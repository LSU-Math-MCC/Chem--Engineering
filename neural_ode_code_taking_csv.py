"""This is a sample neural ODE got from gemini. I've been looking at and playing with it"""
"""With generous aid form google gemini, I have set it up to run with the results from the CSV from the simulation.
At current, It is taking in the state variables from the CSV and using them to train a neural ODE to fit the simulation data.
Now also includes input temperature T_i as an additional input to the neural ODE function.

Newest changes: With the help of Google Gemini, State variables have been removed as inputs or outputs so now the only inputs
are the 7 flow variables and the input temperature, and the outputs are the measured output variables. Instead of tracking the
state variables, the neural ODE uses a hidden state whose size Optuna picks to simulate the state variables.
It should also save the model into a .pth file for later use."""
"""With the help of Google Gemini, I have made some changes to the code to make it more efficient and easier to read.
I have also added Optuna hyperparameter optimization to the training process, allowing for automatic tuning of learning rate,
weight decay, and latent dimension size. The best hyperparameters found during the search will be used for the final training of the model.
I made other small changes such as reducing the Nodes in the code as well
as made an optimization to the interpolation of the input features to make it more efficient.
I also added a progress report for the training loop. I also added the ability to take a random sample from the dataset
"""
"""Trains on many simulator runs (python generate_training_data.py makes them) so the model learns how the inputs
drive the reactor instead of memorizing one curve: with one run it scored R2 -11 on unseen runs, with 24 varied runs +0.93.
Time is scaled by T_SCALE because integrating over raw seconds pushes the hidden state into the hundreds or more,
where the tanh layers saturate and the model stops responding to its inputs.
Optuna scores each trial on validation runs, not on the training loss, so it cannot prefer a model that memorizes.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchdiffeq import odeint
import matplotlib.pyplot as plt
import optuna

# Suppress excessive Optuna logging during search if desired
optuna.logging.set_verbosity(optuna.logging.WARNING)

# 1. Configuration & Data Loading
HERE = Path(__file__).parent
DATA_DIR = HERE / 'data'
MODEL_PATH = HERE / 'neural_ode_reactor_model.pth'
input_vars = ['T_i', 'flow_M1', 'flow_M2', 'flow_M3', 'flow_C6', 'flow_cat', 'flow_Al', 'flow_H2']
output_vars = ['Mw_meas', 'Mn_meas', 'ML_meas', 'X_M1_meas', 'X_M3_meas', 'T_meas', 'PR_meas']

# Select frac to be what fraction of each run's rows you want to use for training (e.g., 0.5 for 50%).
# Lower values only drop rows from the loss; they do not add information about the inputs.
sample_frac = 1
T_SCALE = 300.0     # seconds per unit of solver time, about the reactor residence time
SEED = 0
N_TRIALS = 10
TRIAL_EPOCHS = 100  # long enough to tell which settings keep learning
FINAL_EPOCHS = 2500


def load_run(path, frac=1):
    """One simulator CSV as a DataFrame sorted by time. Sampling with a fixed seed keeps the same rows in every run."""
    raw = pd.read_csv(path)
    blank = [name for name in output_vars if name not in raw or raw[name].isna().all()]
    if blank:
        raise ValueError(f"{path}: no data in {', '.join(blank)}. Measurement case 1 leaves Mw and Mn blank; "
                         "use runs with measure case 2.")
    odf = raw.dropna(subset=output_vars).reset_index(drop=True)
    return odf.sample(frac=frac, random_state=11).sort_values('t').reset_index(drop=True)


def scale_inputs(raw):
    """Mean and scale for the inputs. An input that never changes is scaled by its own size, so a different
    value in a later run stays a modest number instead of a huge one."""
    mean, std = raw.mean(axis=0), raw.std(axis=0)
    return mean, np.where(std == 0.0, np.maximum(np.abs(mean), 1.0), std)


def to_tensors(dfs, x_mean, x_std, y_mean, y_std):
    """(tau, x, y) for runs that share one time grid. tau: [time], x and y: [runs, time, columns]."""
    t_np = dfs[0]['t'].values
    if any(not np.array_equal(df['t'].values, t_np) for df in dfs):
        raise ValueError("All runs must have the same time steps; use the same N and dt.")
    tau = torch.tensor((t_np - t_np[0]) / T_SCALE, dtype=torch.float32)
    x = torch.tensor(np.stack([(df[input_vars].values - x_mean) / x_std for df in dfs]), dtype=torch.float32)
    y = torch.tensor(np.stack([(df[output_vars].values - y_mean) / y_std for df in dfs]), dtype=torch.float32)
    return tau, x, y


# 2. Define the Neural Network ODE Function
class ODEFunc(nn.Module):
    def __init__(self, latent_dim=36, input_dim=len(input_vars), output_dim=len(output_vars)):
        super(ODEFunc, self).__init__()
        self.latent_dim = latent_dim

        # Learnable initial state (h0) shared by the runs, which all start from the same reactor conditions
        self.h0 = nn.Parameter(torch.zeros(1, latent_dim))

        self.ode_net = nn.Sequential(
            nn.Linear(latent_dim + input_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, latent_dim)
        )
        self.readout = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.Tanh(),
            nn.Linear(64, output_dim)
        )

    def set_run_context(self, t_tensor, x_tensor):
        # Store time [time] and input data [runs, time, input_dim] as PyTorch tensors
        self.current_t = t_tensor
        self.current_x = x_tensor

    def forward(self, t_tensor, h):
        # 1. Find the upper boundary index using searchsorted
        idx_hi = torch.searchsorted(self.current_t, t_tensor, right=False)
        idx_hi = torch.clamp(idx_hi, min=1, max=self.current_t.shape[0] - 1)
        idx_lo = idx_hi - 1

        # 2. Get bounding times and input feature vectors
        t_lo = self.current_t[idx_lo]
        t_hi = self.current_t[idx_hi]

        x_lo = self.current_x[:, idx_lo]  # Shape: [runs, input_dim]
        x_hi = self.current_x[:, idx_hi]

        # 3. Linearly interpolate external features: x_lo + alpha * (x_hi - x_lo)
        alpha = (t_tensor - t_lo) / (t_hi - t_lo)
        ext_vals = x_lo + alpha * (x_hi - x_lo)

        # 4. Concatenate latent states and interpolated external inputs
        return self.ode_net(torch.cat([h, ext_vals], dim=-1))

    def predict(self, tau, x):
        """Scaled outputs [runs, time, output_dim] for scaled inputs x, along with the hidden states [time, runs, latent_dim]."""
        self.set_run_context(tau, x)
        h0 = self.h0.expand(x.shape[0], -1)
        # Fixed steps: varied runs have step changes at many different times, which makes the adaptive solver about 4x slower.
        h = odeint(self, h0, tau, method='rk4', options={'step_size': (tau[1:] - tau[:-1]).min().item()})
        return self.readout(h).transpose(0, 1), h


def r2_per_output(pred, true):
    """R2 of each output over all runs and steps. 1 is perfect; 0 is no better than predicting the mean."""
    pred, true = pred.reshape(-1, true.shape[-1]).numpy(), true.reshape(-1, true.shape[-1]).numpy()
    return 1 - ((pred - true) ** 2).sum(axis=0) / ((true - true.mean(axis=0)) ** 2).sum(axis=0)


def train_model(train_data, val_data, lr, weight_decay, latent_dim, epochs, report):
    """Train one model. report(epoch, train_loss, val_loss) is called every 10 epochs."""
    torch.manual_seed(SEED)
    tau, x_train, y_train = train_data
    tau_val, x_val, y_val = val_data
    func = ODEFunc(latent_dim=latent_dim, input_dim=len(input_vars), output_dim=len(output_vars))
    optimizer = optim.Adam(func.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    criterion = nn.MSELoss()

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        pred_y, _ = func.predict(tau, x_train)
        loss = criterion(pred_y, y_train)
        loss.backward()
        nn.utils.clip_grad_norm_(func.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if epoch % 10 == 0 or epoch == epochs:
            with torch.no_grad():
                val_loss = criterion(func.predict(tau_val, x_val)[0], y_val).item()
            report(epoch, loss.item(), val_loss)
    return func, val_loss


def main():
    runs = {}
    for split in ('train', 'val', 'test'):
        files = sorted(DATA_DIR.glob(f'{split}_*.csv'))
        if not files:
            raise SystemExit(f"No {split} runs in {DATA_DIR}. Run: python generate_training_data.py")
        runs[split] = [load_run(file, sample_frac if split == 'train' else 1) for file in files]
    print(f"Runs: {len(runs['train'])} train, {len(runs['val'])} validation, {len(runs['test'])} test")

    # 3. Feature Scaling for both Inputs and Outputs, from the training runs only
    x_mean, x_std = scale_inputs(np.concatenate([df[input_vars].values for df in runs['train']]))
    y_raw = np.concatenate([df[output_vars].values for df in runs['train']])
    y_mean, y_std = y_raw.mean(axis=0), y_raw.std(axis=0)
    y_std[y_std == 0.0] = 1.0
    data = {split: to_tensors(dfs, x_mean, x_std, y_mean, y_std) for split, dfs in runs.items()}

    # 4. Optuna Hyperparameter Optimization Study
    print("Starting Optuna Hyperparameter Search...")

    def objective(trial):
        # Suggest hyperparameters to tune
        lr = trial.suggest_float("lr", 3e-4, 5e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        latent_dim = trial.suggest_categorical("latent_dim", [12, 24, 36, 48, 60, 72])

        def report(epoch, train_loss, val_loss):
            # Report progress to Optuna for pruning poor trials
            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        _, val_loss = train_model(data['train'], data['val'], lr, weight_decay, latent_dim, TRIAL_EPOCHS, report)
        print(f"  Trial {trial.number}: validation loss {val_loss:.4f}", flush=True)
        return val_loss

    # Run the study (adjust N_TRIALS based on your compute availability)
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=N_TRIALS)

    print("\nHyperparameter Search Complete!")
    best_params = study.best_params
    print("Best Parameters Found:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")

    # 5. Full Training Loop using Best Parameters
    print("\nTraining Final Model with Best Parameters...")

    def report(epoch, train_loss, val_loss):
        if epoch % 50 == 0:
            print(f"Epoch {epoch:04d} | Train loss: {train_loss:.6f} | Validation loss: {val_loss:.6f}", flush=True)

    func, _ = train_model(data['train'], data['val'], best_params['lr'], best_params['weight_decay'],
                          best_params['latent_dim'], FINAL_EPOCHS, report)
    print("\nFinal Training Complete!")

    # 6. Evaluation: the test runs were not used for training or for choosing settings
    with torch.no_grad():
        predictions = {split: func.predict(tau, x)[0] for split, (tau, x, _) in data.items()}
    for split, (_, _, y) in data.items():
        r2 = r2_per_output(predictions[split], y)
        print(f"R2 on {split}: mean {r2.mean():.3f} | " + ", ".join(f"{name} {v:.3f}" for name, v in zip(output_vars, r2)))

    # 7. Inverse Scaling and Plotting for the first test run
    output_index = 0  # Change index to plot different variables (0 to 6)
    output_name = output_vars[output_index]
    test_df = runs['test'][0]
    t_np = test_df['t'].values
    true_vals = test_df[output_vars].values[:, output_index]
    pred_vals = predictions['test'][0].numpy()[:, output_index] * y_std[output_index] + y_mean[output_index]

    plt.figure(figsize=(9, 5))
    plt.plot(t_np, true_vals, label=f"Simulation {output_name}", color="blue", linewidth=2, linestyle="--")
    plt.plot(t_np, pred_vals, label=f"Neural ODE {output_name}", color="red", linewidth=2)

    plt.xlabel("Time ($t$)", fontsize=12)
    plt.ylabel(f"Value ({output_name})", fontsize=12)
    plt.title(f"Neural ODE on an unseen run: {output_name}", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.show()

    # 8. Save trained model weights, feature scaling parameters, and metadata
    checkpoint = {
        'model_state_dict': func.state_dict(),
        'x_mean': x_mean,
        'x_std': x_std,
        'y_mean': y_mean,
        'y_std': y_std,
        't_scale': T_SCALE,
        'input_vars': input_vars,
        'output_vars': output_vars,
        'latent_dim': best_params['latent_dim'],
        'best_hyperparameters': best_params
    }

    torch.save(checkpoint, MODEL_PATH)
    print(f"Optimized model successfully exported to '{MODEL_PATH.name}'!")


if __name__ == "__main__":
    main()
