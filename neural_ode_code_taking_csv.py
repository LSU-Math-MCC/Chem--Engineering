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
I also added a progress report for the training loop.
"""
"""Trains on many simulator runs (python generate_training_data.py makes them) so the model learns how the inputs
drive the reactor instead of memorizing one curve. Time is scaled by T_SCALE because integrating over raw seconds
pushes the hidden state into the hundreds or more, where the tanh layers saturate and the model stops responding
to its inputs. Optuna scores each trial on validation runs, not on the training loss, so it cannot prefer a model
that memorizes.

The runs follow EPDM_Data_Generation: 1,450-step DOE step tests for the four grades (parameter sets). One model
covers all grades: the grade is an extra one-hot input, since the same feeds give very different polymer in each
grade. Mw, Mn, ML and PR are log-scaled because they span orders of magnitude across grades. Training uses random
WINDOW-step windows cut from the long runs; an encoder turns the ENC_STEPS steps before a window into its starting
hidden state. A whole run is predicted the same way, starting after its first ENC_STEPS steps. The four original
EPDM_Data_Generation datasets are the final benchmark and are never trained on.
"""
import re
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
REFERENCE_DIR = HERE / 'EPDM_Data_Generation' / 'Generated_Data'
MODEL_PATH = HERE / 'neural_ode_reactor_model.pth'
input_vars = ['T_i', 'flow_M1', 'flow_M2', 'flow_M3', 'flow_C6', 'flow_cat', 'flow_Al', 'flow_H2']
output_vars = ['Mw_meas', 'Mn_meas', 'ML_meas', 'X_M1_meas', 'X_M3_meas', 'T_meas', 'PR_meas']
GRADES = (1, 2, 3, 4)
LOG_OUTPUTS = ['Mw_meas', 'Mn_meas', 'ML_meas', 'PR_meas']   # log1p before scaling
LOG_IDX = [output_vars.index(name) for name in LOG_OUTPUTS]

T_SCALE = 300.0     # seconds per unit of solver time, about the reactor residence time
ENC_STEPS = 10      # steps of history the encoder reads before a window starts
WINDOW = 150        # steps per training window
BATCH = 64          # windows per training step
VAL_WINDOWS = 64    # fixed validation windows, so every trial is scored on the same ones
SEED = 0
N_TRIALS = 10
TRIAL_EPOCHS = 100  # long enough to tell which settings keep learning
FINAL_EPOCHS = 2000
PLOT_GRADE = 1      # which reference dataset to plot at the end


def grade_of(path):
    """Grade number from a file name: data/train_g3_000.csv or results_grade3.csv."""
    m = re.search(r'(?:_g|grade)(\d+)', Path(path).stem)
    if not m or int(m.group(1)) not in GRADES:
        raise ValueError(f"{path}: cannot tell the grade from the file name (expected _g<n> or grade<n>, n in {GRADES}).")
    return int(m.group(1))


def load_run(path):
    """One simulator CSV as a DataFrame sorted by time, rows with all measurements only."""
    raw = pd.read_csv(path)
    blank = [name for name in output_vars if name not in raw or raw[name].isna().all()]
    if blank:
        raise ValueError(f"{path}: no data in {', '.join(blank)}. Measurement case 1 leaves Mw and Mn blank; "
                         "use runs with measure case 2.")
    return raw.dropna(subset=output_vars).sort_values('t').reset_index(drop=True)


def scale_inputs(raw):
    """Mean and scale for the inputs. An input that never changes is scaled by its own size, so a different
    value in a later run stays a modest number instead of a huge one."""
    mean, std = raw.mean(axis=0), raw.std(axis=0)
    return mean, np.where(std == 0.0, np.maximum(np.abs(mean), 1.0), std)


def transform_outputs(y):
    """Log-scale the outputs that span orders of magnitude (noise can make a reading slightly negative)."""
    y = np.array(y, dtype=float)
    y[:, LOG_IDX] = np.log1p(np.maximum(y[:, LOG_IDX], 0.0))
    return y


def untransform_outputs(y):
    y = np.array(y, dtype=float)
    y[:, LOG_IDX] = np.expm1(y[:, LOG_IDX])
    return y


def run_arrays(df, grade, scales):
    """Scaled inputs plus the grade one-hot [time, inputs + grades], and scaled outputs [time, outputs]."""
    x_mean, x_std, y_mean, y_std = scales
    onehot = np.zeros((len(df), len(GRADES)))
    onehot[:, GRADES.index(grade)] = 1.0
    x = np.hstack([(df[input_vars].values - x_mean) / x_std, onehot])
    y = (transform_outputs(df[output_vars].values) - y_mean) / y_std
    return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


def tau_grid(df, steps):
    """Solver times for `steps` steps after a window's start (all runs share dt)."""
    dt = float(df['t'].values[1] - df['t'].values[0])
    return torch.arange(steps + 1, dtype=torch.float32) * (dt / T_SCALE)


def make_windows(arrays, picks):
    """Batch for (run, start) picks: encoder history [B, ENC_STEPS + 1, in + out], inputs and targets
    over the window [B, WINDOW + 1, ...]."""
    hist, xs, ys = [], [], []
    for i, s in picks:
        x, y = arrays[i]
        hist.append(torch.cat([x[s - ENC_STEPS:s + 1], y[s - ENC_STEPS:s + 1]], dim=-1))
        xs.append(x[s:s + WINDOW + 1])
        ys.append(y[s:s + WINDOW + 1])
    return torch.stack(hist), torch.stack(xs), torch.stack(ys)


def random_picks(arrays, count, rng):
    return [(i, int(rng.integers(ENC_STEPS, len(arrays[i][0]) - WINDOW))) for i in rng.integers(0, len(arrays), count)]


# 2. Define the Neural Network ODE Function
class ODEFunc(nn.Module):
    def __init__(self, latent_dim=36, input_dim=len(input_vars) + len(GRADES), output_dim=len(output_vars)):
        super(ODEFunc, self).__init__()
        self.latent_dim = latent_dim

        # Encoder: the last ENC_STEPS + 1 steps of inputs and measurements -> hidden state at a window's start
        self.encoder = nn.Sequential(
            nn.Linear((ENC_STEPS + 1) * (input_dim + output_dim), 128),
            nn.Tanh(),
            nn.Linear(128, latent_dim)
        )
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

    def predict(self, tau, x, history):
        """Scaled outputs [runs, time, output_dim] from a window's history and its inputs x, with the hidden
        states [time, runs, latent_dim]."""
        self.set_run_context(tau, x)
        h0 = self.encoder(history.flatten(1))
        # Fixed steps: step changes come at many different times, which makes the adaptive solver much slower.
        h = odeint(self, h0, tau, method='rk4', options={'step_size': (tau[1:] - tau[:-1]).min().item()})
        return self.readout(h).transpose(0, 1), h


def forecast(func, df, grade, scales):
    """Predict a whole run from its first ENC_STEPS steps. Returns (t, predicted, measured) in real units."""
    x, y = run_arrays(df, grade, scales)
    steps = len(df) - 1 - ENC_STEPS
    history = torch.cat([x[:ENC_STEPS + 1], y[:ENC_STEPS + 1]], dim=-1).unsqueeze(0)
    with torch.no_grad():
        pred, _ = func.predict(tau_grid(df, steps), x[ENC_STEPS:].unsqueeze(0), history)
    _, _, y_mean, y_std = scales
    pred = untransform_outputs(pred[0].numpy() * y_std + y_mean)
    return df['t'].values[ENC_STEPS:], pred, df[output_vars].values[ENC_STEPS:]


def r2_per_output(pred, true):
    """R2 of each output. 1 is perfect; 0 is no better than predicting the mean."""
    pred, true = np.asarray(pred).reshape(-1, true.shape[-1]), np.asarray(true).reshape(-1, true.shape[-1])
    return 1 - ((pred - true) ** 2).sum(axis=0) / ((true - true.mean(axis=0)) ** 2).sum(axis=0)


def train_model(train_arrays, val_batch, tau, lr, weight_decay, latent_dim, epochs, report):
    """Train one model on random windows. report(epoch, train_loss, val_loss) is called every 10 epochs."""
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    func = ODEFunc(latent_dim=latent_dim)
    optimizer = optim.Adam(func.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    criterion = nn.MSELoss()

    for epoch in range(1, epochs + 1):
        history, x, y = make_windows(train_arrays, random_picks(train_arrays, BATCH, rng))
        optimizer.zero_grad()
        pred_y, _ = func.predict(tau, x, history)
        loss = criterion(pred_y, y)
        loss.backward()
        nn.utils.clip_grad_norm_(func.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if epoch % 10 == 0 or epoch == epochs:
            with torch.no_grad():
                val_loss = criterion(func.predict(tau, val_batch[1], val_batch[0])[0], val_batch[2]).item()
            report(epoch, loss.item(), val_loss)
    return func, val_loss


def main():
    runs = {}
    for split in ('train', 'val', 'test'):
        files = sorted(DATA_DIR.glob(f'{split}_g*.csv'))
        if not files:
            raise SystemExit(f"No {split} runs in {DATA_DIR}. Run: python generate_training_data.py")
        runs[split] = [(load_run(file), grade_of(file), file.name) for file in files]
    runs['reference'] = [(load_run(file), grade_of(file), file.name) for file in sorted(REFERENCE_DIR.glob('results_grade*.csv'))]
    print("Runs: " + ", ".join(f"{len(v)} {k}" for k, v in runs.items()))

    # 3. Feature Scaling for both Inputs and Outputs, from the training runs only
    x_mean, x_std = scale_inputs(np.concatenate([df[input_vars].values for df, _, _ in runs['train']]))
    y_raw = np.concatenate([transform_outputs(df[output_vars].values) for df, _, _ in runs['train']])
    y_mean, y_std = y_raw.mean(axis=0), y_raw.std(axis=0)
    y_std[y_std == 0.0] = 1.0
    scales = (x_mean, x_std, y_mean, y_std)
    arrays = {split: [run_arrays(df, g, scales) for df, g, _ in v] for split, v in runs.items()}
    tau = tau_grid(runs['train'][0][0], WINDOW)
    val_batch = make_windows(arrays['val'], random_picks(arrays['val'], VAL_WINDOWS, np.random.default_rng(SEED + 1)))

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

        _, val_loss = train_model(arrays['train'], val_batch, tau, lr, weight_decay, latent_dim, TRIAL_EPOCHS, report)
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

    func, _ = train_model(arrays['train'], val_batch, tau, best_params['lr'], best_params['weight_decay'],
                          best_params['latent_dim'], FINAL_EPOCHS, report)
    func.eval()
    print("\nFinal Training Complete!")

    # 6. Evaluation on whole runs, per grade, in real units. Test and reference runs were never trained on.
    for split in ('test', 'reference'):
        print(f"\nR2 on whole {split} runs (predicted from step {ENC_STEPS} to the end):")
        for df, grade, name in runs[split]:
            _, pred, true = forecast(func, df, grade, scales)
            r2 = r2_per_output(pred, true)
            print(f"  grade {grade} {name}: mean {r2.mean():.3f} | " + ", ".join(f"{n} {v:.3f}" for n, v in zip(output_vars, r2)))

    # 7. Plot all output variables for one reference dataset
    df, grade, name = next(r for r in runs['reference'] if r[1] == PLOT_GRADE)
    t_np, pred, true = forecast(func, df, grade, scales)

    # Create a 4x2 grid, sharing the x-axis to eliminate duplicate tick labels
    fig, axes = plt.subplots(4, 2, figsize=(14, 18), sharex=True)
    axes = axes.flatten()

    for i, output_name in enumerate(output_vars):
        ax = axes[i]
        ax.plot(t_np, true[:, i], label="Simulation", color="blue", linewidth=2, linestyle="--")
        ax.plot(t_np, pred[:, i], label="Neural ODE", color="red", linewidth=2)

        # Only label the x-axis on the bottom row to keep it clean
        if i >= len(output_vars) - 2:
            ax.set_xlabel("Time ($t$)", fontsize=11)

        ax.set_ylabel("Value", fontsize=11)
        ax.set_title(f"{name}: {output_name}", fontsize=12, pad=10)
        ax.legend(fontsize=9)
        ax.grid(True, linestyle=":", alpha=0.7)

    # Hide the 8th empty subplot slot since there are only 7 output variables
    if len(output_vars) < len(axes):
        axes[-1].axis('off')

    # Give rows plenty of breathing room (hspace=0.55)
    plt.subplots_adjust(hspace=0.55, wspace=0.25, top=0.94, bottom=0.06)
    plt.show()

    # 8. Save trained model weights, feature scaling parameters, and metadata
    checkpoint = {
        'model_state_dict': func.state_dict(),
        'x_mean': x_mean,
        'x_std': x_std,
        'y_mean': y_mean,
        'y_std': y_std,
        't_scale': T_SCALE,
        'enc_steps': ENC_STEPS,
        'grades': GRADES,
        'log_outputs': LOG_OUTPUTS,
        'input_vars': input_vars,
        'output_vars': output_vars,
        'latent_dim': best_params['latent_dim'],
        'best_hyperparameters': best_params
    }

    torch.save(checkpoint, MODEL_PATH)
    print(f"Optimized model successfully exported to '{MODEL_PATH.name}'!")


if __name__ == "__main__":
    main()
