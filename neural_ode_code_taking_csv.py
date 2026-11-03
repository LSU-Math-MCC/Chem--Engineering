"""This is a sample neural ODE got from gemini. I've been looking at and playing with it"""
"""With generous aid form google gemini, I have set it up to run with the results from the CSV from the simulation.
At current, It is taking in the state variables from the CSV and using them to train a neural ODE to fit the simulation data.
Now also includes input temperature T_i as an additional input to the neural ODE function.

Newest changes: With the help of Google Gemini, State variables have been removed as inputs or outputs so now the only inputs
are the 7 flow variables and the input temperature, and the outputs are the measured output variables. Instead of tracking the 
state variables, the neural ODE uses a hidden layer of 36 nodes to simulate the state variables.
It should also be able to take multiple CSV files and train on all of them.
It should also save the model into a .pth file for later use."""
"""With the help of Google Gemini, I have made some changes to the code to make it more efficient and easier to read.
I have also added Optuna hyperparameter optimization to the training process, allowing for automatic tuning of learning rate, 
weight decay, and latent dimension size. The best hyperparameters found during the search will be used for the final training of the model.
I also changed the code to once again run only one CSV file. I made other small changes such as reducing the Nodes in the code as well
as made an optimization to the interpolation of the input features to make it more efficient. 
I also added a progress report for the training loop. I also added the ability to take a random sample from the dataset
"""
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
csv_file = 'results_grade1.csv'  # Add more CSV file paths here if needed
input_vars = ['T_i', 'flow_M1', 'flow_M2', 'flow_M3', 'flow_C6', 'flow_cat', 'flow_Al', 'flow_H2']
output_vars = ['Mw_meas', 'Mn_meas', 'ML_meas', 'X_M1_meas', 'X_M3_meas', 'T_meas', 'PR_meas']

odf = pd.read_csv(csv_file).dropna(subset=output_vars).reset_index(drop=True)

# Randomly sample a percentage of the dataset, then sort by time 't' to preserve sequence order
df = odf.sample(frac=0.25, random_state=11).sort_values('t').reset_index(drop=True)

# 2. Feature Scaling for both Inputs and Outputs
x_raw = df[input_vars].values
x_mean = x_raw.mean(axis=0)
x_std = x_raw.std(axis=0)
x_std[x_std == 0.0] = 1.0  # Prevent division by zero
x_scaled = (x_raw - x_mean) / x_std

y_raw = df[output_vars].values
y_mean = y_raw.mean(axis=0)
y_std = y_raw.std(axis=0)
y_std[y_std == 0.0] = 1.0
y_scaled = (y_raw - y_mean) / y_std

t_np = df['t'].values
t_tensor = torch.tensor(t_np, dtype=torch.float32)
true_y = torch.tensor(y_scaled, dtype=torch.float32).unsqueeze(1)  # Shape: [time_steps, 1, output_dim]

# 3. Define the Neural Network ODE Function with Independent Initial States
class ODEFunc(nn.Module):
    def __init__(self, latent_dim=36, input_dim=len(input_vars), output_dim=len(output_vars)):
        super(ODEFunc, self).__init__()
        self.latent_dim = latent_dim
        
        # Single learnable initial state (h0) for the single dataset
        self.h0 = nn.Parameter(torch.zeros(1, latent_dim))

        self.ode_net = nn.Sequential(
            nn.Linear(latent_dim + input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, latent_dim)
        )
        self.readout = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.Tanh(),
            nn.Linear(32, output_dim)
        )
        
    def set_run_context(self, t_tensor, x_tensor):
        # Store time and input data as PyTorch tensors on the active device
        self.current_t = t_tensor
        self.current_x = x_tensor

    def forward(self, t_tensor, h):
        # 1. Find the upper boundary index using GPU-accelerated searchsorted
        idx_hi = torch.searchsorted(self.current_t, t_tensor, right=False)
        idx_hi = torch.clamp(idx_hi, min=1, max=self.current_t.shape[0] - 1)
        idx_lo = idx_hi - 1

        # 2. Get bounding times and input feature vectors
        t_lo = self.current_t[idx_lo]
        t_hi = self.current_t[idx_hi]
        
        x_lo = self.current_x[idx_lo] # Shape: [input_dim]
        x_hi = self.current_x[idx_hi] # Shape: [input_dim]

        # 3. Compute interpolation weight (alpha) safely avoiding division by zero
        denom = t_hi - t_lo
        alpha = (t_tensor - t_lo) / torch.where(denom == 0, torch.tensor(1.0, device=denom.device), denom)

        # 4. Linearly interpolate external features: x_lo + alpha * (x_hi - x_lo)
        ext_vals = x_lo + alpha.unsqueeze(-1) * (x_hi - x_lo) # Shape: [input_dim]

        # 5. Match batch size dimensions
        batch_size = h.shape[0]
        ext_tensor = ext_vals.unsqueeze(0).expand(batch_size, -1)
        
        # 6. Concatenate latent states and interpolated external inputs
        h_aug = torch.cat([h, ext_tensor], dim=-1)
        
        return self.ode_net(h_aug)

t_tensor_gpu = torch.tensor(t_np, dtype=torch.float32)
x_tensor_gpu = torch.tensor(x_scaled, dtype=torch.float32)

# 4. Optuna Hyperparameter Optimization Study
print("Starting Optuna Hyperparameter Search...")

def objective(trial):
    # Suggest hyperparameters to tune
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    latent_dim = trial.suggest_categorical("latent_dim", [12, 24, 36, 48])
    
    # Initialize model with trial parameters
    func_trial = ODEFunc(latent_dim=latent_dim, input_dim=len(input_vars), output_dim=len(output_vars))
    func_trial.set_run_context(t_tensor_gpu, x_tensor_gpu)

    optimizer_trial = optim.Adam(func_trial.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    
    # Run a shorter training period for tuning efficiency
    for epoch in range(1, 16):
        optimizer_trial.zero_grad()

        pred_h = odeint(func_trial, func_trial.h0, t_tensor) 
        pred_y = func_trial.readout(pred_h) 
        
        loss = criterion(pred_y, true_y)        
        
        loss.backward()
        optimizer_trial.step()
        
        # Report progress to Optuna for pruning poor trials
        trial.report(loss.item(), epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
            
    return loss.item()

# Run the study (adjust n_trials based on your compute availability)
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=5)

print("\nHyperparameter Search Complete!")
best_params = study.best_params
print("Best Parameters Found:")
for key, value in best_params.items():
    print(f"  {key}: {value}")

# 5. Initialize Final Model with Best Hyperparameters
best_latent_dim = best_params['latent_dim']
best_lr = best_params['lr']
best_weight_decay = best_params['weight_decay']

func = ODEFunc(latent_dim=best_latent_dim, input_dim=len(input_vars), output_dim=len(output_vars))
func.set_run_context(t_tensor_gpu, x_tensor_gpu)

optimizer = optim.Adam(func.parameters(), lr=best_lr, weight_decay=best_weight_decay)
criterion = nn.MSELoss()

# 6. Full Training Loop using Best Parameters
print("\nTraining Final Model with Best Parameters...")
for epoch in range(1, 251):
    optimizer.zero_grad()

    pred_h = odeint(func, func.h0, t_tensor) 
    pred_y = func.readout(pred_h) 
    
    loss = criterion(pred_y, true_y)
    loss.backward()
    optimizer.step()
    
    if epoch % 1 == 0 or epoch == 1:
        print(f"Epoch {epoch:03d} | Loss: {loss.item():.6f}", flush=True)

print("\nFinal Training Complete!")

# 7. Evaluation and Inverse Scaling for Plotting
with torch.no_grad():
    final_pred_h = odeint(func, func.h0, t_tensor) 
    final_pred_y_scaled = func.readout(final_pred_h) 

final_pred_y = final_pred_y_scaled.numpy() * y_std + y_mean
true_y_original = df[output_vars].values

output_index = 0  # Change index to plot different variables (0 to 6)
output_name = output_vars[output_index]

true_vals = true_y_original[:, output_index]
pred_vals = final_pred_y[:, 0, output_index]

# Plotting results
plt.figure(figsize=(9, 5))
plt.plot(t_np, true_vals, label=f"Simulation {output_name}", color="blue", linewidth=2, linestyle="--")
plt.plot(t_np, pred_vals, label=f"Neural ODE {output_name}", color="red", linewidth=2)

plt.xlabel("Time ($t$)", fontsize=12)
plt.ylabel(f"Value ({output_name})", fontsize=12)
plt.title(f"Neural ODE Fit for: {output_name} (Optimized)", fontsize=14)
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
    'input_vars': input_vars,
    'output_vars': output_vars,
    'latent_dim': best_latent_dim,
    'best_hyperparameters': best_params
}

torch.save(checkpoint, 'neural_ode_reactor_model.pth')
print("Optimized model successfully exported to 'neural_ode_reactor_model.pth'!")