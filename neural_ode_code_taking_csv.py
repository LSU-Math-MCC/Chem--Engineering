"""This is a sample neural ODE got from gemini. I've been looking at and playing with it"""
"""With generous aid form google gemini, I have set it up to run with the results from the CSV from the simulation.
At current, It is taking in the state variables from the CSV and using them to train a neural ODE to fit the simulation data.
Now also includes input temperature T_i as an additional input to the neural ODE function.

Newest changes: With the help of Google Gemini, State variables have been removed as inputs or outputs so now the only inputs
are the 7 flow variables and the input temperature, and the outputs are the measured output variables. Instead of tracking the 
state variables, the neural ODE uses a hidden layer of 36 nodes to simulate the state variables.
It should also be able to take multiple CSV files and train on all of them.
It should also save the model into a .pth file for later use."""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchdiffeq import odeint
import matplotlib.pyplot as plt

#csv files
csv_files = ['results.csv']
#input columns (T_i plus the 7 flow variables)
input_vars = ['T_i', 'flow_M1', 'flow_M2', 'flow_M3', 'flow_C6', 'flow_cat', 'flow_Al', 'flow_H2']
#output columns (measured output variables)
output_vars = ['Mw_meas', 'Mn_meas', 'ML_meas', 'X_M1_meas', 'X_M3_meas', 'T_meas', 'PR_meas'] #remove 'Mw_meas' and 'Mn_meas' if tracking them is not desired

raw_datasets = []
for file in csv_files:
    df = pd.read_csv(file).dropna(subset=output_vars).reset_index(drop=True)
    raw_datasets.append(df)

# 2. Feature Scaling (keep inputs and outputs on comparable scales) 
all_outputs_concat = np.concatenate([df[output_vars].values for df in raw_datasets], axis=0)
y_mean = all_outputs_concat.mean(axis=0)
y_std = all_outputs_concat.std(axis=0)
y_std[y_std == 0.0] = 1.0

runs = []
for df in raw_datasets:
    t_np = df['t'].values
    y_raw = df[output_vars].values
    y_scaled = (y_raw - y_mean) / y_std
    
    runs.append({
        't_np': t_np,
        't': torch.tensor(t_np, dtype=torch.float32),
        'true_y': torch.tensor(y_scaled, dtype=torch.float32).unsqueeze(1), # [time_steps, 1, 7]
        'df': df
    })

latent_layer_size = 36  

# 3. Define the Neural Network ODE Function with all External Inputs
class ODEFunc(nn.Module):
    def __init__(self, latent_dim=latent_layer_size, input_dim=(len(input_vars)), output_dim=len(output_vars)):
        super(ODEFunc, self).__init__()
        self.latent_dim = latent_dim

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
    def set_run_context(self, df, t_np):  #Sets the active CSV dataframe and time array for interpolation
            self.current_df = df
            self.current_t_np = t_np

    def forward(self, t_tensor, h):
        t_val = t_tensor.item()
        
        # Linearly interpolate standardized external inputs at the current solver time.
        ext_vals = np.array([np.interp(t_val, self.current_t_np, self.current_df[input_vars].values[:, i]) for i in range(len(input_vars))])
        
        # Match batch size dimensions
        batch_size = h.shape[0]
        ext_tensor = torch.tensor(ext_vals, dtype=h.dtype, device=h.device).unsqueeze(0).expand(batch_size, -1) # Shape: [batch_size, 8]
        
        # Concatenate states and external inputs
        h_aug = torch.cat([h, ext_tensor], dim=-1)
        
        return self.ode_net(h_aug)

# 4. Initialize Model, Optimizer, and Loss
func = ODEFunc(latent_dim=latent_layer_size, input_dim=(len(input_vars)), output_dim=len(output_vars))
optimizer = optim.Adam(func.parameters(), lr=0.005, weight_decay=1e-5)
criterion = nn.MSELoss()

# 5. Training Loop (test with a few epochs first)
print("Training Neural ODE...")
for epoch in range(1, 251):
    optimizer.zero_grad()
    total_loss = 0.0

    for run in runs:
        # Tell the ODE function which CSV context to interpolate from
        func.set_run_context(run['df'], run['t_np'])
        
        pred_h = odeint(func, func.h0, run['t']) 
        pred_y = func.readout(pred_h) 
        
        loss = criterion(pred_y, run['true_y'])
        total_loss += loss
    
    total_loss.backward()
    optimizer.step()
    
    print(f"Epoch {epoch:03d} | Loss: {total_loss.item():.6f}", flush=True)

print("\nTraining Complete!")

# 6. Evaluation and Inverse Scaling for Plotting
run_index = 0 
target_run = runs[run_index]

# Set the context for that specific run
func.set_run_context(target_run['df'], target_run['t_np'])

with torch.no_grad():
    final_pred_h = odeint(func, func.h0, target_run['t'])  # Shape: [time_steps, 1, latent_dim]
    final_pred_y_scaled = func.readout(final_pred_h)  # Shape: [time_steps, 1, output_dim]

final_pred_y = final_pred_y_scaled.numpy() * y_std + y_mean
true_y_original = target_run['df'][output_vars].values
t_np_plot = target_run['t_np']

output_index = 0  # Change this index to plot different state variables (0 to 6) 
output_name = output_vars[output_index]

true_vals = true_y_original[:, output_index]
pred_vals = final_pred_y[:, 0, output_index]

# Create the plot
plt.figure(figsize=(9, 5))
plt.plot(t_np_plot, true_vals, label=f"Simulation {output_name}", color="blue", linewidth=2, linestyle="--")
plt.plot(t_np_plot, pred_vals, label=f"Neural ODE {output_name}", color="red", linewidth=2)

plt.xlabel("Time ($t$)", fontsize=12)
plt.ylabel(f"Value ({output_name})", fontsize=12)
plt.title(f"Neural ODE Fit for: {output_name}", fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, linestyle=":", alpha=0.7)
plt.show()

# Save trained model weights and feature scaling parameters
checkpoint = {
    'model_state_dict': func.state_dict(),
    'y_mean': y_mean,
    'y_std': y_std,
    'input_vars': input_vars,
    'output_vars': output_vars,
    'latent_dim': latent_layer_size
}

torch.save(checkpoint, 'neural_ode_reactor_model.pth')
print("Model successfully exported to 'neural_ode_reactor_model.pth'!")