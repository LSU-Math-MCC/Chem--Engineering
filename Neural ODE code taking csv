"""This is a sample neural ODE got from gemini. I've been looking at and playing with it"""
"""With generous aid form google gemini, I have set it up to run with the results from the CSV from the simulation.
At current, It is taking in the state variables from the CSV and using them to train a neural ODE to fit the simulation data.
Now also includes input temperature T_i as an additional input to the neural ODE function."""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchdiffeq import odeint
import matplotlib.pyplot as plt
import derivatives
from derivatives import STATE_VARS

# 1. Load and Verify CSV
df = pd.read_csv('results.csv')

missing_vars = [var for var in STATE_VARS if var not in df.columns]
if missing_vars:
    print(f"Warning: These STATE_VARS were not found in the CSV: {missing_vars}")
else:
    print("Success: All STATE_VARS matched your CSV columns!")

#external input columns (T_i plus the 7 flow variables)
input_vars = ['T_i', 'flow_M1', 'flow_M2', 'flow_M3', 'flow_C6', 'flow_cat', 'flow_Al', 'flow_H2']

# Extract time, states, and external inputs
t_np = df['t'].values
y_raw = df[STATE_VARS].values           # Shape: [time_steps, 36]
inputs_raw = df[input_vars].values      # Shape: [time_steps, 8]

# 2. Feature Scaling (Scaling states to ensure balanced gradients)
y_mean = y_raw.mean(axis=0)
y_std = y_raw.std(axis=0)
y_std[y_std == 0.0] = 1.0 
y_scaled = (y_raw - y_mean) / y_std

# Convert to PyTorch Tensors
t = torch.tensor(t_np, dtype=torch.float32)
true_y = torch.tensor(y_scaled, dtype=torch.float32).unsqueeze(1)  # Shape: [time_steps, 1, 36]
true_y0 = true_y[0]  # Shape: [1, 36]

# 3. Define the Neural Network ODE Function with all External Inputs
class ODEFunc(nn.Module):
    def __init__(self):
        super(ODEFunc, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(44, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 36)  #36 state derivatives
        )

    def forward(self, t_tensor, y):
        t_val = t_tensor.item()
        
        # Linearly interpolate all 8 external inputs at the current solver time t_val
        ext_vals = np.array([np.interp(t_val, t_np, df[col].values) for col in input_vars]) # Shape: [8]
        
        # Match batch size dimensions (y shape: [batch_size, 36])
        batch_size = y.shape[0]
        ext_tensor = torch.tensor(ext_vals, dtype=y.dtype, device=y.device).unsqueeze(0).expand(batch_size, -1) # Shape: [batch_size, 8]
        
        # Concatenate states and external inputs: shape [batch_size, 44]
        y_aug = torch.cat([y, ext_tensor], dim=-1)
        
        return self.net(y_aug)

# 4. Initialize Model, Optimizer, and Loss
func = ODEFunc()
optimizer = optim.Adam(func.parameters(), lr=0.005, weight_decay=1e-5)
criterion = nn.MSELoss()

# 5. Training Loop (test with a few epochs first)
print("Training Neural ODE...")
for epoch in range(1, 11):
    optimizer.zero_grad()
    
    pred_y = odeint(func, true_y0, t) 
    loss = criterion(pred_y, true_y)
    
    loss.backward()
    optimizer.step()
    
    print(f"Epoch {epoch:03d} | Loss: {loss.item():.6f}", flush=True)

print("\nTraining Complete!")

# 6. Evaluation and Inverse Scaling for Plotting
with torch.no_grad():
    final_pred_y_scaled = odeint(func, true_y0, t)

final_pred_y = final_pred_y_scaled.numpy() * y_std + y_mean
true_y_original = y_raw

t_np = t.numpy()
state_index = 0  
state_name = STATE_VARS[state_index]

true_vals = true_y_original[:, state_index]
pred_vals = final_pred_y[:, 0, state_index]

# Create the plot
plt.figure(figsize=(9, 5))
plt.plot(t_np, true_vals, label=f"Simulation {state_name}", color="blue", linewidth=2, linestyle="--")
plt.plot(t_np, pred_vals, label=f"Neural ODE {state_name}", color="red", linewidth=2)

plt.xlabel("Time ($t$)", fontsize=12)
plt.ylabel(f"Value ({state_name})", fontsize=12)
plt.title(f"Neural ODE Fit for: {state_name}", fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, linestyle=":", alpha=0.7)
plt.show()
