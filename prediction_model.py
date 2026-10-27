import torch
import pandas as pd
from torchdiffeq import odeint
import matplotlib.pyplot as plt

# Import your ODEFunc directly from your training file!
from NODE_with_multiple_csv_data import ODEFunc, input_vars, output_vars

# Load the saved checkpoint
checkpoint = torch.load('neural_ode_reactor_model.pth')

y_mean = checkpoint['y_mean']
y_std = checkpoint['y_std']
latent_dim = checkpoint['latent_dim']

# Instantiate model and load weights
model = ODEFunc(latent_dim=latent_dim)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Load new CSV data for prediction
new_df = pd.read_csv('results_new_run.csv').dropna(subset=output_vars).reset_index(drop=True)
t_new_np = new_df['t'].values
t_new_tensor = torch.tensor(t_new_np, dtype=torch.float32)

# Set context using the DataFrame version
model.set_run_context(new_df, t_new_np)

# Run prediction
with torch.no_grad():
    pred_h = odeint(model, model.h0, t_new_tensor)
    pred_y_scaled = model.readout(pred_h)

# Inverse scale back to physical units
pred_y = pred_y_scaled.numpy() * y_std + y_mean
true_y_original = new_df[output_vars].values

# Plotting
output_index = 2  # e.g., ML_meas
output_name = output_vars[output_index]

plt.figure(figsize=(9, 5))
plt.plot(t_new_np, true_y_original[:, output_index], label=f"True {output_name}", color="blue", linestyle="--")
plt.plot(t_new_np, pred_y[:, 0, output_index], label=f"Neural ODE Predicted {output_name}", color="red")
plt.xlabel("Time ($t$)")
plt.ylabel(output_name)
plt.title(f"Neural ODE Generalization Test: {output_name}")
plt.legend()
plt.grid(True, linestyle=":", alpha=0.7)
plt.show()
