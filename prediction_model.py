"""Framework from Google Gemini made to match the neural_ode_code_taking_csv.py output pth file

    python prediction_model.py                     # first test run from generate_training_data.py
    python prediction_model.py some_run.csv        # any run with measure case 2
"""

import sys

import torch
import matplotlib.pyplot as plt

# Import your ODEFunc directly from your training file!
from neural_ode_code_taking_csv import DATA_DIR, MODEL_PATH, ODEFunc, T_SCALE, load_run, output_vars, r2_per_output, to_tensors

csv_path = sys.argv[1] if len(sys.argv) > 1 else DATA_DIR / 'test_000.csv'

# Load the saved checkpoint (our own file; it holds NumPy scaling arrays, which weights_only would refuse)
checkpoint = torch.load(MODEL_PATH, weights_only=False)
if checkpoint.get('t_scale') != T_SCALE:
    sys.exit(f"{MODEL_PATH.name} was trained with different time scaling. Retrain: python neural_ode_code_taking_csv.py")

y_mean = checkpoint['y_mean']
y_std = checkpoint['y_std']

# Instantiate model and load weights
model = ODEFunc(latent_dim=checkpoint['latent_dim'])
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Load new CSV data for prediction
new_df = load_run(csv_path)
t_new_np = new_df['t'].values
tau, x, y = to_tensors([new_df], checkpoint['x_mean'], checkpoint['x_std'], y_mean, y_std)

# Run prediction
with torch.no_grad():
    pred_y_scaled, _ = model.predict(tau, x)

r2 = r2_per_output(pred_y_scaled, y)
print(f"R2 on {csv_path}: mean {r2.mean():.3f} | " + ", ".join(f"{name} {v:.3f}" for name, v in zip(output_vars, r2)))

# Inverse scale back to physical units
pred_y = pred_y_scaled[0].numpy() * y_std + y_mean
true_y_original = new_df[output_vars].values

# Plotting
output_index = 2  # e.g., ML_meas
output_name = output_vars[output_index]

plt.figure(figsize=(9, 5))
plt.plot(t_new_np, true_y_original[:, output_index], label=f"True {output_name}", color="blue", linestyle="--")
plt.plot(t_new_np, pred_y[:, output_index], label=f"Neural ODE Predicted {output_name}", color="red")
plt.xlabel("Time ($t$)")
plt.ylabel(output_name)
plt.title(f"Neural ODE Generalization Test: {output_name}")
plt.legend()
plt.grid(True, linestyle=":", alpha=0.7)
plt.show()
