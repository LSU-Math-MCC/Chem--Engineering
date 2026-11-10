"""Framework from Google Gemini made to match the neural_ode_code_taking_csv.py output pth file

    python prediction_model.py                                   # reference dataset of grade 1
    python prediction_model.py EPDM_Data_Generation/Generated_Data/results_grade3.csv
    python prediction_model.py some_run.csv 2                    # any case 2 run; grade given

The grade comes from the file name (_g<n> or grade<n>) unless it is given after the path.
The whole run is predicted from its first ENC_STEPS steps.
"""

import sys

import torch
import matplotlib.pyplot as plt

# Import your ODEFunc directly from your training file!
from neural_ode_code_taking_csv import (ENC_STEPS, GRADES, LOG_OUTPUTS, MODEL_PATH, REFERENCE_DIR, T_SCALE,
                                        ODEFunc, forecast, grade_of, load_run, output_vars, r2_per_output)

csv_path = sys.argv[1] if len(sys.argv) > 1 else REFERENCE_DIR / 'results_grade1.csv'
grade = int(sys.argv[2]) if len(sys.argv) > 2 else grade_of(csv_path)

# Load the saved checkpoint (our own file; it holds NumPy scaling arrays, which weights_only would refuse)
checkpoint = torch.load(MODEL_PATH, weights_only=False)
if (checkpoint.get('t_scale') != T_SCALE or checkpoint.get('enc_steps') != ENC_STEPS
        or tuple(checkpoint.get('grades', ())) != GRADES or checkpoint.get('log_outputs') != LOG_OUTPUTS):
    sys.exit(f"{MODEL_PATH.name} was trained with a different setup. Retrain: python neural_ode_code_taking_csv.py")
scales = (checkpoint['x_mean'], checkpoint['x_std'], checkpoint['y_mean'], checkpoint['y_std'])

# Instantiate model and load weights
model = ODEFunc(latent_dim=checkpoint['latent_dim'])
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Predict the whole run and compare in real units
t_np, pred_y, true_y = forecast(model, load_run(csv_path), grade, scales)
r2 = r2_per_output(pred_y, true_y)
print(f"R2 on {csv_path} (grade {grade}): mean {r2.mean():.3f} | " + ", ".join(f"{name} {v:.3f}" for name, v in zip(output_vars, r2)))

# Plot all output variables (already in real units)
# Create a 4x2 grid, sharing the x-axis to eliminate duplicate tick labels
fig, axes = plt.subplots(4, 2, figsize=(14, 18), sharex=True)
axes = axes.flatten()

for i, output_name in enumerate(output_vars):
    ax = axes[i]
    ax.plot(t_np, true_y[:, i], label="Simulation", color="blue", linewidth=2, linestyle="--")
    ax.plot(t_np, pred_y[:, i], label="Neural ODE", color="red", linewidth=2)

    # Only label the x-axis on the bottom row to keep it clean
    if i >= len(output_vars) - 2:
        ax.set_xlabel("Time ($t$)", fontsize=11)

    ax.set_ylabel("Value", fontsize=11)
    ax.set_title(f"Grade {grade}: {output_name} (R² {r2[i]:.3f})", fontsize=12, pad=10)
    ax.legend(fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.7)

# Hide the 8th empty subplot slot since there are only 7 output variables
if len(output_vars) < len(axes):
    axes[-1].axis('off')

# Give rows plenty of breathing room (hspace=0.55)
plt.subplots_adjust(hspace=0.55, wspace=0.25, top=0.94, bottom=0.06)
plt.show()
