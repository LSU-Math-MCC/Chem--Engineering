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

# Plotting
output_index = 2  # e.g., ML_meas
output_name = output_vars[output_index]

plt.figure(figsize=(9, 5))
plt.plot(t_np, true_y[:, output_index], label=f"True {output_name}", color="blue", linestyle="--")
plt.plot(t_np, pred_y[:, output_index], label=f"Neural ODE Predicted {output_name}", color="red")
plt.xlabel("Time ($t$)")
plt.ylabel(output_name)
plt.title(f"Neural ODE Generalization Test: {output_name} (grade {grade})")
plt.legend()
plt.grid(True, linestyle=":", alpha=0.7)
plt.show()
