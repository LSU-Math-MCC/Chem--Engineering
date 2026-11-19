"""
Grey-box neural ODE: the black-box model plus measured monomer balances (IN - OUT) as inputs.

    python neural_ode_inout.py                             # train (same data and steps as neural_ode_code_taking_csv.py)
    python neural_ode_inout.py predict <run.csv> [grade]   # predict a whole run and plot it

Outputs are the same as the black-box model (Mw, Mn, ML, X_M1, X_M3, T, PR measurements).
Inputs are the black-box inputs (T_i, the 7 feed flows, the grade) plus, for each monomer,
IN - OUT in lb/hr: the feed rate minus the unreacted flow leaving the reactor, which is
measured as the total volumetric flow F times the outlet concentration (times MW). IN - OUT
is what the reactor consumes plus what accumulates in it, a known piece of the mass balance
handed to the network, which makes the model slightly grey-box.

The simulator does not add noise to the outlet concentrations, so measurement noise is added
here: each concentration reading is C (1 + SIGMA e), e ~ N(0, 1) per step, seeded from the file
name so a run always gets the same noise. Because the features come from measurements made
along the run, a forecast needs the outlet measurements over the period it predicts.
"""

import sys
import zlib
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

import model
from derivatives import L_PER_FT3
import neural_ode_code_taking_csv as node

MODEL_PATH = node.HERE / 'neural_ode_inout_model.pth'
SIGMA = 0.02            # relative noise on each outlet concentration reading, as for the other analyzers
MONOMERS = ('M1', 'M2', 'M3')
MW = model.DEFAULT_CONFIG['p0']['Mw']                  # monomer weights (the same in all four grades)
FEATURES = tuple(f'in_out_{m}' for m in MONOMERS)


def add_in_out(df, path):
    """IN - OUT for each monomer, lb/hr: feed minus F (ft3/hr) x L/ft3 x measured C (mol/L) x MW / (g/lb)."""
    df = df.copy()
    rng = np.random.default_rng(zlib.crc32(Path(path).name.encode()))
    litres_per_hr = df['F'] * L_PER_FT3
    for m in MONOMERS:
        measured = df[f'C_{m}'] * (1 + SIGMA * rng.standard_normal(len(df)))
        out = litres_per_hr * np.maximum(measured, 0.0) * MW[m] / model.G_PER_LB
        df[f'in_out_{m}'] = df[f'flow_{m}'] - out
    return df


INOUT = node.Target('inout', node.PROPERTIES.outputs, node.PROPERTIES.log_outputs, MODEL_PATH, add_in_out,
                    inputs=tuple(node.input_vars) + FEATURES)


def predict(csv_path, grade=None):
    grade = grade or node.grade_of(csv_path)
    checkpoint = torch.load(MODEL_PATH, weights_only=False)
    if checkpoint.get('target') != INOUT.name or checkpoint.get('input_vars') != list(INOUT.inputs):
        sys.exit(f"{MODEL_PATH.name} is not the grey-box in/out model. Train it: python neural_ode_inout.py")
    scales = (checkpoint['x_mean'], checkpoint['x_std'], checkpoint['y_mean'], checkpoint['y_std'])
    func = node.ODEFunc(latent_dim=checkpoint['latent_dim'], input_dim=INOUT.input_dim)
    func.load_state_dict(checkpoint['model_state_dict'])
    func.eval()

    t_np, pred, true = node.forecast(func, node.load_run(csv_path, INOUT), grade, scales, INOUT)
    r2 = node.r2_per_output(pred, true)
    print(f"R2 on {csv_path} (grade {grade}): mean {r2.mean():.3f} | "
          + ", ".join(f"{name} {v:.3f}" for name, v in zip(INOUT.outputs, r2)))

    fig, axes = plt.subplots(4, 2, figsize=(14, 18), sharex=True)
    axes = axes.flatten()
    for i, name in enumerate(INOUT.outputs):
        axes[i].plot(t_np, true[:, i], label="Simulation", color="blue", linewidth=2, linestyle="--")
        axes[i].plot(t_np, pred[:, i], label="Neural ODE (grey-box)", color="red", linewidth=2)
        axes[i].set_title(f"Grade {grade}: {name} (R² {r2[i]:.3f})", fontsize=12, pad=10)
        axes[i].grid(True, linestyle=":", alpha=0.7)
        if i >= len(INOUT.outputs) - 2:
            axes[i].set_xlabel("Time ($t$)", fontsize=11)
    axes[0].legend(fontsize=9)
    axes[-1].axis('off')
    plt.subplots_adjust(hspace=0.55, wspace=0.25, top=0.94, bottom=0.06)
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == 'predict':
        if len(sys.argv) < 3:
            sys.exit("Usage: python neural_ode_inout.py predict <run.csv> [grade]")
        predict(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else None)
    else:
        node.main(INOUT)
