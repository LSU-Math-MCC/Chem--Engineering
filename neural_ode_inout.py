"""
Neural ODE for the reactor as a flow balance: feed flows in -> stream flows out.

    python neural_ode_inout.py                             # train (same data and steps as neural_ode_code_taking_csv.py)
    python neural_ode_inout.py predict <run.csv> [grade]   # predict a whole run and plot it

Inputs are the constant feed rates (lb/hr of ethylene, propylene, diene, hexane, catalyst,
Al alkyl, hydrogen), the inlet temperature and the grade. Outputs are what leaves the reactor,
in lb/hr: the unreacted monomers, hydrogen, Al alkyl and catalyst, and the polymer made (PR).
They are computed from each run's outlet flow F and outlet concentrations, so any simulator CSV
works and no new data is needed. Hexane is left out: it is inert and not tracked, so out = in.
These are the simulator's true (noise-free) values; they are not analyzer readings.
"""

import sys

import torch
import matplotlib.pyplot as plt

import model
from derivatives import L_PER_FT3
import neural_ode_code_taking_csv as node

MODEL_PATH = node.HERE / 'neural_ode_inout_model.pth'
MW = {key: mw for key, _, mw in model.STREAMS if mw is not None}
MW.update(model.DEFAULT_CONFIG['p0']['Mw'])           # monomer weights (the same in all four grades)
OUTLETS = {'out_M1': (['C_M1'], MW['M1']), 'out_M2': (['C_M2'], MW['M2']), 'out_M3': (['C_M3'], MW['M3']),
           'out_H2': (['C_H2'], MW['H2']), 'out_Al': (['C_Al'], MW['Al']),
           'out_cat': (['C_C1', 'C_C2'], MW['cat'])}   # inactive + active catalyst, as VCl4


def add_outlet_flows(df):
    """Outlet mass flows, lb/hr = F (ft3/hr) x L/ft3 x C (mol/L) x MW (g/mol) / (g/lb)."""
    df = df.copy()
    litres_per_hr = df['F'] * L_PER_FT3
    for name, (states, mw) in OUTLETS.items():
        df[name] = litres_per_hr * df[states].sum(axis=1) * mw / model.G_PER_LB
    return df


# Every output is a positive flow that can span orders of magnitude, so all are log-scaled.
INOUT = node.Target('inout', tuple(OUTLETS) + ('PR',), tuple(OUTLETS) + ('PR',), MODEL_PATH, add_outlet_flows)


def predict(csv_path, grade=None):
    grade = grade or node.grade_of(csv_path)
    checkpoint = torch.load(MODEL_PATH, weights_only=False)
    if checkpoint.get('target') != INOUT.name or checkpoint.get('output_vars') != list(INOUT.outputs):
        sys.exit(f"{MODEL_PATH.name} is not an in/out model. Train it: python neural_ode_inout.py")
    scales = (checkpoint['x_mean'], checkpoint['x_std'], checkpoint['y_mean'], checkpoint['y_std'])
    func = node.ODEFunc(latent_dim=checkpoint['latent_dim'], output_dim=len(INOUT.outputs))
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
        axes[i].plot(t_np, pred[:, i], label="Neural ODE", color="red", linewidth=2)
        axes[i].set_title(f"Grade {grade}: {name}, lb/hr (R² {r2[i]:.3f})", fontsize=12, pad=10)
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
