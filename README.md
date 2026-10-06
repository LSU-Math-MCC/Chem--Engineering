# EPDM Reactor Simulator

A dynamic simulation of a continuous stirred-tank reactor (CSTR) making EPDM
rubber (ethylene–propylene–diene) with a Ziegler–Natta catalyst. The kinetics
follow Cozewith (1988). The model tracks 37 states: 7 species
concentrations, 30 moments of the chain-length distribution, and the reactor
temperature. From those it reports the polymer properties a plant would watch:
molecular weight, dispersity, composition, conversion, production rate and
Mooney viscosity.

You can run it from the command line or from an interactive dashboard, and you
can plan a run (step changes, ramps, recorded plant data) in a CSV file.

All the equations are in **[EQUATIONS.md](EQUATIONS.md)**.

---

## Quick start

Needs Python 3 with `numpy` and `scipy`.

```bash
python dashboard_server.py        # then open http://127.0.0.1:8000
```

The dashboard runs the model when it starts. Edit the inputs on the page and
press **Run simulation** to run it again.

Other ways to run it:

```bash
python model.py                       # default run -> results.csv + results_config.json
python model.py plan_example.csv      # run a plan CSV instead (see below)
python dashboard.py                   # results.csv -> dashboard.html (static, no editing)
```

---

## Files

| File | What it does |
|---|---|
| `derivatives.py` | The model equations: dy/dt for all 37 states (A.54–A.82, the dead-chain second moments, and the energy balance). |
| `model.py` | Runs the simulation. Turns lb/hr feeds into model units, steps through time, computes polymer properties, writes `results.csv`. Holds `DEFAULT_CONFIG`, the default run. |
| `plan.py` | Reads a plan CSV and turns it into a run config. |
| `dashboard_server.py` | Local web server for the editable dashboard (127.0.0.1 only). |
| `dashboard.py` / `dashboard_template.html` | Builds the dashboard page. `dashboard.html` is a standalone snapshot. |
| `results.csv` | Output of the last run: one row per step. |
| `results_config.json` | The inputs that produced `results.csv`. |
| `plan_example.csv`, `plan_timeline_example.csv`, `fake_plant_schedule.csv` | Example plan files. |
| `cozewith88_model_36state.py` | An earlier standalone version of the model. The simulator does not use it. |

---

## How a run works

The simulation is a series of steps of length Δt. During each step, the inputs
are held constant and the equations are integrated over that step:

```
config  ──►  build_inputs()  ──►  u(0..N-1), p(0..N-1), states(0)
                                        │
                     for k = 0 … N-1:   ▼
                     states(k+1) = move(states(k), u(k), p(k), Δt)     # LSODA over [t_k, t_k+Δt]
                                        │
                                        ▼
                     properties(states(k), u(k), p(k))  ──►  results.csv
```

- **u(k), the inputs:** the mass feed flow of each stream (lb/hr) and the inlet
  temperature Tᵢ (°F). These are what an operator can change.
- **p(k), the parameters:** reactor volume V, rate constants, molecular weights,
  heats of reaction, heat of mixing, and the Mooney constants a and b. A step
  change can alter some of these too (V, ΔH_mix and the rate constants), for
  example to imitate catalyst decay.
- **Starting state:** the reactor starts full of the feed mixture at temperature
  T0, with no active catalyst or polymer yet.
- **Feeds:** seven streams: ethylene (M1), propylene (M2), diene (M3), hexane
  solvent (C6), catalyst VCl₄ (fed in its inactive form), aluminum alkyl
  co-catalyst, and hydrogen.
- **Solver:** each step uses SciPy's `solve_ivp` with LSODA (rtol 1e-8,
  atol 1e-14). The model is stiff, because catalyst activation and chain
  growth are far faster than the residence time.

### What a run is made of (the config)

A run is described by one JSON-style dict (see `DEFAULT_CONFIG` in `model.py`):

| Key | Meaning |
|---|---|
| `N`, `dt` | Number of steps and step length in seconds (default 150 × 10 s). |
| `u0` | Starting inputs: `flows` (lb/hr per stream) and `T_i` (°F). |
| `p0` | Parameters: `V` (ft³), `T0` (°F), rate constants, `Mw`, `neg_dH`, `H_mix`, `ML_a`, `ML_b`. |
| `events` | Step changes applied on top of `u0`/`p0` (see below). |

Each event has `start` and `end` (steps, where `end: null` means until the
end of the run), `target` (`flow.H2`, `T_i`, `p.k11` …), `mode` (`set` or
`multiply`), and either a single `value` or a list of `values`, one per step.

---

## The dashboard

`python dashboard_server.py` serves the editable version.

- **Simulation inputs:** run length, starting feeds and inlet temperature, step
  changes, and all parameters. **Run simulation** reruns the model and updates
  `results.csv` and `results_config.json`. **Load plan CSV…** fills the whole
  form from a plan file.
- **Time slider and summary tiles:** pick any step. The tiles, the equations and
  the full data table all show values at that time.
- **Changes during the run:** lists the step changes. When you've edited the
  form but haven't run it yet, it says **not run yet** and has a dashed border.
- **Equations panel:** the equations behind each property, with the numbers
  at the selected time plugged in, e.g. `PDI = Mw / Mn = 86,987 / 43,566 = 1.997`.
- **Charts:** the inputs u(k), temperature, polymerization rate, molecular
  weight, dispersity, composition, conversion, Mooney viscosity and chain counts.
  Hover to read values, or click to select a time.

---

## Planning a run with a CSV file

A plan CSV sets up a whole run. Load it with **Load plan CSV…** on the
dashboard, or run `python model.py plan.csv`. The header row tells the program
which of two layouts the file uses.

### 1. List layout: one row per setting or change

```csv
input,value,change,start,end
N,200
dt,10
Ethylene,12000
T0,90
k11,800
H2,3,multiply,50,
T_i,-20,set,80,120
```

- A row with **no start step** sets a starting value: run length, step size,
  a feed flow, Tᵢ, T0, or any parameter (`k11`, `V`, `Mw.M1`, `ML_a` …).
- A row **with a start step** is a step change. `change` is `set` (the default)
  or `multiply`. A blank `end` means it lasts until the end of the run.

### 2. Timeline layout: one row per time, one column per input

```csv
step,H2,T_i,Catalyst
0,2,-32,50
50,6,-32,50
80,6,-20,50
```

- The first column is `step`, or `t` in seconds (rounded to the nearest step).
- Each value holds until the next row. A blank cell keeps the previous value.
- If the last row is past the end of the run, the run is lengthened to cover it.

This layout suits recorded plant data or a schedule with ramps. For an example,
see `fake_plant_schedule.csv`.

### Input names

Any of these work: `flow_M1`, `flow.M1`, `M1`, `Ethylene`, `Ethylene, lb/hr`
(text after a comma or bracket is ignored). The streams are Ethylene / M1,
Propylene / M2, Diene / M3, Hexane / C6, Catalyst / cat, Aluminum / Al and
Hydrogen / H2. The other inputs are `T_i`, `T0`, `V`, `H_mix`, the rate constants
(`k11`, `k_tr1` …), `Mw.M1`–`Mw.M3`, `neg_dH.M1`–`neg_dH.M3`, `ML_a`, `ML_b`,
`N` and `dt`.

A mistake in the file gives an error naming the row, for example
`plan.csv row 4 (H2): change must be 'set' or 'multiply'`.

---

## Output: results.csv

There is one row per step k = 0 … N:

| Columns | Content |
|---|---|
| `step`, `t` | Step number and time (s) |
| `F`, `T_i`, `flow_*` | Outlet flow (ft³/hr), inlet temperature (°F), feed flows (lb/hr) |
| `C_M1` … `C_Al` | Species concentrations (mol/L) |
| `P0` … `B011` | Moments of the chain-length distribution |
| `T` | Reactor temperature (°F) |
| `Mn`, `Mw`, `PDI`, `ML` | Molecular weights (g/mol), dispersity, Mooney viscosity |
| `PR`, `polymer_conc` | Polymerization rate (lb/hr), polymer concentration (g/L) |
| `wt_ethylene`, `wt_propylene`, `wt_diene`, `X_M1`, `X_M3` | Composition (wt %) |
| `conv_M1` … `conv_M3` | Monomer conversion (fraction) |
| `live_chains`, `dead_chains` | Chain counts (mol/L) |

The last row (k = N) has no input of its own, so it repeats u(N−1).

---

## Limits and assumptions

- In the energy balance, the reactor contents are treated as hexane:
  enthalpy, heat capacity and density all come from hexane correlations.
  Reaction heat is the only exception. The density correlation holds from
  −200 to 450 °F, and the run stops with a message if T leaves that range.
- The catalyst feed must stay above zero, because the equations divide by C_C1.
- The rate constants are not measured data. With the defaults, the run ends
  near Mn ≈ 22,000 and conversions of about 93 % ethylene, 30 % propylene and
  65 % diene.
- Conversion is (fed − leaving) / fed. During a transient, that includes
  monomer building up in the reactor.
