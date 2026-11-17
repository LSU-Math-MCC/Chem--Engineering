# EPDM Reactor Simulator

A dynamic simulation of a continuous stirred-tank reactor (CSTR) making EPDM
rubber (ethylene–propylene–diene) with a Ziegler–Natta catalyst. The kinetics
follow Cozewith (1988). The model tracks 36 states: 7 species
concentrations, 28 moments of the chain-length distribution (4 zeroth, 12 first,
12 second), and the reactor temperature. From those it reports the polymer properties a plant would watch:
molecular weight, dispersity, composition, conversion, production rate and
Mooney viscosity. It can also report them as a plant would *measure* them, with
optional noise (see [Measurements and noise](#measurements-and-noise)).

You can run it from the command line or from an interactive dashboard, plan a
run (step changes, ramps, recorded plant data) in a CSV file, and close the loop
with your own controller code.

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
python model.py                            # default run -> results.csv + results_config.json
python model.py my_plan.csv                # run a plan CSV instead (see below)
python model.py --controller hold_plan     # closed loop with controllers/hold_plan.py
python dashboard.py                        # results.csv -> dashboard.html (static, no editing)
```

The dashboard server reloads the last saved `results_config.json` when it
starts. Press **Reset to defaults** on the page to go back to `DEFAULT_CONFIG`.

---

## Files

| File | What it does |
|---|---|
| `derivatives.py` | The model equations: dy/dt for all 36 states (A.54–A.82, the dead-chain second moments, and the energy balance). |
| `model.py` | Runs the simulation. Turns lb/hr feeds into model units, steps through time, computes polymer properties, writes `results.csv`. Holds `DEFAULT_CONFIG`, the default run. |
| `plan.py` | Reads a plan CSV and turns it into a run config. |
| `controllers/` | Closed-loop controller files: `hold_plan.py` (template) and `example_pi_temperature.py` (example). |
| `dashboard_server.py` | Local web server for the editable dashboard (127.0.0.1 only). |
| `dashboard.py` / `dashboard_template.html` | Builds the dashboard page. `dashboard.html` is a standalone snapshot. |
| `results.csv` | Output of the last run: one row per step. |
| `results_config.json` | The inputs that produced `results.csv`. |
| `EQUATIONS.md` | Every equation the simulator solves, and the default parameters. |
| `cozewith88_model_36state.py` | An earlier standalone version of the model. The simulator does not use it. |
| `EPDM_Data_Generation/` | Four reference datasets (grades 1–4: four parameter sets) from this simulator, their configs, and the shared DOE step-test plan. See its README. |
| `generate_training_data.py`, `neural_ode_code_taking_csv.py`, `prediction_model.py`, `nn_and_node_research.py` | Research scripts: simulator runs in the style of `EPDM_Data_Generation` (the DOE steps in random orders, for each grade) written to `data/`, one neural ODE trained on them for all grades (inputs `T_i`, `flow_*` and the grade; outputs the `*_meas` columns) and saved to `neural_ode_reactor_model.pth`, and a script that predicts a whole run with it (`python prediction_model.py <run.csv> [grade]`). Run them in that order; the four reference datasets are the test. Not used by the simulator; they need `torch`, `torchdiffeq`, `optuna`, `pandas` and `matplotlib`. |
| `neural_ode_inout.py` | A second neural ODE on the same data, framed as a flow balance: feed flows in (lb/hr) → stream flows out (lb/hr of unreacted ethylene, propylene, diene, H₂, Al alkyl and catalyst, plus polymer made, PR), computed from each run's `F` and outlet concentrations. `python neural_ode_inout.py` trains it to `neural_ode_inout_model.pth`; `python neural_ode_inout.py predict <run.csv> [grade]` predicts a whole run. |

`.csv` and `.json` files are git-ignored, so plan files and results stay local.
The plan examples below can be copied into a file.

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
                     measure(properties, T, noise)      ──►  *_meas columns
```

- **u(k), the inputs:** the mass feed flow of each of the seven streams (lb/hr)
  and the inlet temperature Tᵢ (°F). These are what an operator, or a
  controller, can change every step.
- **p(k), the parameters:** reactor volume V, rate constants, molecular weights,
  heats of reaction, heat of mixing, the Mooney constants a and b, the
  long chain branching constants, and the critical catalyst concentration. A
  step change can alter some of these too (V, ΔH_mix and the rate constants),
  for example to imitate catalyst decay.
- **Starting state:** by default (the Grade 1 base case) the reactor starts
  **empty**: every concentration 0, at T0 = 45 °F. Under *Initial conditions*
  you can give any species its own starting concentration, or leave it blank
  to start at the feed composition (the reactor full of feed).
- **Rate constants:** each one follows the Arrhenius equation
  k = k₀·exp(−E·(1/T − 1/Tr)) at the current reactor temperature. Defaults
  are Tr = 90 °F, E = 1000 °R for the propagation constants (k11 … k31) and
  E = 0 (constant k) for the rest.
- **Long chain branching:** Mw can be scaled by a factor α once branching sets
  in, controlled by `LCB_kb` and `LCB_beta`. The defaults (0 and 1) switch it off.
  See [EQUATIONS.md](EQUATIONS.md), section 8.
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
| `p0` | Parameters: `V` (ft³), `T0` (°F), rate constants (k₀ at `T_ref`), `Mw`, `neg_dH`, `H_mix`, `ML_a`, `ML_b`, the long chain branching constants `LCB_kb` and `LCB_beta`, the critical catalyst concentration `CC1_crit` (mol/L), and the Arrhenius data `T_ref` (°F) and `E` (°R, one per rate constant). |
| `x0` | Starting concentrations (mol/L) by species, e.g. `{"C_M1": 0.5}`. Species left out start at the feed composition. The default sets all seven to 0 (empty reactor). |
| `events` | Step changes applied on top of `u0`/`p0` (see below). |
| `controllers` | Closed loop: a list of `{"name": file in controllers/, "params": {SETTING: value}}`, run in order every step; `[]` for open loop. The older `"controller": "name"` is still accepted. |
| `measure` | What is measured and its noise; see [Measurements and noise](#measurements-and-noise). |

Each event has `start` and `end` (steps, where `end: null` means until the
end of the run), `target` (`flow.H2`, `T_i`, `p.k11` …), `mode` (`set` or
`multiply`), and either a single `value` or a list of `values`, one per step.

### Measurements and noise

A plant can't see the true state; it reads instruments. The `measure` entry
says which properties are measured and whether the readings are noisy:

```python
"measure": {"case": 1, "noise": False, "seed": 0,
            "sigma": {"ML": 0.02, "Mw": 0.02, "Mn": 0.02, "X_M1": 0.02,
                      "X_M3": 0.02, "T": 1.0, "PR": 0.02}}
```

| Case | Measured |
|---|---|
| 1 | ML, X_C2 (`X_M1`, ethylene in the C2 + C3 part), X_diene (`X_M3`), T, PR |
| 2 | ML, Mw, Mn, X_C2, X_diene, T, PR |

- **Only the measured variables get noise.** PDI, wt %, conversions and the
  states stay exact.
- **Noise** (when `noise` is on): each reading gets e ~ N(0, 1), a new draw each
  step. Most readings are `value · (1 + σ·e)` with σ a fraction (0.02 = 2 %).
  T is `T + σ·e` with σ in °F, since a percentage of a °F reading has no
  physical meaning.
- **Physical range:** readings are clipped to it: ML, Mw, Mn and PR ≥ 0;
  X_C2 and X_diene between 0 and 100 %. T is not clipped. Before there is
  any polymer, the polymer readings are blank.
- **Seed:** the same seed gives exactly the same noise, so two controllers can
  be compared on identical noise; change it for a different noise pattern. A
  variable gets the same noise in case 1 and case 2.
- **The plant never sees the noise.** It only changes what the controller is
  given and the `_meas` columns. In closed loop, the inputs still change if
  the controller reacts to noisy readings, as in a real plant.

In a plan CSV, set these with `measure.case`, `measure.noise` (1 or 0),
`measure.seed` and `measure.sigma.T`, `measure.sigma.ML` … (starting values
only).

---

## The dashboard

`python dashboard_server.py` serves the editable version.

- **Simulation inputs:** run length, starting feeds and inlet temperature,
  initial conditions (T0 and the starting concentrations; leave a box blank to
  use the feed composition, shown in grey), the closed-loop controller,
  **Measurements and noise** (case, on/off, seed and each σ; the σ boxes for
  Mw and Mn appear only in case 2), step changes, and all parameters,
  including long chain branching (k_b, β). Each rate constant has an **E** box
  next to it, and the Arrhenius section has **Tr**. **Run simulation** reruns
  the model and updates `results.csv` and `results_config.json`. **Load plan
  CSV…** fills the whole form from a plan file, and **Reset to defaults** goes
  back to `DEFAULT_CONFIG`.
- **Time slider and summary tiles:** pick any step. The tiles, the equations and
  the full data table all show values at that time.
- **Changes during the run:** lists the step changes. When you've edited the
  form but haven't run it yet, it says **not run yet** and has a dashed border.
- **Equations panel:** the equations behind each property, with the numbers
  at the selected time plugged in, e.g. `PDI = Mw / Mn = 86,987 / 43,566 = 1.997`.
- **Charts:** the inputs u(k), temperature, polymerization rate, molecular
  weight, dispersity, composition, conversion, Mooney viscosity and chain counts.
  Hover to read values, or click to select a time.
  - The inputs chart shows only inputs that change during the run (up to three),
    as % of their starting value. Tᵢ is on the temperature chart.
  - When noise is on, each measured variable gets a thinner, faded line in the
    same colour as its true line, and a *Measured composition* chart shows
    X_C2 and X_diene. In case 1 the molecular weight chart notes that Mw and Mn
    are not measured.

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

This layout suits recorded plant data or a schedule with ramps.

### Input names

Any of these work: `flow_M1`, `flow.M1`, `M1`, `Ethylene`, `Ethylene, lb/hr`
(text after a comma or bracket is ignored). The streams are Ethylene / M1,
Propylene / M2, Diene / M3, Hexane / C6, Catalyst / cat, Aluminum / Al and
Hydrogen / H2. The other inputs are `T_i`, `T0`, `V`, `H_mix`, the rate constants
(`k11`, `k_tr1` …), `Mw.M1`–`Mw.M3`, `neg_dH.M1`–`neg_dH.M3`, `ML_a`, `ML_b`,
`LCB_kb`, `LCB_beta`, `CC1_crit`, `N` and `dt`. The list layout can also set:

- **Arrhenius data:** `T_ref` (or `Tr`) and `E.k11`, `E.k_tr1` …
- **Starting concentrations:** `C_M1` … `C_Al` (or `init.C_M1`).
- **Measurements:** `measure.case`, `measure.noise`, `measure.seed`,
  `measure.sigma.ML` … `measure.sigma.T`.
- **Controller:** `controller,my_controller` (`a+b` for several, in order; or `none`).

These are starting values only, so they can't be step changes:

```csv
input,value
Tr,90
E.k11,3000
C_M1,0.5
T0,100
```

A mistake in the file gives an error naming the row, for example
`plan.csv row 4 (H2): change must be 'set' or 'multiply'`.

---

## Closed-loop control: code that decides the inputs every step

By default a run is open loop: every input u(k) comes from the starting inputs
and the step changes. In closed loop, your own code is called at **every step**.
It sees the current reactor state and properties, and it decides that step's
inputs: any of the 7 feed flows and the inlet temperature Tᵢ. The simulator only
provides this hook. The control logic is up to you: a rule, a PID, a neural
network, and so on.

### Option A: a controller file

Copy `controllers/hold_plan.py` to `controllers/my_controller.py` and fill in
`control(obs)`:

```python
def reset(config):          # optional: called once before the run
    ...

def control(obs):           # called every step k, before that step is simulated
    T = obs["meas"]["T"]                     # measured reactor temperature, deg F
    ML = obs["meas"]["ML"]                   # measured Mooney viscosity (see "measure")
    h2 = obs["u_planned"]["flows"]["H2"]     # what the plan says for this step
    return {"flows": {"H2": h2 * 1.1}, "T_i": -35.0}   # or None to keep the plan
```

Then pick it in any of three ways:

- **Dashboard:** **Add controller** under Closed-loop controllers in Simulation inputs.
- **Command line:** `python model.py --controller my_controller [plan.csv]`.
- **Plan CSV:** a `controller,my_controller` row in the list layout.

**Settings.** Module-level UPPERCASE number constants (`SETPOINT = 110.0`,
`T_I_MIN, T_I_MAX = -80.0, 40.0`) are the controller's settings. The dashboard
shows a box for each one, and a changed value is set on the module before
`reset()` for that run only; the file is not edited. In a config they are
`{"name": "example_pi_temperature", "params": {"SETPOINT": 105}}`; on the
command line, `--controller example_pi_temperature:SETPOINT=105,KC=1.2`.
Constants computed from other constants at import time keep the file's values.

**Several controllers.** A run can use any number of controllers (add more
rows in the dashboard, repeat `--controller`, or write `a+b` in a plan CSV).
They are called in order every step: each sees `obs["u_planned"]` with the
inputs set by the ones before it, and the last one to set an input wins. For
example, one controller can hold T with Tᵢ while another adjusts H2 for ML.
**Reload files** in the dashboard picks up new controller files or changed
defaults without restarting the server.

`controllers/example_pi_temperature.py` is a worked example: a PI loop that
holds the reactor T at 110 °F by moving Tᵢ. It isn't a recommended design, just
a demonstration of the pattern, including state kept between steps and
anti-windup.

### Option B: drive the loop yourself

For scripts that should own the loop, such as training or testing a neural
network:

```python
import model

sim = model.Reactor(config)            # e.g. model.DEFAULT_CONFIG
while not sim.done:
    obs = sim.observe()
    u = my_logic(obs)                  # dict or None
    sim.step(u)
sim.write_csv("results.csv")           # open it with python dashboard.py
t, X, U, P = sim.results()             # or use the arrays directly
```

### What the controller sees (`obs`)

| Key | Content |
|---|---|
| `k`, `t`, `dt`, `N` | Step number, time (s), step length (s), total steps |
| `meas` | The measured variables of the case (case 1: ML, X_M1, X_M3, T, PR; case 2: + Mw, Mn), noisy when noise is on. A realistic controller uses only these. |
| `state` | All 36 states at t_k: `C_M1` … `C_Al`, the moments, and `T` (°F). `T` is the measured T. |
| `props` | Mn, Mw, PDI, ML, PR, wt %, conversions …, with the measured values for the variables in `meas`; the rest are exact |
| `state_true`, `props_true` | The same without noise (as in `results.csv`) |
| `u_planned` | u(k) from the starting inputs and step changes: `{"flows": {M1, M2, M3, C6, cat, Al, H2}, "T_i": …}` |
| `u_prev` | u(k−1) actually applied (`None` at k = 0) |
| `p` | Parameters p(k): V, rate constants … (read only) |

### What it returns

- **`None`:** keep the planned inputs.
- **A dict with `"flows"`** (any subset of M1, M2, M3, C6, cat, Al, H2, in lb/hr,
  ≥ 0) and/or **`"T_i"`** (°F). Anything left out stays as planned. The
  returned inputs are held over [t_k, t_k + Δt].

Bad values (a negative flow, an unknown stream, Tᵢ outside −200 to 450 °F) stop
the run with a message naming the step.

In `results.csv`, the `flow_*` and `T_i` columns are the inputs the controller
actually applied. On the dashboard, "Changes during the run" shows the
controller's name, and the step changes are marked "(planned)", because those
are what the controller was given, not necessarily what it applied.

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
| `live_chains`, `dead_chains`, `polymer_conc` | Chain counts (mol/L), polymer concentration (g/L) |
| `Mn`, `Mw`, `PDI`, `LCB_alpha` | Molecular weights (g/mol), dispersity, long chain branching factor α (1 = no branching) |
| `wt_ethylene`, `wt_propylene`, `wt_diene` | Composition (wt % of polymer) |
| `X_M1`, `X_M3` | X_C2 (ethylene in the C2 + C3 part) and X_diene (wt %) |
| `PR`, `ML` | Polymerization rate (lb/hr), Mooney viscosity |
| `conv_M1` … `conv_M3` | Monomer conversion (fraction) |
| `ML_meas`, `Mw_meas`, `Mn_meas`, `X_M1_meas`, `X_M3_meas`, `T_meas`, `PR_meas` | Measured values (with noise when on). Blank for a variable the case does not measure (Mw, Mn in case 1) and before there is polymer. |

All columns except `*_meas` are true values. The last row (k = N) has no input
of its own, so it repeats u(N−1). Row k evaluates PR and the conversions with
u(k), while the controller at step k sees them with u(k−1), the inputs that
produced that state, so the two differ slightly at a step change.

---

## Limits and assumptions

- In the energy balance, the reactor contents are treated as hexane:
  enthalpy, heat capacity and density all come from hexane correlations.
  Reaction heat is the only exception. The density correlation holds from
  −200 to 450 °F, and the run stops with a message if T leaves that range.
- The catalyst feed can be 0. Below the critical concentration `CC1_crit`
  (default 1e-7 mol/L), the C_Al/C_C1 term is smoothed so that it goes to 0
  instead of dividing by zero. See [EQUATIONS.md](EQUATIONS.md), section 1.
- The rate constants are not measured data. The defaults are the Grade 1 base
  case (empty reactor at 45 °F); without step changes the run ends near
  Mn ≈ 54,000 and conversions of about 91 % ethylene, 32 % propylene and
  69 % diene.
- Conversion is (fed − leaving) / fed. During a transient, that includes
  monomer building up in the reactor.
