"""
Discrete-step simulation driver for the Cozewith '88 CSTR model.

Structure (from the whiteboard):

    der(t, states, u, p)          -> dstates/dt
    move(states(k), u(k), p(k))   -> states(k+1)     integrate one step dt
    simulate(states(0), U, P)     -> states(0..N)    call move N times
    properties(states, u, p)      -> Mn, Mw, PDI, polymerization rate, ...
    write_csv(...)                -> results.csv

u(k) holds the manipulated inputs that can change every step: the mass feed
flows F_i of each stream (lb/hr) and the inlet temperature T_i (deg F).
From "more equations.pdf":

    outlet flow   F = sum_i F_i / rho(T_i)        (ft3/hr)
    molar feeds   m_i,f = F_i / MW_i              (lbmol/hr)

p(k) holds the reactor volume (ft3), rate constants, molecular weights and
the energy-balance data; it may also change per step (e.g. catalyst decay).

The component and moment equations come from derivatives.py, which was
checked term-by-term against A.54-A.82 and the dead-chain second moments.
derivatives.py works in mol/L, L and s, so operating_point() converts the
lb/hr and ft3 inputs; temperatures stay in deg F.
"""

import csv
from copy import deepcopy

import numpy as np
from scipy.integrate import solve_ivp

from derivatives import L_PER_FT3, STATE_VARS, derivatives, stream_density


KINETIC_KEYS = [
    "k_i1", "k_i2",
    "k11", "k12", "k13", "k21", "k22", "k31",
    "k_a", "k_x", "k_x2", "k_x3",
    "k_t", "k_t2", "k_t3",
    "k_tr", "k_tr1", "k_trM2",
]

# Feed streams: key, model state it feeds (None = solvent only adds volume),
# molecular weight (lb/lbmol; monomers use p["Mw"] instead).
STREAMS = [
    ("M1", "C_M1", None), ("M2", "C_M2", None), ("M3", "C_M3", None),
    ("C6", None, 86.0),          # hexane solvent
    ("cat", "C_C1", 192.75),     # VCl4, fed as inactive catalyst
    ("Al", "C_Al", 120.56),      # aluminum alkyl co-catalyst (DEAC)
    ("H2", "C_H2", 2.0),
]
STREAM_KEYS = [key for key, _, _ in STREAMS]

G_PER_LB = 453.59237
MOL_S_PER_LBMOL_HR = G_PER_LB / 3600.0
L_S_PER_FT3_HR = L_PER_FT3 / 3600.0

# Reactor temperature range where the density correlation holds (deg F).
T_RANGE = (-200.0, 450.0)

PROPERTY_NAMES = [
    "live_chains", "dead_chains",
    "polymer_conc",
    "Mn", "Mw", "PDI",
    "wt_ethylene", "wt_propylene", "wt_diene",
    "X_M1", "X_M3",
    "PR", "ML",
    "conv_M1", "conv_M2", "conv_M3",
]


def operating_point(u, p):
    """Outlet flow and molar feeds from the mass flows.

    Returns (F in L/s, feed in mol/s keyed by state, F in ft3/hr).
    """
    F_ft3_hr = sum(u["flows"].values()) / stream_density(u["T_i"])
    feed = {}
    for key, state, mw in STREAMS:
        if state is not None:
            mw = p["Mw"][key] if mw is None else mw
            feed[state] = u["flows"][key] / mw * MOL_S_PER_LBMOL_HR
    return F_ft3_hr * L_S_PER_FT3_HR, feed, F_ft3_hr


# ---------------------------------------------------------------------------
# der / move / simulate
# ---------------------------------------------------------------------------

def der(t, states, u, p):
    """dstates/dt for inputs u and parameters p (derivation A.54-A.82+)."""
    if states[STATE_VARS.index("C_C1")] <= 0.0:
        raise ValueError("C_C1 must stay positive: the model contains C_Al/C_C1.")
    T = states[STATE_VARS.index("T")]
    if not T_RANGE[0] < T < T_RANGE[1]:
        raise ValueError(f"Reactor temperature reached {T:.0f} deg F, outside the "
                         f"{T_RANGE[0]:.0f} to {T_RANGE[1]:.0f} deg F range of the density correlation.")
    F, feed, _ = operating_point(u, p)
    rates = {key: p[key] for key in KINETIC_KEYS}
    rates.update(F=F, V=p["V"] * L_PER_FT3)
    rates["energy"] = {"T_i": u["T_i"], "Mw": p["Mw"], "neg_dH": p["neg_dH"], "H_mix": p["H_mix"]}
    return derivatives(t, states, rates, feed=feed)


def move(states, u, p, dt, t=0.0, rtol=1e-8, atol=1e-14):
    """Integrate from t to t+dt with u and p held constant; return states(k+1)."""
    sol = solve_ivp(
        der, (t, t + dt), np.asarray(states, dtype=float),
        args=(u, p), method="LSODA", rtol=rtol, atol=atol,
    )
    if not sol.success:
        raise RuntimeError(f"move failed at t={t}: {sol.message}")
    return sol.y[:, -1]


def simulate(states0, U, P, dt, t0=0.0):
    """Apply move N times, with N = len(U) = len(P).

    Returns t (N+1,), X (N+1, n_states) where X[k] is states(k).
    u(k), p(k) act over [t_k, t_k+1].
    """
    if len(U) != len(P):
        raise ValueError(f"U and P must have the same length ({len(U)} vs {len(P)}).")

    X = [np.asarray(states0, dtype=float)]
    t = [float(t0)]
    for k in range(len(U)):
        X.append(move(X[-1], U[k], P[k], dt, t=t[-1]))
        t.append(t[-1] + dt)
    return np.array(t), np.array(X)


# ---------------------------------------------------------------------------
# Properties from the moments ("more equations.pdf", pp. 2-3)
# ---------------------------------------------------------------------------

def properties(states, u, p):
    """Polymer properties from one state vector.

    Uses total (live + dead) moments:
        lambda_0    = P0 + Q0 + R0 + B0
        lambda_ijk  = first/second moments of live + dead chains
    Molecular weights come from p["Mw"] (M1, M2, M3 repeat-unit weights).
    """
    s = dict(zip(STATE_VARS, states))
    m1, m2, m3 = p["Mw"]["M1"], p["Mw"]["M2"], p["Mw"]["M3"]

    live0 = s["P0"] + s["Q0"] + s["R0"]
    lam0 = live0 + s["B0"]

    lam100 = s["P100"] + s["Q100"] + s["R100"] + s["B100"]
    lam010 = s["P010"] + s["Q010"] + s["R010"] + s["B010"]
    lam001 = s["P001"] + s["Q001"] + s["R001"] + s["B001"]

    lam200 = s["L200"] + s["B200"]
    lam020 = s["L020"] + s["B020"]
    lam002 = s["L002"] + s["B002"]
    lam110 = s["L110"] + s["B110"]
    lam101 = s["L101"] + s["B101"]
    lam011 = s["L011"] + s["B011"]

    # Mass of polymer per volume (P1^m, g/L), and sum of (chain mass)^2 (P2^m).
    mass1 = m1 * lam100 + m2 * lam010 + m3 * lam001
    mass2 = (m1**2 * lam200 + m2**2 * lam020 + m3**2 * lam002
             + 2 * m1 * m2 * lam110 + 2 * m1 * m3 * lam101 + 2 * m2 * m3 * lam011)

    Mn = mass1 / lam0 if lam0 > 0 else np.nan
    Mw = mass2 / mass1 if mass1 > 0 else np.nan
    PDI = Mw / Mn if Mn > 0 else np.nan

    F, feed, _ = operating_point(u, p)

    def conversion(state):
        fed = feed.get(state, 0.0)
        return (fed - F * s[state]) / fed if fed > 0 else np.nan

    def pct(num, den):
        return 100 * num / den if den > 0 else np.nan

    return {
        "live_chains": live0,
        "dead_chains": s["B0"],
        "polymer_conc": mass1,
        "Mn": Mn,
        "Mw": Mw,
        "PDI": PDI,
        "wt_ethylene": pct(m1 * lam100, mass1),
        "wt_propylene": pct(m2 * lam010, mass1),
        "wt_diene": pct(m3 * lam001, mass1),
        # X_M1: ethylene in the ethylene + propylene part; X_M3: diene in all.
        "X_M1": pct(m1 * lam100, m1 * lam100 + m2 * lam010),
        "X_M3": pct(m3 * lam001, mass1),
        # Polymerization rate PR = F * lambda_0 * Mn = F * P1^m, in lb/hr.
        "PR": F * mass1 * 3600.0 / G_PER_LB,
        # Mooney viscosity ML = (a + b PD) Mw.
        "ML": (p["ML_a"] + p["ML_b"] * PDI) * Mw,
        "conv_M1": conversion("C_M1"),
        "conv_M2": conversion("C_M2"),
        "conv_M3": conversion("C_M3"),
    }


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

def write_csv(path, t, X, U, P):
    """One row per step k: time, inputs u(k), states(k), properties(k).

    F is the outlet flow (ft3/hr) that follows from the inputs. The last row
    (k = N) has no u(N) of its own; it repeats u(N-1) so its properties can
    still be evaluated.
    """
    header = (["step", "t", "F", "T_i"] + [f"flow_{key}" for key in STREAM_KEYS]
              + STATE_VARS + PROPERTY_NAMES)

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for k in range(len(t)):
            u = U[min(k, len(U) - 1)]
            p = P[min(k, len(P) - 1)]
            props = properties(X[k], u, p)
            row = ([k, t[k], operating_point(u, p)[2], u["T_i"]]
                   + [u["flows"][key] for key in STREAM_KEYS]
                   + list(X[k]) + [props[name] for name in PROPERTY_NAMES])
            writer.writerow(["" if isinstance(v, float) and np.isnan(v) else f"{v:.10g}"
                             if isinstance(v, float) else v for v in row])


# ---------------------------------------------------------------------------
# Run configuration
# ---------------------------------------------------------------------------
#
# A run is described by one plain dict (JSON-friendly, so the dashboard can
# send it).  Edit DEFAULT_CONFIG to change the default run.
#
#   N, dt     number of steps and step length (s)
#   u0        baseline inputs: mass flows of each stream (lb/hr) and the
#             inlet temperature T_i (deg F)
#   p0        reactor volume (ft3), initial reactor temperature T0 (deg F),
#             rate constants, molecular weights, energy-balance data,
#             Mooney viscosity constants
#   events    step changes applied on top of u0 / p0:
#               start   first step k it applies to
#               end     last step k (None = until the end of the run)
#               target  "flow.<stream>" (e.g. "flow.H2"), "T_i" or "p.<key>"
#               mode    "multiply" (x value) or "set" (= value)
#               value   number
#               values  instead of value: a list of numbers (e.g. a CSV
#                       column), one per step from start; the last is held
#               source  label for values, e.g. "feeds.csv: H2" (display only)
#
# Flows, MWs, T_i, T, heats of reaction and dH_mix are from the course notes.
# The rate constants (mol/L, s) are not from data: the propagation and
# transfer constants were fitted so the steady state with these feeds hits
# the whiteboard targets (Mn ~50k, conversions ethylene ~87.5%, propylene
# ~30%, diene ~70%). The scaled Table 4.29 grades need their scale factors.

DEFAULT_CONFIG = {
    "N": 150,
    "dt": 10.0,
    "u0": {
        "flows": {"M1": 12000.0, "M2": 20000.0, "M3": 2000.0, "C6": 300000.0,
                  "cat": 50.0, "Al": 156.37, "H2": 2.0},
        "T_i": -32.0,
    },
    "p0": {
        "V": 420.0,
        "T0": 90.0,
        # Initiation / propagation
        "k_i1": 50.0, "k_i2": 5.0,
        "k11": 385.0, "k12": 29.0, "k13": 189.0,
        "k21": 308.0, "k22": 10.9, "k31": 154.0,
        # Catalyst activation / deactivation
        "k_a": 0.5, "k_x": 5e-3, "k_x2": 1e-3, "k_x3": 1e-3,
        # Termination / chain transfer
        "k_t": 0.0202, "k_t2": 1e-3, "k_t3": 1e-3,
        "k_tr": 1.16e-5, "k_tr1": 2.37, "k_trM2": 1e-3,
        "Mw": {"M1": 28.0, "M2": 42.0, "M3": 120.0},
        # Energy balance: -dH_RMj (Btu/lb) and dH_mix (Btu/hr)
        "neg_dH": {"M1": 1400.0, "M2": 950.0, "M3": 700.0},
        "H_mix": 1.0e6,
        # Mooney viscosity ML = (a + b PD) Mw
        "ML_a": 1.0, "ML_b": 1.0,
    },
    "events": [
        # hydrogen feed tripled -> shorter chains
        {"start": 50, "end": None, "target": "flow.H2", "mode": "multiply", "value": 3.0},
        # more propylene -> composition shift
        {"start": 100, "end": None, "target": "flow.M2", "mode": "multiply", "value": 1.5},
    ],
}

MAX_STEPS = 2000

# Keys of p0 that a step change may target.
PARAM_TARGETS = ["V", "H_mix"] + KINETIC_KEYS


def _number(value, name, positive=False, allow_zero=True):
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number (got {value!r}).") from None
    if not np.isfinite(value):
        raise ValueError(f"{name} must be a finite number.")
    if positive and (value < 0 or (value == 0 and not allow_zero)):
        raise ValueError(f"{name} must be {'positive' if not allow_zero else 'zero or positive'}.")
    return value


def _temperature(value, name):
    value = _number(value, name)
    if not T_RANGE[0] < value < T_RANGE[1]:
        raise ValueError(f"{name} must be between {T_RANGE[0]:.0f} and {T_RANGE[1]:.0f} deg F.")
    return value


def validate_config(config):
    """Return a cleaned copy of config, or raise ValueError saying what is wrong."""
    try:
        return _validate(deepcopy(config))
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError(f"The inputs are missing a value or use an older format ({exc}).") from None


def _validate(cfg):
    n = cfg.get("N")
    if not isinstance(n, (int, float)) or int(n) != n or not 1 <= n <= MAX_STEPS:
        raise ValueError(f"Number of steps must be a whole number from 1 to {MAX_STEPS}.")
    cfg["N"] = int(n)
    cfg["dt"] = _number(cfg.get("dt"), "Step size dt", positive=True, allow_zero=False)

    u0 = cfg["u0"]
    flows = u0["flows"]
    for key in list(flows):
        if key not in STREAM_KEYS:
            raise ValueError(f"Unknown feed stream {key!r}.")
    for key in STREAM_KEYS:
        flows[key] = _number(flows.get(key), f"Feed flow {key}", positive=True)
    if flows["cat"] <= 0.0:
        raise ValueError("Catalyst flow must be above zero: the model divides by C_C1.")
    if sum(flows.values()) <= 0.0:
        raise ValueError("The total feed flow must be above zero.")
    u0["T_i"] = _temperature(u0.get("T_i"), "Inlet temperature T_i")

    p0 = cfg["p0"]
    p0["V"] = _number(p0.get("V"), "Volume V", positive=True, allow_zero=False)
    p0["T0"] = _temperature(p0.get("T0"), "Starting reactor temperature")
    for key in KINETIC_KEYS:
        p0[key] = _number(p0.get(key), key, positive=True)
    for key in ("M1", "M2", "M3"):
        p0["Mw"][key] = _number(p0["Mw"].get(key), f"Molecular weight {key}", positive=True, allow_zero=False)
        p0["neg_dH"][key] = _number(p0["neg_dH"].get(key), f"Heat of reaction {key}", positive=True)
    p0["H_mix"] = _number(p0.get("H_mix"), "Heat of mixing")
    p0["ML_a"] = _number(p0.get("ML_a"), "ML constant a")
    p0["ML_b"] = _number(p0.get("ML_b"), "ML constant b")

    for i, e in enumerate(cfg.get("events", []), start=1):
        label = f"Step change {i}"
        target = e.get("target", "")
        if not ((target.startswith("flow.") and target[5:] in STREAM_KEYS)
                or target == "T_i"
                or (target.startswith("p.") and target[2:] in PARAM_TARGETS)):
            raise ValueError(f"{label}: unknown input {target!r}.")
        if e.get("mode") not in ("multiply", "set"):
            raise ValueError(f"{label}: mode must be 'multiply' or 'set'.")
        # Temperatures can be below zero deg F; everything else is zero or more.
        signed = target == "T_i" and e["mode"] == "set"
        if "values" in e:
            values = e["values"]
            if not isinstance(values, list) or not 1 <= len(values) <= MAX_STEPS:
                raise ValueError(f"{label}: the CSV data must have 1 to {MAX_STEPS} values.")
            e["values"] = [_number(v, f"{label} CSV row {j}", positive=not signed)
                           for j, v in enumerate(values, start=1)]
            e["source"] = str(e.get("source", "CSV"))[:200]
            e.pop("value", None)
        else:
            e["value"] = _number(e.get("value"), f"{label} value", positive=not signed)
        e["start"] = int(_number(e.get("start"), f"{label} start step", positive=True))
        if e.get("end") not in (None, ""):
            e["end"] = int(_number(e["end"], f"{label} end step", positive=True))
            if e["end"] < e["start"]:
                raise ValueError(f"{label}: end step is before the start step.")
        else:
            e["end"] = None
    return cfg


def _event_value(event, k):
    """Value of a step change at step k. With CSV data ("values"), row 1 is used
    at the start step, row 2 at the next step, ...; the last row is held after
    the data runs out."""
    if "values" in event:
        values = event["values"]
        return values[min(k - event["start"], len(values) - 1)]
    return event["value"]


def _apply(event, u, p, k):
    target, value = event["target"], _event_value(event, k)
    if target == "T_i":
        holder, key = u, "T_i"
    elif target.startswith("flow."):
        holder, key = u["flows"], target[5:]
    else:
        holder, key = p, target[2:]
    old = holder.get(key, 0.0)
    holder[key] = old * value if event["mode"] == "multiply" else value


def build_inputs(config=DEFAULT_CONFIG):
    """Turn a run config into (x0, U, P, dt) for simulate()."""
    cfg = validate_config(config)
    u0, p0 = cfg["u0"], cfg["p0"]

    U, P = [], []
    for k in range(cfg["N"]):
        u, p = deepcopy(u0), deepcopy(p0)
        for e in cfg["events"]:
            if e["start"] <= k and (e["end"] is None or k <= e["end"]):
                _apply(e, u, p, k)
        if u["flows"]["cat"] <= 0.0:
            raise ValueError(f"Catalyst flow drops to zero at step {k}; the model divides by C_C1.")
        if not T_RANGE[0] < u["T_i"] < T_RANGE[1]:
            raise ValueError(f"Inlet temperature leaves the {T_RANGE[0]:.0f} to {T_RANGE[1]:.0f} deg F range at step {k}.")
        U.append(u)
        P.append(p)

    # Reactor initially full of the feed mixture, no catalyst or polymer yet,
    # at the starting reactor temperature T0.
    F, feed, _ = operating_point(u0, p0)
    x0 = np.zeros(len(STATE_VARS))
    for state, value in feed.items():
        x0[STATE_VARS.index(state)] = value / F
    x0[STATE_VARS.index("T")] = p0["T0"]
    return x0, U, P, cfg["dt"]


def run(config=DEFAULT_CONFIG, csv_path="results.csv"):
    """Build inputs from config, simulate, and write the CSV."""
    x0, U, P, dt = build_inputs(config)
    t, X = simulate(x0, U, P, dt)
    write_csv(csv_path, t, X, U, P)
    return t, X, U, P


def example_inputs():
    """(x0, U, P) for DEFAULT_CONFIG; kept for scripts that used it."""
    x0, U, P, _ = build_inputs(DEFAULT_CONFIG)
    return x0, U, P


if __name__ == "__main__":
    t, X, U, P = run(DEFAULT_CONFIG, "results.csv")
    last = properties(X[-1], U[-1], P[-1])
    T = X[-1][STATE_VARS.index("T")]
    print(f"Simulated {len(U)} steps of {DEFAULT_CONFIG['dt']:g} s -> results.csv")
    print(f"Final Mn = {last['Mn']:.4g}, Mw = {last['Mw']:.4g}, PDI = {last['PDI']:.3f}, "
          f"PR = {last['PR']:.4g} lb/hr, T = {T:.1f} F, ML = {last['ML']:.4g}")
