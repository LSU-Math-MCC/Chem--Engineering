"""
Discrete-step simulation driver for the Cozewith '88 CSTR model.

Structure (from the whiteboard):

    der(t, states, u, p)          -> dstates/dt
    move(states(k), u(k), p(k))   -> states(k+1)     integrate one step dt
    simulate(states(0), U, P)     -> states(0..N)    call move N times
    properties(states, u, p)      -> Mn, Mw, PDI, composition, ...
    write_csv(...)                -> results.csv

u(k) holds the manipulated inputs that can change every step: the outlet
flow F and the molar feed rates m_f (keyed like derivatives.STATE_VARS,
e.g. "C_M1", "C_H2").  p(k) holds the reactor volume, rate constants and
molecular weights; it may also change per step (e.g. catalyst decay).

The component and moment equations come from derivatives.py, which was
checked term-by-term against A.54-A.82 and the dead-chain second moments.
The energy balance (A.50) is not included, so the model is isothermal.
"""

import csv
from copy import deepcopy

import numpy as np
from scipy.integrate import solve_ivp

from derivatives import STATE_VARS, derivatives


KINETIC_KEYS = [
    "k_i1", "k_i2",
    "k11", "k12", "k13", "k21", "k22", "k31",
    "k_a", "k_x", "k_x2", "k_x3",
    "k_t", "k_t2", "k_t3",
    "k_tr", "k_tr1", "k_trM2",
]

PROPERTY_NAMES = [
    "live_chains", "dead_chains",
    "polymer_conc",
    "Mn", "Mw", "PDI",
    "wt_ethylene", "wt_propylene", "wt_diene",
    "conv_M1", "conv_M2", "conv_M3",
]


# ---------------------------------------------------------------------------
# der / move / simulate
# ---------------------------------------------------------------------------

def der(t, states, u, p):
    """dstates/dt for inputs u and parameters p (derivation A.54-A.82+)."""
    if states[STATE_VARS.index("C_C1")] <= 0.0:
        raise ValueError("C_C1 must stay positive: the model contains C_Al/C_C1.")
    rates = {key: p[key] for key in KINETIC_KEYS}
    rates.update(F=u["F"], V=p["V"])
    return derivatives(t, states, rates, feed=u["feed"])


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
# Properties from the moments
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

    # Mass of polymer per volume, and sum of (chain mass)^2 per volume.
    mass1 = m1 * lam100 + m2 * lam010 + m3 * lam001
    mass2 = (m1**2 * lam200 + m2**2 * lam020 + m3**2 * lam002
             + 2 * m1 * m2 * lam110 + 2 * m1 * m3 * lam101 + 2 * m2 * m3 * lam011)

    Mn = mass1 / lam0 if lam0 > 0 else np.nan
    Mw = mass2 / mass1 if mass1 > 0 else np.nan

    def conversion(state, feed_key):
        fed = u["feed"].get(feed_key, 0.0)
        return (fed - u["F"] * s[state]) / fed if fed > 0 else np.nan

    return {
        "live_chains": live0,
        "dead_chains": s["B0"],
        "polymer_conc": mass1,
        "Mn": Mn,
        "Mw": Mw,
        "PDI": Mw / Mn if Mn > 0 else np.nan,
        "wt_ethylene": 100 * m1 * lam100 / mass1 if mass1 > 0 else np.nan,
        "wt_propylene": 100 * m2 * lam010 / mass1 if mass1 > 0 else np.nan,
        "wt_diene": 100 * m3 * lam001 / mass1 if mass1 > 0 else np.nan,
        "conv_M1": conversion("C_M1", "C_M1"),
        "conv_M2": conversion("C_M2", "C_M2"),
        "conv_M3": conversion("C_M3", "C_M3"),
    }


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

def write_csv(path, t, X, U, P):
    """One row per step k: time, inputs u(k), states(k), properties(k).

    The last row (k = N) has no u(N) of its own; it repeats u(N-1) so its
    properties can still be evaluated.
    """
    feed_keys = sorted({key for u in U for key in u["feed"]})
    header = (["step", "t", "F"] + [f"feed_{key}" for key in feed_keys]
              + STATE_VARS + PROPERTY_NAMES)

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for k in range(len(t)):
            u = U[min(k, len(U) - 1)]
            p = P[min(k, len(P) - 1)]
            props = properties(X[k], u, p)
            row = ([k, t[k], u["F"]] + [u["feed"].get(key, 0.0) for key in feed_keys]
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
#   u0        baseline inputs: outlet flow F and molar feed rates m_f
#   p0        reactor volume, rate constants, molecular weights
#   events    step changes applied on top of u0 / p0:
#               start   first step k it applies to
#               end     last step k (None = until the end of the run)
#               target  "F", "feed.<state>" (e.g. "feed.C_H2") or "p.<key>"
#               mode    "multiply" (x value) or "set" (= value)
#               value   number
#
# Units assumed: mol/L, L, s, g/mol.  The rate constants are illustrative
# placeholders, not fitted values: replace them with real kinetics.

DEFAULT_CONFIG = {
    "N": 150,
    "dt": 10.0,
    "u0": {
        "F": 1.0,
        "feed": {"C_M1": 2.0, "C_M2": 2.0, "C_M3": 0.05,
                 "C_C1": 3e-4, "C_H2": 5e-4, "C_Al": 1e-4},
    },
    "p0": {
        "V": 100.0,
        # Initiation / propagation
        "k_i1": 50.0, "k_i2": 5.0,
        "k11": 50.0, "k12": 8.0, "k13": 2.0,
        "k21": 40.0, "k22": 3.0, "k31": 20.0,
        # Catalyst activation / deactivation
        "k_a": 0.5, "k_x": 5e-3, "k_x2": 1e-3, "k_x3": 1e-3,
        # Termination / chain transfer
        "k_t": 1e-2, "k_t2": 1e-3, "k_t3": 1e-3,
        "k_tr": 1e-3, "k_tr1": 100.0, "k_trM2": 1e-3,
        "Mw": {"M1": 28.05, "M2": 42.08, "M3": 120.19},
    },
    "events": [
        # hydrogen feed tripled -> shorter chains
        {"start": 50, "end": None, "target": "feed.C_H2", "mode": "multiply", "value": 3.0},
        # more propylene -> composition shift
        {"start": 100, "end": None, "target": "feed.C_M2", "mode": "multiply", "value": 1.5},
    ],
}

MAX_STEPS = 2000


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


def validate_config(config):
    """Return a cleaned copy of config, or raise ValueError saying what is wrong."""
    cfg = deepcopy(config)

    n = cfg.get("N")
    if not isinstance(n, (int, float)) or int(n) != n or not 1 <= n <= MAX_STEPS:
        raise ValueError(f"Number of steps must be a whole number from 1 to {MAX_STEPS}.")
    cfg["N"] = int(n)
    cfg["dt"] = _number(cfg.get("dt"), "Step size dt", positive=True, allow_zero=False)

    u0 = cfg["u0"]
    u0["F"] = _number(u0.get("F"), "Outlet flow F", positive=True, allow_zero=False)
    for key in list(u0["feed"]):
        if key not in STATE_VARS:
            raise ValueError(f"Unknown feed {key!r}.")
        u0["feed"][key] = _number(u0["feed"][key], f"Feed {key}", positive=True)
    if u0["feed"].get("C_C1", 0.0) <= 0.0:
        raise ValueError("Catalyst feed C_C1 must be above zero: the model divides by C_C1.")

    p0 = cfg["p0"]
    p0["V"] = _number(p0.get("V"), "Volume V", positive=True, allow_zero=False)
    for key in KINETIC_KEYS:
        p0[key] = _number(p0.get(key), key, positive=True)
    for key in ("M1", "M2", "M3"):
        p0["Mw"][key] = _number(p0["Mw"].get(key), f"Molecular weight {key}", positive=True, allow_zero=False)

    for i, e in enumerate(cfg.get("events", []), start=1):
        label = f"Step change {i}"
        target = e.get("target", "")
        if not (target == "F"
                or (target.startswith("feed.") and target[5:] in STATE_VARS)
                or (target.startswith("p.") and target[2:] in ["V"] + KINETIC_KEYS)):
            raise ValueError(f"{label}: unknown input {target!r}.")
        if e.get("mode") not in ("multiply", "set"):
            raise ValueError(f"{label}: mode must be 'multiply' or 'set'.")
        e["value"] = _number(e.get("value"), f"{label} value", positive=True)
        e["start"] = int(_number(e.get("start"), f"{label} start step", positive=True))
        if e.get("end") not in (None, ""):
            e["end"] = int(_number(e["end"], f"{label} end step", positive=True))
            if e["end"] < e["start"]:
                raise ValueError(f"{label}: end step is before the start step.")
        else:
            e["end"] = None
    return cfg


def _apply(event, u, p):
    target, value = event["target"], event["value"]
    if target == "F":
        holder, key = u, "F"
    elif target.startswith("feed."):
        holder, key = u["feed"], target[5:]
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
                _apply(e, u, p)
        if u["feed"].get("C_C1", 0.0) <= 0.0:
            raise ValueError(f"Catalyst feed C_C1 drops to zero at step {k}; the model divides by C_C1.")
        U.append(u)
        P.append(p)

    # Reactor initially full of the feed mixture, no catalyst or polymer yet.
    x0 = np.zeros(len(STATE_VARS))
    for key, value in u0["feed"].items():
        x0[STATE_VARS.index(key)] = value / u0["F"]
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
    print(f"Simulated {len(U)} steps of {DEFAULT_CONFIG['dt']:g} s -> results.csv")
    print(f"Final Mn = {last['Mn']:.4g}, Mw = {last['Mw']:.4g}, PDI = {last['PDI']:.3f}")
