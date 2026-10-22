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
import importlib.util
import re
from copy import deepcopy
from pathlib import Path

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

# Species whose starting concentration (mol/L) can be given in config["x0"];
# any not given start at the feed composition (reactor full of feed).
INITIAL_KEYS = ["C_M1", "C_M2", "C_M3", "C_C1", "C_C2", "C_H2", "C_Al"]
RANKINE = 459.67


def arrhenius(p, T):
    """Rate constants at reactor temperature T (deg F):

        k = k0 * exp(-E * (1/T - 1/T_ref))

    with k0 = p[key] (the value at T_ref), E = p["E"][key] in deg R (Ea/R)
    and both temperatures taken as absolute (Rankine). E = 0 keeps k = k0.
    """
    inv = 1.0 / (T + RANKINE) - 1.0 / (p["T_ref"] + RANKINE)
    return {key: p[key] * np.exp(-p["E"][key] * inv) for key in KINETIC_KEYS}

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

# Measured variables (config["measure"]); the order fixes the noise draws, so
# a variable gets the same noise in case 1 and case 2. X_M1 is X_C2 (ethylene
# in the C2 + C3 part), X_M3 is X_diene. T noise is in deg F, the others are
# a fraction of the value.
MEAS_KEYS = ["ML", "Mw", "Mn", "X_M1", "X_M3", "T", "PR"]
MEAS_CASES = {
    1: ["ML", "X_M1", "X_M3", "T", "PR"],
    2: ["ML", "Mw", "Mn", "X_M1", "X_M3", "T", "PR"],
}
MEAS_ABSOLUTE = {"T"}
# Physical range of each measurement: a noisy reading is clipped to it (an
# analyzer cannot report a negative Mw or more than 100 wt %). T is not clipped.
MEAS_LIMITS = {"ML": (0.0, np.inf), "Mw": (0.0, np.inf), "Mn": (0.0, np.inf), "PR": (0.0, np.inf),
               "X_M1": (0.0, 100.0), "X_M3": (0.0, 100.0), "T": (-np.inf, np.inf)}

PROPERTY_NAMES = [
    "live_chains", "dead_chains",
    "polymer_conc",
    "Mn", "Mw", "PDI", "LCB_alpha",
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
    T = states[STATE_VARS.index("T")]
    if not T_RANGE[0] < T < T_RANGE[1]:
        raise ValueError(f"Reactor temperature reached {T:.0f} deg F, outside the "
                         f"{T_RANGE[0]:.0f} to {T_RANGE[1]:.0f} deg F range of the density correlation.")
    F, feed, _ = operating_point(u, p)
    rates = arrhenius(p, T)
    rates.update(F=F, V=p["V"] * L_PER_FT3, CC1_crit=p["CC1_crit"])
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

    Long chain branching hack: Mw from the moments is scaled by alpha,
        k_gc  = 1 / (4 theta B002)       theta = V/F, residence time (s)
        k_c   = k_b B001 B0              k_b = p["LCB_kb"] (adjustable)
        alpha = 1                        if k_c <= k_gc
                beta k_c / k_gc          if k_c >  k_gc   beta = p["LCB_beta"]
    k_b = 0, beta = 1 gives the unbranched Mw.
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

    F, feed, _ = operating_point(u, p)

    theta = p["V"] * L_PER_FT3 / F
    k_gc = 1.0 / (4.0 * theta * s["B002"]) if s["B002"] > 0 else np.inf
    k_c = p["LCB_kb"] * s["B001"] * s["B0"]
    alpha = p["LCB_beta"] * k_c / k_gc if k_c > k_gc else 1.0

    Mn = mass1 / lam0 if lam0 > 0 else np.nan
    Mw = alpha * mass2 / mass1 if mass1 > 0 else np.nan
    PDI = Mw / Mn if Mn > 0 else np.nan

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
        "LCB_alpha": alpha,
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
# Measurements and noise
# ---------------------------------------------------------------------------

def noise_draws(cfg):
    """e[k, i] ~ N(0, 1) for step k = 0..N and MEAS_KEYS[i] (zeros when off).

    Drawn once from the seed, so step k gets the same noise in open and
    closed loop, and the same seed gives the same run.
    """
    m = cfg["measure"]
    shape = (cfg["N"] + 1, len(MEAS_KEYS))
    if not m["noise"]:
        return np.zeros(shape)
    return np.random.default_rng(m["seed"]).standard_normal(shape)


def measure(props, T, e, m):
    """The measured variables of case m["case"] with noise row e:
    T + sigma e (deg F) for T, value (1 + sigma e) for the rest, clipped to
    MEAS_LIMITS. NaN (no polymer yet) stays NaN."""
    out = {}
    for key in MEAS_CASES[m["case"]]:
        true = T if key == "T" else props[key]
        noise = m["sigma"][key] * e[MEAS_KEYS.index(key)]
        value = true + noise if key in MEAS_ABSOLUTE else true * (1.0 + noise)
        out[key] = float(np.clip(value, *MEAS_LIMITS[key]))
    return out


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

def write_csv(path, t, X, U, P, noise=None, m=None):
    """One row per step k: time, inputs u(k), states(k), properties(k), and
    the measurements <name>_meas.

    F is the outlet flow (ft3/hr) that follows from the inputs. The last row
    (k = N) has no u(N) of its own; it repeats u(N-1) so its properties can
    still be evaluated. States and properties are the true values; the _meas
    columns are measure() with noise[k] (case m, default case 1 without
    noise), blank for variables the case does not measure.
    """
    m = m or {"case": 1, "noise": False, "seed": 0, "sigma": {key: 0.0 for key in MEAS_KEYS}}
    if noise is None:
        noise = np.zeros((len(t), len(MEAS_KEYS)))
    header = (["step", "t", "F", "T_i"] + [f"flow_{key}" for key in STREAM_KEYS]
              + STATE_VARS + PROPERTY_NAMES + [f"{key}_meas" for key in MEAS_KEYS])

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for k in range(len(t)):
            u = U[min(k, len(U) - 1)]
            p = P[min(k, len(P) - 1)]
            props = properties(X[k], u, p)
            meas = measure(props, X[k][STATE_VARS.index("T")], noise[k], m)
            row = ([k, t[k], operating_point(u, p)[2], u["T_i"]]
                   + [u["flows"][key] for key in STREAM_KEYS]
                   + list(X[k]) + [props[name] for name in PROPERTY_NAMES]
                   + [meas.get(key, np.nan) for key in MEAS_KEYS])
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
#             Mooney viscosity constants, Arrhenius data (T_ref, E per k)
#   x0        starting concentrations (mol/L) by species; empty = feed
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
# The rate constants (mol/L, s) are not from data. The base case below is
# the team's Grade 1 sheet (2026-10-22): feeds, V, CC1_crit, rate constants
# with their E, the Mooney constants a, b, and an empty reactor (all
# concentrations 0) at 45 deg F. H2 feed, k_t3, Mw, heats of reaction and
# dH_mix are not on the sheet and keep their earlier values. The scaled
# Table 4.29 grades need their scale factors.

DEFAULT_CONFIG = {
    "N": 150,
    "dt": 10.0,
    "u0": {
        "flows": {"M1": 12000.0, "M2": 20000.0, "M3": 2000.0, "C6": 300000.0,
                  "cat": 50.0, "Al": 160.0, "H2": 2.0},
        "T_i": -32.0,
    },
    "p0": {
        "V": 600.0,
        "T0": 45.0,
        # Initiation / propagation
        "k_i1": 500.0, "k_i2": 50.0,
        "k11": 500.0, "k12": 30.0, "k13": 175.0,
        "k21": 310.0, "k22": 10.0, "k31": 150.0,
        # Catalyst activation / deactivation
        "k_a": 100.0, "k_x": 5e-3, "k_x2": 1e-3, "k_x3": 1e-3,
        # Termination / chain transfer
        "k_t": 0.0202, "k_t2": 1e-3, "k_t3": 1e-3,
        "k_tr": 1.0e-7, "k_tr1": 2.5, "k_trM2": 1e-3,
        "Mw": {"M1": 28.0, "M2": 42.0, "M3": 120.0},
        # Energy balance: -dH_RMj (Btu/lb) and dH_mix (Btu/hr)
        "neg_dH": {"M1": 1400.0, "M2": 950.0, "M3": 700.0},
        "H_mix": 1.0e6,
        # Mooney viscosity ML = (a + b PD) Mw
        "ML_a": 2.0e-4, "ML_b": 1.3e-4,
        # Long chain branching hack (properties()): Mw is scaled by alpha once
        # k_c = LCB_kb B001 B0 passes 1 / (4 theta B002). 0 and 1 = no branching.
        "LCB_kb": 0.0, "LCB_beta": 1.0,
        # Arrhenius k = k0 exp(-E (1/T - 1/T_ref)): the rate constants above are
        # k0 at T_ref (deg F); E in deg R (Ea/R), 0 = no temperature dependence.
        "T_ref": 90.0,
        # Below this C_C1 (mol/L) the C_Al/C_C1 term is smoothed to 0 at
        # C_C1 = 0, so a catalyst feed of 0 runs (derivatives.py).
        "CC1_crit": 1e-7,
        # Propagation constants are temperature dependent; the rest are not.
        "E": {key: 1000.0 if key in ("k11", "k12", "k13", "k21", "k22", "k31") else 0.0
              for key in KINETIC_KEYS},
    },
    # Starting concentrations (mol/L) by species, e.g. {"C_M1": 0.5};
    # species left out start at the feed composition. Base case: empty reactor.
    "x0": {key: 0.0 for key in INITIAL_KEYS},
    # Closed loop: name of a file in controllers/ whose control(obs) picks
    # u(k) every step (see Reactor below); None = open loop, u(k) from events.
    "controller": None,
    # Measurements: case 1 measures ML, X_C2 (X_M1), X_diene (X_M3), T, PR;
    # case 2 also Mw and Mn. With noise on, each gets e ~ N(0, 1) per step
    # from the seed: T + sigma e (sigma in deg F), value (1 + sigma e) for the
    # rest (sigma a fraction). Only these get noise. The controller sees the
    # measured values; results.csv keeps the true ones and adds <name>_meas.
    "measure": {
        "case": 1, "noise": False, "seed": 0,
        "sigma": {"ML": 0.02, "Mw": 0.02, "Mn": 0.02, "X_M1": 0.02, "X_M3": 0.02,
                  "T": 1.0, "PR": 0.02},
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
    # Older saved configs have no branching data: k_b = 0, beta = 1 (no branching).
    p0["LCB_kb"] = _number(p0.get("LCB_kb", 0.0), "Branching constant k_b", positive=True)
    p0["LCB_beta"] = _number(p0.get("LCB_beta", 1.0), "Branching constant beta", positive=True)
    # Older saved configs have no Arrhenius data: use T_ref 90 deg F and E = 0.
    p0["CC1_crit"] = _number(p0.get("CC1_crit", 0.1), "Critical catalyst concentration CC1_crit",
                             positive=True, allow_zero=False)
    p0["T_ref"] = _temperature(p0.get("T_ref", 90.0), "Arrhenius reference temperature T_ref")
    E = p0.get("E") or {}
    for key in E:
        if key not in KINETIC_KEYS:
            raise ValueError(f"Unknown rate constant {key!r} in E.")
    p0["E"] = {key: _number(E.get(key, 0.0), f"Activation energy E for {key}") for key in KINETIC_KEYS}

    x0 = {k: v for k, v in (cfg.get("x0") or {}).items() if v not in (None, "")}
    for key in x0:
        if key not in INITIAL_KEYS:
            raise ValueError(f"Unknown starting concentration {key!r}; use one of {', '.join(INITIAL_KEYS)}.")
        x0[key] = _number(x0[key], f"Starting concentration {key}", positive=True)
    cfg["x0"] = x0

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

    # Older saved configs have no measure entry: case 1, no noise.
    cfg.pop("Mw_noise", None)
    m = cfg.get("measure") or {}
    default = DEFAULT_CONFIG["measure"]
    case = _number(m.get("case", default["case"]), "Measurement case")
    if case not in MEAS_CASES:
        raise ValueError(f"Measurement case must be {' or '.join(map(str, MEAS_CASES))}.")
    noise = m.get("noise", False)
    if noise not in (True, False, 0, 1):
        raise ValueError("Measurement noise must be on or off (true or false).")
    seed = _number(m.get("seed", 0), "Noise seed", positive=True)
    if seed != int(seed):
        raise ValueError("Noise seed must be a whole number.")
    sigma = m.get("sigma") or {}
    for key in sigma:
        if key not in MEAS_KEYS:
            raise ValueError(f"Unknown measured variable {key!r} in sigma; use one of {', '.join(MEAS_KEYS)}.")
    cfg["measure"] = {
        "case": int(case), "noise": bool(noise), "seed": int(seed),
        "sigma": {key: _number(sigma.get(key, default["sigma"][key]), f"Noise sigma for {key}", positive=True)
                  for key in MEAS_KEYS},
    }

    name = cfg.get("controller") or None
    if name is not None:
        name = str(name).strip()
        if name.endswith(".py"):
            name = name[:-3]
        if name not in list_controllers():
            raise ValueError(f"Unknown controller {name!r}: put {name}.py in the controllers folder "
                             f"(available: {', '.join(list_controllers()) or 'none'}).")
    cfg["controller"] = name
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


def planned_inputs(cfg, k):
    """(u(k), p(k)) from u0/p0 plus the step changes active at step k.

    cfg must already be validated. In closed loop this is the *planned* input
    the controller sees; it may override it.
    """
    u, p = deepcopy(cfg["u0"]), deepcopy(cfg["p0"])
    for e in cfg["events"]:
        if e["start"] <= k and (e["end"] is None or k <= e["end"]):
            _apply(e, u, p, k)
    if not T_RANGE[0] < u["T_i"] < T_RANGE[1]:
        raise ValueError(f"Inlet temperature leaves the {T_RANGE[0]:.0f} to {T_RANGE[1]:.0f} deg F range at step {k}.")
    return u, p


def initial_state(cfg):
    """states(0) for a validated cfg."""
    u0, p0 = cfg["u0"], cfg["p0"]
    # Reactor initially full of the feed mixture, no catalyst or polymer yet,
    # at the starting reactor temperature T0; cfg["x0"] overrides any species.
    F, feed, _ = operating_point(u0, p0)
    x0 = np.zeros(len(STATE_VARS))
    for state, value in feed.items():
        x0[STATE_VARS.index(state)] = value / F
    for state, value in cfg["x0"].items():
        x0[STATE_VARS.index(state)] = value
    x0[STATE_VARS.index("T")] = p0["T0"]
    return x0


def build_inputs(config=DEFAULT_CONFIG):
    """Turn a run config into (x0, U, P, dt) for simulate() (open loop)."""
    cfg = validate_config(config)
    U, P = [], []
    for k in range(cfg["N"]):
        u, p = planned_inputs(cfg, k)
        U.append(u)
        P.append(p)
    return initial_state(cfg), U, P, cfg["dt"]


# ---------------------------------------------------------------------------
# Closed loop: outside code decides u(k) at every step
# ---------------------------------------------------------------------------

CONTROLLER_DIR = Path(__file__).with_name("controllers")
_CONTROLLER_NAME = re.compile(r"[A-Za-z0-9_]+")


def list_controllers():
    """Names of the controller files in controllers/ (without .py)."""
    if not CONTROLLER_DIR.is_dir():
        return []
    return sorted(f.stem for f in CONTROLLER_DIR.glob("*.py")
                  if _CONTROLLER_NAME.fullmatch(f.stem) and not f.stem.startswith("_"))


def load_controller(name):
    """(control, reset) from controllers/<name>.py. The file is loaded fresh
    for every run, so edits are picked up and module state starts clean."""
    if not _CONTROLLER_NAME.fullmatch(str(name)) or name not in list_controllers():
        raise ValueError(f"Unknown controller {name!r}.")
    path = CONTROLLER_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"controllers.{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "control", None)):
        raise ValueError(f"controllers/{name}.py has no control(obs) function.")
    return module.control, getattr(module, "reset", None)


def merge_inputs(planned, override, k):
    """u(k) = the planned inputs with a controller's override on top.

    override: None (keep the plan) or a dict with any of
        "flows": {stream: lb/hr, ...}   (any subset of STREAM_KEYS)
        "T_i":   inlet temperature, deg F
    """
    u = deepcopy(planned)
    if override is None:
        return u
    if not isinstance(override, dict):
        raise ValueError(f"Step {k}: the controller must return a dict or None, not {type(override).__name__}.")
    unknown = set(override) - {"flows", "T_i"}
    if unknown:
        raise ValueError(f"Step {k}: the controller returned unknown input(s) {sorted(unknown)}; use 'flows' and 'T_i'.")
    for key, value in (override.get("flows") or {}).items():
        if key not in STREAM_KEYS:
            raise ValueError(f"Step {k}: unknown feed stream {key!r} from the controller; use {', '.join(STREAM_KEYS)}.")
        u["flows"][key] = _number(value, f"Step {k}: controller feed flow {key}", positive=True)
    if "T_i" in override:
        u["T_i"] = _number(override["T_i"], f"Step {k}: controller inlet temperature T_i")
        if not T_RANGE[0] < u["T_i"] < T_RANGE[1]:
            raise ValueError(f"Step {k}: controller inlet temperature {u['T_i']:g} deg F is outside "
                             f"{T_RANGE[0]:.0f} to {T_RANGE[1]:.0f} deg F.")
    if sum(u["flows"].values()) <= 0.0:
        raise ValueError(f"Step {k}: the controller set the total feed flow to zero.")
    return u


class Reactor:
    """The simulation one step at a time, so outside code can decide u(k).

        sim = Reactor(config)
        while not sim.done:
            obs = sim.observe()        # what the reactor looks like now
            sim.step(my_logic(obs))    # u(k) override, or None for the plan
        sim.write_csv("results.csv")

    observe() returns a dict:
        k, t, dt, N   step number, time (s), step length, total steps
        meas          the measured variables of config["measure"]["case"]
                      (case 1: ML, X_M1, X_M3, T, PR; case 2: + Mw, Mn),
                      with noise when it is on
        state         {state name: value} at t_k (concentrations, moments, T);
                      T is the measured T
        props         polymer properties at t_k (Mn, Mw, PDI, PR, ML, wt_*,
                      conv_*, ...), with the measured values for the
                      variables in meas; the rest are exact
        state_true, props_true   the same without noise, as in results.csv
        u_planned     u(k) from u0 + step changes: {"flows": {...}, "T_i": ...}
        u_prev        u(k-1) actually applied (None at k = 0)
        p             parameters p(k) (read only; changes are ignored)
    Everything in obs is a copy, so the caller may change it freely.
    """

    def __init__(self, config=DEFAULT_CONFIG):
        self.config = validate_config(config)
        self.N, self.dt = self.config["N"], self.config["dt"]
        self.k = 0
        self.t = [0.0]
        self.X = [initial_state(self.config)]
        self.U, self.P = [], []
        self.noise = noise_draws(self.config)
        self.measure = self.config["measure"]

    @property
    def done(self):
        return self.k >= self.N

    @property
    def state(self):
        return dict(zip(STATE_VARS, self.X[-1]))

    def observe(self):
        k = min(self.k, self.N - 1)
        u_planned, p = planned_inputs(self.config, k)
        # Properties at t_k use the inputs that produced this state (u(k-1)),
        # falling back to the plan before the first step.
        u_now = self.U[-1] if self.U else u_planned
        state = self.state
        props = properties(self.X[-1], u_now, p)
        meas = measure(props, state["T"], self.noise[self.k], self.measure)
        return {
            "k": self.k, "t": self.t[-1], "dt": self.dt, "N": self.N,
            "meas": meas,
            "state": {**state, **{key: v for key, v in meas.items() if key in state}},
            "props": {**props, **{key: v for key, v in meas.items() if key in props}},
            "state_true": state,
            "props_true": props,
            "u_planned": u_planned,
            "u_prev": deepcopy(self.U[-1]) if self.U else None,
            "p": p,
        }

    def step(self, u=None):
        """Apply u(k) (None = planned inputs) over one step dt; return states(k+1)."""
        if self.done:
            raise RuntimeError(f"The run is finished ({self.N} steps).")
        u_planned, p = planned_inputs(self.config, self.k)
        u_applied = merge_inputs(u_planned, u, self.k)
        x = move(self.X[-1], u_applied, p, self.dt, t=self.t[-1])
        self.X.append(x)
        self.t.append(self.t[-1] + self.dt)
        self.U.append(u_applied)
        self.P.append(p)
        self.k += 1
        return x

    def results(self):
        """(t, X, U, P) like run(): U and P are the inputs actually applied."""
        return np.array(self.t), np.array(self.X), self.U, self.P

    def write_csv(self, path="results.csv"):
        write_csv(path, *self.results(), self.noise, self.measure)


def run(config=DEFAULT_CONFIG, csv_path="results.csv", controller=None):
    """Simulate and write the CSV. Returns (t, X, U, P).

    controller: None to use config["controller"] (a file in controllers/), or
    a function control(obs) -> u override or None, called every step. With
    neither, the run is open loop: u(k) comes from u0 and the step changes.
    """
    cfg = validate_config(config)
    reset = None
    if controller is None and cfg["controller"]:
        controller, reset = load_controller(cfg["controller"])
    if controller is None:
        x0, U, P, dt = build_inputs(cfg)
        t, X = simulate(x0, U, P, dt)
        write_csv(csv_path, t, X, U, P, noise_draws(cfg), cfg["measure"])
        return t, X, U, P

    if reset is not None:
        reset(deepcopy(cfg))
    sim = Reactor(cfg)
    while not sim.done:
        sim.step(controller(sim.observe()))
    sim.write_csv(csv_path)
    return sim.results()


def example_inputs():
    """(x0, U, P) for DEFAULT_CONFIG; kept for scripts that used it."""
    x0, U, P, _ = build_inputs(DEFAULT_CONFIG)
    return x0, U, P


if __name__ == "__main__":
    import json
    import sys
    from pathlib import Path

    from plan import load_plan

    # python model.py [--controller NAME] [plan.csv]: the default inputs, or a
    # plan CSV on top of them; --controller closes the loop with controllers/NAME.py.
    args = sys.argv[1:]
    controller_name = None
    if "--controller" in args:
        i = args.index("--controller")
        if i + 1 >= len(args):
            sys.exit("--controller needs a name, e.g. --controller hold_plan")
        controller_name = args[i + 1]
        del args[i:i + 2]
    config = DEFAULT_CONFIG
    if args:
        plan_path = Path(args[0])
        config, summary = load_plan(plan_path.read_text(encoding="utf-8-sig"), DEFAULT_CONFIG, plan_path.name)
        config = validate_config(config)
        print(f"Plan {plan_path.name}: {summary}")
    if controller_name:
        config = dict(validate_config(config), controller=controller_name)
    config = validate_config(config)
    if config["controller"]:
        print(f"Closed loop: controllers/{config['controller']}.py decides u(k) every step")
    t, X, U, P = run(config, "results.csv")
    Path("results_config.json").write_text(json.dumps(validate_config(config), indent=2), encoding="utf-8")
    last = properties(X[-1], U[-1], P[-1])
    T = X[-1][STATE_VARS.index("T")]
    print(f"Simulated {len(U)} steps of {config['dt']:g} s -> results.csv")
    print(f"Final Mn = {last['Mn']:.4g}, Mw = {last['Mw']:.4g}, PDI = {last['PDI']:.3f}, "
          f"PR = {last['PR']:.4g} lb/hr, T = {T:.1f} F, ML = {last['ML']:.4g}")
