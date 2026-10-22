"""
Read a run plan from a CSV file and turn it into a run config (model.py).

    python model.py plan.csv          # run a plan from the command line

Two layouts are accepted; the header row tells them apart.

1. List: one row per setting or step change.

       input,value,change,start,end
       N,200
       dt,10
       Ethylene,12000
       T0,90
       k11,385
       H2,3,multiply,50,
       T_i,-20,set,80,120

   A row without a start step sets a starting value (run length N, step
   size dt, a feed flow, the inlet temperature, a parameter). A row with a
   start step is a step change: change is "set" (default) or "multiply",
   end is the last step it applies to (blank = until the end of the run).

2. Timeline: one row per step (or per time), one column per input.

       step,H2,T_i,k11
       0,2,-32,385
       50,6,-32,385
       80,6,-20,400

   The first column is "step" or "t" (seconds; rounded to the nearest
   step). Each value holds until the next row, and a blank cell keeps the
   previous value. If the last row is past the end of the run, N is raised
   to cover it.

Inputs are named as in results.csv or the dashboard: flow_M1 / flow.M1 / M1
/ Ethylene, T_i, T0, V, H_mix, k11, Mw.M1, neg_dH.M1, ML_a, ML_b, N, dt,
the Arrhenius data T_ref and E.k11, the measurements measure.case (1/2),
measure.noise (1/0), measure.seed and measure.sigma.ML ... (starting values
only), starting concentrations C_M1 ...
C_Al (or init.C_M1), and controller (a file name in controllers/, or none);
the last four are starting values only.
Text after a comma or bracket in a header is ignored, so "Ethylene, lb/hr"
works.
"""

import csv
import io
import re
from copy import deepcopy

STREAM_ALIASES = {
    "m1": "M1", "ethylene": "M1", "m2": "M2", "propylene": "M2", "m3": "M3", "diene": "M3",
    "c6": "C6", "hexane": "C6", "solvent": "C6", "cat": "cat", "catalyst": "cat", "vcl4": "cat",
    "al": "Al", "aluminum": "Al", "aluminum alkyl": "Al", "h2": "H2", "hydrogen": "H2",
}
MONOMERS = ("M1", "M2", "M3")


def _clean(name):
    """Header/input name without units: "Ethylene feed, lb/hr" -> "ethylene"."""
    name = re.split(r"[,(\[]", str(name), maxsplit=1)[0].strip()
    name = re.sub(r"\s+(feed|flow)$", "", name, flags=re.I)
    return name


def resolve(name, p0):
    """Where an input lives in the config: ("flow", key), ("T_i",), ("p", key),
    ("p_nested", group, key), ("measure", key), ("measure_sigma", key) or
    ("run", "N"/"dt"). Raises ValueError."""
    raw = _clean(name)
    low = raw.lower()
    if low in ("n", "steps"):
        return ("run", "N")
    if low in ("dt", "step size"):
        return ("run", "dt")
    if low == "controller":
        return ("run", "controller")
    if low in ("t_i", "ti", "inlet t", "inlet temperature"):
        return ("T_i",)
    if low in ("t_ref", "tr", "t_r"):
        return ("p", "T_ref")
    m = re.fullmatch(r"(?:init[._]|x0[._]|initial\s+)?(c_(?:m[123]|c[12]|h2|al))", low)
    if m:
        key = "C_" + m.group(1)[2:].upper().replace("AL", "Al")
        return ("x0", key)
    for prefix in ("flow.", "flow_"):
        if low.startswith(prefix):
            low = low[len(prefix):]
            break
    if low in STREAM_ALIASES:
        return ("flow", STREAM_ALIASES[low])
    key = raw[2:] if raw.startswith("p.") else raw
    m = re.fullmatch(r"measure[._](case|noise|seed)", key, flags=re.I)
    if m:
        return ("measure", m.group(1).lower())
    m = re.fullmatch(r"measure[._]sigma[._](\w+)", key, flags=re.I)
    if m:
        k = next((k for k in ("ML", "Mw", "Mn", "X_M1", "X_M3", "T", "PR") if k.lower() == m.group(1).lower()), None)
        if k is None:
            raise ValueError(f"Unknown measured variable in {name!r}.")
        return ("measure_sigma", k)
    m = re.fullmatch(r"(Mw|neg_dH)[._](M[123])", key, flags=re.I)
    if m:
        group = "Mw" if m.group(1).lower() == "mw" else "neg_dH"
        return ("p_nested", group, m.group(2).upper())
    m = re.fullmatch(r"E[._](k\w+)", key, flags=re.I)
    if m:
        k = next((k for k in p0 if k.lower() == m.group(1).lower() and k.startswith("k")), None)
        if k is None:
            raise ValueError(f"Unknown rate constant in {name!r}.")
        return ("p_nested", "E", k)
    for k in p0:
        if k.lower() == key.lower() and not isinstance(p0[k], dict):
            return ("p", k)
    raise ValueError(f"Unknown input {name!r}.")


def _target(where, name):
    """Event target string for an input that can change during the run."""
    if where[0] == "flow":
        return "flow." + where[1]
    if where[0] == "T_i":
        return "T_i"
    if where[0] == "p":
        return "p." + where[1]
    raise ValueError(f"{name!r} can only be set as a starting value, not changed during the run.")


def _num(text, what):
    try:
        return float(text)
    except (TypeError, ValueError):
        raise ValueError(f"{what}: {text!r} is not a number.") from None


def _step(text, what):
    v = _num(text, what)
    if v != int(v) or v < 0:
        raise ValueError(f"{what}: step must be a whole number of 0 or more (got {text!r}).")
    return int(v)


def _set_start(cfg, where, value):
    if where[0] == "run":
        cfg[where[1]] = value
    elif where[0] == "flow":
        cfg["u0"]["flows"][where[1]] = value
    elif where[0] == "T_i":
        cfg["u0"]["T_i"] = value
    elif where[0] == "p":
        cfg["p0"][where[1]] = value
    elif where[0] == "x0":
        cfg.setdefault("x0", {})[where[1]] = value
    elif where[0] == "measure":
        cfg.setdefault("measure", {})[where[1]] = value
    elif where[0] == "measure_sigma":
        cfg.setdefault("measure", {}).setdefault("sigma", {})[where[1]] = value
    else:
        cfg["p0"].setdefault(where[1], {})[where[2]] = value


def _rows(text):
    rows = [r for r in csv.reader(io.StringIO(text.lstrip("﻿"))) if any(c.strip() for c in r)]
    if len(rows) < 2:
        raise ValueError("The plan needs a header row and at least one data row.")
    return [c.strip() for c in rows[0]], [[c.strip() for c in r] for r in rows[1:]]


def load_plan(text, base, source="plan.csv"):
    """Run config from plan CSV text, starting from a copy of `base`.

    Returns (config, summary) where summary is a short description of what
    the plan set. The config still goes through model.validate_config.
    """
    header, rows = _rows(text)
    low = [h.lower() for h in header]
    cfg = deepcopy(base)
    if "input" in low and "value" in low:
        return _load_list(cfg, low, rows, source)
    if low[0] in ("step", "k", "t", "time") or low[0].startswith(("t ", "t,", "time")):
        return _load_timeline(cfg, header, rows, source)
    raise ValueError("Unrecognized plan layout: the header needs 'input' and 'value' columns "
                     "(list layout), or a first column 'step' or 't' (timeline layout).")


def _load_list(cfg, low, rows, source):
    col = {name: low.index(name) for name in ("input", "value", "change", "start", "end") if name in low}
    get = lambda r, name: r[col[name]] if name in col and col[name] < len(r) else ""
    events, starts = [], 0
    for i, r in enumerate(rows, start=2):
        name, value = get(r, "input"), get(r, "value")
        if not name:
            continue
        where = resolve(name, cfg["p0"])
        what = f"{source} row {i} ({name})"
        start = get(r, "start")
        if start == "":
            # The controller is a name (a file in controllers/), not a number; blank or "none" = open loop.
            text_value = where == ("run", "controller")
            _set_start(cfg, where, (value if value.lower() != "none" else None) if text_value else _num(value, what))
            starts += 1
            continue
        mode = (get(r, "change") or "set").lower()
        mode = {"x": "multiply", "×": "multiply", "*": "multiply", "times": "multiply", "=": "set"}.get(mode, mode)
        if mode not in ("set", "multiply"):
            raise ValueError(f"{what}: change must be 'set' or 'multiply' (got {get(r, 'change')!r}).")
        end = get(r, "end")
        events.append({"start": _step(start, what), "end": _step(end, what) if end else None,
                       "target": _target(where, name), "mode": mode, "value": _num(value, what)})
    cfg["events"] = sorted(events, key=lambda e: e["start"])
    return cfg, f"{starts} starting value{'s' * (starts != 1)}, {len(events)} step change{'s' * (len(events) != 1)}"


def _load_timeline(cfg, header, rows, source):
    by_time = header[0].lower().startswith(("t", "time")) and header[0].lower() not in ("step",)
    dt = float(cfg["dt"])
    steps = []
    for i, r in enumerate(rows, start=2):
        what = f"{source} row {i}"
        k = round(_num(r[0], what) / dt) if by_time else _step(r[0], what)
        if k < 0:
            raise ValueError(f"{what}: time before the start of the run.")
        if steps and k <= steps[-1][0]:
            raise ValueError(f"{what}: steps must increase down the file.")
        steps.append((k, r))
    first, last = steps[0][0], steps[-1][0]
    raised = last + 1 > cfg["N"]
    if raised:
        cfg["N"] = last + 1

    events, columns = [], 0
    for j, name in enumerate(header[1:], start=1):
        if not name:
            continue
        where = resolve(name, cfg["p0"])
        target = _target(where, name)
        values, current, row_i = [], None, 0
        for k in range(first, last + 1):
            if row_i < len(steps) and steps[row_i][0] == k:
                cell = steps[row_i][1][j] if j < len(steps[row_i][1]) else ""
                if cell != "":
                    current = _num(cell, f"{source} step {k} ({name})")
                row_i += 1
            values.append(current)
        # Leading blanks: start the profile at its first value.
        lead = next((n for n, v in enumerate(values) if v is not None), None)
        if lead is None:
            continue
        values = values[lead:]
        events.append({"start": first + lead, "end": None, "target": target, "mode": "set",
                       "values": values, "source": f"{source}: {_clean(name)}"})
        columns += 1
    cfg["events"] = events
    return cfg, (f"{columns} input profile{'s' * (columns != 1)} over steps {first}-{last}"
                 + (f"; run length raised to {cfg['N']} steps" if raised else ""))
