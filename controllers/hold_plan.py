"""
Template controller: keeps the planned inputs (same result as open loop).

Copy this file to controllers/<your_name>.py and fill in control(). The
simulator calls control(obs) once per step k, before integrating that step,
and uses what it returns as u(k) over [t_k, t_k + dt].

obs (a copy; change it freely):
    obs["k"], obs["t"], obs["dt"], obs["N"]   step, time (s), step length, total steps
    obs["state"]      {"C_M1": ..., "P0": ..., "T": ...}  all 37 states at t_k
    obs["props"]      {"Mn": ..., "PDI": ..., "PR": ..., "ML": ..., "wt_ethylene": ...}
    obs["u_planned"]  {"flows": {"M1", "M2", "M3", "C6", "cat", "Al", "H2"}, "T_i": ...}
                      u(k) from the starting inputs and step changes
    obs["u_prev"]     u(k-1) actually applied (None at k = 0)
    obs["p"]          parameters p(k): V, rate constants, ... (read only)

Return:
    None                                  keep the planned inputs
    {"flows": {"H2": 3.0}, "T_i": -35.0}  override any feeds (lb/hr, >= 0) and/or
                                          the inlet temperature (deg F); anything
                                          left out stays as planned
"""


def reset(config):
    """Called once before the run with the full run config. Optional."""


def control(obs):
    return None
