"""
Example only: a PI loop that holds the reactor temperature at a setpoint by
moving the inlet temperature T_i (chilling or warming the feed). Every other
input stays as planned.

It shows the pattern for any controller: read obs, keep state between steps
(module variables reset in reset()), return the inputs to change.
"""

SETPOINT = 110.0          # reactor T target, deg F
KC = 0.8                  # gain: deg F of T_i per deg F of error (T_i up -> T up)
TAU_I = 300.0             # integral time, s
T_I_MIN, T_I_MAX = -80.0, 40.0   # feed chiller limits, deg F

_integral = 0.0


def reset(config):
    global _integral
    _integral = 0.0


def control(obs):
    global _integral
    error = SETPOINT - obs["state"]["T"]
    bias = obs["u_planned"]["T_i"]
    unclamped = bias + KC * (error + _integral / TAU_I)
    T_i = min(max(unclamped, T_I_MIN), T_I_MAX)
    # Anti-windup: only integrate while the output is not at a limit.
    if T_i == unclamped:
        _integral += error * obs["dt"]
    return {"T_i": T_i}
