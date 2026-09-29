"""
CSTR dynamic model for the Cozewith '88 EPDM/EPM mechanism.

Translates equations A.54-A.82 (plus the six unlabeled second-moment dead-chain
equations that follow A.82) from "derivations of equations.pdf" into a single
right-hand-side function usable with scipy.integrate.solve_ivp.


Naming:
  M1, M2, M3   = ethylene, propylene, diene monomers
  C1, C2       = inactive / active catalyst
  H2           = hydrogen
  Al           = aluminum alkyl
  P, Q, R      = live chains ending in M1, M2, M3 respectively
  B            = dead chains
  0/100/010/001/200/020/002/110/101/011 = moment order in (M1, M2, M3)
                 e.g. P100 = first moment of P w.r.t. M1, L200 = second
                 moment of live chains w.r.t. M1, etc.
"""

import numpy as np

# ---------------------------------------------------------------------------
# State vector layout
# ---------------------------------------------------------------------------

STATE_VARS = [
    # Components (A.54-A.60)
    "C_M1", "C_M2", "C_M3", "C_C1", "C_C2", "C_H2", "C_Al",
    # Zeroth moments (A.61-A.64)
    "P0", "Q0", "R0", "B0",
    # First moments (A.65-A.76)
    "P100", "P010", "P001",
    "Q100", "Q010", "Q001",
    "R100", "R010", "R001",
    "B100", "B010", "B001",
    # Second moments (A.77-A.82 + trailing B200.. equations)
    "L200", "L020", "L002", "L110", "L101", "L011",
    "B200", "B020", "B002", "B110", "B101", "B011",
]


def unpack(y):
    """1D array -> dict of named state variables."""
    return dict(zip(STATE_VARS, y))


def pack(state):
    """dict of named state variables -> 1D array, in STATE_VARS order."""
    return np.array([state[name] for name in STATE_VARS], dtype=float)


def zero_feed():
    """Default feed dict: m_{f,X} = 0 for every state X."""
    return {name: 0.0 for name in STATE_VARS}


# ---------------------------------------------------------------------------
# Right-hand side
# ---------------------------------------------------------------------------

def derivatives(t, y, p, feed=None):
    """
    Compute dy/dt for the full 29-state Cozewith '88 CSTR model.

    Parameters
    ----------
    t : float
        Time (unused explicitly; kept for solve_ivp's calling convention).
    y : array_like, shape (29,)
        State vector, ordered as STATE_VARS.
    p : dict
        Reactor and rate-constant parameters. Required keys:
            F, V,
            k_i1, k_i2,
            k11, k12, k13,
            k21, k22,
            k31,
            k_a, k_x, k_x2, k_x3,
            k_t, k_t2, k_t3,
            k_tr, k_tr1, k_trM2
    feed : dict, optional
        Molar feed terms m_{f,X} (= F_f * C_{X,f}) keyed by name in
        STATE_VARS. Any key not given defaults to 0 (e.g. moments other
        than the monomers/catalyst/H2/Al typically have no feed).

    Returns
    -------
    dydt : np.ndarray, shape (29,)
        Time derivatives, ordered as STATE_VARS.
    """
    if feed is None:
        feed = {}
    mf = lambda name: feed.get(name, 0.0)

    s = unpack(y)
    C_M1, C_M2, C_M3 = s["C_M1"], s["C_M2"], s["C_M3"]
    C_C1, C_C2 = s["C_C1"], s["C_C2"]
    C_H2, C_Al = s["C_H2"], s["C_Al"]

    P0, Q0, R0, B0 = s["P0"], s["Q0"], s["R0"], s["B0"]

    P100, P010, P001 = s["P100"], s["P010"], s["P001"]
    Q100, Q010, Q001 = s["Q100"], s["Q010"], s["Q001"]
    R100, R010, R001 = s["R100"], s["R010"], s["R001"]
    B100, B010, B001 = s["B100"], s["B010"], s["B001"]

    L200, L020, L002 = s["L200"], s["L020"], s["L002"]
    L110, L101, L011 = s["L110"], s["L101"], s["L011"]
    B200, B020, B002 = s["B200"], s["B020"], s["B002"]
    B110, B101, B011 = s["B110"], s["B101"], s["B011"]

    F, V = p["F"], p["V"]
    k_i1, k_i2 = p["k_i1"], p["k_i2"]
    k11, k12, k13 = p["k11"], p["k12"], p["k13"]
    k21, k22 = p["k21"], p["k22"]
    k31 = p["k31"]
    k_a, k_x, k_x2, k_x3 = p["k_a"], p["k_x"], p["k_x2"], p["k_x3"]
    k_t, k_t2, k_t3 = p["k_t"], p["k_t2"], p["k_t3"]
    k_tr, k_tr1, k_trM2 = p["k_tr"], p["k_tr1"], p["k_trM2"]

    # -- Recurring composite terms -----------------------------------------
    # Transfer-to-alkyl term: regenerates a P-type (M1-ended) center, so it
    # is excluded from P0's own death bracket (A.61) but shows up as a gain
    # term there from Q0/R0, and as a full loss term everywhere else.
    term_Al = k_tr * (C_Al / C_C1 - 1.0)
    # Transfer-to-M2 term: regenerates a Q-type (M2-ended) center, so it is
    # excluded from Q0's own death bracket (A.62) but appears elsewhere.
    term_M2tr = k_trM2 * C_M2

    k_death_base = k_t + k_t2 * C_M2 + k_t3 * C_M3 + k_tr1 * C_H2
    k_death_P0 = k_death_base + term_M2tr             # used only in P0's own loss (A.61)
    k_death_Q0 = k_death_base + term_Al                # used only in Q0's own loss (A.62)
    k_death_full = k_death_base + term_Al + term_M2tr  # used everywhere else

    # ------------------------------------------------------------------
    # Components (A.54-A.60)
    # ------------------------------------------------------------------
    dC_M1 = (mf("C_M1") / V - F / V * C_M1
             - (k_i1 * C_C2 + k11 * P0 + k21 * Q0 + k31 * R0) * C_M1)

    dC_M2 = (mf("C_M2") / V - F / V * C_M2
             - (k_x2 * C_C1 + k_i2 * C_C2 + k12 * P0 + k22 * Q0
                + (k_t2 + k_trM2) * (P0 + Q0 + R0)) * C_M2)

    dC_M3 = (mf("C_M3") / V - F / V * C_M3
             - (k_x3 * C_C1 + k13 * P0 + k_t3 * (P0 + Q0 + R0)) * C_M3)

    dC_C1 = (mf("C_C1") / V - F / V * C_C1
             - (k_a + k_x + k_x2 * C_M2 + k_x3 * C_M3) * C_C1)

    dC_C2 = (mf("C_C2") / V - F / V * C_C2
             + k_a * C_C1
             - (k_i1 * C_M1 + k_i2 * C_M2) * C_C2
             + k_tr1 * C_H2 * (P0 + Q0 + R0))

    dC_H2 = (mf("C_H2") / V - F / V * C_H2
             - k_tr1 * C_H2 * (P0 + Q0 + R0))

    dC_Al = (mf("C_Al") / V - F / V * C_Al
             - term_Al * (P0 + Q0 + R0))

    # ------------------------------------------------------------------
    # Zeroth moments (A.61-A.64)
    # ------------------------------------------------------------------
    dP0 = (mf("P0") / V - F / V * P0
           + k_i1 * C_M1 * C_C2
           + (k21 * Q0 + k31 * R0) * C_M1
           - (k12 * C_M2 + k13 * C_M3) * P0
           - k_death_P0 * P0
           + term_Al * (Q0 + R0))

    dQ0 = (mf("Q0") / V - F / V * Q0
           + k_i2 * C_C2 * C_M2
           + k12 * C_M2 * P0
           - k21 * C_M1 * Q0
           - k_death_Q0 * Q0
           + term_M2tr * (P0 + R0))

    dR0 = (mf("R0") / V - F / V * R0
           + k13 * C_M3 * P0
           - k31 * C_M1 * R0
           - k_death_full * R0)

    dB0 = (mf("B0") / V - F / V * B0
           + k_death_full * (P0 + Q0 + R0))

    # ------------------------------------------------------------------
    # First moments (A.65-A.76)
    # ------------------------------------------------------------------
    dP100 = (mf("P100") / V - F / V * P100
             + k_i1 * C_C2 * C_M1
             + (k11 * P0 + k21 * (Q0 + Q100) + k31 * (R0 + R100)) * C_M1
             - (k12 * C_M2 + k13 * C_M3) * P100
             - k_death_full * P100
             + term_Al * (P0 + Q0 + R0))

    dP010 = (mf("P010") / V - F / V * P010
             - (k12 * C_M2 + k13 * C_M3) * P010
             + k21 * C_M1 * Q010
             + k31 * C_M1 * R010
             - k_death_full * P010)

    dP001 = (mf("P001") / V - F / V * P001
             - (k12 * C_M2 + k13 * C_M3) * P001
             + k21 * C_M1 * Q001
             + k31 * C_M1 * R001
             - k_death_full * P001)

    dQ100 = (mf("Q100") / V - F / V * Q100
             - k21 * C_M1 * Q100
             + k12 * C_M2 * P100
             - k_death_full * Q100)

    dQ010 = (mf("Q010") / V - F / V * Q010
             + (k_i2 * C_C2 + k12 * (P0 + P010) + k22 * Q0) * C_M2
             - k21 * C_M1 * Q010
             - k_death_full * Q010
             + term_M2tr * (P0 + Q0 + R0))

    dQ001 = (mf("Q001") / V - F / V * Q001
             - k21 * C_M1 * Q001
             + k12 * C_M2 * P001
             - k_death_full * Q001)

    dR100 = (mf("R100") / V - F / V * R100
             + k13 * C_M3 * P100
             - k31 * C_M1 * R100
             - k_death_full * R100)

    dR010 = (mf("R010") / V - F / V * R010
             + k13 * C_M3 * P010
             - k31 * C_M1 * R010
             - k_death_full * R010)

    dR001 = (mf("R001") / V - F / V * R001
             + k13 * C_M3 * (P0 + P001)
             - k31 * C_M1 * R001
             - k_death_full * R001)

    dB100 = (mf("B100") / V - F / V * B100
             + k_death_full * (P100 + Q100 + R100))

    dB010 = (mf("B010") / V - F / V * B010
             + k_death_full * (P010 + Q010 + R010))

    dB001 = (mf("B001") / V - F / V * B001
             + k_death_full * (P001 + Q001 + R001))

    # ------------------------------------------------------------------
    # Second moments: live chains L (A.77-A.82), dead chains B (unlabeled,
    # immediately following A.82 in the source)
    # ------------------------------------------------------------------
    dL200 = (mf("L200") / V - F / V * L200
             + k_i1 * C_C2 * C_M1
             + (k11 * (P0 + 2 * P100) + k21 * (Q0 + 2 * Q100)
                + k31 * (R0 + 2 * R100)) * C_M1
             + term_Al * (P0 + Q0 + R0)
             - k_death_full * L200)

    dL020 = (mf("L020") / V - F / V * L020
             + k_i2 * C_M2 * C_C2
             + k12 * (P0 + 2 * P010) * C_M2
             + k22 * (Q0 + 2 * Q010) * C_M2
             + term_M2tr * (P0 + Q0 + R0)
             - k_death_full * L020)

    dL002 = (mf("L002") / V - F / V * L002
             + k13 * C_M3 * (P0 + 2 * P001)
             - k_death_full * L002)

    dL110 = (mf("L110") / V - F / V * L110
             + (k11 * P010 + k21 * Q010 + k31 * R010) * C_M1
             + (k12 * P100 + k22 * Q100) * C_M2
             - k_death_full * L110)

    dL101 = (mf("L101") / V - F / V * L101
             + (k11 * P001 + k21 * Q001 + k31 * R001) * C_M1
             + k13 * C_M3 * P100
             - k_death_full * L101)

    dL011 = (mf("L011") / V - F / V * L011
             + (k12 * P001 + k22 * Q001) * C_M2
             + k13 * C_M3 * P010
             - k_death_full * L011)

    dB200 = mf("B200") / V - F / V * B200 + k_death_full * L200
    dB020 = mf("B020") / V - F / V * B020 + k_death_full * L020
    dB002 = mf("B002") / V - F / V * B002 + k_death_full * L002
    dB110 = mf("B110") / V - F / V * B110 + k_death_full * L110
    dB101 = mf("B101") / V - F / V * B101 + k_death_full * L101
    dB011 = mf("B011") / V - F / V * B011 + k_death_full * L011

    return pack({
        "C_M1": dC_M1, "C_M2": dC_M2, "C_M3": dC_M3,
        "C_C1": dC_C1, "C_C2": dC_C2, "C_H2": dC_H2, "C_Al": dC_Al,
        "P0": dP0, "Q0": dQ0, "R0": dR0, "B0": dB0,
        "P100": dP100, "P010": dP010, "P001": dP001,
        "Q100": dQ100, "Q010": dQ010, "Q001": dQ001,
        "R100": dR100, "R010": dR010, "R001": dR001,
        "B100": dB100, "B010": dB010, "B001": dB001,
        "L200": dL200, "L020": dL020, "L002": dL002,
        "L110": dL110, "L101": dL101, "L011": dL011,
        "B200": dB200, "B020": dB020, "B002": dB002,
        "B110": dB110, "B101": dB101, "B011": dB011,
    })


if __name__ == "__main__":
    # Minimal smoke test: arbitrary parameters/state, just checks it runs
    # and returns the right shape. Replace with real kinetic constants and
    # initial conditions before using for anything quantitative.
    from scipy.integrate import solve_ivp

    p = dict(
        F=1.0, V=100.0,
        k_i1=1e-2, k_i2=1e-2,
        k11=1e-2, k12=1e-2, k13=1e-2,
        k21=1e-2, k22=1e-2,
        k31=1e-2,
        k_a=1e-2, k_x=1e-3, k_x2=1e-3, k_x3=1e-3,
        k_t=1e-2, k_t2=1e-3, k_t3=1e-3,
        k_tr=1e-3, k_tr1=1e-3, k_trM2=1e-3,
    )

    y0 = {name: 0.0 for name in STATE_VARS}
    y0.update(C_M1=2.0, C_M2=1.0, C_M3=0.05, C_C1=1e-3, C_C2=1e-4,
              C_H2=1e-3, C_Al=1e-2)

    feed = zero_feed()
    feed.update(C_M1=2.0, C_M2=1.0, C_M3=0.05, C_C1=1e-3, C_H2=1e-3, C_Al=1e-2)

    sol = solve_ivp(derivatives, [0, 10], pack(y0), args=(p, feed),
                     method="LSODA", dense_output=True)
    print("success:", sol.success)
    print(dict(zip(STATE_VARS, sol.y[:, -1])))

#Dan's addition?
def run_simulation():
    time = 0
    while time < 4:
        if __name__ == "__main__":
            from scipy.integrate import solve_ivp

            p = dict(
                F=1.0, V=3000.0,
                k_i1=1e-2, k_i2=1e-2,
                k11=1e-2, k12=1e-2, k13=1e-2,
                k21=1e-2, k22=1e-2,
                k31=1e-2,
                k_a=1e-2, k_x=1e-3, k_x2=1e-3, k_x3=1e-3,
                k_t=1e-2, k_t2=1e-3, k_t3=1e-3,
                k_tr=1e-3, k_tr1=1e-3, k_trM2=1e-3,
            )

            y0 = {name: 0.0 for name in STATE_VARS}
            y0.update(pack(dict(zip(STATE_VARS, sol.y))))

            feed = zero_feed()
            feed.update(C_M1=2.0, C_M2=1.0, C_M3=0.05, C_C1=1e-3, C_H2=1e-3, C_Al=1e-2)

            sol = solve_ivp(derivatives, [0, 1], pack(y0), args=(p, feed),
                            method="LSODA", dense_output=True)
            print("success:", sol.success)
            print(dict(zip(STATE_VARS, sol.y)))
        return(dict(zip(STATE_VARS, sol.y)))
    time += 1
run_simulation()
