# Model Equations

These are the equations the simulator solves, exactly as `derivatives.py` and
`model.py` compute them. The equation numbers (A.54 …) refer to the
derivation in *"derivations of equations.pdf"*, and the energy balance and
property formulas come from *"more equations.pdf"*.

For how to run the simulator, see [README.md](README.md).

---

## 1. Notation

| Symbol | Meaning |
|---|---|
| M1, M2, M3 | Ethylene, propylene, diene monomers |
| C1, C2 | Inactive and active catalyst sites |
| H2, Al | Hydrogen, aluminum alkyl co-catalyst |
| P, Q, R | Live chains whose last unit is M1, M2, M3 |
| B | Dead chains |
| $X_{ijk}$ | Moment of species X for chain composition (i, j, k) in (M1, M2, M3). For example, $P_{100}$ is the first moment of P in M1. |
| $L_{ijk}$ | Second moments of all live chains (P + Q + R) |
| $C_X$ | Concentration of species X (mol/L) |
| $F$, $V$ | Outlet volumetric flow and reactor volume |
| $m_{f,X}$ | Molar feed rate of X (mol/s) |
| $D = F/V$ | Dilution rate (1/s) |

**Rate constants**

| Constant | Reaction |
|---|---|
| $k_a$ | Catalyst activation C1 → C2 |
| $k_x$, $k_{x2}$, $k_{x3}$ | Deactivation of C1: spontaneous, by M2, by M3 |
| $k_{i1}$, $k_{i2}$ | Initiation of an active site by M1, M2 |
| $k_{jk}$ | Propagation: a chain ending in Mj adds Mk ($k_{11}, k_{12}, k_{13}, k_{21}, k_{22}, k_{31}$) |
| $k_t$, $k_{t2}$, $k_{t3}$ | Termination: spontaneous, with M2, with M3 |
| $k_{tr}$ | Chain transfer to aluminum alkyl |
| $k_{tr1}$ | Chain transfer to hydrogen |
| $k_{trM2}$ | Chain transfer to M2 |

**Shorthand used below**

$$\lambda_L = P_0 + Q_0 + R_0 \qquad \text{(total live chains)}$$

$$\tau_{Al} = k_{tr}\,g(C_{C1}) \qquad \tau_{M2} = k_{trM2}\,C_{M2}$$

where $g$ is the alkyl ratio. It is $C_{Al}/C_{C1} - 1$, except that below a
critical catalyst concentration $c^* $ (`CC1_crit`, default 0.1 mol/L) it is
replaced by a polynomial that goes to 0 at $C_{C1} = 0$. That way the model
never divides by zero, and a catalyst feed of 0 can be simulated:

$$g(C_{C1}) = \begin{cases} \dfrac{C_{Al}}{C_{C1}} - 1 & C_{C1} \ge c^* \[2ex] \dfrac{C_{Al} - C_{C1}}{c^{*2}}\left(6C_{C1} - \dfrac{8}{c^*}C_{C1}^2 + \dfrac{3}{c^{*2}}C_{C1}^3\right) & C_{C1} < c^* \end{cases}$$

The two branches are equal at $C_{C1} = c^*$. If the solver steps $C_{C1}$ a
hair below 0 as the catalyst runs out, it is treated as 0. $c^*$ can be set on
the dashboard (*Critical C_C1*) or with `CC1_crit` in a plan CSV.

The total rate at which a live chain dies (termination plus transfer) is

$$k_d = k_t + k_{t2}C_{M2} + k_{t3}C_{M3} + k_{tr1}C_{H2} + \tau_{Al} + \tau_{M2}$$

Two variants are used only in the zeroth-moment equations. Transfer to Al
gives back a P-type chain and transfer to M2 gives back a Q-type chain, so
each is left out of that chain's own loss term:

$$k_{d,P} = k_d - \tau_{Al} \qquad k_{d,Q} = k_d - \tau_{M2}$$

Every state X gets the same CSTR in/out terms, $\dfrac{m_{f,X}}{V} - D\,X$.
Only M1, M2, M3, C1, H2 and Al are fed, so $m_{f,X} = 0$ for every other state.

---

## 2. Inputs to model units

These are in `model.py`, `operating_point()`.

**Outlet flow.** The feed is treated as hexane at the inlet temperature:

$$F = \frac{\sum_i F_i}{\rho(T_i)} \quad [\text{ft}^3/\text{hr}]$$

with $F_i$ the mass feed of each stream (lb/hr).

**Molar feeds:**

$$m_{f,X} = \frac{F_i}{MW_i} \quad [\text{lbmol/hr}] \;\times\; \frac{453.59}{3600} \;\to\; [\text{mol/s}]$$

The molecular weights of the monomers come from `p0.Mw`. The others are hexane 86, VCl₄ 192.75,
aluminum alkyl 120.56 and H₂ 2.

**Unit conversions:** 1 ft³ = 28.317 L. In the equations below, F is in L/s and V is in L.

**Residence time** (shown on the dashboard):

$$\tau = \frac{V}{F} \times 60 \quad [\text{min}], \quad V \text{ in ft}^3,\ F \text{ in ft}^3/\text{hr}$$

**Where u(k) comes from.** In open loop, the inputs come from the starting
inputs and the step changes. In closed loop, a controller sets them at every
step from the current state; see README, *Closed-loop control*. Either way they
are held constant over each step, and the equations below are unchanged.

**Starting state.** By default, the reactor is full of feed, so
$C_X(0) = m_{f,X}/F$ for the fed species and $T(0) = T_0$. Everything else
(active catalyst and all moments) starts at 0. Any species concentration
($C_{M1} \dots C_{Al}$) can be given its own starting value instead: set it in
*Initial conditions* on the dashboard, in `x0` in the config, or with a
`C_M1,0.5` row in a plan CSV.

---

## 2b. Temperature dependence of the rate constants (Arrhenius)

Every rate constant in sections 3–6 is evaluated at the current reactor
temperature:

$$k(T) = k_0 \exp\!\left[-E\left(\frac{1}{T} - \frac{1}{T_r}\right)\right]$$
ight)
ight]$$

| Symbol | Meaning | Default |
|---|---|---|
| $k_0$ | Value of the constant at $T_r$ (the numbers in section 9) | current defaults |
| $E$ | Activation energy divided by R, in °R (one per constant) | 0.0 |
| $T_r$ | Reference temperature | 90 °F |

T and Tr are in Rankine (°F + 459.67), so 1/T stays finite near 0 °F. With
E = 0, k = k₀ at every temperature, which reproduces the earlier constant-k
model exactly. When E > 0, a hotter reactor gives a larger k. For example, with
E = 3000 °R, k₁₁ goes from 800 at 90 °F to 1,061 at 120 °F.

The code is `arrhenius()` in `model.py`, called on every evaluation of the
right-hand side.

---

## 3. Species balances (A.54–A.60)

**Monomer consumption rates:**

$$R_{M1} = \left(k_{i1}C_{C2} + k_{11}P_0 + k_{21}Q_0 + k_{31}R_0\right) C_{M1}$$

$$R_{M2} = \left(k_{x2}C_{C1} + k_{i2}C_{C2} + k_{12}P_0 + k_{22}Q_0 + (k_{t2} + k_{trM2})\lambda_L\right) C_{M2}$$

$$R_{M3} = \left(k_{x3}C_{C1} + k_{13}P_0 + k_{t3}\lambda_L\right) C_{M3}$$

**Monomers**, for j = 1, 2, 3:

$$\frac{dC_{Mj}}{dt} = \frac{m_{f,Mj}}{V} - D\,C_{Mj} - R_{Mj}$$

**Inactive catalyst:**

$$\frac{dC_{C1}}{dt} = \frac{m_{f,C1}}{V} - D\,C_{C1} - \left(k_a + k_x + k_{x2}C_{M2} + k_{x3}C_{M3}\right) C_{C1}$$

**Active catalyst** (activated from C1, used up by initiation, given back by H₂ transfer):

$$\frac{dC_{C2}}{dt} = -D\,C_{C2} + k_a C_{C1} - \left(k_{i1}C_{M1} + k_{i2}C_{M2}\right) C_{C2} + k_{tr1}C_{H2}\lambda_L$$

**Hydrogen:**

$$\frac{dC_{H2}}{dt} = \frac{m_{f,H2}}{V} - D\,C_{H2} - k_{tr1}C_{H2}\lambda_L$$

**Aluminum alkyl:**

$$\frac{dC_{Al}}{dt} = \frac{m_{f,Al}}{V} - D\,C_{Al} - \tau_{Al}\lambda_L$$

---

## 4. Zeroth moments: number of chains (A.61–A.64)

$$\frac{dP_0}{dt} = -D P_0 + k_{i1}C_{M1}C_{C2} + (k_{21}Q_0 + k_{31}R_0)C_{M1} - (k_{12}C_{M2} + k_{13}C_{M3})P_0 - k_{d,P}P_0 + \tau_{Al}(Q_0 + R_0)$$

$$\frac{dQ_0}{dt} = -D Q_0 + k_{i2}C_{C2}C_{M2} + k_{12}C_{M2}P_0 - k_{21}C_{M1}Q_0 - k_{d,Q}Q_0 + \tau_{M2}(P_0 + R_0)$$

$$\frac{dR_0}{dt} = -D R_0 + k_{13}C_{M3}P_0 - k_{31}C_{M1}R_0 - k_d R_0$$

$$\frac{dB_0}{dt} = -D B_0 + k_d\,\lambda_L$$

---

## 5. First moments: monomer units in chains (A.65–A.76)

**Chains ending in M1 (P):**

$$\frac{dP_{100}}{dt} = -D P_{100} + k_{i1}C_{C2}C_{M1} + \left[k_{11}P_0 + k_{21}(Q_0 + Q_{100}) + k_{31}(R_0 + R_{100})\right]C_{M1} - (k_{12}C_{M2} + k_{13}C_{M3})P_{100} - k_d P_{100} + \tau_{Al}\lambda_L$$

$$\frac{dP_{010}}{dt} = -D P_{010} - (k_{12}C_{M2} + k_{13}C_{M3})P_{010} + k_{21}C_{M1}Q_{010} + k_{31}C_{M1}R_{010} - k_d P_{010}$$

$$\frac{dP_{001}}{dt} = -D P_{001} - (k_{12}C_{M2} + k_{13}C_{M3})P_{001} + k_{21}C_{M1}Q_{001} + k_{31}C_{M1}R_{001} - k_d P_{001}$$

**Chains ending in M2 (Q):**

$$\frac{dQ_{100}}{dt} = -D Q_{100} - k_{21}C_{M1}Q_{100} + k_{12}C_{M2}P_{100} - k_d Q_{100}$$

$$\frac{dQ_{010}}{dt} = -D Q_{010} + \left[k_{i2}C_{C2} + k_{12}(P_0 + P_{010}) + k_{22}Q_0\right]C_{M2} - k_{21}C_{M1}Q_{010} - k_d Q_{010} + \tau_{M2}\lambda_L$$

$$\frac{dQ_{001}}{dt} = -D Q_{001} - k_{21}C_{M1}Q_{001} + k_{12}C_{M2}P_{001} - k_d Q_{001}$$

**Chains ending in M3 (R):**

$$\frac{dR_{100}}{dt} = -D R_{100} + k_{13}C_{M3}P_{100} - k_{31}C_{M1}R_{100} - k_d R_{100}$$

$$\frac{dR_{010}}{dt} = -D R_{010} + k_{13}C_{M3}P_{010} - k_{31}C_{M1}R_{010} - k_d R_{010}$$

$$\frac{dR_{001}}{dt} = -D R_{001} + k_{13}C_{M3}(P_0 + P_{001}) - k_{31}C_{M1}R_{001} - k_d R_{001}$$

**Dead chains (B)**, for $ijk \in \{100, 010, 001\}$:

$$\frac{dB_{ijk}}{dt} = -D B_{ijk} + k_d\left(P_{ijk} + Q_{ijk} + R_{ijk}\right)$$

---

## 6. Second moments (A.77–A.82 and the dead-chain equations after them)

**Live chains (L = P + Q + R):**

$$\frac{dL_{200}}{dt} = -D L_{200} + k_{i1}C_{C2}C_{M1} + \left[k_{11}(P_0 + 2P_{100}) + k_{21}(Q_0 + 2Q_{100}) + k_{31}(R_0 + 2R_{100})\right]C_{M1} + \tau_{Al}\lambda_L - k_d L_{200}$$

$$\frac{dL_{020}}{dt} = -D L_{020} + k_{i2}C_{M2}C_{C2} + \left[k_{12}(P_0 + 2P_{010}) + k_{22}(Q_0 + 2Q_{010})\right]C_{M2} + \tau_{M2}\lambda_L - k_d L_{020}$$

$$\frac{dL_{002}}{dt} = -D L_{002} + k_{13}C_{M3}(P_0 + 2P_{001}) - k_d L_{002}$$

$$\frac{dL_{110}}{dt} = -D L_{110} + (k_{11}P_{010} + k_{21}Q_{010} + k_{31}R_{010})C_{M1} + (k_{12}P_{100} + k_{22}Q_{100})C_{M2} - k_d L_{110}$$

$$\frac{dL_{101}}{dt} = -D L_{101} + (k_{11}P_{001} + k_{21}Q_{001} + k_{31}R_{001})C_{M1} + k_{13}C_{M3}P_{100} - k_d L_{101}$$

$$\frac{dL_{011}}{dt} = -D L_{011} + (k_{12}P_{001} + k_{22}Q_{001})C_{M2} + k_{13}C_{M3}P_{010} - k_d L_{011}$$

**Dead chains**, for $ijk \in \{200, 020, 002, 110, 101, 011\}$:

$$\frac{dB_{ijk}}{dt} = -D B_{ijk} + k_d L_{ijk}$$

---

## 7. Energy balance (A.50, simplified)

The reactor contents are treated as hexane. Only the reaction heat uses
monomer properties.

$$\frac{dT}{dt} = \underbrace{\frac{F}{V}\,\frac{H(T_i) - H(T)}{C_p(T)}}_{\text{feed term}} + \underbrace{\frac{\sum_j R^{p}_{Mj}\,Mw_j\,(-\Delta H_{j}) + \Delta H_{mix}/V}{\rho(T)\,C_p(T)}}_{\text{reaction + mixing heat}} \quad [°\text{F/s}]$$

**Only propagation counts toward the reaction heat:**

$$R^p_{M1} = (k_{11}P_0 + k_{21}Q_0 + k_{31}R_0)C_{M1} \qquad R^p_{M2} = (k_{12}P_0 + k_{22}Q_0)C_{M2} \qquad R^p_{M3} = k_{13}P_0\,C_{M3}$$

**Hexane correlations** (T in °F):

$$H(T) = 0.5073\,T + 2.074\times10^{-4}\,T^2 + 2.53\times10^{-7}\,T^3 \quad [\text{Btu/lb}]$$

$$C_p(T) = \frac{dH}{dT} = 0.5073 + 4.148\times10^{-4}\,T + 7.590\times10^{-7}\,T^2 \quad [\text{Btu/lb/°F}]$$

$$\rho(T) = \frac{3.804}{0.2641^{\,1 + (1 - (T + 459.67)/913.68)^{0.27537}}} \quad [\text{lb/ft}^3] \quad \text{(Rackett form)}$$

**Units in the code:** $R\cdot Mw$ (g/L/s) × 0.062428 → lb/ft³/s; $-\Delta H_j$ in Btu/lb;
$\Delta H_{mix}$ in Btu/hr is divided by 3600 and by V in ft³.

---

## 8. Polymer properties (`model.py`, `properties()`)

Each property uses the total moments of live and dead chains together:

$$\lambda_0 = P_0 + Q_0 + R_0 + B_0 \qquad \lambda_{100} = P_{100} + Q_{100} + R_{100} + B_{100} \;(\text{same for } 010, 001)$$

$$\lambda_{200} = L_{200} + B_{200} \;(\text{same for } 020, 002, 110, 101, 011)$$

**Polymer mass moments**, with $Mw_1, Mw_2, Mw_3$ the monomer molecular weights:

$$P_1^m = Mw_1\lambda_{100} + Mw_2\lambda_{010} + Mw_3\lambda_{001} \quad [\text{g/L}]$$

$$P_2^m = Mw_1^2\lambda_{200} + Mw_2^2\lambda_{020} + Mw_3^2\lambda_{002} + 2Mw_1Mw_2\lambda_{110} + 2Mw_1Mw_3\lambda_{101} + 2Mw_2Mw_3\lambda_{011}$$

| Property | Formula | Units |
|---|---|---|
| Number-average MW | $M_n = P_1^m / \lambda_0$ | g/mol |
| Weight-average MW | $M_w = P_2^m / P_1^m$ | g/mol |
| Dispersity | $PDI = M_w / M_n$ | – |
| Mooney viscosity | $ML = (a + b \cdot PDI)\,M_w$ | – |
| Polymerization rate | $PR = F \cdot P_1^m$ (F in L/s, × 3600 / 453.59) | lb/hr |
| Polymer concentration | $P_1^m$ | g/L |
| Ethylene in polymer | $100\,Mw_1\lambda_{100} / P_1^m$ | wt % |
| Propylene in polymer | $100\,Mw_2\lambda_{010} / P_1^m$ | wt % |
| Diene in polymer | $100\,Mw_3\lambda_{001} / P_1^m$ | wt % |
| $X_{M1}$ (ethylene in the C2 + C3 part) | $100\,Mw_1\lambda_{100} / (Mw_1\lambda_{100} + Mw_2\lambda_{010})$ | wt % |
| $X_{M3}$ (diene) | $100\,Mw_3\lambda_{001} / P_1^m$ | wt % |
| Monomer conversion | $X_j = (m_{f,Mj} - F\,C_{Mj}) / m_{f,Mj}$ | fraction |
| Live chains | $\lambda_L = P_0 + Q_0 + R_0$ | mol/L |
| Dead chains | $B_0$ | mol/L |
| Residence time | $\tau = 60\,V/F$ (V in ft³, F in ft³/hr) | min |

---

## 9. Default parameters

These are from `DEFAULT_CONFIG` in `model.py`. Concentrations are in mol/L
and time is in s.

| Group | Values |
|---|---|
| Reactor | V = 650 ft³, T0 = 90 °F, Tᵢ = −32 °F |
| Feeds (lb/hr) | M1 12,000 · M2 20,000 · M3 2,000 · hexane 300,000 · VCl₄ 50 · Al alkyl 156.37 · H₂ 2 |
| Initiation / propagation | k_i1 50, k_i2 5, k11 800, k12 29, k13 189, k21 500, k22 10.9, k31 150 |
| Catalyst | k_a 0.5, k_x 5e-3, k_x2 1e-3, k_x3 1e-3 |
| Termination / transfer | k_t 0.0202, k_t2 1e-3, k_t3 1e-3, k_tr 1e-4, k_tr1 2.37, k_trM2 1e-3 |
| Molecular weights | M1 28, M2 42, M3 120 g/mol |
| Heats of reaction −ΔH | M1 1400, M2 950, M3 700 Btu/lb; ΔH_mix = 1.0e6 Btu/hr |
| Mooney | a = 2e-4, b = 2e-4 |
| Arrhenius | Tr = 90 °F; E = 0 for every constant (k = k₀) |
| Critical C_C1 | c* = 0.1 mol/L |
| Initial conditions | T0 = 90 °F; species at the feed composition |

The rate constants are not measured data. With these defaults the run ends
near steady state at Mn ≈ 22,000, PDI ≈ 2.0, ML ≈ 27, T ≈ 120 °F, a residence
time of about 5 min, and conversions of about 93 % ethylene, 30 % propylene and
65 % diene.
