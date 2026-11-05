# EPDM Data-Generation Package

This package contains four datasets generated with four distinct simulator parameter sets. Grades 1–4 are dataset labels for the four parameter sets; they are not claims about commercial rubber grades.

## Files

- `Parameter_Sets/results_gradeN_config.json` — complete configuration used for dataset N, including its model parameters, initial conditions, measurement settings, and embedded input-event profiles.
- `Generated_Data/results_gradeN.csv` — generated output data corresponding to that configuration.
- `Experimental_Plan/doe29_plan_c6limited.csv` — shared experimental input plan supplied with the original archive.

Replace `N` with `1`, `2`, `3`, or `4`. Keep each configuration paired with the CSV bearing the same grade number.

## Simulation and measurement settings

The four configuration files specify the same run length and sampling interval: **N = 1450 steps** and **dt = 10 seconds**, for a simulated duration of **14,500 seconds**. The initial feed settings and input-event plan are shared across the four configurations; the model parameter values are what distinguish the four parameter sets.

All four configurations specify measurement **case 2**, measurement noise enabled, and random seed **0**. The configured noise levels (`sigma`) are:

| Measurement | Sigma |
|---|---:|
| ML | 0.02 |
| Mw | 0.02 |
| Mn | 0.02 |
| X_M1 | 0.02 |
| X_M3 | 0.02 |
| T | 1.0 |
| PR | 0.02 |

## Pairing

| Dataset | Configuration | Output data |
|---|---|---|
| Grade 1 | `Parameter_Sets/results_grade1_config.json` | `Generated_Data/results_grade1.csv` |
| Grade 2 | `Parameter_Sets/results_grade2_config.json` | `Generated_Data/results_grade2.csv` |
| Grade 3 | `Parameter_Sets/results_grade3_config.json` | `Generated_Data/results_grade3.csv` |
| Grade 4 | `Parameter_Sets/results_grade4_config.json` | `Generated_Data/results_grade4.csv` |

## Scope

The simulator source code is intentionally not included because it is maintained by the team member who created it. This package is for sharing the parameter configurations, experimental plan, and generated datasets.
