"""
Simulator runs in the style of EPDM_Data_Generation, for training the neural ODE.

    python generate_training_data.py

Each run follows the shared DOE plan (Experimental_Plan/doe29_plan_c6limited.csv):
50 steps at the base feeds, then one feed at a time stepped to another level and held
for 50 steps (each feed at x0.5, 0.75, 1.25 and 1.5 of base; hexane limited to x0.9 to
1.1), 1,450 steps of 10 s in all, measurement case 2 with noise. The parameters are a
grade's set from Parameter_Sets/. Each run takes the same 28 steps in a new random
order with its own noise seed, so the network sees transitions the fixed plan never
makes. The four original datasets stay held out as the benchmark.

Writes data/{train,val,test}_g<grade>_<n>.csv. The same SEED gives the same runs.
results.csv and results_config.json are not touched.
"""

import json
import random
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from pathlib import Path

import model

HERE = Path(__file__).parent
OUT_DIR = HERE / "data"
REFERENCE_DIR = HERE / "EPDM_Data_Generation"
GRADES = (1, 2, 3, 4)
SPLITS = {"train": 6, "val": 1, "test": 1}     # runs per grade
SEED = 4330

HOLD = 50          # steps at each level, as in the DOE plan
N_STEPS = 1450
# Levels as multiples of the base feed, as in doe29_plan_c6limited.csv (hexane limited).
LEVELS = {"M1": (0.5, 0.75, 1.25, 1.5), "M2": (0.5, 0.75, 1.25, 1.5), "M3": (0.5, 0.75, 1.25, 1.5),
          "C6": (0.9, 0.95, 1.05, 1.1), "cat": (0.5, 0.75, 1.25, 1.5), "Al": (0.5, 0.75, 1.25, 1.5),
          "H2": (0.5, 0.75, 1.25, 1.5)}


def grade_config(grade):
    """The team's full configuration for a grade: its p0, u0, x0, N, dt and measurement settings."""
    return json.loads((REFERENCE_DIR / "Parameter_Sets" / f"results_grade{grade}_config.json").read_text())


def doe_events(base_flows, rng):
    """Step-change events for one run: base for the first HOLD steps, then every (feed, level)
    pair once, in a random order, each held HOLD steps. The last level holds to the end."""
    steps = [(feed, level) for feed, levels in LEVELS.items() for level in levels]
    rng.shuffle(steps)
    events = []
    for i, (feed, level) in enumerate(steps):
        start = HOLD * (i + 1)
        last = i == len(steps) - 1
        events.append({"start": start, "end": None if last else start + HOLD - 1, "target": "flow." + feed,
                       "mode": "set", "value": base_flows[feed] * level})
    return events


def run_one(job):
    grade, split, index, order_seed, noise_seed = job
    cfg = grade_config(grade)
    cfg.pop("controller", None)
    cfg.update(N=N_STEPS, controllers=[], events=doe_events(cfg["u0"]["flows"], random.Random(order_seed)))
    cfg["measure"] = dict(cfg["measure"], case=2, noise=True, seed=noise_seed)
    path = OUT_DIR / f"{split}_g{grade}_{index:03d}.csv"
    try:
        model.run(cfg, path)
        return f"{path.name}"
    except (ValueError, RuntimeError) as exc:      # inputs the simulator rejects or cannot solve
        return f"skipped {path.name}: {exc}"


def main():
    OUT_DIR.mkdir(exist_ok=True)
    rng = random.Random(SEED)
    jobs = [(grade, split, i, rng.randrange(1 << 30), rng.randrange(1, 1 << 30))
            for grade in GRADES for split, count in SPLITS.items() for i in range(count)]
    print(f"Simulating {len(jobs)} runs of {N_STEPS} steps ({len(GRADES)} grades) in parallel...", flush=True)
    with ProcessPoolExecutor() as pool:
        results = list(pool.map(run_one, jobs))
    skipped = [r for r in results if r.startswith("skipped")]
    for r in skipped:
        print(r)
    print(f"Done: {len(results) - len(skipped)} runs in {OUT_DIR}" + (f", {len(skipped)} skipped" if skipped else ""))


if __name__ == "__main__":
    main()
