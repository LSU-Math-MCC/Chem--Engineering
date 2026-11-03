"""
Simulator runs with random input changes, for training the neural ODE.

    python generate_training_data.py

Writes data/train_*.csv, val_*.csv and test_*.csv. The same random seed gives the
same runs. results.csv and results_config.json are not touched.
"""

import random
from copy import deepcopy
from pathlib import Path

import model

OUT_DIR = Path(__file__).with_name("data")
SPLITS = {"train": 24, "val": 4, "test": 8}
SEED = 4330

# Range of the random factor for each feed; T_i is set to a value in T_I_RANGE, deg F.
FLOW_RANGES = {"flow.M1": (0.5, 3.0), "flow.M2": (0.5, 2.0), "flow.M3": (0.5, 2.0), "flow.C6": (0.8, 1.2),
               "flow.cat": (0.6, 1.6), "flow.Al": (0.6, 1.6), "flow.H2": (0.5, 3.0)}
T_I_RANGE = (-50.0, -10.0)
TARGETS = list(FLOW_RANGES) + ["T_i"]


def random_event(rng, start):
    target = rng.choice(TARGETS)
    if target == "T_i":
        return {"start": start, "end": None, "target": target, "mode": "set", "value": rng.uniform(*T_I_RANGE)}
    return {"start": start, "end": None, "target": target, "mode": "multiply", "value": rng.uniform(*FLOW_RANGES[target])}


def random_config(rng, seed):
    """DEFAULT_CONFIG with one change from the start, 1 to 3 later ones, and measure case 2 with noise."""
    cfg = deepcopy(model.DEFAULT_CONFIG)
    cfg["measure"].update(case=2, noise=True, seed=seed)
    cfg["events"] = [random_event(rng, 0)] + [random_event(rng, rng.randint(20, 130)) for _ in range(rng.randint(1, 3))]
    return cfg


def main():
    OUT_DIR.mkdir(exist_ok=True)
    rng = random.Random(SEED)
    seed, skipped = 1000, 0
    for split, count in SPLITS.items():
        for i in range(count):
            while True:
                seed += 1
                try:
                    model.run(random_config(rng, seed), OUT_DIR / f"{split}_{i:03d}.csv")
                    break
                except (ValueError, RuntimeError) as exc:     # inputs the simulator rejects or cannot solve
                    skipped += 1
                    print(f"Skipped a run: {exc}")
        print(f"Wrote {count} {split} runs", flush=True)
    print(f"Done: {OUT_DIR}" + (f" ({skipped} runs skipped)" if skipped else ""))


if __name__ == "__main__":
    main()
