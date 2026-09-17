#!/usr/bin/env python3
"""Is there any heteroscedasticity here for a variance head to find?

A variance head that carries no information is one of two very different
results, and the aggregate correlation cannot tell them apart:

* the head failed to learn something that was there, or
* the residual really is homoscedastic and there was nothing to learn.

This measures the second directly, without reference to any fitted variance
head, so the ablation's negative result can be attributed rather than merely
reported.

Three things are measured.

**The theory's prediction.** Phase 3 established that this drift is a stationary
Gaussian process plus narrowband sinusoids, which is why linear prediction is
optimal for the mean. The same Gaussianity forces the *conditional variance* of
the optimal predictor to be constant — for jointly Gaussian variables, the
variance of one given the others does not depend on their values. So the theory
says: no heteroscedasticity, except where the process is not Gaussian.

**Where it is not Gaussian.** The jumps. A cable jump is the one non-Gaussian
event in the model, and rows shortly after one should be genuinely harder to
predict. If the residual is elevated there and nowhere else, the honest
conclusion is not "there is nothing to predict" but "what there is to predict
occupies a fraction of a percent of the rows".

**What is achievable.** An upper bound on any variance head: fit a regressor
from the same features to the log squared error, and see how much of its
variation it can explain. A head cannot beat that bound, so if the bound is
near zero the ablation result is explained and no amount of architecture would
change it.

    python scripts/analyse_variance_head.py --out runs/ablation
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from task3 import FEATURE_SPEC, TRAIN_SEEDS, make_control, make_trajectories  # noqa: E402
from qnet_autopilot.compensation.train import (build_dataset,  # noqa: E402
                                               chronological_split, fit_ridge)


def jump_distance(dataset, trajectories) -> np.ndarray:
    """Seconds since the most recent jump, per dataset row (inf if none yet)."""
    distance = np.full(len(dataset), np.inf)
    for index, trajectory in enumerate(trajectories):
        rows = dataset.episode == index
        if not rows.any():
            continue
        times = dataset.time_s[rows]
        jumps = np.sort(np.asarray(trajectory.jump_times_s, dtype=float))
        if len(jumps) == 0:
            continue
        position = np.searchsorted(jumps, times, side="right") - 1
        elapsed = np.where(position >= 0,
                           times - jumps[np.maximum(position, 0)], np.inf)
        distance[rows] = elapsed
    return distance


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("runs/ablation"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    control = make_control()
    trajectories = make_trajectories(TRAIN_SEEDS)
    dataset = build_dataset(trajectories, control, FEATURE_SPEC, rng=0)
    truth = build_dataset(trajectories, control, FEATURE_SPEC, target="true", rng=0)
    train_mask, validation = chronological_split(dataset, 0.85, 0.02)

    # The optimal linear predictor's residual is what a variance head would have
    # to model. Using ridge rather than the network keeps this independent of
    # how well any particular network happened to fit.
    ridge = fit_ridge(dataset, train_mask)
    scaled = ridge.standardiser.transform(dataset.features)
    predicted = np.asarray(ridge.model.predict(scaled), dtype=float)

    squared_error = ((predicted - dataset.targets) ** 2).sum(axis=1)
    true_squared_error = ((predicted - truth.targets) ** 2).sum(axis=1)
    elapsed = jump_distance(dataset, trajectories)

    report: dict[str, float] = {
        "noise_share": float(1.0 - true_squared_error[validation].mean()
                             / squared_error[validation].mean())
    }

    # Over *all* rows, not just validation: whether the residual is elevated
    # after a jump is a property of the process, not of a fit, and the
    # chronological split happens to leave almost no post-jump rows in the
    # validation window — which would make the question unanswerable there.
    print("=== residual against time since the last cable jump (all rows) ===")
    # The last band is rows in trajectories that have not jumped yet. Leaving
    # it out — which a half-open [2, inf) bin does — silently drops well over
    # half the data off the chart, and the shares no longer add up.
    edges = [0.0, 0.02, 0.05, 0.1, 0.5, 2.0]
    quiet = float(np.median(true_squared_error[elapsed > 2.0]))
    bands = [(low, high, f"{low:g}-{high:g}")
             for low, high in zip(edges[:-1], edges[1:])]
    bands.append((2.0, np.inf, "> 2"))
    bands.append((np.inf, np.inf, "sem salto"))

    rows = []
    for low, high, label in bands:
        if np.isinf(low):
            selector = np.isinf(elapsed)
        else:
            selector = (elapsed >= low) & (elapsed < high) & ~np.isinf(elapsed)
        if not selector.any():
            continue
        rows.append({
            "since_jump_s": label,
            "rows": int(selector.sum()),
            "share_of_rows": float(selector.sum() / len(dataset)),
            "median_squared_error": float(np.median(true_squared_error[selector])),
            "ratio_to_quiet": float(np.median(true_squared_error[selector]) / quiet),
        })
    jump_table = pd.DataFrame(rows)
    print(jump_table.round(4).to_string(index=False))
    jump_table.to_csv(args.out / "variance_by_jump.csv", index=False)

    # An upper bound on any variance head: predict the log squared error from
    # the same features. A head reads those features and estimates that
    # quantity, so it cannot beat a model fitted directly to it.
    target = np.log(np.maximum(true_squared_error, 1e-30))
    design = np.hstack([scaled, np.ones((len(scaled), 1))])
    penalty = 1e-6 * np.mean(scaled ** 2) * len(scaled) * np.eye(design.shape[1])
    penalty[-1, -1] = 0.0
    coefficients = np.linalg.solve(
        design[train_mask].T @ design[train_mask] + penalty,
        design[train_mask].T @ target[train_mask])
    residual = target[validation] - design[validation] @ coefficients
    report["log_error_r_squared"] = float(1.0 - residual.var()
                                          / target[validation].var())
    report["log_error_spread"] = float(target[validation].std())

    print("\n=== ceiling for any variance head ===")
    print(f"  R2 of a linear fit predicting log squared error: "
          f"{report['log_error_r_squared']:+.4f}")

    post_jump = elapsed < 0.1
    report["post_jump_row_share"] = float(post_jump.sum() / len(dataset))
    report["post_jump_error_share"] = float(
        true_squared_error[post_jump].sum() / true_squared_error.sum())

    print("\n=== what a perfect variance head would be reacting to ===")
    print(f"  rows within 100 ms of a jump: {report['post_jump_row_share']:.4%}")
    print(f"  their share of the total squared error: "
          f"{report['post_jump_error_share']:.4%}")
    print(f"  target noise as a share of the residual: {report['noise_share']:.1%}")

    (args.out / "variance_diagnosis.json").write_text(json.dumps(report, indent=2))
    print(f"\nwrote {args.out}/variance_diagnosis.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
