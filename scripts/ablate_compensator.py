#!/usr/bin/env python3
"""Does a variance head help a compensator that already ties the Wiener filter?

Phase 3 measured that a network does not beat linear prediction at this fibre's
*mean*, and explained why. This asks the separate question the negative result
left open: whether predicting the *variance* — and standing down when it is
high — changes the closed-loop QBER.

Three models, one code path
---------------------------
    mean-only     the same network and the same fit, with the likelihood never
                  switched on. The like-for-like control.
    dual-head     mean and log-variance, Gaussian likelihood, correction
                  applied in full.
    dual-head +   the same weights, with the correction shrunk towards
    shrinkage     persistence by the model's own confidence.

The middle rung is what separates "the variance head changed the training" from
"the variance head changed the control", which are different claims and are
usually reported as one.

Before any of that, the variance head is scored on its own: if predicted
variance does not track squared error, the shrinkage is a constant gain wearing
a posterior's clothes, and the closed-loop numbers would be uninterpretable.

    python scripts/ablate_compensator.py --out runs/ablation
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from task3 import (FEATURE_SPEC, TEST_SEEDS, TRAIN_SEEDS,  # noqa: E402
                   VALIDATION_SEEDS, make_control, make_trajectories)
from qnet_autopilot.compensation import run_closed_loop  # noqa: E402
from qnet_autopilot.compensation.heteroscedastic import (  # noqa: E402
    ShrinkingCompensator, fit_heteroscedastic)
from qnet_autopilot.compensation.train import (build_dataset,  # noqa: E402
                                               chronological_split, fit_ridge)


def score(trajectories, compensator, control, seed: int = 7000) -> dict:
    summaries = [run_closed_loop(trajectory, compensator, control, rng=seed + index
                                 ).summary()
                 for index, trajectory in enumerate(trajectories)]
    aggregate = {key: float(np.mean([item[key] for item in summaries]))
                 for key in summaries[0] if key != "compensator"}
    aggregate["compensator"] = compensator.name
    return aggregate


def _spearman(left, right) -> float:
    ranks = lambda values: np.argsort(np.argsort(values))
    return float(np.corrcoef(ranks(left), ranks(right))[0, 1])


def calibration(model, features, targets, true_targets=None) -> dict:
    """Does the predicted variance know anything?

    Three numbers, and the third is the one that decides how to read the other
    two.

    The Spearman correlation between predicted variance and actual squared error
    says whether the *ordering* is right — whether the model knows which rows are
    hard. The ratio of mean squared error to mean predicted variance says
    whether the *scale* is right; a well-calibrated Gaussian puts it at 1.

    The third repeats the correlation against the error measured from the
    **true** future rotation instead of the noisy future measurement. That
    matters because the target a deployable model is fitted on carries the
    monitoring channel's own shot noise, and that noise is the same size on
    every row. If a model knows which rows are genuinely unpredictable, the
    correlation against the true error will show it even when the correlation
    against the measured error is buried — and if neither shows anything, the
    head really does know nothing.
    """
    mean, log_variance = model.forward(features)
    variance = np.exp(log_variance)
    squared_error = (mean - targets) ** 2

    report = {
        "spearman_variance_vs_error": _spearman(variance.ravel(),
                                                squared_error.ravel()),
        "calibration_ratio": float(np.mean(squared_error) / np.mean(variance)),
        "variance_spread": float(np.std(np.log(variance.ravel()))),
        "validation_mse": float(np.mean(squared_error)),
        "saturated": model.saturated_fraction(features),
    }
    if true_targets is not None:
        true_error = (mean - true_targets) ** 2
        report["spearman_variance_vs_true_error"] = _spearman(
            variance.ravel(), true_error.ravel())
        report["true_mse"] = float(np.mean(true_error))
        # How much of the residual is target noise rather than genuine
        # unpredictability: if these are close, there is little left to predict.
        report["noise_share"] = float(1.0 - np.mean(true_error)
                                      / np.mean(squared_error))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("runs/ablation"))
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--hidden", type=int, nargs="+", default=[192, 192])
    parser.add_argument("--test-runs", type=int, default=8)
    parser.add_argument("--seeds", type=int, default=3)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    control = make_control()
    print("building the dataset...")
    training = make_trajectories(TRAIN_SEEDS)
    dataset = build_dataset(training, control, FEATURE_SPEC, rng=0)
    train_mask, validation_mask = chronological_split(dataset, 0.85, 0.02)
    print(f"  {int(train_mask.sum())} train rows, "
          f"{int(validation_mask.sum())} validation rows")

    # The same rows with the noise-free target, for the diagnostic above. Same
    # seed, so the measurement noise in the *features* is identical and only
    # the target changes.
    truth = build_dataset(training, control, FEATURE_SPEC, target="true", rng=0)

    test = make_trajectories(TEST_SEEDS[:args.test_runs])
    ridge = fit_ridge(dataset, train_mask)

    rows, calibrations, histories = [], [], {}

    for seed in range(args.seeds):
        print(f"\n=== seed {seed} ===")

        # The control: identical code, likelihood never switched on.
        mean_only, mean_history = fit_heteroscedastic(
            dataset, train_mask, validation_mask, hidden=tuple(args.hidden),
            epochs=args.epochs, warmup_epochs=args.epochs, seed=seed,
            name="somente média")
        mean_only.shrink = False

        dual, dual_history = fit_heteroscedastic(
            dataset, train_mask, validation_mask, hidden=tuple(args.hidden),
            epochs=args.epochs, seed=seed, name="dual-head")
        histories[f"mean_only_{seed}"] = mean_history
        histories[f"dual_head_{seed}"] = dual_history

        validation_features = dual.standardiser.transform(
            dataset.features[validation_mask])
        validation_targets = dataset.targets[validation_mask]
        report = calibration(dual.model, validation_features, validation_targets,
                             truth.targets[validation_mask])
        report["seed"] = seed
        calibrations.append(report)
        print(f"  variância vs erro: spearman {report['spearman_variance_vs_error']:+.3f}"
              f" (vs erro verdadeiro {report.get('spearman_variance_vs_true_error', float('nan')):+.3f})"
              f"  calibração {report['calibration_ratio']:.2f}"
              f"  espalhamento {report['variance_spread']:.2f}"
              f"  saturado {report['saturated']:.3f}"
              f"  ruído no alvo {report.get('noise_share', float('nan')):.2f}")

        unshrunk = ShrinkingCompensator(
            model=dual.model, spec=dual.spec, standardiser=dual.standardiser,
            prior_variance=dual.prior_variance, shrink=False,
            name="dual-head (sem encolhimento)")

        for compensator in (mean_only, unshrunk, dual):
            record = score(test, compensator, control)
            record["seed"] = seed
            rows.append(record)
            print(f"  {record['compensator']:32s} QBER {record['mean_qber']:.5f}"
                  f"  chave {record['mean_key_rate_hz']:.0f} Hz"
                  f"  disp {record['uptime']:.4f}")

    ridge_record = score(test, ridge, control)
    ridge_record["seed"] = -1
    rows.append(ridge_record)
    print(f"\n  {ridge_record['compensator']:32s} QBER {ridge_record['mean_qber']:.5f}"
          f"  chave {ridge_record['mean_key_rate_hz']:.0f} Hz")

    frame = pd.DataFrame(rows)
    frame.to_csv(args.out / "compensator_ablation.csv", index=False)
    pd.DataFrame(calibrations).to_csv(args.out / "compensator_calibration.csv",
                                      index=False)
    (args.out / "compensator_histories.json").write_text(json.dumps(histories))

    print("\n=== summary over seeds ===")
    summary = frame[frame["seed"] >= 0].groupby("compensator", sort=False).agg(
        mean_qber=("mean_qber", "mean"),
        qber_std=("mean_qber", "std"),
        key_rate=("mean_key_rate_hz", "mean"),
        uptime=("uptime", "mean"))
    print(summary.round(5).to_string())
    print(f"\nridge (Wiener): QBER {ridge_record['mean_qber']:.5f}")

    print("\n=== paired differences over seeds ===")
    pivot = frame[frame["seed"] >= 0].pivot(index="seed", columns="compensator",
                                            values="mean_qber")
    pairs = [("dual-head", "somente média"),
             ("dual-head", "dual-head (sem encolhimento)"),
             ("dual-head (sem encolhimento)", "somente média")]
    differences = []
    for later, earlier in pairs:
        if later not in pivot or earlier not in pivot:
            continue
        delta = (pivot[later] - pivot[earlier]).dropna()
        differences.append({
            "change": f"{later} - {earlier}",
            "mean_delta_qber": float(delta.mean()),
            "stderr": float(delta.std(ddof=1) / np.sqrt(len(delta)))
            if len(delta) > 1 else 0.0,
            "improvements": int((delta < 0).sum()),
            "regressions": int((delta > 0).sum()),
        })
    paired = pd.DataFrame(differences)
    paired.to_csv(args.out / "compensator_paired.csv", index=False)
    print(paired.round(6).to_string(index=False))

    print(f"\nwrote CSVs to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
