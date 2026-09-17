#!/usr/bin/env python3
"""What the compensator costs in fixed point, and where the error comes from.

The question behind this is whether the model needs an FPGA. Timing the
datapath says the arithmetic does not: 46 656 multiply-accumulates at a 2 ms
control period is 23 MMAC/s. The measurement that does decide the port is the
word length — it sizes the multipliers, the parameter ROM and the CORDIC — and
that is what this sweeps.

Three studies, in the order they answer things:

1. **Word length.** The whole datapath, from the SO(3) logarithms to the
   exponential, at several widths, scored by the same closed loop and the same
   held-out seeds as the floating-point model.
2. **CORDIC depth.** How many rotation stages the transcendental blocks need.
   The logarithms are 72% of the datapath's time, so this is the knob that
   sizes the expensive part.
3. **Attribution.** The same widths with the transcendental blocks left in
   double precision, which separates "the weights are too coarse" from "the
   CORDIC is too short". Not a deployment option — a diagnostic one.

    python scripts/evaluate_quantisation.py --out runs/quantisation

Note:
    Calibration runs on the *training* seeds and on traces from the closed loop
    itself, not on the trajectory's true rotations. Both matter: calibrating on
    the test seeds would size the format knowing the excursions it has to
    survive, and calibrating on the truth undersizes it, because in closed loop
    the history is made of noisy monitoring estimates whose logarithms are
    larger.
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from qnet_autopilot.compat import bootstrap  # noqa: E402

BOOTSTRAP_STATUS = bootstrap()

from task3 import (TEST_SEEDS, TRAIN_SEEDS, make_control,  # noqa: E402
                   make_trajectories)
from qnet_autopilot.compensation import run_closed_loop  # noqa: E402
from qnet_autopilot.deploy import (CordicConfig, QuantisationConfig,  # noqa: E402
                                   build_quantised, collect_calibration)

#: Datapath widths the sweep walks. Eight is where an int8 port would sit, 16
#: is one DSP48 multiplier input, 24 and 32 are the widths a port reaches for
#: when 16 turns out not to be enough.
WIDTHS = (8, 10, 12, 14, 16, 18, 20, 24)

#: CORDIC stages. Each buys about one bit of angle, and the logarithms are the
#: dominant block, so this is the knob that sizes the expensive part.
CORDIC_ITERATIONS = (6, 8, 10, 12, 16, 20)

#: Widths the attribution study repeats with the transcendental blocks in
#: double precision.
ATTRIBUTION_WIDTHS = (10, 12, 14, 16, 20)


def score(trajectories, compensator, control, seeds) -> dict:
    """Closed loop over the held-out trajectories, averaged."""
    rows = [run_closed_loop(trajectory, compensator, control, rng=7000 + index)
            .summary()
            for index, trajectory in enumerate(trajectories)]
    aggregate = {key: float(np.mean([row[key] for row in rows]))
                 for key in rows[0] if key != "compensator"}
    aggregate["qber_stderr"] = float(
        np.std([row["mean_qber"] for row in rows], ddof=1) / np.sqrt(len(rows)))
    return aggregate


def study(fitted, models, calibration, trajectories, control, seeds,
          configurations) -> pd.DataFrame:
    """One row per model and configuration, plus the float reference."""
    rows = []
    for name in models:
        reference = score(trajectories, fitted[name], control, seeds)
        reference.update(model=name, width=np.nan, cordic_iterations=np.nan,
                         variant="float64", multiply_accumulates=np.nan,
                         parameter_kib=np.nan, saturations=0)
        rows.append(reference)
        print(f"  {name} float: QBER {reference['mean_qber']:.4f}")

        for config in configurations:
            quantised = build_quantised(fitted[name], calibration[name], config)
            record = score(trajectories, quantised, control, seeds)
            report = quantised.report()
            record.update(model=name, width=config.width,
                          cordic_iterations=report["cordic_iterations"],
                          variant=config.labelled(),
                          multiply_accumulates=report["multiply_accumulates"],
                          parameter_kib=report["parameter_kib"],
                          saturations=report["saturations"])
            record["qber_ratio"] = record["mean_qber"] / reference["mean_qber"]
            rows.append(record)
            print(f"  {name} {config.labelled()}: QBER {record['mean_qber']:.4f} "
                  f"({record['qber_ratio']:.2f}x), sat {report['saturations']}")
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("runs/quantisation"))
    parser.add_argument("--models", type=Path,
                        default=Path("runs/compensator/main.pkl"))
    parser.add_argument("--test-seeds", type=int, default=6)
    parser.add_argument("--calibration-seeds", type=int, default=3)
    parser.add_argument("--only", nargs="+", default=["width", "cordic",
                                                      "attribution"],
                        choices=["width", "cordic", "attribution"])
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    with open(args.models, "rb") as handle:
        fitted = pickle.load(handle)
    models = ["mlp", "ridge"]
    control = make_control()

    calibration_trajectories = make_trajectories(
        TRAIN_SEEDS[:args.calibration_seeds])
    print(f"calibrating on {args.calibration_seeds} training trajectories")
    calibration = {name: collect_calibration(fitted[name],
                                             calibration_trajectories, control)
                   for name in models}
    print(f"  {calibration['mlp']['raw'].shape[0]} windows recorded")

    seeds = TEST_SEEDS[:args.test_seeds]
    trajectories = make_trajectories(seeds)

    if "width" in args.only:
        print(f"\n=== word length, whole datapath, {len(seeds)} held-out seeds ===")
        configurations = [QuantisationConfig(width=width,
                                             cordic=CordicConfig(16, 20, 32))
                          for width in WIDTHS]
        frame = study(fitted, models, calibration, trajectories, control, seeds,
                      configurations)
        frame.to_csv(args.out / "width.csv", index=False)

    if "cordic" in args.only:
        print("\n=== CORDIC depth, at 16 bits ===")
        configurations = [
            QuantisationConfig(width=16,
                               cordic=CordicConfig(iterations, 20, 32),
                               name=f"16 bits, CORDIC {iterations}")
            for iterations in CORDIC_ITERATIONS]
        frame = study(fitted, models, calibration, trajectories, control, seeds,
                      configurations)
        frame.to_csv(args.out / "cordic.csv", index=False)

    if "attribution" in args.only:
        print("\n=== the same widths with the transcendental blocks in float ===")
        configurations = [
            QuantisationConfig(width=width, quantise_transcendental=False,
                               name=f"{width} bits, transcendental em float")
            for width in ATTRIBUTION_WIDTHS]
        frame = study(fitted, models, calibration, trajectories, control, seeds,
                      configurations)
        frame.to_csv(args.out / "attribution.csv", index=False)

    print(f"\nwrote CSVs to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
