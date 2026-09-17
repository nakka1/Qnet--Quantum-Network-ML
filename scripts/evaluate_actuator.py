#!/usr/bin/env python3
"""Does phase 3's result survive a real polarisation controller?

Phase 3 assumed the requested rotation appears on the fibre exactly. Its own
limits section flagged that as the assumption most likely to matter, and named
the endless reset as "exactly the nonlinearity that would change the conclusion"
about neural networks. This measures it.

Three questions, in order of what the numbers turned out to say:

1. **Does the ranking survive a rate-limited device?** The predictor tracks the
   40 Hz vibration band; the classical filter smooths it away. Tracking costs
   actuator motion, so a device that cannot move fast enough should hurt the
   predictor more — possibly enough to reverse the ordering.
2. **How much motion does each compensator actually demand?** The mechanism, if
   the answer to (1) is yes.
3. **Does the reset fire at all, and does it discriminate?** The original
   hypothesis. Resets need accumulated drift, so this uses long runs and a
   tight range, which is the only regime where the question is answerable.

    python scripts/evaluate_actuator.py --out runs/actuator
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

from task3 import TEST_SEEDS, make_control, make_trajectories  # noqa: E402
from qnet_autopilot.compensation import (AlphaBetaFilter, NoCompensation,  # noqa: E402
                                         Persistence, run_closed_loop)
from qnet_autopilot.compensation.actuator import (ActuatorConfig,  # noqa: E402
                                                  PolarisationActuator)

#: Device classes, in radians per second. A motorised waveplate sits at the
#: bottom, a lithium-niobate controller at the top, piezo squeezers between.
SLEW_RATES = (30, 60, 120, 250, 500, 1000, 3000, 10000)


def compensators(fitted):
    return [
        Persistence(),
        AlphaBetaFilter(alpha=0.05, beta=0.005, name="filtro alfa-beta"),
        fitted["ridge"],
        fitted["mlp"],
    ]


def score(trajectories, compensator, control, actuator_config, seed=7000) -> dict:
    rows = []
    for index, trajectory in enumerate(trajectories):
        actuator = (None if actuator_config is None
                    else PolarisationActuator(actuator_config))
        rows.append(run_closed_loop(trajectory, compensator, control,
                                    rng=seed + index, actuator=actuator).summary())
    aggregate = {key: float(np.mean([row[key] for row in rows]))
                 for key in rows[0] if key != "compensator"}
    aggregate["compensator"] = compensator.name
    aggregate["qber_stderr"] = float(
        np.std([row["mean_qber"] for row in rows], ddof=1) / np.sqrt(len(rows)))
    return aggregate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("runs/actuator"))
    parser.add_argument("--models", type=Path,
                        default=Path("runs/compensator/main.pkl"))
    parser.add_argument("--runs", type=int, default=4)
    parser.add_argument("--long-duration", type=float, default=200.0)
    parser.add_argument("--long-runs", type=int, default=3)
    parser.add_argument("--only", nargs="+", default=["slew", "reset", "corner"],
                        choices=["slew", "reset", "corner"])
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    with open(args.models, "rb") as handle:
        fitted = pickle.load(handle)
    control = make_control()

    if "slew" in args.only:
        print("=== how fast does the device have to be? ===")
        test = make_trajectories(TEST_SEEDS[:args.runs])
        rows = []

        for compensator in compensators(fitted):
            ideal = score(test, compensator, control, None)
            ideal.update(slew_rate=np.inf, range_rad=np.inf)
            rows.append(ideal)

        for slew in SLEW_RATES:
            config = ActuatorConfig(slew_rate_rad_per_s=slew)
            for compensator in compensators(fitted):
                record = score(test, compensator, control, config)
                record.update(slew_rate=slew, range_rad=config.range_rad)
                rows.append(record)
            print(f"  {slew:>6} rad/s done")

        slew_frame = pd.DataFrame(rows)
        slew_frame.to_csv(args.out / "slew_sweep.csv", index=False)
        print(slew_frame.pivot(index="slew_rate", columns="compensator",
                               values="mean_qber").round(4).to_string())

        print("\n=== motion demanded, at a device fast enough not to bind ===")
        travel = slew_frame[slew_frame["slew_rate"] == 10000][
            ["compensator", "actuator_travel_rad", "mean_qber"]].copy()
        travel["rad_per_s"] = travel["actuator_travel_rad"] / 20.0
        print(travel.round(2).to_string(index=False))
        travel.to_csv(args.out / "travel.csv", index=False)

    if "reset" in args.only:
        # Resets need accumulated drift, and 20 seconds does not accumulate
        # any. Long runs and a tight range are the only regime in which the
        # original hypothesis is even testable.
        print(f"\n=== do resets fire? ({args.long_duration:g} s runs) ===")
        long_runs = make_trajectories(TEST_SEEDS[:args.long_runs],
                                      duration_s=args.long_duration)
        rows = []
        for range_turns in (2.0, 4.0, 12.0):
            config = ActuatorConfig(range_rad=range_turns * np.pi,
                                    slew_rate_rad_per_s=3000.0)
            for compensator in compensators(fitted):
                record = score(long_runs, compensator, control, config)
                record["range_turns"] = range_turns
                record["resets_per_minute"] = (record["actuator_resets"] * 60.0
                                               / args.long_duration)
                rows.append(record)
            print(f"  range {range_turns:g}pi done")

        reset_frame = pd.DataFrame(rows)
        reset_frame.to_csv(args.out / "reset_sweep.csv", index=False)
        print(reset_frame.pivot(index="range_turns", columns="compensator",
                                values="resets_per_minute").round(2).to_string())
        print()
        print(reset_frame.pivot(index="range_turns", columns="compensator",
                                values="mean_qber").round(4).to_string())

    if "corner" in args.only:
        # The reset sweep used a fast device and found the reset costs nothing;
        # the slew sweep used a generous range and found speed is what matters.
        # Neither covers a device that is slow *and* short of travel at the same
        # time, which is the one corner where the original hypothesis — that the
        # endless reset is the nonlinearity separating the compensators — could
        # still be true. A cheap motorised waveplate is exactly that device.
        print(f"\n=== slow and short of travel at once ({args.long_duration:g} s runs) ===")
        long_runs = make_trajectories(TEST_SEEDS[:args.long_runs],
                                      duration_s=args.long_duration)
        rows = []
        for slew in (30, 60, 120, 250):
            for range_turns in (2.0, 6.0):
                config = ActuatorConfig(range_rad=range_turns * np.pi,
                                        slew_rate_rad_per_s=slew)
                for compensator in compensators(fitted):
                    record = score(long_runs, compensator, control, config)
                    record["slew_rate"] = slew
                    record["range_turns"] = range_turns
                    record["resets_per_minute"] = (record["actuator_resets"] * 60.0
                                                   / args.long_duration)
                    rows.append(record)
                print(f"  {slew:>4} rad/s, {range_turns:g}pi done")

        corner = pd.DataFrame(rows)
        corner.to_csv(args.out / "corner_sweep.csv", index=False)
        print(corner.pivot(index=["slew_rate", "range_turns"],
                           columns="compensator", values="mean_qber")
              .round(4).to_string())
        print()
        print(corner.pivot(index=["slew_rate", "range_turns"],
                           columns="compensator", values="resets_per_minute")
              .round(1).to_string())
        print()
        print(corner.pivot(index=["slew_rate", "range_turns"],
                           columns="compensator",
                           values="actuator_resetting_fraction")
              .round(4).to_string())

    print(f"\nwrote CSVs to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
