#!/usr/bin/env python3
"""Write the compensator out as Verilog, at a chosen word length.

The defaults are the ones ``scripts/evaluate_quantisation.py`` measured: 16
bits for the datapath and 8 CORDIC stages. Sixteen is one comfortable step
above the 14-bit cliff where the port collapses, and eight stages is where the
CORDIC stops mattering.

    python scripts/emit_rtl.py --model ridge --out rtl/ridge
    python scripts/emit_rtl.py --model mlp --width 16 --out rtl/mlp

The emitted directory carries a self-checking testbench whose vectors are the
fixed-point reference's own intermediate values. **It has never been run** —
there is no HDL simulator in the container this was developed in — so the first
thing to do with the output is::

    cd rtl/ridge && iverilog -o qnet_tb qnet_tb.v qnet_mac.v && vvp qnet_tb
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from qnet_autopilot.compat import bootstrap  # noqa: E402

BOOTSTRAP_STATUS = bootstrap()

from task3 import TRAIN_SEEDS, make_control, make_trajectories  # noqa: E402
from qnet_autopilot.deploy import (CordicConfig, QuantisationConfig,  # noqa: E402
                                   build_quantised, collect_calibration,
                                   emit_verilog)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="ridge", choices=["ridge", "mlp"])
    parser.add_argument("--models", type=Path,
                        default=Path("runs/compensator/main.pkl"))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--cordic", type=int, default=8)
    parser.add_argument("--calibration-seeds", type=int, default=3)
    parser.add_argument("--vectors", type=int, default=64)
    args = parser.parse_args()

    destination = args.out or Path("rtl") / args.model
    with open(args.models, "rb") as handle:
        fitted = pickle.load(handle)

    control = make_control()
    print(f"calibrating {args.model} on {args.calibration_seeds} training "
          f"trajectories")
    calibration = collect_calibration(
        fitted[args.model], make_trajectories(TRAIN_SEEDS[:args.calibration_seeds]),
        control)

    config = QuantisationConfig(width=args.width,
                                cordic=CordicConfig(args.cordic, 20, 32))
    quantised = build_quantised(fitted[args.model], calibration, config)
    summary = emit_verilog(quantised, destination, inputs=calibration["raw"],
                           vectors=args.vectors)

    report = quantised.report()
    print(f"\nwrote {len(summary['files'])} files to {destination}")
    print(f"  {report['multiply_accumulates']} MAC por inferência, "
          f"{report['parameter_kib']:.2f} KiB de parâmetros a {args.width} bits")
    print(f"  {report['logarithms_per_step']} logaritmos de SO(3) por passo, "
          f"CORDIC de {args.cordic} estágios")
    print(f"  {summary.get('vectors', 0)} vetores de ouro para o testbench")
    print(f"\nrode o bench antes de confiar no RTL:\n"
          f"  cd {destination} && iverilog -o qnet_tb qnet_tb.v qnet_mac.v "
          f"&& vvp qnet_tb")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
