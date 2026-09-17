"""Command-line entry point: ``python -m qnet_autopilot``.

Four subcommands, matching the four things you do with this package before any
agent exists:

``info``
    What SeQUeNCe is being used, and whether any of its dependencies are
    standing in.  Run this first in a new environment — a dataset generated
    under the compatibility shims is fine, but you should know that it was.

``validate``
    Re-run every cross-check and, optionally, write the HTML report.

``generate``
    Produce a dataset and its metadata sidecar.

``budget``
    Print the link budget for a span or a scenario — the quickest way to see
    whether a topology is even in the feasible regime before spending an hour
    simulating it.

Built on argparse rather than a CLI framework so the package has no dependency
beyond its numerical stack.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

__all__ = ["main", "build_parser"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qnet_autopilot",
        description="Physics and data layer for learned quantum-network control.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("info", help="report the SeQUeNCe build and any shims in use")

    validate = subparsers.add_parser(
        "validate", help="cross-check the analytic models against SeQUeNCe")
    validate.add_argument("--quick", action="store_true",
                          help="smaller samples; a smoke test, not a report")
    validate.add_argument("--report", type=Path, default=None,
                          help="also write the HTML report to this path")
    validate.add_argument("--seed", type=int, default=0)

    generate = subparsers.add_parser("generate", help="produce a dataset")
    generate.add_argument("dataset", choices=["polarization", "link", "capacity", "demands"])
    generate.add_argument("--out", type=Path, required=True, help="output .csv path")
    generate.add_argument("--seed", type=int, default=0)
    generate.add_argument("--duration", type=float, default=600.0,
                          help="seconds of simulated time (time-series datasets)")
    generate.add_argument("--dt", type=float, default=0.1,
                          help="sampling interval in seconds (time-series datasets)")
    generate.add_argument("--episodes", type=int, default=8,
                          help="independent runs (polarization)")
    generate.add_argument("--attempts", type=int, default=200,
                          help="generation attempts per run (link)")
    generate.add_argument("--scenario", default="ring",
                          help="topology name for the traffic datasets")

    budget = subparsers.add_parser("budget", help="print the link budget for a span")
    budget.add_argument("--km", type=float, action="append", default=None,
                        help="span length; repeatable")
    budget.add_argument("--scenario", default=None,
                        help="use every span in this scenario instead of --km")
    budget.add_argument("--scheme", default="single_heralded",
                        choices=["single_heralded", "barrett_kok"])

    return parser


def _command_info() -> int:
    from .compat import bootstrap

    status = bootstrap()
    print(status.summary())
    if status.shimmed:
        print("\nDatasets generated here record which dependencies were shimmed.\n"
              "Install the real ones (`pip install sequence`) when you can.")
    return 0


def _command_validate(args) -> int:
    from .validation import render_report, run_all

    results = run_all(quick=args.quick, seed=args.seed)
    width = max(len(result.name) for result in results)
    for result in results:
        mark = "PASS" if result.passed else "FAIL"
        print(f"[{mark}] {result.name:<{width}}  ({result.kind})")
        print(f"        {result.summary}")

    if args.report is not None:
        path = render_report(results, args.report)
        print(f"\nreport written to {path}")

    return 0 if all(result.passed for result in results) else 1


def _command_generate(args) -> int:
    from .datagen import (generate_capacity_trace, generate_demand_stream,
                          generate_link_dataset, generate_polarization_dataset)
    from .network import scenario

    if args.dataset == "polarization":
        dataset = generate_polarization_dataset(
            num_episodes=args.episodes, duration_s=args.duration, dt_s=args.dt,
            seed=args.seed)
    elif args.dataset == "link":
        dataset = generate_link_dataset(num_attempts=args.attempts, seed=args.seed)
    elif args.dataset == "capacity":
        dataset = generate_capacity_trace(
            scenario(args.scenario), duration_s=args.duration,
            dt_s=max(args.dt, 1.0), seed=args.seed)
    else:
        dataset = generate_demand_stream(
            scenario(args.scenario), duration_s=args.duration, seed=args.seed)

    path = dataset.save(args.out)
    print(dataset.describe())
    print(f"\nwritten to {path}")
    print(f"metadata   {path.with_suffix(path.suffix + '.meta.json')}")
    return 0


def _command_budget(args) -> int:
    from .network import scenario
    from .physics.link import LinkBudget

    if args.scenario:
        graph = scenario(args.scenario)
        spans = sorted({round(float(data["length"]), 3)
                        for *_, data in graph.edges(data=True)})
    else:
        spans = args.km or [10.0, 25.0, 50.0, 100.0]

    header = f"{'km':>8} {'eta/arm':>9} {'p(attempt)':>12} {'pairs/s':>12} {'wait':>12}"
    print(header)
    print("-" * len(header))
    for length_km in spans:
        budget = LinkBudget.compute(length_km, scheme=args.scheme)
        wait = ("never" if budget.expected_time_s == float("inf")
                else f"{budget.expected_time_s:.3g} s")
        print(f"{length_km:8.1f} {budget.arm_efficiency:9.4f} "
              f"{budget.attempt_probability:12.3e} {budget.rate_hz:12.3g} {wait:>12}")
    print(f"\nscheme: {args.scheme}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run the CLI. Returns the process exit code."""
    args = build_parser().parse_args(argv)
    handlers = {
        "info": lambda: _command_info(),
        "validate": lambda: _command_validate(args),
        "generate": lambda: _command_generate(args),
        "budget": lambda: _command_budget(args),
    }
    return handlers[args.command]()


if __name__ == "__main__":
    sys.exit(main())
