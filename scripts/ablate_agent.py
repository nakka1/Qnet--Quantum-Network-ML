#!/usr/bin/env python3
"""Does a shared trunk beat two separate networks? And what else helps?

The experiment is a ladder, not a shootout. Each variant adds exactly one thing
to the one before it, and every variant is trained on the same set of seeds, so
the effect of a change is the paired difference between two adjacent rungs
rather than a comparison of two arbitrarily different setups.

    baseline      as shipped: separate actor and critic, 64x64, 300 iterations
    + norm        running feature normalisation
    + capacity    wider networks, more iterations, decayed step size
    dual-head     the same capacity budget, but one shared state trunk

The dual-head claim is therefore ``dual-head`` minus ``+ capacity``, on shared
seeds — everything else held fixed, including the training loop, which both
architectures drive through the same protocol.

Why several seeds
-----------------
Policy gradient on this task has large seed-to-seed variance; the phase-2 report
says so explicitly and reports one training run. A one-seed comparison between
architectures would measure that variance. With paired seeds the difference has
a standard error, and a change that does not clear it is reported as not
clearing it.

    python scripts/ablate_agent.py --out runs/ablation --seeds 5
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from task import make_env, make_request  # noqa: E402
from qnet_autopilot.agents.actor_critic import (ActorCriticPolicy,  # noqa: E402
                                                TrainingConfig, train)
from qnet_autopilot.agents.dual_head import DualHeadPolicy  # noqa: E402
from qnet_autopilot.agents.baselines import ThresholdPolicy  # noqa: E402
from qnet_autopilot.agents.rollout import evaluate  # noqa: E402

#: Held-out seeds for the final score, disjoint from training and from the
#: validation seeds the checkpoint is selected on.
TEST_EPISODES = 400
TEST_SEED = 5_000_000


def baseline_config(**overrides) -> TrainingConfig:
    parameters = dict(iterations=300, hidden=(64, 64), seed=0)
    parameters.update(overrides)
    return TrainingConfig(**parameters)


def capacity_config(**overrides) -> TrainingConfig:
    """More capacity, more updates, and a step size that decays with them.

    Three changes at once, on purpose: they are the standard package for "train
    it harder", none of them is interesting on its own, and separating them
    would triple the compute to answer a question nobody asked.
    """
    parameters = dict(iterations=600, hidden=(128, 128), learning_rate=3e-3,
                      learning_rate_end=1e-4, batch_episodes=24, seed=0)
    parameters.update(overrides)
    return TrainingConfig(**parameters)


VARIANTS = {
    "baseline": dict(
        config=baseline_config,
        build=lambda cfg, seed: ActorCriticPolicy(hidden=cfg.hidden, rng=seed,
                                                  normalise=False),
        label="linha de base (fase 2)"),
    "normalised": dict(
        config=baseline_config,
        build=lambda cfg, seed: ActorCriticPolicy(hidden=cfg.hidden, rng=seed,
                                                  normalise=True),
        label="+ normalização"),
    "capacity": dict(
        config=capacity_config,
        build=lambda cfg, seed: ActorCriticPolicy(hidden=cfg.hidden, rng=seed,
                                                  normalise=True),
        label="+ capacidade e treino longo"),
    "dual_head": dict(
        config=capacity_config,
        build=lambda cfg, seed: DualHeadPolicy(trunk=(128,), head=(128,),
                                               embedding=128, rng=seed,
                                               normalise=True),
        label="dual-head (trunk compartilhado)"),
}


def run_variant(name: str, seed: int, quiet: bool = False):
    """Train one variant at one seed and score it on held-out episodes."""
    spec = VARIANTS[name]
    config = spec["config"](seed=seed)
    policy = spec["build"](config, seed)

    started = time.time()
    policy, history = train(make_env, make_request, config, policy=policy)
    policy.freeze_normalisers()
    elapsed = time.time() - started

    # Scored the way phase 2 scores: sampling at the training temperature, on
    # seeds disjoint from everything training touched.
    result = evaluate(make_env, [policy.view(temperature=1.0)],
                      episodes=TEST_EPISODES, request_factory=make_request,
                      seed=TEST_SEED)
    summary = result.summary().iloc[0]

    parameters = sum(array.size for group in policy.parameter_groups().values()
                     for array in group)
    row = {
        "variant": name, "label": spec["label"], "seed": seed,
        "success_rate": float(summary["success_rate"]),
        "success_stderr": float(summary["success_stderr"]),
        "raw_return": float(summary["raw_return"]),
        "mean_fidelity": float(summary["mean_fidelity"]),
        "purifications": float(summary["purifications"]),
        "best_validation": float(max(history.eval_success)),
        "final_validation": float(history.eval_success[-1]),
        "parameters": int(parameters),
        "iterations": int(config.iterations),
        "seconds": round(elapsed, 1),
    }
    if not quiet:
        print(f"  {name:12s} seed {seed}  teste {row['success_rate']:.3f}  "
              f"val {row['best_validation']:.3f}  {parameters:>6d} params  "
              f"{elapsed:.0f}s")
    return row, history


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("runs/ablation"))
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS))
    parser.add_argument("--seed-offset", type=int, default=0)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    seeds = [args.seed_offset + index for index in range(args.seeds)]
    rows, histories = [], {}

    for name in args.variants:
        print(f"\n=== {name} ===")
        for seed in seeds:
            row, history = run_variant(name, seed)
            rows.append(row)
            histories[f"{name}_{seed}"] = {
                "iteration": history.iteration,
                "train_success": history.train_success,
                "eval_iteration": history.eval_iteration,
                "eval_success": history.eval_success,
            }
            pd.DataFrame(rows).to_csv(args.out / "agent_ablation.csv", index=False)

    frame = pd.DataFrame(rows)
    (args.out / "agent_histories.json").write_text(json.dumps(histories))

    print("\n=== summary over seeds ===")
    summary = frame.groupby("variant", sort=False).agg(
        label=("label", "first"),
        mean=("success_rate", "mean"),
        std=("success_rate", "std"),
        best=("success_rate", "max"),
        worst=("success_rate", "min"),
        parameters=("parameters", "first"),
        seconds=("seconds", "mean"))
    summary["stderr"] = summary["std"] / np.sqrt(len(seeds))
    print(summary.round(4).to_string())

    print("\n=== paired differences (same seeds) ===")
    pivot = frame.pivot(index="seed", columns="variant", values="success_rate")
    pairs = [("normalised", "baseline"), ("capacity", "normalised"),
             ("dual_head", "capacity"), ("dual_head", "baseline")]
    paired = []
    for later, earlier in pairs:
        if later not in pivot or earlier not in pivot:
            continue
        difference = (pivot[later] - pivot[earlier]).dropna()
        paired.append({
            "change": f"{later} - {earlier}",
            "mean": float(difference.mean()),
            "stderr": float(difference.std(ddof=1) / np.sqrt(len(difference)))
            if len(difference) > 1 else 0.0,
            "wins": int((difference > 0).sum()),
            "losses": int((difference < 0).sum()),
            "seeds": int(len(difference)),
        })
    paired_frame = pd.DataFrame(paired)
    paired_frame.to_csv(args.out / "agent_paired.csv", index=False)
    print(paired_frame.round(4).to_string(index=False))

    print(f"\nwrote CSVs to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
