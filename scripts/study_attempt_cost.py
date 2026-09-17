#!/usr/bin/env python3
"""Why the phase-2 agent collapsed at long deadlines, and what it was measuring.

The retraction in the realism report said the agent ties the heuristics at the
published deadline and loses to ``greedy_fidelity`` at four times it. That is
true and it is not the interesting part. Running the saved long-deadline policy
greedily shows what it actually learned: **nothing at all** — zero generation
attempts, zero swaps, wait until the clock runs out. Its score is the score of
doing nothing, to three decimals.

That is not a training failure, it is a correct answer to the question that was
asked. The environment charges a **flat** ``attempt_cost`` per GENERATE while
the time cost is charged as a *fraction of the deadline*. Quadruple the deadline
and the time cost stays bounded at ``time_cost`` while the attempt cost
quadruples with the number of attempts that now fit. At the published deadline
the best heuristic clears doing-nothing by 0.13; at four times it, by 0.04 — and
``shortest_path`` drops *below* doing nothing. The long-deadline benchmark was a
task with almost nothing in it, and the agent found the floor.

This script measures that, and then measures what happens to the ordering when
the per-attempt charge is made deadline-invariant — which is the one change that
restores the question the sweep was supposed to ask.

    python scripts/study_attempt_cost.py --out runs/attempt_cost

Note:
    Every policy is scored under whichever reward the row names, heuristics
    included. Changing a cost and comparing the new agent against the old
    baselines would manufacture an improvement out of bookkeeping.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from qnet_autopilot.compat import bootstrap  # noqa: E402

BOOTSTRAP_STATUS = bootstrap()

from task import ATTEMPT_COST, MEMORY_SIZE, make_config, make_request  # noqa: E402
from qnet_autopilot.agents.actor_critic import ActorCriticPolicy  # noqa: E402
from qnet_autopilot.agents.baselines import (DoNothingPolicy,  # noqa: E402
                                             GreedyFidelityPolicy,
                                             RandomPolicy, ShortestPathPolicy)
from qnet_autopilot.agents.rollout import run_episode  # noqa: E402
from qnet_autopilot.env.network_env import QuantumNetworkEnv  # noqa: E402
from qnet_autopilot.physics.constants import (  # noqa: E402
    MEASURED_HERALDING_ROUND_TRIPS)

#: Per-attempt charges the sweep walks through. The published value is the last
#: one; the deadline-invariant one at four times the deadline is the second.
ATTEMPT_COSTS = (0.0, 0.0025, 0.005, 0.01)

#: Deadline multipliers. One is the published benchmark; four is the control the
#: corrected heralding cost called for.
DEADLINE_SCALES = (1.0, 4.0)

#: Seeds held out from every training run in this package.
TEST_SEED_BASE = 510_000


def baselines() -> list:
    return [DoNothingPolicy(), RandomPolicy(rng=0), ShortestPathPolicy(),
            GreedyFidelityPolicy()]


def returns_per_episode(env, policy, deadline_scale: float,
                        episodes: int) -> np.ndarray:
    """One row per held-out request, so two policies can be paired on it.

    The request and the environment seed are both derived from the episode
    index, so every policy faces the same task in the same order. Comparing
    unpaired means throws away most of the resolution: the spread across
    requests is several times the spread between policies.
    """
    rows = []
    for index in range(episodes):
        request = make_request(np.random.default_rng(TEST_SEED_BASE + index),
                               deadline_scale=deadline_scale)
        episode = run_episode(env, policy, request=request,
                              memory_size=MEMORY_SIZE,
                              seed=TEST_SEED_BASE + index, collect=False)
        rows.append((episode.succeeded, episode.raw_return, episode.attempts,
                     episode.swaps, episode.discarded))
    return np.asarray(rows, dtype=float)


def score(env, policy, deadline_scale: float, episodes: int) -> dict:
    """Run one policy over the held-out requests and average what came out."""
    values = returns_per_episode(env, policy, deadline_scale, episodes)
    return {
        "policy": policy.name,
        "success": float(values[:, 0].mean()),
        "raw_return": float(values[:, 1].mean()),
        "return_stderr": float(values[:, 1].std(ddof=1) / np.sqrt(episodes)),
        "attempts": float(values[:, 2].mean()),
        "swaps": float(values[:, 3].mean()),
        "discarded": float(values[:, 4].mean()),
    }


def paired_against(env, policy, reference, deadline_scale: float,
                   episodes: int) -> dict:
    """Per-request difference between two policies, with its own error bar."""
    mine = returns_per_episode(env, policy, deadline_scale, episodes)[:, 1]
    theirs = returns_per_episode(env, reference, deadline_scale, episodes)[:, 1]
    difference = mine - theirs
    return {
        "against": reference.name,
        "delta": float(difference.mean()),
        "delta_stderr": float(difference.std(ddof=1) / np.sqrt(episodes)),
        "wins": int(np.sum(difference > 1e-9)),
        "losses": int(np.sum(difference < -1e-9)),
        "ties": int(np.sum(np.abs(difference) <= 1e-9)),
    }


def study_floor(episodes: int, policies=None) -> pd.DataFrame:
    """The heuristics against doing nothing, across cost and deadline.

    The column that carries the finding is ``over_doing_nothing``: how much of a
    delivery separates the best available policy from refusing to play. It is
    the size of the prize any learner is being asked to find.
    """
    rows = []
    for deadline_scale in DEADLINE_SCALES:
        for cost in ATTEMPT_COSTS:
            config = make_config(
                heralding_round_trips=MEASURED_HERALDING_ROUND_TRIPS,
                attempt_cost=cost)
            env = QuantumNetworkEnv(config, rng=0)
            for policy in (policies or baselines()):
                record = score(env, policy, deadline_scale, episodes)
                record.update(deadline_scale=deadline_scale, attempt_cost=cost)
                rows.append(record)
            print(f"  deadline x{deadline_scale:g}, attempt cost {cost:g}: done")
    return pd.DataFrame(rows)


def summarise_floor(frame: pd.DataFrame) -> pd.DataFrame:
    table = frame.pivot_table(index=["deadline_scale", "attempt_cost"],
                              columns="policy", values="raw_return")
    contenders = [name for name in table.columns
                  if name not in ("nao_fazer_nada", "random")]
    table["melhor_heuristica"] = table[contenders].max(axis=1)
    table["over_doing_nothing"] = (table["melhor_heuristica"]
                                   - table["nao_fazer_nada"])
    return table.reset_index()


def study_policies(paths, episodes: int, deadline_scale: float,
                   attempt_cost: float) -> pd.DataFrame:
    """Saved policies and the baselines, all under one reward.

    Each policy is run both greedily and at the temperature it was trained
    with. A policy that scores like ``random`` when sampled and like
    ``nao_fazer_nada`` when greedy has not learned to act; it has learned that
    acting does not pay, and only the greedy run shows which.
    """
    config = make_config(heralding_round_trips=MEASURED_HERALDING_ROUND_TRIPS,
                         attempt_cost=attempt_cost)
    env = QuantumNetworkEnv(config, rng=0)

    best = GreedyFidelityPolicy()
    floor = DoNothingPolicy()

    rows = []
    for policy in baselines():
        record = score(env, policy, deadline_scale, episodes)
        record.update(temperature=np.nan, source="—")
        if policy.name != best.name:
            record.update({f"vs_greedy_{key}": value for key, value
                           in paired_against(env, policy, best, deadline_scale,
                                             episodes).items()
                           if key != "against"})
        record.update({f"vs_floor_{key}": value for key, value
                       in paired_against(env, policy, floor, deadline_scale,
                                         episodes).items() if key != "against"})
        rows.append(record)

    for path in paths:
        path = Path(path)
        if not (path / "policy.npz").is_file():
            print(f"  ({path} has no policy.npz; skipped)")
            continue
        for temperature in (0.0, 1.0):
            agent = ActorCriticPolicy(rng=0, name=f"{path.name}")
            agent.load(path / "policy.npz")
            agent.temperature = temperature
            agent.greedy = temperature <= 0
            record = score(env, agent, deadline_scale, episodes)
            record.update(temperature=temperature, source=str(path))
            record.update({f"vs_greedy_{key}": value for key, value
                           in paired_against(env, agent, best, deadline_scale,
                                             episodes).items()
                           if key != "against"})
            record.update({f"vs_floor_{key}": value for key, value
                           in paired_against(env, agent, floor, deadline_scale,
                                             episodes).items()
                           if key != "against"})
            rows.append(record)
        print(f"  {path.name}: done")
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("runs/attempt_cost"))
    parser.add_argument("--episodes", type=int, default=160)
    parser.add_argument("--policies", type=Path, nargs="*", default=[])
    parser.add_argument("--deadline-scale", type=float, default=4.0)
    parser.add_argument("--attempt-cost", type=float, default=ATTEMPT_COST)
    parser.add_argument("--only", nargs="+", default=["floor", "policies"],
                        choices=["floor", "policies"])
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    if "floor" in args.only:
        print("=== how much is the task worth, against refusing to play? ===")
        floor = study_floor(args.episodes)
        floor.to_csv(args.out / "floor.csv", index=False)
        summary = summarise_floor(floor)
        summary.to_csv(args.out / "floor_summary.csv", index=False)
        print(summary.round(3).to_string(index=False))

    if "policies" in args.only and args.policies:
        print(f"\n=== saved policies at deadline x{args.deadline_scale:g}, "
              f"attempt cost {args.attempt_cost:g} ===")
        policies = study_policies(args.policies, args.episodes,
                                  args.deadline_scale, args.attempt_cost)
        policies.to_csv(args.out / "policies.csv", index=False)
        columns = ["policy", "temperature", "success", "raw_return",
                   "vs_greedy_delta", "vs_greedy_delta_stderr",
                   "vs_greedy_wins", "vs_greedy_losses",
                   "vs_floor_delta", "vs_floor_delta_stderr"]
        print(policies[[c for c in columns if c in policies]]
              .round(3).to_string(index=False))

    print(f"\nwrote CSVs to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
