#!/usr/bin/env python3
"""Measure the three gaps the phase reports left open, and how much they cost.

The applicability review named four things standing between these results and a
deployment claim. Three of them are answerable here, and this script answers
them with numbers rather than with prose:

1. **The twin was never confronted in closed loop.** The physics was validated
   operation by operation; the environment the agent trains in was not. Here a
   whole trajectory is run through the environment and rebuilt independently out
   of SeQUeNCe objects, and the divergence is reported step by step.
2. **Key rates were asymptotic.** Every key figure in phases 3 and 4 assumed
   infinite blocks. This re-scores both against the composable finite-key bound
   at realistic block durations, so the published numbers can be corrected.
3. **The actuator was ideal.** ``scripts/evaluate_actuator.py`` covers that one;
   its CSVs are read by the same report.

The fourth — no real fibre data — cannot be closed in this container and is not
attempted. It is reported as open.

    python scripts/evaluate_realism.py --out runs/realism
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from qnet_autopilot.compat import bootstrap  # noqa: E402

BOOTSTRAP_STATUS = bootstrap()

from task3 import TEST_SEEDS as COMPENSATOR_SEEDS  # noqa: E402
from task3 import make_control, make_trajectories  # noqa: E402
from task4 import FORECAST_HORIZON  # noqa: E402
from task4 import TEST_SEEDS as ALLOCATOR_SEEDS  # noqa: E402
from task4 import make_env  # noqa: E402
from qnet_autopilot.compensation import (AlphaBetaFilter, Persistence,  # noqa: E402
                                         run_closed_loop)
from qnet_autopilot.allocation import (PersistenceForecaster,  # noqa: E402
                                       PredictiveAllocator, ReactiveThreshold,
                                       StaticAdmission, run_episode)
from qnet_autopilot.env.actions import Action, ActionKind  # noqa: E402
from qnet_autopilot.env.network_env import EnvironmentConfig, QuantumNetworkEnv  # noqa: E402
from qnet_autopilot.env.state import Request  # noqa: E402
from qnet_autopilot.physics.blocks import (adaptive_blocks,  # noqa: E402
                                            fixed_blocks, optimal_blocks,
                                            pooled_optimal_blocks)
from qnet_autopilot.validation.twin import SequenceShadow  # noqa: E402

#: Block durations for the finite-key comparison. A tenth of a second is what a
#: fast controller settles up on, ten seconds is what a patient one waits for,
#: and the point of the sweep is that the answer moves a long way between them.
BLOCK_DURATIONS_S = (0.05, 0.1, 0.5, 1.0, 5.0, 10.0)

#: Timer durations the blocking study puts against the controller. Longer than
#: the sweep above, and deliberately including the whole run: the honest
#: baseline for "a fixed block was fine all along" is the best fixed block, not
#: the one second phases 3 and 4 happened to assume.
BLOCK_BLOCKING_S = (0.1, 1.0, 5.0, 10.0, 20.0)


# ---------------------------------------------------------------------------
# 1. the twin, confronted
# ---------------------------------------------------------------------------

def _chain(*lengths_km) -> nx.Graph:
    graph = nx.Graph()
    names = [chr(ord("a") + index) for index in range(len(lengths_km) + 1)]
    for index, length in enumerate(lengths_km):
        graph.add_edge(names[index], names[index + 1], length=float(length))
    return graph


def scripted_trajectory(env, links, purify: bool = True):
    """Generate, optionally purify, swap along the chain, deliver.

    A fixed script rather than the trained policy: the question is whether the
    environment's arithmetic is SeQUeNCe's, and a script covers every operation
    type in a known order, where a policy would cover whatever it happens to
    like. The agent's own action distribution is a separate question, already
    answered in the phase-2 report.
    """
    shadow = SequenceShadow(env)

    for link in links:
        for _ in range(2 if purify else 1):
            for _ in range(80):
                action = next(candidate for candidate in env.legal_actions()
                              if candidate.kind is ActionKind.GENERATE
                              and candidate.link == link)
                shadow.step(action)
                if len(env.state.pairs_on(*link)) >= (2 if purify else 1):
                    break
        if not purify:
            continue
        candidates = [action for action in env.legal_actions()
                      if action.kind is ActionKind.PURIFY
                      and set(env.state.pairs[action.pair_ids[0]].nodes) == set(link)]
        if candidates:
            shadow.step(candidates[0])

    while True:
        swap = next((action for action in env.legal_actions()
                     if action.kind is ActionKind.SWAP), None)
        if swap is None:
            break
        shadow.step(swap)

    deliver = next((action for action in env.legal_actions()
                    if action.kind is ActionKind.DELIVER), None)
    if deliver is not None:
        shadow.step(deliver)
    else:
        shadow.step(Action(ActionKind.WAIT))
    return shadow


def study_twin(seeds) -> pd.DataFrame:
    rows = []
    topologies = {
        "cadeia 2 saltos": (_chain(20.0, 20.0), "a", "c"),
        "cadeia 4 saltos": (_chain(15.0, 25.0, 15.0, 25.0), "a", "e"),
    }

    for label, (graph, source, destination) in topologies.items():
        links = [tuple(sorted(edge)) for edge in graph.edges()]
        for purify in (False, True):
            for seed in seeds:
                config = EnvironmentConfig(graph=graph, gate_fid=0.97,
                                           meas_fid=0.98, purification="dejmps",
                                           swap_success_prob=1.0,
                                           max_pairs_per_link=3)
                env = QuantumNetworkEnv(config, rng=int(seed))
                env.reset(request=Request(source=source, destination=destination,
                                          min_fidelity=0.5, deadline_s=10.0),
                          memory_size=8)
                shadow = scripted_trajectory(env, links, purify=purify)

                for index, step in enumerate(shadow.steps):
                    rows.append({
                        "topology": label,
                        "purify": purify,
                        "seed": int(seed),
                        "step": index,
                        "action": str(step.action).split("(")[0],
                        "outcome": step.outcome,
                        "elapsed_s": step.elapsed_s,
                        "live_pairs": step.live_pairs,
                        "state_divergence": step.state_divergence,
                        "probability_divergence": step.probability_divergence,
                    })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2. finite blocks, on the link and in the network
# ---------------------------------------------------------------------------

def study_finite_key_link(fitted, seeds) -> pd.DataFrame:
    """What the phase-3 compensators are worth once the block is finite."""
    control = make_control()
    trajectories = make_trajectories(seeds)
    compensators = [Persistence(),
                    AlphaBetaFilter(alpha=0.05, beta=0.005, name="filtro alfa-beta"),
                    fitted["ridge"], fitted["mlp"]]

    rows = []
    for compensator in compensators:
        results = [run_closed_loop(trajectory, compensator, control, rng=7000 + index)
                   for index, trajectory in enumerate(trajectories)]
        asymptotic = float(np.mean([result.summary()["mean_key_rate_hz"]
                                    for result in results]))
        for duration in BLOCK_DURATIONS_S:
            finite = float(np.mean([
                result.finite_key_rate_hz(control.sifted_rate_hz, duration)
                for result in results]))
            rows.append({
                "compensator": compensator.name,
                "block_duration_s": duration,
                "block_bits": control.sifted_rate_hz * duration,
                "asymptotic_key_rate_hz": asymptotic,
                "finite_key_rate_hz": finite,
                "ratio": finite / asymptotic if asymptotic > 0 else np.nan,
            })
    return pd.DataFrame(rows)


def study_adaptive_blocks(fitted, seeds) -> pd.DataFrame:
    """Blocks that are not of fixed duration: a controller, a timer, an oracle.

    Every finite-key figure in phases 3 and 4 assumed the receiver settles up
    on a timer. It does not have to. The block boundary is the receiver's to
    choose, and the choice matters for a reason specific to the finite-key
    bound: errors are charged against the whole block, so a stretch of bad
    fibre poisons every bit that shares a block with it.

    Three strategies on the same runs, in
    :mod:`qnet_autopilot.physics.blocks`: the timer at several durations, the
    causal controller, and the dynamic-programming oracle that chooses the
    partition with perfect hindsight. The oracle is the ceiling — no causal
    rule can beat it — so the controller is judged as a fraction of it rather
    than against the timer it is bound to beat.

    Returns:
        One row per compensator, seed and strategy: the key rate over the whole
        run, how many blocks were cut, how much of the run sat inside one, and
        how many of them yielded nothing.
    """
    control = make_control()
    trajectories = make_trajectories(seeds)
    compensators = [Persistence(),
                    AlphaBetaFilter(alpha=0.05, beta=0.005, name="filtro alfa-beta"),
                    fitted["ridge"], fitted["mlp"]]
    rate = control.sifted_rate_hz

    rows = []
    for compensator in compensators:
        for index, trajectory in enumerate(trajectories):
            result = run_closed_loop(trajectory, compensator, control,
                                     rng=7000 + index)
            dt = float(result.time_s[1] - result.time_s[0])
            run_s = float(result.time_s.size * dt)
            plans = [fixed_blocks(result.qber, rate, dt, duration)
                     for duration in BLOCK_BLOCKING_S]
            # The controller is allowed the whole run as one block, so nothing
            # it loses to the longest timer can be blamed on a shorter budget.
            plans.append(adaptive_blocks(result.qber, rate, dt,
                                         max_block_s=run_s, rng=9000 + index))
            plans.append(adaptive_blocks(result.qber, rate, dt,
                                         max_block_s=run_s, rng=9000 + index,
                                         pooling=True))
            plans.append(optimal_blocks(result.qber, rate, dt))
            plans.append(pooled_optimal_blocks(result.qber, rate, dt))
            for plan in plans:
                rows.append({
                    "compensator": compensator.name,
                    "seed": int(seeds[index]),
                    "strategy": plan.strategy,
                    "mean_qber": float(np.mean(result.qber)),
                    "key_rate_hz": plan.mean_key_rate_hz,
                    "blocks": plan.count,
                    "duty": plan.duty,
                    "empty_fraction": plan.empty_fraction,
                    "pieces_per_block": plan.pieces / plan.count
                    if plan.count else 0.0,
                    "mean_block_s": float(np.mean(plan.durations_s))
                    if plan.count else 0.0,
                })
    return pd.DataFrame(rows)


#: Vibration amplitudes the blocking stress sweep walks through. The phase-3
#: operating point is the first; the rest are the disturbance a good
#: compensator starts to lose to, and are where the question "does variable
#: blocking matter for the *good* compensator too" gets answered.
BLOCKING_STRESS_RAD = (0.3, 0.6, 0.9, 1.2)


def study_blocking_stress(fitted, seeds,
                          amplitudes=BLOCKING_STRESS_RAD) -> pd.DataFrame:
    """When the link gets hard, does the block boundary start to matter?

    The controller's advantage on a badly compensated link is easy to dismiss:
    nobody deploys the persistence baseline. So the same comparison is run on
    the *best* compensator with the disturbance turned up, which is the regime
    the amplitude sweep already showed it degrading in. If the advantage
    appears there too, it is a property of hard links rather than of weak
    compensators.

    Returns:
        Per amplitude: the best timer with hindsight, the controller, and the
        oracle.
    """
    control = make_control()
    rate = control.sifted_rate_hz
    compensator = fitted["mlp"]

    rows = []
    for amplitude in amplitudes:
        trajectories = make_trajectories(seeds, vibration_amplitude_rad=amplitude)
        for index, trajectory in enumerate(trajectories):
            result = run_closed_loop(trajectory, compensator, control,
                                     rng=7000 + index)
            dt = float(result.time_s[1] - result.time_s[0])
            run_s = float(result.time_s.size * dt)
            timers = {duration: fixed_blocks(result.qber, rate, dt,
                                             duration).mean_key_rate_hz
                      for duration in BLOCK_BLOCKING_S}
            best = max(timers, key=timers.get)
            controller = adaptive_blocks(result.qber, rate, dt,
                                         max_block_s=run_s, rng=9000 + index)
            pooling = adaptive_blocks(result.qber, rate, dt, max_block_s=run_s,
                                      rng=9000 + index, pooling=True)
            oracle = optimal_blocks(result.qber, rate, dt)
            pooled_oracle = pooled_optimal_blocks(result.qber, rate, dt)
            rows.append({
                "vibration_amplitude_rad": amplitude,
                "seed": int(seeds[index]),
                "mean_qber": float(np.mean(result.qber)),
                "best_timer_s": best,
                "best_timer_hz": timers[best],
                "timer_1s_hz": timers[1.0],
                "adaptive_hz": controller.mean_key_rate_hz,
                "pooling_hz": pooling.mean_key_rate_hz,
                "oracle_hz": oracle.mean_key_rate_hz,
                "pooled_oracle_hz": pooled_oracle.mean_key_rate_hz,
                "adaptive_duty": controller.duty,
                "pooling_duty": pooling.duty,
                "pooling_pieces_per_block": pooling.pieces / pooling.count
                if pooling.count else 0.0,
            })
    return pd.DataFrame(rows)


def summarise_blocking_stress(frame: pd.DataFrame) -> pd.DataFrame:
    """Averaged over seeds, with the two ratios that carry the finding."""
    summary = frame.groupby("vibration_amplitude_rad").agg(
        mean_qber=("mean_qber", "mean"),
        median_best_timer_s=("best_timer_s", "median"),
        best_timer_hz=("best_timer_hz", "mean"),
        timer_1s_hz=("timer_1s_hz", "mean"),
        adaptive_hz=("adaptive_hz", "mean"),
        pooling_hz=("pooling_hz", "mean"),
        oracle_hz=("oracle_hz", "mean"),
        pooled_oracle_hz=("pooled_oracle_hz", "mean"),
        adaptive_duty=("adaptive_duty", "mean"),
        pooling_duty=("pooling_duty", "mean"),
        pieces_per_block=("pooling_pieces_per_block", "mean")).reset_index()
    summary["over_best_timer"] = np.where(
        summary["best_timer_hz"] > 0,
        summary["pooling_hz"] / summary["best_timer_hz"], np.nan)
    summary["pooling_gain"] = np.where(
        summary["adaptive_hz"] > 0,
        summary["pooling_hz"] / summary["adaptive_hz"], np.nan)
    summary["of_oracle"] = np.where(
        summary["pooled_oracle_hz"] > 0,
        summary["pooling_hz"] / summary["pooled_oracle_hz"], np.nan)
    return summary


def summarise_adaptive_blocks(frame: pd.DataFrame) -> pd.DataFrame:
    """Per compensator and strategy, averaged over seeds, against the oracle."""
    summary = frame.groupby(["compensator", "strategy"], sort=False).agg(
        mean_qber=("mean_qber", "mean"),
        key_rate_hz=("key_rate_hz", "mean"),
        blocks=("blocks", "mean"),
        duty=("duty", "mean"),
        empty_fraction=("empty_fraction", "mean"),
        pieces_per_block=("pieces_per_block", "mean"),
        mean_block_s=("mean_block_s", "mean")).reset_index()
    oracle = summary[summary["strategy"] == "oráculo agrupado"].set_index(
        "compensator")
    summary["of_oracle"] = [
        row.key_rate_hz / oracle.loc[row.compensator, "key_rate_hz"]
        if oracle.loc[row.compensator, "key_rate_hz"] > 0 else np.nan
        for row in summary.itertuples()]
    return summary


def study_finite_key_allocation(seeds) -> pd.DataFrame:
    """Whether charging the finite-key cost reorders the phase-4 policies."""
    policies = [StaticAdmission(0.0), StaticAdmission(0.5), StaticAdmission(1.0),
                ReactiveThreshold(low=0.02, high=0.04),
                PredictiveAllocator(
                    forecaster=PersistenceForecaster(horizon=FORECAST_HORIZON),
                    coverage_target=0.5)]

    rows = []
    for finite in (False, True):
        env = make_env(seed=0, finite_key=finite)
        for policy in policies:
            episodes = [run_episode(env, policy, seed=int(seed)).as_row()
                        for seed in seeds]
            frame = pd.DataFrame(episodes)
            rows.append({
                "policy": frame["policy"].iloc[0],
                "finite_key": finite,
                "served_qkd": float(frame["served_qkd"].mean()),
                "served_classical": float(frame["served_classical"].mean()),
                "qkd_stderr": float(frame["served_qkd"].std(ddof=1)
                                    / np.sqrt(len(frame))),
                "mean_admission": float(frame["mean_admission"].mean()),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------

def study_finite_key_frontier(seeds) -> pd.DataFrame:
    """Re-trace the phase-4 frontiers with the finite-block cost charged.

    The phase-4 headline — the same key served with 2.4x the classical
    throughput — is a *matched-service* comparison between two frontiers, so
    knowing that finite blocks lower every policy's key service says nothing
    about whether the headline survives. Only re-tracing both frontiers does,
    and that is what this is.
    """
    env = make_env(seed=0, finite_key=True)
    rows = []

    def point(family, knob, policy):
        episodes = [run_episode(env, policy, seed=int(seed)).as_row()
                    for seed in seeds]
        frame = pd.DataFrame(episodes)
        rows.append({
            "family": family, "knob": knob,
            "policy": frame["policy"].iloc[0],
            "served_qkd": float(frame["served_qkd"].mean()),
            "served_classical": float(frame["served_classical"].mean()),
            "qkd_stderr": float(frame["served_qkd"].std(ddof=1)
                                / np.sqrt(len(frame))),
            "mean_admission": float(frame["mean_admission"].mean()),
        })

    for level in (0.0, 0.25, 0.5, 0.75, 1.0):
        point("admissão estática", level, StaticAdmission(level))
    for low, high in ((0.01, 0.02), (0.02, 0.04), (0.035, 0.06),
                      (0.05, 0.08), (0.07, 0.10)):
        point("limiar reativo", high, ReactiveThreshold(low=low, high=high))
    for coverage in (0.05, 0.25, 0.5, 1.0, 2.0, 5.0):
        point("preditivo", coverage, PredictiveAllocator(
            forecaster=PersistenceForecaster(horizon=FORECAST_HORIZON),
            coverage_target=coverage))

    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("runs/realism"))
    parser.add_argument("--models", type=Path,
                        default=Path("runs/compensator/main.pkl"))
    parser.add_argument("--twin-seeds", type=int, default=4)
    parser.add_argument("--link-seeds", type=int, default=6)
    parser.add_argument("--allocation-seeds", type=int, default=8)
    parser.add_argument("--frontier-seeds", type=int, default=6)
    parser.add_argument("--stress-seeds", type=int, default=6)
    parser.add_argument("--only", nargs="+",
                        default=["twin", "link", "allocation", "frontier",
                                 "adaptive"],
                        choices=["twin", "link", "allocation", "frontier",
                                 "adaptive"])
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    if "twin" in args.only:
        print("=== the environment against SeQUeNCe, step by step ===")
        twin = study_twin(range(args.twin_seeds))
        twin.to_csv(args.out / "twin.csv", index=False)
        summary = twin.groupby(["topology", "purify"]).agg(
            steps=("state_divergence", "size"),
            worst=("state_divergence", "max"),
            worst_probability=("probability_divergence", "max"))
        print(summary.to_string(float_format=lambda value: f"{value:.3e}"))
        print(f"  {len(twin)} steps compared, worst divergence "
              f"{twin['state_divergence'].max():.3e}")

    if "link" in args.only:
        print("\n=== finite blocks on the compensated link ===")
        with open(args.models, "rb") as handle:
            fitted = pickle.load(handle)
        link = study_finite_key_link(fitted, COMPENSATOR_SEEDS[:args.link_seeds])
        link.to_csv(args.out / "finite_key_link.csv", index=False)
        print(link.pivot(index="block_duration_s", columns="compensator",
                         values="ratio").round(3).to_string())

    if "adaptive" in args.only:
        print("\n=== blocks of variable duration: timer, controller, oracle ===")
        with open(args.models, "rb") as handle:
            fitted = pickle.load(handle)
        adaptive = study_adaptive_blocks(fitted, COMPENSATOR_SEEDS[:args.link_seeds])
        adaptive.to_csv(args.out / "adaptive_blocks.csv", index=False)
        summary = summarise_adaptive_blocks(adaptive)
        summary.to_csv(args.out / "adaptive_blocks_summary.csv", index=False)
        print(summary.round(3).to_string(index=False))

        print("\n=== and on the best compensator, as the link gets harder ===")
        stress = study_blocking_stress(fitted,
                                       COMPENSATOR_SEEDS[:args.stress_seeds])
        stress.to_csv(args.out / "blocking_stress.csv", index=False)
        stress_summary = summarise_blocking_stress(stress)
        stress_summary.to_csv(args.out / "blocking_stress_summary.csv",
                              index=False)
        print(stress_summary.round(3).to_string(index=False))

    if "allocation" in args.only:
        print("\n=== finite blocks in the allocation problem ===")
        allocation = study_finite_key_allocation(
            ALLOCATOR_SEEDS[:args.allocation_seeds])
        allocation.to_csv(args.out / "finite_key_allocation.csv", index=False)
        print(allocation.pivot(index="policy", columns="finite_key",
                               values="served_qkd").round(4).to_string())

    if "frontier" in args.only:
        print("\n=== the phase-4 frontiers, re-traced with finite blocks ===")
        frontier = study_finite_key_frontier(
            ALLOCATOR_SEEDS[:args.frontier_seeds])
        frontier.to_csv(args.out / "finite_key_frontier.csv", index=False)
        print(frontier[["family", "knob", "served_qkd", "served_classical",
                        "mean_admission"]].round(4).to_string(index=False))

    print(f"\nwrote CSVs to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
