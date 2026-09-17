#!/usr/bin/env python3
"""Three questions the headline number does not answer.

1. **Why sample rather than greedify?**  A temperature sweep: if performance
   peaks at the temperature the policy was trained at, the explanation is that
   A2C optimised a stochastic policy, not that greedy is broken.
2. **Does it transfer?**  The policy scores actions from their own features and
   never sees a node name, so in principle it runs on a topology it never
   trained on. This measures whether it actually does.
3. **What did it learn?**  When the agent purifies and the threshold heuristic
   does not, and the reverse.
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

from task import MEMORY_SIZE, make_config, make_env, make_request  # noqa: E402
from qnet_autopilot.agents import ShortestPathPolicy, ThresholdPolicy, evaluate  # noqa: E402
from qnet_autopilot.agents.actor_critic import ActorCriticPolicy  # noqa: E402
from qnet_autopilot.agents.rollout import run_episode  # noqa: E402
from qnet_autopilot.env import EnvironmentConfig, QuantumNetworkEnv, Request  # noqa: E402
from qnet_autopilot.env.actions import ActionKind  # noqa: E402
from qnet_autopilot.env.features import ActionFeature, StateFeature  # noqa: E402
from qnet_autopilot.network import scenario  # noqa: E402
from qnet_autopilot.physics.memory import MemoryParameters  # noqa: E402

TEST_SEED = 500_000
#: The seeds training used for checkpoint selection. Choosing anything —
#: including the deployment temperature — on the test seeds would make the
#: headline number a maximum over choices rather than an estimate.
VALIDATION_SEED = 900_000


def load_agent(weights: Path, temperature: float = 1.0, rng: int = 0,
               name: str = "actor_critic") -> ActorCriticPolicy:
    agent = ActorCriticPolicy(rng=rng, name=name)
    agent.load(weights)
    agent.temperature = temperature
    agent.greedy = temperature <= 0
    return agent


def temperature_sweep(weights: Path, episodes: int = 300,
                      seed: int = VALIDATION_SEED) -> pd.DataFrame:
    """Success against softmax temperature.

    Run on the **validation** seeds by default, because its purpose is to
    choose the deployment temperature; running it on the test seeds and then
    quoting the best cell would be selecting on the test set.
    """
    temperatures = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
    policies = [load_agent(weights, temperature=t, rng=11,
                           name=f"T={t:g}") for t in temperatures]
    result = evaluate(lambda: make_env(0.0), policies, episodes=episodes,
                      request_factory=make_request, memory_size=MEMORY_SIZE,
                      seed=seed)
    summary = result.summary()
    summary.insert(1, "temperature", temperatures)
    return summary


def transfer(weights: Path, episodes: int = 250) -> pd.DataFrame:
    """Run the trained policy on topologies it never saw."""
    topologies = {
        "chain-4 (trained on)": (scenario("chain", num_nodes=4, span_km=20), "0", "3"),
        "chain-5 (deeper)": (scenario("chain", num_nodes=5, span_km=20), "0", "4"),
        "chain-3 (shallower)": (scenario("chain", num_nodes=3, span_km=20), "0", "2"),
        "chain-4, 30 km spans": (scenario("chain", num_nodes=4, span_km=30), "0", "3"),
        "ring-6": (scenario("ring", num_nodes=6, span_km=20), "0", "3"),
    }

    rows = []
    for label, (graph, source, destination) in topologies.items():
        base = make_config()
        config = EnvironmentConfig(
            graph=graph, memory=base.memory, gate_fid=base.gate_fid,
            meas_fid=base.meas_fid, coupling_efficiency=base.coupling_efficiency,
            detector_efficiency=base.detector_efficiency,
            generate_budget_s=base.generate_budget_s, wait_slot_s=base.wait_slot_s,
            max_steps=base.max_steps, max_pairs_per_link=base.max_pairs_per_link)

        def request_factory(rng, s=source, d=destination):
            return Request(source=s, destination=d,
                           min_fidelity=float(rng.choice([0.70, 0.74, 0.78])),
                           deadline_s=float(rng.uniform(0.040, 0.090)))

        policies = [ShortestPathPolicy(),
                    ThresholdPolicy(name="threshold_0.90", target_fidelity=0.90),
                    load_agent(weights, temperature=1.0, rng=3)]
        result = evaluate(lambda c=config: QuantumNetworkEnv(c, rng=0), policies,
                          episodes=episodes, request_factory=request_factory,
                          memory_size=MEMORY_SIZE, seed=TEST_SEED)
        summary = result.summary().set_index("policy")
        rows.append({
            "topology": label,
            "hops": len(graph) and _hops(graph, source, destination),
            "shortest_path": summary.loc["shortest_path", "success_rate"],
            "threshold_0.90": summary.loc["threshold_0.90", "success_rate"],
            "actor_critic": summary.loc["actor_critic", "success_rate"],
            "agent_stderr": summary.loc["actor_critic", "success_stderr"],
        })
    return pd.DataFrame(rows)


def _hops(graph, source, destination) -> int:
    import networkx as nx

    relabelled = nx.relabel_nodes(graph, {n: str(n) for n in graph.nodes()}, copy=True)
    return len(nx.shortest_path(relabelled, source, destination, weight="length")) - 1


def purification_behaviour(weights: Path, episodes: int = 200) -> pd.DataFrame:
    """When does each policy purify? Recorded at the moment of the decision."""
    records = []
    for policy in (load_agent(weights, temperature=1.0, rng=5),
                   ThresholdPolicy(name="threshold_0.90", target_fidelity=0.90)):
        env = make_env(0.0)
        for index in range(episodes):
            seed = TEST_SEED + index
            request = make_request(np.random.default_rng(seed))
            observation, _ = env.reset(request, memory_size=MEMORY_SIZE, rng=seed)
            policy.reset()
            while True:
                choice = policy.act(observation)
                state_features, action_features, actions = observation
                action = actions[choice]
                if action.kind is ActionKind.PURIFY:
                    records.append({
                        "policy": policy.name,
                        "time_remaining": state_features[StateFeature.TIME_REMAINING],
                        "input_fidelity": max(action_features[choice, ActionFeature.INPUT_FIDELITY_A],
                                              action_features[choice, ActionFeature.INPUT_FIDELITY_B]),
                        "predicted_gain": action_features[choice, ActionFeature.FIDELITY_GAIN],
                        "success_prob": action_features[choice, ActionFeature.SUCCESS_PROB],
                        "result_hops": action_features[choice, ActionFeature.RESULT_HOPS],
                        "min_fidelity": request.min_fidelity,
                        "deadline_ms": request.deadline_s * 1e3,
                    })
                observation, result = env.step(action)
                if result.done:
                    break
    return pd.DataFrame(records)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--policy", type=Path, default=Path("runs/agent/policy.npz"))
    parser.add_argument("--out", type=Path, default=Path("runs/agent"))
    parser.add_argument("--episodes", type=int, default=250)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print("=== temperature sweep (validation seeds) ===")
    sweep = temperature_sweep(args.policy, episodes=args.episodes)
    print(sweep[["temperature", "success_rate", "success_stderr", "raw_return",
                 "mean_time_ms", "purifications"]].round(4).to_string(index=False))
    sweep.to_csv(args.out / "temperature_sweep.csv", index=False)
    chosen = float(sweep.loc[sweep["success_rate"].idxmax(), "temperature"])
    (args.out / "temperature.json").write_text(json.dumps({"temperature": chosen}))
    print(f"\nchosen deployment temperature (argmax on validation): {chosen:g}")

    print("\n=== transfer to unseen topologies ===")
    table = transfer(args.policy, episodes=args.episodes)
    print(table.round(4).to_string(index=False))
    table.to_csv(args.out / "transfer.csv", index=False)

    print("\n=== purification decisions ===")
    decisions = purification_behaviour(args.policy, episodes=args.episodes)
    decisions.to_csv(args.out / "purification_decisions.csv", index=False)
    print(decisions.groupby("policy").agg(
        rounds=("input_fidelity", "size"),
        mean_input_fidelity=("input_fidelity", "mean"),
        mean_predicted_gain=("predicted_gain", "mean"),
        mean_time_remaining=("time_remaining", "mean"),
        frac_on_elementary_links=("result_hops", lambda s: float((s <= 0.13).mean())),
    ).round(4).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
