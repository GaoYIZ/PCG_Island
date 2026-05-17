"""
Fast reward learnability probe for the island RL environment.

This script does not train SAC/PPO. It checks whether the current reward gives
a short-horizon optimizer a usable signal before spending hours on RL.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np

from reporting import save_json
from rl_environment import IslandGenerationEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe whether the RL reward has a learnable local signal.")
    parser.add_argument("--output", type=str, default="rl_reward_probe_summary.json")
    parser.add_argument("--map-size", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=3)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--candidate-actions", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sampling-profile",
        type=str,
        default="island_voronoi",
        choices=["uniform", "island", "island_voronoi"],
    )
    parser.add_argument("--action-step-scale", type=float, default=0.08)
    parser.add_argument("--reward-current-scale", type=float, default=0.0)
    parser.add_argument("--reward-delta-scale", type=float, default=5.0)
    parser.add_argument("--reward-best-scale", type=float, default=2.0)
    parser.add_argument("--reward-step-penalty", type=float, default=0.01)
    parser.add_argument("--reward-success-bonus", type=float, default=1.0)
    parser.add_argument("--failure-score-threshold", type=float, default=0.10)
    parser.add_argument("--success-score-threshold", type=float, default=0.72)
    parser.add_argument("--success-score-gain-threshold", type=float, default=0.03)
    parser.add_argument("--success-streak-required", type=int, default=3)
    parser.add_argument("--stagnation-patience", type=int, default=5)
    parser.add_argument("--stagnation-delta", type=float, default=1e-3)
    return parser.parse_args()


def make_env(args: argparse.Namespace) -> IslandGenerationEnv:
    return IslandGenerationEnv(
        map_size=args.map_size,
        max_steps=args.max_steps,
        include_latent=False,
        action_step_scale=args.action_step_scale,
        sampling_profile=args.sampling_profile,
        reward_current_scale=args.reward_current_scale,
        reward_delta_scale=args.reward_delta_scale,
        reward_best_scale=args.reward_best_scale,
        reward_step_penalty=args.reward_step_penalty,
        reward_success_bonus=args.reward_success_bonus,
        success_score_threshold=args.success_score_threshold,
        success_score_gain_threshold=args.success_score_gain_threshold,
        failure_score_threshold=args.failure_score_threshold,
        success_streak_required=args.success_streak_required,
        stagnation_patience=args.stagnation_patience,
        stagnation_delta=args.stagnation_delta,
    )


def candidate_reward_and_score(env: IslandGenerationEnv, action: Sequence[float]) -> Tuple[float, float]:
    """Mirror env.step reward for one candidate action without mutating the env."""
    if env.current_params is None or env.previous_score is None:
        raise RuntimeError("Environment must be reset before evaluating candidate actions.")

    candidate_params = env.param_normalizer.apply_normalized_delta(env.current_params, action)
    candidate_params["seed"] = env.current_seed
    heightmap = env.generator.generate_heightmap(candidate_params)
    metrics = env.evaluator.evaluate(heightmap)
    novelty_vector = env.feature_normalizer.transform_metrics(metrics)
    score = env.scorer.score_metrics(
        metrics,
        feature_vector=novelty_vector,
        history_vectors=env.novelty_reference_vectors,
    )

    previous_total = float(env.previous_score.total_score)
    current_total = float(score.total_score)
    delta_score = current_total - previous_total
    best_delta = max(0.0, current_total - float(env.best_total_score))

    reward = (
        env.reward_current_scale * current_total
        + env.reward_delta_scale * delta_score
        + env.reward_best_scale * best_delta
        - env.reward_step_penalty
    )
    score_gain_from_initial = current_total - float(env.initial_total_score)
    projected_success_streak = (
        env.success_streak + 1
        if (
            current_total >= env.success_score_threshold
            and score_gain_from_initial >= env.success_score_gain_threshold
        )
        else 0
    )
    if projected_success_streak >= env.success_streak_required:
        reward += env.reward_success_bonus
    return float(reward), current_total


def choose_action(
    policy_name: str,
    env: IslandGenerationEnv,
    rng: np.random.Generator,
    candidate_actions: int,
) -> np.ndarray:
    action_dim = env.action_space.shape[0]
    if policy_name == "zero":
        return np.zeros(action_dim, dtype=np.float32)
    if policy_name == "random":
        return rng.uniform(-1.0, 1.0, size=action_dim).astype(np.float32)

    candidates = [np.zeros(action_dim, dtype=np.float32)]
    for _ in range(max(0, candidate_actions - 1)):
        candidates.append(rng.uniform(-1.0, 1.0, size=action_dim).astype(np.float32))

    evaluated = [candidate_reward_and_score(env, action) for action in candidates]
    if policy_name == "greedy_reward":
        best_index = int(np.argmax([item[0] for item in evaluated]))
        return candidates[best_index]
    if policy_name == "greedy_score":
        best_index = int(np.argmax([item[1] for item in evaluated]))
        return candidates[best_index]
    raise ValueError(f"Unknown probe policy: {policy_name}")


def rollout_policy(
    policy_name: str,
    args: argparse.Namespace,
    episode_seed: int,
    rng: np.random.Generator,
) -> Dict[str, object]:
    env = make_env(args)
    _, reset_info = env.reset(seed=episode_seed)
    initial_score = reset_info["score"]
    total_reward = 0.0
    final_info = reset_info
    done_reason = "max_steps"
    steps = 0

    for _ in range(args.max_steps):
        action = choose_action(policy_name, env, rng, args.candidate_actions)
        _, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        final_info = info
        steps += 1
        if terminated or truncated:
            done_reason = str(info.get("done_reason", "done"))
            break

    final_score = final_info["score"]
    initial_total = float(initial_score["total_score"])
    final_total = float(final_score["total_score"])
    return {
        "policy": policy_name,
        "seed": int(episode_seed),
        "steps": int(steps),
        "reward": float(total_reward),
        "initial_total_score": initial_total,
        "final_total_score": final_total,
        "score_gain": float(final_total - initial_total),
        "final_path_score": float(final_score.get("path_score", 0.0)),
        "final_connectivity_score": float(final_score.get("connectivity_score", 0.0)),
        "done_reason": done_reason,
    }


def summarize_records(records: Sequence[Mapping[str, object]]) -> Dict[str, float]:
    def values(name: str) -> np.ndarray:
        return np.asarray([float(record[name]) for record in records], dtype=np.float32)

    gains = values("score_gain")
    done_reasons = [str(record["done_reason"]) for record in records]
    return {
        "episodes": float(len(records)),
        "reward_mean": float(values("reward").mean()),
        "reward_std": float(values("reward").std()),
        "initial_total_score_mean": float(values("initial_total_score").mean()),
        "final_total_score_mean": float(values("final_total_score").mean()),
        "score_gain_mean": float(gains.mean()),
        "score_gain_std": float(gains.std()),
        "positive_gain_rate": float(np.mean(gains > 0.0)),
        "final_path_score_mean": float(values("final_path_score").mean()),
        "final_connectivity_score_mean": float(values("final_connectivity_score").mean()),
        "success_rate": float(np.mean([reason == "success_threshold" for reason in done_reasons])),
        "stagnation_rate": float(np.mean([reason == "stagnation" for reason in done_reasons])),
        "steps_mean": float(values("steps").mean()),
    }


def run_noop_check(args: argparse.Namespace) -> Dict[str, object]:
    records: List[Dict[str, float]] = []
    for episode in range(args.episodes):
        env = make_env(args)
        _, reset_info = env.reset(seed=args.seed + episode)
        initial_params = dict(reset_info["params"])
        initial_total = float(reset_info["score"]["total_score"])
        zero_action = np.zeros(env.action_space.shape[0], dtype=np.float32)
        _, _, _, _, info = env.step(zero_action)

        final_params = info["params"]
        param_deltas = [
            abs(float(final_params[name]) - float(initial_params[name]))
            for name in env.param_normalizer.param_names
            if name in initial_params and name in final_params
        ]
        records.append(
            {
                "score_gain": float(info["score"]["total_score"] - initial_total),
                "max_param_abs_delta": float(max(param_deltas) if param_deltas else 0.0),
            }
        )

    score_gains = np.asarray([record["score_gain"] for record in records], dtype=np.float32)
    param_deltas = np.asarray([record["max_param_abs_delta"] for record in records], dtype=np.float32)
    return {
        "episodes": int(len(records)),
        "mean_abs_score_gain": float(np.abs(score_gains).mean()) if score_gains.size else 0.0,
        "max_abs_score_gain": float(np.abs(score_gains).max()) if score_gains.size else 0.0,
        "max_abs_param_delta": float(param_deltas.max()) if param_deltas.size else 0.0,
        "zero_action_is_noop": bool(
            (np.abs(score_gains).max() if score_gains.size else 0.0) < 1e-6
            and (param_deltas.max() if param_deltas.size else 0.0) < 1e-5
        ),
        "records": records,
    }


def build_diagnosis(
    policy_summary: Mapping[str, Mapping[str, float]],
    noop_summary: Mapping[str, object],
) -> Dict[str, object]:
    zero_gain = policy_summary["zero"]["score_gain_mean"]
    random_gain = policy_summary["random"]["score_gain_mean"]
    baseline_gain = max(zero_gain, random_gain)
    greedy_reward_gain = policy_summary["greedy_reward"]["score_gain_mean"]
    greedy_score_gain = policy_summary["greedy_score"]["score_gain_mean"]

    local_signal_margin = 0.02
    reward_has_local_signal = greedy_reward_gain >= baseline_gain + local_signal_margin
    score_has_local_gradient = greedy_score_gain >= baseline_gain + local_signal_margin
    reward_score_gap = greedy_score_gain - greedy_reward_gain

    zero_action_is_noop = bool(noop_summary.get("zero_action_is_noop", False))
    if not zero_action_is_noop:
        recommendation = "Zero action changes the state; fix reset/action parameter ranges before long RL."
    elif reward_has_local_signal:
        recommendation = "Reward has a usable local signal; long SAC/PPO runs are worth trying."
    elif score_has_local_gradient:
        recommendation = "Score can be improved locally, but reward-greedy is weak; adjust reward shaping before long RL."
    else:
        recommendation = "No clear local gradient; fix scoring/reward/environment before spending long RL time."

    return {
        "baseline_score_gain_mean": float(baseline_gain),
        "greedy_reward_gain_margin": float(greedy_reward_gain - baseline_gain),
        "greedy_score_gain_margin": float(greedy_score_gain - baseline_gain),
        "greedy_score_minus_reward_gain": float(reward_score_gap),
        "zero_action_is_noop": zero_action_is_noop,
        "reward_has_local_signal": bool(reward_has_local_signal),
        "score_has_local_gradient": bool(score_has_local_gradient),
        "recommendation": recommendation,
    }


def main() -> None:
    args = parse_args()
    policy_names = ("zero", "random", "greedy_reward", "greedy_score")
    records_by_policy: Dict[str, List[Dict[str, object]]] = {name: [] for name in policy_names}
    noop_summary = run_noop_check(args)

    for episode in range(args.episodes):
        episode_seed = args.seed + episode
        for policy_index, policy_name in enumerate(policy_names):
            rng = np.random.default_rng(args.seed * 100_000 + episode * 10 + policy_index)
            records_by_policy[policy_name].append(rollout_policy(policy_name, args, episode_seed, rng))

    policy_summary = {
        policy_name: summarize_records(records)
        for policy_name, records in records_by_policy.items()
    }
    summary = {
        "config": vars(args),
        "noop_summary": noop_summary,
        "policy_summary": policy_summary,
        "diagnosis": build_diagnosis(policy_summary, noop_summary),
        "records": records_by_policy,
    }

    output_path = Path(args.output)
    save_json(summary, output_path)
    print(f"Saved reward probe summary to {output_path.resolve()}")
    print(
        f"zero_action_noop={noop_summary['zero_action_is_noop']} "
        f"max_param_delta={noop_summary['max_abs_param_delta']:.6f} "
        f"max_score_gain={noop_summary['max_abs_score_gain']:+.6f}"
    )
    for policy_name, stats in policy_summary.items():
        print(
            f"{policy_name:>13}: gain={stats['score_gain_mean']:+.4f}, "
            f"reward={stats['reward_mean']:+.4f}, "
            f"path={stats['final_path_score_mean']:.4f}, "
            f"conn={stats['final_connectivity_score_mean']:.4f}, "
            f"positive={stats['positive_gain_rate']:.2f}"
        )
    print(summary["diagnosis"]["recommendation"])


if __name__ == "__main__":
    main()
