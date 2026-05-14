"""
Run Optuna VAE tuning first, then automatically launch formal RL with the best trial.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sequential Optuna -> formal RL launcher")
    parser.add_argument("--python-exe", type=str, default=sys.executable, help="Python executable to use")
    parser.add_argument(
        "--workspace",
        type=str,
        default=str(Path(__file__).resolve().parent),
        help="Repository root that contains optuna_vae_tuning.py and formal_experiment.py",
    )
    parser.add_argument("--optuna-output-dir", type=str, required=True, help="Output directory for Optuna tuning")
    parser.add_argument("--rl-output-dir", type=str, required=True, help="Output directory for formal RL")
    parser.add_argument("--map-size", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--dataset-samples", type=int, default=1000)
    parser.add_argument("--min-clean-samples", type=int, default=3000)
    parser.add_argument("--max-dataset-samples", type=int, default=12000)
    parser.add_argument(
        "--sampling-profile",
        type=str,
        default="island_voronoi",
        choices=["uniform", "island", "island_voronoi"],
    )
    parser.add_argument("--drop-connectivity-supervision", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--optuna-batch-size", type=int, default=32)
    parser.add_argument("--optuna-epochs", type=int, default=12)
    parser.add_argument("--optuna-trials", type=int, default=12)

    parser.add_argument("--rl-batch-size", type=int, default=16)
    parser.add_argument("--vae-epochs", type=int, default=40)
    parser.add_argument("--ppo-episodes", type=int, default=60)
    parser.add_argument("--ppo-max-steps", type=int, default=50)
    parser.add_argument("--sac-episodes", type=int, default=1000)
    parser.add_argument("--eval-islands", type=int, default=24)
    parser.add_argument(
        "--rl-reset-profile",
        type=str,
        default="",
        choices=["", "uniform", "island", "island_voronoi"],
        help="Optional override for RL reset profile. Empty means follow sampling-profile.",
    )
    parser.add_argument("--sac-learning-starts", type=int, default=512)
    parser.add_argument("--sac-print-interval", type=int, default=10)
    parser.add_argument("--sac-actor-lr", type=float, default=3e-4)
    parser.add_argument("--sac-critic-lr", type=float, default=1e-3)
    parser.add_argument("--sac-alpha-lr", type=float, default=3e-4)

    parser.add_argument("--expert-top-percent", type=float, default=0.10)
    parser.add_argument("--expert-max-samples", type=int, default=256)
    parser.add_argument("--novelty-reference-size", type=int, default=256)
    parser.add_argument("--reward-delta-scale", type=float, default=3.0)
    parser.add_argument("--reward-best-scale", type=float, default=0.3)
    parser.add_argument("--reward-expert-scale", type=float, default=0.05)
    parser.add_argument("--reward-step-penalty", type=float, default=0.005)
    parser.add_argument("--reward-success-bonus", type=float, default=0.8)
    parser.add_argument("--reward-failure-penalty", type=float, default=0.8)
    parser.add_argument("--reward-stagnation-penalty", type=float, default=0.05)
    parser.add_argument("--reward-success-threshold", type=float, default=0.70)
    parser.add_argument("--reward-failure-threshold", type=float, default=0.12)
    parser.add_argument("--reward-success-streak", type=int, default=2)
    parser.add_argument("--reward-stagnation-patience", type=int, default=15)
    parser.add_argument("--reward-stagnation-delta", type=float, default=1e-3)

    parser.add_argument("--reuse-best-trial", action="store_true", help="Skip Optuna when best_trial.json already exists")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them")
    return parser.parse_args()


def render_command(command: list[str]) -> str:
    return subprocess.list2cmdline(command)


def run_command(command: list[str], cwd: Path, dry_run: bool) -> None:
    print(f"\n[Run] {render_command(command)}")
    if dry_run:
        return
    subprocess.run(command, cwd=str(cwd), check=True)


def build_optuna_command(args: argparse.Namespace, workspace: Path) -> list[str]:
    command = [
        args.python_exe,
        str(workspace / "optuna_vae_tuning.py"),
        "--output-dir",
        args.optuna_output_dir,
        "--map-size",
        str(args.map_size),
        "--latent-dim",
        str(args.latent_dim),
        "--dataset-samples",
        str(args.dataset_samples),
        "--min-clean-samples",
        str(args.min_clean_samples),
        "--max-dataset-samples",
        str(args.max_dataset_samples),
        "--sampling-profile",
        args.sampling_profile,
        "--batch-size",
        str(args.optuna_batch_size),
        "--epochs",
        str(args.optuna_epochs),
        "--trials",
        str(args.optuna_trials),
        "--seed",
        str(args.seed),
    ]
    if args.drop_connectivity_supervision:
        command.append("--drop-connectivity-supervision")
    return command


def build_rl_command(args: argparse.Namespace, workspace: Path, best_trial_path: Path) -> list[str]:
    command = [
        args.python_exe,
        str(workspace / "formal_experiment.py"),
        "--formal-rl",
        "--output-dir",
        args.rl_output_dir,
        "--map-size",
        str(args.map_size),
        "--latent-dim",
        str(args.latent_dim),
        "--dataset-samples",
        str(args.dataset_samples),
        "--min-clean-samples",
        str(args.min_clean_samples),
        "--max-dataset-samples",
        str(args.max_dataset_samples),
        "--sampling-profile",
        args.sampling_profile,
        "--batch-size",
        str(args.rl_batch_size),
        "--vae-epochs",
        str(args.vae_epochs),
        "--ppo-episodes",
        str(args.ppo_episodes),
        "--ppo-max-steps",
        str(args.ppo_max_steps),
        "--sac-episodes",
        str(args.sac_episodes),
        "--eval-islands",
        str(args.eval_islands),
        "--optuna-best-trial",
        str(best_trial_path),
        "--reward-delta-scale",
        str(args.reward_delta_scale),
        "--reward-best-scale",
        str(args.reward_best_scale),
        "--reward-expert-scale",
        str(args.reward_expert_scale),
        "--reward-step-penalty",
        str(args.reward_step_penalty),
        "--reward-success-bonus",
        str(args.reward_success_bonus),
        "--reward-failure-penalty",
        str(args.reward_failure_penalty),
        "--reward-stagnation-penalty",
        str(args.reward_stagnation_penalty),
        "--reward-success-threshold",
        str(args.reward_success_threshold),
        "--reward-failure-threshold",
        str(args.reward_failure_threshold),
        "--reward-success-streak",
        str(args.reward_success_streak),
        "--reward-stagnation-patience",
        str(args.reward_stagnation_patience),
        "--reward-stagnation-delta",
        str(args.reward_stagnation_delta),
        "--sac-actor-lr",
        str(args.sac_actor_lr),
        "--sac-critic-lr",
        str(args.sac_critic_lr),
        "--sac-alpha-lr",
        str(args.sac_alpha_lr),
        "--expert-top-percent",
        str(args.expert_top_percent),
        "--expert-max-samples",
        str(args.expert_max_samples),
        "--novelty-reference-size",
        str(args.novelty_reference_size),
        "--sac-learning-starts",
        str(args.sac_learning_starts),
        "--sac-print-interval",
        str(args.sac_print_interval),
        "--seed",
        str(args.seed),
    ]
    if args.drop_connectivity_supervision:
        command.append("--drop-connectivity-supervision")
    if args.rl_reset_profile:
        command.extend(["--rl-reset-profile", args.rl_reset_profile])
    return command


def main() -> None:
    args = parse_args()
    workspace = Path(args.workspace).resolve()
    optuna_output_dir = Path(args.optuna_output_dir)
    best_trial_path = workspace / optuna_output_dir / "best_trial.json"

    if not (workspace / "optuna_vae_tuning.py").exists():
        raise FileNotFoundError(f"Cannot find optuna_vae_tuning.py under {workspace}")
    if not (workspace / "formal_experiment.py").exists():
        raise FileNotFoundError(f"Cannot find formal_experiment.py under {workspace}")

    should_run_optuna = True
    if args.reuse_best_trial and best_trial_path.exists():
        should_run_optuna = False
        print(f"[Info] Reusing existing best trial: {best_trial_path}")

    if should_run_optuna:
        run_command(build_optuna_command(args, workspace), workspace, args.dry_run)

    if not args.dry_run and not best_trial_path.exists():
        raise FileNotFoundError(f"Optuna completed but best_trial.json was not found: {best_trial_path}")

    run_command(build_rl_command(args, workspace, best_trial_path), workspace, args.dry_run)


if __name__ == "__main__":
    main()
