"""
One-command launcher for Optuna tuning followed by formal VAE+RL training.

Default usage:
    python run_optuna_then_rl.py

To reuse an existing Optuna result:
    python run_optuna_then_rl.py --reuse-best-trial
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
        help="Repository root containing optuna_vae_tuning.py and formal_experiment.py",
    )
    parser.add_argument("--optuna-output-dir", type=str, default="optuna_latent64_3k_v3_voronoi_drop_connectivity")
    parser.add_argument("--rl-output-dir", type=str, default="formal_rl_from_3k_v3_voronoi_drop_connectivity_tuned_v2")

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
    parser.add_argument(
        "--drop-connectivity-supervision",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop connectivity from VAE structure supervision by default for this branch.",
    )
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--optuna-batch-size", type=int, default=32)
    parser.add_argument("--optuna-epochs", type=int, default=12)
    parser.add_argument("--optuna-trials", type=int, default=12)

    parser.add_argument("--rl-batch-size", type=int, default=16)
    parser.add_argument(
        "--rl-update-batch-size",
        type=int,
        default=64,
        help="PPO/SAC update batch size. This is separate from --rl-batch-size, which is still the VAE batch size.",
    )
    parser.add_argument("--vae-epochs", type=int, default=40)
    parser.add_argument("--ppo-episodes", type=int, default=2500)
    parser.add_argument("--ppo-max-steps", type=int, default=18)
    parser.add_argument("--ppo-rollout-steps", type=int, default=1024)
    parser.add_argument("--ppo-lr", type=float, default=3e-4)
    parser.add_argument("--sac-episodes", type=int, default=6000)
    parser.add_argument("--eval-islands", type=int, default=96)
    parser.add_argument(
        "--rl-reset-profile",
        type=str,
        default="island_voronoi",
        choices=["uniform", "island", "island_voronoi"],
        help="Sampling profile used for RL resets.",
    )

    parser.add_argument("--rl-action-step-scale", type=float, default=0.05)
    parser.add_argument("--sac-learning-starts", type=int, default=2048)
    parser.add_argument("--sac-print-interval", type=int, default=25)
    parser.add_argument("--sac-hidden-dim", type=int, default=256)
    parser.add_argument("--sac-actor-lr", type=float, default=1e-4)
    parser.add_argument("--sac-critic-lr", type=float, default=3e-4)
    parser.add_argument("--sac-alpha-lr", type=float, default=5e-5)
    parser.add_argument("--sac-target-entropy-scale", type=float, default=0.5)

    parser.add_argument("--expert-top-percent", type=float, default=0.10)
    parser.add_argument("--expert-max-samples", type=int, default=256)
    parser.add_argument("--novelty-reference-size", type=int, default=256)
    parser.add_argument("--reward-current-scale", type=float, default=0.4)
    parser.add_argument("--reward-delta-scale", type=float, default=2.5)
    parser.add_argument("--reward-best-scale", type=float, default=2.0)
    parser.add_argument("--reward-step-penalty", type=float, default=0.01)
    parser.add_argument("--reward-success-bonus", type=float, default=1.0)
    parser.add_argument("--reward-success-threshold", type=float, default=0.72)
    parser.add_argument("--reward-success-gain-threshold", type=float, default=0.03)
    parser.add_argument("--reward-failure-threshold", type=float, default=0.10)
    parser.add_argument("--reward-success-streak", type=int, default=3)
    parser.add_argument("--reward-stagnation-patience", type=int, default=7)
    parser.add_argument("--reward-stagnation-delta", type=float, default=1e-3)
    parser.add_argument("--rl-best-window", type=int, default=50)
    parser.add_argument(
        "--restore-best-policy",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Evaluate the rolling-best checkpoint instead of the last checkpoint.",
    )
    parser.add_argument(
        "--tensorboard",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write training scalars for TensorBoard plus CSV.",
    )
    parser.add_argument("--tensorboard-dir", type=str, default="")
    parser.add_argument("--unity-export-top-k", type=int, default=2)
    parser.add_argument("--unity-terrain-width", type=float, default=512.0)
    parser.add_argument("--unity-terrain-length", type=float, default=512.0)
    parser.add_argument("--unity-terrain-height", type=float, default=80.0)
    parser.add_argument("--unity-sea-level", type=float, default=0.30)
    parser.add_argument("--unity-target-resolution", type=int, default=257)

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
        "--rl-update-batch-size",
        str(args.rl_update_batch_size),
        "--vae-epochs",
        str(args.vae_epochs),
        "--ppo-episodes",
        str(args.ppo_episodes),
        "--ppo-max-steps",
        str(args.ppo_max_steps),
        "--ppo-rollout-steps",
        str(args.ppo_rollout_steps),
        "--ppo-lr",
        str(args.ppo_lr),
        "--sac-episodes",
        str(args.sac_episodes),
        "--eval-islands",
        str(args.eval_islands),
        "--optuna-best-trial",
        str(best_trial_path),
        "--rl-reset-profile",
        args.rl_reset_profile,
        "--rl-action-step-scale",
        str(args.rl_action_step_scale),
        "--reward-current-scale",
        str(args.reward_current_scale),
        "--reward-delta-scale",
        str(args.reward_delta_scale),
        "--reward-best-scale",
        str(args.reward_best_scale),
        "--reward-step-penalty",
        str(args.reward_step_penalty),
        "--reward-success-bonus",
        str(args.reward_success_bonus),
        "--reward-success-threshold",
        str(args.reward_success_threshold),
        "--reward-success-gain-threshold",
        str(args.reward_success_gain_threshold),
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
        "--sac-hidden-dim",
        str(args.sac_hidden_dim),
        "--sac-critic-lr",
        str(args.sac_critic_lr),
        "--sac-alpha-lr",
        str(args.sac_alpha_lr),
        "--sac-target-entropy-scale",
        str(args.sac_target_entropy_scale),
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
        "--rl-best-window",
        str(args.rl_best_window),
        "--unity-export-top-k",
        str(args.unity_export_top_k),
        "--unity-terrain-width",
        str(args.unity_terrain_width),
        "--unity-terrain-length",
        str(args.unity_terrain_length),
        "--unity-terrain-height",
        str(args.unity_terrain_height),
        "--unity-sea-level",
        str(args.unity_sea_level),
        "--unity-target-resolution",
        str(args.unity_target_resolution),
        "--seed",
        str(args.seed),
    ]
    command.append("--restore-best-policy" if args.restore_best_policy else "--no-restore-best-policy")
    command.append("--tensorboard" if args.tensorboard else "--no-tensorboard")
    if args.tensorboard_dir:
        command.extend(["--tensorboard-dir", args.tensorboard_dir])
    if args.drop_connectivity_supervision:
        command.append("--drop-connectivity-supervision")
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

    should_run_optuna = not (args.reuse_best_trial and best_trial_path.exists())
    if should_run_optuna:
        run_command(build_optuna_command(args, workspace), workspace, args.dry_run)
    else:
        print(f"[Info] Reusing existing best trial: {best_trial_path}")

    if not args.dry_run and not best_trial_path.exists():
        raise FileNotFoundError(f"Optuna completed but best_trial.json was not found: {best_trial_path}")

    run_command(build_rl_command(args, workspace, best_trial_path), workspace, args.dry_run)


if __name__ == "__main__":
    main()
