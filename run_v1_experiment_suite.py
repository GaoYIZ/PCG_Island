"""Run the restored-v1 SAC tuning, PPO, VAE, reward, and baseline ablation suite."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any


DEFAULT_SEEDS = [42, 43, 44]
DEFAULT_ABLATION_SEEDS = [42]

V1_REWARD_CONFIG = {
    "current_scale": 0.3,
    "delta_scale": 3.0,
    "best_scale": 2.0,
    "step_penalty": 0.01,
    "success_bonus": 1.0,
    "success_threshold": 0.72,
    "success_gain_threshold": 0.03,
    "failure_threshold": 0.10,
    "success_streak": 3,
    "stagnation_patience": 5,
    "stagnation_delta": 0.001,
}

SAC_SCREEN_VARIANTS = {
    "s1_alpha_slow": {
        "alpha_lr": 5e-5,
        "min_alpha": 0.0,
        "entropy_scale": 1.0,
        "action_step": 0.06,
    },
    "s2_alpha_floor": {
        "alpha_lr": 5e-5,
        "min_alpha": 0.03,
        "entropy_scale": 1.0,
        "action_step": 0.06,
    },
    "s3_entropy_target": {
        "alpha_lr": 5e-5,
        "min_alpha": 0.03,
        "entropy_scale": 0.8,
        "action_step": 0.06,
    },
    "s4_small_step": {
        "alpha_lr": 5e-5,
        "min_alpha": 0.03,
        "entropy_scale": 0.8,
        "action_step": 0.045,
    },
}

PPO_ABLATIONS = {
    "a0_latent64_depth2": {"latent_dim": 64, "include_latent": True, "hidden_layers": 2},
    "a1_no_latent_depth2": {"latent_dim": 64, "include_latent": False, "hidden_layers": 2},
    "a2_latent32_depth2": {"latent_dim": 32, "include_latent": True, "hidden_layers": 2},
    "a3_latent128_depth2": {"latent_dim": 128, "include_latent": True, "hidden_layers": 2},
    "a4_latent64_depth1": {"latent_dim": 64, "include_latent": True, "hidden_layers": 1},
    "a5_latent64_depth3": {"latent_dim": 64, "include_latent": True, "hidden_layers": 3},
}

VAE_ABLATIONS = {
    "full_v1": {
        "description": "Restored v1 structure-aware VAE assets.",
        "reuse_baseline_assets": True,
        "best_trial_patch": {},
    },
    "no_structure_loss": {
        "description": "Reconstruction-only VAE: no structure loss and no metric alignment.",
        "reuse_baseline_assets": False,
        "best_trial_patch": {
            "best_params": {
                "structure_loss_weight": 0.0,
                "metric_alignment_loss_weight": 0.0,
            }
        },
    },
    "no_metric_alignment": {
        "description": "Keep the structure head but remove metric-alignment loss.",
        "reuse_baseline_assets": False,
        "best_trial_patch": {
            "best_params": {
                "metric_alignment_loss_weight": 0.0,
            }
        },
    },
    "with_connectivity_supervision": {
        "description": "Keep connectivity inside VAE supervision instead of the v1 drop-connectivity setup.",
        "reuse_baseline_assets": False,
        "best_trial_patch": {
            "drop_connectivity_supervision": False,
        },
    },
}

REWARD_ABLATIONS = {
    "full_reward": dict(V1_REWARD_CONFIG),
    "absolute_score_only": {
        **V1_REWARD_CONFIG,
        "current_scale": 1.0,
        "delta_scale": 0.0,
        "best_scale": 0.0,
        "step_penalty": 0.0,
        "success_bonus": 0.0,
    },
    "no_best_bonus": {
        **V1_REWARD_CONFIG,
        "best_scale": 0.0,
    },
    "no_success_bonus": {
        **V1_REWARD_CONFIG,
        "success_bonus": 0.0,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="One-command restored-v1 final experiment suite")
    parser.add_argument(
        "--stage",
        choices=[
            "sac-screen",
            "sac-long",
            "ppo-ablation",
            "vae-ablation",
            "reward-ablation",
            "baseline-ablation",
            "all",
        ],
        default="all",
    )
    parser.add_argument("--workspace", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--python-exe", type=str, default=sys.executable)
    parser.add_argument("--map-size", type=int, default=64)
    parser.add_argument(
        "--baseline-assets",
        type=Path,
        default=Path("formal_rl_from_3k_v3_voronoi_drop_connectivity_tuned_v1"),
    )
    parser.add_argument("--output-root", type=Path, default=Path("experiments") / "v1_suite")
    parser.add_argument(
        "--best-trial",
        type=Path,
        default=Path("configs") / "optuna_best_latent64_3k_v3_voronoi_drop_connectivity.json",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--ablation-seeds", nargs="+", type=int, default=list(DEFAULT_ABLATION_SEEDS))
    parser.add_argument("--screen-episodes", type=int, default=1500)
    parser.add_argument("--long-episodes", type=int, default=4000)
    parser.add_argument("--ppo-episodes", type=int, default=1500)
    parser.add_argument("--ablation-episodes", type=int, default=1500)
    parser.add_argument("--eval-islands", type=int, default=64)
    parser.add_argument("--asset-dataset-samples", type=int, default=1000)
    parser.add_argument("--asset-min-clean-samples", type=int, default=3000)
    parser.add_argument("--asset-max-dataset-samples", type=int, default=12000)
    parser.add_argument("--asset-vae-epochs", type=int, default=40)
    parser.add_argument("--cmaes-generations", type=int, default=60)
    parser.add_argument("--cmaes-pop-size", type=int, default=16)
    parser.add_argument("--cmaes-sigma", type=float, default=0.25)
    parser.add_argument("--fast-dev", action="store_true", help="Tiny end-to-end run for server smoke testing.")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def apply_fast_dev(args: argparse.Namespace) -> None:
    if not args.fast_dev:
        return
    args.seeds = [args.seeds[0]]
    args.ablation_seeds = [args.ablation_seeds[0]]
    args.screen_episodes = min(args.screen_episodes, 1)
    args.long_episodes = min(args.long_episodes, 1)
    args.ppo_episodes = min(args.ppo_episodes, 1)
    args.ablation_episodes = min(args.ablation_episodes, 1)
    args.eval_islands = min(args.eval_islands, 1)
    args.asset_dataset_samples = min(args.asset_dataset_samples, 24)
    args.asset_min_clean_samples = min(args.asset_min_clean_samples, 12)
    args.asset_max_dataset_samples = min(args.asset_max_dataset_samples, 80)
    args.asset_vae_epochs = min(args.asset_vae_epochs, 1)
    args.cmaes_generations = min(args.cmaes_generations, 1)
    args.cmaes_pop_size = min(args.cmaes_pop_size, 4)


def render_command(command: list[str]) -> str:
    return subprocess.list2cmdline(command)


def resolve_under_workspace(path: Path, workspace: Path) -> Path:
    return path.resolve() if path.is_absolute() else (workspace / path).resolve()


def reward_args(config: dict[str, Any]) -> list[str]:
    return [
        "--reward-current-scale",
        str(config["current_scale"]),
        "--reward-delta-scale",
        str(config["delta_scale"]),
        "--reward-best-scale",
        str(config["best_scale"]),
        "--reward-step-penalty",
        str(config["step_penalty"]),
        "--reward-success-bonus",
        str(config["success_bonus"]),
        "--reward-success-threshold",
        str(config["success_threshold"]),
        "--reward-success-gain-threshold",
        str(config["success_gain_threshold"]),
        "--reward-failure-threshold",
        str(config["failure_threshold"]),
        "--reward-success-streak",
        str(config["success_streak"]),
        "--reward-stagnation-patience",
        str(config["stagnation_patience"]),
        "--reward-stagnation-delta",
        str(config["stagnation_delta"]),
    ]


def is_complete(output_dir: Path, agent: str | None = None) -> bool:
    if not (output_dir / "final_summary.json").exists():
        return False
    if agent is None:
        return (output_dir / "rl_assets_ready.json").exists()
    if agent in {"cmaes", "sac_best"}:
        return True
    return (output_dir / f"{agent}_training_summary.json").exists() and (
        output_dir / f"{agent}_evaluation_summary.json"
    ).exists()


def has_reusable_assets(path: Path) -> bool:
    return all(
        required.exists()
        for required in (
            path / "artifacts" / "vae_checkpoint.pt",
            path / "artifacts" / "feature_normalizer.json",
            path / "expert_bank.npz",
        )
    )


def write_suite_manifest(
    output_dir: Path,
    family: str,
    variant: str,
    seed: int,
    agent: str | None,
    parameters: dict[str, Any],
    command: list[str] | None = None,
) -> None:
    manifest = {
        "family": family,
        "variant": variant,
        "seed": int(seed),
        "agent": agent,
        "parameters": parameters,
        "command": command or [],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "suite_run.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def run_command(
    args: argparse.Namespace,
    command: list[str],
    output_dir: Path,
    family: str,
    variant: str,
    seed: int,
    agent: str | None,
    parameters: dict[str, Any],
) -> None:
    if args.resume and is_complete(output_dir, agent):
        manifest_path = output_dir / "suite_run.json"
        if not args.dry_run and not manifest_path.exists():
            write_suite_manifest(output_dir, family, variant, seed, agent, parameters, command)
        print(f"[Skip] Completed: {output_dir}")
        return
    print(f"\n[Run] {render_command(command)}")
    if args.dry_run:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(command, cwd=str(args.workspace), check=True)
    write_suite_manifest(output_dir, family, variant, seed, agent, parameters, command)


def common_formal_args(
    args: argparse.Namespace,
    output_dir: Path,
    latent_dim: int,
    seed: int,
    reward_config: dict[str, Any] | None = None,
) -> list[str]:
    return [
        args.python_exe,
        str(args.workspace / "formal_experiment.py"),
        "--formal-rl",
        "--output-dir",
        str(output_dir),
        "--map-size",
        str(args.map_size),
        "--latent-dim",
        str(latent_dim),
        "--sampling-profile",
        "island_voronoi",
        "--rl-reset-profile",
        "island_voronoi",
        "--batch-size",
        "16",
        "--rl-update-batch-size",
        "64",
        "--ppo-max-steps",
        "15",
        "--ppo-rollout-steps",
        "1024",
        "--ppo-lr",
        "0.0003",
        "--eval-islands",
        str(args.eval_islands),
        "--rl-best-window",
        "50",
        "--seed",
        str(seed),
        "--restore-best-policy",
        "--tensorboard",
        *reward_args(reward_config or V1_REWARD_CONFIG),
    ]


def add_asset_training_args(args: argparse.Namespace, command: list[str], best_trial: Path) -> None:
    command.extend(
        [
            "--prepare-rl-assets-only",
            "--dataset-samples",
            str(args.asset_dataset_samples),
            "--min-clean-samples",
            str(args.asset_min_clean_samples),
            "--max-dataset-samples",
            str(args.asset_max_dataset_samples),
            "--vae-epochs",
            str(args.asset_vae_epochs),
            "--optuna-best-trial",
            str(best_trial),
            "--preserve-latent-dim",
        ]
    )


def patched_best_trial(args: argparse.Namespace, variant_name: str, patch: dict[str, Any]) -> Path:
    target = args.output_root / "configs" / f"best_trial_{variant_name}.json"
    if args.dry_run:
        return target
    if args.resume and target.exists():
        return target
    if not args.best_trial.exists():
        raise FileNotFoundError(f"Optuna best trial not found: {args.best_trial}")
    data = json.loads(args.best_trial.read_text(encoding="utf-8"))
    data.setdefault("best_params", {})
    for key, value in patch.items():
        if key == "best_params":
            data["best_params"].update(value)
        else:
            data[key] = value
    data["suite_variant"] = variant_name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return target


def run_sac_screen(args: argparse.Namespace) -> None:
    for name, params in SAC_SCREEN_VARIANTS.items():
        output_dir = args.output_root / "sac_screen" / name
        command = common_formal_args(args, output_dir, latent_dim=64, seed=42)
        command.extend(
            [
                "--rl-agent",
                "sac",
                "--reuse-rl-assets-from",
                str(args.baseline_assets),
                "--sac-episodes",
                str(args.screen_episodes),
                "--sac-learning-starts",
                "4096",
                "--sac-hidden-dim",
                "256",
                "--sac-actor-lr",
                "0.0001",
                "--sac-critic-lr",
                "0.0003",
                "--sac-alpha-lr",
                str(params["alpha_lr"]),
                "--sac-min-alpha",
                str(params["min_alpha"]),
                "--sac-target-entropy-scale",
                str(params["entropy_scale"]),
                "--rl-action-step-scale",
                str(params["action_step"]),
                "--sac-print-interval",
                "25",
            ]
        )
        run_command(args, command, output_dir, "sac_screen", name, 42, "sac", params)


def sac_screen_score(run_dir: Path) -> dict[str, float] | None:
    path = run_dir / "sac_training_summary.json"
    if not path.exists():
        return None
    episodes = json.loads(path.read_text(encoding="utf-8")).get("episodes", [])
    tail = episodes[-200:]
    if not tail:
        return None
    final_scores = [float(row.get("score", {}).get("total_score", float("nan"))) for row in tail]
    gains = [float(row.get("score_gain", float("nan"))) for row in tail]
    loss_values = [
        float(value)
        for row in tail
        for value in row.get("losses", {}).values()
        if isinstance(value, (int, float))
    ]
    if not all(math.isfinite(value) for value in final_scores + gains + loss_values):
        return None
    successes = [float(row.get("done_reason") == "success_threshold") for row in tail]
    return {
        "last200_final_score": sum(final_scores) / len(final_scores),
        "last200_score_gain": sum(gains) / len(gains),
        "last200_success_rate": sum(successes) / len(successes),
    }


def select_top_sac_variants(args: argparse.Namespace) -> list[str]:
    ranked = []
    for name in SAC_SCREEN_VARIANTS:
        metrics = sac_screen_score(args.output_root / "sac_screen" / name)
        if metrics is not None:
            ranked.append((metrics["last200_final_score"], metrics["last200_score_gain"], name, metrics))
    if len(ranked) < 2:
        raise RuntimeError("SAC long stage requires at least two completed, finite screen runs.")
    ranked.sort(reverse=True)
    selected = [row[2] for row in ranked[:2]]
    selection = {"selected": selected, "ranking": [row[3] | {"variant": row[2]} for row in ranked]}
    if not args.dry_run:
        (args.output_root / "sac_selection.json").write_text(
            json.dumps(selection, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    print(f"[Select] SAC long variants: {', '.join(selected)}")
    return selected


def run_sac_long(args: argparse.Namespace) -> None:
    if args.dry_run and not all(
        (args.output_root / "sac_screen" / name / "sac_training_summary.json").exists()
        for name in SAC_SCREEN_VARIANTS
    ):
        print("[DryRun] SAC long commands are selected after completed sac-screen results.")
        return
    for name in select_top_sac_variants(args):
        params = SAC_SCREEN_VARIANTS[name]
        output_dir = args.output_root / "sac_long" / name
        command = common_formal_args(args, output_dir, latent_dim=64, seed=42)
        command.extend(
            [
                "--rl-agent",
                "sac",
                "--reuse-rl-assets-from",
                str(args.baseline_assets),
                "--sac-episodes",
                str(args.long_episodes),
                "--sac-learning-starts",
                "4096",
                "--sac-hidden-dim",
                "256",
                "--sac-actor-lr",
                "0.0001",
                "--sac-critic-lr",
                "0.0003",
                "--sac-alpha-lr",
                str(params["alpha_lr"]),
                "--sac-min-alpha",
                str(params["min_alpha"]),
                "--sac-target-entropy-scale",
                str(params["entropy_scale"]),
                "--rl-action-step-scale",
                str(params["action_step"]),
                "--sac-print-interval",
                "25",
            ]
        )
        run_command(args, command, output_dir, "sac_long", name, 42, "sac", params)


def prepare_latent_assets(args: argparse.Namespace, latent_dim: int) -> Path:
    if latent_dim == 64:
        return args.baseline_assets
    output_dir = args.output_root / "assets" / f"latent_{latent_dim}"
    command = common_formal_args(args, output_dir, latent_dim=latent_dim, seed=42)
    add_asset_training_args(args, command, args.best_trial)
    command.append("--drop-connectivity-supervision")
    run_command(
        args,
        command,
        output_dir,
        "ppo_assets",
        f"latent_{latent_dim}",
        42,
        None,
        {"latent_dim": latent_dim},
    )
    return output_dir


def ensure_baseline_assets(args: argparse.Namespace) -> None:
    if has_reusable_assets(args.baseline_assets):
        return
    output_dir = args.output_root / "assets" / "latent_64_v1"
    args.baseline_assets = output_dir
    command = common_formal_args(args, output_dir, latent_dim=64, seed=42)
    add_asset_training_args(args, command, args.best_trial)
    command.append("--drop-connectivity-supervision")
    run_command(
        args,
        command,
        output_dir,
        "ppo_assets",
        "latent_64_v1",
        42,
        None,
        {"latent_dim": 64, "source": "restored_v1_optuna_config"},
    )


def run_ppo_ablations(args: argparse.Namespace) -> None:
    asset_dirs = {dim: prepare_latent_assets(args, dim) for dim in (32, 64, 128)}
    for variant, params in PPO_ABLATIONS.items():
        for seed in args.seeds:
            output_dir = args.output_root / "ppo_ablation" / variant / f"seed_{seed}"
            command = common_formal_args(args, output_dir, latent_dim=params["latent_dim"], seed=seed)
            command.extend(
                [
                    "--rl-agent",
                    "ppo",
                    "--reuse-rl-assets-from",
                    str(asset_dirs[params["latent_dim"]]),
                    "--ppo-episodes",
                    str(args.ppo_episodes),
                    "--ppo-hidden-dim",
                    "256",
                    "--ppo-hidden-layers",
                    str(params["hidden_layers"]),
                    "--rl-action-step-scale",
                    "0.06",
                    "--include-latent-state" if params["include_latent"] else "--no-include-latent-state",
                    "--sac-episodes",
                    "0",
                ]
            )
            run_command(args, command, output_dir, "ppo_ablation", variant, seed, "ppo", params)


def prepare_vae_ablation_assets(args: argparse.Namespace, variant: str, params: dict[str, Any]) -> Path:
    if params["reuse_baseline_assets"]:
        return args.baseline_assets
    output_dir = args.output_root / "assets" / "vae_ablation" / variant
    trial_path = patched_best_trial(args, f"vae_{variant}", params["best_trial_patch"])
    command = common_formal_args(args, output_dir, latent_dim=64, seed=42)
    add_asset_training_args(args, command, trial_path)
    run_command(
        args,
        command,
        output_dir,
        "vae_assets",
        variant,
        42,
        None,
        {"latent_dim": 64, "description": params["description"]},
    )
    return output_dir


def run_vae_ablations(args: argparse.Namespace) -> None:
    for variant, params in VAE_ABLATIONS.items():
        asset_dir = prepare_vae_ablation_assets(args, variant, params)
        for seed in args.ablation_seeds:
            output_dir = args.output_root / "vae_ablation" / variant / f"seed_{seed}"
            command = common_formal_args(args, output_dir, latent_dim=64, seed=seed)
            command.extend(
                [
                    "--rl-agent",
                    "ppo",
                    "--reuse-rl-assets-from",
                    str(asset_dir),
                    "--ppo-episodes",
                    str(args.ablation_episodes),
                    "--ppo-hidden-dim",
                    "256",
                    "--ppo-hidden-layers",
                    "2",
                    "--rl-action-step-scale",
                    "0.06",
                    "--include-latent-state",
                    "--sac-episodes",
                    "0",
                ]
            )
            run_command(
                args,
                command,
                output_dir,
                "vae_ablation",
                variant,
                seed,
                "ppo",
                {"asset_dir": str(asset_dir), "description": params["description"]},
            )


def run_reward_ablations(args: argparse.Namespace) -> None:
    for variant, config in REWARD_ABLATIONS.items():
        for seed in args.ablation_seeds:
            output_dir = args.output_root / "reward_ablation" / variant / f"seed_{seed}"
            command = common_formal_args(args, output_dir, latent_dim=64, seed=seed, reward_config=config)
            command.extend(
                [
                    "--rl-agent",
                    "ppo",
                    "--reuse-rl-assets-from",
                    str(args.baseline_assets),
                    "--ppo-episodes",
                    str(args.ablation_episodes),
                    "--ppo-hidden-dim",
                    "256",
                    "--ppo-hidden-layers",
                    "2",
                    "--rl-action-step-scale",
                    "0.06",
                    "--include-latent-state",
                    "--sac-episodes",
                    "0",
                ]
            )
            run_command(args, command, output_dir, "reward_ablation", variant, seed, "ppo", config)


def run_baseline_ppo_zero_random(args: argparse.Namespace) -> None:
    for seed in args.ablation_seeds:
        output_dir = args.output_root / "baseline_ablation" / "ppo_vs_zero_random" / f"seed_{seed}"
        command = common_formal_args(args, output_dir, latent_dim=64, seed=seed)
        command.extend(
            [
                "--rl-agent",
                "ppo",
                "--reuse-rl-assets-from",
                str(args.baseline_assets),
                "--ppo-episodes",
                str(args.ablation_episodes),
                "--ppo-hidden-dim",
                "256",
                "--ppo-hidden-layers",
                "2",
                "--rl-action-step-scale",
                "0.06",
                "--include-latent-state",
                "--sac-episodes",
                "0",
            ]
        )
        run_command(
            args,
            command,
            output_dir,
            "baseline_ablation",
            "ppo_vs_zero_random",
            seed,
            "ppo",
            {"baselines": ["zero_action", "random_action", "ppo"]},
        )


def mean_dict(records: list[dict[str, float]]) -> dict[str, float]:
    if not records:
        return {}
    keys = sorted({key for record in records for key in record})
    return {
        key: float(statistics.fmean(float(record[key]) for record in records if key in record))
        for key in keys
    }


def run_cmaes_baselines(args: argparse.Namespace) -> None:
    for seed in args.ablation_seeds:
        output_dir = args.output_root / "baseline_ablation" / "cmaes" / f"seed_{seed}"
        if args.resume and is_complete(output_dir, "cmaes"):
            if not args.dry_run and not (output_dir / "suite_run.json").exists():
                write_suite_manifest(
                    output_dir,
                    "baseline_ablation",
                    "cmaes",
                    seed,
                    "cmaes",
                    {"generations": args.cmaes_generations, "pop_size": args.cmaes_pop_size},
                )
            print(f"[Skip] Completed: {output_dir}")
            continue
        print(
            f"\n[Run] CMA-ES baseline seed={seed} generations={args.cmaes_generations} "
            f"pop_size={args.cmaes_pop_size}"
        )
        if args.dry_run:
            continue
        from cmaes_baseline import CMAESOptimizer
        from map_scoring import MapScorer
        from pcg_generator import PCGIslandGenerator
        from structure_evaluator import StructureEvaluator

        output_dir.mkdir(parents=True, exist_ok=True)
        generator = PCGIslandGenerator(map_size=args.map_size)
        optimizer = CMAESOptimizer(
            generator.get_param_ranges(args.map_size),
            sigma0=args.cmaes_sigma,
            pop_size=args.cmaes_pop_size,
            map_size=args.map_size,
        )
        optimizer.rng = __import__("numpy").random.default_rng(seed)
        best_vector, best_fitness = optimizer.optimize(generations=args.cmaes_generations, verbose=True)

        evaluator = StructureEvaluator(map_size=args.map_size)
        scorer = MapScorer()
        score_records: list[dict[str, float]] = []
        metric_records: list[dict[str, float]] = []
        eval_records: list[dict[str, Any]] = []
        for index in range(args.eval_islands):
            theta = optimizer.param_normalizer.denormalize_vector(best_vector)
            theta["seed"] = int(seed * 100_000 + index)
            heightmap = generator.generate_heightmap(theta)
            metrics = evaluator.evaluate(heightmap)
            score = scorer.score_metrics(metrics).as_dict()
            score_records.append(score)
            metric_records.append({key: float(value) for key, value in metrics.items()})
            eval_records.append(
                {
                    "index": index + 1,
                    "theta": theta,
                    "metrics": {key: float(value) for key, value in metrics.items()},
                    "final_score": score,
                }
            )
        final_score_mean = mean_dict(score_records)
        final_metric_mean = mean_dict(metric_records)
        zero_gain = {key: 0.0 for key in final_score_mean}
        summary = {
            "mode": "cmaes_baseline",
            "status": "complete",
            "seed": int(seed),
            "map_size": int(args.map_size),
            "generations": int(args.cmaes_generations),
            "pop_size": int(args.cmaes_pop_size),
            "sigma": float(args.cmaes_sigma),
            "best_training_fitness": float(best_fitness),
            "best_normalized_theta": [float(value) for value in best_vector],
            "best_theta": optimizer.param_normalizer.denormalize_vector(best_vector),
            "cmaes_summary": {
                "final_score_mean": final_score_mean,
                "score_gain_mean": zero_gain,
                "final_metric_mean": final_metric_mean,
                "reward_summary": {
                    "count": int(args.eval_islands),
                    "mean": float(final_score_mean.get("total_score", 0.0)),
                    "std": 0.0,
                    "min": float(min(record.get("total_score", 0.0) for record in score_records)),
                    "max": float(max(record.get("total_score", 0.0) for record in score_records)),
                },
            },
        }
        (output_dir / "final_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (output_dir / "evaluation_details.json").write_text(
            json.dumps({"records": eval_records}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (output_dir / "best_theta.json").write_text(
            json.dumps(summary["best_theta"], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        write_suite_manifest(
            output_dir,
            "baseline_ablation",
            "cmaes",
            seed,
            "cmaes",
            {"generations": args.cmaes_generations, "pop_size": args.cmaes_pop_size},
        )


def score_section(summary: dict[str, Any], agent: str) -> dict[str, Any] | None:
    section = summary.get(f"{agent.lower()}_summary")
    if not isinstance(section, dict):
        return None
    return extract_scores_from_eval_summary(section)


def extract_scores_from_eval_summary(section: dict[str, Any]) -> dict[str, Any] | None:
    final_score = section.get("final_score_mean", {})
    gain = section.get("score_gain_mean", {})
    if not isinstance(final_score, dict):
        return None
    return {
        "final_total_score": final_score.get("total_score"),
        "total_score_gain": gain.get("total_score") if isinstance(gain, dict) else None,
        "final_structure_score": final_score.get("structure_score"),
        "structure_score_gain": gain.get("structure_score") if isinstance(gain, dict) else None,
        "final_path_score": final_score.get("path_score"),
        "path_score_gain": gain.get("path_score") if isinstance(gain, dict) else None,
        "final_land_score": final_score.get("land_score"),
        "land_score_gain": gain.get("land_score") if isinstance(gain, dict) else None,
        "final_novelty_score": final_score.get("novelty_score"),
        "final_connectivity_score": final_score.get("connectivity_score"),
        "final_navigable_score": final_score.get("navigable_score"),
        "final_coast_score": final_score.get("coast_score"),
        "final_variance_score": final_score.get("variance_score"),
    }


def extract_vae_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    vae_summary = summary.get("vae_pipeline_summary", {})
    if not isinstance(vae_summary, dict):
        return {}
    test_summary = vae_summary.get("test_summary", {})
    if not isinstance(test_summary, dict):
        return {}
    r2_values = [
        float(value)
        for value in test_summary.get("latent_predictive_r2", {}).values()
        if isinstance(value, (int, float)) and math.isfinite(float(value))
    ]
    structure_mae_values = [
        float(value)
        for value in test_summary.get("structure_head_mae", {}).values()
        if isinstance(value, (int, float)) and math.isfinite(float(value))
    ]
    return {
        "vae_test_pixel_mae": test_summary.get("pixel_mae"),
        "vae_test_land_pixel_mae": test_summary.get("land_pixel_mae"),
        "vae_test_coast_band_mae": test_summary.get("coast_band_mae"),
        "vae_test_latent_r2_mean": statistics.fmean(r2_values) if r2_values else None,
        "vae_test_structure_mae_mean": statistics.fmean(structure_mae_values) if structure_mae_values else None,
        "vae_active_latent_dims": test_summary.get("active_latent_dims"),
    }


def row_from_section(
    family: str,
    variant: str,
    seed: int,
    agent: str,
    summary: dict[str, Any],
    section: dict[str, Any],
    status: str = "complete",
) -> dict[str, Any] | None:
    scores = extract_scores_from_eval_summary(section)
    if scores is None:
        return None
    return {
        "family": family,
        "variant": variant,
        "seed": int(seed),
        "agent": agent,
        "status": status,
        **scores,
        **extract_vae_metrics(summary),
    }


def find_best_sac_long_row(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any], Path] | None:
    best: tuple[float, dict[str, Any], dict[str, Any], Path] | None = None
    for summary_path in (args.output_root / "sac_long").glob("*/final_summary.json"):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        section = summary.get("sac_summary")
        scores = extract_scores_from_eval_summary(section) if isinstance(section, dict) else None
        if scores is None or scores.get("final_total_score") is None:
            continue
        value = float(scores["final_total_score"])
        if best is None or value > best[0]:
            best = (value, summary, section, summary_path.parent)
    if best is None:
        return None
    return best[1], best[2], best[3]


def write_sac_best_baseline(args: argparse.Namespace) -> None:
    seed = args.ablation_seeds[0]
    output_dir = args.output_root / "baseline_ablation" / "sac_best" / f"seed_{seed}"
    if args.resume and is_complete(output_dir, "sac_best"):
        if not args.dry_run and not (output_dir / "suite_run.json").exists():
            write_suite_manifest(output_dir, "baseline_ablation", "sac_best", seed, "sac", {})
        print(f"[Skip] Completed: {output_dir}")
        return
    print("\n[Run] SAC best baseline pointer")
    if args.dry_run:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    best = find_best_sac_long_row(args)
    if best is None:
        summary = {
            "mode": "sac_best_pointer",
            "status": "unstable_or_missing",
            "message": "No completed sac_long run was available. Keep SAC as an unstable supplementary result.",
        }
    else:
        source_summary, sac_section, source_dir = best
        summary = {
            "mode": "sac_best_pointer",
            "status": "complete",
            "source_run": str(source_dir),
            "sac_summary": sac_section,
            "source_final_summary": source_summary,
        }
    (output_dir / "final_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_suite_manifest(output_dir, "baseline_ablation", "sac_best", seed, "sac", {})


def run_baseline_ablations(args: argparse.Namespace) -> None:
    run_baseline_ppo_zero_random(args)
    run_cmaes_baselines(args)
    write_sac_best_baseline(args)


def collect_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    baseline_summary_path = args.baseline_assets / "final_summary.json"
    if baseline_summary_path.exists():
        summary = json.loads(baseline_summary_path.read_text(encoding="utf-8"))
        for agent in ("ppo", "sac"):
            section = summary.get(f"{agent}_summary")
            if isinstance(section, dict):
                row = row_from_section("v1_baseline", "archived_v1", 42, agent, summary, section)
                if row is not None:
                    rows.append(row)

    for manifest_path in args.output_root.rglob("suite_run.json"):
        run_dir = manifest_path.parent
        summary_path = run_dir / "final_summary.json"
        if not summary_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        family = manifest["family"]
        variant = manifest["variant"]
        seed = int(manifest["seed"])
        agent = manifest.get("agent")

        if family in {"ppo_assets", "vae_assets"}:
            continue
        if family == "baseline_ablation" and variant == "ppo_vs_zero_random":
            for section_name, row_variant, row_agent in (
                ("zero_summary", "zero_action", "zero"),
                ("random_summary", "random_action", "random"),
                ("ppo_summary", "ppo", "ppo"),
            ):
                section = summary.get(section_name)
                if isinstance(section, dict):
                    row = row_from_section(family, row_variant, seed, row_agent, summary, section)
                    if row is not None:
                        rows.append(row)
            continue
        if family == "baseline_ablation" and variant == "cmaes":
            section = summary.get("cmaes_summary")
            if isinstance(section, dict):
                row = row_from_section(family, "cmaes", seed, "cmaes", summary, section)
                if row is not None:
                    rows.append(row)
            continue
        if family == "baseline_ablation" and variant == "sac_best":
            if summary.get("status") != "complete":
                rows.append(
                    {
                        "family": family,
                        "variant": "sac_best",
                        "seed": seed,
                        "agent": "sac",
                        "status": str(summary.get("status", "missing")),
                    }
                )
                continue
            section = summary.get("sac_summary")
            if isinstance(section, dict):
                row = row_from_section(family, "sac_best", seed, "sac", summary, section)
                if row is not None:
                    rows.append(row)
            continue

        if agent in {"ppo", "sac"}:
            section = summary.get(f"{agent}_summary")
            if isinstance(section, dict):
                row = row_from_section(family, variant, seed, agent, summary, section)
                if row is not None:
                    rows.append(row)
    return rows


def aggregate_results(args: argparse.Namespace) -> None:
    rows = collect_rows(args)
    args.output_root.mkdir(parents=True, exist_ok=True)
    json_path = args.output_root / "suite_summary.json"
    csv_path = args.output_root / "suite_summary.csv"
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    fieldnames = sorted({key for row in rows for key in row}) if rows else ["family", "variant", "seed", "agent"]
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    numeric_fields = sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if key not in {"family", "variant", "seed", "agent", "status"}
            and isinstance(value, (int, float))
            and math.isfinite(float(value))
        }
    )
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["family"], row["variant"], row["agent"]), []).append(row)
    aggregate_rows = []
    for (family, variant, agent), group_rows in sorted(grouped.items()):
        aggregate_row: dict[str, Any] = {
            "family": family,
            "variant": variant,
            "agent": agent,
            "seed_count": len(group_rows),
            "status": "complete" if all(row.get("status") == "complete" for row in group_rows) else "incomplete",
        }
        for field in numeric_fields:
            values = [
                float(row[field])
                for row in group_rows
                if isinstance(row.get(field), (int, float)) and math.isfinite(float(row[field]))
            ]
            aggregate_row[f"{field}_mean"] = statistics.fmean(values) if values else None
            aggregate_row[f"{field}_std"] = statistics.pstdev(values) if len(values) > 1 else 0.0
        aggregate_rows.append(aggregate_row)

    aggregate_json_path = args.output_root / "suite_aggregate.json"
    aggregate_csv_path = args.output_root / "suite_aggregate.csv"
    aggregate_json_path.write_text(json.dumps(aggregate_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    aggregate_fields = sorted({key for row in aggregate_rows for key in row}) if aggregate_rows else [
        "family",
        "variant",
        "agent",
    ]
    with aggregate_csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=aggregate_fields)
        writer.writeheader()
        writer.writerows(aggregate_rows)
    plot_suite_comparisons(aggregate_rows, args.output_root)
    print(f"[Summary] {csv_path}")
    print(f"[Aggregate] {aggregate_csv_path}")


def plot_suite_comparisons(aggregate_rows: list[dict[str, Any]], output_root: Path) -> None:
    plot_family_comparison(
        [row for row in aggregate_rows if row["family"] in {"ppo_ablation", "v1_baseline"} and row["agent"] == "ppo"],
        output_root / "suite_comparison_ppo.png",
        "PPO ablation comparison",
    )
    plot_family_comparison(
        [row for row in aggregate_rows if row["family"] in {"sac_screen", "sac_long", "v1_baseline"} and row["agent"] == "sac"],
        output_root / "suite_comparison_sac.png",
        "SAC tuning comparison",
    )
    plot_family_comparison(
        [row for row in aggregate_rows if row["family"] == "vae_ablation"],
        output_root / "suite_comparison_vae.png",
        "VAE representation ablation",
    )
    plot_family_comparison(
        [row for row in aggregate_rows if row["family"] == "reward_ablation"],
        output_root / "suite_comparison_reward.png",
        "Reward shaping ablation",
    )
    plot_family_comparison(
        [row for row in aggregate_rows if row["family"] == "baseline_ablation"],
        output_root / "suite_comparison_baseline.png",
        "Traditional and policy baseline comparison",
    )


def plot_family_comparison(rows: list[dict[str, Any]], output_path: Path, title: str) -> None:
    if not rows:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [f"{row['variant']}" if row["agent"] in {"ppo", "sac"} else f"{row['variant']}({row['agent']})" for row in rows]
    final_means = [float(row.get("final_total_score_mean") or 0.0) for row in rows]
    final_stds = [float(row.get("final_total_score_std") or 0.0) for row in rows]
    gain_means = [float(row.get("total_score_gain_mean") or 0.0) for row in rows]
    gain_stds = [float(row.get("total_score_gain_std") or 0.0) for row in rows]
    fig, axes = plt.subplots(2, 1, figsize=(max(10, len(rows) * 1.55), 8))
    positions = list(range(len(rows)))
    axes[0].bar(positions, final_means, yerr=final_stds, capsize=4, color="#177e89")
    axes[0].set_ylabel("final total score")
    axes[0].set_title(title)
    axes[1].bar(positions, gain_means, yerr=gain_stds, capsize=4, color="#db7c26")
    axes[1].set_ylabel("total score gain")
    for axis in axes:
        axis.set_xticks(positions, labels, rotation=25, ha="right")
        axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def export_plots(args: argparse.Namespace) -> None:
    script = args.workspace / "export_tensorboard_plots.py"
    if args.dry_run or not script.exists():
        return
    subprocess.run(
        [
            args.python_exe,
            str(script),
            "--run-dir",
            str(args.output_root),
            "--recursive",
            "--source",
            "csv",
            "--max-individual-plots",
            "24",
        ],
        cwd=str(args.workspace),
        check=True,
    )


def main() -> int:
    args = parse_args()
    apply_fast_dev(args)
    args.workspace = args.workspace.resolve()
    args.baseline_assets = resolve_under_workspace(args.baseline_assets, args.workspace)
    args.output_root = resolve_under_workspace(args.output_root, args.workspace)
    args.best_trial = resolve_under_workspace(args.best_trial, args.workspace)
    if not (args.workspace / "formal_experiment.py").exists():
        raise FileNotFoundError(f"formal_experiment.py not found under {args.workspace}")
    if not args.best_trial.exists() and not args.dry_run:
        raise FileNotFoundError(f"Optuna best trial not found: {args.best_trial}")
    ensure_baseline_assets(args)

    if args.stage in {"sac-screen", "all"}:
        run_sac_screen(args)
    if args.stage in {"sac-long", "all"}:
        run_sac_long(args)
    if args.stage in {"ppo-ablation", "all"}:
        run_ppo_ablations(args)
    if args.stage in {"vae-ablation", "all"}:
        run_vae_ablations(args)
    if args.stage in {"reward-ablation", "all"}:
        run_reward_ablations(args)
    if args.stage in {"baseline-ablation", "all"}:
        run_baseline_ablations(args)
    if not args.dry_run:
        aggregate_results(args)
        export_plots(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
