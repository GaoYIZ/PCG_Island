"""Run the restored-v1 SAC tuning and PPO ablation suite."""

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


V1_REWARD_ARGS = [
    "--reward-current-scale",
    "0.3",
    "--reward-delta-scale",
    "3.0",
    "--reward-best-scale",
    "2.0",
    "--reward-step-penalty",
    "0.01",
    "--reward-success-bonus",
    "1.0",
    "--reward-success-threshold",
    "0.72",
    "--reward-success-gain-threshold",
    "0.03",
    "--reward-failure-threshold",
    "0.10",
    "--reward-success-streak",
    "3",
    "--reward-stagnation-patience",
    "5",
    "--reward-stagnation-delta",
    "0.001",
]

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="One-command v1 SAC tuning and PPO ablation suite")
    parser.add_argument(
        "--stage",
        choices=["sac-screen", "sac-long", "ppo-ablation", "all"],
        default="all",
    )
    parser.add_argument("--workspace", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--python-exe", type=str, default=sys.executable)
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
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--screen-episodes", type=int, default=1500)
    parser.add_argument("--long-episodes", type=int, default=4000)
    parser.add_argument("--ppo-episodes", type=int, default=1500)
    parser.add_argument("--eval-islands", type=int, default=64)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def render_command(command: list[str]) -> str:
    return subprocess.list2cmdline(command)


def resolve_under_workspace(path: Path, workspace: Path) -> Path:
    return path.resolve() if path.is_absolute() else (workspace / path).resolve()


def is_complete(output_dir: Path, agent: str | None = None) -> bool:
    if not (output_dir / "final_summary.json").exists():
        return False
    if agent is None:
        return (output_dir / "rl_assets_ready.json").exists()
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
    manifest = {
        "family": family,
        "variant": variant,
        "seed": int(seed),
        "agent": agent,
        "parameters": parameters,
        "command": command,
    }
    if args.resume and is_complete(output_dir, agent):
        manifest_path = output_dir / "suite_run.json"
        if not args.dry_run and not manifest_path.exists():
            manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[Skip] Completed: {output_dir}")
        return
    print(f"\n[Run] {render_command(command)}")
    if args.dry_run:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(command, cwd=str(args.workspace), check=True)
    (output_dir / "suite_run.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def common_formal_args(args: argparse.Namespace, output_dir: Path, latent_dim: int, seed: int) -> list[str]:
    return [
        args.python_exe,
        str(args.workspace / "formal_experiment.py"),
        "--formal-rl",
        "--output-dir",
        str(output_dir),
        "--map-size",
        "64",
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
        *V1_REWARD_ARGS,
    ]


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
    (args.output_root / "sac_selection.json").write_text(
        json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[Select] SAC long variants: {', '.join(selected)}")
    return selected


def run_sac_long(args: argparse.Namespace) -> None:
    if args.dry_run and not all(
        (args.output_root / "sac_screen" / name / "sac_training_summary.json").exists()
        for name in SAC_SCREEN_VARIANTS
    ):
        print("[DryRun] SAC long commands are selected from completed screen results.")
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
    command.extend(
        [
            "--prepare-rl-assets-only",
            "--dataset-samples",
            "1000",
            "--min-clean-samples",
            "3000",
            "--max-dataset-samples",
            "12000",
            "--vae-epochs",
            "40",
            "--optuna-best-trial",
            str(args.best_trial),
            "--preserve-latent-dim",
            "--drop-connectivity-supervision",
        ]
    )
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
    command.extend(
        [
            "--prepare-rl-assets-only",
            "--dataset-samples",
            "1000",
            "--min-clean-samples",
            "3000",
            "--max-dataset-samples",
            "12000",
            "--vae-epochs",
            "40",
            "--optuna-best-trial",
            str(args.best_trial),
            "--preserve-latent-dim",
            "--drop-connectivity-supervision",
        ]
    )
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


def score_section(summary: dict[str, Any], agent: str) -> dict[str, Any] | None:
    section = summary.get(f"{agent}_summary")
    if not isinstance(section, dict):
        return None
    final_score = section.get("final_score_mean", {})
    gain = section.get("score_gain_mean", {})
    return {
        "final_total_score": final_score.get("total_score"),
        "total_score_gain": gain.get("total_score"),
        "final_path_score": final_score.get("path_score"),
        "final_connectivity_score": final_score.get("connectivity_score"),
        "final_navigable_score": final_score.get("navigable_score"),
        "final_coast_score": final_score.get("coast_score"),
        "final_land_score": final_score.get("land_score"),
        "final_variance_score": final_score.get("variance_score"),
    }


def aggregate_results(args: argparse.Namespace) -> None:
    rows: list[dict[str, Any]] = []
    baseline_summary_path = args.baseline_assets / "final_summary.json"
    if baseline_summary_path.exists():
        summary = json.loads(baseline_summary_path.read_text(encoding="utf-8"))
        for agent in ("ppo", "sac"):
            scores = score_section(summary, agent)
            if scores is not None:
                rows.append(
                    {"family": "v1_baseline", "variant": "archived_v1", "seed": 42, "agent": agent, **scores}
                )

    for manifest_path in args.output_root.rglob("suite_run.json"):
        run_dir = manifest_path.parent
        summary_path = run_dir / "final_summary.json"
        if not summary_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        agent = manifest.get("agent")
        if agent not in {"ppo", "sac"}:
            continue
        scores = score_section(summary, agent)
        if scores is None:
            continue
        rows.append(
            {
                "family": manifest["family"],
                "variant": manifest["variant"],
                "seed": manifest["seed"],
                "agent": agent,
                **scores,
            }
        )

    args.output_root.mkdir(parents=True, exist_ok=True)
    json_path = args.output_root / "suite_summary.json"
    csv_path = args.output_root / "suite_summary.csv"
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    fieldnames = list(rows[0].keys()) if rows else ["family", "variant", "seed", "agent"]
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    numeric_fields = [
        "final_total_score",
        "total_score_gain",
        "final_path_score",
        "final_connectivity_score",
        "final_navigable_score",
        "final_coast_score",
        "final_land_score",
        "final_variance_score",
    ]
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
        }
        for field in numeric_fields:
            values = [float(row[field]) for row in group_rows if row.get(field) is not None]
            aggregate_row[f"{field}_mean"] = statistics.fmean(values) if values else None
            aggregate_row[f"{field}_std"] = statistics.pstdev(values) if len(values) > 1 else 0.0
        aggregate_rows.append(aggregate_row)

    aggregate_json_path = args.output_root / "suite_aggregate.json"
    aggregate_csv_path = args.output_root / "suite_aggregate.csv"
    aggregate_json_path.write_text(json.dumps(aggregate_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    aggregate_fields = list(aggregate_rows[0].keys()) if aggregate_rows else ["family", "variant", "agent"]
    with aggregate_csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=aggregate_fields)
        writer.writeheader()
        writer.writerows(aggregate_rows)
    plot_suite_comparison(aggregate_rows, args.output_root / "suite_comparison.png")
    print(f"[Summary] {csv_path}")
    print(f"[Aggregate] {aggregate_csv_path}")


def plot_suite_comparison(aggregate_rows: list[dict[str, Any]], output_path: Path) -> None:
    if not aggregate_rows:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for agent in ("ppo", "sac"):
        rows = [row for row in aggregate_rows if row["agent"] == agent]
        if not rows:
            continue
        labels = [row["variant"] for row in rows]
        final_means = [float(row["final_total_score_mean"] or 0.0) for row in rows]
        final_stds = [float(row["final_total_score_std"] or 0.0) for row in rows]
        gain_means = [float(row["total_score_gain_mean"] or 0.0) for row in rows]
        gain_stds = [float(row["total_score_gain_std"] or 0.0) for row in rows]
        fig, axes = plt.subplots(2, 1, figsize=(max(10, len(rows) * 1.6), 8))
        positions = list(range(len(rows)))
        axes[0].bar(positions, final_means, yerr=final_stds, capsize=4, color="#177e89")
        axes[0].set_ylabel("final total score")
        axes[0].set_title(f"{agent.upper()} variant comparison")
        axes[1].bar(positions, gain_means, yerr=gain_stds, capsize=4, color="#db7c26")
        axes[1].set_ylabel("total score gain")
        for axis in axes:
            axis.set_xticks(positions, labels, rotation=25, ha="right")
            axis.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        agent_path = output_path.with_name(f"{output_path.stem}_{agent}{output_path.suffix}")
        fig.savefig(agent_path, dpi=160)
        plt.close(fig)


def export_plots(args: argparse.Namespace) -> None:
    script = args.workspace / "export_tensorboard_plots.py"
    if args.dry_run or not script.exists():
        return
    subprocess.run(
        [args.python_exe, str(script), "--run-dir", str(args.output_root), "--recursive"],
        cwd=str(args.workspace),
        check=True,
    )


def main() -> int:
    args = parse_args()
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
    if not args.dry_run:
        aggregate_results(args)
        export_plots(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
