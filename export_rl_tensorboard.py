"""
Export RL training JSON logs to CSV and TensorBoard event files.

Usage:
    python export_rl_tensorboard.py --run-dir formal_rl_from_3k_v3_voronoi_drop_connectivity
    tensorboard --logdir formal_rl_from_3k_v3_voronoi_drop_connectivity/tensorboard_export
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert PPO/SAC JSON summaries into TensorBoard-friendly logs.")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("formal_rl_from_3k_v3_voronoi_drop_connectivity"),
        help="Directory containing ppo_training_summary.json and sac_training_summary.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <run-dir>/tensorboard_export.",
    )
    return parser.parse_args()


def scalar(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def flatten(prefix: str, value: Any, out: dict[str, float]) -> None:
    direct = scalar(value)
    if direct is not None:
        out[prefix] = direct
        return
    if isinstance(value, dict):
        for key, child in value.items():
            flatten(f"{prefix}/{key}" if prefix else str(key), child, out)


def load_episodes(run_dir: Path, agent: str) -> list[dict[str, Any]]:
    path = run_dir / f"{agent}_training_summary.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return list(data.get("episodes", []))


def episode_scalars(agent: str, episode: dict[str, Any]) -> dict[str, float]:
    values: dict[str, float] = {}
    for key in (
        "reward",
        "score_gain",
        "total_env_steps",
        "positive_delta_steps",
        "best_improve_steps",
        "episode_steps",
    ):
        if key in episode:
            direct = scalar(episode[key])
            if direct is not None:
                values[f"{agent}/{key}"] = direct

    for key in ("initial_score", "score", "reward_components", "episode_component_sums", "losses"):
        if key in episode:
            flatten(f"{agent}/{key}", episode[key], values)

    done_reason = episode.get("done_reason")
    if done_reason is not None:
        for reason in ("success_threshold", "stagnation", "max_steps", "failure_threshold"):
            values[f"{agent}/done_reason/{reason}"] = 1.0 if done_reason == reason else 0.0

    return values


def write_csv(rows: list[tuple[str, int, str, float]], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "rl_scalars.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["agent", "episode", "tag", "value"])
        writer.writerows(rows)
    return csv_path


def write_tensorboard(rows: list[tuple[str, int, str, float]], output_dir: Path) -> bool:
    try:
        from torch.utils.tensorboard import SummaryWriter
    except Exception:
        return False

    writers: dict[str, SummaryWriter] = {}
    try:
        for agent, episode, tag, value in rows:
            writer = writers.get(agent)
            if writer is None:
                writer = SummaryWriter(log_dir=str(output_dir / agent))
                writers[agent] = writer
            writer.add_scalar(tag, value, episode)
    finally:
        for writer in writers.values():
            writer.close()
    return True


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    output_dir = (args.output_dir or run_dir / "tensorboard_export").resolve()

    rows: list[tuple[str, int, str, float]] = []
    for agent in ("ppo", "sac"):
        for index, episode in enumerate(load_episodes(run_dir, agent), start=1):
            step = int(episode.get("episode", index))
            for tag, value in sorted(episode_scalars(agent, episode).items()):
                rows.append((agent, step, tag, value))

    if not rows:
        raise FileNotFoundError(f"No PPO/SAC training episodes found under {run_dir}")

    csv_path = write_csv(rows, output_dir)
    tensorboard_written = write_tensorboard(rows, output_dir)

    print(f"Exported {len(rows)} scalar rows")
    print(f"CSV: {csv_path}")
    if tensorboard_written:
        print(f"TensorBoard logs: {output_dir}")
        print(f"Run: tensorboard --logdir {output_dir}")
    else:
        print("TensorBoard package is not installed; CSV export is ready.")
        print("Install when needed: pip install tensorboard")
        print(f"Then rerun this script and open: tensorboard --logdir {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
