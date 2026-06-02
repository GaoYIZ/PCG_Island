"""
Export TensorBoard/RL scalar curves to PNG figures and CSV.

Typical usage:
    python export_tensorboard_plots.py --run-dir formal_rl_from_3k_v3_voronoi_drop_connectivity_tuned_v4

The script first tries TensorBoard event files. If TensorBoard is not installed,
it falls back to scalar CSV files and then PPO/SAC training JSON summaries.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class ScalarPoint:
    agent: str
    step: int
    tag: str
    value: float
    source: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export TensorBoard scalar curves as PNG figures.")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("formal_rl_from_3k_v3_voronoi_drop_connectivity_tuned_v4"),
        help="RL output directory containing tensorboard logs and/or PPO/SAC summaries.",
    )
    parser.add_argument(
        "--tensorboard-dir",
        type=Path,
        default=None,
        help="TensorBoard directory. Defaults to <run-dir>/tensorboard.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <run-dir>/tensorboard_plots.",
    )
    parser.add_argument(
        "--source",
        choices=("auto", "event", "csv", "json"),
        default="auto",
        help="Scalar source. auto tries event, then CSV, then JSON.",
    )
    parser.add_argument("--agents", nargs="+", default=["ppo", "sac"], help="Agents to export.")
    parser.add_argument("--smooth-window", type=int, default=25, help="Moving-average window for dark trend line.")
    parser.add_argument("--dpi", type=int, default=160, help="PNG resolution.")
    parser.add_argument(
        "--max-individual-plots",
        type=int,
        default=0,
        help="Maximum individual curve PNGs to save per agent; 0 means save all.",
    )
    return parser.parse_args()


def finite_float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def normalize_tag(agent: str, tag: str) -> str:
    tag = str(tag).replace("\\", "/").strip("/")
    if tag.startswith(f"{agent}/"):
        return tag
    return f"{agent}/{tag}"


def infer_agent(default_agent: str, tag: str, agents: Iterable[str]) -> str:
    tag = tag.replace("\\", "/").strip("/")
    for agent in agents:
        if tag == agent or tag.startswith(f"{agent}/"):
            return agent
    return default_agent


def sanitize_filename(text: str, max_len: int = 140) -> str:
    text = text.replace("\\", "/").strip("/")
    text = re.sub(r"[^A-Za-z0-9._/-]+", "_", text)
    text = text.replace("/", "__").strip("._")
    return (text or "scalar")[:max_len]


def dedupe_and_sort(points: Iterable[ScalarPoint]) -> list[ScalarPoint]:
    latest: dict[tuple[str, str, int], ScalarPoint] = {}
    for point in points:
        latest[(point.agent, point.tag, point.step)] = point
    return sorted(latest.values(), key=lambda item: (item.agent, item.tag, item.step))


def load_event_scalars(tensorboard_dir: Path, agents: list[str]) -> list[ScalarPoint]:
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except Exception as exc:
        print(f"[Info] TensorBoard event reader unavailable ({exc}); trying fallback sources.")
        return []

    points: list[ScalarPoint] = []
    for agent in agents:
        candidates = [tensorboard_dir / agent]
        if not candidates[0].exists():
            candidates.append(tensorboard_dir)

        for log_dir in candidates:
            event_files = list(log_dir.glob("events.out.tfevents*"))
            if not event_files:
                continue
            accumulator = EventAccumulator(str(log_dir), size_guidance={"scalars": 0})
            accumulator.Reload()
            for raw_tag in accumulator.Tags().get("scalars", []):
                inferred_agent = infer_agent(agent, raw_tag, agents)
                tag = normalize_tag(inferred_agent, raw_tag)
                for event in accumulator.Scalars(raw_tag):
                    value = finite_float(event.value)
                    if value is not None:
                        points.append(
                            ScalarPoint(
                                agent=inferred_agent,
                                step=int(event.step),
                                tag=tag,
                                value=value,
                                source="event",
                            )
                        )
            break
    return dedupe_and_sort(points)


def load_csv_scalars(tensorboard_dir: Path, agents: list[str]) -> list[ScalarPoint]:
    points: list[ScalarPoint] = []
    for agent in agents:
        path = tensorboard_dir / f"{agent}_scalars.csv"
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)
            for row in reader:
                value = finite_float(row.get("value"))
                if value is None:
                    continue
                try:
                    step = int(float(row.get("step", 0)))
                except (TypeError, ValueError):
                    continue
                raw_tag = str(row.get("tag", "scalar"))
                inferred_agent = infer_agent(agent, raw_tag, agents)
                points.append(
                    ScalarPoint(
                        agent=inferred_agent,
                        step=step,
                        tag=normalize_tag(inferred_agent, raw_tag),
                        value=value,
                        source="csv",
                    )
                )
    return dedupe_and_sort(points)


def load_json_scalars(run_dir: Path, agents: list[str]) -> list[ScalarPoint]:
    try:
        from export_rl_tensorboard import episode_scalars, load_episodes
    except Exception as exc:
        print(f"[Info] JSON summary reader unavailable ({exc}).")
        return []

    points: list[ScalarPoint] = []
    for agent in agents:
        for index, episode in enumerate(load_episodes(run_dir, agent), start=1):
            step = int(episode.get("episode", index))
            for raw_tag, raw_value in episode_scalars(agent, episode).items():
                value = finite_float(raw_value)
                if value is None:
                    continue
                inferred_agent = infer_agent(agent, raw_tag, agents)
                points.append(
                    ScalarPoint(
                        agent=inferred_agent,
                        step=step,
                        tag=normalize_tag(inferred_agent, raw_tag),
                        value=value,
                        source="json",
                    )
                )
    return dedupe_and_sort(points)


def load_scalars(args: argparse.Namespace, run_dir: Path, tensorboard_dir: Path) -> list[ScalarPoint]:
    agents = [str(agent).lower() for agent in args.agents]
    if args.source in ("auto", "event"):
        points = load_event_scalars(tensorboard_dir, agents)
        if points or args.source == "event":
            return points
    if args.source in ("auto", "csv"):
        points = load_csv_scalars(tensorboard_dir, agents)
        if points or args.source == "csv":
            return points
    if args.source in ("auto", "json"):
        points = load_json_scalars(run_dir, agents)
        if points or args.source == "json":
            return points
    return []


def write_scalars_csv(points: list[ScalarPoint], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "all_scalars.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["agent", "step", "tag", "value", "source"])
        for point in points:
            writer.writerow([point.agent, point.step, point.tag, point.value, point.source])
    return csv_path


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or values.size == 0:
        return values
    trend = np.empty_like(values, dtype=np.float64)
    for index in range(values.size):
        start = max(0, index - window + 1)
        trend[index] = float(np.mean(values[start : index + 1]))
    return trend


def series_by_agent_tag(points: list[ScalarPoint]) -> dict[tuple[str, str], tuple[np.ndarray, np.ndarray]]:
    grouped: dict[tuple[str, str], list[tuple[int, float]]] = defaultdict(list)
    for point in points:
        grouped[(point.agent, point.tag)].append((point.step, point.value))

    series: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    for key, rows in grouped.items():
        rows.sort(key=lambda item: item[0])
        steps = np.asarray([row[0] for row in rows], dtype=np.int64)
        values = np.asarray([row[1] for row in rows], dtype=np.float64)
        series[key] = (steps, values)
    return series


def plot_one_curve(
    steps: np.ndarray,
    values: np.ndarray,
    title: str,
    output_path: Path,
    smooth_window: int,
    dpi: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(9.5, 4.8))
    if values.size > 1:
        axis.plot(steps, values, color="#9ad7e8", linewidth=0.9, alpha=0.35, label="raw")
        trend = moving_average(values, smooth_window)
        axis.plot(steps, trend, color="#045a70", linewidth=2.1, label=f"moving avg {smooth_window}")
        axis.legend(loc="best", fontsize=8)
    else:
        axis.scatter(steps, values, color="#045a70", s=24)
    axis.set_title(title)
    axis.set_xlabel("episode")
    axis.set_ylabel("value")
    axis.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def export_individual_plots(
    series: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]],
    output_dir: Path,
    smooth_window: int,
    dpi: int,
    max_per_agent: int,
) -> list[Path]:
    paths: list[Path] = []
    counts: dict[str, int] = defaultdict(int)
    for (agent, tag), (steps, values) in sorted(series.items()):
        if max_per_agent > 0 and counts[agent] >= max_per_agent:
            continue
        short_tag = tag[len(agent) + 1 :] if tag.startswith(f"{agent}/") else tag
        path = output_dir / "curves" / agent / f"{sanitize_filename(short_tag)}.png"
        plot_one_curve(steps, values, f"{agent.upper()} {short_tag}", path, smooth_window, dpi)
        paths.append(path)
        counts[agent] += 1
    return paths


def plot_multi_axis(
    axis: plt.Axes,
    series: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]],
    agent: str,
    tags: list[str],
    title: str,
    smooth_window: int,
) -> None:
    plotted = 0
    for short_tag in tags:
        tag = normalize_tag(agent, short_tag)
        values = series.get((agent, tag))
        if values is None:
            continue
        steps, raw_values = values
        label = short_tag.split("/")[-1]
        if raw_values.size > 1:
            axis.plot(steps, moving_average(raw_values, smooth_window), linewidth=1.8, label=label)
        else:
            axis.scatter(steps, raw_values, s=16, label=label)
        plotted += 1
    axis.set_title(title)
    axis.grid(True, alpha=0.25)
    if plotted:
        axis.legend(fontsize=7, loc="best")
    else:
        axis.text(0.5, 0.5, "No matching scalars", ha="center", va="center", transform=axis.transAxes)


def export_overview_plots(
    series: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]],
    agents: list[str],
    output_dir: Path,
    smooth_window: int,
    dpi: int,
) -> list[Path]:
    overview_groups = [
        (
            "Reward",
            ["reward", "reward_components/reward", "score_gain/total_score"],
        ),
        (
            "Loss",
            ["losses/total_loss", "losses/policy_loss", "losses/value_loss", "losses/q_loss", "losses/alpha"],
        ),
        (
            "Scores",
            [
                "score/total_score",
                "score/path_score",
                "score/connectivity_score",
                "score/coast_score",
                "score/land_score",
                "score/balance_score",
            ],
        ),
        (
            "Metrics",
            [
                "metrics/path_reachability",
                "metrics/component_count",
                "metrics/navigable_ratio",
                "metrics/coast_complexity",
                "metrics/steep_slope_ratio",
                "metrics/coast_steep_ratio",
            ],
        ),
    ]
    paths: list[Path] = []
    for agent in agents:
        fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.5))
        axes_flat = list(axes.ravel())
        for axis, (title, tags) in zip(axes_flat, overview_groups):
            plot_multi_axis(axis, series, agent, tags, title, smooth_window)
        fig.suptitle(f"{agent.upper()} TensorBoard Overview", fontsize=14)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        path = output_dir / f"{agent}_overview.png"
        fig.savefig(path, dpi=dpi)
        plt.close(fig)
        paths.append(path)
    return paths


def write_manifest(
    output_dir: Path,
    run_dir: Path,
    tensorboard_dir: Path,
    points: list[ScalarPoint],
    overview_paths: list[Path],
    individual_paths: list[Path],
    csv_path: Path,
) -> Path:
    agents = sorted({point.agent for point in points})
    tags_by_agent: dict[str, list[str]] = defaultdict(list)
    for point in points:
        if point.tag not in tags_by_agent[point.agent]:
            tags_by_agent[point.agent].append(point.tag)

    manifest = {
        "run_dir": str(run_dir),
        "tensorboard_dir": str(tensorboard_dir),
        "output_dir": str(output_dir),
        "scalar_count": len(points),
        "agents": agents,
        "tags_by_agent": {agent: sorted(tags) for agent, tags in tags_by_agent.items()},
        "csv": str(csv_path),
        "overview_pngs": [str(path) for path in overview_paths],
        "individual_png_count": len(individual_paths),
    }
    path = output_dir / "manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    tensorboard_dir = (args.tensorboard_dir or run_dir / "tensorboard").resolve()
    output_dir = (args.output_dir or run_dir / "tensorboard_plots").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    points = load_scalars(args, run_dir, tensorboard_dir)
    if not points:
        raise FileNotFoundError(
            "No scalar data found. Expected TensorBoard events, tensorboard/*_scalars.csv, "
            f"or PPO/SAC training summaries under {run_dir}."
        )

    csv_path = write_scalars_csv(points, output_dir)
    series = series_by_agent_tag(points)
    agents = [agent for agent in [str(agent).lower() for agent in args.agents] if any(key[0] == agent for key in series)]
    overview_paths = export_overview_plots(series, agents, output_dir, args.smooth_window, args.dpi)
    individual_paths = export_individual_plots(
        series=series,
        output_dir=output_dir,
        smooth_window=args.smooth_window,
        dpi=args.dpi,
        max_per_agent=max(0, int(args.max_individual_plots)),
    )
    manifest_path = write_manifest(output_dir, run_dir, tensorboard_dir, points, overview_paths, individual_paths, csv_path)

    print(f"Exported {len(points)} scalar points")
    print(f"CSV: {csv_path}")
    print(f"Overview PNGs: {len(overview_paths)}")
    for path in overview_paths:
        print(f"  {path}")
    print(f"Individual curve PNGs: {len(individual_paths)}")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
