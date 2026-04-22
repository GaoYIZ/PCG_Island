"""
Chinese reporting helpers for IslandTest experiment outputs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence

import numpy as np


METRIC_LABELS_ZH: Dict[str, str] = {
    "connectivity": "连通性",
    "navigable_ratio": "可导航比例",
    "coast_complexity": "海岸复杂度",
    "terrain_variance": "地形方差",
    "path_reachability": "路径可达性",
    "land_ratio": "陆地占比",
    "total_score": "总评分",
    "structure_score": "结构评分",
    "path_score": "路径评分",
    "land_score": "地图大小评分",
    "novelty_score": "新颖性评分",
    "connectivity_score": "连通性子评分",
    "navigable_score": "可导航子评分",
    "coast_score": "海岸复杂度子评分",
    "variance_score": "地形方差子评分",
}


REJECTION_LABELS_ZH: Dict[str, str] = {
    "too_little_land": "陆地面积过小",
    "too_much_land": "陆地面积过大",
    "poor_reachability": "路径可达性不足",
    "weak_connectivity": "连通性较弱",
    "too_flat": "地形过于平坦",
    "low_quality_score": "综合质量分过低",
    "near_duplicate_metrics": "结构指标近似重复",
}


def metric_label(name: str) -> str:
    return METRIC_LABELS_ZH.get(name, name)


def rejection_label(name: str) -> str:
    return REJECTION_LABELS_ZH.get(name, name)


def print_section(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def print_metric_dict(metrics: Mapping[str, float], precision: int = 4) -> None:
    for key, value in metrics.items():
        print(f"{metric_label(key):<18}: {value:.{precision}f}")


def print_distribution_summary(metric_matrix: np.ndarray, metric_names: Sequence[str]) -> None:
    for idx, name in enumerate(metric_names):
        values = metric_matrix[:, idx]
        print(
            f"{metric_label(name):<18}: "
            f"均值={values.mean():.4f}  标准差={values.std():.4f}  "
            f"最小值={values.min():.4f}  最大值={values.max():.4f}"
        )


def print_dataset_summary(summary: Mapping[str, object]) -> None:
    print(f"原始样本数          : {summary['num_samples']}")
    print(f"有效样本数          : {summary['num_valid']}")
    if "valid_ratio" in summary:
        print(f"有效率             : {summary['valid_ratio']:.4f}")
    print(f"平均质量分          : {summary['quality_score_mean']:.4f}")
    print(f"质量分标准差        : {summary['quality_score_std']:.4f}")

    metric_stats = summary.get("metric_stats", {})
    if metric_stats:
        print("\n指标分布统计:")
        for key, stats in metric_stats.items():
            print(
                f"{metric_label(key):<18}: "
                f"均值={stats['mean']:.4f}  标准差={stats['std']:.4f}  "
                f"最小值={stats['min']:.4f}  最大值={stats['max']:.4f}"
            )

    rejection_histogram = summary.get("rejection_histogram", {})
    if rejection_histogram:
        print("\n被过滤原因统计:")
        for key, value in rejection_histogram.items():
            print(f"{rejection_label(key):<18}: {value}")


def save_json(data: Mapping[str, object], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def summarize_rewards(rewards: Iterable[float]) -> Dict[str, float]:
    rewards = np.asarray(list(rewards), dtype=np.float32)
    if rewards.size == 0:
        return {"mean": 0.0, "std": 0.0, "max": 0.0, "min": 0.0}
    return {
        "mean": float(rewards.mean()),
        "std": float(rewards.std()),
        "max": float(rewards.max()),
        "min": float(rewards.min()),
    }
