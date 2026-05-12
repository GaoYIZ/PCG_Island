"""
Unified map scoring for offline dataset evaluation and RL rewards.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Sequence

import numpy as np


@dataclass
class ScoreBreakdown:
    total_score: float
    structure_score: float
    path_score: float
    land_score: float
    novelty_score: float
    component_scores: Dict[str, float]
    constraint_penalty: float = 0.0
    quality_bonus: float = 0.0

    def as_dict(self) -> Dict[str, float]:
        data = {
            "total_score": self.total_score,
            "structure_score": self.structure_score,
            "path_score": self.path_score,
            "land_score": self.land_score,
            "novelty_score": self.novelty_score,
            "constraint_penalty": self.constraint_penalty,
            "quality_bonus": self.quality_bonus,
        }
        data.update(self.component_scores)
        return data


class MapScorer:
    """
    Single source of truth for terrain quality.

    The same scorer is used for:
    - dataset filtering and quality auditing
    - direct map size evaluation via land ratio
    - RL reward computation
    """

    def __init__(
        self,
        preferred_ranges: Optional[Mapping[str, Mapping[str, float]]] = None,
        structure_weights: Optional[Mapping[str, float]] = None,
        total_weights: Optional[Mapping[str, float]] = None,
        novelty_scale: float = 0.35,
        novelty_k: int = 5,
        reward_delta_scale: float = 5.0,
        quality_bonus_threshold: float = 0.85,
        excellent_bonus_threshold: float = 0.92,
        quality_bonus: float = 0.50,
        excellent_bonus: float = 1.00,
        constraint_thresholds: Optional[Mapping[str, float]] = None,
        constraint_penalties: Optional[Mapping[str, float]] = None,
    ):
        self.preferred_ranges = {
            "navigable_ratio": {
                "minimum": 0.70,
                "ideal_low": 0.85,
                "ideal_high": 1.00,
                "maximum": 1.00,
            },
            "coast_complexity": {
                "minimum": 1.20,
                "ideal_low": 2.50,
                "ideal_high": 6.50,
                "maximum": 10.50,
            },
            "terrain_variance": {
                "minimum": 0.04,
                "ideal_low": 0.08,
                "ideal_high": 0.18,
                "maximum": 0.28,
            },
            "land_ratio": {
                "minimum": 0.10,
                "ideal_low": 0.18,
                "ideal_high": 0.42,
                "maximum": 0.60,
            },
        }
        self.structure_weights = {
            "connectivity": 0.40,
            "navigable_ratio": 0.30,
            "coast_complexity": 0.15,
            "terrain_variance": 0.15,
        }
        self.total_weights = {
            "structure": 0.60,
            "path": 0.25,
            "land": 0.10,
            "novelty": 0.05,
        }
        self.preferred_ranges.update({key: dict(value) for key, value in (preferred_ranges or {}).items()})
        self.structure_weights.update({key: float(value) for key, value in (structure_weights or {}).items()})
        self.total_weights.update({key: float(value) for key, value in (total_weights or {}).items()})
        self.novelty_scale = float(novelty_scale)
        self.novelty_k = int(max(1, novelty_k))
        self.reward_delta_scale = float(reward_delta_scale)
        self.quality_bonus_threshold = float(quality_bonus_threshold)
        self.excellent_bonus_threshold = float(excellent_bonus_threshold)
        self.quality_bonus_value = float(quality_bonus)
        self.excellent_bonus_value = float(excellent_bonus)
        self.constraint_thresholds = {
            "connectivity": 0.90,
            "path_reachability": 0.25,
            "land_ratio_min": 0.10,
            "land_ratio_max": 0.60,
            "navigable_ratio": 0.70,
            "terrain_variance": 0.04,
            "quality_score": 0.50,
        }
        self.constraint_penalties = {
            "connectivity": 1.25,
            "path_reachability": 1.25,
            "land_ratio": 0.75,
            "navigable_ratio": 0.50,
            "terrain_variance": 0.25,
        }
        if constraint_thresholds is not None:
            self.constraint_thresholds.update({key: float(value) for key, value in constraint_thresholds.items()})
        if constraint_penalties is not None:
            self.constraint_penalties.update({key: float(value) for key, value in constraint_penalties.items()})

    @staticmethod
    def _smoothstep(t: float) -> float:
        clipped = float(np.clip(t, 0.0, 1.0))
        return clipped * clipped * (3.0 - 2.0 * clipped)

    @classmethod
    def _band_score(
        cls,
        value: float,
        minimum: float,
        ideal_low: float,
        ideal_high: float,
        maximum: float,
    ) -> float:
        if value < minimum or value > maximum:
            return 0.0
        if ideal_low <= value <= ideal_high:
            return 1.0
        if value < ideal_low:
            if ideal_low <= minimum + 1e-8:
                return 1.0
            return cls._smoothstep((value - minimum) / max(ideal_low - minimum, 1e-8))
        if maximum <= ideal_high + 1e-8:
            return 1.0
        return cls._smoothstep((maximum - value) / max(maximum - ideal_high, 1e-8))

    def describe(self) -> Dict[str, object]:
        return {
            "preferred_ranges": self.preferred_ranges,
            "structure_weights": self.structure_weights,
            "total_weights": self.total_weights,
            "novelty_scale": self.novelty_scale,
            "novelty_k": self.novelty_k,
            "reward_delta_scale": self.reward_delta_scale,
            "quality_bonus_threshold": self.quality_bonus_threshold,
            "excellent_bonus_threshold": self.excellent_bonus_threshold,
            "quality_bonus": self.quality_bonus_value,
            "excellent_bonus": self.excellent_bonus_value,
            "constraint_thresholds": self.constraint_thresholds,
            "constraint_penalties": self.constraint_penalties,
        }

    def compute_novelty_score(
        self,
        feature_vector: Optional[Sequence[float]],
        history_vectors: Optional[Iterable[Sequence[float]]] = None,
    ) -> float:
        if feature_vector is None or history_vectors is None:
            return 0.0

        feature_vector = np.asarray(feature_vector, dtype=np.float32)
        history = [np.asarray(vector, dtype=np.float32) for vector in history_vectors]
        if len(history) == 0:
            return 0.0

        history_matrix = np.stack(history, axis=0)
        distances = np.linalg.norm(history_matrix - feature_vector[None, :], axis=1)
        positive_distances = distances[distances > 1e-6]
        if positive_distances.size == 0:
            return 0.0

        neighbor_count = min(self.novelty_k, positive_distances.size)
        nearest_distances = np.partition(positive_distances, neighbor_count - 1)[:neighbor_count]
        neighborhood_distance = float(np.mean(nearest_distances))
        adaptive_scale = max(
            float(np.median(positive_distances)),
            self.novelty_scale * float(np.sqrt(feature_vector.size)),
            1e-6,
        )
        novelty_score = 1.0 - float(np.exp(-neighborhood_distance / adaptive_scale))
        return float(np.clip(novelty_score, 0.0, 1.0))

    def score_metrics(
        self,
        metrics: Mapping[str, float],
        feature_vector: Optional[Sequence[float]] = None,
        history_vectors: Optional[Iterable[Sequence[float]]] = None,
    ) -> ScoreBreakdown:
        structure_components = {
            "connectivity": float(np.clip(metrics["connectivity"], 0.0, 1.0)),
            "navigable_ratio": self._band_score(
                metrics["navigable_ratio"],
                **self.preferred_ranges["navigable_ratio"],
            ),
            "coast_complexity": self._band_score(
                metrics["coast_complexity"],
                **self.preferred_ranges["coast_complexity"],
            ),
            "terrain_variance": self._band_score(
                metrics["terrain_variance"],
                **self.preferred_ranges["terrain_variance"],
            ),
        }

        structure_score = 0.0
        for name, weight in self.structure_weights.items():
            structure_score += weight * structure_components[name]

        path_score = float(np.clip(metrics["path_reachability"], 0.0, 1.0))
        land_score = self._band_score(
            metrics["land_ratio"],
            **self.preferred_ranges["land_ratio"],
        )
        novelty_score = self.compute_novelty_score(feature_vector, history_vectors)

        total_score = (
            self.total_weights["structure"] * structure_score
            + self.total_weights["path"] * path_score
            + self.total_weights["land"] * land_score
            + self.total_weights["novelty"] * novelty_score
        )

        component_scores = {
            "connectivity_score": structure_components["connectivity"],
            "navigable_score": structure_components["navigable_ratio"],
            "coast_score": structure_components["coast_complexity"],
            "variance_score": structure_components["terrain_variance"],
        }
        return ScoreBreakdown(
            total_score=float(total_score),
            structure_score=float(structure_score),
            path_score=path_score,
            land_score=float(land_score),
            novelty_score=float(novelty_score),
            component_scores=component_scores,
        )

    def compute_constraint_penalty(self, metrics: Mapping[str, float]) -> float:
        penalty = 0.0
        if float(metrics["connectivity"]) < self.constraint_thresholds["connectivity"]:
            penalty += self.constraint_penalties["connectivity"]
        if float(metrics["path_reachability"]) < self.constraint_thresholds["path_reachability"]:
            penalty += self.constraint_penalties["path_reachability"]
        land_ratio = float(metrics["land_ratio"])
        if (
            land_ratio < self.constraint_thresholds["land_ratio_min"]
            or land_ratio > self.constraint_thresholds["land_ratio_max"]
        ):
            penalty += self.constraint_penalties["land_ratio"]
        if float(metrics["navigable_ratio"]) < self.constraint_thresholds["navigable_ratio"]:
            penalty += self.constraint_penalties["navigable_ratio"]
        if float(metrics["terrain_variance"]) < self.constraint_thresholds["terrain_variance"]:
            penalty += self.constraint_penalties["terrain_variance"]
        return float(penalty)

    def compute_quality_bonus(self, total_score: float) -> float:
        bonus = 0.0
        if total_score >= self.quality_bonus_threshold:
            bonus += self.quality_bonus_value
        if total_score >= self.excellent_bonus_threshold:
            bonus += self.excellent_bonus_value
        return float(bonus)

    def transition_reward(
        self,
        previous_score: ScoreBreakdown,
        current_score: ScoreBreakdown,
        metrics: Mapping[str, float],
    ) -> tuple[float, Dict[str, float]]:
        delta_quality = float(current_score.total_score - previous_score.total_score)
        penalty = self.compute_constraint_penalty(metrics)
        bonus = self.compute_quality_bonus(current_score.total_score)
        reward = self.reward_delta_scale * delta_quality - penalty + bonus
        reward_terms = {
            "delta_quality": delta_quality,
            "reward_delta_scale": self.reward_delta_scale,
            "constraint_penalty": penalty,
            "quality_bonus": bonus,
            "previous_total_score": float(previous_score.total_score),
            "current_total_score": float(current_score.total_score),
            "shaped_reward": float(reward),
        }
        current_score.constraint_penalty = penalty
        current_score.quality_bonus = bonus
        return float(reward), reward_terms
