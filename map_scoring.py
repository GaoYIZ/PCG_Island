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

    def as_dict(self) -> Dict[str, float]:
        data = {
            "total_score": self.total_score,
            "structure_score": self.structure_score,
            "path_score": self.path_score,
            "land_score": self.land_score,
            "novelty_score": self.novelty_score,
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
        novelty_scale: float = 0.35,
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
                "minimum": 0.03,
                "ideal_low": 0.08,
                "ideal_high": 0.18,
                "maximum": 0.28,
            },
            "land_ratio": {
                "minimum": 0.08,
                "ideal_low": 0.18,
                "ideal_high": 0.42,
                "maximum": 0.62,
            },
        }
        self.structure_weights = {
            "connectivity": 0.30,
            "navigable_ratio": 0.25,
            "coast_complexity": 0.20,
            "terrain_variance": 0.25,
        }
        self.total_weights = {
            "structure": 0.45,
            "path": 0.20,
            "land": 0.25,
            "novelty": 0.10,
        }
        self.novelty_scale = float(novelty_scale)

    @staticmethod
    def _band_score(
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
            return float((value - minimum) / max(ideal_low - minimum, 1e-8))
        if maximum <= ideal_high + 1e-8:
            return 1.0
        return float((maximum - value) / max(maximum - ideal_high, 1e-8))

    def describe(self) -> Dict[str, object]:
        return {
            "preferred_ranges": self.preferred_ranges,
            "structure_weights": self.structure_weights,
            "total_weights": self.total_weights,
            "novelty_scale": self.novelty_scale,
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

        min_distance = min(float(np.linalg.norm(feature_vector - previous)) for previous in history)
        scale = max(self.novelty_scale * np.sqrt(feature_vector.size), 1e-6)
        return float(np.clip(min_distance / scale, 0.0, 1.0))

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
