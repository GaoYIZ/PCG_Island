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

    The score follows a compact reward definition:
    R = sum(w_i * R_i) + lambda * R_novelty.
    """

    def __init__(
        self,
        novelty_scale: float = 0.35,
        novelty_k: int = 1,
        connectivity_alpha: float = 0.35,
        navigable_mu: float = 0.90,
        navigable_sigma: float = 0.18,
        coast_mu: float = 3.80,
        coast_sigma: float = 1.70,
        variance_mu: float = 0.13,
        variance_sigma: float = 0.08,
        land_mu: float = 0.23,
        land_sigma: float = 0.11,
    ):
        self.connectivity_alpha = float(connectivity_alpha)
        self.gaussian_targets = {
            "navigable_ratio": {"mu": float(navigable_mu), "sigma": float(navigable_sigma)},
            "coast_complexity": {"mu": float(coast_mu), "sigma": float(coast_sigma)},
            "terrain_variance": {"mu": float(variance_mu), "sigma": float(variance_sigma)},
            "land_ratio": {"mu": float(land_mu), "sigma": float(land_sigma)},
        }
        self.total_weights = {
            "connectivity": 0.28,
            "path": 0.28,
            "navigable": 0.10,
            "coast": 0.16,
            "variance": 0.06,
            "land": 0.08,
            "novelty": 0.02,
        }
        self.novelty_scale = float(novelty_scale)
        self.novelty_k = int(max(1, novelty_k))

    @staticmethod
    def _gaussian_score(value: float, mu: float, sigma: float) -> float:
        sigma = max(float(sigma), 1e-8)
        score = np.exp(-((float(value) - float(mu)) ** 2) / (sigma**2))
        return float(np.clip(score, 0.0, 1.0))

    def _connectivity_score(self, metrics: Mapping[str, float]) -> float:
        if "component_count" in metrics:
            component_count = max(float(metrics["component_count"]), 1.0)
            score = np.exp(-self.connectivity_alpha * (component_count - 1.0) ** 2)
            return float(np.clip(score, 0.0, 1.0))

        connectivity = float(np.clip(metrics["connectivity"], 0.0, 1.0))
        return self._gaussian_score(connectivity, mu=1.0, sigma=0.25)

    def describe(self) -> Dict[str, object]:
        return {
            "formula": "R = sum(w_i * R_i) + lambda * R_novelty",
            "connectivity": "exp(-alpha * (component_count - 1)^2), fallback uses connectivity ratio",
            "path": "continuous path_reachability",
            "gaussian_targets": self.gaussian_targets,
            "connectivity_alpha": self.connectivity_alpha,
            "total_weights": self.total_weights,
            "novelty_scale": self.novelty_scale,
            "novelty_k": self.novelty_k,
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
        nearest_distance = float(np.mean(nearest_distances))
        scale = max(self.novelty_scale * float(np.sqrt(feature_vector.size)), 1e-6)
        novelty_score = nearest_distance / (nearest_distance + scale)
        return float(np.clip(novelty_score, 0.0, 1.0))

    def score_metrics(
        self,
        metrics: Mapping[str, float],
        feature_vector: Optional[Sequence[float]] = None,
        history_vectors: Optional[Iterable[Sequence[float]]] = None,
    ) -> ScoreBreakdown:
        connectivity_score = self._connectivity_score(metrics)
        navigable_score = self._gaussian_score(
            metrics["navigable_ratio"],
            **self.gaussian_targets["navigable_ratio"],
        )
        coast_score = self._gaussian_score(
            metrics["coast_complexity"],
            **self.gaussian_targets["coast_complexity"],
        )
        variance_score = self._gaussian_score(
            metrics["terrain_variance"],
            **self.gaussian_targets["terrain_variance"],
        )
        path_score = float(np.clip(metrics["path_reachability"], 0.0, 1.0))
        land_score = self._gaussian_score(
            metrics["land_ratio"],
            **self.gaussian_targets["land_ratio"],
        )
        novelty_score = self.compute_novelty_score(feature_vector, history_vectors)

        total_score = (
            self.total_weights["connectivity"] * connectivity_score
            + self.total_weights["path"] * path_score
            + self.total_weights["navigable"] * navigable_score
            + self.total_weights["coast"] * coast_score
            + self.total_weights["variance"] * variance_score
            + self.total_weights["land"] * land_score
            + self.total_weights["novelty"] * novelty_score
        )

        structure_score = (
            self.total_weights["connectivity"] * connectivity_score
            + self.total_weights["navigable"] * navigable_score
            + self.total_weights["coast"] * coast_score
            + self.total_weights["variance"] * variance_score
        ) / (
            self.total_weights["connectivity"]
            + self.total_weights["navigable"]
            + self.total_weights["coast"]
            + self.total_weights["variance"]
        )

        component_scores = {
            "connectivity_score": connectivity_score,
            "navigable_score": navigable_score,
            "coast_score": coast_score,
            "variance_score": variance_score,
        }
        return ScoreBreakdown(
            total_score=float(np.clip(total_score, 0.0, 1.0)),
            structure_score=float(np.clip(structure_score, 0.0, 1.0)),
            path_score=path_score,
            land_score=float(land_score),
            novelty_score=float(novelty_score),
            component_scores=component_scores,
        )
