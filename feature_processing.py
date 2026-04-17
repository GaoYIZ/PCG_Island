"""
Feature and parameter normalization utilities for IslandTest.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Sequence

import numpy as np


DEFAULT_METRIC_BOUNDS: Dict[str, tuple[float, float]] = {
    "connectivity": (0.0, 1.0),
    "navigable_ratio": (0.0, 1.0),
    "coast_complexity": (0.8, 3.5),
    "terrain_variance": (0.0, 0.35),
    "path_reachability": (0.0, 1.0),
    "land_ratio": (0.0, 1.0),
}


class FixedRangeNormalizer:
    """Maps bounded features to [-1, 1]."""

    def __init__(self, low: Sequence[float], high: Sequence[float]):
        self.low = np.asarray(low, dtype=np.float32)
        self.high = np.asarray(high, dtype=np.float32)
        self.scale = np.maximum(self.high - self.low, 1e-8)

    def transform(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        normalized = 2.0 * (values - self.low) / self.scale - 1.0
        return np.clip(normalized, -1.0, 1.0)

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        clipped = np.clip(values, -1.0, 1.0)
        return (clipped + 1.0) * 0.5 * self.scale + self.low


class ZScoreClipNormalizer:
    """Standardizes features and clips them back into [-1, 1]."""

    def __init__(self, clip: float = 3.0):
        self.clip = float(clip)
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None

    @property
    def fitted(self) -> bool:
        return self.mean is not None and self.std is not None

    def fit(self, values: np.ndarray) -> "ZScoreClipNormalizer":
        values = np.asarray(values, dtype=np.float32)
        if values.ndim == 1:
            values = values[None, :]
        self.mean = values.mean(axis=0)
        self.std = np.maximum(values.std(axis=0), 1e-6)
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("ZScoreClipNormalizer must be fitted before use.")
        values = np.asarray(values, dtype=np.float32)
        z_score = (values - self.mean) / self.std
        return np.clip(z_score / self.clip, -1.0, 1.0)


class ParameterSpaceNormalizer:
    """Normalizes PCG parameters to a shared scale and applies bounded deltas."""

    def __init__(
        self,
        param_ranges: Mapping[str, tuple[float, float]],
        param_names: Optional[Iterable[str]] = None,
        step_scale: float = 0.15,
    ):
        self.param_ranges = dict(param_ranges)
        self.param_names = tuple(
            param_names if param_names is not None else [name for name in self.param_ranges if name != "seed"]
        )
        low = [self.param_ranges[name][0] for name in self.param_names]
        high = [self.param_ranges[name][1] for name in self.param_names]
        self.normalizer = FixedRangeNormalizer(low=low, high=high)
        self.step_scale = float(step_scale)

    def normalize_params(self, params: Mapping[str, float]) -> np.ndarray:
        values = np.array([params[name] for name in self.param_names], dtype=np.float32)
        return self.normalizer.transform(values)

    def denormalize_vector(self, normalized_params: Sequence[float]) -> Dict[str, float]:
        values = self.normalizer.inverse_transform(np.asarray(normalized_params, dtype=np.float32))
        result: Dict[str, float] = {}
        for idx, name in enumerate(self.param_names):
            value = float(values[idx])
            if name == "N_octaves":
                value = int(round(value))
            result[name] = value
        return result

    def apply_normalized_delta(
        self,
        params: Mapping[str, float],
        action: Sequence[float],
    ) -> Dict[str, float]:
        base_vector = self.normalize_params(params)
        action = np.asarray(action, dtype=np.float32)
        if action.shape != base_vector.shape:
            raise ValueError(
                f"Action shape {action.shape} does not match normalized parameter shape {base_vector.shape}."
            )
        updated_vector = np.clip(base_vector + action * self.step_scale, -1.0, 1.0)
        updated_params = dict(params)
        updated_params.update(self.denormalize_vector(updated_vector))
        return updated_params


@dataclass
class IslandFeatureNormalizer:
    """
    Normalizes structured metrics and optional latent vectors onto a shared scale.

    Metrics use fixed physical bounds before fitting and empirical z-score clipping after fitting.
    Latent vectors use empirical z-score clipping when stats are available.
    """

    metric_names: Sequence[str]
    clip: float = 3.0

    def __post_init__(self) -> None:
        metric_low = [DEFAULT_METRIC_BOUNDS[name][0] for name in self.metric_names]
        metric_high = [DEFAULT_METRIC_BOUNDS[name][1] for name in self.metric_names]
        self.metric_bound_normalizer = FixedRangeNormalizer(metric_low, metric_high)
        self.metric_stat_normalizer = ZScoreClipNormalizer(clip=self.clip)
        self.latent_normalizer = ZScoreClipNormalizer(clip=self.clip)

    def fit(
        self,
        metric_matrix: np.ndarray,
        latent_matrix: Optional[np.ndarray] = None,
    ) -> "IslandFeatureNormalizer":
        self.metric_stat_normalizer.fit(metric_matrix)
        if latent_matrix is not None and len(latent_matrix) > 0:
            self.latent_normalizer.fit(latent_matrix)
        return self

    def transform_metrics(self, metrics: Mapping[str, float]) -> np.ndarray:
        values = np.array([metrics[name] for name in self.metric_names], dtype=np.float32)
        if self.metric_stat_normalizer.fitted:
            return self.metric_stat_normalizer.transform(values)
        return self.metric_bound_normalizer.transform(values)

    def transform_latent(self, latent_vector: Sequence[float]) -> np.ndarray:
        latent_vector = np.asarray(latent_vector, dtype=np.float32)
        if self.latent_normalizer.fitted:
            return self.latent_normalizer.transform(latent_vector)
        return np.tanh(latent_vector / self.clip)

    def transform_state(
        self,
        metrics: Mapping[str, float],
        latent_vector: Optional[Sequence[float]] = None,
    ) -> np.ndarray:
        metric_features = self.transform_metrics(metrics)
        if latent_vector is None:
            return metric_features.astype(np.float32)
        latent_features = self.transform_latent(latent_vector)
        return np.concatenate([latent_features, metric_features]).astype(np.float32)
