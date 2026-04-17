"""
Dataset construction, cleaning, evaluation, and feature-normalizer fitting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from feature_processing import IslandFeatureNormalizer, ParameterSpaceNormalizer
from map_scoring import MapScorer
from pcg_generator import PCGIslandGenerator
from structure_evaluator import StructureEvaluator


@dataclass
class DatasetSample:
    params: Dict[str, float]
    normalized_params: np.ndarray
    heightmap: np.ndarray
    metrics: Dict[str, float]
    score: float
    score_details: Dict[str, float]
    valid: bool
    rejection_reasons: List[str]


class IslandDatasetBuilder:
    """Builds, cleans, and evaluates island datasets before VAE/RL training."""

    def __init__(
        self,
        map_size: int = 64,
        scorer: Optional[MapScorer] = None,
    ):
        self.map_size = map_size
        self.generator = PCGIslandGenerator(map_size=map_size)
        self.evaluator = StructureEvaluator(map_size=map_size)
        self.scorer = scorer or MapScorer()
        self.param_ranges = self.generator.get_param_ranges(map_size)
        self.param_normalizer = ParameterSpaceNormalizer(self.param_ranges)

    def generate_samples(
        self,
        n_samples: int,
        seed: int = 42,
    ) -> List[DatasetSample]:
        rng = np.random.default_rng(seed)
        samples: List[DatasetSample] = []

        for _ in range(n_samples):
            params = self.generator.sample_random_params(rng)
            normalized_params = self.param_normalizer.normalize_params(params)
            heightmap = self.generator.generate_heightmap(params)
            metrics = self.evaluator.evaluate(heightmap)
            score_breakdown = self.scorer.score_metrics(metrics)
            rejection_reasons = self._collect_rejection_reasons(metrics, score_breakdown.total_score)
            samples.append(
                DatasetSample(
                    params=params,
                    normalized_params=normalized_params,
                    heightmap=heightmap,
                    metrics=metrics,
                    score=score_breakdown.total_score,
                    score_details=score_breakdown.as_dict(),
                    valid=len(rejection_reasons) == 0,
                    rejection_reasons=rejection_reasons,
                )
            )

        return samples

    def clean_samples(
        self,
        samples: Sequence[DatasetSample],
        deduplicate_decimals: int = 3,
    ) -> List[DatasetSample]:
        seen_signatures = set()
        cleaned: List[DatasetSample] = []

        for sample in samples:
            signature = tuple(
                round(float(sample.metrics[name]), deduplicate_decimals)
                for name in self.evaluator.metric_names
            )
            if signature in seen_signatures:
                sample.valid = False
                sample.rejection_reasons.append("near_duplicate_metrics")
                continue
            if not sample.valid:
                continue

            seen_signatures.add(signature)
            cleaned.append(sample)

        return cleaned

    def evaluate_dataset(self, samples: Sequence[DatasetSample]) -> Dict[str, object]:
        metrics_matrix = np.array(
            [[sample.metrics[name] for name in self.evaluator.metric_names] for sample in samples],
            dtype=np.float32,
        )
        scores = np.array([sample.score for sample in samples], dtype=np.float32)

        metric_stats = {}
        for idx, name in enumerate(self.evaluator.metric_names):
            values = metrics_matrix[:, idx]
            metric_stats[name] = {
                "mean": float(values.mean()),
                "std": float(values.std()),
                "min": float(values.min()),
                "max": float(values.max()),
            }

        rejection_histogram: Dict[str, int] = {}
        for sample in samples:
            for reason in sample.rejection_reasons:
                rejection_histogram[reason] = rejection_histogram.get(reason, 0) + 1

        return {
            "num_samples": len(samples),
            "num_valid": int(sum(sample.valid for sample in samples)),
            "quality_score_mean": float(scores.mean()),
            "quality_score_std": float(scores.std()),
            "metric_stats": metric_stats,
            "rejection_histogram": rejection_histogram,
        }

    def build_training_arrays(self, samples: Sequence[DatasetSample]) -> Dict[str, np.ndarray]:
        valid_samples = [sample for sample in samples if sample.valid]
        return {
            "heightmaps": np.stack([sample.heightmap for sample in valid_samples], axis=0),
            "normalized_params": np.stack([sample.normalized_params for sample in valid_samples], axis=0),
            "quality_scores": np.array([sample.score for sample in valid_samples], dtype=np.float32),
            "metric_matrix": np.array(
                [[sample.metrics[name] for name in self.evaluator.metric_names] for sample in valid_samples],
                dtype=np.float32,
            ),
        }

    def fit_feature_normalizer(
        self,
        samples: Sequence[DatasetSample],
        latent_matrix: Optional[np.ndarray] = None,
    ) -> IslandFeatureNormalizer:
        valid_samples = [sample for sample in samples if sample.valid]
        metric_matrix = np.array(
            [[sample.metrics[name] for name in self.evaluator.metric_names] for sample in valid_samples],
            dtype=np.float32,
        )
        normalizer = IslandFeatureNormalizer(metric_names=self.evaluator.metric_names)
        normalizer.fit(metric_matrix=metric_matrix, latent_matrix=latent_matrix)
        return normalizer

    def _collect_rejection_reasons(self, metrics: Dict[str, float], score: float) -> List[str]:
        reasons: List[str] = []
        if metrics["land_ratio"] < 0.08:
            reasons.append("too_little_land")
        if metrics["land_ratio"] > 0.85:
            reasons.append("too_much_land")
        if metrics["connectivity"] < 0.75:
            reasons.append("weak_connectivity")
        if metrics["path_reachability"] < 0.5:
            reasons.append("poor_reachability")
        if metrics["terrain_variance"] < 0.02:
            reasons.append("too_flat")
        if score < 0.35:
            reasons.append("low_quality_score")
        return reasons
