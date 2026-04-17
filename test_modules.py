"""
Smoke tests for the IslandTest core pipeline.
"""

from __future__ import annotations

import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader

from cmaes_baseline import CMAESOptimizer
from dataset_pipeline import IslandDatasetBuilder
from feature_processing import IslandFeatureNormalizer, ParameterSpaceNormalizer
from map_scoring import MapScorer
from pcg_generator import PCGIslandGenerator
from rl_environment import IslandGenerationEnv
from structure_evaluator import StructureEvaluator
from vae_model import BetaVAE, HeightmapDataset, encode_heightmaps, train_vae


class IslandPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.generator = PCGIslandGenerator(map_size=64)
        self.evaluator = StructureEvaluator(map_size=64)
        self.scorer = MapScorer()

    def test_generator_uses_amplitude_and_returns_normalized_map(self) -> None:
        rng = np.random.default_rng(0)
        params = self.generator.sample_random_params(rng)
        params["A"] = 0.6
        low_amp = self.generator.generate_heightmap(params)
        params["A"] = 1.8
        high_amp = self.generator.generate_heightmap(params)

        self.assertEqual(low_amp.shape, (64, 64))
        self.assertGreaterEqual(float(low_amp.min()), 0.0)
        self.assertLessEqual(float(low_amp.max()), 1.0)
        self.assertNotAlmostEqual(float(low_amp.std()), float(high_amp.std()), places=3)

    def test_structure_evaluator_reports_size_metric(self) -> None:
        heightmap = self.generator.generate_heightmap(self.generator.sample_random_params(np.random.default_rng(1)))
        metrics = self.evaluator.evaluate(heightmap)

        self.assertIn("land_ratio", metrics)
        self.assertEqual(set(metrics.keys()), set(self.evaluator.metric_names))
        self.assertTrue(0.0 <= metrics["land_ratio"] <= 1.0)

    def test_dataset_builder_cleans_and_summarizes(self) -> None:
        builder = IslandDatasetBuilder(map_size=64, scorer=self.scorer)
        raw_samples = builder.generate_samples(n_samples=12, seed=123)
        clean_samples = builder.clean_samples(raw_samples)
        summary = builder.evaluate_dataset(raw_samples)
        arrays = builder.build_training_arrays(clean_samples)

        self.assertEqual(summary["num_samples"], 12)
        self.assertGreater(summary["num_valid"], 0)
        self.assertIn("land_ratio", summary["metric_stats"])
        self.assertEqual(arrays["heightmaps"].ndim, 3)
        self.assertEqual(arrays["normalized_params"].shape[1], len(builder.param_normalizer.param_names))

    def test_feature_normalization_keeps_values_on_shared_scale(self) -> None:
        metrics_list = [
            self.evaluator.evaluate(
                self.generator.generate_heightmap(self.generator.sample_random_params(np.random.default_rng(seed)))
            )
            for seed in range(8)
        ]
        metric_matrix = np.array(
            [[metrics[name] for name in self.evaluator.metric_names] for metrics in metrics_list],
            dtype=np.float32,
        )
        latent_matrix = np.random.default_rng(42).normal(size=(8, 16)).astype(np.float32)

        normalizer = IslandFeatureNormalizer(metric_names=self.evaluator.metric_names)
        normalizer.fit(metric_matrix=metric_matrix, latent_matrix=latent_matrix)

        state = normalizer.transform_state(metrics_list[0], latent_matrix[0])
        self.assertTrue(np.all(state <= 1.0 + 1e-6))
        self.assertTrue(np.all(state >= -1.0 - 1e-6))

    def test_parameter_normalizer_is_round_trip_safe(self) -> None:
        normalizer = ParameterSpaceNormalizer(self.generator.get_param_ranges(64))
        params = self.generator.sample_random_params(np.random.default_rng(99))
        normalized = normalizer.normalize_params(params)
        reconstructed = normalizer.denormalize_vector(normalized)

        for name in normalizer.param_names:
            if name == "N_octaves":
                self.assertEqual(int(params[name]), int(reconstructed[name]))
            else:
                self.assertAlmostEqual(float(params[name]), float(reconstructed[name]), places=5)

    def test_environment_uses_normalized_state_and_reward(self) -> None:
        env = IslandGenerationEnv(map_size=64, max_steps=5)
        state, info = env.reset(seed=7)
        self.assertEqual(state.shape, env.observation_space.shape)
        self.assertTrue(np.all(state <= 1.0 + 1e-6))
        self.assertTrue(np.all(state >= -1.0 - 1e-6))
        self.assertIn("land_score", info["score"])

        next_state, reward, terminated, truncated, info = env.step(env.action_space.sample())
        self.assertEqual(next_state.shape, env.observation_space.shape)
        self.assertIsInstance(reward, float)
        self.assertFalse(truncated)
        self.assertIn("total_score", info["score"])

    def test_scorer_matches_metrics_interface(self) -> None:
        heightmap = self.generator.generate_heightmap(self.generator.sample_random_params(np.random.default_rng(5)))
        metrics = self.evaluator.evaluate(heightmap)
        score = self.scorer.score_metrics(metrics)
        self.assertGreaterEqual(score.total_score, 0.0)
        self.assertLessEqual(score.total_score, 1.0)
        self.assertIn("variance_score", score.component_scores)

    def test_vae_helpers_extract_latents(self) -> None:
        heightmaps = np.stack(
            [
                self.generator.generate_heightmap(self.generator.sample_random_params(np.random.default_rng(seed)))
                for seed in range(10)
            ],
            axis=0,
        )
        dataset = HeightmapDataset(heightmaps)
        dataloader = DataLoader(dataset, batch_size=5, shuffle=False)
        vae = BetaVAE(map_size=64, latent_dim=8, beta=2.0)
        history = train_vae(vae, dataloader, epochs=1, device="cpu")
        latents = encode_heightmaps(vae, heightmaps, batch_size=5, device="cpu")

        self.assertEqual(len(history), 1)
        self.assertEqual(latents.shape, (10, 8))

    def test_cmaes_baseline_runs_in_normalized_space(self) -> None:
        optimizer = CMAESOptimizer(
            param_ranges=self.generator.get_param_ranges(64),
            sigma0=0.20,
            pop_size=8,
            map_size=64,
        )
        best_params, best_fitness = optimizer.optimize(generations=2, verbose=False)
        self.assertEqual(best_params.shape, (len(optimizer.param_names),))
        self.assertTrue(np.all(best_params <= 1.0 + 1e-6))
        self.assertTrue(np.all(best_params >= -1.0 - 1e-6))
        self.assertGreaterEqual(best_fitness, 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
