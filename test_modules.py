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
from ppo_baseline import PPOAgent
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
        for name in self.evaluator.metric_names:
            self.assertIn(name, metrics)
        self.assertIn("component_count", metrics)
        self.assertIn("path_exists", metrics)
        self.assertTrue(0.0 <= metrics["land_ratio"] <= 1.0)

    def test_navigability_responds_to_local_terrain_roughness(self) -> None:
        rng = np.random.default_rng(0)
        rough_heightmap = (0.35 + 0.65 * rng.random((64, 64))).astype(np.float32)
        metrics = self.evaluator.evaluate(rough_heightmap)

        self.assertLess(metrics["navigable_ratio"], 0.8)
        self.assertGreaterEqual(metrics["navigable_ratio"], 0.0)

    def test_astar_reachability_distinguishes_connected_paths(self) -> None:
        evaluator = StructureEvaluator(map_size=8, path_sample_points=4, max_path_pairs=6)
        slope = np.zeros((8, 8), dtype=np.float32)

        connected = np.zeros((8, 8), dtype=bool)
        connected[1:7, 3] = True
        connected[4, 1:7] = True

        disconnected = np.zeros((8, 8), dtype=bool)
        disconnected[1:3, 1:3] = True
        disconnected[5:7, 5:7] = True

        connected_score = evaluator._check_path_reachability(connected, slope=slope)
        disconnected_score = evaluator._check_path_reachability(disconnected, slope=slope)

        self.assertGreater(connected_score, disconnected_score)
        self.assertAlmostEqual(connected_score, 1.0, places=6)
        self.assertLess(disconnected_score, 1.0)

    def test_coast_complexity_rewards_jagged_coastline(self) -> None:
        smooth = np.zeros((16, 16), dtype=bool)
        smooth[4:12, 4:12] = True

        jagged = np.zeros((16, 16), dtype=bool)
        jagged[4:12, 4:12] = True
        jagged[3, 6:10] = True
        jagged[12, 5:11] = True
        jagged[6:10, 3] = True
        jagged[5:11, 12] = True

        smooth_complexity = self.evaluator._calculate_coast_complexity(smooth)
        jagged_complexity = self.evaluator._calculate_coast_complexity(jagged)

        self.assertGreater(jagged_complexity, smooth_complexity)

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

        param_vector = np.linspace(-1.0, 1.0, 9, dtype=np.float32)
        state = normalizer.transform_state(param_vector=param_vector, metrics=metrics_list[0], latent_vector=latent_matrix[0])
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
        param_dim = len(env.param_normalizer.param_names)
        expected_params = env.param_normalizer.normalize_params(env.current_params)
        np.testing.assert_allclose(state[:param_dim], expected_params, atol=1e-6)

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

    def test_scorer_directly_penalizes_oversized_maps(self) -> None:
        balanced_metrics = {
            "connectivity": 1.0,
            "navigable_ratio": 0.95,
            "coast_complexity": 4.5,
            "terrain_variance": 0.14,
            "path_reachability": 1.0,
            "land_ratio": 0.32,
        }
        oversized_metrics = dict(balanced_metrics)
        oversized_metrics["land_ratio"] = 0.82

        balanced_score = self.scorer.score_metrics(balanced_metrics)
        oversized_score = self.scorer.score_metrics(oversized_metrics)

        self.assertGreater(balanced_score.land_score, oversized_score.land_score)
        self.assertGreater(balanced_score.total_score, oversized_score.total_score)

    def test_vae_helpers_extract_latents(self) -> None:
        heightmaps = np.stack(
            [
                self.generator.generate_heightmap(self.generator.sample_random_params(np.random.default_rng(seed)))
                for seed in range(10)
            ],
            axis=0,
        )
        structure_targets = np.array(
            [
                [self.evaluator.evaluate(heightmap)[name] for name in self.evaluator.core_metric_names]
                for heightmap in heightmaps
            ],
            dtype=np.float32,
        )
        dataset = HeightmapDataset(heightmaps, structure_targets=structure_targets)
        dataloader = DataLoader(dataset, batch_size=5, shuffle=False)
        vae = BetaVAE(
            map_size=64,
            latent_dim=8,
            beta=0.25,
            beta_start=0.0,
            free_bits=0.01,
            gradient_loss_weight=0.20,
            structure_dim=structure_targets.shape[1],
            structure_loss_weight=0.2,
        )
        history = train_vae(vae, dataloader, epochs=1, device="cpu", warmup_epochs=1)
        latents = encode_heightmaps(vae, heightmaps, batch_size=5, device="cpu")

        self.assertEqual(len(history), 1)
        self.assertIn("beta", history[0])
        self.assertIn("gradient_loss", history[0])
        self.assertIn("mask_loss", history[0])
        self.assertIn("land_dice_loss", history[0])
        self.assertIn("coast_loss", history[0])
        self.assertIn("coast_dice_loss", history[0])
        self.assertIn("weighted_mse_loss", history[0])
        self.assertIn("weighted_l1_loss", history[0])
        self.assertIn("structure_loss", history[0])
        self.assertIn("kl_raw", history[0])
        self.assertEqual(latents.shape, (10, 8))

    def test_vae_supports_128_heightmaps(self) -> None:
        generator = PCGIslandGenerator(map_size=128)
        heightmaps = np.stack(
            [
                generator.generate_heightmap(generator.sample_random_params(np.random.default_rng(seed)))
                for seed in range(2)
            ],
            axis=0,
        )
        batch = torch.from_numpy(heightmaps).unsqueeze(1)
        vae = BetaVAE(map_size=128, latent_dim=128)

        reconstruction, mu, logvar, structure_prediction = vae(batch)

        self.assertEqual(reconstruction.shape, batch.shape)
        self.assertEqual(mu.shape, (2, 128))
        self.assertEqual(logvar.shape, (2, 128))
        self.assertIsNone(structure_prediction)

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

    def test_ppo_agent_can_collect_and_update(self) -> None:
        env = IslandGenerationEnv(map_size=64, max_steps=6)
        agent = PPOAgent(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.shape[0],
            hidden_dim=64,
            batch_size=4,
            epoch=2,
            action_range=1.0,
        ).to("cpu")

        memory = []
        state, _ = env.reset(seed=123)
        for _ in range(8):
            action, log_prob = agent.get_action_and_log_prob(state, deterministic=False)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            memory.append((state, action, reward, next_state, done, log_prob))
            state = next_state
            if done:
                state, _ = env.reset()

        losses = agent.update(memory)
        self.assertIn("total_loss", losses)
        self.assertIn("policy_loss", losses)
        self.assertIn("value_loss", losses)


if __name__ == "__main__":
    unittest.main(verbosity=2)
