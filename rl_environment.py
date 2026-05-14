"""
Gymnasium environment for normalized island parameter optimization.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from feature_processing import IslandFeatureNormalizer, ParameterSpaceNormalizer
from map_scoring import MapScorer
from pcg_generator import PCGIslandGenerator
from structure_evaluator import StructureEvaluator


class IslandGenerationEnv(gym.Env):
    """Optimizes PCG parameters in normalized action/state spaces."""

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        map_size: int = 128,
        max_steps: int = 50,
        vae_model: Optional[Any] = None,
        feature_normalizer: Optional[IslandFeatureNormalizer] = None,
        scorer: Optional[MapScorer] = None,
        include_latent: bool = True,
        action_step_scale: float = 0.15,
        sampling_profile: str = "island",
        novelty_reference_vectors: Optional[Sequence[Sequence[float]]] = None,
        expert_param_vectors: Optional[Sequence[Sequence[float]]] = None,
        reward_delta_scale: float = 2.0,
        reward_best_scale: float = 0.5,
        reward_expert_scale: float = 0.0,
        reward_step_penalty: float = 0.005,
        reward_success_bonus: float = 0.80,
        reward_failure_penalty: float = 0.80,
        reward_stagnation_penalty: float = 0.10,
        success_score_threshold: float = 0.70,
        failure_score_threshold: float = 0.12,
        success_streak_required: int = 2,
        stagnation_patience: int = 6,
        stagnation_delta: float = 1e-3,
    ):
        super().__init__()

        self.map_size = map_size
        self.max_steps = max_steps
        self.vae_model = vae_model
        self.include_latent = include_latent and vae_model is not None

        self.generator = PCGIslandGenerator(map_size=map_size)
        self.evaluator = StructureEvaluator(map_size=map_size)
        self.scorer = scorer or MapScorer()
        self.sampling_profile = sampling_profile
        self.novelty_reference_vectors = None if novelty_reference_vectors is None else [
            np.asarray(vector, dtype=np.float32) for vector in novelty_reference_vectors
        ]
        self.expert_param_vectors = None if expert_param_vectors is None else np.asarray(
            expert_param_vectors, dtype=np.float32
        )
        self.reward_delta_scale = float(reward_delta_scale)
        self.reward_best_scale = float(reward_best_scale)
        self.reward_expert_scale = float(reward_expert_scale)
        self.reward_step_penalty = float(reward_step_penalty)
        self.reward_success_bonus = float(reward_success_bonus)
        self.reward_failure_penalty = float(reward_failure_penalty)
        self.reward_stagnation_penalty = float(reward_stagnation_penalty)
        self.success_score_threshold = float(success_score_threshold)
        self.failure_score_threshold = float(failure_score_threshold)
        self.success_streak_required = int(max(1, success_streak_required))
        self.stagnation_patience = int(max(1, stagnation_patience))
        self.stagnation_delta = float(max(0.0, stagnation_delta))

        self.param_ranges = self.generator.get_param_ranges(map_size, profile=sampling_profile)
        self.param_normalizer = ParameterSpaceNormalizer(
            param_ranges=self.param_ranges,
            step_scale=action_step_scale,
        )
        self.feature_normalizer = feature_normalizer or IslandFeatureNormalizer(
            metric_names=self.evaluator.metric_names
        )

        self.latent_dim = int(getattr(vae_model, "latent_dim", 0)) if self.include_latent else 0
        self.metric_dim = len(self.evaluator.metric_names)
        self.param_dim = len(self.param_normalizer.param_names)
        self.state_dim = self.param_dim + self.metric_dim + self.latent_dim

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(len(self.param_normalizer.param_names),),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.state_dim,),
            dtype=np.float32,
        )

        self.current_params: Dict[str, float] | None = None
        self.current_heightmap: np.ndarray | None = None
        self.current_metrics: Dict[str, float] | None = None
        self.current_latent: np.ndarray | None = None
        self.current_seed: int = 42
        self.steps = 0
        self.previous_score = None
        self.previous_expert_distance: float | None = None
        self.best_total_score = float("-inf")
        self.success_streak = 0
        self.stagnation_steps = 0

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        self.current_params = self.generator.sample_random_params(self.np_random, profile=self.sampling_profile)
        self.current_seed = int(self.current_params["seed"])
        self.current_heightmap = self.generator.generate_heightmap(self.current_params)
        self.current_metrics = self.evaluator.evaluate(self.current_heightmap)
        self.current_latent = self._encode_latent(self.current_heightmap)
        self.steps = 0
        self.previous_score = self._score_current_state()
        self.previous_expert_distance = self._get_expert_distance() if self.reward_expert_scale != 0.0 else None
        self.best_total_score = float(self.previous_score.total_score)
        self.success_streak = 0
        self.stagnation_steps = 0

        state = self._get_state()
        info = {
            "metrics": dict(self.current_metrics),
            "params": dict(self.current_params),
            "score": self.previous_score.as_dict(),
            "reward_components": {
                "reward": 0.0,
                "delta_score": 0.0,
                "best_delta": 0.0,
                "expert_delta": 0.0,
                "delta_term": 0.0,
                "best_term": 0.0,
                "expert_term": 0.0,
                "step_penalty_term": 0.0,
                "success_bonus": 0.0,
                "failure_penalty": 0.0,
                "stagnation_penalty": 0.0,
                "previous_total_score": float(self.previous_score.total_score),
                "current_total_score": float(self.previous_score.total_score),
                "best_total_score": float(self.best_total_score),
                "expert_distance": self.previous_expert_distance,
                "novelty_score": float(self.previous_score.novelty_score),
                "done_reason": "reset",
            },
        }
        return state, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        if self.current_params is None:
            raise RuntimeError("Environment must be reset before step().")

        self.steps += 1
        self.current_params = self.param_normalizer.apply_normalized_delta(self.current_params, action)
        self.current_params["seed"] = self.current_seed

        self.current_heightmap = self.generator.generate_heightmap(self.current_params)
        self.current_metrics = self.evaluator.evaluate(self.current_heightmap)
        self.current_latent = self._encode_latent(self.current_heightmap)
        score = self._score_current_state()
        previous_total_score = float(self.previous_score.total_score) if self.previous_score is not None else 0.0
        delta_score = float(score.total_score - previous_total_score)
        previous_best_total_score = float(self.best_total_score)
        best_delta = max(0.0, float(score.total_score) - previous_best_total_score)
        current_expert_distance = self._get_expert_distance() if self.reward_expert_scale != 0.0 else None
        expert_delta = 0.0
        if current_expert_distance is not None and self.previous_expert_distance is not None:
            expert_delta = float(self.previous_expert_distance - current_expert_distance)

        delta_term = self.reward_delta_scale * delta_score
        best_term = self.reward_best_scale * best_delta
        expert_term = self.reward_expert_scale * expert_delta
        step_penalty_term = -self.reward_step_penalty
        reward = delta_term + best_term + expert_term + step_penalty_term
        success_bonus = 0.0
        failure_penalty = 0.0
        stagnation_penalty = 0.0
        done_reason = "in_progress"

        if float(score.total_score) > self.best_total_score + self.stagnation_delta:
            self.best_total_score = float(score.total_score)
            self.stagnation_steps = 0
        else:
            self.stagnation_steps += 1

        if float(score.total_score) >= self.success_score_threshold:
            self.success_streak += 1
        else:
            self.success_streak = 0

        terminated = False
        truncated = False
        if self.success_streak >= self.success_streak_required:
            success_bonus = self.reward_success_bonus
            reward += success_bonus
            terminated = True
            done_reason = "success_threshold"
        elif float(score.total_score) <= self.failure_score_threshold:
            failure_penalty = self.reward_failure_penalty
            reward -= failure_penalty
            truncated = True
            done_reason = "failure_threshold"
        elif self.stagnation_steps >= self.stagnation_patience:
            stagnation_penalty = self.reward_stagnation_penalty
            reward -= stagnation_penalty
            truncated = True
            done_reason = "stagnation"
        elif self.steps >= self.max_steps:
            terminated = True
            done_reason = "max_steps"

        self.previous_score = score
        self.previous_expert_distance = current_expert_distance
        state = self._get_state()

        info = {
            "metrics": dict(self.current_metrics),
            "params": dict(self.current_params),
            "score": score.as_dict(),
            "heightmap": self.current_heightmap,
            "done_reason": done_reason,
            "reward_components": {
                "reward": float(reward),
                "delta_score": delta_score,
                "best_delta": float(best_delta),
                "expert_delta": expert_delta,
                "delta_term": float(delta_term),
                "best_term": float(best_term),
                "expert_term": float(expert_term),
                "step_penalty_term": float(step_penalty_term),
                "success_bonus": success_bonus,
                "failure_penalty": failure_penalty,
                "stagnation_penalty": stagnation_penalty,
                "previous_total_score": previous_total_score,
                "current_total_score": float(score.total_score),
                "best_total_score": float(self.best_total_score),
                "expert_distance": current_expert_distance,
                "novelty_score": float(score.novelty_score),
                "done_reason": done_reason,
            },
        }
        return state, float(reward), terminated, truncated, info

    def render(self, mode: str = "human") -> None:
        if mode == "human" and self.current_metrics is not None:
            print(f"Step: {self.steps}")
            for key, value in self.current_metrics.items():
                print(f"  {key}: {value:.4f}")

    def _get_state(self) -> np.ndarray:
        if self.current_metrics is None:
            raise RuntimeError("No metrics available. Call reset() first.")
        return self.feature_normalizer.transform_state(
            param_vector=self.param_normalizer.normalize_params(self.current_params or {}),
            metrics=self.current_metrics,
            latent_vector=self.current_latent if self.include_latent else None,
        )

    def _encode_latent(self, heightmap: np.ndarray) -> Optional[np.ndarray]:
        if not self.include_latent or self.vae_model is None:
            return None
        return self.vae_model.encode_heightmap(heightmap, deterministic=True)

    def _get_novelty_vector(self) -> Optional[np.ndarray]:
        if self.current_metrics is None:
            return None
        return self.feature_normalizer.transform_metrics(self.current_metrics)

    def _score_current_state(self):
        novelty_vector = self._get_novelty_vector()
        return self.scorer.score_metrics(
            self.current_metrics,
            feature_vector=novelty_vector,
            history_vectors=self.novelty_reference_vectors,
        )

    def _get_expert_distance(self) -> Optional[float]:
        if self.current_params is None or self.expert_param_vectors is None or len(self.expert_param_vectors) == 0:
            return None
        current_vector = self.param_normalizer.normalize_params(self.current_params)
        distances = np.linalg.norm(self.expert_param_vectors - current_vector[None, :], axis=1)
        return float(np.min(distances))


if __name__ == "__main__":
    env = IslandGenerationEnv(map_size=128, max_steps=5)
    state, info = env.reset(seed=42)
    print(f"State shape: {state.shape}")
    print(f"Initial score: {info['score']['total_score']:.4f}")

    for step in range(3):
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, info = env.step(action)
        print(f"Step {step + 1}: reward={reward:.4f}, state_range=[{next_state.min():.3f}, {next_state.max():.3f}]")
        if terminated or truncated:
            break
