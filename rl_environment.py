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
        latent_novelty: bool = True,
        action_step_scale: float = 0.08,
        sampling_profile: str = "island",
        novelty_reference_vectors: Optional[Sequence[Sequence[float]]] = None,
        expert_param_vectors: Optional[Sequence[Sequence[float]]] = None,
        reward_current_scale: float = 0.0,
        reward_delta_scale: float = 5.0,
        reward_best_scale: float = 2.0,
        reward_step_penalty: float = 0.01,
        reward_success_bonus: float = 1.0,
        success_score_threshold: float = 0.72,
        success_score_gain_threshold: float = 0.03,
        failure_score_threshold: float = 0.10,
        success_streak_required: int = 3,
        stagnation_patience: int = 5,
        stagnation_delta: float = 1e-3,
    ):
        super().__init__()

        self.map_size = map_size
        self.max_steps = max_steps
        self.vae_model = vae_model
        self.include_latent = include_latent and vae_model is not None
        self.latent_novelty = latent_novelty and vae_model is not None
        self.encode_latent = self.include_latent or self.latent_novelty

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

        self.reward_current_scale = float(reward_current_scale)
        self.reward_delta_scale = float(reward_delta_scale)
        self.reward_best_scale = float(reward_best_scale)
        self.reward_step_penalty = float(reward_step_penalty)
        self.reward_success_bonus = float(reward_success_bonus)
        self.success_score_threshold = float(success_score_threshold)
        self.success_score_gain_threshold = float(max(0.0, success_score_gain_threshold))
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
        self.initial_total_score = 0.0
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
        self.initial_total_score = float(self.previous_score.total_score)
        self.best_total_score = float(self.previous_score.total_score)
        self.success_streak = 0
        self.stagnation_steps = 0

        state = self._get_state()
        info = {
            "metrics": dict(self.current_metrics),
            "params": dict(self.current_params),
            "score": self.previous_score.as_dict(),
            "reward_components": self._empty_reward_components(done_reason="reset"),
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
        current_total_score = float(score.total_score)
        delta_score = current_total_score - previous_total_score
        score_gain_from_initial = current_total_score - float(self.initial_total_score)
        previous_best_total_score = float(self.best_total_score)
        best_delta = max(0.0, current_total_score - previous_best_total_score)

        current_term = self.reward_current_scale * current_total_score
        delta_term = self.reward_delta_scale * delta_score
        best_term = self.reward_best_scale * best_delta
        step_penalty_term = -self.reward_step_penalty
        reward = current_term + delta_term + best_term + step_penalty_term
        success_bonus = 0.0
        done_reason = "in_progress"

        if best_delta > self.stagnation_delta:
            self.best_total_score = current_total_score
            self.stagnation_steps = 0
        else:
            self.stagnation_steps += 1

        if (
            current_total_score >= self.success_score_threshold
            and score_gain_from_initial >= self.success_score_gain_threshold
        ):
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
        elif current_total_score <= self.failure_score_threshold:
            truncated = True
            done_reason = "failure_threshold"
        elif self.stagnation_steps >= self.stagnation_patience:
            truncated = True
            done_reason = "stagnation"
        elif self.steps >= self.max_steps:
            terminated = True
            done_reason = "max_steps"

        self.previous_score = score
        state = self._get_state()

        info = {
            "metrics": dict(self.current_metrics),
            "params": dict(self.current_params),
            "score": score.as_dict(),
            "heightmap": self.current_heightmap,
            "done_reason": done_reason,
            "reward_components": {
                "reward": float(reward),
                "current_score": current_total_score,
                "delta_score": delta_score,
                "best_delta": best_delta,
                "score_gain_from_initial": score_gain_from_initial,
                "current_term": float(current_term),
                "delta_term": float(delta_term),
                "best_term": float(best_term),
                "step_penalty_term": float(step_penalty_term),
                "success_bonus": success_bonus,
                "previous_total_score": previous_total_score,
                "current_total_score": current_total_score,
                "best_total_score": float(self.best_total_score),
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

    def _empty_reward_components(self, done_reason: str) -> Dict[str, float | str]:
        score = 0.0 if self.previous_score is None else float(self.previous_score.total_score)
        return {
            "reward": 0.0,
            "current_score": score,
            "delta_score": 0.0,
            "best_delta": 0.0,
            "score_gain_from_initial": 0.0,
            "current_term": 0.0,
            "delta_term": 0.0,
            "best_term": 0.0,
            "step_penalty_term": 0.0,
            "success_bonus": 0.0,
            "previous_total_score": score,
            "current_total_score": score,
            "best_total_score": score,
            "novelty_score": 0.0 if self.previous_score is None else float(self.previous_score.novelty_score),
            "done_reason": done_reason,
        }

    def _get_state(self) -> np.ndarray:
        if self.current_metrics is None:
            raise RuntimeError("No metrics available. Call reset() first.")
        return self.feature_normalizer.transform_state(
            param_vector=self.param_normalizer.normalize_params(self.current_params or {}),
            metrics=self.current_metrics,
            latent_vector=self.current_latent if self.include_latent else None,
        )

    def _encode_latent(self, heightmap: np.ndarray) -> Optional[np.ndarray]:
        if not self.encode_latent or self.vae_model is None:
            return None
        return self.vae_model.encode_heightmap(heightmap, deterministic=True)

    def _get_novelty_vector(self) -> Optional[np.ndarray]:
        if self.current_metrics is None:
            return None
        if self.current_latent is not None:
            return self.feature_normalizer.transform_latent(self.current_latent)
        return self.feature_normalizer.transform_metrics(self.current_metrics)

    def _score_current_state(self):
        novelty_vector = self._get_novelty_vector()
        return self.scorer.score_metrics(
            self.current_metrics,
            feature_vector=novelty_vector,
            history_vectors=self.novelty_reference_vectors,
        )


if __name__ == "__main__":
    env = IslandGenerationEnv(map_size=128, max_steps=5)
    state, info = env.reset(seed=42)
    print(f"State shape: {state.shape}")
    print(f"Initial score: {info['score']['total_score']:.4f}")

    for step in range(3):
        next_state, reward, terminated, truncated, info = env.step(env.action_space.sample())
        print(f"Step {step + 1}: reward={reward:.4f}, state_range=[{next_state.min():.3f}, {next_state.max():.3f}]")
        if terminated or truncated:
            break
