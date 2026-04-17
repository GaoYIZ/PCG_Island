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
        map_size: int = 64,
        max_steps: int = 50,
        vae_model: Optional[Any] = None,
        feature_normalizer: Optional[IslandFeatureNormalizer] = None,
        scorer: Optional[MapScorer] = None,
        include_latent: bool = True,
        action_step_scale: float = 0.15,
    ):
        super().__init__()

        self.map_size = map_size
        self.max_steps = max_steps
        self.vae_model = vae_model
        self.include_latent = include_latent and vae_model is not None

        self.generator = PCGIslandGenerator(map_size=map_size)
        self.evaluator = StructureEvaluator(map_size=map_size)
        self.scorer = scorer or MapScorer()

        self.param_ranges = self.generator.get_param_ranges(map_size)
        self.param_normalizer = ParameterSpaceNormalizer(
            param_ranges=self.param_ranges,
            step_scale=action_step_scale,
        )
        self.feature_normalizer = feature_normalizer or IslandFeatureNormalizer(
            metric_names=self.evaluator.metric_names
        )

        self.latent_dim = int(getattr(vae_model, "latent_dim", 0)) if self.include_latent else 0
        self.metric_dim = len(self.evaluator.metric_names)
        self.state_dim = self.metric_dim + self.latent_dim

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
        self.history_buffer: list[np.ndarray] = []
        self.buffer_size = 100

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        self.current_params = self.generator.sample_random_params(self.np_random)
        self.current_seed = int(self.current_params["seed"])
        self.current_heightmap = self.generator.generate_heightmap(self.current_params)
        self.current_metrics = self.evaluator.evaluate(self.current_heightmap)
        self.current_latent = self._encode_latent(self.current_heightmap)
        self.history_buffer = []
        self.steps = 0

        state = self._get_state()
        info = {
            "metrics": dict(self.current_metrics),
            "params": dict(self.current_params),
            "score": self.scorer.score_metrics(self.current_metrics).as_dict(),
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
        novelty_vector = self._get_novelty_vector()

        score = self.scorer.score_metrics(
            self.current_metrics,
            feature_vector=novelty_vector,
            history_vectors=self.history_buffer,
        )

        if novelty_vector is not None:
            if len(self.history_buffer) >= self.buffer_size:
                self.history_buffer.pop(0)
            self.history_buffer.append(novelty_vector.copy())

        terminated = self.steps >= self.max_steps
        truncated = False
        state = self._get_state()

        info = {
            "metrics": dict(self.current_metrics),
            "params": dict(self.current_params),
            "score": score.as_dict(),
            "heightmap": self.current_heightmap,
        }
        return state, float(score.total_score), terminated, truncated, info

    def render(self, mode: str = "human") -> None:
        if mode == "human" and self.current_metrics is not None:
            print(f"Step: {self.steps}")
            for key, value in self.current_metrics.items():
                print(f"  {key}: {value:.4f}")

    def _get_state(self) -> np.ndarray:
        if self.current_metrics is None:
            raise RuntimeError("No metrics available. Call reset() first.")
        return self.feature_normalizer.transform_state(
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
        if self.include_latent and self.current_latent is not None:
            return self.feature_normalizer.transform_latent(self.current_latent)
        return self.feature_normalizer.transform_metrics(self.current_metrics)


if __name__ == "__main__":
    env = IslandGenerationEnv(map_size=64, max_steps=5)
    state, info = env.reset(seed=42)
    print(f"State shape: {state.shape}")
    print(f"Initial score: {info['score']['total_score']:.4f}")

    for step in range(3):
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, info = env.step(action)
        print(f"Step {step + 1}: reward={reward:.4f}, state_range=[{next_state.min():.3f}, {next_state.max():.3f}]")
        if terminated or truncated:
            break
