"""
CMA-ES baseline operating in the same normalized parameter space as RL.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from feature_processing import ParameterSpaceNormalizer
from map_scoring import MapScorer
from pcg_generator import PCGIslandGenerator
from structure_evaluator import StructureEvaluator


class CMAESOptimizer:
    """Simple CMA-ES baseline on normalized parameters."""

    def __init__(
        self,
        param_ranges: Dict[str, Tuple[float, float]],
        sigma0: float = 0.25,
        pop_size: int | None = None,
        map_size: int = 64,
    ):
        self.param_ranges = dict(param_ranges)
        self.param_normalizer = ParameterSpaceNormalizer(self.param_ranges)
        self.param_names = self.param_normalizer.param_names
        self.n_params = len(self.param_names)
        self.pop_size = pop_size or (4 + int(3 * np.log(self.n_params)))

        self.sigma = sigma0
        self.mu = self.pop_size // 2
        self.weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights /= np.sum(self.weights)
        self.mu_eff = np.sum(self.weights) ** 2 / np.sum(self.weights**2)

        self.cc = (4 + self.mu_eff / self.n_params) / (self.n_params + 4 + 2 * self.mu_eff / self.n_params)
        self.cs = (self.mu_eff + 2) / (self.n_params + self.mu_eff + 5)
        self.c1 = 2 / ((self.n_params + 1.3) ** 2 + self.mu_eff)
        self.cmu = min(
            1 - self.c1,
            2 * (self.mu_eff - 2 + 1 / self.mu_eff) / ((self.n_params + 2) ** 2 + self.mu_eff),
        )
        self.damps = 1 + 2 * max(0, np.sqrt((self.mu_eff - 1) / (self.n_params + 1)) - 1) + self.cs

        self.xmean = np.zeros(self.n_params, dtype=np.float32)
        self.pc = np.zeros(self.n_params, dtype=np.float32)
        self.ps = np.zeros(self.n_params, dtype=np.float32)
        self.C = np.eye(self.n_params, dtype=np.float32)
        self.chiN = np.sqrt(self.n_params) * (1 - 1 / (4 * self.n_params) + 1 / (21 * self.n_params**2))
        self.eigenvalues = np.ones(self.n_params, dtype=np.float32)
        self.eigenvectors = np.eye(self.n_params, dtype=np.float32)
        self._update_eigen()

        self.generator = PCGIslandGenerator(map_size=map_size)
        self.evaluator = StructureEvaluator(map_size=map_size)
        self.scorer = MapScorer()
        self.rng = np.random.default_rng(42)

    def evaluate_fitness(self, normalized_params: np.ndarray) -> float:
        param_dict = self.param_normalizer.denormalize_vector(normalized_params)
        param_dict["seed"] = int(self.rng.integers(0, 1_000_000))
        heightmap = self.generator.generate_heightmap(param_dict)
        metrics = self.evaluator.evaluate(heightmap)
        return self.scorer.score_metrics(metrics).total_score

    def optimize(self, generations: int = 50, verbose: bool = True) -> Tuple[np.ndarray, float]:
        best_fitness = -np.inf
        best_params = self.xmean.copy()

        for generation in range(generations):
            population = self._sample_population()
            fitnesses = np.array([self.evaluate_fitness(individual) for individual in population], dtype=np.float32)
            order = np.argsort(fitnesses)[::-1]
            population = population[order]
            fitnesses = fitnesses[order]

            if fitnesses[0] > best_fitness:
                best_fitness = float(fitnesses[0])
                best_params = population[0].copy()

            previous_mean = self.xmean.copy()
            self.xmean = np.sum(self.weights[:, None] * population[: self.mu], axis=0)
            self.xmean = np.clip(self.xmean, -1.0, 1.0)

            direction = (self.xmean - previous_mean) / max(self.sigma, 1e-6)
            self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.mu_eff) * direction
            hsig = float(
                np.linalg.norm(self.ps) / np.sqrt(1 - (1 - self.cs) ** (2 * (generation + 1)))
                < (1.4 + 2 / (self.n_params + 1)) * self.chiN
            )
            self.pc = (1 - self.cc) * self.pc + hsig * np.sqrt(self.cc * (2 - self.cc) * self.mu_eff) * direction

            centered = population[: self.mu] - previous_mean
            self.C = (
                (1 - self.c1 - self.cmu) * self.C
                + self.c1 * (np.outer(self.pc, self.pc) + (1 - hsig) * self.cc * (2 - self.cc) * self.C)
                + self.cmu
                * np.sum(
                    self.weights[:, None, None] * np.einsum("ni,nj->nij", centered, centered),
                    axis=0,
                )
            )
            self.sigma *= np.exp((self.cs / self.damps) * (np.linalg.norm(self.ps) / self.chiN - 1))

            if generation % 5 == 0:
                self._update_eigen()

            if verbose and (generation + 1) % 10 == 0:
                print(
                    f"Generation {generation + 1}/{generations} "
                    f"best={best_fitness:.4f} "
                    f"mean_top={np.mean(fitnesses[: self.mu]):.4f}"
                )

        return best_params, best_fitness

    def generate_islands(self, n_islands: int = 20) -> List[Tuple[np.ndarray, Dict[str, float]]]:
        results = []
        for _ in range(n_islands):
            params = self.param_normalizer.denormalize_vector(self.xmean)
            params["seed"] = int(self.rng.integers(0, 1_000_000))
            heightmap = self.generator.generate_heightmap(params)
            metrics = self.evaluator.evaluate(heightmap)
            results.append((heightmap, metrics))
        return results

    def _sample_population(self) -> np.ndarray:
        population = np.zeros((self.pop_size, self.n_params), dtype=np.float32)
        sqrt_diag = np.sqrt(np.maximum(self.eigenvalues, 1e-8))
        transform = self.eigenvectors @ np.diag(sqrt_diag)
        for idx in range(self.pop_size):
            noise = self.rng.standard_normal(self.n_params)
            population[idx] = np.clip(self.xmean + self.sigma * (transform @ noise), -1.0, 1.0)
        return population

    def _update_eigen(self) -> None:
        self.C = (self.C + self.C.T) / 2.0
        self.eigenvalues, self.eigenvectors = np.linalg.eigh(self.C)
        self.eigenvalues = np.maximum(self.eigenvalues, 1e-8)


if __name__ == "__main__":
    generator = PCGIslandGenerator(map_size=64)
    optimizer = CMAESOptimizer(generator.get_param_ranges(64), sigma0=0.25, pop_size=16, map_size=64)
    best_params, best_fitness = optimizer.optimize(generations=5, verbose=True)
    decoded = optimizer.param_normalizer.denormalize_vector(best_params)
    print(f"Best fitness: {best_fitness:.4f}")
    print("Decoded parameters:")
    for name, value in decoded.items():
        print(f"  {name}: {value}")
