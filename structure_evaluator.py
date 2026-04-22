"""
Terrain structure evaluator used for dataset curation and reward computation.
"""

from __future__ import annotations

from collections import deque
from typing import Dict, Iterable, Sequence

import numpy as np
from scipy import ndimage


class StructureEvaluator:
    """Extracts compact terrain descriptors from an island heightmap."""

    metric_names: Sequence[str] = (
        "connectivity",
        "navigable_ratio",
        "coast_complexity",
        "terrain_variance",
        "path_reachability",
        "land_ratio",
    )

    def __init__(self, map_size: int = 64, water_threshold: float = 0.30, slope_threshold: float = 30.0):
        self.map_size = map_size
        self.water_threshold = float(water_threshold)
        self.slope_threshold = float(slope_threshold)

    def evaluate(self, heightmap: np.ndarray) -> Dict[str, float]:
        land_mask = heightmap > self.water_threshold
        slope = self._compute_slope(heightmap)
        navigable_mask = land_mask & (slope < self.slope_threshold)

        return {
            "connectivity": self._check_connectivity(land_mask),
            "navigable_ratio": self._calculate_navigable_ratio(land_mask, navigable_mask),
            "coast_complexity": self._calculate_coast_complexity(land_mask),
            "terrain_variance": self._calculate_terrain_variance(heightmap, land_mask),
            "path_reachability": self._check_path_reachability(navigable_mask),
            "land_ratio": self._calculate_land_ratio(land_mask),
        }

    def get_feature_vector(
        self,
        heightmap: np.ndarray | None = None,
        metrics: Dict[str, float] | None = None,
        metric_names: Iterable[str] | None = None,
    ) -> np.ndarray:
        if metrics is None:
            if heightmap is None:
                raise ValueError("Either heightmap or metrics must be provided.")
            metrics = self.evaluate(heightmap)

        ordered_names = tuple(metric_names or self.metric_names)
        return np.array([metrics[name] for name in ordered_names], dtype=np.float32)

    def _compute_slope(self, heightmap: np.ndarray) -> np.ndarray:
        grad_y, grad_x = np.gradient(heightmap)
        return np.degrees(np.arctan(np.sqrt(grad_x**2 + grad_y**2)))

    def _check_connectivity(self, land_mask: np.ndarray) -> float:
        if not np.any(land_mask):
            return 0.0

        structure = np.ones((3, 3), dtype=np.int32)
        labeled, num_components = ndimage.label(land_mask, structure=structure)
        if num_components <= 1:
            return 1.0

        component_sizes = ndimage.sum(land_mask, labeled, index=np.arange(1, num_components + 1))
        total_land = float(np.sum(land_mask))
        return float(np.max(component_sizes) / max(total_land, 1.0))

    @staticmethod
    def _calculate_navigable_ratio(land_mask: np.ndarray, navigable_mask: np.ndarray) -> float:
        total_land = float(np.sum(land_mask))
        if total_land == 0.0:
            return 0.0
        return float(np.sum(navigable_mask) / total_land)

    @staticmethod
    def _calculate_coast_complexity(land_mask: np.ndarray) -> float:
        area = int(np.sum(land_mask))
        if area == 0:
            return 0.0

        eroded = ndimage.binary_erosion(land_mask, iterations=1)
        boundary = land_mask & ~eroded
        perimeter = int(np.sum(boundary))
        return float(perimeter / (2.0 * np.sqrt(np.pi * area)))

    @staticmethod
    def _calculate_terrain_variance(heightmap: np.ndarray, land_mask: np.ndarray) -> float:
        if not np.any(land_mask):
            return 0.0
        return float(np.std(heightmap[land_mask]))

    def _check_path_reachability(self, navigable_mask: np.ndarray) -> float:
        if not np.any(navigable_mask):
            return 0.0

        start = self._find_anchor_point(navigable_mask)
        if start is None:
            return 0.0

        queue = deque([start])
        visited = np.zeros_like(navigable_mask, dtype=bool)
        visited[start] = True

        while queue:
            y, x = queue.popleft()
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < self.map_size and 0 <= nx < self.map_size:
                    if navigable_mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        queue.append((ny, nx))

        return float(np.sum(visited) / max(float(np.sum(navigable_mask)), 1.0))

    @staticmethod
    def _calculate_land_ratio(land_mask: np.ndarray) -> float:
        return float(np.mean(land_mask))

    def _find_anchor_point(self, navigable_mask: np.ndarray) -> tuple[int, int] | None:
        center = np.array([self.map_size // 2, self.map_size // 2], dtype=np.int32)
        coordinates = np.argwhere(navigable_mask)
        if len(coordinates) == 0:
            return None
        distances = np.linalg.norm(coordinates - center[None, :], axis=1)
        return tuple(coordinates[np.argmin(distances)])

    def _is_boundary(self, y: int, x: int) -> bool:
        return y == 0 or x == 0 or y == self.map_size - 1 or x == self.map_size - 1


if __name__ == "__main__":
    from pcg_generator import PCGIslandGenerator

    generator = PCGIslandGenerator(map_size=64)
    evaluator = StructureEvaluator(map_size=64)
    heightmap = generator.generate_heightmap(generator.sample_random_params(np.random.default_rng(42)))
    metrics = evaluator.evaluate(heightmap)
    print("Structure metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
