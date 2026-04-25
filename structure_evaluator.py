"""
Terrain structure evaluator used for dataset curation and reward computation.
"""

from __future__ import annotations

from heapq import heappop, heappush
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

    def __init__(
        self,
        map_size: int = 64,
        water_threshold: float = 0.30,
        slope_threshold: float = 30.0,
        height_scale: float = 3.5,
        cell_size: float = 1.0,
        path_sample_points: int = 6,
        max_path_pairs: int = 8,
        slope_cost_weight: float = 0.35,
    ):
        self.map_size = map_size
        self.water_threshold = float(water_threshold)
        self.slope_threshold = float(slope_threshold)
        self.height_scale = float(height_scale)
        self.cell_size = float(cell_size)
        self.path_sample_points = int(path_sample_points)
        self.max_path_pairs = int(max_path_pairs)
        self.slope_cost_weight = float(slope_cost_weight)

    def evaluate(self, heightmap: np.ndarray) -> Dict[str, float]:
        land_mask = heightmap > self.water_threshold
        slope = self._compute_slope(heightmap)
        navigable_mask = land_mask & (slope < self.slope_threshold)
        connectivity, component_count = self._analyze_connectivity(land_mask)
        path_reachability = self._check_path_reachability(navigable_mask, slope=slope)

        return {
            "connectivity": connectivity,
            "component_count": float(component_count),
            "navigable_ratio": self._calculate_navigable_ratio(land_mask, navigable_mask),
            "coast_complexity": self._calculate_coast_complexity(land_mask),
            "terrain_variance": self._calculate_terrain_variance(heightmap, land_mask),
            "path_reachability": path_reachability,
            "path_exists": float(path_reachability > 0.0),
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
        elevation = np.asarray(heightmap, dtype=np.float32) * self.height_scale
        grad_y, grad_x = np.gradient(elevation, self.cell_size, self.cell_size)
        return np.degrees(np.arctan(np.sqrt(grad_x**2 + grad_y**2)))

    def _analyze_connectivity(self, land_mask: np.ndarray) -> tuple[float, int]:
        if not np.any(land_mask):
            return 0.0, 0

        structure = np.ones((3, 3), dtype=np.int32)
        labeled, num_components = ndimage.label(land_mask, structure=structure)
        if num_components <= 1:
            return 1.0, int(num_components)

        component_sizes = ndimage.sum(land_mask, labeled, index=np.arange(1, num_components + 1))
        total_land = float(np.sum(land_mask))
        connectivity = float(np.max(component_sizes) / max(total_land, 1.0))
        return connectivity, int(num_components)

    def _check_connectivity(self, land_mask: np.ndarray) -> float:
        connectivity, _ = self._analyze_connectivity(land_mask)
        return connectivity

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

        # Estimate coastline length from land-water transitions on the pixel grid.
        padded = np.pad(land_mask.astype(np.int8), pad_width=1, mode="constant", constant_values=0)
        perimeter = float(
            np.sum(np.abs(np.diff(padded, axis=0))) + np.sum(np.abs(np.diff(padded, axis=1)))
        )
        return float(perimeter / (2.0 * np.sqrt(np.pi * area)))

    @staticmethod
    def _calculate_terrain_variance(heightmap: np.ndarray, land_mask: np.ndarray) -> float:
        if not np.any(land_mask):
            return 0.0
        return float(np.std(heightmap[land_mask]))

    def _check_path_reachability(self, navigable_mask: np.ndarray, slope: np.ndarray | None = None) -> float:
        if not np.any(navigable_mask):
            return 0.0

        coordinates = np.argwhere(navigable_mask)
        if len(coordinates) < 2:
            return 0.0

        sample_points = self._select_representative_points(coordinates)
        if len(sample_points) < 2:
            return 0.0

        tested_pairs = 0
        successful_pairs = 0
        slope = np.zeros_like(navigable_mask, dtype=np.float32) if slope is None else slope

        for idx, start in enumerate(sample_points):
            for goal in sample_points[idx + 1 :]:
                tested_pairs += 1
                if self._astar_has_path(start, goal, navigable_mask, slope):
                    successful_pairs += 1
                if tested_pairs >= self.max_path_pairs:
                    break
            if tested_pairs >= self.max_path_pairs:
                break

        if tested_pairs == 0:
            return 0.0
        return float(successful_pairs / tested_pairs)

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

    def _select_representative_points(self, coordinates: np.ndarray) -> list[tuple[int, int]]:
        max_points = min(self.path_sample_points, len(coordinates))
        if max_points <= 0:
            return []

        center = np.array([self.map_size / 2.0, self.map_size / 2.0], dtype=np.float32)
        distances_to_center = np.linalg.norm(coordinates - center[None, :], axis=1)
        selected_indices = [int(np.argmin(distances_to_center))]

        while len(selected_indices) < max_points:
            selected_coords = coordinates[selected_indices]
            min_distances = np.min(
                np.linalg.norm(coordinates[:, None, :] - selected_coords[None, :, :], axis=2),
                axis=1,
            )
            min_distances[selected_indices] = -1.0
            next_index = int(np.argmax(min_distances))
            if min_distances[next_index] <= 0.0:
                break
            selected_indices.append(next_index)

        return [tuple(int(value) for value in coordinates[index]) for index in selected_indices]

    def _astar_has_path(
        self,
        start: tuple[int, int],
        goal: tuple[int, int],
        navigable_mask: np.ndarray,
        slope: np.ndarray,
    ) -> bool:
        if start == goal:
            return True

        open_heap: list[tuple[float, float, tuple[int, int]]] = []
        heappush(open_heap, (self._heuristic(start, goal), 0.0, start))
        best_cost = {start: 0.0}

        while open_heap:
            _, current_cost, current = heappop(open_heap)
            if current == goal:
                return True
            if current_cost > best_cost.get(current, float("inf")) + 1e-8:
                continue

            y, x = current
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = y + dy, x + dx
                if not (0 <= ny < self.map_size and 0 <= nx < self.map_size):
                    continue
                if not navigable_mask[ny, nx]:
                    continue

                neighbor = (ny, nx)
                step_cost = 1.0 + self.slope_cost_weight * float(
                    (slope[y, x] + slope[ny, nx]) / max(2.0 * self.slope_threshold, 1e-6)
                )
                new_cost = current_cost + step_cost
                if new_cost + 1e-8 < best_cost.get(neighbor, float("inf")):
                    best_cost[neighbor] = new_cost
                    priority = new_cost + self._heuristic(neighbor, goal)
                    heappush(open_heap, (priority, new_cost, neighbor))

        return False

    @staticmethod
    def _heuristic(start: tuple[int, int], goal: tuple[int, int]) -> float:
        return float(abs(start[0] - goal[0]) + abs(start[1] - goal[1]))

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
