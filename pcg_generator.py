"""
PCG terrain generator for island heightmaps.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np


class SimplexNoise:
    """Lightweight gradient-noise generator used as the terrain backbone."""

    def __init__(self, seed: Optional[int] = None):
        self.perm = np.arange(256, dtype=np.int32)
        rng = np.random.default_rng(seed)
        rng.shuffle(self.perm)
        self.perm = np.concatenate([self.perm, self.perm])

    def noise2d(self, x: float, y: float) -> float:
        x0 = int(np.floor(x)) & 255
        y0 = int(np.floor(y)) & 255
        xf = x - np.floor(x)
        yf = y - np.floor(y)

        u = self._fade(xf)
        v = self._fade(yf)

        aa = self.perm[self.perm[x0] + y0]
        ab = self.perm[self.perm[x0] + y0 + 1]
        ba = self.perm[self.perm[x0 + 1] + y0]
        bb = self.perm[self.perm[x0 + 1] + y0 + 1]

        x1 = self._lerp(self._grad(aa, xf, yf), self._grad(ba, xf - 1.0, yf), u)
        x2 = self._lerp(self._grad(ab, xf, yf - 1.0), self._grad(bb, xf - 1.0, yf - 1.0), u)
        return self._lerp(x1, x2, v)

    @staticmethod
    def _fade(t: float) -> float:
        return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)

    @staticmethod
    def _lerp(a: float, b: float, t: float) -> float:
        return a + t * (b - a)

    @staticmethod
    def _grad(hash_value: int, x: float, y: float) -> float:
        h = hash_value & 3
        if h == 0:
            return x + y
        if h == 1:
            return -x + y
        if h == 2:
            return x - y
        return -x - y


class PCGIslandGenerator:
    """Generates normalized island heightmaps from parameterized noise."""

    def __init__(self, map_size: int = 128):
        self.map_size = map_size
        self.noise = SimplexNoise(seed=42)

    @staticmethod
    def get_param_ranges(map_size: int) -> Dict[str, tuple[float, float]]:
        return {
            "f": (1.0, 100.0),
            "A": (0.5, 2.0),
            "N_octaves": (3.0, 6.0),
            "persistence": (0.3, 0.7),
            "lacunarity": (1.5, 2.5),
            "warp_strength": (0.0, 1.0),
            "warp_frequency": (1.0, 10.0),
            "falloff_radius": (map_size * 0.30, map_size * 0.80),
            "falloff_exponent": (1.0, 4.0),
        }

    @classmethod
    def get_sampling_ranges(cls, map_size: int, profile: str = "uniform") -> Dict[str, tuple[float, float]]:
        base_ranges = cls.get_param_ranges(map_size)
        if profile == "uniform":
            return base_ranges
        if profile == "island":
            return {
                "f": (6.0, 28.0),
                "A": (0.70, 1.40),
                "N_octaves": (3.0, 5.0),
                "persistence": (0.35, 0.60),
                "lacunarity": (1.60, 2.30),
                "warp_strength": (0.05, 0.65),
                "warp_frequency": (2.0, 7.5),
                "falloff_radius": (map_size * 0.24, map_size * 0.48),
                "falloff_exponent": (1.60, 3.20),
            }
        raise ValueError(f"Unsupported sampling profile: {profile}")

    def sample_random_params(
        self,
        rng: Optional[np.random.Generator] = None,
        profile: str = "uniform",
    ) -> Dict[str, float]:
        rng = rng or np.random.default_rng()
        params = {}
        for name, (low, high) in self.get_sampling_ranges(self.map_size, profile=profile).items():
            value = float(rng.uniform(low, high))
            if name == "N_octaves":
                value = int(round(value))
            params[name] = value
        params["seed"] = int(rng.integers(0, 1_000_000))
        return params

    def generate_heightmap(self, params: Dict[str, float]) -> np.ndarray:
        param_ranges = self.get_param_ranges(self.map_size)
        defaults = {name: (low + high) * 0.5 for name, (low, high) in param_ranges.items()}

        frequency = float(params.get("f", defaults["f"]))
        amplitude_scale = float(params.get("A", defaults["A"]))
        octaves = int(round(params.get("N_octaves", defaults["N_octaves"])))
        persistence = float(params.get("persistence", defaults["persistence"]))
        lacunarity = float(params.get("lacunarity", defaults["lacunarity"]))
        seed = int(params.get("seed", 42))
        warp_strength = float(params.get("warp_strength", defaults["warp_strength"]))
        warp_frequency = float(params.get("warp_frequency", defaults["warp_frequency"]))
        falloff_radius = float(params.get("falloff_radius", defaults["falloff_radius"]))
        falloff_exponent = float(params.get("falloff_exponent", defaults["falloff_exponent"]))

        self.noise = SimplexNoise(seed=seed)

        heightmap = self._fbm(
            frequency=frequency,
            octaves=octaves,
            persistence=persistence,
            lacunarity=lacunarity,
        )
        if warp_strength > 0.0:
            heightmap = self._domain_warping(heightmap, strength=warp_strength, frequency=warp_frequency)
        heightmap = self._normalize(heightmap)

        heightmap = np.clip(0.5 + (heightmap - 0.5) * amplitude_scale, 0.0, 1.0)
        heightmap = self._radial_falloff(heightmap, radius=falloff_radius, exponent=falloff_exponent)
        heightmap = np.clip(heightmap, 0.0, 1.0)
        return heightmap.astype(np.float32)

    def _fbm(
        self,
        frequency: float,
        octaves: int,
        persistence: float,
        lacunarity: float,
    ) -> np.ndarray:
        heightmap = np.zeros((self.map_size, self.map_size), dtype=np.float32)
        current_amplitude = 1.0
        current_frequency = frequency

        for _ in range(octaves):
            for y in range(self.map_size):
                for x in range(self.map_size):
                    nx = x / self.map_size * current_frequency
                    ny = y / self.map_size * current_frequency
                    heightmap[y, x] += self.noise.noise2d(nx, ny) * current_amplitude

            current_amplitude *= persistence
            current_frequency *= lacunarity

        return heightmap

    def _domain_warping(self, heightmap: np.ndarray, strength: float, frequency: float) -> np.ndarray:
        warped = np.zeros_like(heightmap)

        for y in range(self.map_size):
            for x in range(self.map_size):
                dx = self.noise.noise2d(x / self.map_size * frequency, y / self.map_size * frequency) * strength
                dy = self.noise.noise2d(
                    x / self.map_size * frequency + 31.7,
                    y / self.map_size * frequency + 11.3,
                ) * strength
                nx = int(np.clip(x + dx * 10.0, 0, self.map_size - 1))
                ny = int(np.clip(y + dy * 10.0, 0, self.map_size - 1))
                warped[y, x] = heightmap[ny, nx]

        return warped

    def _radial_falloff(self, heightmap: np.ndarray, radius: float, exponent: float) -> np.ndarray:
        center_x = self.map_size / 2.0
        center_y = self.map_size / 2.0
        falloff_map = np.zeros_like(heightmap)

        for y in range(self.map_size):
            for x in range(self.map_size):
                distance = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
                normalized_distance = distance / max(radius, 1e-6)
                falloff_map[y, x] = 1.0 - min(1.0, normalized_distance**exponent)

        return heightmap * falloff_map

    @staticmethod
    def _normalize(heightmap: np.ndarray) -> np.ndarray:
        min_value = float(heightmap.min())
        max_value = float(heightmap.max())
        return (heightmap - min_value) / (max_value - min_value + 1e-8)


if __name__ == "__main__":
    generator = PCGIslandGenerator(map_size=128)
    params = generator.sample_random_params(np.random.default_rng(42))
    heightmap = generator.generate_heightmap(params)
    print(f"Heightmap shape: {heightmap.shape}")
    print(f"Heightmap range: [{heightmap.min():.3f}, {heightmap.max():.3f}]")
