"""
PCG基座模块 - 基于Simplex噪声的岛屿地图生成
包含：Simplex噪声、fBm叠加、Domain Warping、Radial Falloff
"""

import numpy as np
from typing import Tuple, Dict


class SimplexNoise:
    """Simplex噪声实现"""
    
    def __init__(self, seed: int = None):
        self.perm = np.arange(256)
        if seed is not None:
            np.random.seed(seed)
        np.random.shuffle(self.perm)
        self.perm = np.concatenate([self.perm, self.perm])
        
    def noise2d(self, x: float, y: float) -> float:
        """2D Simplex噪声"""
        # 简化的Perlin噪声实现（用于演示）
        X = int(np.floor(x)) & 255
        Y = int(np.floor(y)) & 255
        x -= np.floor(x)
        y -= np.floor(y)
        
        u = self._fade(x)
        v = self._fade(y)
        
        A = self.perm[X] + Y
        B = self.perm[(X + 1) & 255] + Y
        
        return self._lerp(u, 
                         self._lerp(v, self._grad(self.perm[A & 255], x, y),
                                   self._grad(self.perm[B & 255], x, y - 1)),
                         self._lerp(v, self._grad(self.perm[(A + 1) & 255], x - 1, y),
                                   self._grad(self.perm[(B + 1) & 255], x - 1, y - 1)))
    
    def _fade(self, t: float) -> float:
        return t * t * t * (t * (t * 6 - 15) + 10)
    
    def _lerp(self, t: float, a: float, b: float) -> float:
        return a + t * (b - a)
    
    def _grad(self, hash_val: int, x: float, y: float) -> float:
        h = hash_val & 3
        if h == 0:
            return x + y
        elif h == 1:
            return -x + y
        elif h == 2:
            return x - y
        else:
            return -x - y


class PCGIslandGenerator:
    """岛屿地图生成器"""
    
    def __init__(self, map_size: int = 64):
        self.map_size = map_size
        self.noise = SimplexNoise()
        
    def generate_heightmap(self, params: Dict) -> np.ndarray:
        """
        生成高度图
        
        参数:
            params: 包含以下键的字典
                - f: 基础频率 (1-100)
                - A: 振幅缩放 (0.5-2.0)
                - N_octaves: 八度数 (1-8)
                - persistence: 持久性 (0-1)
                - lacunarity: 间隙度 (1.5-2.5)
                - seed: 随机种子
                - warp_strength: 扭曲强度 (0-1)
                - warp_frequency: 扭曲频率 (1-10)
                - falloff_radius: 衰减半径 (0-地图半径)
                - falloff_exponent: 衰减指数 (1-4)
        
        返回:
            归一化的高度图 (0-1)
        """
        # 设置参数默认值
        f = params.get('f', 10)
        A = params.get('A', 1.0)
        N_octaves = params.get('N_octaves', 4)
        persistence = params.get('persistence', 0.5)
        lacunarity = params.get('lacunarity', 2.0)
        seed = params.get('seed', 42)
        warp_strength = params.get('warp_strength', 0.5)
        warp_frequency = params.get('warp_frequency', 2)
        falloff_radius = params.get('falloff_radius', self.map_size / 2)
        falloff_exponent = params.get('falloff_exponent', 2)
        
        # 重新初始化噪声
        self.noise = SimplexNoise(seed)
        
        # 生成fBm噪声
        heightmap = self._fbm(f, N_octaves, persistence, lacunarity)
        
        # 应用Domain Warping
        if warp_strength > 0:
            heightmap = self._domain_warping(heightmap, warp_strength, warp_frequency)
        
        # 应用Radial Falloff
        heightmap = self._radial_falloff(heightmap, falloff_radius, falloff_exponent)
        
        # 归一化到[0, 1]
        heightmap = (heightmap - heightmap.min()) / (heightmap.max() - heightmap.min() + 1e-8)
        
        return heightmap
    
    def _fbm(self, frequency: float, octaves: int, persistence: float, 
             lacunarity: float) -> np.ndarray:
        """分数布朗运动(fBm)"""
        heightmap = np.zeros((self.map_size, self.map_size))
        amplitude = 1.0
        freq = frequency
        
        for _ in range(octaves):
            for y in range(self.map_size):
                for x in range(self.map_size):
                    nx = x / self.map_size * freq
                    ny = y / self.map_size * freq
                    heightmap[y, x] += self.noise.noise2d(nx, ny) * amplitude
            
            amplitude *= persistence
            freq *= lacunarity
        
        return heightmap
    
    def _domain_warping(self, heightmap: np.ndarray, strength: float, 
                       frequency: float) -> np.ndarray:
        """Domain Warping - 坐标扭曲"""
        warped = np.zeros_like(heightmap)
        
        for y in range(self.map_size):
            for x in range(self.map_size):
                # 计算扭曲偏移
                dx = self.noise.noise2d(x / self.map_size * frequency, 
                                       y / self.map_size * frequency) * strength
                dy = self.noise.noise2d(y / self.map_size * frequency + 100, 
                                       x / self.map_size * frequency + 100) * strength
                
                # 应用扭曲
                nx = int(np.clip(x + dx * 10, 0, self.map_size - 1))
                ny = int(np.clip(y + dy * 10, 0, self.map_size - 1))
                warped[y, x] = heightmap[ny, nx]
        
        return warped
    
    def _radial_falloff(self, heightmap: np.ndarray, radius: float, 
                       exponent: float) -> np.ndarray:
        """径向衰减 - 确保岛屿形态"""
        center_x, center_y = self.map_size / 2, self.map_size / 2
        max_dist = np.sqrt(center_x**2 + center_y**2)
        
        falloff_map = np.zeros((self.map_size, self.map_size))
        for y in range(self.map_size):
            for x in range(self.map_size):
                dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                normalized_dist = dist / radius
                falloff_map[y, x] = 1.0 - min(1.0, normalized_dist ** exponent)
        
        return heightmap * falloff_map
    
    def update_params(self, delta_params: Dict) -> Dict:
        """更新参数字典"""
        # 这里可以根据RL输出的动作更新参数
        pass


# 测试代码
if __name__ == "__main__":
    generator = PCGIslandGenerator(map_size=64)
    
    # 示例参数
    params = {
        'f': 10,
        'A': 1.0,
        'N_octaves': 4,
        'persistence': 0.5,
        'lacunarity': 2.0,
        'seed': 42,
        'warp_strength': 0.5,
        'warp_frequency': 2,
        'falloff_radius': 32,
        'falloff_exponent': 2
    }
    
    heightmap = generator.generate_heightmap(params)
    print(f"Heightmap shape: {heightmap.shape}")
    print(f"Heightmap range: [{heightmap.min():.3f}, {heightmap.max():.3f}]")
