"""
结构评估模块 - 提取高度图的物理拓扑特征
包含：连通性、可导航比例、海岸复杂度、地形方差、路径可达性
"""

import numpy as np
from typing import Dict, Tuple
from scipy import ndimage


class StructureEvaluator:
    """岛屿结构评估器"""
    
    def __init__(self, map_size: int = 64):
        self.map_size = map_size
        self.water_threshold = 0.3  # 水域阈值
        self.slope_threshold = 30   # 坡度阈值（度）
        
    def evaluate(self, heightmap: np.ndarray) -> Dict[str, float]:
        """
        评估高度图的结构特征
        
        参数:
            heightmap: 归一化的高度图 (0-1)
        
        返回:
            包含各项指标的字典
        """
        metrics = {}
        
        # 1. 连通性检测
        metrics['connectivity'] = self._check_connectivity(heightmap)
        
        # 2. 可导航比例
        metrics['navigable_ratio'] = self._calculate_navigable_ratio(heightmap)
        
        # 3. 海岸复杂度
        metrics['coast_complexity'] = self._calculate_coast_complexity(heightmap)
        
        # 4. 地形方差
        metrics['terrain_variance'] = self._calculate_terrain_variance(heightmap)
        
        # 5. 路径可达性（简化版，仅检查是否存在可行路径）
        metrics['path_reachability'] = self._check_path_reachability(heightmap)
        
        return metrics
    
    def _check_connectivity(self, heightmap: np.ndarray) -> float:
        """
        检查陆地连通性
        返回1.0表示单连通，0.0表示多个不连通区域
        """
        # 二值化：陆地 vs 水域
        land_mask = heightmap > self.water_threshold
        
        # 使用连通分量标记
        labeled, num_features = ndimage.label(land_mask)
        
        # 如果只有一个连通分量，返回1.0
        if num_features == 1:
            return 1.0
        elif num_features == 0:
            return 0.0
        else:
            # 计算最大连通分量的占比
            component_sizes = [np.sum(labeled == i) for i in range(1, num_features + 1)]
            max_component = max(component_sizes)
            total_land = np.sum(land_mask)
            return max_component / total_land if total_land > 0 else 0.0
    
    def _calculate_navigable_ratio(self, heightmap: np.ndarray) -> float:
        """
        计算可导航比例（坡度<30°的单元格占比）
        """
        # 计算梯度（坡度）
        grad_y, grad_x = np.gradient(heightmap)
        
        # 计算坡度角度
        slope = np.arctan(np.sqrt(grad_x**2 + grad_y**2)) * 180 / np.pi
        
        # 只考虑陆地区域
        land_mask = heightmap > self.water_threshold
        
        # 计算可导航比例
        navigable = np.sum((slope < self.slope_threshold) & land_mask)
        total_land = np.sum(land_mask)
        
        return navigable / total_land if total_land > 0 else 0.0
    
    def _calculate_coast_complexity(self, heightmap: np.ndarray) -> float:
        """
        计算海岸复杂度（周长-面积比）
        C = P / (2 * sqrt(π * A))
        """
        # 二值化
        land_mask = heightmap > self.water_threshold
        
        # 计算面积
        area = np.sum(land_mask)
        
        if area == 0:
            return 0.0
        
        # 使用边缘检测计算周长
        # 通过形态学操作找到边界
        eroded = ndimage.binary_erosion(land_mask, iterations=1)
        boundary = land_mask & ~eroded
        perimeter = np.sum(boundary)
        
        # 计算复杂度指标
        complexity = perimeter / (2 * np.sqrt(np.pi * area))
        
        return complexity
    
    def _calculate_terrain_variance(self, heightmap: np.ndarray) -> float:
        """
        计算地形高程标准差
        """
        land_mask = heightmap > self.water_threshold
        
        if np.sum(land_mask) == 0:
            return 0.0
        
        # 只考虑陆地区域的高程变化
        land_heights = heightmap[land_mask]
        return np.std(land_heights)
    
    def _check_path_reachability(self, heightmap: np.ndarray) -> float:
        """
        简化的路径可达性检查
        检查是否存在从中心到边缘的可行路径
        """
        land_mask = heightmap > self.water_threshold
        
        # 找到中心点
        center_y, center_x = self.map_size // 2, self.map_size // 2
        
        # 如果中心不是陆地，尝试找到最近的陆地点
        if not land_mask[center_y, center_x]:
            # 寻找最近的陆地点
            distances = np.indices((self.map_size, self.map_size))
            dist_to_center = np.sqrt((distances[0] - center_y)**2 + 
                                    (distances[1] - center_x)**2)
            dist_to_center[~land_mask] = np.inf
            nearest = np.unravel_index(np.argmin(dist_to_center), dist_to_center.shape)
            center_y, center_x = nearest
        
        # 使用BFS检查是否能到达边界
        visited = np.zeros_like(land_mask, dtype=bool)
        queue = [(center_y, center_x)]
        visited[center_y, center_x] = True
        
        reached_boundary = False
        
        while queue:
            y, x = queue.pop(0)
            
            # 检查是否到达边界
            if y == 0 or y == self.map_size - 1 or x == 0 or x == self.map_size - 1:
                if land_mask[y, x]:
                    reached_boundary = True
                    break
            
            # 探索邻居
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < self.map_size and 0 <= nx < self.map_size:
                    if land_mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        queue.append((ny, nx))
        
        return 1.0 if reached_boundary else 0.0
    
    def get_feature_vector(self, heightmap: np.ndarray) -> np.ndarray:
        """
        获取结构特征向量
        """
        metrics = self.evaluate(heightmap)
        
        feature_vector = np.array([
            metrics['connectivity'],
            metrics['navigable_ratio'],
            metrics['coast_complexity'],
            metrics['terrain_variance'],
            metrics['path_reachability']
        ])
        
        return feature_vector


# 测试代码
if __name__ == "__main__":
    from pcg_generator import PCGIslandGenerator
    
    # 生成测试地图
    generator = PCGIslandGenerator(map_size=64)
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
    
    # 评估结构
    evaluator = StructureEvaluator(map_size=64)
    metrics = evaluator.evaluate(heightmap)
    
    print("Structure Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    feature_vec = evaluator.get_feature_vector(heightmap)
    print(f"\nFeature vector shape: {feature_vec.shape}")
    print(f"Feature vector: {feature_vec}")
