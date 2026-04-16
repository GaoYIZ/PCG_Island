"""
强化学习环境 - Gymnasium环境封装
状态：[z, metrics]，动作：Δθ
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Tuple, Optional
from pcg_generator import PCGIslandGenerator
from structure_evaluator import StructureEvaluator


class IslandGenerationEnv(gym.Env):
    """岛屿生成强化学习环境"""
    
    metadata = {'render_modes': ['human', 'rgb_array']}
    
    def __init__(self, map_size: int = 64, max_steps: int = 50):
        super().__init__()
        
        self.map_size = map_size
        self.max_steps = max_steps
        
        # 初始化组件
        self.generator = PCGIslandGenerator(map_size=map_size)
        self.evaluator = StructureEvaluator(map_size=map_size)
        
        # PCG参数范围定义
        self.param_ranges = {
            'f': (1, 100),
            'A': (0.5, 2.0),
            'N_octaves': (3, 6),
            'persistence': (0.3, 0.7),
            'lacunarity': (1.5, 2.5),
            'warp_strength': (0.0, 1.0),
            'warp_frequency': (1, 10),
            'falloff_radius': (map_size * 0.3, map_size * 0.8),
            'falloff_exponent': (1, 4)
        }
        
        # 参数量
        self.n_params = len(self.param_ranges)
        
        # 动作空间：连续动作，每个参数的增量
        action_dim = self.n_params
        self.action_space = spaces.Box(
            low=-0.1, high=0.1, shape=(action_dim,), dtype=np.float32
        )
        
        # 状态空间：隐变量 + 结构指标（简化版，暂不使用VAE）
        # 为了demo，我们只使用结构指标作为状态
        state_dim = 5  # connectivity, navigable_ratio, coast_complexity, terrain_variance, path_reachability
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32
        )
        
        # 当前状态
        self.current_params = None
        self.current_heightmap = None
        self.current_metrics = None
        self.steps = 0
        
        # 历史新颖性缓冲区
        self.history_buffer = []
        self.buffer_size = 100
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        """重置环境"""
        super().reset(seed=seed)
        
        # 随机初始化参数
        self.current_params = self._random_params()
        
        # 生成初始高度图
        self.current_heightmap = self.generator.generate_heightmap(self.current_params)
        
        # 评估结构
        self.current_metrics = self.evaluator.evaluate(self.current_heightmap)
        
        self.steps = 0
        self.history_buffer = []
        
        # 状态 = 结构指标
        state = self._get_state()
        
        return state, {}
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """执行一步"""
        self.steps += 1
        
        # 更新参数
        self._update_params(action)
        
        # 生成新的高度图
        self.current_heightmap = self.generator.generate_heightmap(self.current_params)
        
        # 评估新结构
        new_metrics = self.evaluator.evaluate(self.current_heightmap)
        
        # 计算奖励
        reward = self._calculate_reward(new_metrics)
        
        # 检查是否终止
        terminated = self.steps >= self.max_steps
        truncated = False
        
        # 更新状态
        self.current_metrics = new_metrics
        state = self._get_state()
        
        # 添加到历史缓冲区
        feature_vec = self.evaluator.get_feature_vector(self.current_heightmap)
        if len(self.history_buffer) >= self.buffer_size:
            self.history_buffer.pop(0)
        self.history_buffer.append(feature_vec)
        
        info = {
            'metrics': new_metrics,
            'heightmap': self.current_heightmap,
            'params': self.current_params.copy()
        }
        
        return state, reward, terminated, truncated, info
    
    def _get_state(self) -> np.ndarray:
        """获取当前状态"""
        state = np.array([
            self.current_metrics['connectivity'],
            self.current_metrics['navigable_ratio'],
            self.current_metrics['coast_complexity'],
            self.current_metrics['terrain_variance'],
            self.current_metrics['path_reachability']
        ], dtype=np.float32)
        
        return state
    
    def _random_params(self) -> Dict:
        """随机生成参数"""
        params = {}
        for key, (low, high) in self.param_ranges.items():
            if key == 'N_octaves':
                params[key] = np.random.randint(int(low), int(high) + 1)
            else:
                params[key] = np.random.uniform(low, high)
        return params
    
    def _update_params(self, action: np.ndarray):
        """根据动作更新参数"""
        param_keys = list(self.param_ranges.keys())
        
        for i, key in enumerate(param_keys):
            low, high = self.param_ranges[key]
            
            # 应用增量
            if key == 'N_octaves':
                # 离散参数
                delta = action[i] * 2  # 放大增量
                new_value = self.current_params[key] + int(round(delta))
                new_value = np.clip(new_value, low, high)
            else:
                # 连续参数
                range_size = high - low
                delta = action[i] * range_size
                new_value = self.current_params[key] + delta
                new_value = np.clip(new_value, low, high)
            
            self.current_params[key] = new_value
    
    def _calculate_reward(self, metrics: Dict) -> float:
        """
        计算奖励函数
        R = w1*R_struct + w2*R_reach + w3*R_novelty
        """
        # 1. 结构奖励
        r_connectivity = metrics['connectivity']
        r_navigable = np.exp(-((metrics['navigable_ratio'] - 0.7) ** 2) / 0.1)
        r_coast = np.exp(-((metrics['coast_complexity'] - 1.2) ** 2) / 0.2)
        r_variance = np.exp(-((metrics['terrain_variance'] - 0.15) ** 2) / 0.01)
        
        r_struct = 0.3 * r_connectivity + 0.3 * r_navigable + 0.2 * r_coast + 0.2 * r_variance
        
        # 2. 路径可达性奖励
        r_reach = metrics['path_reachability']
        
        # 3. 新颖性奖励（与历史的距离）
        if len(self.history_buffer) > 0:
            current_features = self.evaluator.get_feature_vector(self.current_heightmap)
            distances = [np.linalg.norm(current_features - hist) 
                        for hist in self.history_buffer]
            r_novelty = min(distances) / 10.0  # 归一化
        else:
            r_novelty = 0.5
        
        # 总奖励
        reward = 0.5 * r_struct + 0.3 * r_reach + 0.2 * r_novelty
        
        return reward
    
    def render(self, mode='human'):
        """渲染环境（可选）"""
        if mode == 'human':
            print(f"Step: {self.steps}")
            print(f"Metrics: {self.current_metrics}")
            print(f"Reward: {self._calculate_reward(self.current_metrics):.4f}")


# 测试代码
if __name__ == "__main__":
    env = IslandGenerationEnv(map_size=64, max_steps=10)
    
    state, info = env.reset()
    print(f"Initial state: {state}")
    print(f"State shape: {state.shape}")
    
    # 随机动作测试
    for i in range(5):
        action = env.action_space.sample()
        state, reward, terminated, truncated, info = env.step(action)
        print(f"\nStep {i+1}:")
        print(f"  Reward: {reward:.4f}")
        print(f"  State: {state}")
        print(f"  Terminated: {terminated}")
        
        if terminated:
            break
    
    env.close()
