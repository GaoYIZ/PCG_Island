"""
CMA-ES (Covariance Matrix Adaptation Evolution Strategy) 基线模型
用于岛屿生成参数优化的无梯度搜索算法
"""

import numpy as np
from typing import Dict, List, Callable, Tuple
from pcg_generator import PCGIslandGenerator
from structure_evaluator import StructureEvaluator


class CMAESOptimizer:
    """CMA-ES优化器"""
    
    def __init__(self, 
                 param_ranges: Dict[str, Tuple[float, float]],
                 sigma0: float = 0.5,
                 pop_size: int = None):
        """
        初始化CMA-ES
        
        参数:
            param_ranges: 参数范围字典 {param_name: (min, max)}
            sigma0: 初始步长
            pop_size: 种群大小（默认4 + floor(3 * ln(n))）
        """
        self.param_names = list(param_ranges.keys())
        self.param_ranges = param_ranges
        self.n_params = len(param_ranges)
        
        # 设置种群大小
        if pop_size is None:
            self.pop_size = 4 + int(3 * np.log(self.n_params))
        else:
            self.pop_size = pop_size
        
        # CMA-ES参数
        self.sigma = sigma0
        self.mu = self.pop_size // 2
        self.weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights /= np.sum(self.weights)
        self.mu_eff = np.sum(self.weights)**2 / np.sum(self.weights**2)
        
        # 学习率
        self.cc = (4 + self.mu_eff / self.n_params) / (self.n_params + 4 + 2 * self.mu_eff / self.n_params)
        self.cs = (self.mu_eff + 2) / (self.n_params + self.mu_eff + 5)
        self.c1 = 2 / ((self.n_params + 1.3)**2 + self.mu_eff)
        self.cmu = min(1 - self.c1, 2 * (self.mu_eff - 2 + 1/self.mu_eff) / ((self.n_params + 2)**2 + self.mu_eff))
        self.damps = 1 + 2 * max(0, np.sqrt((self.mu_eff - 1) / (self.n_params + 1)) - 1) + self.cs
        
        # 初始化状态
        self.xmean = self._random_params()
        self.pc = np.zeros(self.n_params)
        self.ps = np.zeros(self.n_params)
        self.C = np.eye(self.n_params)
        self.chiN = np.sqrt(self.n_params) * (1 - 1/(4*self.n_params) + 1/(21*self.n_params**2))
        
        # 特征值分解
        self.eigenvalues = None
        self.eigenvectors = None
        self.invsqrtC = None
        
        self._update_eigen()
        
        # 评估函数
        self.generator = PCGIslandGenerator(map_size=64)
        self.evaluator = StructureEvaluator(map_size=64)
        
    def _random_params(self) -> np.ndarray:
        """随机初始化参数"""
        params = []
        for name in self.param_names:
            low, high = self.param_ranges[name]
            params.append(np.random.uniform(low, high))
        return np.array(params)
    
    def _update_eigen(self):
        """更新特征值分解"""
        self.C = (self.C + self.C.T) / 2
        self.eigenvalues, self.eigenvectors = np.linalg.eigh(self.C)
        self.eigenvalues = np.maximum(self.eigenvalues, 0)
        self.invsqrtC = self.eigenvectors @ np.diag(1 / np.sqrt(self.eigenvalues)) @ self.eigenvectors.T
    
    def _sample_population(self) -> np.ndarray:
        """采样种群"""
        population = np.zeros((self.pop_size, self.n_params))
        for i in range(self.pop_size):
            z = np.random.randn(self.n_params)
            population[i] = self.xmean + self.sigma * self.eigenvectors @ np.sqrt(np.diag(self.eigenvalues)) @ z
        return population
    
    def _clip_params(self, params: np.ndarray) -> np.ndarray:
        """裁剪参数到合法范围"""
        clipped = params.copy()
        for i, name in enumerate(self.param_names):
            low, high = self.param_ranges[name]
            clipped[i] = np.clip(clipped[i], low, high)
        return clipped
    
    def _params_to_dict(self, params: np.ndarray) -> Dict:
        """参数数组转字典"""
        result = {}
        for i, name in enumerate(self.param_names):
            if name == 'N_octaves':
                result[name] = int(round(params[i]))
            else:
                result[name] = params[i]
        return result
    
    def evaluate_fitness(self, params: np.ndarray) -> float:
        """
        评估适应度（奖励函数）
        
        参数:
            params: 参数数组
        
        返回:
            适应度值（越高越好）
        """
        param_dict = self._params_to_dict(params)
        param_dict['seed'] = np.random.randint(0, 10000)
        
        # 生成高度图
        heightmap = self.generator.generate_heightmap(param_dict)
        
        # 评估结构
        metrics = self.evaluator.evaluate(heightmap)
        
        # 计算奖励（与RL环境一致）
        r_conn = metrics['connectivity']
        r_nav = np.exp(-((metrics['navigable_ratio'] - 0.7)**2) / 0.1)
        r_coast = np.exp(-((metrics['coast_complexity'] - 1.2)**2) / 0.2)
        r_var = np.exp(-((metrics['terrain_variance'] - 0.15)**2) / 0.01)
        r_struct = 0.3 * r_conn + 0.3 * r_nav + 0.2 * r_coast + 0.2 * r_var
        r_reach = metrics['path_reachability']
        
        fitness = 0.5 * r_struct + 0.3 * r_reach + 0.2 * 0.5  # 固定novelty
        
        return fitness
    
    def optimize(self, generations: int = 50, verbose: bool = True) -> Tuple[np.ndarray, float]:
        """
        运行CMA-ES优化
        
        参数:
            generations: 迭代代数
            verbose: 是否打印进度
        
        返回:
            最优参数和最优适应度
        """
        best_fitness = -np.inf
        best_params = self.xmean.copy()
        
        for gen in range(generations):
            # 采样种群
            population = self._sample_population()
            
            # 评估适应度
            fitnesses = np.array([self.evaluate_fitness(self._clip_params(p)) for p in population])
            
            # 排序
            indices = np.argsort(fitnesses)[::-1]
            population = population[indices]
            fitnesses = fitnesses[indices]
            
            # 更新最优解
            if fitnesses[0] > best_fitness:
                best_fitness = fitnesses[0]
                best_params = self._clip_params(population[0]).copy()
            
            # 更新均值
            xold = self.xmean.copy()
            self.xmean = np.sum(self.weights[:, np.newaxis] * population[:self.mu], axis=0)
            self.xmean = self._clip_params(self.xmean)
            
            # 更新进化路径
            z = (self.xmean - xold) / self.sigma
            self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.mu_eff) * z
            
            hsig = float(np.linalg.norm(self.ps) / np.sqrt(1 - (1 - self.cs)**(2*(gen+1))) < 
                        (1.4 + 2/(self.n_params+1)) * self.chiN)
            
            self.pc = (1 - self.cc) * self.pc + hsig * np.sqrt(self.cc * (2 - self.cc) * self.mu_eff) * z
            
            # 更新协方差矩阵
            diff = population[:self.mu] - xold
            self.C = (1 - self.c1 - self.cmu) * self.C + \
                    self.c1 * (np.outer(self.pc, self.pc) + (1 - hsig) * self.cc * (2 - self.cc) * self.C) + \
                    self.cmu * np.sum(self.weights[:self.mu, np.newaxis, np.newaxis] * 
                                     np.einsum('ni,nj->nij', diff, diff), axis=0)
            
            # 更新步长
            self.sigma *= np.exp((self.cs / self.damps) * (np.linalg.norm(self.ps) / self.chiN - 1))
            
            # 更新特征值分解
            if gen % 5 == 0:
                self._update_eigen()
            
            if verbose and (gen + 1) % 10 == 0:
                print(f"Generation {gen+1}/{generations}, "
                      f"Best Fitness: {best_fitness:.4f}, "
                      f"Mean Fitness: {np.mean(fitnesses[:10]):.4f}")
        
        return best_params, best_fitness
    
    def generate_islands(self, n_islands: int = 20) -> List[Tuple[np.ndarray, Dict]]:
        """
        使用优化后的参数生成多个岛屿
        
        参数:
            n_islands: 生成岛屿数量
        
        返回:
            列表，每个元素为(heightmap, metrics)元组
        """
        results = []
        for i in range(n_islands):
            param_dict = self._params_to_dict(self.xmean)
            param_dict['seed'] = i * 100
            
            heightmap = self.generator.generate_heightmap(param_dict)
            metrics = self.evaluator.evaluate(heightmap)
            
            results.append((heightmap, metrics))
        
        return results


# 测试代码
if __name__ == "__main__":
    # 定义参数范围
    param_ranges = {
        'f': (5, 20),
        'A': (0.5, 1.5),
        'N_octaves': (3, 6),
        'persistence': (0.3, 0.7),
        'lacunarity': (1.5, 2.5),
        'warp_strength': (0.2, 0.8),
        'warp_frequency': (1, 5),
        'falloff_radius': (20, 40),
        'falloff_exponent': (1.5, 3)
    }
    
    # 创建优化器
    optimizer = CMAESOptimizer(param_ranges, sigma0=0.5, pop_size=20)
    
    # 运行优化
    print("开始CMA-ES优化...")
    best_params, best_fitness = optimizer.optimize(generations=30, verbose=True)
    
    print(f"\n最优适应度: {best_fitness:.4f}")
    print(f"最优参数:")
    for name, value in zip(optimizer.param_names, best_params):
        print(f"  {name}: {value:.4f}")
    
    # 生成测试岛屿
    print("\n生成测试岛屿...")
    islands = optimizer.generate_islands(n_islands=5)
    
    for i, (heightmap, metrics) in enumerate(islands):
        print(f"\n岛屿 {i+1}:")
        print(f"  连通性: {metrics['connectivity']:.4f}")
        print(f"  可导航比例: {metrics['navigable_ratio']:.4f}")
        print(f"  海岸复杂度: {metrics['coast_complexity']:.4f}")
    
    print("\n✅ CMA-ES优化完成！")
