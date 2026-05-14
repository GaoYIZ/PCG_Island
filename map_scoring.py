"""
Unified map scoring for offline dataset evaluation and RL rewards.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Sequence

import numpy as np


@dataclass
class ScoreBreakdown:
    total_score: float
    structure_score: float
    path_score: float
    land_score: float
    novelty_score: float
    component_scores: Dict[str, float]

    def as_dict(self) -> Dict[str, float]:
        data = {
            "total_score": self.total_score,
            "structure_score": self.structure_score,
            "path_score": self.path_score,
            "land_score": self.land_score,
            "novelty_score": self.novelty_score,
        }
        data.update(self.component_scores)
        return data


class MapScorer:
    """
    Single source of truth for terrain quality.

    The same scorer is used for:
    - dataset filtering and quality auditing
    - direct map size evaluation via land ratio
    - RL reward computation
    """

    def __init__(
        self,
        novelty_scale: float = 0.35,
        novelty_k: int = 5,
        # 高斯惩罚函数参数
        conn_alpha: float = 2.0,            # 连通性惩罚系数 α
        nav_mu: float = 0.90,               # 导航性理想值 μ
        nav_sigma: float = 0.15,            # 导航性标准差 σ
        coast_mu: float = 4.0,              # 海岸复杂度理想值
        coast_sigma: float = 1.5,           # 海岸复杂度标准差
        var_mu: float = 0.12,               # 地形方差理想值
        var_sigma: float = 0.05,            # 地形方差标准差
    ):
        # 高斯惩罚函数参数
        self.conn_alpha = float(conn_alpha)
        self.nav_mu = float(nav_mu)
        self.nav_sigma = float(nav_sigma)
        self.coast_mu = float(coast_mu)
        self.coast_sigma = float(coast_sigma)
        self.var_mu = float(var_mu)
        self.var_sigma = float(var_sigma)
        
        # 总权重配置：连通性和路径可达性为第一梯队，land为最后权重
        self.total_weights = {
            "connectivity": 0.25,    # 连通性 (第一梯队)
            "navigable": 0.10,       # 导航性 (第二梯队)
            "coast": 0.10,           # 海岸复杂度 (第三梯队)
            "variance": 0.10,        # 地形方差 (第三梯队)
            "path": 0.25,            # 路径可达性 (第一梯队)
            "land": 0.05,            # 陆地比例 (最后权重)
            "novelty": 0.15,         # 新颖性 (最后权重)
        }
        self.novelty_scale = float(novelty_scale)
        self.novelty_k = int(max(1, novelty_k))

    @staticmethod
    def _gaussian_penalty(
        value: float,
        mu: float,
        sigma: float,
    ) -> float:
        """高斯惩罚函数: R = exp(-(x - μ)² / σ²)"""
        return float(np.exp(-((value - mu) ** 2) / (sigma ** 2)))

    @staticmethod
    def _connectivity_gaussian(
        connectivity: float,
        alpha: float,
    ) -> float:
        """连通性高斯惩罚函数: R_conn = exp(-α(C-1)²)"""
        return float(np.exp(-alpha * (connectivity - 1.0) ** 2))

    def describe(self) -> Dict[str, object]:
        return {
            "gaussian_params": {
                "conn_alpha": self.conn_alpha,
                "nav_mu": self.nav_mu,
                "nav_sigma": self.nav_sigma,
                "coast_mu": self.coast_mu,
                "coast_sigma": self.coast_sigma,
                "var_mu": self.var_mu,
                "var_sigma": self.var_sigma,
            },
            "total_weights": self.total_weights,
            "novelty_scale": self.novelty_scale,
            "novelty_k": self.novelty_k,
        }

    def compute_novelty_score(
        self,
        feature_vector: Optional[Sequence[float]],
        history_vectors: Optional[Iterable[Sequence[float]]] = None,
    ) -> float:
        if feature_vector is None or history_vectors is None:
            return 0.0

        feature_vector = np.asarray(feature_vector, dtype=np.float32)
        history = [np.asarray(vector, dtype=np.float32) for vector in history_vectors]
        if len(history) == 0:
            return 0.0

        history_matrix = np.stack(history, axis=0)
        distances = np.linalg.norm(history_matrix - feature_vector[None, :], axis=1)
        positive_distances = distances[distances > 1e-6]
        if positive_distances.size == 0:
            return 0.0

        neighbor_count = min(self.novelty_k, positive_distances.size)
        nearest_distances = np.partition(positive_distances, neighbor_count - 1)[:neighbor_count]
        neighborhood_distance = float(np.mean(nearest_distances))
        adaptive_scale = max(
            float(np.median(positive_distances)),
            self.novelty_scale * float(np.sqrt(feature_vector.size)),
            1e-6,
        )
        novelty_score = 1.0 - float(np.exp(-neighborhood_distance / adaptive_scale))
        return float(np.clip(novelty_score, 0.0, 1.0))

    def score_metrics(
        self,
        metrics: Mapping[str, float],
        feature_vector: Optional[Sequence[float]] = None,
        history_vectors: Optional[Iterable[Sequence[float]]] = None,
    ) -> ScoreBreakdown:
        # 基础结构奖励（使用高斯惩罚函数）
        conn_score = self._connectivity_gaussian(
            float(np.clip(metrics["connectivity"], 0.0, 1.0)),
            self.conn_alpha
        )
        nav_score = self._gaussian_penalty(
            metrics["navigable_ratio"],
            self.nav_mu,
            self.nav_sigma
        )
        coast_score = self._gaussian_penalty(
            metrics["coast_complexity"],
            self.coast_mu,
            self.coast_sigma
        )
        variance_score = self._gaussian_penalty(
            metrics["terrain_variance"],
            self.var_mu,
            self.var_sigma
        )

        # 路径可达奖励（连续指标）
        path_score = float(np.clip(metrics["path_reachability"], 0.0, 1.0))
        
        # 陆地比例奖励（使用高斯惩罚函数，理想值 0.30）
        land_score = self._gaussian_penalty(
            metrics["land_ratio"],
            mu=0.30,
            sigma=0.12
        )
        
        # 新颖性奖励
        novelty_score = self.compute_novelty_score(feature_vector, history_vectors)

        # 总奖励函数：R = ΣwᵢRᵢ + λR_novelty（简化版，无嵌套）
        total_score = (
            self.total_weights["connectivity"] * conn_score
            + self.total_weights["navigable"] * nav_score
            + self.total_weights["coast"] * coast_score
            + self.total_weights["variance"] * variance_score
            + self.total_weights["path"] * path_score
            + self.total_weights["land"] * land_score
            + self.total_weights["novelty"] * novelty_score
        )

        # 结构分数（前四项的加权和，仅用于参考）
        structure_score = (
            conn_score * 0.25 + nav_score * 0.25 + coast_score * 0.25 + variance_score * 0.25
        )

        component_scores = {
            "connectivity_score": conn_score,
            "navigable_score": nav_score,
            "coast_score": coast_score,
            "variance_score": variance_score,
        }
        return ScoreBreakdown(
            total_score=float(total_score),
            structure_score=float(structure_score),
            path_score=path_score,
            land_score=float(land_score),
            novelty_score=float(novelty_score),
            component_scores=component_scores,
        )
