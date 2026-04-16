"""
多样性分析工具 - 计算隐空间离散度和可视化
"""

import numpy as np
import torch
from typing import List, Dict
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


class DiversityAnalyzer:
    """多样性分析器"""
    
    def __init__(self):
        pass
    
    @staticmethod
    def compute_latent_discreteness(latent_vectors: np.ndarray) -> float:
        """
        计算隐空间离散度
        
        D_latent = 1/|B| Σ_{z_i∈B} min_{z_j∈B, j≠i} ||z_i - z_j||
        
        参数:
            latent_vectors: 形状为 (N, d) 的隐向量数组
        
        返回:
            平均最近邻距离
        """
        n_samples = len(latent_vectors)
        
        if n_samples < 2:
            return 0.0
        
        # 计算每对样本之间的距离
        min_distances = []
        
        for i in range(n_samples):
            # 计算第i个样本到其他所有样本的距离
            distances = np.linalg.norm(latent_vectors - latent_vectors[i], axis=1)
            
            # 排除自身（距离为0）
            distances[i] = np.inf
            
            # 找到最近邻距离
            min_dist = np.min(distances)
            min_distances.append(min_dist)
        
        # 平均最近邻距离
        D_latent = np.mean(min_distances)
        
        return D_latent
    
    @staticmethod
    def compute_pairwise_distances(latent_vectors: np.ndarray) -> np.ndarray:
        """
        计算成对距离矩阵
        
        参数:
            latent_vectors: 形状为 (N, d) 的隐向量数组
        
        返回:
            距离矩阵 (N, N)
        """
        n_samples = len(latent_vectors)
        dist_matrix = np.zeros((n_samples, n_samples))
        
        for i in range(n_samples):
            for j in range(i+1, n_samples):
                dist = np.linalg.norm(latent_vectors[i] - latent_vectors[j])
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist
        
        return dist_matrix
    
    @staticmethod
    def visualize_latent_space_2d(latent_vectors: np.ndarray, 
                                  labels: List[str] = None,
                                  method: str = 'PCA',
                                  save_path: str = None):
        """
        2D可视化隐空间
        
        参数:
            latent_vectors: 形状为 (N, d) 的隐向量数组
            labels: 标签列表（可选）
            method: 降维方法 ('PCA' 或 't-SNE')
            save_path: 保存路径（可选）
        """
        n_samples = len(latent_vectors)
        
        if n_samples < 3:
            print("样本数量太少，无法可视化")
            return
        
        # 降维到2D
        if method == 'PCA':
            reducer = PCA(n_components=2)
            embedding = reducer.fit_transform(latent_vectors)
            explained_var = reducer.explained_variance_ratio_.sum()
            title = f'PCA Visualization (Explained Var: {explained_var:.2%})'
        elif method == 't-SNE':
            perplexity = min(30, n_samples - 1)
            reducer = TSNE(n_components=2, perplexity=perplexity, random_state=42)
            embedding = reducer.fit_transform(latent_vectors)
            title = 't-SNE Visualization'
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # 可视化
        plt.figure(figsize=(10, 8))
        
        if labels is not None:
            scatter = plt.scatter(embedding[:, 0], embedding[:, 1], 
                                 c=range(n_samples), cmap='viridis', 
                                 alpha=0.6, s=50)
            plt.colorbar(scatter, label='Sample Index')
        else:
            plt.scatter(embedding[:, 0], embedding[:, 1], 
                       c='blue', alpha=0.6, s=50)
        
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"可视化已保存到: {save_path}")
        
        plt.show()
    
    @staticmethod
    def interpolate_latent(vae_model, z1: np.ndarray, z2: np.ndarray, 
                          n_steps: int = 10, device: str = 'cpu'):
        """
        隐空间插值
        
        参数:
            vae_model: 训练好的VAE模型
            z1: 起始隐向量
            z2: 终止隐向量
            n_steps: 插值步数
            device: 设备
        
        返回:
            插值生成的图像列表
        """
        vae_model.eval()
        
        # 线性插值
        alphas = np.linspace(0, 1, n_steps)
        interpolated_images = []
        
        with torch.no_grad():
            for alpha in alphas:
                # 插值
                z_interp = (1 - alpha) * z1 + alpha * z2
                z_tensor = torch.FloatTensor(z_interp).unsqueeze(0).to(device)
                
                # 解码
                reconstructed = vae_model.decode(z_tensor)
                image = reconstructed.squeeze().cpu().numpy()
                
                interpolated_images.append(image)
        
        return interpolated_images
    
    @staticmethod
    def visualize_interpolation(interpolated_images: List[np.ndarray], 
                               save_path: str = None):
        """
        可视化插值序列
        
        参数:
            interpolated_images: 插值图像列表
            save_path: 保存路径（可选）
        """
        n_steps = len(interpolated_images)
        
        fig, axes = plt.subplots(1, n_steps, figsize=(3*n_steps, 3))
        
        if n_steps == 1:
            axes = [axes]
        
        for i, img in enumerate(interpolated_images):
            axes[i].imshow(img, cmap='terrain')
            axes[i].set_title(f'Step {i+1}')
            axes[i].axis('off')
        
        plt.suptitle('Latent Space Interpolation', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"插值可视化已保存到: {save_path}")
        
        plt.show()
    
    @staticmethod
    def compare_diversity(methods_data: Dict[str, np.ndarray]):
        """
        对比不同方法的多样性
        
        参数:
            methods_data: 字典 {method_name: latent_vectors}
        
        返回:
            多样性指标字典
        """
        diversity_scores = {}
        
        for method_name, latents in methods_data.items():
            D_latent = DiversityAnalyzer.compute_latent_discreteness(latents)
            diversity_scores[method_name] = D_latent
            
            print(f"{method_name:20s}: D_latent = {D_latent:.4f}")
        
        # 可视化对比
        plt.figure(figsize=(10, 6))
        methods = list(diversity_scores.keys())
        scores = list(diversity_scores.values())
        
        bars = plt.bar(methods, scores, color=['skyblue', 'lightcoral', 'lightgreen'])
        plt.xlabel('Method')
        plt.ylabel('D_latent (Average Nearest Neighbor Distance)')
        plt.title('Diversity Comparison Across Methods')
        plt.grid(True, alpha=0.3, axis='y')
        
        # 在柱状图上添加数值
        for bar, score in zip(bars, scores):
            plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('diversity_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        return diversity_scores


# 测试代码
if __name__ == "__main__":
    # 生成测试数据
    np.random.seed(42)
    n_samples = 50
    latent_dim = 32
    
    # 模拟不同方法的latent vectors
    random_latents = np.random.randn(n_samples, latent_dim) * 0.5
    clustered_latents = np.random.randn(n_samples, latent_dim) * 0.2
    diverse_latents = np.random.randn(n_samples, latent_dim) * 1.0
    
    analyzer = DiversityAnalyzer()
    
    # 计算离散度
    print("计算多样性指标:")
    print("=" * 50)
    D_random = analyzer.compute_latent_discreteness(random_latents)
    D_clustered = analyzer.compute_latent_discreteness(clustered_latents)
    D_diverse = analyzer.compute_latent_discreteness(diverse_latents)
    
    print(f"Random:    D_latent = {D_random:.4f}")
    print(f"Clustered: D_latent = {D_clustered:.4f}")
    print(f"Diverse:   D_latent = {D_diverse:.4f}")
    
    # 对比可视化
    print("\n对比不同方法的多样性:")
    print("=" * 50)
    methods_data = {
        'Random': random_latents,
        'Clustered': clustered_latents,
        'Diverse': diverse_latents
    }
    analyzer.compare_diversity(methods_data)
    
    # 2D可视化
    print("\n生成2D可视化...")
    analyzer.visualize_latent_space_2d(diverse_latents, method='PCA', 
                                      save_path='latent_pca.png')
    
    print("\n✅ 多样性分析工具测试完成！")
