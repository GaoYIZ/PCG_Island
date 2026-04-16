"""
β-VAE表征学习模块 - 将高度图映射为低维隐变量
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple


class BetaVAE(nn.Module):
    """β-VAE模型"""
    
    def __init__(self, map_size: int = 64, latent_dim: int = 32, beta: float = 4.0):
        super(BetaVAE, self).__init__()
        
        self.map_size = map_size
        self.latent_dim = latent_dim
        self.beta = beta
        
        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),  # 64x64 -> 32x32
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 32x32 -> 16x16
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 16x16 -> 8x8
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU()
        )
        
        # 隐空间参数
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)
        
        # 解码器
        self.decoder_input = nn.Linear(latent_dim, 256)
        
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (256, 1, 1)),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 1x1 -> 2x2
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 2x2 -> 4x4
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 4x4 -> 8x8
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),  # 8x8 -> 16x16
            nn.Sigmoid()
        )
        
        # 调整输出大小到map_size
        if map_size == 64:
            self.output_adjust = nn.Upsample(size=(64, 64), mode='bilinear', align_corners=False)
        elif map_size == 128:
            self.output_adjust = nn.Sequential(
                nn.ConvTranspose2d(1, 1, kernel_size=4, stride=2, padding=1),  # 16x16 -> 32x32
                nn.Sigmoid(),
                nn.ConvTranspose2d(1, 1, kernel_size=4, stride=2, padding=1),  # 32x32 -> 64x64
                nn.Sigmoid()
            )
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """编码：输入高度图，输出隐变量的均值和方差"""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """重参数化技巧"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """解码：从隐变量重建高度图"""
        h = self.decoder_input(z)
        reconstructed = self.decoder(h)
        
        # 调整输出大小
        if hasattr(self, 'output_adjust'):
            reconstructed = self.output_adjust(reconstructed)
        
        return reconstructed
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """前向传播"""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar
    
    def loss_function(self, x_recon: torch.Tensor, x: torch.Tensor, 
                     mu: torch.Tensor, logvar: torch.Tensor) -> dict:
        """
        β-VAE损失函数
        Loss = Reconstruction Loss + β * KL Divergence
        """
        # 重建损失（MSE）
        recon_loss = F.mse_loss(x_recon, x, reduction='sum')
        
        # KL散度
        kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        # 总损失
        total_loss = recon_loss + self.beta * kl_divergence
        
        return {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'kl_divergence': kl_divergence
        }


class HeightmapDataset(Dataset):
    """高度图数据集"""
    
    def __init__(self, heightmaps: np.ndarray):
        """
        参数:
            heightmaps: 形状为 (N, H, W) 的高度图数组
        """
        self.heightmaps = heightmaps
    
    def __len__(self):
        return len(self.heightmaps)
    
    def __getitem__(self, idx):
        # 转换为tensor并添加通道维度
        heightmap = self.heightmaps[idx]
        heightmap = torch.FloatTensor(heightmap).unsqueeze(0)  # (1, H, W)
        return heightmap


def train_vae(vae: BetaVAE, dataloader: DataLoader, epochs: int = 50, 
              learning_rate: float = 1e-3, device: str = 'cpu') -> list:
    """
    训练β-VAE
    
    返回:
        训练损失历史
    """
    vae = vae.to(device)
    optimizer = torch.optim.Adam(vae.parameters(), lr=learning_rate)
    
    loss_history = []
    
    for epoch in range(epochs):
        epoch_loss = 0
        epoch_recon_loss = 0
        epoch_kl_loss = 0
        n_batches = 0
        
        for batch in dataloader:
            batch = batch.to(device)
            
            optimizer.zero_grad()
            
            # 前向传播
            x_recon, mu, logvar = vae(batch)
            
            # 计算损失
            losses = vae.loss_function(x_recon, batch, mu, logvar)
            
            # 反向传播
            losses['total_loss'].backward()
            optimizer.step()
            
            epoch_loss += losses['total_loss'].item()
            epoch_recon_loss += losses['recon_loss'].item()
            epoch_kl_loss += losses['kl_divergence'].item()
            n_batches += 1
        
        # 记录平均损失
        avg_loss = epoch_loss / n_batches
        avg_recon = epoch_recon_loss / n_batches
        avg_kl = epoch_kl_loss / n_batches
        
        loss_history.append({
            'epoch': epoch + 1,
            'total_loss': avg_loss,
            'recon_loss': avg_recon,
            'kl_loss': avg_kl
        })
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] "
                  f"Total Loss: {avg_loss:.4f} "
                  f"Recon: {avg_recon:.4f} "
                  f"KL: {avg_kl:.4f}")
    
    return loss_history


# 测试代码
if __name__ == "__main__":
    from pcg_generator import PCGIslandGenerator
    
    # 生成测试数据
    print("Generating test data...")
    generator = PCGIslandGenerator(map_size=64)
    heightmaps = []
    
    for i in range(100):
        params = {
            'f': np.random.uniform(5, 20),
            'A': np.random.uniform(0.5, 1.5),
            'N_octaves': np.random.randint(3, 6),
            'persistence': np.random.uniform(0.3, 0.7),
            'lacunarity': np.random.uniform(1.5, 2.5),
            'seed': i,
            'warp_strength': np.random.uniform(0.2, 0.8),
            'warp_frequency': np.random.uniform(1, 5),
            'falloff_radius': np.random.uniform(20, 40),
            'falloff_exponent': np.random.uniform(1.5, 3)
        }
        heightmap = generator.generate_heightmap(params)
        heightmaps.append(heightmap)
    
    heightmaps = np.array(heightmaps)
    print(f"Generated {len(heightmaps)} heightmaps with shape {heightmaps.shape}")
    
    # 创建数据集和数据加载器
    dataset = HeightmapDataset(heightmaps)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    # 创建VAE模型
    vae = BetaVAE(map_size=64, latent_dim=32, beta=4.0)
    
    # 训练（少量epoch用于测试）
    print("\nTraining VAE...")
    loss_history = train_vae(vae, dataloader, epochs=20, learning_rate=1e-3)
    
    # 测试编码
    test_map = torch.FloatTensor(heightmaps[0]).unsqueeze(0).unsqueeze(0)  # (1, 1, 64, 64)
    mu, logvar = vae.encode(test_map)
    print(f"\nLatent vector shape: {mu.shape}")
    print(f"Latent vector mean: {mu.mean().item():.4f}")
    print(f"Latent vector std: {mu.std().item():.4f}")
