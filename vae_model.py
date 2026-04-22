"""
Beta-VAE model and latent extraction helpers for island heightmaps.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


class BetaVAE(nn.Module):
    """Beta-VAE with KL warmup and edge-aware reconstruction for 64x64 heightmaps."""

    def __init__(
        self,
        map_size: int = 64,
        latent_dim: int = 32,
        beta: float = 0.25,
        beta_start: float = 0.0,
        free_bits: float = 0.01,
        gradient_loss_weight: float = 0.20,
    ):
        super().__init__()
        if map_size != 64:
            raise ValueError("This reference implementation currently supports map_size=64.")

        self.map_size = map_size
        self.latent_dim = latent_dim
        self.beta = float(beta)
        self.beta_start = float(beta_start)
        self.current_beta = float(beta_start)
        self.free_bits = float(free_bits)
        self.gradient_loss_weight = float(gradient_loss_weight)

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )
        self.fc_mu = nn.Linear(256 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(256 * 4 * 4, latent_dim)

        self.decoder_input = nn.Linear(latent_dim, 256 * 4 * 4)
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (256, 4, 4)),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def set_beta(self, beta: float) -> None:
        self.current_beta = float(beta)

    def update_beta(self, epoch: int, total_epochs: int, warmup_epochs: int) -> float:
        if warmup_epochs <= 0:
            self.current_beta = self.beta
            return self.current_beta
        progress = min(max((epoch + 1) / warmup_epochs, 0.0), 1.0)
        self.current_beta = self.beta_start + (self.beta - self.beta_start) * progress
        return self.current_beta

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encoder(x)
        return self.fc_mu(hidden), self.fc_logvar(hidden)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        hidden = self.decoder_input(z)
        return self.decoder(hidden)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        return reconstruction, mu, logvar

    @staticmethod
    def _gradient_map(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        grad_x = x[:, :, :, 1:] - x[:, :, :, :-1]
        grad_y = x[:, :, 1:, :] - x[:, :, :-1, :]
        return grad_x, grad_y

    def loss_function(
        self,
        reconstruction: torch.Tensor,
        target: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> dict:
        mse_loss = F.mse_loss(reconstruction, target, reduction="mean")
        l1_loss = F.l1_loss(reconstruction, target, reduction="mean")

        recon_grad_x, recon_grad_y = self._gradient_map(reconstruction)
        target_grad_x, target_grad_y = self._gradient_map(target)
        gradient_loss = (
            F.l1_loss(recon_grad_x, target_grad_x, reduction="mean")
            + F.l1_loss(recon_grad_y, target_grad_y, reduction="mean")
        )

        recon_loss = 0.70 * mse_loss + 0.30 * l1_loss + self.gradient_loss_weight * gradient_loss

        kl_per_dim = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp(), dim=0)
        kl_divergence = torch.mean(torch.clamp(kl_per_dim, min=self.free_bits))
        total_loss = recon_loss + self.current_beta * kl_divergence
        return {
            "total_loss": total_loss,
            "recon_loss": recon_loss,
            "mse_loss": mse_loss,
            "l1_loss": l1_loss,
            "gradient_loss": gradient_loss,
            "kl_divergence": kl_divergence,
            "kl_raw": torch.mean(kl_per_dim),
            "beta": torch.tensor(self.current_beta, device=target.device),
        }

    def encode_heightmap(self, heightmap: np.ndarray, device: str = "cpu", deterministic: bool = True) -> np.ndarray:
        self.eval()
        model_device = next(self.parameters()).device
        target_device = model_device if device == "cpu" else torch.device(device)
        tensor = torch.as_tensor(heightmap, dtype=torch.float32, device=target_device).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            mu, logvar = self.encode(tensor)
            latent = mu if deterministic else self.reparameterize(mu, logvar)
        return latent.squeeze(0).cpu().numpy()


class HeightmapDataset(Dataset):
    """Simple dataset wrapper for heightmaps shaped as (N, H, W)."""

    def __init__(self, heightmaps: np.ndarray, augment: bool = False):
        self.heightmaps = np.asarray(heightmaps, dtype=np.float32)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.heightmaps)

    def __getitem__(self, index: int) -> torch.Tensor:
        heightmap = self.heightmaps[index]
        if self.augment:
            rotation_k = int(np.random.randint(0, 4))
            heightmap = np.rot90(heightmap, k=rotation_k).copy()
            if np.random.rand() < 0.5:
                heightmap = np.flip(heightmap, axis=0).copy()
            if np.random.rand() < 0.5:
                heightmap = np.flip(heightmap, axis=1).copy()
        return torch.from_numpy(heightmap).unsqueeze(0)


def train_vae(
    vae: BetaVAE,
    dataloader: DataLoader,
    epochs: int = 50,
    learning_rate: float = 1e-3,
    device: str = "cpu",
    warmup_epochs: int = 10,
) -> List[dict]:
    vae = vae.to(device)
    optimizer = torch.optim.Adam(vae.parameters(), lr=learning_rate)
    history: List[dict] = []

    for epoch in range(epochs):
        vae.train()
        beta = vae.update_beta(epoch=epoch, total_epochs=epochs, warmup_epochs=warmup_epochs)
        total_loss = 0.0
        recon_loss = 0.0
        mse_loss = 0.0
        l1_loss = 0.0
        gradient_loss = 0.0
        kl_loss = 0.0
        kl_raw = 0.0
        num_batches = 0

        for batch in dataloader:
            batch = batch.to(device)
            optimizer.zero_grad()

            reconstruction, mu, logvar = vae(batch)
            losses = vae.loss_function(reconstruction, batch, mu, logvar)
            losses["total_loss"].backward()
            optimizer.step()

            total_loss += float(losses["total_loss"].item())
            recon_loss += float(losses["recon_loss"].item())
            mse_loss += float(losses["mse_loss"].item())
            l1_loss += float(losses["l1_loss"].item())
            gradient_loss += float(losses["gradient_loss"].item())
            kl_loss += float(losses["kl_divergence"].item())
            kl_raw += float(losses["kl_raw"].item())
            num_batches += 1

        epoch_metrics = {
            "epoch": epoch + 1,
            "beta": float(beta),
            "total_loss": total_loss / max(num_batches, 1),
            "recon_loss": recon_loss / max(num_batches, 1),
            "mse_loss": mse_loss / max(num_batches, 1),
            "l1_loss": l1_loss / max(num_batches, 1),
            "gradient_loss": gradient_loss / max(num_batches, 1),
            "kl_loss": kl_loss / max(num_batches, 1),
            "kl_raw": kl_raw / max(num_batches, 1),
        }
        history.append(epoch_metrics)

        if (epoch + 1) % 5 == 0 or epoch == 0 or epoch + 1 == epochs:
            print(
                f"Epoch [{epoch + 1}/{epochs}] "
                f"beta={epoch_metrics['beta']:.3f} "
                f"total={epoch_metrics['total_loss']:.4f} "
                f"recon={epoch_metrics['recon_loss']:.4f} "
                f"grad={epoch_metrics['gradient_loss']:.4f} "
                f"kl={epoch_metrics['kl_loss']:.4f}"
            )

    return history


def encode_heightmaps(
    vae: BetaVAE,
    heightmaps: np.ndarray,
    batch_size: int = 64,
    device: str = "cpu",
    deterministic: bool = True,
) -> np.ndarray:
    vae = vae.to(device)
    vae.eval()
    dataset = HeightmapDataset(heightmaps)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    latent_batches = []
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            mu, logvar = vae.encode(batch)
            latent = mu if deterministic else vae.reparameterize(mu, logvar)
            latent_batches.append(latent.cpu().numpy())

    return np.concatenate(latent_batches, axis=0) if latent_batches else np.empty((0, vae.latent_dim))


if __name__ == "__main__":
    from pcg_generator import PCGIslandGenerator

    generator = PCGIslandGenerator(map_size=64)
    heightmaps = np.stack(
        [generator.generate_heightmap(generator.sample_random_params(np.random.default_rng(i))) for i in range(32)],
        axis=0,
    )
    dataset = HeightmapDataset(heightmaps)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    vae = BetaVAE(map_size=64, latent_dim=16, beta=0.25, beta_start=0.0, free_bits=0.01)
    history = train_vae(vae, dataloader, epochs=2, warmup_epochs=1)
    latents = encode_heightmaps(vae, heightmaps, batch_size=8)
    print(f"Training steps: {len(history)}")
    print(f"Latent matrix shape: {latents.shape}")
