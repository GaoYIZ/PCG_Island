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
    """Compact Beta-VAE for 64x64 heightmaps."""

    def __init__(self, map_size: int = 64, latent_dim: int = 32, beta: float = 4.0):
        super().__init__()
        if map_size != 64:
            raise ValueError("This reference implementation currently supports map_size=64.")

        self.map_size = map_size
        self.latent_dim = latent_dim
        self.beta = beta

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

        self.decoder_input = nn.Linear(latent_dim, 256)
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (256, 1, 1)),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )
        self.output_adjust = nn.Upsample(size=(64, 64), mode="bilinear", align_corners=False)

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
        reconstruction = self.decoder(hidden)
        return self.output_adjust(reconstruction)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        return reconstruction, mu, logvar

    def loss_function(
        self,
        reconstruction: torch.Tensor,
        target: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> dict:
        recon_loss = F.mse_loss(reconstruction, target, reduction="mean")
        kl_divergence = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        total_loss = recon_loss + self.beta * kl_divergence
        return {
            "total_loss": total_loss,
            "recon_loss": recon_loss,
            "kl_divergence": kl_divergence,
        }

    def encode_heightmap(self, heightmap: np.ndarray, device: str = "cpu", deterministic: bool = True) -> np.ndarray:
        self.eval()
        tensor = torch.as_tensor(heightmap, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            mu, logvar = self.encode(tensor)
            latent = mu if deterministic else self.reparameterize(mu, logvar)
        return latent.squeeze(0).cpu().numpy()


class HeightmapDataset(Dataset):
    """Simple dataset wrapper for heightmaps shaped as (N, H, W)."""

    def __init__(self, heightmaps: np.ndarray):
        self.heightmaps = np.asarray(heightmaps, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.heightmaps)

    def __getitem__(self, index: int) -> torch.Tensor:
        return torch.from_numpy(self.heightmaps[index]).unsqueeze(0)


def train_vae(
    vae: BetaVAE,
    dataloader: DataLoader,
    epochs: int = 50,
    learning_rate: float = 1e-3,
    device: str = "cpu",
) -> List[dict]:
    vae = vae.to(device)
    optimizer = torch.optim.Adam(vae.parameters(), lr=learning_rate)
    history: List[dict] = []

    for epoch in range(epochs):
        vae.train()
        total_loss = 0.0
        recon_loss = 0.0
        kl_loss = 0.0
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
            kl_loss += float(losses["kl_divergence"].item())
            num_batches += 1

        epoch_metrics = {
            "epoch": epoch + 1,
            "total_loss": total_loss / max(num_batches, 1),
            "recon_loss": recon_loss / max(num_batches, 1),
            "kl_loss": kl_loss / max(num_batches, 1),
        }
        history.append(epoch_metrics)

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch [{epoch + 1}/{epochs}] "
                f"total={epoch_metrics['total_loss']:.4f} "
                f"recon={epoch_metrics['recon_loss']:.4f} "
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

    vae = BetaVAE(map_size=64, latent_dim=16, beta=4.0)
    history = train_vae(vae, dataloader, epochs=2)
    latents = encode_heightmaps(vae, heightmaps, batch_size=8)
    print(f"Training steps: {len(history)}")
    print(f"Latent matrix shape: {latents.shape}")
