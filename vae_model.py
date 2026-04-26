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


class ResidualRefineBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.block(x))


class BetaVAE(nn.Module):
    """Beta-VAE with KL warmup and edge-aware reconstruction for square heightmaps."""

    def __init__(
        self,
        map_size: int = 128,
        latent_dim: int = 128,
        beta: float = 0.25,
        beta_start: float = 0.0,
        free_bits: float = 0.01,
        gradient_loss_weight: float = 0.20,
        land_threshold: float = 0.30,
        mask_loss_weight: float = 0.15,
        coast_loss_weight: float = 0.20,
        land_dice_loss_weight: float = 0.10,
        coast_dice_loss_weight: float = 0.30,
        structure_dim: int = 0,
        structure_loss_weight: float = 0.0,
        land_recon_focus_weight: float = 1.5,
        coast_recon_focus_weight: float = 2.0,
    ):
        super().__init__()
        if map_size % 16 != 0:
            raise ValueError("This reference implementation requires map_size to be divisible by 16.")

        self.map_size = map_size
        self.encoder_output_size = map_size // 16
        self.encoder_channels = 256
        self.latent_dim = latent_dim
        self.beta = float(beta)
        self.beta_start = float(beta_start)
        self.current_beta = float(beta_start)
        self.free_bits = float(free_bits)
        self.gradient_loss_weight = float(gradient_loss_weight)
        self.land_threshold = float(land_threshold)
        self.mask_loss_weight = float(mask_loss_weight)
        self.coast_loss_weight = float(coast_loss_weight)
        self.land_dice_loss_weight = float(land_dice_loss_weight)
        self.coast_dice_loss_weight = float(coast_dice_loss_weight)
        self.structure_dim = int(structure_dim)
        self.structure_loss_weight = float(structure_loss_weight)
        self.land_recon_focus_weight = float(land_recon_focus_weight)
        self.coast_recon_focus_weight = float(coast_recon_focus_weight)

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
        flattened_dim = self.encoder_channels * self.encoder_output_size * self.encoder_output_size
        self.fc_mu = nn.Linear(flattened_dim, latent_dim)
        self.fc_logvar = nn.Linear(flattened_dim, latent_dim)

        self.decoder_input = nn.Linear(latent_dim, flattened_dim)
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (self.encoder_channels, self.encoder_output_size, self.encoder_output_size)),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.output_head = nn.Sequential(
            ResidualRefineBlock(16),
            ResidualRefineBlock(16),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )
        if self.structure_dim > 0:
            self.structure_predictor = nn.Sequential(
                nn.Linear(latent_dim, latent_dim),
                nn.ReLU(inplace=True),
                nn.Linear(latent_dim, self.structure_dim),
            )
        else:
            self.structure_predictor = None

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
        decoded = self.decoder(hidden)
        return self.output_head(decoded)

    def predict_structure(self, latent: torch.Tensor) -> torch.Tensor | None:
        if self.structure_predictor is None:
            return None
        return self.structure_predictor(latent)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        structure_prediction = self.predict_structure(mu)
        return reconstruction, mu, logvar, structure_prediction

    @staticmethod
    def _gradient_map(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        grad_x = x[:, :, :, 1:] - x[:, :, :, :-1]
        grad_y = x[:, :, 1:, :] - x[:, :, :-1, :]
        return grad_x, grad_y

    def _soft_land_mask(self, x: torch.Tensor, temperature: float = 25.0) -> torch.Tensor:
        return torch.sigmoid((x - self.land_threshold) * temperature)

    def _coast_response(self, x: torch.Tensor) -> torch.Tensor:
        grad_x, grad_y = self._gradient_map(x)
        grad_x = F.pad(grad_x.abs(), (0, 1, 0, 0))
        grad_y = F.pad(grad_y.abs(), (0, 0, 0, 1))
        return torch.clamp(grad_x + grad_y, 0.0, 1.0)

    @staticmethod
    def _weighted_mean(loss_map: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return torch.sum(loss_map * weights) / torch.sum(weights)

    @staticmethod
    def _dice_loss(prediction: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        dims = tuple(range(1, prediction.ndim))
        intersection = torch.sum(prediction * target, dim=dims)
        denominator = torch.sum(prediction, dim=dims) + torch.sum(target, dim=dims)
        dice = (2.0 * intersection + eps) / (denominator + eps)
        return 1.0 - dice.mean()

    def loss_function(
        self,
        reconstruction: torch.Tensor,
        target: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        structure_prediction: torch.Tensor | None = None,
        structure_target: torch.Tensor | None = None,
    ) -> dict:
        mse_loss = F.mse_loss(reconstruction, target, reduction="mean")
        l1_loss = F.l1_loss(reconstruction, target, reduction="mean")

        target_land_mask = self._soft_land_mask(target)
        target_coast = self._coast_response(target_land_mask)
        pixel_weights = 1.0 + self.land_recon_focus_weight * target_land_mask + self.coast_recon_focus_weight * target_coast
        weighted_mse_loss = self._weighted_mean((reconstruction - target).pow(2), pixel_weights)
        weighted_l1_loss = self._weighted_mean((reconstruction - target).abs(), pixel_weights)

        recon_grad_x, recon_grad_y = self._gradient_map(reconstruction)
        target_grad_x, target_grad_y = self._gradient_map(target)
        gradient_loss = (
            self._weighted_mean((recon_grad_x - target_grad_x).abs(), pixel_weights[:, :, :, 1:])
            + self._weighted_mean((recon_grad_y - target_grad_y).abs(), pixel_weights[:, :, 1:, :])
        )

        recon_land_mask = self._soft_land_mask(reconstruction)
        mask_loss = F.l1_loss(recon_land_mask, target_land_mask, reduction="mean")
        land_dice_loss = self._dice_loss(recon_land_mask, target_land_mask)

        recon_coast = self._coast_response(recon_land_mask)
        coast_loss = F.l1_loss(recon_coast, target_coast, reduction="mean")
        coast_dice_loss = self._dice_loss(recon_coast, target_coast)

        recon_loss = (
            0.20 * mse_loss
            + 0.10 * l1_loss
            + 0.40 * weighted_mse_loss
            + 0.30 * weighted_l1_loss
            + self.gradient_loss_weight * gradient_loss
            + self.mask_loss_weight * mask_loss
            + self.coast_loss_weight * coast_loss
            + self.land_dice_loss_weight * land_dice_loss
            + self.coast_dice_loss_weight * coast_dice_loss
        )

        kl_per_dim = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp(), dim=0)
        kl_divergence = torch.mean(torch.clamp(kl_per_dim, min=self.free_bits))
        structure_loss = torch.zeros((), device=target.device)
        if (
            self.structure_loss_weight > 0.0
            and structure_prediction is not None
            and structure_target is not None
        ):
            structure_loss = F.smooth_l1_loss(structure_prediction, structure_target, reduction="mean")

        total_loss = recon_loss + self.current_beta * kl_divergence + self.structure_loss_weight * structure_loss
        return {
            "total_loss": total_loss,
            "recon_loss": recon_loss,
            "mse_loss": mse_loss,
            "l1_loss": l1_loss,
            "weighted_mse_loss": weighted_mse_loss,
            "weighted_l1_loss": weighted_l1_loss,
            "gradient_loss": gradient_loss,
            "mask_loss": mask_loss,
            "land_dice_loss": land_dice_loss,
            "coast_loss": coast_loss,
            "coast_dice_loss": coast_dice_loss,
            "structure_loss": structure_loss,
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

    def __init__(
        self,
        heightmaps: np.ndarray,
        structure_targets: np.ndarray | None = None,
        augment: bool = False,
    ):
        self.heightmaps = np.asarray(heightmaps, dtype=np.float32)
        self.structure_targets = (
            None if structure_targets is None else np.asarray(structure_targets, dtype=np.float32)
        )
        self.augment = augment

    def __len__(self) -> int:
        return len(self.heightmaps)

    def __getitem__(self, index: int):
        heightmap = self.heightmaps[index]
        if self.augment:
            rotation_k = int(np.random.randint(0, 4))
            heightmap = np.rot90(heightmap, k=rotation_k).copy()
            if np.random.rand() < 0.5:
                heightmap = np.flip(heightmap, axis=0).copy()
            if np.random.rand() < 0.5:
                heightmap = np.flip(heightmap, axis=1).copy()
        heightmap_tensor = torch.from_numpy(heightmap).unsqueeze(0)
        if self.structure_targets is None:
            return heightmap_tensor
        return heightmap_tensor, torch.from_numpy(self.structure_targets[index])


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
        mask_loss = 0.0
        land_dice_loss = 0.0
        coast_loss = 0.0
        coast_dice_loss = 0.0
        weighted_mse_loss = 0.0
        weighted_l1_loss = 0.0
        structure_loss = 0.0
        kl_loss = 0.0
        kl_raw = 0.0
        num_batches = 0

        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                batch, structure_target = batch
                structure_target = structure_target.to(device)
            else:
                structure_target = None
            batch = batch.to(device)
            optimizer.zero_grad()

            reconstruction, mu, logvar, structure_prediction = vae(batch)
            losses = vae.loss_function(
                reconstruction,
                batch,
                mu,
                logvar,
                structure_prediction=structure_prediction,
                structure_target=structure_target,
            )
            losses["total_loss"].backward()
            optimizer.step()

            total_loss += float(losses["total_loss"].item())
            recon_loss += float(losses["recon_loss"].item())
            mse_loss += float(losses["mse_loss"].item())
            l1_loss += float(losses["l1_loss"].item())
            weighted_mse_loss += float(losses["weighted_mse_loss"].item())
            weighted_l1_loss += float(losses["weighted_l1_loss"].item())
            gradient_loss += float(losses["gradient_loss"].item())
            mask_loss += float(losses["mask_loss"].item())
            land_dice_loss += float(losses["land_dice_loss"].item())
            coast_loss += float(losses["coast_loss"].item())
            coast_dice_loss += float(losses["coast_dice_loss"].item())
            structure_loss += float(losses["structure_loss"].item())
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
            "weighted_mse_loss": weighted_mse_loss / max(num_batches, 1),
            "weighted_l1_loss": weighted_l1_loss / max(num_batches, 1),
            "gradient_loss": gradient_loss / max(num_batches, 1),
            "mask_loss": mask_loss / max(num_batches, 1),
            "land_dice_loss": land_dice_loss / max(num_batches, 1),
            "coast_loss": coast_loss / max(num_batches, 1),
            "coast_dice_loss": coast_dice_loss / max(num_batches, 1),
            "structure_loss": structure_loss / max(num_batches, 1),
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
                f"wmse={epoch_metrics['weighted_mse_loss']:.4f} "
                f"grad={epoch_metrics['gradient_loss']:.4f} "
                f"coast={epoch_metrics['coast_loss']:.4f} "
                f"coast_dice={epoch_metrics['coast_dice_loss']:.4f} "
                f"struct={epoch_metrics['structure_loss']:.4f} "
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

    generator = PCGIslandGenerator(map_size=128)
    heightmaps = np.stack(
        [generator.generate_heightmap(generator.sample_random_params(np.random.default_rng(i))) for i in range(32)],
        axis=0,
    )
    dataset = HeightmapDataset(heightmaps)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    vae = BetaVAE(map_size=128, latent_dim=128, beta=0.25, beta_start=0.0, free_bits=0.01)
    history = train_vae(vae, dataloader, epochs=2, warmup_epochs=1)
    latents = encode_heightmaps(vae, heightmaps, batch_size=8)
    print(f"Training steps: {len(history)}")
    print(f"Latent matrix shape: {latents.shape}")
