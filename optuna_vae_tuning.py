"""
Optuna-based hyperparameter tuning for the IslandTest VAE.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import optuna
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from dataset_pipeline import IslandDatasetBuilder
from vae_model import BetaVAE, HeightmapDataset, encode_heightmaps, train_vae

COAST_THRESHOLD = 0.30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optuna 调优 IslandTest 的 VAE")
    parser.add_argument("--output-dir", type=str, default="optuna_vae_outputs", help="调参输出目录")
    parser.add_argument("--map-size", type=int, default=128, help="地图尺寸")
    parser.add_argument("--latent-dim", type=int, default=128, help="固定的 VAE latent 维度")
    parser.add_argument("--dataset-samples", type=int, default=120, help="每轮数据采样量")
    parser.add_argument("--min-clean-samples", type=int, default=64, help="最少保留样本数")
    parser.add_argument("--max-dataset-samples", type=int, default=480, help="最大原始样本数")
    parser.add_argument("--sampling-profile", type=str, default="island", choices=["uniform", "island"])
    parser.add_argument("--batch-size", type=int, default=32, help="训练批大小")
    parser.add_argument("--epochs", type=int, default=12, help="每个 trial 的训练轮数")
    parser.add_argument("--trials", type=int, default=12, help="Optuna trial 数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_arrays(args: argparse.Namespace) -> dict[str, np.ndarray]:
    builder = IslandDatasetBuilder(map_size=args.map_size, sampling_profile=args.sampling_profile)

    raw_samples = []
    clean_samples = []
    round_index = 0
    while len(clean_samples) < args.min_clean_samples and len(raw_samples) < args.max_dataset_samples:
        remaining_budget = args.max_dataset_samples - len(raw_samples)
        chunk_size = min(args.dataset_samples, remaining_budget)
        chunk = builder.generate_samples(
            n_samples=chunk_size,
            seed=args.seed + round_index * 997,
            sampling_profile=args.sampling_profile,
        )
        raw_samples.extend(chunk)
        clean_samples = builder.clean_samples(raw_samples)
        round_index += 1

    return builder.build_training_arrays(clean_samples)


def reconstruct(
    vae: BetaVAE,
    heightmaps: np.ndarray,
    batch_size: int,
    device: torch.device,
    deterministic: bool = True,
) -> np.ndarray:
    vae.eval()
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(heightmaps), batch_size):
            batch = torch.as_tensor(heightmaps[start : start + batch_size], dtype=torch.float32, device=device)
            inputs = batch.unsqueeze(1)
            if deterministic:
                mu, _ = vae.encode(inputs)
                recon = vae.decode(mu)
            else:
                recon, _, _, _ = vae(inputs)
            outputs.append(recon.squeeze(1).cpu().numpy())
    return np.concatenate(outputs, axis=0)


def build_focus_masks(
    heightmaps: np.ndarray,
    water_threshold: float = COAST_THRESHOLD,
) -> tuple[np.ndarray, np.ndarray]:
    land_mask = heightmaps > water_threshold
    padded = np.pad(land_mask, ((0, 0), (1, 1), (1, 1)), mode="edge")
    up = padded[:, :-2, 1:-1]
    down = padded[:, 2:, 1:-1]
    left = padded[:, 1:-1, :-2]
    right = padded[:, 1:-1, 2:]
    coast_band = land_mask != up
    coast_band |= land_mask != down
    coast_band |= land_mask != left
    coast_band |= land_mask != right
    return land_mask, coast_band


def masked_mae(
    original: np.ndarray,
    reconstructed: np.ndarray,
    mask: np.ndarray,
) -> float:
    if mask.size == 0 or not np.any(mask):
        return 0.0
    return float(np.mean(np.abs(original[mask] - reconstructed[mask])))


def predict_structure(vae: BetaVAE, heightmaps: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    if vae.structure_predictor is None:
        return np.empty((len(heightmaps), 0), dtype=np.float32)
    vae.eval()
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(heightmaps), batch_size):
            batch = torch.as_tensor(heightmaps[start : start + batch_size], dtype=torch.float32, device=device)
            mu, _ = vae.encode(batch.unsqueeze(1))
            pred = vae.predict_structure(mu)
            if pred is not None:
                outputs.append(pred.cpu().numpy())
    return np.concatenate(outputs, axis=0)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    arrays = build_arrays(args)
    train_heightmaps, val_heightmaps, train_metrics, val_metrics = train_test_split(
        arrays["heightmaps"],
        arrays["core_metric_matrix"],
        test_size=0.2,
        random_state=args.seed,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def objective(trial: optuna.Trial) -> float:
        beta = trial.suggest_float("beta", 0.05, 0.5, log=True)
        beta_start = trial.suggest_float("beta_start", 0.0, 0.08)
        free_bits = trial.suggest_float("free_bits", 0.001, 0.05, log=True)
        gradient_loss_weight = trial.suggest_float("gradient_loss_weight", 0.05, 0.40)
        mask_loss_weight = trial.suggest_float("mask_loss_weight", 0.05, 0.35)
        coast_loss_weight = trial.suggest_float("coast_loss_weight", 0.05, 0.35)
        structure_loss_weight = trial.suggest_float("structure_loss_weight", 0.15, 0.80)
        land_recon_focus_weight = trial.suggest_float("land_recon_focus_weight", 0.8, 2.5)
        coast_recon_focus_weight = trial.suggest_float("coast_recon_focus_weight", 1.0, 3.5)
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 2e-3, log=True)
        warmup_epochs = trial.suggest_int("warmup_epochs", 4, max(args.epochs, 4))

        train_dataset = HeightmapDataset(train_heightmaps, structure_targets=train_metrics, augment=True)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        vae = BetaVAE(
            map_size=args.map_size,
            latent_dim=args.latent_dim,
            beta=beta,
            beta_start=beta_start,
            free_bits=free_bits,
            gradient_loss_weight=gradient_loss_weight,
            mask_loss_weight=mask_loss_weight,
            coast_loss_weight=coast_loss_weight,
            structure_dim=train_metrics.shape[1],
            structure_loss_weight=structure_loss_weight,
            land_recon_focus_weight=land_recon_focus_weight,
            coast_recon_focus_weight=coast_recon_focus_weight,
        ).to(device)

        history = train_vae(
            vae,
            train_loader,
            epochs=args.epochs,
            learning_rate=learning_rate,
            device=str(device),
            warmup_epochs=warmup_epochs,
        )
        reconstructions = reconstruct(vae, val_heightmaps, args.batch_size, device)
        predictions = predict_structure(vae, val_heightmaps, args.batch_size, device)
        latents = encode_heightmaps(vae, val_heightmaps, batch_size=args.batch_size, device=str(device))

        pixel_mse = float(np.mean((val_heightmaps - reconstructions) ** 2))
        pixel_mae = float(np.mean(np.abs(val_heightmaps - reconstructions)))
        land_mask, coast_band = build_focus_masks(val_heightmaps)
        land_mae = masked_mae(val_heightmaps, reconstructions, land_mask)
        coast_band_mae = masked_mae(val_heightmaps, reconstructions, coast_band)
        structure_mae = float(np.mean(np.abs(val_metrics - predictions))) if predictions.size else 0.0
        final_kl = float(history[-1]["kl_loss"]) if history else 0.0
        latent_global_std = float(latents.std()) if len(latents) > 0 else 0.0
        latent_std_per_dim = latents.std(axis=0) if len(latents) > 0 else np.zeros((args.latent_dim,), dtype=np.float32)
        active_threshold = max(1e-3, latent_global_std * 0.02)
        active_ratio = float(np.mean(latent_std_per_dim > active_threshold)) if args.latent_dim > 0 else 0.0
        collapse_penalty = max(0.0, 0.35 - active_ratio)
        objective_value = (
            0.30 * pixel_mae
            + 0.20 * land_mae
            + 0.35 * coast_band_mae
            + 0.45 * structure_mae
            + 0.08 * final_kl
            + 0.30 * collapse_penalty
        )

        trial.set_user_attr("pixel_mse", pixel_mse)
        trial.set_user_attr("pixel_mae", pixel_mae)
        trial.set_user_attr("land_mae", land_mae)
        trial.set_user_attr("coast_band_mae", coast_band_mae)
        trial.set_user_attr("structure_mae", structure_mae)
        trial.set_user_attr("final_kl", final_kl)
        trial.set_user_attr("active_ratio", active_ratio)
        trial.set_user_attr("latent_global_std", latent_global_std)
        return objective_value

    study = optuna.create_study(direction="minimize", study_name="island_vae_tuning")
    study.optimize(objective, n_trials=args.trials)

    best = {
        "best_value": float(study.best_value),
        "best_params": study.best_trial.params,
        "best_attrs": study.best_trial.user_attrs,
    }
    best["best_params"]["latent_dim"] = args.latent_dim
    (output_dir / "best_trial.json").write_text(json.dumps(best, ensure_ascii=False, indent=2), encoding="utf-8")
    print("最优结果已保存到:", output_dir / "best_trial.json")
    print(json.dumps(best, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
