"""
Formal experiment runner for IslandTest.

Pipeline:
1. 数据集构建 / 清洗 / 评估
2. VAE 训练与隐向量提取
3. 特征归一化拟合
4. VAE 表征有效性评估
5. PPO 正式训练
6. 随机/零动作基线对比
7. 结果汇总与中文输出
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import font_manager
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score
from torch.utils.data import DataLoader

from dataset_pipeline import IslandDatasetBuilder
from feature_processing import IslandFeatureNormalizer, ParameterSpaceNormalizer
from pcg_generator import PCGIslandGenerator
from ppo_baseline import PPOAgent
from reporting import (
    metric_label,
    print_dataset_summary,
    print_metric_dict,
    print_section,
    save_json,
    summarize_rewards,
)
from rl_environment import IslandGenerationEnv
from sac_agent import ReplayBuffer, SACAgent
from vae_model import BetaVAE, HeightmapDataset, encode_heightmaps, train_vae


COAST_THRESHOLD = 0.30
STRUCTURE_SUPERVISION_WEIGHTS: Dict[str, float] = {
    "connectivity": 1.8,
    "navigable_ratio": 1.3,
    "coast_complexity": 1.5,
    "terrain_variance": 1.0,
    "path_reachability": 2.2,
    "land_ratio": 0.8,
    "component_count": 1.8,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="IslandTest formal experiment runner")
    parser.add_argument("--output-dir", type=str, default="formal_outputs", help="Result output directory")
    parser.add_argument("--map-size", type=int, default=128, help="Heightmap size")
    parser.add_argument("--dataset-samples", type=int, default=120, help="Raw samples generated per sampling round")
    parser.add_argument("--min-clean-samples", type=int, default=48, help="Minimum clean samples kept after filtering")
    parser.add_argument("--max-dataset-samples", type=int, default=480, help="Maximum raw samples allowed during dataset building")
    parser.add_argument(
        "--sampling-profile",
        type=str,
        default="island",
        choices=["uniform", "island", "island_voronoi"],
        help="Parameter sampling strategy",
    )
    parser.add_argument(
        "--drop-connectivity-supervision",
        action="store_true",
        help="Remove connectivity from VAE structure supervision while leaving dataset generation unchanged.",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for VAE/PPO")
    parser.add_argument("--latent-dim", type=int, default=128, help="VAE latent dimension")
    parser.add_argument("--vae-epochs", type=int, default=30, help="VAE training epochs")
    parser.add_argument("--vae-beta", type=float, default=0.25, help="Beta-VAE beta")
    parser.add_argument("--vae-beta-start", type=float, default=0.0, help="KL warmup starting beta")
    parser.add_argument("--vae-warmup-epochs", type=int, default=12, help="KL warmup epochs")
    parser.add_argument("--vae-free-bits", type=float, default=0.01, help="Minimum KL per latent dimension")
    parser.add_argument("--vae-gradient-loss-weight", type=float, default=0.20, help="Gradient reconstruction loss weight")
    parser.add_argument("--vae-mask-loss-weight", type=float, default=0.15, help="Land-mask reconstruction loss weight")
    parser.add_argument("--vae-coast-loss-weight", type=float, default=0.28, help="Coast reconstruction loss weight")
    parser.add_argument("--vae-land-dice-loss-weight", type=float, default=0.12, help="Land-region Dice loss weight")
    parser.add_argument("--vae-coast-dice-loss-weight", type=float, default=0.32, help="Coast-band Dice loss weight")
    parser.add_argument("--vae-structure-loss-weight", type=float, default=0.12, help="Structure-supervision loss weight")
    parser.add_argument("--vae-metric-alignment-loss-weight", type=float, default=0.35, help="Latent geometry alignment loss weight for structure metrics")
    parser.add_argument(
        "--vae-connectivity-supervision-weight",
        type=float,
        default=STRUCTURE_SUPERVISION_WEIGHTS["connectivity"],
        help="Per-metric supervision weight for connectivity",
    )
    parser.add_argument(
        "--vae-path-reachability-supervision-weight",
        type=float,
        default=STRUCTURE_SUPERVISION_WEIGHTS["path_reachability"],
        help="Per-metric supervision weight for path reachability",
    )
    parser.add_argument("--vae-land-recon-focus-weight", type=float, default=2.0, help="Extra reconstruction focus on land pixels")
    parser.add_argument("--vae-coast-recon-focus-weight", type=float, default=3.2, help="Extra reconstruction focus on coast-band pixels")
    parser.add_argument("--vae-lr", type=float, default=8e-4, help="VAE learning rate")
    parser.add_argument("--ppo-episodes", type=int, default=60, help="PPO training episodes")
    parser.add_argument("--ppo-max-steps", type=int, default=30, help="Maximum steps per PPO episode")
    parser.add_argument("--ppo-rollout-steps", type=int, default=1024, help="Transitions collected before each PPO update")
    parser.add_argument("--ppo-hidden-dim", type=int, default=256, help="PPO hidden dimension")
    parser.add_argument("--ppo-hidden-layers", type=int, default=2, help="Number of hidden layers in PPO actor and critic")
    parser.add_argument("--ppo-lr", type=float, default=3e-4, help="PPO learning rate")
    parser.add_argument("--sac-episodes", type=int, default=0, help="Extra SAC training episodes; 0 disables SAC")
    parser.add_argument(
        "--rl-update-batch-size",
        type=int,
        default=0,
        help="Mini-batch size used by PPO/SAC updates; 0 reuses --batch-size. Keep this separate from VAE batch size.",
    )
    parser.add_argument(
        "--rl-reset-profile",
        type=str,
        default=None,
        choices=["uniform", "island", "island_voronoi"],
        help="Sampling profile used for RL environment resets; defaults to --sampling-profile when omitted",
    )
    parser.add_argument("--sac-hidden-dim", type=int, default=256, help="SAC hidden dimension")
    parser.add_argument("--sac-actor-lr", type=float, default=3e-4, help="Actor learning rate for SAC")
    parser.add_argument("--sac-critic-lr", type=float, default=1e-3, help="Critic learning rate for SAC")
    parser.add_argument("--sac-alpha-lr", type=float, default=3e-4, help="Entropy-temperature learning rate for SAC")
    parser.add_argument("--sac-min-alpha", type=float, default=0.0, help="Optional lower bound for SAC entropy temperature")
    parser.add_argument(
        "--sac-target-entropy-scale",
        type=float,
        default=1.0,
        help="Scale applied to SAC target entropy (-action_dim * scale)",
    )
    parser.add_argument("--sac-learning-starts", type=int, default=512, help="Number of environment steps collected before SAC updates start")
    parser.add_argument("--sac-print-interval", type=int, default=5, help="Episode interval for SAC progress printing")
    parser.add_argument("--expert-top-percent", type=float, default=0.10, help="Top fraction of cleaned samples saved as expert references")
    parser.add_argument("--expert-max-samples", type=int, default=256, help="Maximum number of expert samples exported and used for expert guidance")
    parser.add_argument("--novelty-reference-size", type=int, default=256, help="Maximum number of cleaned samples used as the fixed novelty reference bank")
    parser.add_argument("--rl-action-step-scale", type=float, default=0.08, help="Normalized PCG-parameter step size applied to each RL action")
    parser.add_argument("--reward-current-scale", type=float, default=0.0, help="Dense reward weight for current total score; 0 keeps reward focused on improvement")
    parser.add_argument("--reward-delta-scale", type=float, default=5.0, help="Multiplier applied to total-score improvement between consecutive states")
    parser.add_argument("--reward-best-scale", type=float, default=2.0, help="Extra reward for improving beyond the best score seen in the current episode")
    parser.add_argument("--reward-step-penalty", type=float, default=0.01, help="Small per-step penalty to encourage faster convergence")
    parser.add_argument("--reward-success-bonus", type=float, default=1.0, help="Bonus added when the success threshold is reached")
    parser.add_argument("--reward-success-threshold", type=float, default=0.72, help="Episode ends successfully once the total score reaches this threshold enough times")
    parser.add_argument("--reward-success-gain-threshold", type=float, default=0.03, help="Minimum total-score gain required before success bonus/termination")
    parser.add_argument("--reward-failure-threshold", type=float, default=0.10, help="Episode fails early if the total score falls below this threshold")
    parser.add_argument("--reward-success-streak", type=int, default=3, help="Number of consecutive successful steps required for early success termination")
    parser.add_argument("--reward-stagnation-patience", type=int, default=5, help="Early-stop an episode after this many non-improving steps")
    parser.add_argument("--reward-stagnation-delta", type=float, default=1e-3, help="Minimum score improvement counted as progress")
    parser.add_argument("--rl-best-window", type=int, default=50, help="Rolling final-score window used for best checkpoint selection")
    parser.add_argument(
        "--restore-best-policy",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Evaluate the best rolling-score checkpoint instead of the last checkpoint.",
    )
    parser.add_argument(
        "--tensorboard",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write PPO/SAC scalar logs for TensorBoard when tensorboard is installed; CSV is always written.",
    )
    parser.add_argument(
        "--tensorboard-dir",
        type=str,
        default="",
        help="TensorBoard output directory; defaults to <output-dir>/tensorboard.",
    )
    parser.add_argument("--eval-islands", type=int, default=12, help="Number of final evaluation islands")
    parser.add_argument("--skip-rl", action="store_true", help="Stop after VAE evaluation and skip RL/baselines")
    parser.add_argument(
        "--formal-vae-only",
        action="store_true",
        help="Run a full VAE-only evaluation with train/val/test splits and no RL",
    )
    parser.add_argument(
        "--formal-rl",
        action="store_true",
        help="Run the formal VAE split pipeline first, then train/evaluate RL with the frozen VAE.",
    )
    parser.add_argument(
        "--rl-agent",
        choices=["ppo", "sac", "both"],
        default="both",
        help="Train PPO only, SAC only, or both agents.",
    )
    parser.add_argument(
        "--include-latent-state",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include the VAE latent vector in the policy observation.",
    )
    parser.add_argument(
        "--reuse-rl-assets-from",
        type=str,
        default="",
        help="Reuse VAE, normalizer, and expert bank from an existing formal RL output directory.",
    )
    parser.add_argument(
        "--prepare-rl-assets-only",
        action="store_true",
        help="Build formal VAE and RL reference assets, then stop before agent training.",
    )
    parser.add_argument(
        "--preserve-latent-dim",
        action="store_true",
        help="Keep --latent-dim when applying an Optuna trial; used for latent-capacity ablations.",
    )
    parser.add_argument("--vae-train-ratio", type=float, default=0.70, help="Train split ratio for formal VAE-only evaluation")
    parser.add_argument("--vae-val-ratio", type=float, default=0.15, help="Validation split ratio for formal VAE-only evaluation")
    parser.add_argument(
        "--optuna-best-trial",
        type=str,
        default="",
        help="Optional path to best_trial.json produced by optuna_vae_tuning.py; overrides VAE hyperparameters.",
    )
    parser.add_argument(
        "--fast-profile",
        action="store_true",
        help="Use a lighter-weight experimental profile for faster Colab iteration, especially with 64x64 maps.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def apply_formal_vae_preset(args: argparse.Namespace) -> None:
    if not args.formal_vae_only:
        return

    args.skip_rl = True
    if args.dataset_samples == 120:
        args.dataset_samples = 500
    if args.min_clean_samples == 48:
        args.min_clean_samples = 500
    if args.max_dataset_samples == 480:
        args.max_dataset_samples = 2000
    if args.vae_epochs == 30:
        args.vae_epochs = 50
    if args.batch_size == 32:
        args.batch_size = 16


def apply_formal_rl_preset(args: argparse.Namespace) -> None:
    if not args.formal_rl:
        return

    args.skip_rl = False
    if args.dataset_samples == 120:
        args.dataset_samples = 500
    if args.min_clean_samples == 48:
        args.min_clean_samples = 500
    if args.max_dataset_samples == 480:
        args.max_dataset_samples = 2000
    if args.vae_epochs == 30:
        args.vae_epochs = 50
    if args.batch_size == 32:
        args.batch_size = 16
    if args.sac_episodes == 0 and getattr(args, "rl_agent", "both") in {"sac", "both"}:
        args.sac_episodes = 80
    if args.eval_islands == 12:
        args.eval_islands = 24


def apply_optuna_best_trial(args: argparse.Namespace) -> None:
    if not args.optuna_best_trial:
        return

    trial_path = Path(args.optuna_best_trial)
    if not trial_path.exists():
        raise FileNotFoundError(f"Optuna best trial file not found: {trial_path}")

    trial_data = json.loads(trial_path.read_text(encoding="utf-8"))
    if "drop_connectivity_supervision" in trial_data:
        args.drop_connectivity_supervision = bool(trial_data["drop_connectivity_supervision"])
    best_params = trial_data.get("best_params", {})
    mapping = {
        "latent_dim": "latent_dim",
        "beta": "vae_beta",
        "beta_start": "vae_beta_start",
        "warmup_epochs": "vae_warmup_epochs",
        "free_bits": "vae_free_bits",
        "gradient_loss_weight": "vae_gradient_loss_weight",
        "mask_loss_weight": "vae_mask_loss_weight",
        "coast_loss_weight": "vae_coast_loss_weight",
        "land_dice_loss_weight": "vae_land_dice_loss_weight",
        "coast_dice_loss_weight": "vae_coast_dice_loss_weight",
        "structure_loss_weight": "vae_structure_loss_weight",
        "metric_alignment_loss_weight": "vae_metric_alignment_loss_weight",
        "connectivity_supervision_weight": "vae_connectivity_supervision_weight",
        "path_reachability_supervision_weight": "vae_path_reachability_supervision_weight",
        "land_recon_focus_weight": "vae_land_recon_focus_weight",
        "coast_recon_focus_weight": "vae_coast_recon_focus_weight",
        "learning_rate": "vae_lr",
    }
    for source_name, target_name in mapping.items():
        if source_name == "latent_dim" and getattr(args, "preserve_latent_dim", False):
            continue
        if source_name in best_params:
            setattr(args, target_name, best_params[source_name])


def apply_fast_profile(args: argparse.Namespace) -> None:
    if not args.fast_profile:
        return

    if args.map_size <= 64:
        if args.dataset_samples in (120, 500):
            args.dataset_samples = 240
        if args.min_clean_samples in (48, 500):
            args.min_clean_samples = 240
        if args.max_dataset_samples in (480, 2000):
            args.max_dataset_samples = 960
        if args.vae_epochs in (30, 50):
            args.vae_epochs = 24
        if args.batch_size in (16, 32):
            args.batch_size = 32
        if args.latent_dim == 128:
            args.latent_dim = 64
        if args.ppo_episodes == 60:
            args.ppo_episodes = 36
        if args.sac_episodes == 80:
            args.sac_episodes = 40
        elif args.sac_episodes == 0 and args.formal_rl:
            args.sac_episodes = 40
        if args.eval_islands in (12, 24):
            args.eval_islands = 12
    else:
        if args.dataset_samples in (120, 500):
            args.dataset_samples = 320
        if args.min_clean_samples in (48, 500):
            args.min_clean_samples = 320
        if args.max_dataset_samples in (480, 2000):
            args.max_dataset_samples = 1280
        if args.vae_epochs in (30, 50):
            args.vae_epochs = 30
        if args.batch_size in (16, 32):
            args.batch_size = 16
        if args.ppo_episodes == 60:
            args.ppo_episodes = 40
        if args.sac_episodes == 80:
            args.sac_episodes = 50
        elif args.sac_episodes == 0 and args.formal_rl:
            args.sac_episodes = 50
        if args.eval_islands in (12, 24):
            args.eval_islands = 16


def split_indices(
    num_samples: int,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if num_samples < 3:
        raise ValueError("Formal VAE evaluation requires at least 3 clean samples.")
    if not (0.0 < train_ratio < 1.0):
        raise ValueError("vae_train_ratio must be between 0 and 1.")
    if not (0.0 <= val_ratio < 1.0):
        raise ValueError("vae_val_ratio must be between 0 and 1.")
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("vae_train_ratio + vae_val_ratio must be less than 1.")

    indices = np.arange(num_samples, dtype=np.int64)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)

    train_count = max(1, int(round(num_samples * train_ratio)))
    val_count = max(1, int(round(num_samples * val_ratio)))
    if train_count + val_count >= num_samples:
        val_count = max(1, num_samples - train_count - 1)
    test_count = num_samples - train_count - val_count
    if test_count <= 0:
        test_count = 1
        if val_count > 1:
            val_count -= 1
        else:
            train_count -= 1

    train_idx = indices[:train_count]
    val_idx = indices[train_count : train_count + val_count]
    test_idx = indices[train_count + val_count :]
    return train_idx, val_idx, test_idx


def subset_arrays(arrays: Dict[str, np.ndarray], indices: np.ndarray) -> Dict[str, np.ndarray]:
    return {name: values[indices] for name, values in arrays.items()}


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_matplotlib_chinese() -> None:
    candidate_fonts = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Noto Sans CJK JP",
        "WenQuanYi Zen Hei",
        "Arial Unicode MS",
    ]
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    selected = [font for font in candidate_fonts if font in available_fonts]
    if selected:
        plt.rcParams["font.sans-serif"] = selected + ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def get_structure_supervision_weights(
    metric_names: Sequence[str],
    connectivity_weight: float | None = None,
    path_reachability_weight: float | None = None,
) -> List[float]:
    weight_map = dict(STRUCTURE_SUPERVISION_WEIGHTS)
    if connectivity_weight is not None:
        weight_map["connectivity"] = float(connectivity_weight)
    if path_reachability_weight is not None:
        weight_map["path_reachability"] = float(path_reachability_weight)
    return [float(weight_map.get(name, 1.0)) for name in metric_names]


def get_selected_supervision_metric_names(
    args: argparse.Namespace,
    metric_names: Sequence[str],
) -> Tuple[str, ...]:
    selected_metric_names = tuple(metric_names)
    if args.drop_connectivity_supervision:
        selected_metric_names = tuple(name for name in selected_metric_names if name != "connectivity")
    if not selected_metric_names:
        raise ValueError("At least one supervision metric must remain enabled for VAE training.")
    return selected_metric_names


def select_supervision_metrics(
    arrays: Dict[str, np.ndarray],
    available_metric_names: Sequence[str],
    selected_metric_names: Sequence[str],
) -> Dict[str, np.ndarray]:
    selected_indices = [available_metric_names.index(name) for name in selected_metric_names]
    selected_arrays = dict(arrays)
    selected_arrays["supervision_metric_matrix"] = arrays["supervision_metric_matrix"][:, selected_indices]
    return selected_arrays


def draw_heightmap_with_coast(
    axis,
    heightmap: np.ndarray,
    title: str,
    coast_threshold: float = COAST_THRESHOLD,
) -> None:
    axis.imshow(heightmap, cmap="terrain")
    coast_mask = np.asarray(heightmap, dtype=np.float32)
    axis.contour(
        coast_mask,
        levels=[coast_threshold],
        colors=["white"],
        linewidths=1.0,
        alpha=0.95,
    )
    axis.set_title(title)
    axis.axis("off")


def _draw_curve_on_axis(axis, values: Sequence[float], title: str, ylabel: str) -> None:
    axis.clear()
    axis.plot(values, linewidth=2, alpha=0.85)
    if len(values) >= 10:
        moving_average = np.convolve(values, np.ones(10) / 10, mode="valid")
        axis.plot(range(9, len(values)), moving_average, linewidth=2, color="red", label="10轮滑动均值")
        axis.legend()
    axis.set_title(title)
    axis.set_xlabel("轮次")
    axis.set_ylabel(ylabel)
    axis.grid(True, alpha=0.3)


def _draw_optional_curve_on_axis(axis, values: Sequence[float | None], title: str, ylabel: str) -> None:
    axis.clear()
    points = [
        (index, float(value))
        for index, value in enumerate(values)
        if value is not None and np.isfinite(float(value))
    ]
    if points:
        xs, ys = zip(*points)
        axis.plot(xs, ys, linewidth=2, alpha=0.85)
        if len(points) >= 10:
            moving_average = np.convolve(np.asarray(ys, dtype=np.float32), np.ones(10) / 10, mode="valid")
            axis.plot(xs[9:], moving_average, linewidth=2, color="red", label="10轮滑动均值")
            axis.legend()
    axis.set_title(title)
    axis.set_xlabel("轮次")
    axis.set_ylabel(ylabel)
    axis.grid(True, alpha=0.3)


def _draw_multi_curve_on_axis(
    axis,
    series: Dict[str, Sequence[float | None]],
    title: str,
    ylabel: str,
) -> None:
    axis.clear()
    has_points = False
    for label, values in series.items():
        points = [
            (index, float(value))
            for index, value in enumerate(values)
            if value is not None and np.isfinite(float(value))
        ]
        if not points:
            continue
        xs, ys = zip(*points)
        axis.plot(xs, ys, linewidth=2, alpha=0.85, label=label)
        has_points = True
    if has_points:
        axis.legend()
    axis.set_title(title)
    axis.set_xlabel("episode")
    axis.set_ylabel(ylabel)
    axis.grid(True, alpha=0.3)


def plot_curve(values: Sequence[float], title: str, ylabel: str, output_path: Path) -> None:
    plt.figure(figsize=(10, 5))
    axis = plt.gca()
    _draw_curve_on_axis(axis, values, title, ylabel)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


class LiveCurvePlotter:
    """Keeps a reward curve figure refreshed during training and mirrors it to disk."""

    def __init__(
        self,
        title: str,
        ylabel: str,
        output_path: Path,
        refresh_every: int = 1,
    ) -> None:
        self.title = title
        self.ylabel = ylabel
        self.output_path = output_path
        self.refresh_every = max(1, int(refresh_every))
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        self.figure = None
        self.axis = None
        self.interactive = False
        try:
            plt.ion()
            self.figure, self.axis = plt.subplots(figsize=(10, 5))
            self.interactive = True
        except Exception:
            self.figure = None
            self.axis = None
            self.interactive = False

    def update(self, values: Sequence[float], force: bool = False) -> None:
        if not values:
            return
        if not force and len(values) % self.refresh_every != 0:
            return

        if self.interactive and self.figure is not None and self.axis is not None:
            _draw_curve_on_axis(self.axis, values, self.title, self.ylabel)
            self.figure.tight_layout()
            self.figure.savefig(self.output_path, dpi=150, bbox_inches="tight")
            try:
                self.figure.canvas.draw_idle()
                self.figure.canvas.flush_events()
                plt.pause(0.001)
            except Exception:
                pass
        else:
            plot_curve(values, self.title, self.ylabel, self.output_path)

    def close(self, values: Sequence[float]) -> None:
        self.update(values, force=True)
        if self.interactive and self.figure is not None:
            try:
                plt.close(self.figure)
            except Exception:
                pass


class LiveTrainingDashboard:
    """Keeps a 2x2 training dashboard refreshed during training and saved to disk."""

    def __init__(
        self,
        title: str,
        output_path: Path,
        refresh_every: int = 1,
        ylabel: str | None = None,
    ) -> None:
        self.title = title
        self.output_path = output_path
        self.refresh_every = max(1, int(refresh_every))
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        self.figure = None
        self.axes = None
        self.interactive = False
        try:
            plt.ion()
            self.figure, self.axes = plt.subplots(2, 2, figsize=(12, 8))
            self.interactive = True
        except Exception:
            self.figure = None
            self.axes = None
            self.interactive = False

    def update(self, series: Dict[str, Sequence[float | None]], force: bool = False) -> None:
        rewards = series.get("reward", [])
        if not rewards:
            return
        if not force and len(rewards) % self.refresh_every != 0:
            return

        if self.interactive and self.figure is not None and self.axes is not None:
            axes = self.axes.flatten()
            _draw_optional_curve_on_axis(axes[0], rewards, "Reward", "episode reward")
            _draw_optional_curve_on_axis(axes[1], series.get("loss", []), "Loss", "loss")
            _draw_optional_curve_on_axis(axes[2], series.get("final_score", []), "Final total_score", "score")
            _draw_multi_curve_on_axis(
                axes[3],
                {
                    "score_gain": series.get("score_gain", []),
                    "path_score": series.get("path_score", []),
                    "connectivity_score": series.get("connectivity_score", []),
                },
                "Gain / path / connectivity",
                "score",
            )
            self.figure.suptitle(self.title)
            self.figure.tight_layout()
            self.figure.savefig(self.output_path, dpi=150, bbox_inches="tight")
            try:
                self.figure.canvas.draw_idle()
                self.figure.canvas.flush_events()
                plt.pause(0.001)
            except Exception:
                pass
        else:
            self._save_static(series)

    def close(self, series: Dict[str, Sequence[float | None]]) -> None:
        self.update(series, force=True)
        if self.interactive and self.figure is not None:
            try:
                plt.close(self.figure)
            except Exception:
                pass

    def _save_static(self, series: Dict[str, Sequence[float | None]]) -> None:
        figure, axes_grid = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes_grid.flatten()
        _draw_optional_curve_on_axis(axes[0], series.get("reward", []), "Reward", "episode reward")
        _draw_optional_curve_on_axis(axes[1], series.get("loss", []), "Loss", "loss")
        _draw_optional_curve_on_axis(axes[2], series.get("final_score", []), "Final total_score", "score")
        _draw_multi_curve_on_axis(
            axes[3],
            {
                "score_gain": series.get("score_gain", []),
                "path_score": series.get("path_score", []),
                "connectivity_score": series.get("connectivity_score", []),
            },
            "Gain / path / connectivity",
            "score",
        )
        figure.suptitle(self.title)
        figure.tight_layout()
        figure.savefig(self.output_path, dpi=150, bbox_inches="tight")
        plt.close(figure)


class TrainingScalarLogger:
    """Writes RL scalar logs to CSV and, when available, TensorBoard."""

    def __init__(self, log_dir: Path, agent_name: str, enabled: bool = True) -> None:
        self.log_dir = log_dir
        self.agent_name = agent_name
        self.enabled = bool(enabled)
        self.rows: List[Tuple[int, str, float]] = []
        self.writer = None
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.log_dir / f"{agent_name}_scalars.csv"

        if self.enabled:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self.writer = SummaryWriter(log_dir=str(self.log_dir / agent_name))
                print(f"{agent_name.upper()} TensorBoard logdir: {(self.log_dir / agent_name).resolve()}")
            except Exception as exc:
                print(f"{agent_name.upper()} TensorBoard disabled ({exc}). CSV scalars will still be written.")

    def add_scalar(self, tag: str, value: object, step: int) -> None:
        number = self._to_float(value)
        if number is None:
            return
        full_tag = f"{self.agent_name}/{tag}"
        self.rows.append((int(step), full_tag, number))
        if self.writer is not None:
            self.writer.add_scalar(full_tag, number, int(step))

    def add_dict(self, prefix: str, values: object, step: int) -> None:
        if isinstance(values, dict):
            for key, value in values.items():
                child_prefix = f"{prefix}/{key}" if prefix else str(key)
                self.add_dict(child_prefix, value, step)
            return
        self.add_scalar(prefix, values, step)

    def close(self) -> None:
        with self.csv_path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["step", "tag", "value"])
            writer.writerows(self.rows)
        if self.writer is not None:
            self.writer.close()
        print(f"{self.agent_name.upper()} scalar CSV: {self.csv_path.resolve()}")

    @staticmethod
    def _to_float(value: object) -> Optional[float]:
        if isinstance(value, bool):
            return float(value)
        if isinstance(value, (int, float, np.floating, np.integer)):
            number = float(value)
            return number if np.isfinite(number) else None
        return None


def resolve_tensorboard_dir(args: argparse.Namespace, output_dir: Path) -> Path:
    return Path(args.tensorboard_dir).resolve() if args.tensorboard_dir else output_dir / "tensorboard"


def resolve_rl_update_batch_size(args: argparse.Namespace) -> int:
    value = int(getattr(args, "rl_update_batch_size", 0) or 0)
    return max(1, value if value > 0 else int(args.batch_size))


def rolling_finite_mean(values: Sequence[float | None], window: int) -> Optional[float]:
    recent = [float(value) for value in values[-max(1, int(window)) :] if value is not None and np.isfinite(value)]
    if not recent:
        return None
    return float(np.mean(recent))


def write_episode_scalars(
    logger: TrainingScalarLogger,
    episode: int,
    episode_reward: float,
    losses: Dict[str, float] | None,
    reset_info: dict,
    last_info: Optional[dict],
    extra: Optional[Dict[str, object]] = None,
) -> None:
    logger.add_scalar("reward", episode_reward, episode)
    if last_info is not None:
        logger.add_dict("score", last_info.get("score", {}), episode)
        logger.add_dict("metrics", last_info.get("metrics", {}), episode)
        logger.add_dict("reward_components", last_info.get("reward_components", {}), episode)
        score = last_info.get("score", {})
        reset_score = reset_info.get("score", {})
        logger.add_scalar(
            "score_gain/total_score",
            float(score.get("total_score", 0.0)) - float(reset_score.get("total_score", 0.0)),
            episode,
        )
        done_reason = last_info.get("done_reason")
        for reason in ("success_threshold", "stagnation", "max_steps", "failure_threshold"):
            logger.add_scalar(f"done_reason/{reason}", float(done_reason == reason), episode)
    if losses:
        logger.add_dict("losses", losses, episode)
    if extra:
        logger.add_dict("extra", extra, episode)


def plot_dataset_samples(heightmaps: np.ndarray, output_path: Path, num_samples: int = 9) -> None:
    if len(heightmaps) == 0:
        return
    num_samples = min(num_samples, len(heightmaps))
    cols = 3
    rows = int(np.ceil(num_samples / cols))
    plt.figure(figsize=(12, 4 * rows))
    for idx in range(num_samples):
        axis = plt.subplot(rows, cols, idx + 1)
        draw_heightmap_with_coast(axis, heightmaps[idx], f"样本 {idx + 1}")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_reconstruction(originals: np.ndarray, reconstructions: np.ndarray, output_path: Path) -> None:
    if len(originals) == 0:
        return
    num_samples = min(6, len(originals))
    fig, axes = plt.subplots(2, num_samples, figsize=(3 * num_samples, 6))
    axes = np.asarray(axes).reshape(2, num_samples)
    for idx in range(num_samples):
        draw_heightmap_with_coast(axes[0, idx], originals[idx], f"原图 {idx + 1}")
        draw_heightmap_with_coast(axes[1, idx], reconstructions[idx], f"重建 {idx + 1}")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_metric_bars(metric_values: Dict[str, float], title: str, ylabel: str, output_path: Path) -> None:
    if len(metric_values) == 0:
        return
    names = list(metric_values.keys())
    values = [metric_values[name] for name in names]

    plt.figure(figsize=(10, 5))
    display_names = [metric_label(name) for name in names]
    plt.bar(display_names, values, color="#4C78A8")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=30, ha="right")
    plt.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_latent_projection(
    latents: np.ndarray,
    score_values: np.ndarray,
    land_values: np.ndarray,
    output_path: Path,
) -> None:
    if len(latents) < 2:
        return

    if latents.shape[1] >= 2:
        points = PCA(n_components=2).fit_transform(latents)
    else:
        points = np.concatenate([latents, np.zeros((len(latents), 1), dtype=np.float32)], axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    scatter_1 = axes[0].scatter(points[:, 0], points[:, 1], c=score_values, cmap="viridis", s=36)
    axes[0].set_title("隐空间 PCA（按总评分着色）")
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")
    fig.colorbar(scatter_1, ax=axes[0], fraction=0.046, pad=0.04)

    scatter_2 = axes[1].scatter(points[:, 0], points[:, 1], c=land_values, cmap="plasma", s=36)
    axes[1].set_title("隐空间 PCA（按陆地占比着色）")
    axes[1].set_xlabel("PC1")
    axes[1].set_ylabel("PC2")
    fig.colorbar(scatter_2, ax=axes[1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_ranked_maps(
    heightmaps: Sequence[np.ndarray],
    scores: Sequence[float],
    output_path: Path,
    title_prefix: str,
) -> None:
    if len(heightmaps) == 0:
        return

    num_samples = min(4, len(heightmaps))
    ranked_indices = np.argsort(np.asarray(scores))
    bottom_indices = ranked_indices[:num_samples]
    top_indices = ranked_indices[-num_samples:][::-1]

    fig, axes = plt.subplots(2, num_samples, figsize=(3.2 * num_samples, 6))
    axes = np.asarray(axes).reshape(2, num_samples)
    for col, idx in enumerate(top_indices):
        draw_heightmap_with_coast(axes[0, col], heightmaps[idx], f"高分 {col + 1}\n{scores[idx]:.3f}")
    for col, idx in enumerate(bottom_indices):
        draw_heightmap_with_coast(axes[1, col], heightmaps[idx], f"低分 {col + 1}\n{scores[idx]:.3f}")

    fig.suptitle(title_prefix)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def reconstruct_heightmaps(
    vae: BetaVAE,
    heightmaps: np.ndarray,
    batch_size: int,
    device: torch.device,
    deterministic: bool = True,
) -> np.ndarray:
    vae.eval()
    batches: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(heightmaps), batch_size):
            batch = torch.as_tensor(heightmaps[start : start + batch_size], dtype=torch.float32, device=device)
            inputs = batch.unsqueeze(1)
            if deterministic:
                reconstruction, _, _, _ = vae.reconstruct_from_input(inputs, deterministic=True)
            else:
                reconstruction, _, _, _ = vae(inputs)
            batches.append(reconstruction.squeeze(1).cpu().numpy())
    return np.concatenate(batches, axis=0) if batches else np.empty_like(heightmaps)


def build_focus_masks(
    heightmaps: np.ndarray,
    water_threshold: float = COAST_THRESHOLD,
) -> Tuple[np.ndarray, np.ndarray]:
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


def predict_structure_targets(
    vae: BetaVAE,
    heightmaps: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    if vae.structure_predictor is None:
        return np.empty((len(heightmaps), 0), dtype=np.float32)

    vae.eval()
    predictions: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(heightmaps), batch_size):
            batch = torch.as_tensor(heightmaps[start : start + batch_size], dtype=torch.float32, device=device)
            mu, _ = vae.encode(batch.unsqueeze(1))
            predicted = vae.predict_structure(mu)
            if predicted is not None:
                predictions.append(predicted.cpu().numpy())
    if not predictions:
        return np.empty((len(heightmaps), 0), dtype=np.float32)
    return np.concatenate(predictions, axis=0)


def build_dataset(
    args: argparse.Namespace,
    output_dir: Path,
) -> Tuple[IslandDatasetBuilder, List, Dict[str, np.ndarray], Dict[str, object]]:
    print_section("第一阶段：数据集构建 / 清洗 / 评估")
    builder = IslandDatasetBuilder(map_size=args.map_size, sampling_profile=args.sampling_profile)

    raw_samples = []
    clean_samples = []
    round_index = 0

    while len(clean_samples) < args.min_clean_samples and len(raw_samples) < args.max_dataset_samples:
        remaining_budget = args.max_dataset_samples - len(raw_samples)
        chunk_size = min(args.dataset_samples, remaining_budget)
        if chunk_size <= 0:
            break

        chunk = builder.generate_samples(
            n_samples=chunk_size,
            seed=args.seed + round_index * 997,
            sampling_profile=args.sampling_profile,
        )
        raw_samples.extend(chunk)
        clean_samples = builder.clean_samples(raw_samples)
        round_index += 1

    raw_summary = builder.evaluate_dataset(raw_samples)
    clean_summary = builder.evaluate_dataset(clean_samples)
    raw_summary["sampling_profile"] = args.sampling_profile
    raw_summary["sampling_rounds"] = round_index
    raw_summary["target_clean_samples"] = args.min_clean_samples
    clean_summary["sampling_profile"] = args.sampling_profile
    clean_summary["sampling_rounds"] = round_index
    clean_summary["target_clean_samples"] = args.min_clean_samples

    print("原始数据集统计:")
    print_dataset_summary(raw_summary)
    print("\n清洗后数据集统计:")
    print_dataset_summary(clean_summary)
    print(f"\n采样轮数            : {round_index}")
    print(f"采样策略            : {args.sampling_profile}")
    print(f"目标清洗样本数      : {args.min_clean_samples}")

    arrays = builder.build_training_arrays(clean_samples)
    plot_dataset_samples(arrays["heightmaps"], output_dir / "dataset_samples.png")
    plot_ranked_maps(
        [sample.heightmap for sample in clean_samples],
        [sample.score for sample in clean_samples],
        output_dir / "dataset_score_extremes.png",
        title_prefix="清洗后数据集评分极值样本",
    )

    save_json(raw_summary, output_dir / "dataset_summary_raw.json")
    save_json(clean_summary, output_dir / "dataset_summary_clean.json")
    return builder, clean_samples, arrays, clean_summary


def resolve_rl_reset_profile(args: argparse.Namespace) -> str:
    return args.rl_reset_profile or args.sampling_profile


def build_rl_param_normalizer(args: argparse.Namespace) -> ParameterSpaceNormalizer:
    profile = resolve_rl_reset_profile(args)
    param_ranges = PCGIslandGenerator.get_param_ranges(args.map_size, profile=profile)
    return ParameterSpaceNormalizer(param_ranges=param_ranges)


def build_rl_reference_bank(
    args: argparse.Namespace,
    clean_samples: Sequence,
    feature_normalizer: IslandFeatureNormalizer,
    param_normalizer,
    output_dir: Path,
    latent_matrix: np.ndarray | None = None,
) -> Dict[str, np.ndarray]:
    valid_samples = [sample for sample in clean_samples if sample.valid]
    if not valid_samples:
        raise ValueError("No valid cleaned samples are available for RL reference-bank construction.")

    novelty_candidates = sorted(valid_samples, key=lambda sample: sample.score)
    novelty_count = min(max(1, args.novelty_reference_size), len(novelty_candidates))
    novelty_indices = np.unique(np.linspace(0, len(novelty_candidates) - 1, novelty_count, dtype=int))
    novelty_samples = [novelty_candidates[int(idx)] for idx in novelty_indices]
    sample_to_valid_index = {id(sample): index for index, sample in enumerate(valid_samples)}
    novelty_reference_kind = "metrics"
    if latent_matrix is not None:
        if len(latent_matrix) != len(valid_samples):
            raise ValueError(
                f"latent_matrix length must match valid sample count: {len(latent_matrix)} vs {len(valid_samples)}."
            )
        novelty_reference_vectors = np.stack(
            [
                feature_normalizer.transform_latent(latent_matrix[sample_to_valid_index[id(sample)]])
                for sample in novelty_samples
            ],
            axis=0,
        ).astype(np.float32)
        novelty_reference_kind = "latent"
    else:
        novelty_reference_vectors = np.stack(
            [feature_normalizer.transform_metrics(sample.metrics) for sample in novelty_samples],
            axis=0,
        ).astype(np.float32)

    ranked_samples = sorted(valid_samples, key=lambda sample: sample.score, reverse=True)
    expert_count = min(
        max(1, int(np.ceil(len(ranked_samples) * args.expert_top_percent))),
        max(1, args.expert_max_samples),
        len(ranked_samples),
    )
    expert_samples = ranked_samples[:expert_count]
    expert_param_vectors = np.stack(
        [param_normalizer.normalize_params(sample.params) for sample in expert_samples],
        axis=0,
    ).astype(np.float32)
    expert_scores = np.asarray([sample.score for sample in expert_samples], dtype=np.float32)
    expert_metric_matrix = np.asarray(
        [[float(value) for value in sample.metrics.values()] for sample in expert_samples],
        dtype=np.float32,
    )
    expert_heightmaps = np.stack([sample.heightmap for sample in expert_samples], axis=0).astype(np.float32)

    np.savez_compressed(
        output_dir / "expert_bank.npz",
        normalized_params=expert_param_vectors,
        quality_scores=expert_scores,
        metric_matrix=expert_metric_matrix,
        heightmaps=expert_heightmaps,
        novelty_reference_vectors=novelty_reference_vectors,
        novelty_reference_kind=np.asarray([novelty_reference_kind]),
    )
    save_json(
        {
            "expert_count": int(expert_count),
            "expert_top_percent": float(args.expert_top_percent),
            "novelty_reference_count": int(len(novelty_reference_vectors)),
            "novelty_reference_kind": novelty_reference_kind,
            "experts": [
                {
                    "rank": index + 1,
                    "score": float(sample.score),
                    "params": {
                        key: float(value) if isinstance(value, (int, float, np.floating, np.integer)) else value
                        for key, value in sample.params.items()
                    },
                    "metrics": {key: float(value) for key, value in sample.metrics.items()},
                    "normalized_params": [
                        float(value)
                        for value in param_normalizer.normalize_params(sample.params)
                    ],
                }
                for index, sample in enumerate(expert_samples)
            ],
        },
        output_dir / "expert_bank_summary.json",
    )
    return {
        "novelty_reference_vectors": novelty_reference_vectors,
        "novelty_reference_kind": novelty_reference_kind,
        "expert_param_vectors": expert_param_vectors,
        "expert_scores": expert_scores,
    }


def _mean_training_loss(losses: Dict[str, float] | None) -> float | None:
    if not losses:
        return None
    preferred_names = ("total_loss", "q_loss", "policy_loss", "value_loss")
    values = [
        abs(float(losses[name]))
        for name in preferred_names
        if name in losses and np.isfinite(float(losses[name]))
    ]
    return float(np.mean(values)) if values else None


def _append_training_dashboard_point(
    series: Dict[str, List[float | None]],
    episode_reward: float,
    losses: Dict[str, float] | None,
    reset_info: Dict[str, object],
    last_info: Optional[dict],
) -> None:
    initial_score = reset_info.get("score", {}) if reset_info is not None else {}
    final_info = last_info or reset_info
    final_score = final_info.get("score", {}) if final_info is not None else {}

    initial_total = float(initial_score.get("total_score", np.nan))
    final_total = float(final_score.get("total_score", np.nan))
    score_gain = final_total - initial_total if np.isfinite(initial_total) and np.isfinite(final_total) else None

    series["reward"].append(float(episode_reward))
    series["loss"].append(_mean_training_loss(losses))
    series["final_score"].append(final_total if np.isfinite(final_total) else None)
    series["score_gain"].append(score_gain)
    series["path_score"].append(float(final_score["path_score"]) if "path_score" in final_score else None)
    series["connectivity_score"].append(
        float(final_score["connectivity_score"]) if "connectivity_score" in final_score else None
    )


def train_formal_vae(
    args: argparse.Namespace,
    arrays: Dict[str, np.ndarray],
    metric_names: Sequence[str],
    output_dir: Path,
    device: torch.device,
) -> Tuple[BetaVAE, np.ndarray, List[dict]]:
    print_section("第二阶段：VAE 训练与隐向量提取")
    dataset = HeightmapDataset(
        arrays["heightmaps"],
        structure_targets=arrays["supervision_metric_matrix"],
        augment=True,
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    structure_metric_names = tuple(metric_names)

    vae = BetaVAE(
        map_size=args.map_size,
        latent_dim=args.latent_dim,
        beta=args.vae_beta,
        beta_start=args.vae_beta_start,
        free_bits=args.vae_free_bits,
        gradient_loss_weight=args.vae_gradient_loss_weight,
        mask_loss_weight=args.vae_mask_loss_weight,
        coast_loss_weight=args.vae_coast_loss_weight,
        land_dice_loss_weight=args.vae_land_dice_loss_weight,
        coast_dice_loss_weight=args.vae_coast_dice_loss_weight,
        structure_dim=arrays["supervision_metric_matrix"].shape[1],
        structure_loss_weight=args.vae_structure_loss_weight,
        metric_alignment_loss_weight=args.vae_metric_alignment_loss_weight,
        structure_loss_weights=get_structure_supervision_weights(
            structure_metric_names,
            connectivity_weight=args.vae_connectivity_supervision_weight,
            path_reachability_weight=args.vae_path_reachability_supervision_weight,
        ),
        land_recon_focus_weight=args.vae_land_recon_focus_weight,
        coast_recon_focus_weight=args.vae_coast_recon_focus_weight,
    ).to(device)
    start_time = time.time()
    history = train_vae(
        vae,
        dataloader,
        epochs=args.vae_epochs,
        learning_rate=args.vae_lr,
        device=str(device),
        warmup_epochs=args.vae_warmup_epochs,
    )
    duration = time.time() - start_time
    print(f"VAE 训练完成，用时 {duration / 60:.2f} 分钟")

    latents = encode_heightmaps(vae, arrays["heightmaps"], batch_size=args.batch_size, device=str(device))
    print(f"latent 矩阵形状: {latents.shape}")
    print(f"latent 均值: {latents.mean():.4f}")
    print(f"latent 标准差: {latents.std():.4f}")

    plot_curve(
        [entry["total_loss"] for entry in history],
        title="VAE 训练损失曲线",
        ylabel="损失值",
        output_path=output_dir / "vae_training_curve.png",
    )

    reconstructions = reconstruct_heightmaps(vae, arrays["heightmaps"][:6], args.batch_size, device)
    plot_reconstruction(
        arrays["heightmaps"][:6],
        reconstructions,
        output_dir / "vae_reconstruction.png",
    )

    save_json(
        {
            "latent_shape": list(latents.shape),
            "latent_mean": float(latents.mean()),
            "latent_std": float(latents.std()),
            "vae_config": {
                "supervision_metric_names": list(structure_metric_names),
                "beta": args.vae_beta,
                "beta_start": args.vae_beta_start,
                "warmup_epochs": args.vae_warmup_epochs,
                "free_bits": args.vae_free_bits,
                "gradient_loss_weight": args.vae_gradient_loss_weight,
                "mask_loss_weight": args.vae_mask_loss_weight,
                "coast_loss_weight": args.vae_coast_loss_weight,
                "land_dice_loss_weight": args.vae_land_dice_loss_weight,
                "coast_dice_loss_weight": args.vae_coast_dice_loss_weight,
                "structure_loss_weight": args.vae_structure_loss_weight,
                "metric_alignment_loss_weight": args.vae_metric_alignment_loss_weight,
                "structure_supervision_weights": {
                    name: weight
                    for name, weight in zip(
                        structure_metric_names,
                        get_structure_supervision_weights(
                            structure_metric_names,
                            connectivity_weight=args.vae_connectivity_supervision_weight,
                            path_reachability_weight=args.vae_path_reachability_supervision_weight,
                        ),
                    )
                },
                "connectivity_supervision_weight": args.vae_connectivity_supervision_weight,
                "path_reachability_supervision_weight": args.vae_path_reachability_supervision_weight,
                "land_recon_focus_weight": args.vae_land_recon_focus_weight,
                "coast_recon_focus_weight": args.vae_coast_recon_focus_weight,
                "learning_rate": args.vae_lr,
            },
            "vae_history": history,
        },
        output_dir / "vae_summary.json",
    )
    return vae, latents, history


def evaluate_vae_representation(
    args: argparse.Namespace,
    builder: IslandDatasetBuilder,
    arrays: Dict[str, np.ndarray],
    structure_metric_names: Sequence[str],
    vae: BetaVAE,
    latents: np.ndarray,
    output_dir: Path,
    device: torch.device,
) -> Dict[str, object]:
    print_section("第四阶段：VAE 表征有效性评估")

    reconstructions = reconstruct_heightmaps(
        vae,
        arrays["heightmaps"],
        args.batch_size,
        device,
        deterministic=True,
    )
    plot_reconstruction(
        arrays["heightmaps"][:6],
        reconstructions[:6],
        output_dir / "vae_reconstruction.png",
    )
    original_metrics = arrays["supervision_metric_matrix"]
    structure_metric_names = tuple(structure_metric_names)
    land_mask, coast_band = build_focus_masks(arrays["heightmaps"])
    reconstructed_metrics = np.array(
        [
            [builder.evaluator.evaluate(heightmap)[name] for name in structure_metric_names]
            for heightmap in reconstructions
        ],
        dtype=np.float32,
    )
    structure_predictions = predict_structure_targets(vae, arrays["heightmaps"], args.batch_size, device)

    pixel_mse = float(np.mean((arrays["heightmaps"] - reconstructions) ** 2))
    pixel_mae = float(np.mean(np.abs(arrays["heightmaps"] - reconstructions)))
    land_pixel_mae = masked_mae(arrays["heightmaps"], reconstructions, land_mask)
    coast_band_mae = masked_mae(arrays["heightmaps"], reconstructions, coast_band)
    metric_mae = {
        name: float(np.mean(np.abs(original_metrics[:, idx] - reconstructed_metrics[:, idx])))
        for idx, name in enumerate(structure_metric_names)
    }
    metric_corr = {}
    for idx, name in enumerate(structure_metric_names):
        origin = original_metrics[:, idx]
        recon = reconstructed_metrics[:, idx]
        if np.std(origin) < 1e-8 or np.std(recon) < 1e-8:
            metric_corr[name] = 0.0
        else:
            metric_corr[name] = float(np.corrcoef(origin, recon)[0, 1])

    structure_head_mae = {}
    structure_head_corr = {}
    if structure_predictions.shape == original_metrics.shape:
        for idx, name in enumerate(structure_metric_names):
            predicted = structure_predictions[:, idx]
            origin = original_metrics[:, idx]
            structure_head_mae[name] = float(np.mean(np.abs(origin - predicted)))
            if np.std(origin) < 1e-8 or np.std(predicted) < 1e-8:
                structure_head_corr[name] = 0.0
            else:
                structure_head_corr[name] = float(np.corrcoef(origin, predicted)[0, 1])
    else:
        structure_head_mae = {name: 0.0 for name in structure_metric_names}
        structure_head_corr = {name: 0.0 for name in structure_metric_names}

    latent_std_per_dim = latents.std(axis=0) if len(latents) > 0 else np.zeros((args.latent_dim,), dtype=np.float32)
    latent_global_std = float(latents.std()) if len(latents) > 0 else 0.0
    active_dim_threshold = max(1e-3, latent_global_std * 0.02)
    active_dims = int(np.sum(latent_std_per_dim > active_dim_threshold))

    predictive_r2 = {}
    if len(latents) >= 6:
        n_splits = min(5, len(latents))
        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=args.seed)
        for idx, name in enumerate(structure_metric_names):
            target = original_metrics[:, idx]
            if float(np.std(target)) < 1e-8:
                predictive_r2[name] = 0.0
                continue
            scores = cross_val_score(
                LinearRegression(),
                latents,
                target,
                cv=kfold,
                scoring="r2",
            )
            predictive_r2[name] = float(np.mean(scores))
    else:
        predictive_r2 = {name: 0.0 for name in structure_metric_names}

    land_index = builder.evaluator.metric_names.index("land_ratio")
    plot_metric_bars(metric_mae, "VAE 结构指标重建误差", "平均绝对误差", output_dir / "vae_metric_mae.png")
    plot_metric_bars(
        structure_head_mae,
        "VAE latent 结构头误差",
        "平均绝对误差",
        output_dir / "vae_structure_head_mae.png",
    )
    plot_metric_bars(
        predictive_r2,
        "隐向量对结构指标的预测能力",
        "R²",
        output_dir / "vae_latent_predictiveness.png",
    )
    plot_latent_projection(
        latents,
        arrays["quality_scores"],
        original_metrics[:, land_index],
        output_dir / "vae_latent_space.png",
    )

    summary = {
        "reconstruction_mode": "deterministic_mu_decode",
        "pixel_mse": pixel_mse,
        "pixel_mae": pixel_mae,
        "land_pixel_mae": land_pixel_mae,
        "coast_band_mae": coast_band_mae,
        "metric_reconstruction_mae": metric_mae,
        "metric_reconstruction_correlation": metric_corr,
        "structure_head_mae": structure_head_mae,
        "structure_head_correlation": structure_head_corr,
        "latent_global_mean": float(latents.mean()) if len(latents) > 0 else 0.0,
        "latent_global_std": latent_global_std,
        "active_latent_threshold": active_dim_threshold,
        "active_latent_dims": active_dims,
        "latent_std_per_dim": {f"z{idx:02d}": float(value) for idx, value in enumerate(latent_std_per_dim)},
        "latent_predictive_r2": predictive_r2,
    }

    print("重建评估模式        : deterministic_mu_decode")
    print(f"像素级 MSE          : {pixel_mse:.6f}")
    print(f"像素级 MAE          : {pixel_mae:.6f}")
    print(f"陆地区域 MAE        : {land_pixel_mae:.6f}")
    print(f"海岸带 MAE          : {coast_band_mae:.6f}")
    print(f"活跃 latent 维度    : {active_dims} / {latents.shape[1]}")
    print("\n重建后结构指标 MAE:")
    print_metric_dict(metric_mae, precision=4)
    print("\n结构监督头 MAE:")
    print_metric_dict(structure_head_mae, precision=4)
    print("\nlatent 对结构指标的预测 R2:")
    print_metric_dict(predictive_r2, precision=4)

    save_json(summary, output_dir / "vae_representation_summary.json")
    return summary


def save_trained_vae_artifacts(
    args: argparse.Namespace,
    vae: BetaVAE,
    feature_normalizer: IslandFeatureNormalizer,
    metric_names: Sequence[str],
    output_dir: Path,
) -> None:
    artifact_dir = output_dir / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": vae.state_dict(),
            "map_size": args.map_size,
            "latent_dim": args.latent_dim,
            "vae_config": {
                "supervision_metric_names": list(metric_names),
                "beta": args.vae_beta,
                "beta_start": args.vae_beta_start,
                "free_bits": args.vae_free_bits,
                "gradient_loss_weight": args.vae_gradient_loss_weight,
                "mask_loss_weight": args.vae_mask_loss_weight,
                "coast_loss_weight": args.vae_coast_loss_weight,
                "land_dice_loss_weight": args.vae_land_dice_loss_weight,
                "coast_dice_loss_weight": args.vae_coast_dice_loss_weight,
                "structure_loss_weight": args.vae_structure_loss_weight,
                "metric_alignment_loss_weight": args.vae_metric_alignment_loss_weight,
                "structure_supervision_weights": {
                    name: weight
                    for name, weight in zip(
                        metric_names,
                        get_structure_supervision_weights(
                            metric_names,
                            connectivity_weight=args.vae_connectivity_supervision_weight,
                            path_reachability_weight=args.vae_path_reachability_supervision_weight,
                        ),
                    )
                },
                "connectivity_supervision_weight": args.vae_connectivity_supervision_weight,
                "path_reachability_supervision_weight": args.vae_path_reachability_supervision_weight,
                "land_recon_focus_weight": args.vae_land_recon_focus_weight,
                "coast_recon_focus_weight": args.vae_coast_recon_focus_weight,
            },
        },
        artifact_dir / "vae_checkpoint.pt",
    )
    save_json(feature_normalizer.to_dict(), artifact_dir / "feature_normalizer.json")


def _load_torch_checkpoint(path: Path, device: torch.device) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def load_reusable_rl_assets(
    args: argparse.Namespace,
    source_dir: Path,
    device: torch.device,
) -> Dict[str, object]:
    source_dir = source_dir.resolve()
    checkpoint_path = source_dir / "artifacts" / "vae_checkpoint.pt"
    normalizer_path = source_dir / "artifacts" / "feature_normalizer.json"
    expert_bank_path = source_dir / "expert_bank.npz"
    for path in (checkpoint_path, normalizer_path, expert_bank_path):
        if not path.exists():
            raise FileNotFoundError(f"Required reusable RL asset not found: {path}")

    checkpoint = _load_torch_checkpoint(checkpoint_path, device)
    checkpoint_map_size = int(checkpoint["map_size"])
    checkpoint_latent_dim = int(checkpoint["latent_dim"])
    if int(args.map_size) != checkpoint_map_size:
        raise ValueError(
            f"Map size mismatch for reusable assets: command={args.map_size}, checkpoint={checkpoint_map_size}."
        )
    if int(args.latent_dim) != checkpoint_latent_dim:
        raise ValueError(
            f"Latent dimension mismatch for reusable assets: command={args.latent_dim}, checkpoint={checkpoint_latent_dim}."
        )

    vae_config = dict(checkpoint.get("vae_config", {}))
    supervision_names = list(vae_config.get("supervision_metric_names", []))
    supervision_weight_map = dict(vae_config.get("structure_supervision_weights", {}))
    structure_weights = [float(supervision_weight_map.get(name, 1.0)) for name in supervision_names]
    vae = BetaVAE(
        map_size=checkpoint_map_size,
        latent_dim=checkpoint_latent_dim,
        beta=float(vae_config.get("beta", args.vae_beta)),
        beta_start=float(vae_config.get("beta_start", args.vae_beta_start)),
        free_bits=float(vae_config.get("free_bits", args.vae_free_bits)),
        gradient_loss_weight=float(vae_config.get("gradient_loss_weight", args.vae_gradient_loss_weight)),
        mask_loss_weight=float(vae_config.get("mask_loss_weight", args.vae_mask_loss_weight)),
        coast_loss_weight=float(vae_config.get("coast_loss_weight", args.vae_coast_loss_weight)),
        land_dice_loss_weight=float(vae_config.get("land_dice_loss_weight", args.vae_land_dice_loss_weight)),
        coast_dice_loss_weight=float(vae_config.get("coast_dice_loss_weight", args.vae_coast_dice_loss_weight)),
        structure_dim=len(supervision_names),
        structure_loss_weight=float(vae_config.get("structure_loss_weight", args.vae_structure_loss_weight)),
        metric_alignment_loss_weight=float(
            vae_config.get("metric_alignment_loss_weight", args.vae_metric_alignment_loss_weight)
        ),
        structure_loss_weights=structure_weights,
        land_recon_focus_weight=float(
            vae_config.get("land_recon_focus_weight", args.vae_land_recon_focus_weight)
        ),
        coast_recon_focus_weight=float(
            vae_config.get("coast_recon_focus_weight", args.vae_coast_recon_focus_weight)
        ),
    ).to(device)
    vae.load_state_dict(checkpoint["state_dict"])
    vae.eval()

    normalizer_data = json.loads(normalizer_path.read_text(encoding="utf-8"))
    feature_normalizer = IslandFeatureNormalizer.from_dict(normalizer_data)
    if feature_normalizer.latent_normalizer.mean is None:
        raise ValueError(f"Reusable feature normalizer has no latent statistics: {normalizer_path}")
    if len(feature_normalizer.latent_normalizer.mean) != checkpoint_latent_dim:
        raise ValueError(
            "Reusable normalizer latent dimension does not match checkpoint: "
            f"{len(feature_normalizer.latent_normalizer.mean)} vs {checkpoint_latent_dim}."
        )

    with np.load(expert_bank_path) as bank:
        novelty_reference_vectors = np.asarray(bank["novelty_reference_vectors"], dtype=np.float32)
        expert_param_vectors = np.asarray(bank["normalized_params"], dtype=np.float32)
        expert_scores = np.asarray(bank["quality_scores"], dtype=np.float32)
        novelty_reference_kind = str(np.asarray(bank["novelty_reference_kind"]).reshape(-1)[0])
    if novelty_reference_kind == "latent" and novelty_reference_vectors.shape[1] != checkpoint_latent_dim:
        raise ValueError(
            "Reusable novelty bank latent dimension does not match checkpoint: "
            f"{novelty_reference_vectors.shape[1]} vs {checkpoint_latent_dim}."
        )

    vae_summary_path = source_dir / "vae_final_summary.json"
    vae_summary = (
        json.loads(vae_summary_path.read_text(encoding="utf-8"))
        if vae_summary_path.exists()
        else {"mode": "reused_rl_assets", "source_dir": str(source_dir)}
    )
    return {
        "vae": vae,
        "feature_normalizer": feature_normalizer,
        "reference_bank": {
            "novelty_reference_vectors": novelty_reference_vectors,
            "novelty_reference_kind": novelty_reference_kind,
            "expert_param_vectors": expert_param_vectors,
            "expert_scores": expert_scores,
        },
        "vae_summary": vae_summary,
        "source_dir": source_dir,
    }


def run_formal_vae_pipeline(
    args: argparse.Namespace,
    builder: IslandDatasetBuilder,
    clean_samples: List,
    arrays: Dict[str, np.ndarray],
    output_dir: Path,
    device: torch.device,
) -> Dict[str, object]:
    print_section("Formal VAE-only evaluation")
    selected_metric_names = get_selected_supervision_metric_names(
        args,
        builder.evaluator.supervision_metric_names,
    )

    train_idx, val_idx, test_idx = split_indices(
        num_samples=len(arrays["heightmaps"]),
        train_ratio=args.vae_train_ratio,
        val_ratio=args.vae_val_ratio,
        seed=args.seed,
    )
    split_indices_map = {
        "train": train_idx,
        "val": val_idx,
        "test": test_idx,
    }
    split_arrays_map = {
        name: select_supervision_metrics(
            subset_arrays(arrays, idx),
            builder.evaluator.supervision_metric_names,
            selected_metric_names,
        )
        for name, idx in split_indices_map.items()
    }
    split_clean_samples = {name: [clean_samples[int(i)] for i in idx] for name, idx in split_indices_map.items()}

    print(f"train/val/test sizes   : {len(train_idx)} / {len(val_idx)} / {len(test_idx)}")

    train_dir = output_dir / "train_split"
    train_dir.mkdir(parents=True, exist_ok=True)
    vae, train_latents, history = train_formal_vae(
        args,
        split_arrays_map["train"],
        selected_metric_names,
        train_dir,
        device,
    )
    feature_normalizer = builder.fit_feature_normalizer(
        split_clean_samples["train"],
        latent_matrix=train_latents,
    )
    save_trained_vae_artifacts(args, vae, feature_normalizer, selected_metric_names, output_dir)

    split_summaries: Dict[str, Dict[str, object]] = {}
    split_latents: Dict[str, np.ndarray] = {"train": train_latents}
    for split_name in ("val", "test"):
        split_latents[split_name] = encode_heightmaps(
            vae,
            split_arrays_map[split_name]["heightmaps"],
            batch_size=args.batch_size,
            device=str(device),
        )

    print_section("Formal VAE split evaluation")
    for split_name in ("train", "val", "test"):
        split_dir = output_dir / f"{split_name}_split"
        split_dir.mkdir(parents=True, exist_ok=True)
        split_summary = evaluate_vae_representation(
            args,
            builder,
            split_arrays_map[split_name],
            selected_metric_names,
            vae,
            split_latents[split_name],
            split_dir,
            device,
        )
        split_summary["num_samples"] = int(len(split_indices_map[split_name]))
        split_summary["quality_score_mean"] = float(split_arrays_map[split_name]["quality_scores"].mean())
        split_summary["quality_score_std"] = float(split_arrays_map[split_name]["quality_scores"].std())
        split_summaries[split_name] = split_summary
        save_json(builder.evaluate_dataset(split_clean_samples[split_name]), split_dir / "dataset_summary.json")

    generalization_gap = {
        "pixel_mae_gap_test_minus_train": float(
            split_summaries["test"]["pixel_mae"] - split_summaries["train"]["pixel_mae"]
        ),
        "land_pixel_mae_gap_test_minus_train": float(
            split_summaries["test"]["land_pixel_mae"] - split_summaries["train"]["land_pixel_mae"]
        ),
        "coast_band_mae_gap_test_minus_train": float(
            split_summaries["test"]["coast_band_mae"] - split_summaries["train"]["coast_band_mae"]
        ),
    }

    final_summary: Dict[str, object] = {
        "mode": "formal_vae_only",
        "preset_applied": True,
        "map_size": args.map_size,
        "latent_dim": args.latent_dim,
        "vae_epochs": args.vae_epochs,
        "batch_size": args.batch_size,
        "sampling_profile": args.sampling_profile,
        "drop_connectivity_supervision": bool(args.drop_connectivity_supervision),
        "supervision_metric_names": list(selected_metric_names),
        "clean_dataset_size": int(len(clean_samples)),
        "split_sizes": {
            "train": int(len(train_idx)),
            "val": int(len(val_idx)),
            "test": int(len(test_idx)),
        },
        "train_summary": split_summaries["train"],
        "val_summary": split_summaries["val"],
        "test_summary": split_summaries["test"],
        "generalization_gap": generalization_gap,
        "vae_config": {
            "beta": args.vae_beta,
            "beta_start": args.vae_beta_start,
            "warmup_epochs": args.vae_warmup_epochs,
            "free_bits": args.vae_free_bits,
            "gradient_loss_weight": args.vae_gradient_loss_weight,
            "mask_loss_weight": args.vae_mask_loss_weight,
            "coast_loss_weight": args.vae_coast_loss_weight,
            "land_dice_loss_weight": args.vae_land_dice_loss_weight,
            "coast_dice_loss_weight": args.vae_coast_dice_loss_weight,
            "structure_loss_weight": args.vae_structure_loss_weight,
            "metric_alignment_loss_weight": args.vae_metric_alignment_loss_weight,
            "structure_supervision_weights": {
                name: weight
                for name, weight in zip(
                    selected_metric_names,
                    get_structure_supervision_weights(
                        selected_metric_names,
                        connectivity_weight=args.vae_connectivity_supervision_weight,
                        path_reachability_weight=args.vae_path_reachability_supervision_weight,
                    ),
                )
            },
            "connectivity_supervision_weight": args.vae_connectivity_supervision_weight,
            "path_reachability_supervision_weight": args.vae_path_reachability_supervision_weight,
            "land_recon_focus_weight": args.vae_land_recon_focus_weight,
            "coast_recon_focus_weight": args.vae_coast_recon_focus_weight,
            "learning_rate": args.vae_lr,
        },
        "train_history_epochs": len(history),
    }
    save_json(final_summary, output_dir / "final_summary.json")
    return {
        "vae": vae,
        "train_latents": train_latents,
        "feature_normalizer": feature_normalizer,
        "history": history,
        "split_summaries": split_summaries,
        "final_summary": final_summary,
    }


def run_formal_vae_only_evaluation(
    args: argparse.Namespace,
    builder: IslandDatasetBuilder,
    clean_samples: List,
    arrays: Dict[str, np.ndarray],
    output_dir: Path,
    device: torch.device,
) -> Dict[str, object]:
    pipeline = run_formal_vae_pipeline(
        args=args,
        builder=builder,
        clean_samples=clean_samples,
        arrays=arrays,
        output_dir=output_dir,
        device=device,
    )
    return pipeline["final_summary"]


def build_env_factory(
    args: argparse.Namespace,
    vae: BetaVAE,
    feature_normalizer: IslandFeatureNormalizer,
    reference_bank: Dict[str, np.ndarray],
) -> Callable[[], IslandGenerationEnv]:
    def factory() -> IslandGenerationEnv:
        return IslandGenerationEnv(
            map_size=args.map_size,
            max_steps=args.ppo_max_steps,
            vae_model=vae,
            feature_normalizer=feature_normalizer,
            include_latent=args.include_latent_state,
            latent_novelty=True,
            action_step_scale=args.rl_action_step_scale,
            sampling_profile=resolve_rl_reset_profile(args),
            novelty_reference_vectors=reference_bank["novelty_reference_vectors"],
            expert_param_vectors=reference_bank["expert_param_vectors"],
            reward_current_scale=args.reward_current_scale,
            reward_delta_scale=args.reward_delta_scale,
            reward_best_scale=args.reward_best_scale,
            reward_step_penalty=args.reward_step_penalty,
            reward_success_bonus=args.reward_success_bonus,
            success_score_threshold=args.reward_success_threshold,
            success_score_gain_threshold=args.reward_success_gain_threshold,
            failure_score_threshold=args.reward_failure_threshold,
            success_streak_required=args.reward_success_streak,
            stagnation_patience=args.reward_stagnation_patience,
            stagnation_delta=args.reward_stagnation_delta,
        )

    return factory


def run_rl_agents_with_assets(
    args: argparse.Namespace,
    builder: IslandDatasetBuilder,
    vae: BetaVAE,
    feature_normalizer: IslandFeatureNormalizer,
    reference_bank: Dict[str, np.ndarray],
    vae_summary: Dict[str, object],
    output_dir: Path,
    device: torch.device,
    asset_source: Optional[Path] = None,
) -> Dict[str, object]:
    env_factory = build_env_factory(args, vae, feature_normalizer, reference_bank)
    zero_policy = ZeroPolicy(action_dim=len(builder.param_normalizer.param_names))
    random_policy = RandomPolicy(action_dim=len(builder.param_normalizer.param_names), seed=args.seed + 3000)
    zero_summary = evaluate_agent_with_gain("Zero", zero_policy, env_factory, output_dir, args.eval_islands, args.seed + 1000)
    random_summary = evaluate_agent_with_gain("Random", random_policy, env_factory, output_dir, args.eval_islands, args.seed + 1500)

    ppo_summary = None
    if args.rl_agent in {"ppo", "both"}:
        ppo_agent, _, _ = train_ppo(args, env_factory, output_dir, device)
        ppo_summary = evaluate_agent_with_gain(
            "PPO", ppo_agent, env_factory, output_dir, args.eval_islands, args.seed + 2000
        )

    sac_summary = None
    if args.rl_agent in {"sac", "both"} and args.sac_episodes > 0:
        sac_agent, _ = train_sac_with_logging(args, env_factory, output_dir, device)
        sac_summary = evaluate_agent_with_gain("SAC", sac_agent, env_factory, output_dir, args.eval_islands, args.seed + 4000)

    policy_summaries = {
        "Zero": zero_summary,
        "Random": random_summary,
    }
    if ppo_summary is not None:
        policy_summaries["PPO"] = ppo_summary
    if sac_summary is not None:
        policy_summaries["SAC"] = sac_summary

    env = env_factory()
    state_components = ["theta_norm", "metrics_norm"]
    if args.include_latent_state:
        state_components.insert(1, "z_norm")
    final_summary: Dict[str, object] = {
        "mode": "formal_rl",
        "vae_pipeline_summary": vae_summary,
        "state_definition": {
            "components": state_components,
            "param_dim": len(builder.param_normalizer.param_names),
            "latent_dim": int(vae.latent_dim) if args.include_latent_state else 0,
            "encoder_latent_dim": int(vae.latent_dim),
            "metric_dim": len(builder.evaluator.metric_names),
            "state_dim": int(env.observation_space.shape[0]),
            "action": "delta_theta_norm",
        },
        "reward_definition": builder.scorer.describe(),
        "rl_reference_bank": {
            "rl_reset_profile": resolve_rl_reset_profile(args),
            "expert_count": int(len(reference_bank["expert_param_vectors"])),
            "novelty_reference_count": int(len(reference_bank["novelty_reference_vectors"])),
            "novelty_reference_kind": str(reference_bank["novelty_reference_kind"]),
        },
        "rl_training_config": {
            "ppo_episodes": int(args.ppo_episodes),
            "ppo_max_steps": int(args.ppo_max_steps),
            "ppo_rollout_steps": int(args.ppo_rollout_steps),
            "ppo_hidden_dim": int(args.ppo_hidden_dim),
            "ppo_hidden_layers": int(args.ppo_hidden_layers),
            "ppo_lr": float(args.ppo_lr),
            "sac_episodes": int(args.sac_episodes),
            "sac_learning_starts": int(args.sac_learning_starts),
            "sac_actor_lr": float(args.sac_actor_lr),
            "sac_critic_lr": float(args.sac_critic_lr),
            "sac_alpha_lr": float(args.sac_alpha_lr),
            "sac_min_alpha": float(args.sac_min_alpha),
            "sac_target_entropy_scale": float(args.sac_target_entropy_scale),
            "rl_update_batch_size": int(resolve_rl_update_batch_size(args)),
            "rl_action_step_scale": float(args.rl_action_step_scale),
            "rl_agent": args.rl_agent,
            "include_latent_state": bool(args.include_latent_state),
            "asset_source": None if asset_source is None else str(asset_source.resolve()),
            "restore_best_policy": bool(args.restore_best_policy),
            "rl_best_window": int(args.rl_best_window),
            "tensorboard_dir": str(resolve_tensorboard_dir(args, output_dir)),
        },
        "policy_comparison": compare_policy_summaries_with_gain(policy_summaries),
        "zero_summary": zero_summary,
        "random_summary": random_summary,
    }
    if ppo_summary is not None:
        final_summary["ppo_summary"] = ppo_summary
    if sac_summary is not None:
        final_summary["sac_summary"] = sac_summary

    save_json(vae_summary, output_dir / "vae_final_summary.json")
    save_json(final_summary, output_dir / "final_summary.json")
    save_json(final_summary["policy_comparison"], output_dir / "policy_comparison.json")
    save_json(
        {
            "mode": "rl_experiment",
            "seed": int(args.seed),
            "agent": args.rl_agent,
            "include_latent_state": bool(args.include_latent_state),
            "ppo_hidden_layers": int(args.ppo_hidden_layers),
            "asset_source": None if asset_source is None else str(asset_source.resolve()),
        },
        output_dir / "experiment_manifest.json",
    )
    return final_summary


def run_formal_rl_experiment(
    args: argparse.Namespace,
    builder: IslandDatasetBuilder,
    clean_samples: List,
    arrays: Dict[str, np.ndarray],
    output_dir: Path,
    device: torch.device,
) -> Dict[str, object]:
    pipeline = run_formal_vae_pipeline(
        args=args,
        builder=builder,
        clean_samples=clean_samples,
        arrays=arrays,
        output_dir=output_dir,
        device=device,
    )
    vae = pipeline["vae"]
    feature_normalizer = pipeline["feature_normalizer"]
    vae_summary = pipeline["final_summary"]
    rl_latents = encode_heightmaps(
        vae,
        arrays["heightmaps"],
        batch_size=args.batch_size,
        device=str(device),
    )
    reference_bank = build_rl_reference_bank(
        args,
        clean_samples,
        feature_normalizer,
        build_rl_param_normalizer(args),
        output_dir,
        latent_matrix=rl_latents,
    )
    if args.prepare_rl_assets_only:
        summary = {
            "mode": "rl_assets_only",
            "map_size": int(args.map_size),
            "latent_dim": int(args.latent_dim),
            "sampling_profile": args.sampling_profile,
            "expert_count": int(len(reference_bank["expert_param_vectors"])),
            "novelty_reference_count": int(len(reference_bank["novelty_reference_vectors"])),
            "novelty_reference_kind": str(reference_bank["novelty_reference_kind"]),
            "vae_pipeline_summary": vae_summary,
        }
        save_json(vae_summary, output_dir / "vae_final_summary.json")
        save_json(summary, output_dir / "final_summary.json")
        save_json(summary, output_dir / "rl_assets_ready.json")
        return summary
    return run_rl_agents_with_assets(
        args=args,
        builder=builder,
        vae=vae,
        feature_normalizer=feature_normalizer,
        reference_bank=reference_bank,
        vae_summary=vae_summary,
        output_dir=output_dir,
        device=device,
    )


def run_formal_rl_with_reused_assets(
    args: argparse.Namespace,
    output_dir: Path,
    device: torch.device,
) -> Dict[str, object]:
    source_dir = Path(args.reuse_rl_assets_from)
    assets = load_reusable_rl_assets(args, source_dir, device)
    builder = IslandDatasetBuilder(
        map_size=args.map_size,
        sampling_profile=resolve_rl_reset_profile(args),
    )
    if tuple(builder.evaluator.metric_names) != tuple(assets["feature_normalizer"].metric_names):
        raise ValueError(
            "Reusable normalizer metric names do not match the restored v1 evaluator: "
            f"{assets['feature_normalizer'].metric_names} vs {builder.evaluator.metric_names}."
        )
    return run_rl_agents_with_assets(
        args=args,
        builder=builder,
        vae=assets["vae"],
        feature_normalizer=assets["feature_normalizer"],
        reference_bank=assets["reference_bank"],
        vae_summary=assets["vae_summary"],
        output_dir=output_dir,
        device=device,
        asset_source=assets["source_dir"],
    )


def train_ppo(
    args: argparse.Namespace,
    env_factory: Callable[[], IslandGenerationEnv],
    output_dir: Path,
    device: torch.device,
) -> Tuple[PPOAgent, List[float], List[dict]]:
    print_section("第五阶段：PPO 正式训练")
    env = env_factory()
    curve_path = output_dir / "ppo_training_curve.png"
    live_plotter = LiveTrainingDashboard(
        title="PPO training dashboard",
        output_path=curve_path,
        refresh_every=1,
    )
    print(f"PPO training dashboard: {curve_path.resolve()}")
    rl_update_batch_size = resolve_rl_update_batch_size(args)
    scalar_logger = TrainingScalarLogger(resolve_tensorboard_dir(args, output_dir), "ppo", enabled=args.tensorboard)
    print(f"PPO update batch size: {rl_update_batch_size}")
    agent = PPOAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        hidden_dim=args.ppo_hidden_dim,
        hidden_layers=args.ppo_hidden_layers,
        learning_rate=args.ppo_lr,
        batch_size=rl_update_batch_size,
        action_range=1.0,
    ).to(device)

    episode_rewards: List[float] = []
    episode_logs: List[dict] = []
    dashboard_series: Dict[str, List[float | None]] = {
        "reward": [],
        "loss": [],
        "final_score": [],
        "score_gain": [],
        "path_score": [],
        "connectivity_score": [],
    }
    rollout_memory = []
    rollout_target_steps = max(1, int(args.ppo_rollout_steps))
    best_checkpoint = {
        "episode": 0,
        "rolling_final_score": float("-inf"),
        "window": int(args.rl_best_window),
        "path": str(output_dir / "ppo_agent_best.pth"),
    }
    best_path = output_dir / "ppo_agent_best.pth"

    for episode in range(args.ppo_episodes):
        state, reset_info = env.reset(seed=args.seed + episode)
        memory = []
        episode_reward = 0.0
        last_info: Optional[dict] = None

        for _ in range(env.max_steps):
            action, log_prob = agent.get_action_and_log_prob(state, deterministic=False)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            memory.append((state, action, reward, next_state, done, log_prob))
            episode_reward += reward
            state = next_state
            last_info = info

            if done:
                break

        rollout_memory.extend(memory)
        should_update = len(rollout_memory) >= rollout_target_steps or episode == args.ppo_episodes - 1
        losses = agent.update(rollout_memory) if should_update else None
        if should_update:
            rollout_memory = []
        episode_rewards.append(float(episode_reward))
        _append_training_dashboard_point(dashboard_series, episode_reward, losses, reset_info, last_info)
        episode_logs.append(
            {
                "episode": episode + 1,
                "reward": float(episode_reward),
                "losses": losses or {},
                "updated_policy": bool(should_update),
                "initial_score": reset_info["score"],
                "score": last_info["score"] if last_info is not None else {},
                "score_gain": dashboard_series["score_gain"][-1],
            }
        )
        write_episode_scalars(
            scalar_logger,
            episode + 1,
            float(episode_reward),
            losses or {},
            reset_info,
            last_info,
            extra={
                "updated_policy": bool(should_update),
                "rollout_memory_steps": len(rollout_memory),
            },
        )
        rolling_score = rolling_finite_mean(dashboard_series["final_score"], args.rl_best_window)
        if rolling_score is not None and rolling_score > float(best_checkpoint["rolling_final_score"]):
            best_checkpoint.update(
                {
                    "episode": episode + 1,
                    "rolling_final_score": float(rolling_score),
                }
            )
            torch.save(agent.network.state_dict(), best_path)
        live_plotter.update(dashboard_series)

        if (episode + 1) % 10 == 0 or episode == 0:
            print(
                f"第 {episode + 1:>3} 轮 / {args.ppo_episodes} 轮 | "
                f"累计奖励 {episode_reward:.4f} | "
                f"最近10轮平均奖励 {np.mean(episode_rewards[-10:]):.4f}"
            )

    live_plotter.close(dashboard_series)
    scalar_logger.close()
    torch.save(agent.network.state_dict(), output_dir / "ppo_agent_final.pth")
    selected_checkpoint = "final"
    if args.restore_best_policy and best_path.exists():
        agent.load(str(best_path))
        selected_checkpoint = "best"
    torch.save(agent.network.state_dict(), output_dir / "ppo_agent.pth")
    save_json(
        {
            "reward_summary": summarize_rewards(episode_rewards),
            "episodes": episode_logs,
            "update_batch_size": int(rl_update_batch_size),
            "learning_rate": float(args.ppo_lr),
            "hidden_dim": int(args.ppo_hidden_dim),
            "hidden_layers": int(args.ppo_hidden_layers),
            "state_dim": int(env.observation_space.shape[0]),
            "best_checkpoint": best_checkpoint,
            "selected_checkpoint": selected_checkpoint,
            "tensorboard_dir": str(resolve_tensorboard_dir(args, output_dir)),
        },
        output_dir / "ppo_training_summary.json",
    )
    return agent, episode_rewards, episode_logs


def train_sac_with_logging(
    args: argparse.Namespace,
    env_factory: Callable[[], IslandGenerationEnv],
    output_dir: Path,
    device: torch.device,
) -> Tuple[SACAgent, List[float]]:
    print_section("补充阶段：SAC 训练")
    env = env_factory()
    curve_path = output_dir / "sac_training_curve.png"
    live_plotter = LiveTrainingDashboard(
        title="SAC training dashboard",
        output_path=curve_path,
        refresh_every=1,
    )
    print(f"SAC training dashboard: {curve_path.resolve()}")
    rl_update_batch_size = resolve_rl_update_batch_size(args)
    scalar_logger = TrainingScalarLogger(resolve_tensorboard_dir(args, output_dir), "sac", enabled=args.tensorboard)
    print(f"SAC update batch size: {rl_update_batch_size}")
    agent = SACAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        hidden_dim=args.sac_hidden_dim,
        action_range=1.0,
        actor_learning_rate=args.sac_actor_lr,
        critic_learning_rate=args.sac_critic_lr,
        alpha_learning_rate=args.sac_alpha_lr,
        min_alpha=args.sac_min_alpha,
        target_entropy_scale=args.sac_target_entropy_scale,
    ).to(device)
    replay_buffer = ReplayBuffer(capacity=100_000)

    episode_rewards: List[float] = []
    episode_logs: List[dict] = []
    dashboard_series: Dict[str, List[float | None]] = {
        "reward": [],
        "loss": [],
        "final_score": [],
        "score_gain": [],
        "path_score": [],
        "connectivity_score": [],
    }
    best_checkpoint = {
        "episode": 0,
        "rolling_final_score": float("-inf"),
        "window": int(args.rl_best_window),
        "path": str(output_dir / "sac_agent_best.pth"),
    }
    best_path = output_dir / "sac_agent_best.pth"
    total_env_steps = 0
    for episode in range(args.sac_episodes):
        state, reset_info = env.reset(seed=args.seed + 1000 + episode)
        episode_reward = 0.0
        last_info: Optional[dict] = None
        last_losses: Dict[str, float] = {}
        episode_component_sums = {
            "current_term": 0.0,
            "delta_term": 0.0,
            "best_term": 0.0,
            "expert_term": 0.0,
            "step_penalty_term": 0.0,
            "success_bonus": 0.0,
            "failure_penalty": 0.0,
            "stagnation_penalty": 0.0,
        }
        positive_delta_steps = 0
        best_improve_steps = 0
        episode_steps = 0

        for _ in range(env.max_steps):
            if total_env_steps < args.sac_learning_starts:
                action = env.action_space.sample()
            else:
                action = agent.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            replay_buffer.push(state, action, reward, next_state, done)
            total_env_steps += 1
            if total_env_steps >= args.sac_learning_starts and len(replay_buffer) >= max(256, rl_update_batch_size * 4):
                losses = agent.update(replay_buffer, batch_size=rl_update_batch_size)
                if losses is not None:
                    last_losses = {key: float(value) for key, value in losses.items()}

            episode_reward += reward
            state = next_state
            last_info = info
            episode_steps += 1
            step_reward_components = info.get("reward_components", {})
            for key in episode_component_sums:
                episode_component_sums[key] += float(step_reward_components.get(key, 0.0))
            if float(step_reward_components.get("delta_score", 0.0)) > 0.0:
                positive_delta_steps += 1
            if float(step_reward_components.get("best_delta", 0.0)) > 0.0:
                best_improve_steps += 1
            if done:
                break

        episode_rewards.append(float(episode_reward))
        _append_training_dashboard_point(dashboard_series, episode_reward, last_losses, reset_info, last_info)
        episode_logs.append(
            {
                "episode": episode + 1,
                "reward": float(episode_reward),
                "total_env_steps": int(total_env_steps),
                "initial_score": reset_info["score"],
                "score": {} if last_info is None else last_info["score"],
                "score_gain": dashboard_series["score_gain"][-1],
                "reward_components": {} if last_info is None else last_info["reward_components"],
                "episode_component_sums": episode_component_sums,
                "positive_delta_steps": int(positive_delta_steps),
                "best_improve_steps": int(best_improve_steps),
                "episode_steps": int(episode_steps),
                "done_reason": None if last_info is None else last_info.get("done_reason"),
                "losses": last_losses,
            }
        )
        write_episode_scalars(
            scalar_logger,
            episode + 1,
            float(episode_reward),
            last_losses,
            reset_info,
            last_info,
            extra={
                "total_env_steps": int(total_env_steps),
                "episode_steps": int(episode_steps),
                "positive_delta_steps": int(positive_delta_steps),
                "best_improve_steps": int(best_improve_steps),
                "replay_buffer_size": int(len(replay_buffer)),
            },
        )
        rolling_score = rolling_finite_mean(dashboard_series["final_score"], args.rl_best_window)
        if rolling_score is not None and rolling_score > float(best_checkpoint["rolling_final_score"]):
            best_checkpoint.update(
                {
                    "episode": episode + 1,
                    "rolling_final_score": float(rolling_score),
                }
            )
            agent.save(best_path)
        live_plotter.update(dashboard_series)
        if (episode + 1) % args.sac_print_interval == 0 or episode == 0:
            reward_components = {} if last_info is None else last_info.get("reward_components", {})
            score = {} if last_info is None else last_info.get("score", {})
            avg_episode_reward = episode_reward / max(episode_steps, 1)
            print(
                f"第 {episode + 1:>3} 轮 / {args.sac_episodes} 轮 | "
                f"累计奖励 {episode_reward:.4f} | "
                f"单步平均奖励 {avg_episode_reward:.4f} | "
                f"最近均值 {np.mean(episode_rewards[-args.sac_print_interval:]):.4f} | "
                f"最终总评分 {score.get('total_score', float('nan')):.4f} | "
                f"末步增量 {reward_components.get('delta_score', float('nan')):.4f} | "
                f"最佳增量 {reward_components.get('best_delta', float('nan')):.4f} | "
                f"新颖性 {score.get('novelty_score', float('nan')):.4f} | "
                f"专家距离改善 {reward_components.get('expert_delta', float('nan')):.4f} | "
                f"结束原因 {None if last_info is None else last_info.get('done_reason')}"
            )
            print(
                f"    奖励累计: "
                f"delta={episode_component_sums['delta_term']:.4f}, "
                f"best={episode_component_sums['best_term']:.4f}, "
                f"expert={episode_component_sums['expert_term']:.4f}, "
                f"step={episode_component_sums['step_penalty_term']:.4f}, "
                f"success={episode_component_sums['success_bonus']:.4f}, "
                f"failure=-{episode_component_sums['failure_penalty']:.4f}, "
                f"stagnation=-{episode_component_sums['stagnation_penalty']:.4f}"
            )
            print(
                f"    奖励单步均值: "
                f"delta={episode_component_sums['delta_term'] / max(episode_steps, 1):.4f}, "
                f"best={episode_component_sums['best_term'] / max(episode_steps, 1):.4f}, "
                f"expert={episode_component_sums['expert_term'] / max(episode_steps, 1):.4f}, "
                f"step={episode_component_sums['step_penalty_term'] / max(episode_steps, 1):.4f}"
            )
            print(
                f"    训练诊断: "
                f"正增量步数={positive_delta_steps}/{max(episode_steps, 1)}, "
                f"刷新最佳步数={best_improve_steps}/{max(episode_steps, 1)}, "
                f"最终结构={score.get('structure_score', float('nan')):.4f}, "
                f"最终路径={score.get('path_score', float('nan')):.4f}, "
                f"最终面积={score.get('land_score', float('nan')):.4f}, "
                f"最终新颖性={score.get('novelty_score', float('nan')):.4f}"
            )
            if last_losses:
                print(
                    f"    损失监控: q={last_losses.get('q_loss', float('nan')):.4f}, "
                    f"policy={last_losses.get('policy_loss', float('nan')):.4f}, "
                    f"alpha={last_losses.get('alpha_loss', float('nan')):.4f}, "
                    f"alpha值={last_losses.get('alpha', float('nan')):.4f}"
                )

    live_plotter.close(dashboard_series)
    scalar_logger.close()
    agent.save(output_dir / "sac_agent_final.pth")
    selected_checkpoint = "final"
    if args.restore_best_policy and best_path.exists():
        agent.load(best_path)
        selected_checkpoint = "best"
    agent.save(output_dir / "sac_agent.pth")
    save_json(
        {
            "reward_summary": summarize_rewards(episode_rewards),
            "episodes": episode_logs,
            "total_env_steps": int(total_env_steps),
            "learning_starts": int(args.sac_learning_starts),
            "update_batch_size": int(rl_update_batch_size),
            "hidden_dim": int(args.sac_hidden_dim),
            "actor_lr": float(args.sac_actor_lr),
            "critic_lr": float(args.sac_critic_lr),
            "alpha_lr": float(args.sac_alpha_lr),
            "min_alpha": float(args.sac_min_alpha),
            "target_entropy_scale": float(args.sac_target_entropy_scale),
            "state_dim": int(env.observation_space.shape[0]),
            "best_checkpoint": best_checkpoint,
            "selected_checkpoint": selected_checkpoint,
            "tensorboard_dir": str(resolve_tensorboard_dir(args, output_dir)),
        },
        output_dir / "sac_training_summary.json",
    )
    return agent, episode_rewards


class ZeroPolicy:
    def __init__(self, action_dim: int):
        self.action_dim = action_dim

    def select_action(self, state: np.ndarray, deterministic: bool = True, evaluate: bool = True) -> np.ndarray:
        return np.zeros(self.action_dim, dtype=np.float32)


class RandomPolicy:
    def __init__(self, action_dim: int, seed: int = 42):
        self.action_dim = action_dim
        self.rng = np.random.default_rng(seed)

    def select_action(self, state: np.ndarray, deterministic: bool = False, evaluate: bool = False) -> np.ndarray:
        return self.rng.uniform(-1.0, 1.0, size=self.action_dim).astype(np.float32)


def select_eval_action(agent, state: np.ndarray) -> np.ndarray:
    if isinstance(agent, PPOAgent):
        return agent.select_action(state, deterministic=True)
    if isinstance(agent, SACAgent):
        return agent.select_action(state, evaluate=True)
    return agent.select_action(state)


def evaluate_agent(
    name: str,
    agent,
    env_factory: Callable[[], IslandGenerationEnv],
    output_dir: Path,
    num_islands: int,
    seed_offset: int,
) -> Dict[str, object]:
    print_section(f"第六阶段：{name} 最终评估")
    initial_metrics_list: List[Dict[str, float]] = []
    metrics_list: List[Dict[str, float]] = []
    initial_score_list: List[Dict[str, float]] = []
    score_list: List[Dict[str, float]] = []
    score_gain_list: List[Dict[str, float]] = []
    rewards: List[float] = []
    heightmaps: List[np.ndarray] = []
    records: List[Dict[str, object]] = []
    env = env_factory()

    for index in range(num_islands):
        state, reset_info = env.reset(seed=seed_offset + index)
        episode_reward = 0.0
        info = {}
        for _ in range(env.max_steps):
            action = select_eval_action(agent, state)
            state, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            if terminated or truncated:
                break

        initial_metrics = {key: float(value) for key, value in reset_info["metrics"].items()}
        final_metrics = {key: float(value) for key, value in info["metrics"].items()}
        initial_score = {key: float(value) for key, value in reset_info["score"].items()}
        final_score = {key: float(value) for key, value in info["score"].items()}
        score_gain = {
            key: float(final_score[key] - initial_score[key])
            for key in final_score.keys()
        }

        rewards.append(float(episode_reward))
        initial_metrics_list.append(initial_metrics)
        metrics_list.append(final_metrics)
        initial_score_list.append(initial_score)
        score_list.append(final_score)
        score_gain_list.append(score_gain)
        heightmaps.append(info["heightmap"])
        record = {
            "index": index + 1,
            "reward": float(episode_reward),
            "done_reason": info.get("done_reason"),
            "initial_metrics": initial_metrics,
            "final_metrics": final_metrics,
            "initial_score": initial_score,
            "final_score": final_score,
            "score_gain": score_gain,
        }
        records.append(record)

    metric_names = list(metrics_list[0].keys())
    summary = {
        "奖励统计": summarize_rewards(rewards),
        "指标均值": {key: float(np.mean([metrics[key] for metrics in metrics_list])) for key in metric_names},
        "指标标准差": {key: float(np.std([metrics[key] for metrics in metrics_list])) for key in metric_names},
        "评分均值": {key: float(np.mean([score[key] for score in score_list])) for key in score_list[0].keys()},
    }

    print("指标均值:")
    print_metric_dict(summary["指标均值"])
    print("\n评分均值:")
    print_metric_dict(summary["评分均值"])

    total_scores = [record["score"]["total_score"] for record in records]
    plot_dataset_samples(np.asarray(heightmaps), output_dir / f"{name.lower()}_generated_islands.png", num_samples=9)
    plot_ranked_maps(
        heightmaps,
        total_scores,
        output_dir / f"{name.lower()}_score_extremes.png",
        title_prefix=f"{name} 评分极值样本",
    )

    save_json(summary, output_dir / f"{name.lower()}_evaluation_summary.json")
    save_json({"records": records}, output_dir / f"{name.lower()}_evaluation_details.json")
    return summary


def evaluate_agent_with_gain(
    name: str,
    agent,
    env_factory: Callable[[], IslandGenerationEnv],
    output_dir: Path,
    num_islands: int,
    seed_offset: int,
) -> Dict[str, object]:
    print_section(f"{name} 最终评估（含提升量）")
    initial_metrics_list: List[Dict[str, float]] = []
    final_metrics_list: List[Dict[str, float]] = []
    initial_score_list: List[Dict[str, float]] = []
    final_score_list: List[Dict[str, float]] = []
    score_gain_list: List[Dict[str, float]] = []
    rewards: List[float] = []
    heightmaps: List[np.ndarray] = []
    records: List[Dict[str, object]] = []
    env = env_factory()

    for index in range(num_islands):
        state, reset_info = env.reset(seed=seed_offset + index)
        episode_reward = 0.0
        info = dict(reset_info)
        for _ in range(env.max_steps):
            action = select_eval_action(agent, state)
            state, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            if terminated or truncated:
                break

        initial_metrics = {key: float(value) for key, value in reset_info["metrics"].items()}
        final_metrics = {key: float(value) for key, value in info["metrics"].items()}
        initial_score = {key: float(value) for key, value in reset_info["score"].items()}
        final_score = {key: float(value) for key, value in info["score"].items()}
        score_gain = {key: float(final_score[key] - initial_score[key]) for key in final_score.keys()}

        rewards.append(float(episode_reward))
        initial_metrics_list.append(initial_metrics)
        final_metrics_list.append(final_metrics)
        initial_score_list.append(initial_score)
        final_score_list.append(final_score)
        score_gain_list.append(score_gain)
        heightmaps.append(info["heightmap"])
        records.append(
            {
                "index": index + 1,
                "reward": float(episode_reward),
                "done_reason": info.get("done_reason"),
                "initial_metrics": initial_metrics,
                "final_metrics": final_metrics,
                "initial_score": initial_score,
                "final_score": final_score,
                "score_gain": score_gain,
            }
        )

    metric_names = list(final_metrics_list[0].keys())
    score_names = list(final_score_list[0].keys())
    summary = {
        "reward_summary": summarize_rewards(rewards),
        "initial_metric_mean": {key: float(np.mean([metrics[key] for metrics in initial_metrics_list])) for key in metric_names},
        "final_metric_mean": {key: float(np.mean([metrics[key] for metrics in final_metrics_list])) for key in metric_names},
        "final_metric_std": {key: float(np.std([metrics[key] for metrics in final_metrics_list])) for key in metric_names},
        "initial_score_mean": {key: float(np.mean([score[key] for score in initial_score_list])) for key in score_names},
        "final_score_mean": {key: float(np.mean([score[key] for score in final_score_list])) for key in score_names},
        "score_gain_mean": {key: float(np.mean([score[key] for score in score_gain_list])) for key in score_names},
        "score_gain_std": {key: float(np.std([score[key] for score in score_gain_list])) for key in score_names},
        "濂栧姳缁熻": summarize_rewards(rewards),
        "初始指标均值": {key: float(np.mean([metrics[key] for metrics in initial_metrics_list])) for key in metric_names},
        "鎸囨爣鍧囧€?": {key: float(np.mean([metrics[key] for metrics in final_metrics_list])) for key in metric_names},
        "鎸囨爣鏍囧噯宸?": {key: float(np.std([metrics[key] for metrics in final_metrics_list])) for key in metric_names},
        "初始评分均值": {key: float(np.mean([score[key] for score in initial_score_list])) for key in score_names},
        "璇勫垎鍧囧€?": {key: float(np.mean([score[key] for score in final_score_list])) for key in score_names},
        "评分提升均值": {key: float(np.mean([score[key] for score in score_gain_list])) for key in score_names},
        "评分提升标准差": {key: float(np.std([score[key] for score in score_gain_list])) for key in score_names},
    }

    print("初始指标均值:")
    print_metric_dict(summary["初始指标均值"])
    print("\n最终指标均值:")
    print_metric_dict(summary["鎸囨爣鍧囧€?"])
    print("\n初始评分均值:")
    print_metric_dict(summary["初始评分均值"])
    print("\n最终评分均值:")
    print_metric_dict(summary["璇勫垎鍧囧€?"])
    print("\n评分提升均值:")
    print_metric_dict(summary["评分提升均值"])

    total_scores = [record["final_score"]["total_score"] for record in records]
    plot_dataset_samples(np.asarray(heightmaps), output_dir / f"{name.lower()}_generated_islands.png", num_samples=9)
    plot_ranked_maps(
        heightmaps,
        total_scores,
        output_dir / f"{name.lower()}_score_extremes.png",
        title_prefix=f"{name} 最终评分样本",
    )

    save_json(summary, output_dir / f"{name.lower()}_evaluation_summary.json")
    save_json({"records": records}, output_dir / f"{name.lower()}_evaluation_details.json")
    return summary


def compare_policy_summaries_with_gain(policy_summaries: Dict[str, Dict[str, object]]) -> Dict[str, object]:
    comparison = {
        policy_name: {
            "initial_total_score": float(summary["initial_score_mean"]["total_score"]),
            "final_total_score": float(summary["final_score_mean"]["total_score"]),
            "total_score_gain": float(summary["score_gain_mean"]["total_score"]),
            "structure_score_gain": float(summary["score_gain_mean"]["structure_score"]),
            "path_score_gain": float(summary["score_gain_mean"]["path_score"]),
            "land_score_gain": float(summary["score_gain_mean"]["land_score"]),
            "reward_mean": float(summary["reward_summary"]["mean"]),
        }
        for policy_name, summary in policy_summaries.items()
    }

    baseline_final_score = comparison["Zero"]["final_total_score"]
    baseline_score_gain = comparison["Zero"]["total_score_gain"]
    for values in comparison.values():
        values["delta_vs_zero_final_total_score"] = float(values["final_total_score"] - baseline_final_score)
        values["delta_vs_zero_total_score_gain"] = float(values["total_score_gain"] - baseline_score_gain)

    return comparison


def compare_policy_summaries(policy_summaries: Dict[str, Dict[str, object]]) -> Dict[str, object]:
    comparison = {
        policy_name: {
            "total_score": float(summary["评分均值"]["total_score"]),
            "structure_score": float(summary["评分均值"]["structure_score"]),
            "path_score": float(summary["评分均值"]["path_score"]),
            "land_score": float(summary["评分均值"]["land_score"]),
            "reward_mean": float(summary["奖励统计"]["mean"]),
        }
        for policy_name, summary in policy_summaries.items()
    }

    baseline_score = comparison["Zero"]["total_score"]
    for policy_name, values in comparison.items():
        values["delta_vs_zero_total_score"] = float(values["total_score"] - baseline_score)

    return comparison


def main() -> None:
    args = parse_args()
    apply_formal_vae_preset(args)
    apply_formal_rl_preset(args)
    apply_optuna_best_trial(args)
    apply_fast_profile(args)
    set_seed(args.seed)
    configure_matplotlib_chinese()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_section("??????")
    print(f"??????            : {device}")
    print(f"??????            : {output_dir.resolve()}")
    print(f"???????????     : {args.dataset_samples}")
    print(f"???????????     : {args.min_clean_samples}")
    print(f"???????????      : {args.max_dataset_samples}")
    print(f"??????            : {args.sampling_profile}")
    print(f"VAE ??????        : {args.vae_epochs}")
    print(f"PPO ??????        : {args.ppo_episodes}")
    print(f"SAC ??????        : {args.sac_episodes}")
    print(f"RL agent             : {args.rl_agent}")
    print(f"Include latent state : {args.include_latent_state}")
    print(f"??? VAE-only ???   : {args.formal_vae_only}")
    print(f"??? formal RL ???  : {args.formal_rl}")
    print(f"??? fast profile ? : {args.fast_profile}")
    if args.optuna_best_trial:
        print(f"Optuna ??????     : {Path(args.optuna_best_trial).resolve()}")
    if args.reuse_rl_assets_from:
        print(f"Reuse RL assets      : {Path(args.reuse_rl_assets_from).resolve()}")

    if args.formal_rl and args.reuse_rl_assets_from:
        if args.prepare_rl_assets_only:
            raise ValueError("--prepare-rl-assets-only cannot be combined with --reuse-rl-assets-from.")
        final_summary = run_formal_rl_with_reused_assets(args, output_dir, device)
        print_section("RL experiment complete")
        print(f"Output directory      : {output_dir.resolve()}")
        print(f"State dimension       : {final_summary['state_definition']['state_dim']}")
        print("- ppo/sac_training_summary.json (selected agent only)")
        print("- ppo/sac_evaluation_summary.json (selected agent only)")
        print("- tensorboard/ and policy_comparison.json")
        return

    builder, clean_samples, arrays, clean_summary = build_dataset(args, output_dir)

    if args.formal_vae_only:
        final_summary = run_formal_vae_only_evaluation(
            args=args,
            builder=builder,
            clean_samples=clean_samples,
            arrays=arrays,
            output_dir=output_dir,
            device=device,
        )
        print_section("??????")
        print(f"????????????    : {output_dir.resolve()}")
        print("?????????        : ??? VAE-only")
        print(
            f"train/val/test      : {final_summary['split_sizes']['train']} / "
            f"{final_summary['split_sizes']['val']} / {final_summary['split_sizes']['test']}"
        )
        print("- dataset_summary_raw.json / dataset_summary_clean.json")
        print("- train_split|val_split|test_split/vae_reconstruction.png")
        print("- train_split|val_split|test_split/vae_representation_summary.json")
        print("- final_summary.json")
        return

    if args.formal_rl:
        final_summary = run_formal_rl_experiment(
            args=args,
            builder=builder,
            clean_samples=clean_samples,
            arrays=arrays,
            output_dir=output_dir,
            device=device,
        )
        print_section("??????")
        print(f"????????????    : {output_dir.resolve()}")
        print("?????????        : ??? RL + frozen VAE")
        print("- artifacts/vae_checkpoint.pt / feature_normalizer.json")
        print("- train_split|val_split|test_split/vae_reconstruction.png")
        print("- zero/random/ppo/sac_evaluation_summary.json")
        print("- policy_comparison.json / final_summary.json")
        if args.prepare_rl_assets_only:
            print("- rl_assets_ready.json / expert_bank.npz")
        else:
            print(f"RL ????????      : {final_summary['state_definition']['state_dim']}")
        return

    selected_metric_names = get_selected_supervision_metric_names(
        args,
        builder.evaluator.supervision_metric_names,
    )
    selected_arrays = select_supervision_metrics(
        arrays,
        builder.evaluator.supervision_metric_names,
        selected_metric_names,
    )
    vae, latents, _ = train_formal_vae(args, selected_arrays, selected_metric_names, output_dir, device)

    print_section("第三阶段：特征归一化拟合")
    metric_names = builder.evaluator.metric_names
    feature_normalizer = builder.fit_feature_normalizer(clean_samples, latent_matrix=latents)
    param_dim = len(builder.param_normalizer.param_names)
    print(f"状态特征维度        : {param_dim + len(metric_names) + latents.shape[1]}")
    print(f"参数维度            : {param_dim}")
    print(f"结构指标维度        : {len(metric_names)}")
    print(f"latent 维度         : {latents.shape[1]}")

    vae_summary = evaluate_vae_representation(
        args,
        builder,
        selected_arrays,
        selected_metric_names,
        vae,
        latents,
        output_dir,
        device,
    )

    if args.skip_rl:
        final_summary: Dict[str, object] = {
            "清洗后有效样本数": clean_summary["num_valid"],
            "采样策略": args.sampling_profile,
            "VAE latent 维度": latents.shape[1],
            "评分器配置": builder.scorer.describe(),
            "VAE 表征评估": vae_summary,
        }
        save_json(final_summary, output_dir / "final_summary.json")
        print_section("实验完成")
        print(f"结果文件已保存到    : {output_dir.resolve()}")
        print("本次运行模式        : 仅 VAE / 跳过 RL")
        print("核心输出包括        :")
        print("- dataset_summary_raw.json / dataset_summary_clean.json")
        print("- dataset_samples.png / dataset_score_extremes.png")
        print("- vae_training_curve.png / vae_reconstruction.png")
        print("- vae_representation_summary.json / vae_metric_mae.png / vae_structure_head_mae.png / vae_latent_space.png")
        print("- final_summary.json")
        return

    reference_bank = build_rl_reference_bank(
        args,
        clean_samples,
        feature_normalizer,
        build_rl_param_normalizer(args),
        output_dir,
    )
    env_factory = build_env_factory(args, vae, feature_normalizer, reference_bank)
    zero_policy = ZeroPolicy(action_dim=len(builder.param_normalizer.param_names))
    random_policy = RandomPolicy(action_dim=len(builder.param_normalizer.param_names), seed=args.seed + 3000)
    zero_summary = evaluate_agent("Zero", zero_policy, env_factory, output_dir, args.eval_islands, args.seed + 1000)
    random_summary = evaluate_agent("Random", random_policy, env_factory, output_dir, args.eval_islands, args.seed + 1500)

    ppo_agent, _, _ = train_ppo(args, env_factory, output_dir, device)
    ppo_summary = evaluate_agent("PPO", ppo_agent, env_factory, output_dir, args.eval_islands, args.seed + 2000)

    policy_summaries = {
        "Zero": zero_summary,
        "Random": random_summary,
        "PPO": ppo_summary,
    }

    final_summary: Dict[str, object] = {
        "清洗后有效样本数": clean_summary["num_valid"],
        "采样策略": args.sampling_profile,
        "VAE latent 维度": latents.shape[1],
        "评分器配置": builder.scorer.describe(),
        "VAE 表征评估": vae_summary,
        "策略对比": compare_policy_summaries(policy_summaries),
        "PPO 评估总结": ppo_summary,
    }

    if args.sac_episodes > 0:
        sac_agent, _ = train_sac_with_logging(args, env_factory, output_dir, device)
        sac_summary = evaluate_agent("SAC", sac_agent, env_factory, output_dir, args.eval_islands, args.seed + 4000)
        final_summary["SAC 评估总结"] = sac_summary

    save_json(final_summary, output_dir / "final_summary.json")
    save_json(final_summary["策略对比"], output_dir / "policy_comparison.json")

    print_section("实验完成")
    print(f"结果文件已保存到    : {output_dir.resolve()}")
    print("核心输出包括        :")
    print("- dataset_summary_raw.json / dataset_summary_clean.json")
    print("- dataset_samples.png / dataset_score_extremes.png")
    print("- vae_training_curve.png / vae_reconstruction.png")
    print("- vae_representation_summary.json / vae_metric_mae.png / vae_structure_head_mae.png / vae_latent_space.png")
    print("- zero/random/ppo_evaluation_summary.json")
    print("- policy_comparison.json / final_summary.json")
    if args.sac_episodes > 0:
        print("- sac_training_curve.png / sac_evaluation_summary.json")


if __name__ == "__main__":
    main()
