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
from feature_processing import IslandFeatureNormalizer
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="IslandTest 正式实验运行脚本")
    parser.add_argument("--output-dir", type=str, default="formal_outputs", help="结果输出目录")
    parser.add_argument("--map-size", type=int, default=64, help="地图尺寸")
    parser.add_argument("--dataset-samples", type=int, default=120, help="每轮采样的原始样本数")
    parser.add_argument("--min-clean-samples", type=int, default=48, help="清洗后至少保留的样本数")
    parser.add_argument("--max-dataset-samples", type=int, default=480, help="数据集构建阶段允许的最大原始样本数")
    parser.add_argument("--sampling-profile", type=str, default="island", choices=["uniform", "island"], help="参数采样策略")
    parser.add_argument("--batch-size", type=int, default=32, help="VAE 和 PPO 的批大小")
    parser.add_argument("--latent-dim", type=int, default=16, help="VAE latent 维度")
    parser.add_argument("--vae-epochs", type=int, default=30, help="VAE 训练轮数")
    parser.add_argument("--vae-beta", type=float, default=0.25, help="Beta-VAE 的 beta")
    parser.add_argument("--vae-beta-start", type=float, default=0.0, help="KL warmup 起始 beta")
    parser.add_argument("--vae-warmup-epochs", type=int, default=12, help="KL warmup 轮数")
    parser.add_argument("--vae-free-bits", type=float, default=0.01, help="每个 latent 维度的最小 KL")
    parser.add_argument("--vae-gradient-loss-weight", type=float, default=0.20, help="边界/梯度重建损失权重")
    parser.add_argument("--vae-mask-loss-weight", type=float, default=0.15, help="陆地掩码重建损失权重")
    parser.add_argument("--vae-coast-loss-weight", type=float, default=0.20, help="海岸线重建损失权重")
    parser.add_argument("--vae-lr", type=float, default=8e-4, help="VAE 学习率")
    parser.add_argument("--ppo-episodes", type=int, default=60, help="PPO 训练轮数")
    parser.add_argument("--ppo-max-steps", type=int, default=30, help="PPO 单轮最大步数")
    parser.add_argument("--ppo-hidden-dim", type=int, default=256, help="PPO 隐层维度")
    parser.add_argument("--sac-episodes", type=int, default=0, help="SAC 补充训练轮数，0 表示跳过")
    parser.add_argument("--eval-islands", type=int, default=12, help="最终评估的地图数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    return parser.parse_args()


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


def plot_curve(values: Sequence[float], title: str, ylabel: str, output_path: Path) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(values, linewidth=2, alpha=0.85)
    if len(values) >= 10:
        moving_average = np.convolve(values, np.ones(10) / 10, mode="valid")
        plt.plot(range(9, len(values)), moving_average, linewidth=2, color="red", label="10轮滑动均值")
        plt.legend()
    plt.title(title)
    plt.xlabel("轮次")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


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
) -> np.ndarray:
    vae.eval()
    batches: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(heightmaps), batch_size):
            batch = torch.as_tensor(heightmaps[start : start + batch_size], dtype=torch.float32, device=device)
            reconstruction, _, _ = vae(batch.unsqueeze(1))
            batches.append(reconstruction.squeeze(1).cpu().numpy())
    return np.concatenate(batches, axis=0) if batches else np.empty_like(heightmaps)


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


def train_formal_vae(
    args: argparse.Namespace,
    arrays: Dict[str, np.ndarray],
    output_dir: Path,
    device: torch.device,
) -> Tuple[BetaVAE, np.ndarray, List[dict]]:
    print_section("第二阶段：VAE 训练与隐向量提取")
    dataset = HeightmapDataset(arrays["heightmaps"], augment=True)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    vae = BetaVAE(
        map_size=args.map_size,
        latent_dim=args.latent_dim,
        beta=args.vae_beta,
        beta_start=args.vae_beta_start,
        free_bits=args.vae_free_bits,
        gradient_loss_weight=args.vae_gradient_loss_weight,
        mask_loss_weight=args.vae_mask_loss_weight,
        coast_loss_weight=args.vae_coast_loss_weight,
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
                "beta": args.vae_beta,
                "beta_start": args.vae_beta_start,
                "warmup_epochs": args.vae_warmup_epochs,
                "free_bits": args.vae_free_bits,
                "gradient_loss_weight": args.vae_gradient_loss_weight,
                "mask_loss_weight": args.vae_mask_loss_weight,
                "coast_loss_weight": args.vae_coast_loss_weight,
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
    vae: BetaVAE,
    latents: np.ndarray,
    output_dir: Path,
    device: torch.device,
) -> Dict[str, object]:
    print_section("第四阶段：VAE 表征有效性评估")

    reconstructions = reconstruct_heightmaps(vae, arrays["heightmaps"], args.batch_size, device)
    original_metrics = arrays["metric_matrix"]
    reconstructed_metrics = np.array(
        [
            [builder.evaluator.evaluate(heightmap)[name] for name in builder.evaluator.metric_names]
            for heightmap in reconstructions
        ],
        dtype=np.float32,
    )

    pixel_mse = float(np.mean((arrays["heightmaps"] - reconstructions) ** 2))
    pixel_mae = float(np.mean(np.abs(arrays["heightmaps"] - reconstructions)))
    metric_mae = {
        name: float(np.mean(np.abs(original_metrics[:, idx] - reconstructed_metrics[:, idx])))
        for idx, name in enumerate(builder.evaluator.metric_names)
    }
    metric_corr = {}
    for idx, name in enumerate(builder.evaluator.metric_names):
        origin = original_metrics[:, idx]
        recon = reconstructed_metrics[:, idx]
        if np.std(origin) < 1e-8 or np.std(recon) < 1e-8:
            metric_corr[name] = 0.0
        else:
            metric_corr[name] = float(np.corrcoef(origin, recon)[0, 1])

    latent_std_per_dim = latents.std(axis=0) if len(latents) > 0 else np.zeros((args.latent_dim,), dtype=np.float32)
    latent_global_std = float(latents.std()) if len(latents) > 0 else 0.0
    active_dim_threshold = max(1e-3, latent_global_std * 0.02)
    active_dims = int(np.sum(latent_std_per_dim > active_dim_threshold))

    predictive_r2 = {}
    if len(latents) >= 6:
        n_splits = min(5, len(latents))
        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=args.seed)
        for idx, name in enumerate(builder.evaluator.metric_names):
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
        predictive_r2 = {name: 0.0 for name in builder.evaluator.metric_names}

    land_index = builder.evaluator.metric_names.index("land_ratio")
    plot_metric_bars(metric_mae, "VAE 结构指标重建误差", "平均绝对误差", output_dir / "vae_metric_mae.png")
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
        "pixel_mse": pixel_mse,
        "pixel_mae": pixel_mae,
        "metric_reconstruction_mae": metric_mae,
        "metric_reconstruction_correlation": metric_corr,
        "latent_global_mean": float(latents.mean()) if len(latents) > 0 else 0.0,
        "latent_global_std": latent_global_std,
        "active_latent_threshold": active_dim_threshold,
        "active_latent_dims": active_dims,
        "latent_std_per_dim": {f"z{idx:02d}": float(value) for idx, value in enumerate(latent_std_per_dim)},
        "latent_predictive_r2": predictive_r2,
    }

    print(f"像素级 MSE          : {pixel_mse:.6f}")
    print(f"像素级 MAE          : {pixel_mae:.6f}")
    print(f"活跃 latent 维度    : {active_dims} / {latents.shape[1]}")
    print("\n重建后结构指标 MAE:")
    print_metric_dict(metric_mae, precision=4)
    print("\nlatent 对结构指标的预测 R2:")
    print_metric_dict(predictive_r2, precision=4)

    save_json(summary, output_dir / "vae_representation_summary.json")
    return summary


def build_env_factory(
    args: argparse.Namespace,
    vae: BetaVAE,
    feature_normalizer: IslandFeatureNormalizer,
) -> Callable[[], IslandGenerationEnv]:
    def factory() -> IslandGenerationEnv:
        return IslandGenerationEnv(
            map_size=args.map_size,
            max_steps=args.ppo_max_steps,
            vae_model=vae,
            feature_normalizer=feature_normalizer,
            include_latent=True,
            sampling_profile=args.sampling_profile,
        )

    return factory


def train_ppo(
    args: argparse.Namespace,
    env_factory: Callable[[], IslandGenerationEnv],
    output_dir: Path,
    device: torch.device,
) -> Tuple[PPOAgent, List[float], List[dict]]:
    print_section("第五阶段：PPO 正式训练")
    env = env_factory()
    agent = PPOAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        hidden_dim=args.ppo_hidden_dim,
        batch_size=args.batch_size,
        action_range=1.0,
    ).to(device)

    episode_rewards: List[float] = []
    episode_logs: List[dict] = []

    for episode in range(args.ppo_episodes):
        state, _ = env.reset(seed=args.seed + episode)
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

        losses = agent.update(memory)
        episode_rewards.append(float(episode_reward))
        episode_logs.append(
            {
                "episode": episode + 1,
                "reward": float(episode_reward),
                "losses": losses,
                "score": last_info["score"] if last_info is not None else {},
            }
        )

        if (episode + 1) % 10 == 0 or episode == 0:
            print(
                f"第 {episode + 1:>3} 轮 / {args.ppo_episodes} 轮 | "
                f"累计奖励 {episode_reward:.4f} | "
                f"最近10轮平均奖励 {np.mean(episode_rewards[-10:]):.4f}"
            )

    plot_curve(
        episode_rewards,
        title="PPO 训练奖励曲线",
        ylabel="奖励值",
        output_path=output_dir / "ppo_training_curve.png",
    )
    torch.save(agent.network.state_dict(), output_dir / "ppo_agent.pth")
    save_json(
        {
            "reward_summary": summarize_rewards(episode_rewards),
            "episodes": episode_logs,
        },
        output_dir / "ppo_training_summary.json",
    )
    return agent, episode_rewards, episode_logs


def train_sac(
    args: argparse.Namespace,
    env_factory: Callable[[], IslandGenerationEnv],
    output_dir: Path,
    device: torch.device,
) -> Tuple[SACAgent, List[float]]:
    print_section("补充阶段：SAC 训练")
    env = env_factory()
    agent = SACAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        action_range=1.0,
    ).to(device)
    replay_buffer = ReplayBuffer(capacity=10000)

    episode_rewards: List[float] = []
    for episode in range(args.sac_episodes):
        state, _ = env.reset(seed=args.seed + 1000 + episode)
        episode_reward = 0.0

        for _ in range(env.max_steps):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            replay_buffer.push(state, action, reward, next_state, done)
            if len(replay_buffer) >= max(256, args.batch_size * 4):
                agent.update(replay_buffer, batch_size=args.batch_size)

            episode_reward += reward
            state = next_state
            if done:
                break

        episode_rewards.append(float(episode_reward))
        if (episode + 1) % 10 == 0 or episode == 0:
            print(
                f"第 {episode + 1:>3} 轮 / {args.sac_episodes} 轮 | "
                f"累计奖励 {episode_reward:.4f} | "
                f"最近10轮平均奖励 {np.mean(episode_rewards[-10:]):.4f}"
            )

    plot_curve(
        episode_rewards,
        title="SAC 训练奖励曲线",
        ylabel="奖励值",
        output_path=output_dir / "sac_training_curve.png",
    )
    agent.save(output_dir / "sac_agent.pth")
    save_json(
        {"reward_summary": summarize_rewards(episode_rewards)},
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
    metrics_list: List[Dict[str, float]] = []
    score_list: List[Dict[str, float]] = []
    rewards: List[float] = []
    heightmaps: List[np.ndarray] = []
    records: List[Dict[str, object]] = []

    for index in range(num_islands):
        env = env_factory()
        state, _ = env.reset(seed=seed_offset + index)
        episode_reward = 0.0
        info = {}
        for _ in range(env.max_steps):
            action = select_eval_action(agent, state)
            state, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            if terminated or truncated:
                break

        rewards.append(float(episode_reward))
        metrics_list.append(dict(info["metrics"]))
        score_list.append(dict(info["score"]))
        heightmaps.append(info["heightmap"])
        record = {
            "index": index + 1,
            "reward": float(episode_reward),
            "metrics": {key: float(value) for key, value in info["metrics"].items()},
            "score": {key: float(value) for key, value in info["score"].items()},
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
    set_seed(args.seed)
    configure_matplotlib_chinese()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_section("实验启动")
    print(f"当前设备            : {device}")
    print(f"输出目录            : {output_dir.resolve()}")
    print(f"每轮原始样本数      : {args.dataset_samples}")
    print(f"目标清洗样本数      : {args.min_clean_samples}")
    print(f"最大原始样本数      : {args.max_dataset_samples}")
    print(f"采样策略            : {args.sampling_profile}")
    print(f"VAE 训练轮数        : {args.vae_epochs}")
    print(f"PPO 训练轮数        : {args.ppo_episodes}")
    print(f"SAC 训练轮数        : {args.sac_episodes}")

    builder, clean_samples, arrays, clean_summary = build_dataset(args, output_dir)
    vae, latents, _ = train_formal_vae(args, arrays, output_dir, device)

    print_section("第三阶段：特征归一化拟合")
    metric_names = builder.evaluator.metric_names
    feature_normalizer = builder.fit_feature_normalizer(clean_samples, latent_matrix=latents)
    print(f"状态特征维度        : {len(metric_names) + latents.shape[1]}")
    print(f"结构指标维度        : {len(metric_names)}")
    print(f"latent 维度         : {latents.shape[1]}")

    vae_summary = evaluate_vae_representation(args, builder, arrays, vae, latents, output_dir, device)

    env_factory = build_env_factory(args, vae, feature_normalizer)
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
        sac_agent, _ = train_sac(args, env_factory, output_dir, device)
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
    print("- vae_representation_summary.json / vae_metric_mae.png / vae_latent_space.png")
    print("- zero/random/ppo_evaluation_summary.json")
    print("- policy_comparison.json / final_summary.json")
    if args.sac_episodes > 0:
        print("- sac_training_curve.png / sac_evaluation_summary.json")


if __name__ == "__main__":
    main()
