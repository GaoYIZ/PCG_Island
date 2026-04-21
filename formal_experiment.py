"""
Formal experiment runner for IslandTest.

Pipeline:
1. 数据集构建 / 清洗 / 评估
2. VAE 训练与 latent 提取
3. 特征归一化拟合
4. PPO 正式训练
5. 可选 SAC 补充训练
6. 结果汇总与中文报告输出
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset_pipeline import IslandDatasetBuilder
from feature_processing import IslandFeatureNormalizer
from ppo_baseline import PPOAgent
from reporting import (
    print_dataset_summary,
    print_metric_dict,
    print_section,
    save_json,
    summarize_rewards,
)
from rl_environment import IslandGenerationEnv
from sac_agent import ReplayBuffer, SACAgent
from vae_model import BetaVAE, HeightmapDataset, encode_heightmaps, train_vae


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="IslandTest 正式实验运行脚本")
    parser.add_argument("--output-dir", type=str, default="formal_outputs", help="结果输出目录")
    parser.add_argument("--map-size", type=int, default=64, help="地图尺寸")
    parser.add_argument("--dataset-samples", type=int, default=300, help="原始生成样本数")
    parser.add_argument("--batch-size", type=int, default=32, help="VAE 和 RL 的批大小")
    parser.add_argument("--latent-dim", type=int, default=16, help="VAE latent 维度")
    parser.add_argument("--vae-epochs", type=int, default=20, help="VAE 训练轮数")
    parser.add_argument("--vae-beta", type=float, default=4.0, help="Beta-VAE 的 beta")
    parser.add_argument("--vae-lr", type=float, default=1e-3, help="VAE 学习率")
    parser.add_argument("--ppo-episodes", type=int, default=60, help="PPO 训练轮数")
    parser.add_argument("--ppo-max-steps", type=int, default=30, help="PPO 单轮最大步数")
    parser.add_argument("--ppo-hidden-dim", type=int, default=256, help="PPO 隐层维度")
    parser.add_argument("--sac-episodes", type=int, default=0, help="SAC 训练轮数，0 表示跳过")
    parser.add_argument("--eval-islands", type=int, default=12, help="最终评估地图数量")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def plot_curve(values: Sequence[float], title: str, ylabel: str, output_path: Path) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(values, linewidth=2, alpha=0.8)
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
    num_samples = min(num_samples, len(heightmaps))
    cols = 3
    rows = int(np.ceil(num_samples / cols))
    plt.figure(figsize=(12, 4 * rows))
    for idx in range(num_samples):
        plt.subplot(rows, cols, idx + 1)
        plt.imshow(heightmaps[idx], cmap="terrain")
        plt.title(f"Sample {idx + 1}")
        plt.axis("off")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_reconstruction(originals: np.ndarray, reconstructions: np.ndarray, output_path: Path) -> None:
    num_samples = min(6, len(originals))
    fig, axes = plt.subplots(2, num_samples, figsize=(3 * num_samples, 6))
    for idx in range(num_samples):
        axes[0, idx].imshow(originals[idx], cmap="terrain")
        axes[0, idx].set_title(f"Original {idx + 1}")
        axes[0, idx].axis("off")
        axes[1, idx].imshow(reconstructions[idx], cmap="terrain")
        axes[1, idx].set_title(f"Recon {idx + 1}")
        axes[1, idx].axis("off")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def build_dataset(
    args: argparse.Namespace,
    output_dir: Path,
) -> Tuple[IslandDatasetBuilder, List, Dict[str, np.ndarray], Dict[str, object]]:
    print_section("第一阶段：数据集构建 / 清洗 / 评估")
    builder = IslandDatasetBuilder(map_size=args.map_size)
    raw_samples = builder.generate_samples(n_samples=args.dataset_samples, seed=args.seed)
    clean_samples = builder.clean_samples(raw_samples)
    raw_summary = builder.evaluate_dataset(raw_samples)
    clean_summary = builder.evaluate_dataset(clean_samples)

    print("原始数据集统计:")
    print_dataset_summary(raw_summary)
    print("\n清洗后数据集统计:")
    print_dataset_summary(clean_summary)

    arrays = builder.build_training_arrays(clean_samples)
    plot_dataset_samples(arrays["heightmaps"], output_dir / "dataset_samples.png")

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
    dataset = HeightmapDataset(arrays["heightmaps"])
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    vae = BetaVAE(map_size=args.map_size, latent_dim=args.latent_dim, beta=args.vae_beta).to(device)
    start_time = time.time()
    history = train_vae(
        vae,
        dataloader,
        epochs=args.vae_epochs,
        learning_rate=args.vae_lr,
        device=str(device),
    )
    duration = time.time() - start_time
    print(f"VAE 训练完成，用时 {duration / 60:.2f} 分钟")

    latents = encode_heightmaps(vae, arrays["heightmaps"], batch_size=args.batch_size, device=str(device))
    print(f"latent 矩阵形状: {latents.shape}")
    print(f"latent 均值: {latents.mean():.4f}")
    print(f"latent 标准差: {latents.std():.4f}")

    plot_curve(
        [entry["total_loss"] for entry in history],
        title="VAE Training Loss",
        ylabel="Loss",
        output_path=output_dir / "vae_training_curve.png",
    )

    vae.eval()
    with torch.no_grad():
        originals = torch.as_tensor(arrays["heightmaps"][:6], dtype=torch.float32, device=device).unsqueeze(1)
        reconstructions, _, _ = vae(originals)
    plot_reconstruction(
        arrays["heightmaps"][:6],
        reconstructions.squeeze(1).cpu().numpy(),
        output_dir / "vae_reconstruction.png",
    )

    save_json(
        {
            "latent_shape": list(latents.shape),
            "latent_mean": float(latents.mean()),
            "latent_std": float(latents.std()),
            "vae_history": history,
        },
        output_dir / "vae_summary.json",
    )
    return vae, latents, history


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
        )

    return factory


def train_ppo(
    args: argparse.Namespace,
    env_factory: Callable[[], IslandGenerationEnv],
    output_dir: Path,
    device: torch.device,
) -> Tuple[PPOAgent, List[float], List[dict]]:
    print_section("第三阶段：PPO 正式训练")
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
        title="PPO Training Reward",
        ylabel="Reward",
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
    print_section("第四阶段：SAC 补充训练")
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
                f"SAC 第 {episode + 1:>3} 轮 / {args.sac_episodes} 轮 | "
                f"累计奖励 {episode_reward:.4f} | "
                f"最近10轮平均奖励 {np.mean(episode_rewards[-10:]):.4f}"
            )

    plot_curve(
        episode_rewards,
        title="SAC Training Reward",
        ylabel="Reward",
        output_path=output_dir / "sac_training_curve.png",
    )
    agent.save(output_dir / "sac_agent.pth")
    save_json(
        {"reward_summary": summarize_rewards(episode_rewards)},
        output_dir / "sac_training_summary.json",
    )
    return agent, episode_rewards


def evaluate_agent(
    name: str,
    agent,
    env_factory: Callable[[], IslandGenerationEnv],
    output_dir: Path,
    num_islands: int,
    seed_offset: int,
) -> Dict[str, object]:
    print_section(f"第五阶段：{name} 最终评估")
    metrics_list: List[Dict[str, float]] = []
    score_list: List[Dict[str, float]] = []
    rewards: List[float] = []
    heightmaps: List[np.ndarray] = []

    for index in range(num_islands):
        env = env_factory()
        state, _ = env.reset(seed=seed_offset + index)
        episode_reward = 0.0
        info = {}
        for _ in range(env.max_steps):
            if hasattr(agent, "select_action"):
                if name.upper() == "PPO":
                    action = agent.select_action(state, deterministic=True)
                else:
                    action = agent.select_action(state, evaluate=True)
            else:
                raise ValueError("Agent does not expose select_action.")
            state, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            if terminated or truncated:
                break

        rewards.append(float(episode_reward))
        metrics_list.append(dict(info["metrics"]))
        score_list.append(dict(info["score"]))
        heightmaps.append(info["heightmap"])

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

    plot_dataset_samples(np.asarray(heightmaps), output_dir / f"{name.lower()}_generated_islands.png", num_samples=9)
    save_json(summary, output_dir / f"{name.lower()}_evaluation_summary.json")
    return summary


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_section("实验启动")
    print(f"当前设备            : {device}")
    print(f"输出目录            : {output_dir.resolve()}")
    print(f"原始样本数          : {args.dataset_samples}")
    print(f"VAE 训练轮数        : {args.vae_epochs}")
    print(f"PPO 训练轮数        : {args.ppo_episodes}")
    print(f"SAC 训练轮数        : {args.sac_episodes}")

    builder, clean_samples, arrays, clean_summary = build_dataset(args, output_dir)
    vae, latents, _ = train_formal_vae(args, arrays, output_dir, device)

    print_section("特征归一化拟合")
    metric_names = builder.evaluator.metric_names
    feature_normalizer = builder.fit_feature_normalizer(clean_samples, latent_matrix=latents)
    print(f"状态特征维度        : {len(metric_names) + latents.shape[1]}")
    print(f"结构指标维度        : {len(metric_names)}")
    print(f"latent 维度         : {latents.shape[1]}")

    env_factory = build_env_factory(args, vae, feature_normalizer)
    ppo_agent, _, _ = train_ppo(args, env_factory, output_dir, device)
    ppo_summary = evaluate_agent("PPO", ppo_agent, env_factory, output_dir, args.eval_islands, args.seed + 2000)

    final_summary: Dict[str, object] = {
        "清洗后有效样本数": clean_summary["num_valid"],
        "VAE latent 维度": latents.shape[1],
        "PPO 评估总结": ppo_summary,
    }

    if args.sac_episodes > 0:
        sac_agent, _ = train_sac(args, env_factory, output_dir, device)
        sac_summary = evaluate_agent("SAC", sac_agent, env_factory, output_dir, args.eval_islands, args.seed + 4000)
        final_summary["SAC 评估总结"] = sac_summary

    save_json(final_summary, output_dir / "final_summary.json")
    print_section("实验完成")
    print(f"结果文件已保存到    : {output_dir.resolve()}")
    print("核心输出包括        :")
    print("- dataset_summary_raw.json / dataset_summary_clean.json")
    print("- dataset_samples.png")
    print("- vae_training_curve.png / vae_reconstruction.png")
    print("- ppo_training_curve.png / ppo_evaluation_summary.json")
    if args.sac_episodes > 0:
        print("- sac_training_curve.png / sac_evaluation_summary.json")


if __name__ == "__main__":
    main()
