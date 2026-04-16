"""
PPO (Proximal Policy Optimization) 基线模型
用于与SAC算法对比
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
from collections import deque
import random


class PPONetwork(nn.Module):
    """PPO网络 - Actor-Critic架构"""
    
    def __init__(self, state_dim, action_dim, hidden_dim=256, action_range=0.1):
        super(PPONetwork, self).__init__()
        
        self.action_range = action_range
        
        # Actor网络
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.actor_logstd = nn.Linear(hidden_dim, action_dim)
        
        # Critic网络
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state):
        """前向传播"""
        # Actor
        actor_features = self.actor(state)
        mean = self.actor_mean(actor_features)
        logstd = self.actor_logstd(actor_features)
        logstd = torch.clamp(logstd, -20, 2)
        
        # Critic
        value = self.critic(state)
        
        return mean, logstd, value
    
    def sample_action(self, state, epsilon=1e-6):
        """采样动作"""
        mean, logstd, _ = self.forward(state)
        std = logstd.exp()
        
        normal = Normal(mean, std)
        z = normal.rsample()
        action = torch.tanh(z) * self.action_range
        
        log_prob = normal.log_prob(z)
        log_prob -= torch.log(1 - action.pow(2) / (self.action_range**2) + epsilon)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        return action, log_prob
    
    def evaluate(self, state, action, epsilon=1e-6):
        """评估动作"""
        mean, logstd, value = self.forward(state)
        std = logstd.exp()
        
        normal = Normal(mean, std)
        z = torch.atanh(action / self.action_range)
        log_prob = normal.log_prob(z)
        log_prob -= torch.log(1 - action.pow(2) / (self.action_range**2) + epsilon)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        entropy = normal.entropy().sum(dim=-1, keepdim=True)
        
        return log_prob, value, entropy


class PPOAgent:
    """PPO智能体"""
    
    def __init__(self, state_dim, action_dim, hidden_dim=256,
                 learning_rate=3e-4, gamma=0.99, lam=0.95,
                 clip_epsilon=0.2, epoch=10, batch_size=64,
                 action_range=0.1):
        """
        初始化PPO智能体
        
        参数:
            state_dim: 状态维度
            action_dim: 动作维度
            hidden_dim: 隐藏层维度
            learning_rate: 学习率
            gamma: 折扣因子
            lam: GAE参数
            clip_epsilon: PPO裁剪参数
            epoch: 每次更新的epoch数
            batch_size: 批大小
            action_range: 动作范围
        """
        self.gamma = gamma
        self.lam = lam
        self.clip_epsilon = clip_epsilon
        self.epoch = epoch
        self.batch_size = batch_size
        
        self.network = PPONetwork(state_dim, action_dim, hidden_dim, action_range)
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
    
    def select_action(self, state, deterministic=False):
        """选择动作"""
        state = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            if deterministic:
                mean, _, _ = self.network(state)
                action = torch.tanh(mean) * self.network.action_range
            else:
                action, _ = self.network.sample_action(state)
        
        return action.detach().numpy()[0]
    
    def compute_gae(self, rewards, values, dones, next_value):
        """计算Generalized Advantage Estimation"""
        advantages = []
        gae = 0
        
        values = values + [next_value]
        
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.lam * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        
        returns = [adv + val for adv, val in zip(advantages, values[:-1])]
        
        return advantages, returns
    
    def update(self, memory):
        """
        更新网络
        
        参数:
            memory: 经验列表 [(state, action, reward, next_state, done, log_prob_old)]
        
        返回:
            损失字典
        """
        states, actions, rewards, next_states, dones, old_log_probs = zip(*memory)
        
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        dones = torch.FloatTensor(dones).unsqueeze(1)
        old_log_probs = torch.FloatTensor(old_log_probs)
        
        # 计算values
        with torch.no_grad():
            _, _, values = self.network(states)
            _, _, next_value = self.network(torch.FloatTensor(next_states[-1]).unsqueeze(0))
        
        # 计算优势
        advantages, returns = self.compute_gae(
            rewards.numpy().flatten().tolist(),
            values.squeeze().numpy().tolist(),
            dones.numpy().flatten().tolist(),
            next_value.squeeze().item()
        )
        
        advantages = torch.FloatTensor(advantages).unsqueeze(1)
        returns = torch.FloatTensor(returns).unsqueeze(1)
        
        # 标准化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO更新
        total_loss = 0
        total_policy_loss = 0
        total_value_loss = 0
        
        for _ in range(self.epoch):
            # 随机打乱
            indices = torch.randperm(len(states))
            
            for start in range(0, len(states), self.batch_size):
                end = start + self.batch_size
                idx = indices[start:end]
                
                batch_states = states[idx]
                batch_actions = actions[idx]
                batch_old_log_probs = old_log_probs[idx]
                batch_advantages = advantages[idx]
                batch_returns = returns[idx]
                
                # 计算新log prob和value
                new_log_probs, values, entropy = self.network.evaluate(batch_states, batch_actions)
                
                # 比率
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                
                # 策略损失（裁剪）
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # 价值损失
                value_loss = nn.MSELoss()(values, batch_returns)
                
                # 熵奖励
                entropy_loss = -0.01 * entropy.mean()
                
                # 总损失
                loss = policy_loss + 0.5 * value_loss + entropy_loss
                
                # 优化
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
                self.optimizer.step()
                
                total_loss += loss.item()
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
        
        n_updates = self.epoch * max(1, len(states) // self.batch_size)
        
        return {
            'total_loss': total_loss / n_updates,
            'policy_loss': total_policy_loss / n_updates,
            'value_loss': total_value_loss / n_updates
        }
    
    def save(self, path):
        """保存模型"""
        torch.save(self.network.state_dict(), path)
    
    def load(self, path):
        """加载模型"""
        self.network.load_state_dict(torch.load(path))


# 测试代码
if __name__ == "__main__":
    from rl_environment import IslandGenerationEnv
    
    # 创建环境
    env = IslandGenerationEnv(map_size=64, max_steps=30)
    
    # 创建PPO智能体
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    agent = PPOAgent(state_dim, action_dim, hidden_dim=256)
    
    print(f"状态维度: {state_dim}")
    print(f"动作维度: {action_dim}")
    
    # 测试动作选择
    state, _ = env.reset(seed=42)
    action = agent.select_action(state)
    print(f"\n测试动作: {action}")
    print(f"动作范围: [{action.min():.4f}, {action.max():.4f}]")
    
    # 简单训练测试
    print("\n开始简单训练测试...")
    memory = []
    
    for step in range(100):
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # 获取旧log prob
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_tensor = torch.FloatTensor(action).unsqueeze(0)
        with torch.no_grad():
            log_prob, _, _ = agent.network.evaluate(state_tensor, action_tensor)
        
        memory.append((state, action, reward, next_state, done, log_prob.item()))
        
        state = next_state
        
        if done:
            state, _ = env.reset()
    
    # 更新
    losses = agent.update(memory)
    print(f"\n更新损失:")
    print(f"  总损失: {losses['total_loss']:.4f}")
    print(f"  策略损失: {losses['policy_loss']:.4f}")
    print(f"  价值损失: {losses['value_loss']:.4f}")
    
    print("\n✅ PPO Agent测试成功！")
