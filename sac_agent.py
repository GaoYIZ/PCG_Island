"""
SAC (Soft Actor-Critic) 强化学习算法实现
用于岛屿生成参数优化
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from collections import deque
import random


class ReplayBuffer:
    """经验回放缓冲区"""
    
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """添加经验"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """随机采样"""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))
    
    def __len__(self):
        return len(self.buffer)


class QNetwork(nn.Module):
    """Q网络 - Critic"""
    
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(QNetwork, self).__init__()
        
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state, action):
        """前向传播"""
        sa = torch.cat([state, action], dim=-1)
        q1 = self.q1(sa)
        q2 = self.q2(sa)
        return q1, q2


class PolicyNetwork(nn.Module):
    """策略网络 - Actor"""
    
    def __init__(self, state_dim, action_dim, hidden_dim=256, action_range=0.1):
        super(PolicyNetwork, self).__init__()
        
        self.action_range = action_range
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)
        
        # 限制log_std的范围
        self.LOG_STD_MIN = -20
        self.LOG_STD_MAX = 2
    
    def forward(self, state):
        """前向传播"""
        x = self.network(state)
        mean = self.mean_head(x)
        log_std = self.log_std_head(x)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        
        return mean, log_std
    
    def sample(self, state, epsilon=1e-6):
        """从策略中采样动作（带重参数化技巧）"""
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        # 采样
        normal = Normal(mean, std)
        z = normal.rsample()  # 重参数化
        
        # tanh变换
        action = torch.tanh(z)
        
        # 计算对数概率（考虑tanh变换的修正）
        log_prob = normal.log_prob(z)
        log_prob -= torch.log(1 - action.pow(2) + epsilon)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        # 缩放到实际动作范围
        action = action * self.action_range
        
        return action, log_prob


class SACAgent:
    """SAC智能体"""
    
    def __init__(self, state_dim, action_dim, hidden_dim=256, 
                 learning_rate=3e-4, gamma=0.99, tau=0.005, 
                 alpha=0.2, action_range=0.1):
        """
        初始化SAC智能体
        
        参数:
            state_dim: 状态维度
            action_dim: 动作维度
            hidden_dim: 隐藏层维度
            learning_rate: 学习率
            gamma: 折扣因子
            tau: 软更新系数
            alpha: 熵温度系数
            action_range: 动作范围
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.action_range = action_range
        
        # 创建网络
        self.q_network = QNetwork(state_dim, action_dim, hidden_dim)
        self.q_target = QNetwork(state_dim, action_dim, hidden_dim)
        self.policy = PolicyNetwork(state_dim, action_dim, hidden_dim, action_range)
        
        # 复制Q网络参数到目标网络
        self._soft_update(self.q_network, self.q_target, 1.0)
        
        # 优化器
        self.q_optimizer = torch.optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)
        
        # 自动调整alpha
        self.target_entropy = -action_dim
        self.log_alpha = torch.zeros(1, requires_grad=True)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=learning_rate)
    
    def to(self, device):
        """将模型移动到指定设备"""
        self.q_network = self.q_network.to(device)
        self.q_target = self.q_target.to(device)
        self.policy = self.policy.to(device)
        self.log_alpha = self.log_alpha.to(device)
        return self
    
    def select_action(self, state, evaluate=False):
        """选择动作"""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.q_network.parameters().__next__().device)
        
        if evaluate:
            # 评估时使用确定性策略
            mean, _ = self.policy(state)
            action = torch.tanh(mean) * self.action_range
        else:
            # 训练时采样
            action, _ = self.policy.sample(state)
        
        return action.detach().cpu().numpy()[0]
    
    def update(self, replay_buffer, batch_size=64):
        """更新网络参数"""
        if len(replay_buffer) < batch_size:
            return
        
        # 采样
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
        
        # 获取设备
        device = self.q_network.parameters().__next__().device
        
        states = torch.FloatTensor(states).to(device)
        actions = torch.FloatTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(device)
        
        # 更新Q网络
        q_loss = self._update_q(states, actions, rewards, next_states, dones)
        
        # 更新策略网络
        policy_loss = self._update_policy(states)
        
        # 更新alpha
        alpha_loss = self._update_alpha(states)
        
        return {
            'q_loss': q_loss,
            'policy_loss': policy_loss,
            'alpha_loss': alpha_loss,
            'alpha': self.alpha
        }
    
    def _update_q(self, states, actions, rewards, next_states, dones):
        """更新Q网络"""
        with torch.no_grad():
            # 下一步的动作和对数概率
            next_actions, next_log_probs = self.policy.sample(next_states)
            
            # 目标Q值
            next_q1, next_q2 = self.q_target(next_states, next_actions)
            min_next_q = torch.min(next_q1, next_q2)
            
            # Bellman备份
            target_q = rewards + (1 - dones) * self.gamma * (min_next_q - self.alpha * next_log_probs)
        
        # 当前Q值
        current_q1, current_q2 = self.q_network(states, actions)
        
        # Q损失
        q1_loss = F.mse_loss(current_q1, target_q)
        q2_loss = F.mse_loss(current_q2, target_q)
        q_loss = q1_loss + q2_loss
        
        # 优化
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()
        
        # 软更新目标网络
        self._soft_update(self.q_network, self.q_target, self.tau)
        
        return q_loss.item()
    
    def _update_policy(self, states):
        """更新策略网络"""
        # 采样动作
        actions, log_probs = self.policy.sample(states)
        
        # Q值
        q1, q2 = self.q_network(states, actions)
        min_q = torch.min(q1, q2)
        
        # 策略损失（最大化Q值 + 熵）
        policy_loss = -(min_q - self.alpha * log_probs).mean()
        
        # 优化
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        return policy_loss.item()
    
    def _update_alpha(self, states):
        """更新温度系数alpha"""
        with torch.no_grad():
            _, log_probs = self.policy.sample(states)
        
        alpha_loss = -(self.log_alpha.exp() * (log_probs + self.target_entropy).detach()).mean()
        
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        self.alpha = self.log_alpha.exp().item()
        
        return alpha_loss.item()
    
    def _soft_update(self, source, target, tau):
        """软更新目标网络"""
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(tau * source_param.data + (1 - tau) * target_param.data)
    
    def save(self, path):
        """保存模型"""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'q_target': self.q_target.state_dict(),
            'policy': self.policy.state_dict(),
            'q_optimizer': self.q_optimizer.state_dict(),
            'policy_optimizer': self.policy_optimizer.state_dict(),
            'alpha_optimizer': self.alpha_optimizer.state_dict(),
            'log_alpha': self.log_alpha
        }, path)
    
    def load(self, path):
        """加载模型"""
        checkpoint = torch.load(path)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.q_target.load_state_dict(checkpoint['q_target'])
        self.policy.load_state_dict(checkpoint['policy'])
        self.q_optimizer.load_state_dict(checkpoint['q_optimizer'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer'])
        self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer'])
        self.log_alpha = checkpoint['log_alpha']
        self.alpha = self.log_alpha.exp().item()


# 测试代码
if __name__ == "__main__":
    # 创建简单的测试环境
    state_dim = 5
    action_dim = 9
    
    agent = SACAgent(state_dim, action_dim, hidden_dim=128)
    
    # 测试动作选择
    state = np.random.randn(state_dim)
    action = agent.select_action(state)
    print(f"State shape: {state.shape}")
    print(f"Action shape: {action.shape}")
    print(f"Action: {action}")
    
    # 测试更新
    replay_buffer = ReplayBuffer(capacity=1000)
    
    # 添加一些随机经验
    for _ in range(100):
        s = np.random.randn(state_dim)
        a = np.random.randn(action_dim) * 0.1
        r = np.random.randn()
        s_next = np.random.randn(state_dim)
        d = False
        replay_buffer.push(s, a, r, s_next, d)
    
    # 更新
    losses = agent.update(replay_buffer, batch_size=32)
    print(f"\nLosses: {losses}")
    print("\n✅ SAC Agent测试成功！")
