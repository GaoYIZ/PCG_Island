"""
Soft Actor-Critic implementation for normalized island parameter optimization.
"""

from __future__ import annotations

from collections import deque
import random
from typing import Deque, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class ReplayBuffer:
    """Simple replay buffer for off-policy RL."""

    def __init__(self, capacity: int = 10_000):
        self.buffer: Deque[Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]] = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done) -> None:
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.float32),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)


class QNetwork(nn.Module):
    """Twin-critic network."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        state_action = torch.cat([state, action], dim=-1)
        return self.q1(state_action), self.q2(state_action)


class PolicyNetwork(nn.Module):
    """Gaussian policy with tanh squashing."""

    LOG_STD_MIN = -20
    LOG_STD_MAX = 2

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256, action_range: float = 1.0):
        super().__init__()
        self.action_range = float(action_range)
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden = self.network(state)
        mean = self.mean_head(hidden)
        log_std = torch.clamp(self.log_std_head(hidden), self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mean, log_std

    def sample(self, state: torch.Tensor, epsilon: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.forward(state)
        std = log_std.exp()
        distribution = Normal(mean, std)
        pre_tanh = distribution.rsample()
        squashed_action = torch.tanh(pre_tanh)

        log_prob = distribution.log_prob(pre_tanh)
        log_prob -= torch.log(1.0 - squashed_action.pow(2) + epsilon)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        action = squashed_action * self.action_range
        return action, log_prob


class SACAgent:
    """Soft Actor-Critic agent with separate actor, critic, and alpha learning rates."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        learning_rate: float = 3e-4,
        actor_learning_rate: Optional[float] = None,
        critic_learning_rate: Optional[float] = None,
        alpha_learning_rate: Optional[float] = None,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        action_range: float = 1.0,
    ):
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.action_range = float(action_range)

        self.actor_lr = float(actor_learning_rate if actor_learning_rate is not None else learning_rate)
        self.critic_lr = float(critic_learning_rate if critic_learning_rate is not None else learning_rate)
        self.alpha_lr = float(alpha_learning_rate if alpha_learning_rate is not None else learning_rate)

        self.q_network = QNetwork(state_dim, action_dim, hidden_dim)
        self.q_target = QNetwork(state_dim, action_dim, hidden_dim)
        self.policy = PolicyNetwork(state_dim, action_dim, hidden_dim, action_range)
        self._soft_update(self.q_network, self.q_target, tau=1.0)

        self.q_optimizer = torch.optim.Adam(self.q_network.parameters(), lr=self.critic_lr)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.actor_lr)

        initial_log_alpha = np.log(max(float(alpha), 1e-6))
        self.log_alpha = torch.tensor([initial_log_alpha], dtype=torch.float32, requires_grad=True)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.alpha_lr)
        self.target_entropy = -float(action_dim)
        self.alpha = float(alpha)

    def to(self, device) -> "SACAgent":
        self.q_network = self.q_network.to(device)
        self.q_target = self.q_target.to(device)
        self.policy = self.policy.to(device)
        self.log_alpha = self.log_alpha.detach().to(device).requires_grad_(True)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.alpha_lr)
        self.alpha = float(self.log_alpha.exp().item())
        return self

    def select_action(self, state, evaluate: bool = False) -> np.ndarray:
        device = next(self.q_network.parameters()).device
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

        if evaluate:
            mean, _ = self.policy(state_tensor)
            action = torch.tanh(mean) * self.action_range
        else:
            action, _ = self.policy.sample(state_tensor)

        return action.detach().cpu().numpy()[0]

    def update(self, replay_buffer: ReplayBuffer, batch_size: int = 64) -> Optional[Dict[str, float]]:
        if len(replay_buffer) < batch_size:
            return None

        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
        device = next(self.q_network.parameters()).device

        states_tensor = torch.as_tensor(states, dtype=torch.float32, device=device)
        actions_tensor = torch.as_tensor(actions, dtype=torch.float32, device=device)
        rewards_tensor = torch.as_tensor(rewards, dtype=torch.float32, device=device).unsqueeze(1)
        next_states_tensor = torch.as_tensor(next_states, dtype=torch.float32, device=device)
        dones_tensor = torch.as_tensor(dones, dtype=torch.float32, device=device).unsqueeze(1)

        q_loss, target_q_mean = self._update_q(
            states_tensor,
            actions_tensor,
            rewards_tensor,
            next_states_tensor,
            dones_tensor,
        )
        policy_loss, policy_log_prob = self._update_policy(states_tensor)
        alpha_loss = self._update_alpha(policy_log_prob)

        return {
            "q_loss": float(q_loss),
            "policy_loss": float(policy_loss),
            "alpha_loss": float(alpha_loss),
            "alpha": float(self.alpha),
            "batch_reward_mean": float(rewards_tensor.mean().item()),
            "target_q_mean": float(target_q_mean),
        }

    def _update_q(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
    ) -> Tuple[float, float]:
        with torch.no_grad():
            next_actions, next_log_probs = self.policy.sample(next_states)
            next_q1, next_q2 = self.q_target(next_states, next_actions)
            min_next_q = torch.min(next_q1, next_q2)
            alpha = self.log_alpha.exp()
            target_q = rewards + (1.0 - dones) * self.gamma * (min_next_q - alpha * next_log_probs)

        current_q1, current_q2 = self.q_network(states, actions)
        q1_loss = F.mse_loss(current_q1, target_q)
        q2_loss = F.mse_loss(current_q2, target_q)
        q_loss = q1_loss + q2_loss

        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()
        self._soft_update(self.q_network, self.q_target, self.tau)

        return q_loss.item(), target_q.mean().item()

    def _update_policy(self, states: torch.Tensor) -> Tuple[float, torch.Tensor]:
        actions, log_probs = self.policy.sample(states)
        q1, q2 = self.q_network(states, actions)
        min_q = torch.min(q1, q2)
        alpha = self.log_alpha.exp().detach()
        policy_loss = (alpha * log_probs - min_q).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        return policy_loss.item(), log_probs.detach()

    def _update_alpha(self, log_probs: torch.Tensor) -> float:
        alpha = self.log_alpha.exp()
        alpha_loss = -(alpha * (log_probs + self.target_entropy)).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self.alpha = float(self.log_alpha.exp().item())
        return alpha_loss.item()

    @staticmethod
    def _soft_update(source: nn.Module, target: nn.Module, tau: float) -> None:
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)

    def save(self, path) -> None:
        torch.save(
            {
                "q_network": self.q_network.state_dict(),
                "q_target": self.q_target.state_dict(),
                "policy": self.policy.state_dict(),
                "q_optimizer": self.q_optimizer.state_dict(),
                "policy_optimizer": self.policy_optimizer.state_dict(),
                "alpha_optimizer": self.alpha_optimizer.state_dict(),
                "log_alpha": self.log_alpha.detach().cpu(),
                "alpha": float(self.alpha),
            },
            path,
        )

    def load(self, path) -> None:
        device = next(self.q_network.parameters()).device
        checkpoint = torch.load(path, map_location=device)

        self.q_network.load_state_dict(checkpoint["q_network"])
        self.q_target.load_state_dict(checkpoint["q_target"])
        self.policy.load_state_dict(checkpoint["policy"])
        self.q_optimizer.load_state_dict(checkpoint["q_optimizer"])
        self.policy_optimizer.load_state_dict(checkpoint["policy_optimizer"])

        self.log_alpha = checkpoint["log_alpha"].to(device).detach().requires_grad_(True)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.alpha_lr)
        if "alpha_optimizer" in checkpoint:
            self.alpha_optimizer.load_state_dict(checkpoint["alpha_optimizer"])
        self.alpha = float(checkpoint.get("alpha", self.log_alpha.exp().item()))
