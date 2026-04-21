"""
PPO baseline with device-safe updates for formal experiments.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal


class PPONetwork(nn.Module):
    """Actor-critic network used by PPO."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256, action_range: float = 1.0):
        super().__init__()
        self.action_range = float(action_range)

        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.actor_logstd = nn.Linear(hidden_dim, action_dim)

        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        actor_features = self.actor(state)
        mean = self.actor_mean(actor_features)
        logstd = torch.clamp(self.actor_logstd(actor_features), -20.0, 2.0)
        value = self.critic(state)
        return mean, logstd, value

    def sample_action(self, state: torch.Tensor, epsilon: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor]:
        mean, logstd, _ = self.forward(state)
        std = logstd.exp()

        distribution = Normal(mean, std)
        z = distribution.rsample()
        squashed = torch.tanh(z)
        action = squashed * self.action_range

        log_prob = distribution.log_prob(z)
        log_prob -= torch.log(1 - squashed.pow(2) + epsilon)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob

    def evaluate(self, state: torch.Tensor, action: torch.Tensor, epsilon: float = 1e-6):
        mean, logstd, value = self.forward(state)
        std = logstd.exp()
        distribution = Normal(mean, std)

        normalized_action = torch.clamp(action / self.action_range, -0.999999, 0.999999)
        z = torch.atanh(normalized_action)

        log_prob = distribution.log_prob(z)
        log_prob -= torch.log(1 - normalized_action.pow(2) + epsilon)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        entropy = distribution.entropy().sum(dim=-1, keepdim=True)
        return log_prob, value, entropy


class PPOAgent:
    """PPO agent for normalized continuous action spaces."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        lam: float = 0.95,
        clip_epsilon: float = 0.2,
        epoch: int = 10,
        batch_size: int = 64,
        action_range: float = 1.0,
    ):
        self.gamma = gamma
        self.lam = lam
        self.clip_epsilon = clip_epsilon
        self.epoch = epoch
        self.batch_size = batch_size
        self.network = PPONetwork(state_dim, action_dim, hidden_dim, action_range)
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)

    def to(self, device: str | torch.device):
        self.network = self.network.to(device)
        return self

    @property
    def device(self) -> torch.device:
        return next(self.network.parameters()).device

    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        action, _ = self.get_action_and_log_prob(state, deterministic=deterministic)
        return action

    def get_action_and_log_prob(self, state: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, float]:
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            if deterministic:
                mean, _, _ = self.network(state_tensor)
                action = torch.tanh(mean) * self.network.action_range
                log_prob = torch.zeros((1, 1), device=self.device)
            else:
                action, log_prob = self.network.sample_action(state_tensor)
        return action.squeeze(0).cpu().numpy(), float(log_prob.item())

    def compute_gae(
        self,
        rewards: Sequence[float],
        values: Sequence[float],
        dones: Sequence[float],
        next_value: float,
    ) -> Tuple[List[float], List[float]]:
        advantages: List[float] = []
        gae = 0.0
        values_with_bootstrap = list(values) + [next_value]

        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values_with_bootstrap[t + 1] * (1 - dones[t]) - values_with_bootstrap[t]
            gae = delta + self.gamma * self.lam * (1 - dones[t]) * gae
            advantages.insert(0, gae)

        returns = [adv + val for adv, val in zip(advantages, values_with_bootstrap[:-1])]
        return advantages, returns

    def update(self, memory: Sequence[Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool, float]]):
        if len(memory) == 0:
            return {"total_loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0}

        states, actions, rewards, next_states, dones, old_log_probs = zip(*memory)

        states_tensor = torch.as_tensor(np.asarray(states), dtype=torch.float32, device=self.device)
        actions_tensor = torch.as_tensor(np.asarray(actions), dtype=torch.float32, device=self.device)
        old_log_probs_tensor = torch.as_tensor(np.asarray(old_log_probs), dtype=torch.float32, device=self.device).unsqueeze(1)

        with torch.no_grad():
            _, _, values_tensor = self.network(states_tensor)
            last_next_state = torch.as_tensor(next_states[-1], dtype=torch.float32, device=self.device).unsqueeze(0)
            _, _, next_value_tensor = self.network(last_next_state)

        values = values_tensor.squeeze(1).cpu().numpy().tolist()
        advantages, returns = self.compute_gae(
            rewards=list(rewards),
            values=values,
            dones=[float(done) for done in dones],
            next_value=float(next_value_tensor.item()),
        )

        advantages_tensor = torch.as_tensor(np.asarray(advantages), dtype=torch.float32, device=self.device).unsqueeze(1)
        returns_tensor = torch.as_tensor(np.asarray(returns), dtype=torch.float32, device=self.device).unsqueeze(1)
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)

        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        num_updates = 0

        for _ in range(self.epoch):
            indices = torch.randperm(states_tensor.size(0), device=self.device)
            for start in range(0, states_tensor.size(0), self.batch_size):
                batch_indices = indices[start : start + self.batch_size]
                batch_states = states_tensor[batch_indices]
                batch_actions = actions_tensor[batch_indices]
                batch_old_log_probs = old_log_probs_tensor[batch_indices]
                batch_advantages = advantages_tensor[batch_indices]
                batch_returns = returns_tensor[batch_indices]

                new_log_probs, values_pred, entropy = self.network.evaluate(batch_states, batch_actions)
                ratio = torch.exp(new_log_probs - batch_old_log_probs)

                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = nn.MSELoss()(values_pred, batch_returns)
                entropy_bonus = -0.01 * entropy.mean()

                loss = policy_loss + 0.5 * value_loss + entropy_bonus
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
                self.optimizer.step()

                total_loss += float(loss.item())
                total_policy_loss += float(policy_loss.item())
                total_value_loss += float(value_loss.item())
                num_updates += 1

        denom = max(num_updates, 1)
        return {
            "total_loss": total_loss / denom,
            "policy_loss": total_policy_loss / denom,
            "value_loss": total_value_loss / denom,
        }

    def save(self, path: str) -> None:
        torch.save(self.network.state_dict(), path)

    def load(self, path: str) -> None:
        self.network.load_state_dict(torch.load(path, map_location=self.device))
