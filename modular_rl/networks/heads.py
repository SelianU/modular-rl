import torch
import torch.nn as nn
from typing import Tuple, Optional


# ══════════════════════════════════════════════════════════════════════════════
# Value-based heads (DQN family)
# ══════════════════════════════════════════════════════════════════════════════

class QHead(nn.Module):
    """Discrete action Q-value head. Maps features → Q(s, a) for each action."""

    def __init__(self, feature_dim: int, action_dim: int):
        super().__init__()
        self.fc = nn.Linear(feature_dim, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class DuelingQHead(nn.Module):
    """
    Dueling Network head.
    Decomposes Q(s,a) = V(s) + A(s,a) − mean_a'[A(s,a')]
    which improves value estimation by learning state value and action
    advantages separately. Drop-in replacement for QHead.
    Works for any input shape (..., feature_dim).
    """

    def __init__(self, feature_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.value_stream = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value = self.value_stream(x)                         # (..., 1)
        advantage = self.advantage_stream(x)                  # (..., action_dim)
        return value + advantage - advantage.mean(dim=-1, keepdim=True)


# ══════════════════════════════════════════════════════════════════════════════
# Continuous actor heads (SAC / PPO / TD3)
# ══════════════════════════════════════════════════════════════════════════════

class GaussianPolicyHead(nn.Module):
    """
    Squashed-Gaussian stochastic policy head (SAC).
    Outputs action samples and their log-probabilities via the reparameterization
    trick with tanh squashing to respect action bounds.
    """

    def __init__(
        self,
        feature_dim: int,
        action_dim: int,
        action_limit_low: float = -1.0,
        action_limit_high: float = 1.0,
    ):
        super().__init__()
        self.mu = nn.Linear(feature_dim, action_dim)
        self.log_std = nn.Linear(feature_dim, action_dim)
        self.action_limit_low = action_limit_low
        self.action_limit_high = action_limit_high
        self.scale = (action_limit_high - action_limit_low) / 2.0
        self.bias = (action_limit_high + action_limit_low) / 2.0

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu = self.mu(x)
        log_std = self.log_std(x).clamp(-20, 2)
        return mu, log_std

    def sample(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu, log_std = self.forward(x)
        std = log_std.exp()
        dist = torch.distributions.Normal(mu, std)
        x_t = dist.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.scale + self.bias
        log_prob = dist.log_prob(x_t) - torch.log(self.scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob

    def get_mean_action(self, x: torch.Tensor) -> torch.Tensor:
        mu, _ = self.forward(x)
        return torch.tanh(mu) * self.scale + self.bias


class DiagonalGaussianHead(nn.Module):
    """
    Diagonal-Gaussian policy head for PPO (no tanh squashing).
    Uses a state-independent learnable log_std vector.
    The sampled action is the raw Gaussian sample; the environment is
    responsible for clipping to its action bounds.
    """

    def __init__(
        self,
        feature_dim: int,
        action_dim: int,
        log_std_init: float = 0.0,
    ):
        super().__init__()
        self.mu = nn.Linear(feature_dim, action_dim)
        self.log_std = nn.Parameter(torch.full((action_dim,), log_std_init))

    def _dist(self, features: torch.Tensor) -> torch.distributions.Normal:
        mu = self.mu(features)
        std = self.log_std.clamp(-4, 2).exp().expand_as(mu)
        return torch.distributions.Normal(mu, std)

    def get_action_and_logprob(
        self, features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist = self._dist(features)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)    # scalar per sample
        entropy = dist.entropy().sum(dim=-1)
        return action, log_prob, entropy

    def evaluate_actions(
        self, features: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        dist = self._dist(features)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy

    def get_mean_action(self, features: torch.Tensor) -> torch.Tensor:
        return self.mu(features)


class DeterministicPolicyHead(nn.Module):
    """
    Deterministic continuous action head (TD3 / DDPG).
    Outputs a scaled tanh action: action = tanh(fc(x)) * scale + bias.
    """

    def __init__(
        self,
        feature_dim: int,
        action_dim: int,
        action_limit_low: float = -1.0,
        action_limit_high: float = 1.0,
    ):
        super().__init__()
        self.fc = nn.Linear(feature_dim, action_dim)
        self.scale = (action_limit_high - action_limit_low) / 2.0
        self.bias = (action_limit_high + action_limit_low) / 2.0
        self.action_limit_low = action_limit_low
        self.action_limit_high = action_limit_high

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.fc(x)) * self.scale + self.bias


# ══════════════════════════════════════════════════════════════════════════════
# Discrete actor head (PPO)
# ══════════════════════════════════════════════════════════════════════════════

class CategoricalPolicyHead(nn.Module):
    """
    Categorical policy head for discrete-action PPO.
    forward() returns a Categorical distribution.
    """

    def __init__(self, feature_dim: int, action_dim: int):
        super().__init__()
        self.fc = nn.Linear(feature_dim, action_dim)

    def forward(self, x: torch.Tensor) -> torch.distributions.Categorical:
        return torch.distributions.Categorical(logits=self.fc(x))

    def get_action_and_logprob(
        self, features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist = self.forward(features)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy

    def evaluate_actions(
        self, features: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        dist = self.forward(features)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_prob, entropy


# ══════════════════════════════════════════════════════════════════════════════
# Critic heads
# ══════════════════════════════════════════════════════════════════════════════

class DoubleQCriticHead(nn.Module):
    """
    Twin Q-critic head (SAC / TD3).
    Outputs two independent Q-values to mitigate overestimation bias.
    """

    def __init__(self, feature_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.q1 = nn.Sequential(
            nn.Linear(feature_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(feature_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self, features: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([features, actions], dim=-1)
        return self.q1(x), self.q2(x)


class ValueHead(nn.Module):
    """State-value head V(s) — used by PPO critic."""

    def __init__(self, feature_dim: int):
        super().__init__()
        self.fc = nn.Linear(feature_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)
