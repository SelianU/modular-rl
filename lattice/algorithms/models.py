import torch
import torch.nn as nn
from typing import Optional, Tuple, Union

from lattice.networks import (
    QHead, GaussianPolicyHead, DoubleQCriticHead,
    CategoricalPolicyHead, DiagonalGaussianHead, DeterministicPolicyHead, ValueHead,
)
from lattice.networks.encoders import RNN, Transformer


# ══════════════════════════════════════════════════════════════════════════════
# DQN
# ══════════════════════════════════════════════════════════════════════════════

class QNetwork(nn.Module):
    """
    Assembles a Backbone and a Q-value head (QHead or DuelingQHead).
    Handles RNN (returns hidden state) and Transformer (returns None placeholder)
    transparently so DQNAgent can use is_recurrent=True for both.
    """

    def __init__(self, backbone: nn.Module, head: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(
        self,
        x: torch.Tensor,
        hx: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[Tuple]]]:
        if isinstance(self.backbone, RNN):
            features, next_hx = self.backbone(x, hx)
            return self.head(features), next_hx
        elif isinstance(self.backbone, Transformer):
            features = self.backbone(x)
            return self.head(features), None
        else:
            features = self.backbone(x)
            return self.head(features)


# ══════════════════════════════════════════════════════════════════════════════
# SAC
# ══════════════════════════════════════════════════════════════════════════════

class SACActor(nn.Module):
    """Backbone + GaussianPolicyHead stochastic policy (SAC)."""

    def __init__(self, backbone: nn.Module, head: GaussianPolicyHead):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.head(self.backbone(x))

    def sample(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.head.sample(self.backbone(x))

    def get_mean_action(self, x: torch.Tensor) -> torch.Tensor:
        return self.head.get_mean_action(self.backbone(x))


class SACCritic(nn.Module):
    """Backbone + DoubleQCriticHead twin Q-critic (SAC / TD3)."""

    def __init__(self, backbone: nn.Module, head: DoubleQCriticHead):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.head(self.backbone(state), action)


# ══════════════════════════════════════════════════════════════════════════════
# PPO
# ══════════════════════════════════════════════════════════════════════════════

class PPOActor(nn.Module):
    """
    Backbone + policy head for PPO.
    Accepts CategoricalPolicyHead (discrete) or DiagonalGaussianHead (continuous).
    Both heads expose get_action_and_logprob() and evaluate_actions().
    """

    def __init__(self, backbone: nn.Module, head: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def get_action_and_logprob(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (action, log_prob, entropy) — used during rollout collection."""
        return self.head.get_action_and_logprob(self.backbone(x))

    def evaluate_actions(
        self, x: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (log_prob, entropy) — used during PPO update."""
        return self.head.evaluate_actions(self.backbone(x), actions)

    def get_mean_action(self, x: torch.Tensor) -> torch.Tensor:
        """Deterministic action for evaluation."""
        features = self.backbone(x)
        if isinstance(self.head, CategoricalPolicyHead):
            return self.head(features).probs.argmax(dim=-1)
        return self.head.get_mean_action(features)


class PPOCritic(nn.Module):
    """Backbone + ValueHead state-value function V(s) (PPO)."""

    def __init__(self, backbone: nn.Module, head: ValueHead):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


# ══════════════════════════════════════════════════════════════════════════════
# TD3
# ══════════════════════════════════════════════════════════════════════════════

class TD3Actor(nn.Module):
    """Backbone + DeterministicPolicyHead (TD3 / DDPG)."""

    def __init__(self, backbone: nn.Module, head: DeterministicPolicyHead):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


# TD3 reuses SACCritic (identical twin-Q architecture)
TD3Critic = SACCritic
