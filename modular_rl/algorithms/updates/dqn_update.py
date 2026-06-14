from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class DQNUpdateBatch:
    """Batch used by one standard DQN update."""

    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_states: torch.Tensor
    dones: torch.Tensor
    importance_weights: Optional[torch.Tensor] = None


@dataclass(frozen=True)
class RecurrentDQNUpdateBatch:
    """Sequence batch used by one recurrent DQN update."""

    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_states: torch.Tensor
    dones: torch.Tensor
    masks: torch.Tensor


@dataclass(frozen=True)
class DQNUpdateMetrics:
    """Metrics returned by a DQN update."""

    loss: float
    td_errors: Optional[torch.Tensor] = None

    def as_dict(self) -> dict:
        return {"loss": self.loss}


def run_dqn_update(
    q_network: nn.Module,
    target_q_network: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    batch: DQNUpdateBatch,
    gamma: float,
    is_double: bool = False,
    gradient_clip_norm: Optional[float] = 10.0,
) -> DQNUpdateMetrics:
    """Run one standard DQN gradient update."""
    q_values = q_network(batch.states)
    state_action_values = q_values.gather(1, batch.actions.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        if is_double:
            next_actions = q_network(batch.next_states).argmax(dim=1, keepdim=True)
            max_next_q_values = target_q_network(batch.next_states).gather(1, next_actions).squeeze(1)
        else:
            max_next_q_values = target_q_network(batch.next_states).max(dim=1)[0]

        target_action_values = (
            batch.rewards.squeeze(1)
            + gamma * max_next_q_values * (1.0 - batch.dones.squeeze(1))
        )

    td_errors = state_action_values - target_action_values
    if batch.importance_weights is not None:
        element_loss = F.smooth_l1_loss(
            state_action_values,
            target_action_values,
            reduction="none",
        )
        loss = (element_loss * batch.importance_weights.squeeze(1)).mean()
    else:
        loss = loss_fn(state_action_values, target_action_values)

    optimizer.zero_grad()
    loss.backward()
    if gradient_clip_norm is not None:
        torch.nn.utils.clip_grad_norm_(q_network.parameters(), max_norm=gradient_clip_norm)
    optimizer.step()

    return DQNUpdateMetrics(loss=loss.item(), td_errors=td_errors.detach())


def run_recurrent_dqn_update(
    q_network: nn.Module,
    target_q_network: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    batch: RecurrentDQNUpdateBatch,
    gamma: float,
    gradient_clip_norm: Optional[float] = 10.0,
) -> DQNUpdateMetrics:
    """Run one recurrent DQN gradient update over a sampled sequence batch."""
    q_sequence, _ = q_network(batch.states)
    state_action_values = q_sequence.gather(2, batch.actions.unsqueeze(-1))

    with torch.no_grad():
        next_q_sequence, _ = target_q_network(batch.next_states)
        max_next_q_values = next_q_sequence.max(dim=-1, keepdim=True)[0]
        target_action_values = (
            batch.rewards + gamma * max_next_q_values * (1.0 - batch.dones)
        )

    if hasattr(loss_fn, "reduction") and loss_fn.reduction != "none":
        unreduced_loss_fn = type(loss_fn)(reduction="none")
        raw_loss = unreduced_loss_fn(state_action_values, target_action_values)
    else:
        raw_loss = loss_fn(state_action_values, target_action_values)

    masked_loss = raw_loss * batch.masks
    loss = masked_loss.sum() / (batch.masks.sum() + 1e-8)

    optimizer.zero_grad()
    loss.backward()
    if gradient_clip_norm is not None:
        torch.nn.utils.clip_grad_norm_(q_network.parameters(), max_norm=gradient_clip_norm)
    optimizer.step()

    td_errors = (state_action_values - target_action_values).detach()
    return DQNUpdateMetrics(loss=loss.item(), td_errors=td_errors)
