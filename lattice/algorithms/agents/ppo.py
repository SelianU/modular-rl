import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseAgent
from ..config import PPOConfig
from ..buffers import RolloutBuffer


class PPOAgent(BaseAgent):
    """
    Proximal Policy Optimization (PPO) Agent.

    On-policy algorithm that clips the policy update ratio to stabilise training.
    Supports both discrete (CategoricalPolicyHead) and continuous
    (DiagonalGaussianHead) action spaces via the PPOActor interface.

    The agent uses the shared RLTrainer loop:
    - select_action() computes V(s) and log π(a|s) and stores them as pending.
    - observe() flushes the pending data into the RolloutBuffer.
    - update() returns {} until the buffer is full, then performs K epochs of
      mini-batch PPO updates and resets the buffer.
    """

    def __init__(
        self,
        actor: nn.Module,
        critic: nn.Module,
        actor_optimizer: torch.optim.Optimizer,
        critic_optimizer: torch.optim.Optimizer,
        rollout_buffer: RolloutBuffer,
        config: PPOConfig,
    ):
        self.actor = actor.to(config.device)
        self.critic = critic.to(config.device)
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.rollout_buffer = rollout_buffer
        self.config = config
        self.device = torch.device(config.device)
        self.global_step = 0

        # State shared between select_action() and observe()
        self._pending_value: float = 0.0
        self._pending_log_prob: float = 0.0
        self._last_next_state = None
        self._last_done: bool = False

    # ------------------------------------------------------------------ #
    # BaseAgent interface                                                   #
    # ------------------------------------------------------------------ #

    def observe(self, state, action, reward: float, next_state, done: bool) -> None:
        self.rollout_buffer.push(
            state, action, reward,
            self._pending_value, self._pending_log_prob, done,
        )
        self._last_next_state = next_state
        self._last_done = bool(done)

    def select_action(self, state: np.ndarray, evaluation: bool = False):
        self.global_step += 1
        state_t = torch.from_numpy(np.array(state, dtype=np.float32)).unsqueeze(0).to(self.device)

        with torch.no_grad():
            value: float = self.critic(state_t).squeeze(-1).item()

            if evaluation:
                action_t = self.actor.get_mean_action(state_t)
                action = self._tensor_to_action(action_t)
                log_prob = 0.0
            else:
                action_t, log_prob_t, _ = self.actor.get_action_and_logprob(state_t)
                action = self._tensor_to_action(action_t)
                log_prob = log_prob_t.item()

        self._pending_value = value
        self._pending_log_prob = log_prob
        return action

    def update(self) -> dict:
        if not self.rollout_buffer.is_full():
            return {}

        # Bootstrap value for the state that follows the rollout
        with torch.no_grad():
            if self._last_done or self._last_next_state is None:
                last_value = 0.0
            else:
                last_s = torch.from_numpy(
                    np.array(self._last_next_state, dtype=np.float32)
                ).unsqueeze(0).to(self.device)
                last_value = self.critic(last_s).squeeze(-1).item()

        actor_losses, value_losses, entropies = [], [], []

        for _ in range(self.config.n_epochs):
            for states, actions, old_log_probs, advantages, returns in \
                    self.rollout_buffer.get_batches(self.config.batch_size, last_value):

                log_probs, entropy = self.actor.evaluate_actions(states, actions)

                ratio = (log_probs - old_log_probs).exp()
                clip_ratio = ratio.clamp(1.0 - self.config.clip_eps, 1.0 + self.config.clip_eps)
                actor_loss = -torch.min(ratio * advantages, clip_ratio * advantages).mean()

                values = self.critic(states).squeeze(-1)
                value_loss = F.mse_loss(values, returns)

                entropy_mean = entropy.mean()
                total_loss = (
                    actor_loss
                    + self.config.value_coeff * value_loss
                    - self.config.entropy_coeff * entropy_mean
                )

                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.max_grad_norm)
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.max_grad_norm)
                self.actor_optimizer.step()
                self.critic_optimizer.step()

                actor_losses.append(actor_loss.item())
                value_losses.append(value_loss.item())
                entropies.append(entropy_mean.item())

        self.rollout_buffer.reset()

        return {
            "actor_loss": float(np.mean(actor_losses)),
            "value_loss": float(np.mean(value_losses)),
            "entropy": float(np.mean(entropies)),
        }

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "global_step": self.global_step,
        }, filepath)

    def load(self, filepath: str):
        ckpt = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.actor_optimizer.load_state_dict(ckpt["actor_optimizer"])
        self.critic_optimizer.load_state_dict(ckpt["critic_optimizer"])
        self.global_step = ckpt["global_step"]

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _tensor_to_action(t: torch.Tensor):
        t = t.squeeze(0)
        if t.dim() == 0 or t.numel() == 1:
            return t.item()
        return t.cpu().numpy()
