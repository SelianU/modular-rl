import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseAgent
from ..config import TD3Config
from ..buffers import ReplayBuffer


class TD3Agent(BaseAgent):
    """
    Twin Delayed Deep Deterministic Policy Gradient (TD3) Agent.

    Improvements over DDPG:
    - Twin critics to reduce overestimation bias.
    - Delayed actor updates (every `policy_delay` critic steps).
    - Target policy smoothing (clipped Gaussian noise on target actions).

    Parameters
    ----------
    actor, actor_target        : TD3Actor (DeterministicPolicyHead).
    critic, critic_target      : SACCritic / TD3Critic (DoubleQCriticHead).
    action_dim                 : Number of action dimensions.
    action_low / action_high   : Scalar bounds applied during exploration and
                                 target-policy clamping.
    """

    def __init__(
        self,
        actor: nn.Module,
        actor_target: nn.Module,
        critic: nn.Module,
        critic_target: nn.Module,
        actor_optimizer: torch.optim.Optimizer,
        critic_optimizer: torch.optim.Optimizer,
        replay_buffer: ReplayBuffer,
        config: TD3Config,
        action_dim: int,
        action_low: float = -1.0,
        action_high: float = 1.0,
    ):
        self.actor = actor.to(config.device)
        self.actor_target = actor_target.to(config.device)
        self.critic = critic.to(config.device)
        self.critic_target = critic_target.to(config.device)

        self.actor_target.load_state_dict(actor.state_dict())
        self.critic_target.load_state_dict(critic.state_dict())
        self.critic_target.eval()
        self.actor_target.eval()

        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.replay_buffer = replay_buffer
        self.config = config
        self.action_dim = action_dim
        self.action_low = action_low
        self.action_high = action_high
        self.device = torch.device(config.device)

        self.global_step = 0
        self._update_count = 0

    # ------------------------------------------------------------------ #
    # BaseAgent interface                                                   #
    # ------------------------------------------------------------------ #

    def select_action(self, state: np.ndarray, evaluation: bool = False) -> np.ndarray:
        if not evaluation and self.global_step < self.config.learning_starts:
            return np.random.uniform(self.action_low, self.action_high, self.action_dim)

        self.global_step += 1
        state_t = torch.from_numpy(np.array(state, dtype=np.float32)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action = self.actor(state_t).squeeze(0).cpu().numpy()

        if not evaluation:
            noise = np.random.normal(0.0, self.config.exploration_noise, size=self.action_dim)
            action = np.clip(action + noise, self.action_low, self.action_high)

        return action

    def update(self) -> dict:
        if len(self.replay_buffer) < self.config.learning_starts:
            return {}

        self.global_step += 1
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.config.batch_size)
        self._update_count += 1

        # ---- Critic update -------------------------------------------
        with torch.no_grad():
            noise = (torch.randn_like(actions) * self.config.target_noise).clamp(
                -self.config.noise_clip, self.config.noise_clip
            )
            next_actions = (self.actor_target(next_states) + noise).clamp(
                self.action_low, self.action_high
            )
            next_q1, next_q2 = self.critic_target(next_states, next_actions)
            target_q = rewards + self.config.gamma * torch.min(next_q1, next_q2) * (1.0 - dones)

        current_q1, current_q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        metrics = {"critic_loss": critic_loss.item()}

        # ---- Delayed actor update ------------------------------------
        if self._update_count % self.config.policy_delay == 0:
            actor_loss = -self.critic(states, self.actor(states))[0].mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self._soft_update(self.actor, self.actor_target)
            self._soft_update(self.critic, self.critic_target)

            metrics["actor_loss"] = actor_loss.item()

        return metrics

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save({
            "actor": self.actor.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "global_step": self.global_step,
        }, filepath)

    def load(self, filepath: str):
        ckpt = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.actor_target.load_state_dict(ckpt["actor_target"])
        self.critic.load_state_dict(ckpt["critic"])
        self.critic_target.load_state_dict(ckpt["critic_target"])
        self.actor_optimizer.load_state_dict(ckpt["actor_optimizer"])
        self.critic_optimizer.load_state_dict(ckpt["critic_optimizer"])
        self.global_step = ckpt["global_step"]

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _soft_update(self, source: nn.Module, target: nn.Module):
        tau = self.config.tau
        for s_p, t_p in zip(source.parameters(), target.parameters()):
            t_p.data.copy_(tau * s_p.data + (1.0 - tau) * t_p.data)
