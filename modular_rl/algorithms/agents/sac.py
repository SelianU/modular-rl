import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Union, Tuple

from .base import BaseAgent
from ..config import SACConfig
from ..buffers import ReplayBuffer

class SACAgent(BaseAgent):
    """
    Modular Soft Actor-Critic (SAC) Agent for continuous action spaces.
    Decoupled from network design via dependency injection.
    """
    def __init__(
        self,
        actor: nn.Module,
        critic: nn.Module,
        critic_target: nn.Module,
        actor_optimizer: optim.Optimizer,
        critic_optimizer: optim.Optimizer,
        replay_buffer: ReplayBuffer,
        config: SACConfig,
        action_dim: int
    ):
        self.actor = actor.to(config.device)
        self.critic = critic.to(config.device)
        self.critic_target = critic_target.to(config.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_target.eval()
        
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.replay_buffer = replay_buffer
        self.config = config
        self.action_dim = action_dim
        self.device = torch.device(config.device)
        
        self.gamma = config.gamma
        self.tau = config.tau
        
        self.global_step = 0
        
        # Entropy temperature parameter (alpha) setup
        self.auto_alpha = config.auto_alpha
        if self.auto_alpha:
            if config.target_entropy is not None:
                self.target_entropy = config.target_entropy
            else:
                self.target_entropy = -float(action_dim)
                
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=config.actor_lr)
            self.alpha = self.log_alpha.exp().item()
        else:
            self.alpha = config.alpha

    def select_action(self, state: np.ndarray, evaluation: bool = False) -> np.ndarray:
        """
        Select action. If evaluating, return deterministic mean action.
        Else, sample stochastic action.
        """
        self.global_step += 1
        
        # Random actions at the very beginning of training
        if not evaluation and self.global_step < self.config.learning_starts:
            # Random sample in continuous range [-1, 1] mapped to action limit
            # Assumes action bounds mapping inside the actor.head
            # For pure random exploration, we can sample standard normal and squash or use actor bounds:
            if hasattr(self.actor, 'head'):
                low = self.actor.head.action_limit_low
                high = self.actor.head.action_limit_high
                return np.random.uniform(low, high, size=self.action_dim)
            return np.random.uniform(-1.0, 1.0, size=self.action_dim)

        state_t = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            if evaluation:
                if hasattr(self.actor, 'get_mean_action'):
                    action = self.actor.get_mean_action(state_t)
                else:
                    action, _ = self.actor.sample(state_t) # fallback
            else:
                action, _ = self.actor.sample(state_t)
                
        return action.squeeze(0).cpu().numpy()

    def update(self) -> dict:
        """
        Perform a single gradient descent update for Critic, Actor, and Temperature.
        """
        if len(self.replay_buffer) < self.config.learning_starts:
            return {}
            
        # Sample replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.config.batch_size)
        
        # ------------------ Update Critic ------------------
        with torch.no_grad():
            # Sample next actions and log-probs from current policy
            next_actions, next_log_probs = self.actor.sample(next_states)
            
            # Compute next twin Q-values from target Critic
            next_q1, next_q2 = self.critic_target(next_states, next_actions)
            min_next_q = torch.min(next_q1, next_q2)
            
            # Target Q value: y = r + gamma * (min(Q_target) - alpha * log_prob) * (1 - done)
            target_q = rewards + self.gamma * (min_next_q - self.alpha * next_log_probs) * (1.0 - dones)
            
        # Current twin Q values
        current_q1, current_q2 = self.critic(states, actions)
        
        # Critic loss (Mean Squared Error)
        critic_loss = nn.MSELoss()(current_q1, target_q) + nn.MSELoss()(current_q2, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # ------------------ Update Actor ------------------
        # Sample current actions and log-probs from policy
        sampled_actions, log_probs = self.actor.sample(states)
        
        # Current twin Q values for sampled actions
        q1_new, q2_new = self.critic(states, sampled_actions)
        min_q_new = torch.min(q1_new, q2_new)
        
        # Actor loss: minimize (alpha * log_prob - min_q)
        actor_loss = (self.alpha * log_probs - min_q_new).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # ------------------ Update Alpha (Temperature) ------------------
        metrics = {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha": self.alpha
        }
        
        if self.auto_alpha:
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            
            self.alpha = self.log_alpha.exp().item()
            metrics["alpha_loss"] = alpha_loss.item()
            metrics["alpha"] = self.alpha
            
        # ------------------ Soft Update Critic Target ------------------
        self._soft_update_target_network(self.critic, self.critic_target, self.tau)
        
        return metrics

    def _soft_update_target_network(self, source: nn.Module, target: nn.Module, tau: float):
        """Polyak averaging target updates: target = tau * source + (1 - tau) * target"""
        for source_param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        checkpoint = {
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'global_step': self.global_step
        }
        if self.auto_alpha:
            checkpoint['log_alpha'] = self.log_alpha
            checkpoint['alpha_optimizer_state_dict'] = self.alpha_optimizer.state_dict()
            
        torch.save(checkpoint, filepath)
        
    def load(self, filepath: str):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.global_step = checkpoint['global_step']
        if self.auto_alpha and 'log_alpha' in checkpoint:
            self.log_alpha.data.copy_(checkpoint['log_alpha'].data)
            self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])
            self.alpha = self.log_alpha.exp().item()
