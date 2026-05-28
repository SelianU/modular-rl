import os
import random
import numpy as np
import torch
import torch.nn as nn
from typing import Union, Tuple, Optional

import torch.nn.functional as F

from .base import BaseAgent
from ..config import DQNConfig
from ..buffers import ReplayBuffer, SequenceReplayBuffer, PrioritizedReplayBuffer

class DQNAgent(BaseAgent):
    """
    Modular DQN Agent. Supports standard (MLP, CNN) and recurrent (RNN/LSTM) architectures.
    Decoupled from network design and loss function via dependency injection.
    """
    def __init__(
        self,
        q_network: nn.Module,
        target_network: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        replay_buffer: Union[ReplayBuffer, SequenceReplayBuffer],
        config: DQNConfig,
        action_dim: int,
        is_recurrent: bool = False
    ):
        self.q_network = q_network.to(config.device)
        self.target_network = target_network.to(config.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        self.optimizer = optimizer
        self.criterion = criterion
        self.replay_buffer = replay_buffer
        self.config = config
        self.action_dim = action_dim
        self.is_recurrent = is_recurrent
        self.device = torch.device(config.device)
        
        self.global_step = 0
        self.epsilon = config.epsilon_start
        self.is_double = getattr(config, "is_double", False)

        # Recurrent hidden state for online decision making (only used if is_recurrent is True)
        self.hidden_state = None

    def reset_hidden_state(self):
        """Reset the RNN hidden state (should be called at the beginning of each episode)."""
        self.hidden_state = None

    def select_action(self, state: np.ndarray, evaluation: bool = False) -> int:
        """
        Select action using epsilon-greedy policy.
        """
        # Linear decay of epsilon during training
        if not evaluation:
            self.global_step += 1
            # Epsilon decay calculation
            decay_frac = min(1.0, self.global_step / self.config.epsilon_decay_steps)
            self.epsilon = self.config.epsilon_start + decay_frac * (self.config.epsilon_end - self.config.epsilon_start)
            
        # Select action
        if not evaluation and random.random() < self.epsilon:
            action = random.randint(0, self.action_dim - 1)
            # If recurrent, we must still forward to update the hidden state sequence
            if self.is_recurrent:
                with torch.no_grad():
                    state_t = torch.from_numpy(state).float().unsqueeze(0).to(self.device) # shape: (1, state_dim)
                    _, self.hidden_state = self.q_network(state_t, self.hidden_state)
            return action
        else:
            with torch.no_grad():
                # state has shape (state_dim,) -> (1, state_dim)
                state_t = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
                if self.is_recurrent:
                    q_values, self.hidden_state = self.q_network(state_t, self.hidden_state)
                else:
                    q_values = self.q_network(state_t)
                return q_values.argmax(dim=-1).item()

    def update(self) -> dict:
        """
        Sample a batch and run one step of gradient descent.
        Supports standard TD loss and Sequence BPTT loss for RNNs.
        """
        if len(self.replay_buffer) < self.config.learning_starts:
            return {}

        if self.is_recurrent:
            return self._update_recurrent()
        else:
            return self._update_standard()

    def _update_standard(self) -> dict:
        is_per = isinstance(self.replay_buffer, PrioritizedReplayBuffer)

        if is_per:
            states, actions, rewards, next_states, dones, weights, idxs = \
                self.replay_buffer.sample(self.config.batch_size)
        else:
            states, actions, rewards, next_states, dones = \
                self.replay_buffer.sample(self.config.batch_size)
            weights = None

        # Compute online Q-values: Q(s, a)
        q_values = self.q_network(states)
        state_action_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Compute target Q-values
        with torch.no_grad():
            if self.is_double:
                next_actions = self.q_network(next_states).argmax(dim=1, keepdim=True)
                max_next_q_values = self.target_network(next_states).gather(1, next_actions).squeeze(1)
            else:
                max_next_q_values = self.target_network(next_states).max(dim=1)[0]
            target_action_values = (
                rewards.squeeze(1) + self.config.gamma * max_next_q_values * (1.0 - dones.squeeze(1))
            )

        # Compute loss (element-wise for PER, criterion-based otherwise)
        if is_per:
            element_loss = F.smooth_l1_loss(state_action_values, target_action_values, reduction="none")
            loss = (element_loss * weights.squeeze(1)).mean()
            self.replay_buffer.update_priorities(idxs, element_loss.detach().cpu().numpy())
        else:
            loss = self.criterion(state_action_values, target_action_values)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
        self.optimizer.step()

        if self.global_step % self.config.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        return {"loss": loss.item(), "epsilon": self.epsilon}

    def _update_recurrent(self) -> dict:
        # Sequence batch size and sequence length
        # For RNNs, we sample sequences of length L
        seq_len = 10  # This could be configurable
        states, actions, rewards, next_states, dones, masks = self.replay_buffer.sample(
            self.config.batch_size, seq_len
        )
        
        # states shape: (B, L, state_dim)
        # We forward the entire sequence through the RNN. We start with zero hidden states.
        q_seq, _ = self.q_network(states)  # (B, L, action_dim)
        
        # Gather Q-values for the chosen actions
        # actions shape: (B, L) -> expand to (B, L, 1)
        state_action_values = q_seq.gather(2, actions.unsqueeze(-1))  # (B, L, 1)
        
        with torch.no_grad():
            # Get target sequence Q-values
            next_q_seq, _ = self.target_network(next_states)  # (B, L, action_dim)
            max_next_q = next_q_seq.max(dim=-1, keepdim=True)[0]  # (B, L, 1)
            target_action_values = rewards + self.config.gamma * max_next_q * (1.0 - dones)
            
        # Compute element-wise loss
        # criterion must have reduction='none' so we can mask it manually
        if hasattr(self.criterion, 'reduction') and self.criterion.reduction != 'none':
            # Create a temporary none-reduction criterion
            temp_criterion = type(self.criterion)(reduction='none')
            raw_loss = temp_criterion(state_action_values, target_action_values)
        else:
            raw_loss = self.criterion(state_action_values, target_action_values)
            
        # Apply sequence mask (ignore padded frames)
        masked_loss = raw_loss * masks
        loss = masked_loss.sum() / (masks.sum() + 1e-8)
        
        # Optimize online network
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
        self.optimizer.step()
        
        # Update target network
        if self.global_step % self.config.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
            
        return {"loss": loss.item(), "epsilon": self.epsilon}

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'global_step': self.global_step
        }, filepath)
        
    def load(self, filepath: str):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.global_step = checkpoint['global_step']
