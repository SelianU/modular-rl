import os
import random
import numpy as np
import torch
import torch.nn as nn
from typing import Union

from .base import BaseAgent
from ..config import DQNConfig
from ..buffers import ReplayBuffer, SequenceReplayBuffer, PrioritizedReplayBuffer
from ..updates import (
    DQNUpdateBatch,
    RecurrentDQNUpdateBatch,
    run_dqn_update,
    run_recurrent_dqn_update,
)


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
        batch = DQNUpdateBatch(
            states=states,
            actions=actions,
            rewards=rewards,
            next_states=next_states,
            dones=dones,
            importance_weights=weights,
        )
        update_metrics = run_dqn_update(
            q_network=self.q_network,
            target_q_network=self.target_network,
            optimizer=self.optimizer,
            loss_fn=self.criterion,
            batch=batch,
            gamma=self.config.gamma,
            is_double=self.is_double,
        )
        if is_per:
            self.replay_buffer.update_priorities(
                idxs,
                update_metrics.td_errors.detach().cpu().numpy(),
            )

        self._maybe_update_target_network()
        metrics = update_metrics.as_dict()
        metrics["epsilon"] = self.epsilon
        return metrics

    def _update_recurrent(self) -> dict:
        # Sequence batch size and sequence length
        # For RNNs, we sample sequences of length L
        seq_len = 10  # This could be configurable
        states, actions, rewards, next_states, dones, masks = self.replay_buffer.sample(
            self.config.batch_size, seq_len
        )
        
        batch = RecurrentDQNUpdateBatch(
            states=states,
            actions=actions,
            rewards=rewards,
            next_states=next_states,
            dones=dones,
            masks=masks,
        )
        update_metrics = run_recurrent_dqn_update(
            q_network=self.q_network,
            target_q_network=self.target_network,
            optimizer=self.optimizer,
            loss_fn=self.criterion,
            batch=batch,
            gamma=self.config.gamma,
        )
        self._maybe_update_target_network()
        metrics = update_metrics.as_dict()
        metrics["epsilon"] = self.epsilon
        return metrics

    def _maybe_update_target_network(self) -> None:
        if self.global_step % self.config.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

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
