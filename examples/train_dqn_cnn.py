import numpy as np
import torch
import torch.nn as nn
from typing import Tuple

from modular_rl.networks import CNN, QHead
from modular_rl.algorithms import DQNConfig, ReplayBuffer, DQNAgent, QNetwork
from modular_rl.training import CustomEnvWrapper, Trainer, ConsoleLogger, MatplotlibLogger, CompositeLogger


class GridWorldEnv:
    """
    16x16 grid navigation task.
    Observation: float32 image of shape (1, 16, 16).
    Actions: 0=Up, 1=Down, 2=Left, 3=Right.
    """

    def __init__(self, size: int = 16):
        self.size = size
        self.max_steps = 60
        self.reset()

    def reset(self) -> Tuple[np.ndarray, dict]:
        self.agent_pos = [0, 0]
        self.goal_pos = [self.size - 1, self.size - 1]
        self.steps = 0
        return self._obs(), {}

    def _obs(self) -> np.ndarray:
        grid = np.zeros((1, self.size, self.size), dtype=np.float32)
        grid[0, self.goal_pos[0], self.goal_pos[1]] = 0.5
        grid[0, self.agent_pos[0], self.agent_pos[1]] = 1.0
        return grid

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        self.steps += 1
        r, c = self.agent_pos
        if action == 0:   r = max(0, r - 1)
        elif action == 1: r = min(self.size - 1, r + 1)
        elif action == 2: c = max(0, c - 1)
        elif action == 3: c = min(self.size - 1, c + 1)
        self.agent_pos = [r, c]

        reached = self.agent_pos == self.goal_pos
        reward = 10.0 if reached else -0.1
        truncated = self.steps >= self.max_steps
        return self._obs(), reward, reached, truncated, {}

    def close(self):
        pass


def build_cnn_q_net(state_shape: Tuple, action_dim: int) -> QNetwork:
    backbone = CNN(
        input_shape=state_shape, feature_dim=32,
        channels=[16, 16], kernels=[3, 3], strides=[1, 1], paddings=[1, 1],
    )
    return QNetwork(backbone, QHead(backbone.output_dim, action_dim))


def train():
    STATE_SHAPE = (1, 16, 16)
    ACTION_DIM = 4

    config = DQNConfig(
        env_name="GridWorld-16x16",
        total_timesteps=8000,
        learning_rate=1e-3,
        buffer_size=5000,
        batch_size=32,
        gamma=0.95,
        target_update_freq=150,
        learning_starts=200,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay_steps=2000,
        log_interval=10,
        save_dir="checkpoints",
    )
    print(f"Using device: {config.device}")

    env = CustomEnvWrapper(GridWorldEnv(size=16), state_dim=STATE_SHAPE, action_dim=ACTION_DIM)

    q_net = build_cnn_q_net(STATE_SHAPE, ACTION_DIM)
    target_q_net = build_cnn_q_net(STATE_SHAPE, ACTION_DIM)

    agent = DQNAgent(
        q_network=q_net,
        target_network=target_q_net,
        optimizer=torch.optim.Adam(q_net.parameters(), lr=config.learning_rate),
        criterion=nn.MSELoss(),
        replay_buffer=ReplayBuffer(config.buffer_size, config.device),
        config=config,
        action_dim=ACTION_DIM,
    )

    logger = CompositeLogger([
        ConsoleLogger(log_interval=config.log_interval, prefix="GridWorld "),
        MatplotlibLogger(save_path="dqn_cnn_results.png"),
    ])

    Trainer(agent, env, config, logger, save_path="checkpoints/dqn_cnn_gridworld.pt").train()


if __name__ == "__main__":
    train()
