import numpy as np
import torch
import torch.nn as nn

from modular_rl.networks import MLP, RNN, QHead
from modular_rl.algorithms import DQNConfig, SequenceReplayBuffer, DQNAgent, QNetwork
from modular_rl.training import GymEnvWrapper, Trainer, ConsoleLogger, MatplotlibLogger, CompositeLogger


def mask_velocity(obs: np.ndarray) -> np.ndarray:
    """Mask cart velocity and pole angular velocity to create a POMDP."""
    masked = obs.copy()
    masked[1] = 0.0  # cart velocity
    masked[3] = 0.0  # pole angular velocity
    return masked


def build_rnn_q_net(state_dim: int, action_dim: int) -> QNetwork:
    base = MLP(input_dim=state_dim, hidden_dims=[64])
    rnn = RNN(base_backbone=base, rnn_type="LSTM", rnn_hidden_dim=64)
    return QNetwork(rnn, QHead(rnn.output_dim, action_dim))


def train():
    config = DQNConfig(
        env_name="CartPole-v1",
        total_timesteps=15000,
        learning_rate=1e-3,
        buffer_size=1000,
        batch_size=32,
        gamma=0.99,
        target_update_freq=200,
        learning_starts=300,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay_steps=4000,
        log_interval=10,
        save_dir="checkpoints",
    )
    print(f"Using device: {config.device}")

    env = GymEnvWrapper(config.env_name, obs_transform=mask_velocity)

    q_net = build_rnn_q_net(env.state_dim, env.action_dim)
    target_q_net = build_rnn_q_net(env.state_dim, env.action_dim)

    agent = DQNAgent(
        q_network=q_net,
        target_network=target_q_net,
        optimizer=torch.optim.Adam(q_net.parameters(), lr=config.learning_rate),
        criterion=nn.MSELoss(reduction="none"),
        replay_buffer=SequenceReplayBuffer(config.buffer_size, config.device),
        config=config,
        action_dim=env.action_dim,
        is_recurrent=True,
    )

    logger = CompositeLogger([
        ConsoleLogger(log_interval=config.log_interval, prefix="DRQN "),
        MatplotlibLogger(save_path="dqn_rnn_results.png"),
    ])

    Trainer(agent, env, config, logger, save_path="checkpoints/dqn_rnn_cartpole.pt").train()


if __name__ == "__main__":
    train()
