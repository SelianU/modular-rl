import torch
import torch.nn as nn

from modular_rl.networks import MLP, QHead
from modular_rl.algorithms import DQNConfig, ReplayBuffer, DQNAgent, QNetwork
from modular_rl.training import GymEnvWrapper, Trainer, ConsoleLogger, MatplotlibLogger, CompositeLogger


def train():
    config = DQNConfig(
        env_name="CartPole-v1",
        total_timesteps=15000,
        learning_rate=1e-3,
        buffer_size=10000,
        batch_size=64,
        gamma=0.99,
        target_update_freq=200,
        learning_starts=500,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay_steps=3000,
        log_interval=10,
        save_dir="checkpoints",
    )
    print(f"Using device: {config.device}")

    env = GymEnvWrapper(config.env_name)

    backbone = MLP(input_dim=env.state_dim, hidden_dims=[128, 128])
    q_net = QNetwork(backbone, QHead(backbone.output_dim, env.action_dim))

    target_backbone = MLP(input_dim=env.state_dim, hidden_dims=[128, 128])
    target_q_net = QNetwork(target_backbone, QHead(target_backbone.output_dim, env.action_dim))

    agent = DQNAgent(
        q_network=q_net,
        target_network=target_q_net,
        optimizer=torch.optim.Adam(q_net.parameters(), lr=config.learning_rate),
        criterion=nn.SmoothL1Loss(),
        replay_buffer=ReplayBuffer(config.buffer_size, config.device),
        config=config,
        action_dim=env.action_dim,
    )

    logger = CompositeLogger([
        ConsoleLogger(log_interval=config.log_interval, prefix="DQN-MLP "),
        MatplotlibLogger(save_path="dqn_mlp_results.png"),
    ])

    Trainer(agent, env, config, logger, save_path="checkpoints/dqn_mlp_cartpole.pt").train()


if __name__ == "__main__":
    train()
