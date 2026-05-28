"""
DQN + Transformer on CartPole-v1 (POMDP variant).

The Transformer processes a sequence of L steps during training,
applying causal self-attention so each step only attends to past context.
At inference, a single step is passed (seq_len=1); to fully leverage history
at inference, pass a sliding window of recent observations instead.
"""
import torch
import torch.nn as nn

from modular_rl.networks import MLP, QHead
from modular_rl.networks.encoders import Transformer
from modular_rl.algorithms import DQNConfig, SequenceReplayBuffer, DQNAgent, QNetwork
from modular_rl.training import GymEnvWrapper, Trainer, ConsoleLogger, MatplotlibLogger, CompositeLogger


def build_transformer_q_net(state_dim: int, action_dim: int) -> QNetwork:
    base = MLP(input_dim=state_dim, hidden_dims=[64])
    transformer = Transformer(
        base_backbone=base,
        num_heads=4,
        num_layers=2,
        embed_dim=128,
        dim_feedforward=256,
        max_seq_len=100,
    )
    return QNetwork(transformer, QHead(transformer.output_dim, action_dim))


def train():
    config = DQNConfig(
        env_name="CartPole-v1",
        total_timesteps=15000,
        learning_rate=5e-4,
        buffer_size=1000,       # Sequence buffer capacity (episodes)
        batch_size=32,
        gamma=0.99,
        target_update_freq=200,
        learning_starts=300,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay_steps=4000,
        is_double=True,         # Double DQN for stability
        log_interval=10,
        save_dir="checkpoints",
    )
    print(f"Using device: {config.device}")

    env = GymEnvWrapper(config.env_name)

    q_net = build_transformer_q_net(env.state_dim, env.action_dim)
    target_q_net = build_transformer_q_net(env.state_dim, env.action_dim)

    agent = DQNAgent(
        q_network=q_net,
        target_network=target_q_net,
        optimizer=torch.optim.Adam(q_net.parameters(), lr=config.learning_rate),
        criterion=nn.MSELoss(reduction="none"),
        replay_buffer=SequenceReplayBuffer(config.buffer_size, config.device),
        config=config,
        action_dim=env.action_dim,
        is_recurrent=True,      # Enables sequence sampling and BPTT
    )

    logger = CompositeLogger([
        ConsoleLogger(log_interval=config.log_interval, prefix="DQN-Transformer "),
        MatplotlibLogger(save_path="dqn_transformer_results.png"),
    ])

    Trainer(
        agent, env, config, logger,
        save_path="checkpoints/dqn_transformer_cartpole.pt",
    ).train()


if __name__ == "__main__":
    train()
