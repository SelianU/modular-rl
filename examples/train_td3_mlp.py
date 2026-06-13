import torch

from modular_rl.networks import MLP, DoubleQCriticHead, DeterministicPolicyHead
from modular_rl.algorithms import TD3Config, ReplayBuffer, TD3Agent
from modular_rl.algorithms.models import TD3Actor, TD3Critic
from modular_rl.training import GymEnvWrapper, Trainer, ConsoleLogger, MatplotlibLogger, CompositeLogger


def build_actor(state_dim, action_dim, hidden_dims, action_low, action_high):
    backbone = MLP(input_dim=state_dim, hidden_dims=hidden_dims)
    return TD3Actor(
        backbone,
        DeterministicPolicyHead(backbone.output_dim, action_dim, action_low, action_high),
    )


def build_critic(state_dim, action_dim, hidden_dims):
    backbone = MLP(input_dim=state_dim, hidden_dims=hidden_dims)
    return TD3Critic(backbone, DoubleQCriticHead(backbone.output_dim, action_dim, hidden_dims[-1]))


def train():
    config = TD3Config(
        env_name="Pendulum-v1",
        total_timesteps=50_000,
        learning_rate=3e-4,
        buffer_size=100_000,
        batch_size=256,
        gamma=0.99,
        tau=0.005,
        policy_delay=2,
        exploration_noise=0.1,
        target_noise=0.2,
        noise_clip=0.5,
        learning_starts=1000,
        log_interval=10,
        save_dir="checkpoints",
    )
    print(f"Using device: {config.device}")

    environment = GymEnvWrapper(config.env_name)
    state_dim = environment.state_dim
    action_dim = environment.action_dim
    action_low = environment.action_low
    action_high = environment.action_high
    hidden_dims = [256, 256]

    actor = build_actor(state_dim, action_dim, hidden_dims, action_low, action_high)
    actor_target = build_actor(state_dim, action_dim, hidden_dims, action_low, action_high)
    critic = build_critic(state_dim, action_dim, hidden_dims)
    critic_target = build_critic(state_dim, action_dim, hidden_dims)

    agent = TD3Agent(
        actor=actor,
        actor_target=actor_target,
        critic=critic,
        critic_target=critic_target,
        actor_optimizer=torch.optim.Adam(actor.parameters(), lr=config.learning_rate),
        critic_optimizer=torch.optim.Adam(critic.parameters(), lr=config.learning_rate),
        replay_buffer=ReplayBuffer(config.buffer_size, config.device),
        config=config,
        action_dim=action_dim,
        action_low=action_low,
        action_high=action_high,
    )

    logger = CompositeLogger([
        ConsoleLogger(log_interval=config.log_interval, prefix="TD3-MLP "),
        MatplotlibLogger(save_path="td3_mlp_results.png"),
    ])

    Trainer(agent, environment, config, logger, save_path="checkpoints/td3_mlp_pendulum.pt").train()


if __name__ == "__main__":
    train()
