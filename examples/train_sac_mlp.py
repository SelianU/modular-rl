import torch

from modular_rl.networks import MLP, GaussianPolicyHead, DoubleQCriticHead
from modular_rl.algorithms import SACConfig, ReplayBuffer, SACAgent, SACActor, SACCritic
from modular_rl.training import GymEnvWrapper, Trainer, ConsoleLogger, MatplotlibLogger, CompositeLogger


def train():
    config = SACConfig(
        env_name="Pendulum-v1",
        total_timesteps=20000,
        actor_lr=3e-4,
        critic_lr=3e-4,
        buffer_size=50000,
        batch_size=128,
        gamma=0.99,
        tau=0.005,
        alpha=0.2,
        auto_alpha=True,
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

    actor_backbone = MLP(input_dim=state_dim, hidden_dims=[256, 256])
    actor = SACActor(
        actor_backbone,
        GaussianPolicyHead(actor_backbone.output_dim, action_dim, action_low, action_high),
    )

    critic_backbone = MLP(input_dim=state_dim, hidden_dims=[256, 256])
    critic = SACCritic(
        critic_backbone,
        DoubleQCriticHead(critic_backbone.output_dim, action_dim, 256),
    )

    critic_target_backbone = MLP(input_dim=state_dim, hidden_dims=[256, 256])
    critic_target = SACCritic(
        critic_target_backbone,
        DoubleQCriticHead(critic_target_backbone.output_dim, action_dim, 256),
    )

    agent = SACAgent(
        actor=actor,
        critic=critic,
        critic_target=critic_target,
        actor_optimizer=torch.optim.Adam(actor.parameters(), lr=config.actor_lr),
        critic_optimizer=torch.optim.Adam(critic.parameters(), lr=config.critic_lr),
        replay_buffer=ReplayBuffer(config.buffer_size, config.device),
        config=config,
        action_dim=action_dim,
    )

    logger = CompositeLogger([
        ConsoleLogger(log_interval=config.log_interval, prefix="SAC-MLP "),
        MatplotlibLogger(save_path="sac_mlp_results.png"),
    ])

    Trainer(agent, environment, config, logger, save_path="checkpoints/sac_mlp_pendulum.pt").train()


if __name__ == "__main__":
    train()
