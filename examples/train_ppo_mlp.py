import torch

from modular_rl.networks import MLP, CategoricalPolicyHead, ValueHead
from modular_rl.algorithms import PPOConfig, PPOAgent
from modular_rl.algorithms.models import PPOActor, PPOCritic
from modular_rl.algorithms.buffers import RolloutBuffer
from modular_rl.training import GymEnvWrapper, Trainer, ConsoleLogger, MatplotlibLogger, CompositeLogger


def train():
    config = PPOConfig(
        env_name="CartPole-v1",
        total_timesteps=200_000,
        n_steps=512,
        n_epochs=10,
        batch_size=64,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.2,
        value_coeff=0.5,
        entropy_coeff=0.01,
        log_interval=10,
        save_dir="checkpoints",
    )
    print(f"Using device: {config.device}")

    env = GymEnvWrapper(config.env_name)

    actor_bb = MLP(input_dim=env.state_dim, hidden_dims=[64, 64])
    actor = PPOActor(actor_bb, CategoricalPolicyHead(actor_bb.output_dim, env.action_dim))

    critic_bb = MLP(input_dim=env.state_dim, hidden_dims=[64, 64])
    critic = PPOCritic(critic_bb, ValueHead(critic_bb.output_dim))

    agent = PPOAgent(
        actor=actor,
        critic=critic,
        actor_optimizer=torch.optim.Adam(actor.parameters(), lr=config.learning_rate),
        critic_optimizer=torch.optim.Adam(critic.parameters(), lr=config.learning_rate),
        rollout_buffer=RolloutBuffer(config.n_steps, config.device, config.gamma, config.gae_lambda),
        config=config,
    )

    logger = CompositeLogger([
        ConsoleLogger(log_interval=config.log_interval, prefix="PPO-MLP "),
        MatplotlibLogger(save_path="ppo_mlp_results.png"),
    ])

    Trainer(agent, env, config, logger, save_path="checkpoints/ppo_mlp_cartpole.pt").train()


if __name__ == "__main__":
    train()
