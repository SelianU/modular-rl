"""
One-liner factory functions for the most common training setups.

    trainer = quick_dqn("CartPole-v1")
    trainer.train()

    trainer = quick_sac("Pendulum-v1", hidden_dims=[256, 256], total_timesteps=50000)
    trainer.train()
"""
from typing import List, Optional

import torch
import torch.nn as nn

from modular_rl.networks import (
    MLP, QHead, GaussianPolicyHead, DoubleQCriticHead,
    CategoricalPolicyHead, DeterministicPolicyHead, ValueHead,
)
from modular_rl.networks.encoders import RNN
from modular_rl.algorithms import (
    DQNConfig, SACConfig, PPOConfig, TD3Config,
    ReplayBuffer, SequenceReplayBuffer, RolloutBuffer,
    DQNAgent, SACAgent, PPOAgent, TD3Agent,
    QNetwork, SACActor, SACCritic, PPOActor, PPOCritic, TD3Actor, TD3Critic,
)

from .env_wrapper import GymEnvWrapper
from .logger import BaseLogger, ConsoleLogger, MatplotlibLogger, CompositeLogger
from .trainer import Trainer


def quick_dqn(
    env_name: str,
    backbone: str = "mlp",
    hidden_dims: Optional[List[int]] = None,
    save_path: Optional[str] = None,
    plot_path: Optional[str] = None,
    logger: Optional[BaseLogger] = None,
    **config_kwargs,
) -> Trainer:
    """
    Build and return a Trainer for a DQN experiment in one call.

    Parameters
    ----------
    env_name    : Gymnasium environment name (e.g. "CartPole-v1").
    backbone    : "mlp" (default) or "rnn" (LSTM-based DRQN).
    hidden_dims : Hidden layer sizes. Defaults to [128, 128].
    save_path   : Where to save the trained model (.pt). Optional.
    plot_path   : Where to save the learning-curve PNG. Optional.
    logger      : Custom BaseLogger. Defaults to ConsoleLogger [+ MatplotlibLogger].
    **config_kwargs : Any DQNConfig field (total_timesteps, learning_rate, …).

    Returns
    -------
    Trainer — call .train() to start, .evaluate() to assess.
    """
    if hidden_dims is None:
        hidden_dims = [128, 128]

    config = DQNConfig(env_name=env_name, **config_kwargs)
    env = GymEnvWrapper(env_name)

    if backbone == "mlp":
        bb = MLP(input_dim=env.state_dim, hidden_dims=hidden_dims)
        t_bb = MLP(input_dim=env.state_dim, hidden_dims=hidden_dims)
        replay_buffer = ReplayBuffer(config.buffer_size, config.device)
        is_recurrent = False
    elif backbone == "rnn":
        base = MLP(input_dim=env.state_dim, hidden_dims=hidden_dims)
        bb = RNN(base_backbone=base, rnn_type="LSTM", rnn_hidden_dim=hidden_dims[-1])
        t_base = MLP(input_dim=env.state_dim, hidden_dims=hidden_dims)
        t_bb = RNN(base_backbone=t_base, rnn_type="LSTM", rnn_hidden_dim=hidden_dims[-1])
        replay_buffer = SequenceReplayBuffer(config.buffer_size, config.device)
        is_recurrent = True
    else:
        raise ValueError(f"quick_dqn supports backbone='mlp' or 'rnn'. Got '{backbone}'.")

    q_net = QNetwork(bb, QHead(bb.output_dim, env.action_dim))
    target_q_net = QNetwork(t_bb, QHead(t_bb.output_dim, env.action_dim))

    criterion = nn.MSELoss(reduction="none") if is_recurrent else nn.SmoothL1Loss()
    agent = DQNAgent(
        q_network=q_net,
        target_network=target_q_net,
        optimizer=torch.optim.Adam(q_net.parameters(), lr=config.learning_rate),
        criterion=criterion,
        replay_buffer=replay_buffer,
        config=config,
        action_dim=env.action_dim,
        is_recurrent=is_recurrent,
    )

    if logger is None:
        loggers: list = [ConsoleLogger(log_interval=config.log_interval)]
        if plot_path:
            loggers.append(MatplotlibLogger(save_path=plot_path))
        logger = CompositeLogger(loggers) if len(loggers) > 1 else loggers[0]

    return Trainer(agent, env, config, logger, save_path=save_path)


def quick_sac(
    env_name: str,
    hidden_dims: Optional[List[int]] = None,
    save_path: Optional[str] = None,
    plot_path: Optional[str] = None,
    logger: Optional[BaseLogger] = None,
    **config_kwargs,
) -> Trainer:
    """
    Build and return a Trainer for a SAC experiment in one call.

    Parameters
    ----------
    env_name    : Gymnasium environment name (e.g. "Pendulum-v1").
    hidden_dims : Hidden layer sizes for actor and critic. Defaults to [256, 256].
    save_path   : Where to save the trained model (.pt). Optional.
    plot_path   : Where to save the learning-curve PNG. Optional.
    logger      : Custom BaseLogger. Defaults to ConsoleLogger [+ MatplotlibLogger].
    **config_kwargs : Any SACConfig field (total_timesteps, actor_lr, …).

    Returns
    -------
    Trainer — call .train() to start, .evaluate() to assess.
    """
    if hidden_dims is None:
        hidden_dims = [256, 256]

    config = SACConfig(env_name=env_name, **config_kwargs)
    env = GymEnvWrapper(env_name)

    s_dim = env.state_dim
    a_dim = env.action_dim
    a_low = env.action_low
    a_high = env.action_high

    actor_bb = MLP(input_dim=s_dim, hidden_dims=hidden_dims)
    actor = SACActor(actor_bb, GaussianPolicyHead(actor_bb.output_dim, a_dim, a_low, a_high))

    critic_bb = MLP(input_dim=s_dim, hidden_dims=hidden_dims)
    critic = SACCritic(critic_bb, DoubleQCriticHead(critic_bb.output_dim, a_dim, hidden_dims[-1]))

    critic_target_bb = MLP(input_dim=s_dim, hidden_dims=hidden_dims)
    critic_target = SACCritic(
        critic_target_bb,
        DoubleQCriticHead(critic_target_bb.output_dim, a_dim, hidden_dims[-1]),
    )

    agent = SACAgent(
        actor=actor,
        critic=critic,
        critic_target=critic_target,
        actor_optimizer=torch.optim.Adam(actor.parameters(), lr=config.actor_lr),
        critic_optimizer=torch.optim.Adam(critic.parameters(), lr=config.critic_lr),
        replay_buffer=ReplayBuffer(config.buffer_size, config.device),
        config=config,
        action_dim=a_dim,
    )

    if logger is None:
        loggers: list = [ConsoleLogger(log_interval=config.log_interval)]
        if plot_path:
            loggers.append(MatplotlibLogger(save_path=plot_path))
        logger = CompositeLogger(loggers) if len(loggers) > 1 else loggers[0]

    return Trainer(agent, env, config, logger, save_path=save_path)


def quick_ppo(
    env_name: str,
    hidden_dims: Optional[List[int]] = None,
    save_path: Optional[str] = None,
    plot_path: Optional[str] = None,
    logger: Optional[BaseLogger] = None,
    **config_kwargs,
) -> Trainer:
    """
    Build and return a Trainer for a PPO experiment in one call.
    Automatically uses CategoricalPolicyHead for discrete envs,
    DiagonalGaussianHead for continuous envs.

    Parameters
    ----------
    env_name     : Gymnasium environment name (e.g. "CartPole-v1").
    hidden_dims  : Hidden layer sizes for actor and critic. Defaults to [64, 64].
    **config_kwargs : Any PPOConfig field (total_timesteps, n_steps, clip_eps, …).
    """
    if hidden_dims is None:
        hidden_dims = [64, 64]

    config = PPOConfig(env_name=env_name, **config_kwargs)
    env = GymEnvWrapper(env_name)

    actor_bb = MLP(input_dim=env.state_dim, hidden_dims=hidden_dims)
    if env.is_continuous:
        from modular_rl.networks import DiagonalGaussianHead
        actor_head = DiagonalGaussianHead(actor_bb.output_dim, env.action_dim)
    else:
        actor_head = CategoricalPolicyHead(actor_bb.output_dim, env.action_dim)
    actor = PPOActor(actor_bb, actor_head)

    critic_bb = MLP(input_dim=env.state_dim, hidden_dims=hidden_dims)
    critic = PPOCritic(critic_bb, ValueHead(critic_bb.output_dim))

    agent = PPOAgent(
        actor=actor,
        critic=critic,
        actor_optimizer=torch.optim.Adam(actor.parameters(), lr=config.learning_rate),
        critic_optimizer=torch.optim.Adam(critic.parameters(), lr=config.learning_rate),
        rollout_buffer=RolloutBuffer(config.n_steps, config.device, config.gamma, config.gae_lambda),
        config=config,
    )

    if logger is None:
        loggers: list = [ConsoleLogger(log_interval=config.log_interval)]
        if plot_path:
            loggers.append(MatplotlibLogger(save_path=plot_path))
        logger = CompositeLogger(loggers) if len(loggers) > 1 else loggers[0]

    return Trainer(agent, env, config, logger, save_path=save_path)


def quick_td3(
    env_name: str,
    hidden_dims: Optional[List[int]] = None,
    save_path: Optional[str] = None,
    plot_path: Optional[str] = None,
    logger: Optional[BaseLogger] = None,
    **config_kwargs,
) -> Trainer:
    """
    Build and return a Trainer for a TD3 experiment in one call.

    Parameters
    ----------
    env_name     : Gymnasium environment name (e.g. "Pendulum-v1").
    hidden_dims  : Hidden layer sizes. Defaults to [256, 256].
    **config_kwargs : Any TD3Config field (total_timesteps, policy_delay, …).
    """
    if hidden_dims is None:
        hidden_dims = [256, 256]

    config = TD3Config(env_name=env_name, **config_kwargs)
    env = GymEnvWrapper(env_name)
    s_dim = env.state_dim
    a_dim = env.action_dim
    a_low = env.action_low
    a_high = env.action_high

    def _actor():
        bb = MLP(input_dim=s_dim, hidden_dims=hidden_dims)
        return TD3Actor(bb, DeterministicPolicyHead(bb.output_dim, a_dim, a_low, a_high))

    def _critic():
        bb = MLP(input_dim=s_dim, hidden_dims=hidden_dims)
        return TD3Critic(bb, DoubleQCriticHead(bb.output_dim, a_dim, hidden_dims[-1]))

    actor, actor_target = _actor(), _actor()
    critic, critic_target = _critic(), _critic()

    agent = TD3Agent(
        actor=actor, actor_target=actor_target,
        critic=critic, critic_target=critic_target,
        actor_optimizer=torch.optim.Adam(actor.parameters(), lr=config.learning_rate),
        critic_optimizer=torch.optim.Adam(critic.parameters(), lr=config.learning_rate),
        replay_buffer=ReplayBuffer(config.buffer_size, config.device),
        config=config,
        action_dim=a_dim,
        action_low=a_low,
        action_high=a_high,
    )

    if logger is None:
        loggers: list = [ConsoleLogger(log_interval=config.log_interval)]
        if plot_path:
            loggers.append(MatplotlibLogger(save_path=plot_path))
        logger = CompositeLogger(loggers) if len(loggers) > 1 else loggers[0]

    return Trainer(agent, env, config, logger, save_path=save_path)
