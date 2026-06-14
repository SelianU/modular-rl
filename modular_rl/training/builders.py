from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch.nn as nn

from modular_rl.algorithms import (
    DQNAgent,
    PPOAgent,
    SACAgent,
    TD3Agent,
    QNetwork,
    PPOActor,
    PPOCritic,
    SACActor,
    TD3Actor,
    TD3Critic,
)
from modular_rl.networks import (
    CategoricalPolicyHead,
    DeterministicPolicyHead,
    DiagonalGaussianHead,
    DoubleQCriticHead,
    GaussianPolicyHead,
    QHead,
    ValueHead,
)
from modular_rl.networks.encoders import MLP, RNN

from .logger import BaseLogger, CompositeLogger
from .registry import Registry
from .rl_trainer import RLTrainer


@dataclass
class BuildContext:
    algorithm: str
    env: Any
    config: Any

    @property
    def state_dim(self):
        return self.env.state_dim

    @property
    def action_dim(self) -> int:
        return self.env.action_dim

    @property
    def action_low(self) -> float:
        return getattr(self.env, "action_low", -1.0) or -1.0

    @property
    def action_high(self) -> float:
        return getattr(self.env, "action_high", 1.0) or 1.0


class ExperimentBuilder:
    """Build RLTrainer instances from plain experiment dictionaries."""

    def __init__(self, registry: type = Registry):
        self.registry = registry

    def build_trainer(self, spec: Dict[str, Any]) -> RLTrainer:
        spec = dict(spec)
        algorithm = spec.get("algorithm", "dqn")
        if algorithm not in self.registry.list_configs():
            raise KeyError(
                f"Algorithm '{algorithm}' not found. Available: {self.registry.list_configs()}"
            )

        environment = self._build_env(spec.get("env", {}), algorithm)
        algorithm_config = self._build_config(algorithm, environment, spec.get("config", {}))
        context = BuildContext(algorithm=algorithm, env=environment, config=algorithm_config)
        logger = self._build_logger(spec.get("logger"), algorithm_config, spec.get("plot_path"))

        agent_builders = {
            "dqn": self._build_dqn_agent,
            "sac": self._build_sac_agent,
            "ppo": self._build_ppo_agent,
            "td3": self._build_td3_agent,
        }
        agent = agent_builders[algorithm](context, spec)
        return RLTrainer(agent, environment, algorithm_config, logger, save_path=spec.get("save_path"))

    def _build_env(self, env_spec: Dict[str, Any], algorithm: str):
        env_spec = dict(env_spec)
        env_type = env_spec.pop("type", "gym")
        env_name = env_spec.pop("name", None) or self._default_env_name(algorithm)

        if env_type == "gym":
            return self.registry.build_env(env_type, env_name=env_name, **env_spec)
        return self.registry.build_env(env_type, **env_spec)

    def _build_config(self, algorithm: str, env, config_spec: Dict[str, Any]):
        config_spec = dict(config_spec)
        gym_env = getattr(env, "_env", None)
        default_name = getattr(getattr(gym_env, "spec", None), "id", None)
        config_spec.setdefault("env_name", default_name or self._default_env_name(algorithm))
        return self.registry.build_config(algorithm, **config_spec)

    def _build_logger(
        self,
        logger_spec: Optional[Any],
        config,
        plot_path: Optional[str] = None,
    ) -> BaseLogger:
        if isinstance(logger_spec, BaseLogger):
            return logger_spec

        if logger_spec is None:
            loggers = [self.registry.build_logger("console", log_interval=config.log_interval)]
            if plot_path:
                loggers.append(self.registry.build_logger("matplotlib", save_path=plot_path))
            return CompositeLogger(loggers) if len(loggers) > 1 else loggers[0]

        if isinstance(logger_spec, list):
            return CompositeLogger([self._build_logger(item, config) for item in logger_spec])

        logger_spec = dict(logger_spec)
        logger_type = logger_spec.pop("type", "console")
        if logger_type == "console":
            logger_spec.setdefault("log_interval", config.log_interval)
        return self.registry.build_logger(logger_type, **logger_spec)

    def _build_dqn_agent(self, context: BuildContext, spec: Dict[str, Any]) -> DQNAgent:
        hidden_dims = self._hidden_dims(spec, default=[128, 128])
        backbone_type = self._backbone_type(spec, default="mlp")

        if backbone_type == "mlp":
            q_backbone = MLP(input_dim=context.state_dim, hidden_dims=hidden_dims)
            target_backbone = MLP(input_dim=context.state_dim, hidden_dims=hidden_dims)
            replay_buffer = self.registry.build_buffer(
                self._buffer_type(spec, "replay"),
                capacity=context.config.buffer_size,
                device=context.config.device,
            )
            criterion = nn.SmoothL1Loss()
            is_recurrent = False
        elif backbone_type == "rnn":
            q_backbone = RNN(
                base_backbone=MLP(input_dim=context.state_dim, hidden_dims=hidden_dims),
                rnn_type="LSTM",
                rnn_hidden_dim=hidden_dims[-1],
            )
            target_backbone = RNN(
                base_backbone=MLP(input_dim=context.state_dim, hidden_dims=hidden_dims),
                rnn_type="LSTM",
                rnn_hidden_dim=hidden_dims[-1],
            )
            replay_buffer = self.registry.build_buffer(
                self._buffer_type(spec, "sequence_replay"),
                capacity_episodes=context.config.buffer_size,
                device=context.config.device,
            )
            criterion = nn.MSELoss(reduction="none")
            is_recurrent = True
        else:
            q_backbone = self._build_backbone_from_spec(context, spec, hidden_dims)
            target_backbone = self._build_backbone_from_spec(context, spec, hidden_dims)
            replay_buffer = self.registry.build_buffer(
                self._buffer_type(spec, "replay"),
                capacity=context.config.buffer_size,
                device=context.config.device,
            )
            criterion = nn.SmoothL1Loss()
            is_recurrent = False

        q_network = QNetwork(q_backbone, QHead(q_backbone.output_dim, context.action_dim))
        target_q_network = QNetwork(target_backbone, QHead(target_backbone.output_dim, context.action_dim))
        return DQNAgent(
            q_network=q_network,
            target_network=target_q_network,
            optimizer=self._optimizer(q_network.parameters(), spec, context.config.learning_rate),
            criterion=criterion,
            replay_buffer=replay_buffer,
            config=context.config,
            action_dim=context.action_dim,
            is_recurrent=is_recurrent,
        )

    def _build_sac_agent(self, context: BuildContext, spec: Dict[str, Any]) -> SACAgent:
        self._require_continuous(context, "sac")
        hidden_dims = self._hidden_dims(spec, default=[256, 256])

        actor_backbone = self._mlp(context.state_dim, hidden_dims)
        actor = SACActor(
            actor_backbone,
            GaussianPolicyHead(actor_backbone.output_dim, context.action_dim, context.action_low, context.action_high),
        )
        critic = self._double_q_critic(context, hidden_dims)
        critic_target = self._double_q_critic(context, hidden_dims)
        return SACAgent(
            actor=actor,
            critic=critic,
            critic_target=critic_target,
            actor_optimizer=self._optimizer(actor.parameters(), spec, context.config.actor_lr, "actor_optimizer"),
            critic_optimizer=self._optimizer(critic.parameters(), spec, context.config.critic_lr, "critic_optimizer"),
            replay_buffer=self.registry.build_buffer(
                self._buffer_type(spec, "replay"),
                capacity=context.config.buffer_size,
                device=context.config.device,
            ),
            config=context.config,
            action_dim=context.action_dim,
        )

    def _build_ppo_agent(self, context: BuildContext, spec: Dict[str, Any]) -> PPOAgent:
        hidden_dims = self._hidden_dims(spec, default=[64, 64])
        actor_backbone = self._mlp(context.state_dim, hidden_dims)
        actor_head = (
            DiagonalGaussianHead(actor_backbone.output_dim, context.action_dim)
            if context.env.is_continuous
            else CategoricalPolicyHead(actor_backbone.output_dim, context.action_dim)
        )
        actor = PPOActor(actor_backbone, actor_head)

        critic_backbone = self._mlp(context.state_dim, hidden_dims)
        critic = PPOCritic(critic_backbone, ValueHead(critic_backbone.output_dim))
        return PPOAgent(
            actor=actor,
            critic=critic,
            actor_optimizer=self._optimizer(actor.parameters(), spec, context.config.learning_rate, "actor_optimizer"),
            critic_optimizer=self._optimizer(critic.parameters(), spec, context.config.learning_rate, "critic_optimizer"),
            rollout_buffer=self.registry.build_buffer(
                self._buffer_type(spec, "rollout"),
                n_steps=context.config.n_steps,
                device=context.config.device,
                gamma=context.config.gamma,
                gae_lambda=context.config.gae_lambda,
            ),
            config=context.config,
        )

    def _build_td3_agent(self, context: BuildContext, spec: Dict[str, Any]) -> TD3Agent:
        self._require_continuous(context, "td3")
        hidden_dims = self._hidden_dims(spec, default=[256, 256])

        actor = self._td3_actor(context, hidden_dims)
        actor_target = self._td3_actor(context, hidden_dims)
        critic = self._double_q_critic(context, hidden_dims)
        critic_target = self._double_q_critic(context, hidden_dims)
        return TD3Agent(
            actor=actor,
            actor_target=actor_target,
            critic=critic,
            critic_target=critic_target,
            actor_optimizer=self._optimizer(actor.parameters(), spec, context.config.learning_rate, "actor_optimizer"),
            critic_optimizer=self._optimizer(critic.parameters(), spec, context.config.learning_rate, "critic_optimizer"),
            replay_buffer=self.registry.build_buffer(
                self._buffer_type(spec, "replay"),
                capacity=context.config.buffer_size,
                device=context.config.device,
            ),
            config=context.config,
            action_dim=context.action_dim,
            action_low=context.action_low,
            action_high=context.action_high,
        )

    def _build_backbone_from_spec(self, context: BuildContext, spec: Dict[str, Any], hidden_dims):
        backbone_spec = dict(spec.get("model", {}).get("backbone", {}))
        backbone_type = backbone_spec.pop("type", "mlp")
        backbone_spec.setdefault("input_dim", context.state_dim)
        backbone_spec.setdefault("hidden_dims", hidden_dims)
        return self.registry.build_backbone(backbone_type, **backbone_spec)

    def _double_q_critic(self, context: BuildContext, hidden_dims):
        backbone = self._mlp(context.state_dim, hidden_dims)
        return TD3Critic(
            backbone,
            DoubleQCriticHead(backbone.output_dim, context.action_dim, hidden_dims[-1]),
        )

    def _td3_actor(self, context: BuildContext, hidden_dims) -> TD3Actor:
        backbone = self._mlp(context.state_dim, hidden_dims)
        return TD3Actor(
            backbone,
            DeterministicPolicyHead(
                backbone.output_dim,
                context.action_dim,
                context.action_low,
                context.action_high,
            ),
        )

    def _optimizer(self, params, spec: Dict[str, Any], lr: float, key: str = "optimizer"):
        optimizer_spec = dict(spec.get(key, spec.get("optimizer", {})))
        optimizer_type = optimizer_spec.pop("type", "adam")
        optimizer_spec.setdefault("lr", lr)
        return self.registry.build_optimizer(optimizer_type, params, **optimizer_spec)

    def _hidden_dims(self, spec: Dict[str, Any], default):
        return spec.get("model", {}).get("backbone", {}).get("hidden_dims", default)

    def _backbone_type(self, spec: Dict[str, Any], default: str):
        return spec.get("model", {}).get("backbone", {}).get("type", default)

    def _buffer_type(self, spec: Dict[str, Any], default: str) -> str:
        return spec.get("buffer", {}).get("type", default)

    @staticmethod
    def _mlp(input_dim, hidden_dims):
        return MLP(input_dim=input_dim, hidden_dims=hidden_dims)

    @staticmethod
    def _default_env_name(algorithm: str) -> str:
        return {
            "dqn": "CartPole-v1",
            "ppo": "CartPole-v1",
            "sac": "Pendulum-v1",
            "td3": "Pendulum-v1",
        }[algorithm]

    @staticmethod
    def _require_continuous(context: BuildContext, algorithm: str) -> None:
        if not context.env.is_continuous:
            raise ValueError(f"{algorithm} requires a continuous action space.")


def build_trainer(spec: Dict[str, Any]) -> RLTrainer:
    return ExperimentBuilder().build_trainer(spec)
