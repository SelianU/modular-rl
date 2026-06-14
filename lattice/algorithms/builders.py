from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch.nn as nn

from lattice.networks import (
    CategoricalPolicyHead,
    DeterministicPolicyHead,
    DiagonalGaussianHead,
    DoubleQCriticHead,
    GaussianPolicyHead,
    QHead,
    ValueHead,
)
from lattice.networks.encoders import MLP, RNN

from .agents import DQNAgent, PPOAgent, SACAgent, TD3Agent
from .models import QNetwork, PPOActor, PPOCritic, SACActor, TD3Actor, TD3Critic


@dataclass
class AgentBuildContext:
    """Shape and action-space information needed to build an RL agent."""

    algorithm: str
    state_dim: int
    action_dim: int
    config: Any
    is_continuous: bool = False
    action_low: float = -1.0
    action_high: float = 1.0


class AgentBuilder:
    """Build RL agents from plain specs without requiring an environment."""

    def __init__(self, registry=None):
        if registry is None:
            from lattice.training import Registry

            registry = Registry
        self.registry = registry

    def build_agent(
        self,
        spec: Dict[str, Any],
        context: Optional[AgentBuildContext] = None,
    ):
        spec = dict(spec)
        algorithm = spec.get("algorithm", "dqn")
        if algorithm not in self.registry.list_configs():
            raise KeyError(
                f"Algorithm '{algorithm}' not found. Available: {self.registry.list_configs()}"
            )

        if context is None:
            config = self._build_config(algorithm, spec.get("config", {}))
            context = self._build_context(algorithm, spec, config)

        agent_builders = {
            "dqn": self._build_dqn_agent,
            "sac": self._build_sac_agent,
            "ppo": self._build_ppo_agent,
            "td3": self._build_td3_agent,
        }
        return agent_builders[algorithm](context, spec)

    def _build_config(self, algorithm: str, config_spec: Dict[str, Any]):
        config_spec = dict(config_spec)
        config_spec.setdefault("env_name", self._default_env_name(algorithm))
        return self.registry.build_config(algorithm, **config_spec)

    @staticmethod
    def _build_context(algorithm: str, spec: Dict[str, Any], config) -> AgentBuildContext:
        missing_keys = [key for key in ("state_dim", "action_dim") if key not in spec]
        if missing_keys:
            raise ValueError(
                "Agent-only construction requires explicit shape information: "
                f"missing {missing_keys}. Provide state_dim and action_dim."
            )

        return AgentBuildContext(
            algorithm=algorithm,
            state_dim=spec["state_dim"],
            action_dim=spec["action_dim"],
            config=config,
            is_continuous=spec.get("is_continuous", False),
            action_low=spec.get("action_low", -1.0),
            action_high=spec.get("action_high", 1.0),
        )

    def _build_dqn_agent(self, context: AgentBuildContext, spec: Dict[str, Any]) -> DQNAgent:
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

    def _build_sac_agent(self, context: AgentBuildContext, spec: Dict[str, Any]) -> SACAgent:
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

    def _build_ppo_agent(self, context: AgentBuildContext, spec: Dict[str, Any]) -> PPOAgent:
        hidden_dims = self._hidden_dims(spec, default=[64, 64])
        actor_backbone = self._mlp(context.state_dim, hidden_dims)
        actor_head = (
            DiagonalGaussianHead(actor_backbone.output_dim, context.action_dim)
            if context.is_continuous
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

    def _build_td3_agent(self, context: AgentBuildContext, spec: Dict[str, Any]) -> TD3Agent:
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

    def _build_backbone_from_spec(self, context: AgentBuildContext, spec: Dict[str, Any], hidden_dims):
        backbone_spec = dict(spec.get("model", {}).get("backbone", {}))
        backbone_type = backbone_spec.pop("type", "mlp")
        backbone_spec.setdefault("input_dim", context.state_dim)
        backbone_spec.setdefault("hidden_dims", hidden_dims)
        return self.registry.build_backbone(backbone_type, **backbone_spec)

    def _double_q_critic(self, context: AgentBuildContext, hidden_dims):
        backbone = self._mlp(context.state_dim, hidden_dims)
        return TD3Critic(
            backbone,
            DoubleQCriticHead(backbone.output_dim, context.action_dim, hidden_dims[-1]),
        )

    def _td3_actor(self, context: AgentBuildContext, hidden_dims) -> TD3Actor:
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

    def _optimizer(self, params, spec: Dict[str, Any], learning_rate: float, key: str = "optimizer"):
        optimizer_spec = dict(spec.get(key, spec.get("optimizer", {})))
        optimizer_type = optimizer_spec.pop("type", "adam")
        optimizer_spec.setdefault("lr", learning_rate)
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
    def _require_continuous(context: AgentBuildContext, algorithm: str) -> None:
        if not context.is_continuous:
            raise ValueError(f"{algorithm} requires a continuous action space.")


def build_agent(
    spec: Dict[str, Any],
    context: Optional[AgentBuildContext] = None,
    registry=None,
):
    return AgentBuilder(registry=registry).build_agent(spec, context=context)
