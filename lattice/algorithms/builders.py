import copy
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

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
from lattice.networks.encoders import CNN, MLP, RNN, Transformer

from .agents import DQNAgent, PPOAgent, SACAgent, TD3Agent
from .models import QNetwork, PPOActor, PPOCritic, SACActor, TD3Actor, TD3Critic


@dataclass
class AgentBuildContext:
    """Shape and action-space information needed to build an RL agent."""

    algorithm: str
    action_dim: int
    config: Any
    state_dim: Optional[int] = None
    input_shape: Optional[Tuple[int, ...]] = None
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
        if "action_dim" not in spec:
            raise ValueError(
                "Agent-only construction requires explicit shape information: "
                "missing ['action_dim']. Provide action_dim."
            )

        state_dim = spec.get("state_dim")
        input_shape = spec.get("input_shape")
        if state_dim is None and input_shape is None and not AgentBuilder._has_prebuilt_model(spec):
            raise ValueError(
                "Agent-only construction requires either state_dim for vector "
                "states, input_shape for image/sequence states, or a prebuilt "
                "model module."
            )

        if input_shape is not None:
            input_shape = tuple(input_shape)
        elif state_dim is not None:
            input_shape = (state_dim,)

        return AgentBuildContext(
            algorithm=algorithm,
            action_dim=spec["action_dim"],
            config=config,
            state_dim=state_dim,
            input_shape=input_shape,
            is_continuous=spec.get("is_continuous", False),
            action_low=spec.get("action_low", -1.0),
            action_high=spec.get("action_high", 1.0),
        )

    def _build_dqn_agent(self, context: AgentBuildContext, spec: Dict[str, Any]) -> DQNAgent:
        if self._has_prebuilt_dqn_model(spec):
            return self._build_custom_dqn_agent(context, spec)

        hidden_dims = self._hidden_dims(spec, default=[128, 128])
        backbone_type = self._backbone_type(spec, default="mlp")

        if backbone_type == "mlp":
            input_dim = self._require_state_dim(context, "mlp")
            q_backbone = MLP(input_dim=input_dim, hidden_dims=hidden_dims)
            target_backbone = MLP(input_dim=input_dim, hidden_dims=hidden_dims)
            replay_buffer = self.registry.build_buffer(
                self._buffer_type(spec, "replay"),
                capacity=context.config.buffer_size,
                device=context.config.device,
            )
            criterion = nn.SmoothL1Loss()
            is_recurrent = False
        elif backbone_type == "rnn":
            input_dim = self._require_state_dim(context, "rnn")
            q_backbone = RNN(
                base_backbone=MLP(input_dim=input_dim, hidden_dims=hidden_dims),
                rnn_type="LSTM",
                rnn_hidden_dim=hidden_dims[-1],
            )
            target_backbone = RNN(
                base_backbone=MLP(input_dim=input_dim, hidden_dims=hidden_dims),
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
                self._buffer_type(
                    spec,
                    "sequence_replay" if backbone_type == "transformer" else "replay",
                ),
                **self._buffer_kwargs(context, backbone_type),
            )
            criterion = (
                nn.MSELoss(reduction="none")
                if backbone_type == "transformer"
                else nn.SmoothL1Loss()
            )
            is_recurrent = backbone_type == "transformer"

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

    def _build_custom_dqn_agent(self, context: AgentBuildContext, spec: Dict[str, Any]) -> DQNAgent:
        model_spec = spec.get("model", {})
        if isinstance(model_spec, nn.Module):
            q_network = model_spec
            target_q_network = copy.deepcopy(q_network)
            is_recurrent = bool(spec.get("is_recurrent", False))
        else:
            q_network = model_spec["q_network"]
            target_q_network = model_spec.get("target_q_network") or copy.deepcopy(q_network)
            is_recurrent = bool(model_spec.get("is_recurrent", spec.get("is_recurrent", False)))

        replay_buffer = self.registry.build_buffer(
            self._buffer_type(spec, "sequence_replay" if is_recurrent else "replay"),
            **self._buffer_kwargs(context, "transformer" if is_recurrent else "mlp"),
        )
        criterion = spec.get(
            "criterion",
            nn.MSELoss(reduction="none") if is_recurrent else nn.SmoothL1Loss(),
        )
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

    def _buffer_kwargs(self, context: AgentBuildContext, backbone_type: str) -> Dict[str, Any]:
        if backbone_type == "transformer":
            return {
                "capacity_episodes": context.config.buffer_size,
                "device": context.config.device,
            }
        return {
            "capacity": context.config.buffer_size,
            "device": context.config.device,
        }

    def _build_sac_agent(self, context: AgentBuildContext, spec: Dict[str, Any]) -> SACAgent:
        self._require_continuous(context, "sac")
        if self._has_prebuilt_actor_critic(spec):
            return self._build_custom_sac_agent(context, spec)

        hidden_dims = self._hidden_dims(spec, default=[256, 256])

        actor_backbone = self._mlp(self._require_state_dim(context, "sac"), hidden_dims)
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

    def _build_custom_sac_agent(self, context: AgentBuildContext, spec: Dict[str, Any]) -> SACAgent:
        model_spec = spec["model"]
        actor = model_spec["actor"]
        critic = model_spec["critic"]
        critic_target = model_spec.get("critic_target") or copy.deepcopy(critic)
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
        if self._has_prebuilt_actor_critic(spec):
            return self._build_custom_ppo_agent(context, spec)

        hidden_dims = self._hidden_dims(spec, default=[64, 64])
        input_dim = self._require_state_dim(context, "ppo")
        actor_backbone = self._mlp(input_dim, hidden_dims)
        actor_head = (
            DiagonalGaussianHead(actor_backbone.output_dim, context.action_dim)
            if context.is_continuous
            else CategoricalPolicyHead(actor_backbone.output_dim, context.action_dim)
        )
        actor = PPOActor(actor_backbone, actor_head)

        critic_backbone = self._mlp(input_dim, hidden_dims)
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

    def _build_custom_ppo_agent(self, context: AgentBuildContext, spec: Dict[str, Any]) -> PPOAgent:
        model_spec = spec["model"]
        actor = model_spec["actor"]
        critic = model_spec["critic"]
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
        if self._has_prebuilt_actor_critic(spec):
            return self._build_custom_td3_agent(context, spec)

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

    def _build_custom_td3_agent(self, context: AgentBuildContext, spec: Dict[str, Any]) -> TD3Agent:
        model_spec = spec["model"]
        actor = model_spec["actor"]
        critic = model_spec["critic"]
        actor_target = model_spec.get("actor_target") or copy.deepcopy(actor)
        critic_target = model_spec.get("critic_target") or copy.deepcopy(critic)
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

        if backbone_type == "cnn":
            backbone_spec.pop("hidden_dims", None)
            backbone_spec.setdefault("input_shape", self._require_input_shape(context, "cnn", rank=3))
            return CNN(**backbone_spec)

        if backbone_type == "transformer":
            backbone_spec.pop("hidden_dims", None)
            return self._build_transformer_backbone(context, backbone_spec, hidden_dims)

        backbone_spec.setdefault("input_dim", self._require_state_dim(context, backbone_type))
        backbone_spec.setdefault("hidden_dims", hidden_dims)
        return self.registry.build_backbone(backbone_type, **backbone_spec)

    def _build_transformer_backbone(
        self,
        context: AgentBuildContext,
        backbone_spec: Dict[str, Any],
        hidden_dims,
    ) -> Transformer:
        base_backbone_spec = dict(backbone_spec.pop("base_backbone", {}))
        input_shape = self._require_input_shape(context, "transformer")

        if base_backbone_spec:
            base_backbone_type = base_backbone_spec.pop("type", "mlp")
            base_backbone = self._build_base_backbone(
                base_backbone_type,
                base_backbone_spec,
                input_shape,
                hidden_dims,
            )
        else:
            base_backbone = self._default_transformer_base_backbone(input_shape, hidden_dims)

        if len(input_shape) >= 2:
            sequence_length = getattr(context.config, "sequence_length", 10)
            backbone_spec.setdefault("max_seq_len", max(input_shape[0], sequence_length))
        return Transformer(base_backbone=base_backbone, **backbone_spec)

    def _build_base_backbone(
        self,
        backbone_type: str,
        backbone_spec: Dict[str, Any],
        input_shape: Tuple[int, ...],
        hidden_dims,
    ):
        if backbone_type == "cnn":
            if len(input_shape) != 4:
                raise ValueError(
                    "Transformer with CNN base expects input_shape=(sequence_length, channels, height, width)."
                )
            backbone_spec.setdefault("input_shape", input_shape[1:])
            return CNN(**backbone_spec)

        if backbone_type == "mlp":
            input_dim = input_shape[-1] if len(input_shape) >= 2 else input_shape[0]
            backbone_spec.setdefault("input_dim", input_dim)
            backbone_spec.setdefault("hidden_dims", hidden_dims)
            return MLP(**backbone_spec)

        return self.registry.build_backbone(backbone_type, **backbone_spec)

    def _default_transformer_base_backbone(self, input_shape: Tuple[int, ...], hidden_dims):
        if len(input_shape) == 2:
            return MLP(input_dim=input_shape[1], hidden_dims=hidden_dims)
        if len(input_shape) == 4:
            return CNN(input_shape=input_shape[1:])
        raise ValueError(
            "Transformer expects input_shape=(sequence_length, feature_dim) "
            "or (sequence_length, channels, height, width)."
        )

    def _double_q_critic(self, context: AgentBuildContext, hidden_dims):
        backbone = self._mlp(self._require_state_dim(context, "critic"), hidden_dims)
        return TD3Critic(
            backbone,
            DoubleQCriticHead(backbone.output_dim, context.action_dim, hidden_dims[-1]),
        )

    def _td3_actor(self, context: AgentBuildContext, hidden_dims) -> TD3Actor:
        backbone = self._mlp(self._require_state_dim(context, "td3"), hidden_dims)
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
    def _has_prebuilt_model(spec: Dict[str, Any]) -> bool:
        model_spec = spec.get("model")
        if isinstance(model_spec, nn.Module):
            return True
        if not isinstance(model_spec, dict):
            return False
        return any(
            key in model_spec
            for key in (
                "q_network",
                "target_q_network",
                "actor",
                "critic",
                "actor_target",
                "critic_target",
            )
        )

    @staticmethod
    def _has_prebuilt_dqn_model(spec: Dict[str, Any]) -> bool:
        model_spec = spec.get("model")
        return isinstance(model_spec, nn.Module) or (
            isinstance(model_spec, dict) and "q_network" in model_spec
        )

    @staticmethod
    def _has_prebuilt_actor_critic(spec: Dict[str, Any]) -> bool:
        model_spec = spec.get("model")
        return isinstance(model_spec, dict) and "actor" in model_spec and "critic" in model_spec

    @staticmethod
    def _require_state_dim(context: AgentBuildContext, component_name: str) -> int:
        if context.state_dim is None:
            raise ValueError(
                f"{component_name} requires state_dim for vector states. "
                "Use input_shape with cnn or transformer backbones."
            )
        return context.state_dim

    @staticmethod
    def _require_input_shape(
        context: AgentBuildContext,
        component_name: str,
        rank: Optional[int] = None,
    ) -> Tuple[int, ...]:
        if context.input_shape is None:
            raise ValueError(f"{component_name} requires input_shape.")
        if rank is not None and len(context.input_shape) != rank:
            raise ValueError(
                f"{component_name} expects input_shape with {rank} values, "
                f"got {context.input_shape}."
            )
        return context.input_shape

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
