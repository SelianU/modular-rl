from typing import Any, Dict, Optional

from .agent_builders import (
    DQNAgentBuilder,
    PPOAgentBuilder,
    SACAgentBuilder,
    TD3AgentBuilder,
)
from .agent_context import AgentBuildContext, has_prebuilt_model


DEFAULT_AGENT_BUILDER_CLASSES = {
    "dqn": DQNAgentBuilder,
    "sac": SACAgentBuilder,
    "ppo": PPOAgentBuilder,
    "td3": TD3AgentBuilder,
}


class AgentBuilder:
    """Dispatch plain agent specs to algorithm-specific builders."""

    def __init__(self, registry=None):
        if registry is None:
            from lattice.training import Registry

            registry = Registry
        self.registry = registry
        self.agent_builder_classes = dict(DEFAULT_AGENT_BUILDER_CLASSES)
        self.agent_builder_classes.update(
            {
                name: self.registry.get_agent_builder_class(name)
                for name in self.registry.list_agent_builders()
            }
        )

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
        if algorithm not in self.agent_builder_classes:
            raise KeyError(f"Algorithm '{algorithm}' has no registered agent builder.")

        if context is None:
            config = self._build_config(algorithm, spec.get("config", {}))
            context = self._build_context(algorithm, spec, config)

        return self._build_algorithm_builder(algorithm).build(context, spec)

    def _build_config(self, algorithm: str, config_spec: Dict[str, Any]):
        config_spec = dict(config_spec)
        default_env_name = self._default_env_name(algorithm)
        if default_env_name is not None:
            config_spec.setdefault("env_name", default_env_name)
        return self.registry.build_config(algorithm, **config_spec)

    def _build_algorithm_builder(self, algorithm: str):
        builder_cls = self.agent_builder_classes[algorithm]
        return builder_cls(registry=self.registry)

    @staticmethod
    def _build_context(algorithm: str, spec: Dict[str, Any], config) -> AgentBuildContext:
        if "action_dim" not in spec:
            raise ValueError(
                "Agent-only construction requires explicit shape information: "
                "missing ['action_dim']. Provide action_dim."
            )

        state_dim = spec.get("state_dim")
        input_shape = spec.get("input_shape")
        if state_dim is None and input_shape is None and not has_prebuilt_model(spec):
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

    @staticmethod
    def _default_env_name(algorithm: str) -> str:
        return {
            "dqn": "CartPole-v1",
            "ppo": "CartPole-v1",
            "sac": "Pendulum-v1",
            "td3": "Pendulum-v1",
        }.get(algorithm)


def build_agent(
    spec: Dict[str, Any],
    context: Optional[AgentBuildContext] = None,
    registry=None,
):
    return AgentBuilder(registry=registry).build_agent(spec, context=context)
