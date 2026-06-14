import copy
from typing import Any, Dict

from lattice.networks import DeterministicPolicyHead, DoubleQCriticHead

from ..agent_context import AgentBuildContext
from ..agents import TD3Agent
from ..models import TD3Actor, TD3Critic
from .base import BaseAgentBuilder


class TD3AgentBuilder(BaseAgentBuilder):
    """Build TD3 agents from shape/model specs."""

    def build(self, context: AgentBuildContext, spec: Dict[str, Any]) -> TD3Agent:
        self._require_continuous(context, "td3")
        if self._has_prebuilt_actor_critic(spec):
            return self._build_custom_agent(context, spec)

        hidden_dims = self._hidden_dims(spec, default=[256, 256])
        actor = self._td3_actor(context, hidden_dims)
        actor_target = self._td3_actor(context, hidden_dims)
        critic = self._double_q_critic(context, hidden_dims)
        critic_target = self._double_q_critic(context, hidden_dims)
        return self._make_agent(context, spec, actor, actor_target, critic, critic_target)

    def _build_custom_agent(self, context: AgentBuildContext, spec: Dict[str, Any]) -> TD3Agent:
        model_spec = spec["model"]
        actor = model_spec["actor"]
        critic = model_spec["critic"]
        actor_target = model_spec.get("actor_target") or copy.deepcopy(actor)
        critic_target = model_spec.get("critic_target") or copy.deepcopy(critic)
        return self._make_agent(context, spec, actor, actor_target, critic, critic_target)

    def _make_agent(self, context, spec, actor, actor_target, critic, critic_target) -> TD3Agent:
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

    @staticmethod
    def _has_prebuilt_actor_critic(spec: Dict[str, Any]) -> bool:
        model_spec = spec.get("model")
        return isinstance(model_spec, dict) and "actor" in model_spec and "critic" in model_spec
