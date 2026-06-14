import copy
from typing import Any, Dict

from lattice.networks import DoubleQCriticHead, GaussianPolicyHead

from ..agent_context import AgentBuildContext
from ..agents import SACAgent
from ..models import SACActor, TD3Critic
from .base import BaseAgentBuilder


class SACAgentBuilder(BaseAgentBuilder):
    """Build SAC agents from shape/model specs."""

    def build(self, context: AgentBuildContext, spec: Dict[str, Any]) -> SACAgent:
        self._require_continuous(context, "sac")
        if self._has_prebuilt_actor_critic(spec):
            return self._build_custom_agent(context, spec)

        hidden_dims = self._hidden_dims(spec, default=[256, 256])
        actor_backbone = self._mlp(self._require_state_dim(context, "sac"), hidden_dims)
        actor = SACActor(
            actor_backbone,
            GaussianPolicyHead(actor_backbone.output_dim, context.action_dim, context.action_low, context.action_high),
        )
        critic = self._double_q_critic(context, hidden_dims)
        critic_target = self._double_q_critic(context, hidden_dims)
        return self._make_agent(context, spec, actor, critic, critic_target)

    def _build_custom_agent(self, context: AgentBuildContext, spec: Dict[str, Any]) -> SACAgent:
        model_spec = spec["model"]
        actor = model_spec["actor"]
        critic = model_spec["critic"]
        critic_target = model_spec.get("critic_target") or copy.deepcopy(critic)
        return self._make_agent(context, spec, actor, critic, critic_target)

    def _make_agent(self, context, spec, actor, critic, critic_target) -> SACAgent:
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

    def _double_q_critic(self, context: AgentBuildContext, hidden_dims):
        backbone = self._mlp(self._require_state_dim(context, "critic"), hidden_dims)
        return TD3Critic(
            backbone,
            DoubleQCriticHead(backbone.output_dim, context.action_dim, hidden_dims[-1]),
        )

    @staticmethod
    def _has_prebuilt_actor_critic(spec: Dict[str, Any]) -> bool:
        model_spec = spec.get("model")
        return isinstance(model_spec, dict) and "actor" in model_spec and "critic" in model_spec
