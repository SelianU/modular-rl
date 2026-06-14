from typing import Any, Dict

from lattice.networks import CategoricalPolicyHead, DiagonalGaussianHead, ValueHead

from ..agent_context import AgentBuildContext
from ..agents import PPOAgent
from ..models import PPOActor, PPOCritic
from .base import BaseAgentBuilder


class PPOAgentBuilder(BaseAgentBuilder):
    """Build PPO agents from shape/model specs."""

    def build(self, context: AgentBuildContext, spec: Dict[str, Any]) -> PPOAgent:
        if self._has_prebuilt_actor_critic(spec):
            return self._build_custom_agent(context, spec)

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
        return self._make_agent(context, spec, actor, critic)

    def _build_custom_agent(self, context: AgentBuildContext, spec: Dict[str, Any]) -> PPOAgent:
        model_spec = spec["model"]
        return self._make_agent(context, spec, model_spec["actor"], model_spec["critic"])

    def _make_agent(self, context, spec, actor, critic) -> PPOAgent:
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

    @staticmethod
    def _has_prebuilt_actor_critic(spec: Dict[str, Any]) -> bool:
        model_spec = spec.get("model")
        return isinstance(model_spec, dict) and "actor" in model_spec and "critic" in model_spec
