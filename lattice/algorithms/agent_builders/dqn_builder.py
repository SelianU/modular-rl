import copy
from typing import Any, Dict

import torch.nn as nn

from lattice.networks import QHead
from lattice.networks.encoders import MLP, RNN

from ..agent_context import AgentBuildContext
from ..agents import DQNAgent
from ..models import QNetwork
from .base import BaseAgentBuilder


class DQNAgentBuilder(BaseAgentBuilder):
    """Build DQN agents from shape/model specs."""

    def build(self, context: AgentBuildContext, spec: Dict[str, Any]) -> DQNAgent:
        if self._has_prebuilt_dqn_model(spec):
            return self._build_custom_agent(context, spec)

        hidden_dims = self._hidden_dims(spec, default=[128, 128])
        backbone_type = self._backbone_type(spec, default="mlp")

        if backbone_type == "mlp":
            q_backbone, target_backbone = self._build_mlp_backbones(context, hidden_dims)
            replay_buffer = self.registry.build_buffer(
                self._buffer_type(spec, "replay"),
                capacity=context.config.buffer_size,
                device=context.config.device,
            )
            criterion = nn.SmoothL1Loss()
            is_recurrent = False
        elif backbone_type == "rnn":
            q_backbone, target_backbone = self._build_rnn_backbones(context, hidden_dims)
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
        return self._make_agent(
            context=context,
            spec=spec,
            q_network=q_network,
            target_q_network=target_q_network,
            replay_buffer=replay_buffer,
            criterion=criterion,
            is_recurrent=is_recurrent,
        )

    def _build_custom_agent(self, context: AgentBuildContext, spec: Dict[str, Any]) -> DQNAgent:
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
        return self._make_agent(
            context=context,
            spec=spec,
            q_network=q_network,
            target_q_network=target_q_network,
            replay_buffer=replay_buffer,
            criterion=criterion,
            is_recurrent=is_recurrent,
        )

    def _make_agent(
        self,
        context: AgentBuildContext,
        spec: Dict[str, Any],
        q_network,
        target_q_network,
        replay_buffer,
        criterion,
        is_recurrent: bool,
    ) -> DQNAgent:
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

    def _build_mlp_backbones(self, context: AgentBuildContext, hidden_dims):
        input_dim = self._require_state_dim(context, "mlp")
        return (
            MLP(input_dim=input_dim, hidden_dims=hidden_dims),
            MLP(input_dim=input_dim, hidden_dims=hidden_dims),
        )

    def _build_rnn_backbones(self, context: AgentBuildContext, hidden_dims):
        input_dim = self._require_state_dim(context, "rnn")
        return (
            RNN(
                base_backbone=MLP(input_dim=input_dim, hidden_dims=hidden_dims),
                rnn_type="LSTM",
                rnn_hidden_dim=hidden_dims[-1],
            ),
            RNN(
                base_backbone=MLP(input_dim=input_dim, hidden_dims=hidden_dims),
                rnn_type="LSTM",
                rnn_hidden_dim=hidden_dims[-1],
            ),
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

    @staticmethod
    def _has_prebuilt_dqn_model(spec: Dict[str, Any]) -> bool:
        model_spec = spec.get("model")
        return isinstance(model_spec, nn.Module) or (
            isinstance(model_spec, dict) and "q_network" in model_spec
        )
