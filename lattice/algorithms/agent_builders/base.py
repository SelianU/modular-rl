from typing import Any, Dict, Optional, Tuple

from lattice.networks.encoders import CNN, MLP, Transformer

from ..agent_context import AgentBuildContext


class BaseAgentBuilder:
    """Shared helpers for algorithm-specific agent builders."""

    def __init__(self, registry=None):
        if registry is None:
            from lattice.training import Registry

            registry = Registry
        self.registry = registry

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
    def _require_continuous(context: AgentBuildContext, algorithm: str) -> None:
        if not context.is_continuous:
            raise ValueError(f"{algorithm} requires a continuous action space.")
