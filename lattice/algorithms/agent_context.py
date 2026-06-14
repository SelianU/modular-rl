from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch.nn as nn


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


def has_prebuilt_model(spec: Dict[str, Any]) -> bool:
    """Return True when an agent spec already provides model modules."""

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
