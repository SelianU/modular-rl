from typing import Any, Dict, List, Optional

from .builders import build_agent


def make_dqn_agent(
    state_dim: Optional[int] = None,
    action_dim: int = 1,
    input_shape: Optional[tuple] = None,
    backbone: str = "mlp",
    hidden_dims: Optional[List[int]] = None,
    model=None,
    **config_kwargs,
):
    """Create a DQN agent without writing a full spec dictionary."""

    return build_agent(
        _agent_spec(
            algorithm="dqn",
            state_dim=state_dim,
            input_shape=input_shape,
            action_dim=action_dim,
            backbone=backbone,
            hidden_dims=hidden_dims,
            model=model,
            config_kwargs=config_kwargs,
        )
    )


def make_ppo_agent(
    state_dim: int,
    action_dim: int,
    hidden_dims: Optional[List[int]] = None,
    model: Optional[Dict[str, Any]] = None,
    is_continuous: bool = False,
    action_low: float = -1.0,
    action_high: float = 1.0,
    **config_kwargs,
):
    """Create a PPO agent for discrete or continuous action spaces."""

    return build_agent(
        _agent_spec(
            algorithm="ppo",
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            model=model,
            is_continuous=is_continuous,
            action_low=action_low,
            action_high=action_high,
            config_kwargs=config_kwargs,
        )
    )


def make_sac_agent(
    state_dim: int,
    action_dim: int,
    hidden_dims: Optional[List[int]] = None,
    model: Optional[Dict[str, Any]] = None,
    action_low: float = -1.0,
    action_high: float = 1.0,
    **config_kwargs,
):
    """Create a SAC agent for continuous action spaces."""

    return build_agent(
        _agent_spec(
            algorithm="sac",
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            model=model,
            is_continuous=True,
            action_low=action_low,
            action_high=action_high,
            config_kwargs=config_kwargs,
        )
    )


def make_td3_agent(
    state_dim: int,
    action_dim: int,
    hidden_dims: Optional[List[int]] = None,
    model: Optional[Dict[str, Any]] = None,
    action_low: float = -1.0,
    action_high: float = 1.0,
    **config_kwargs,
):
    """Create a TD3 agent for continuous action spaces."""

    return build_agent(
        _agent_spec(
            algorithm="td3",
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            model=model,
            is_continuous=True,
            action_low=action_low,
            action_high=action_high,
            config_kwargs=config_kwargs,
        )
    )


def _agent_spec(
    algorithm: str,
    action_dim: int,
    config_kwargs: Dict[str, Any],
    state_dim: Optional[int] = None,
    input_shape: Optional[tuple] = None,
    backbone: str = "mlp",
    hidden_dims: Optional[List[int]] = None,
    model=None,
    is_continuous: bool = False,
    action_low: float = -1.0,
    action_high: float = 1.0,
) -> Dict[str, Any]:
    spec: Dict[str, Any] = {
        "algorithm": algorithm,
        "action_dim": action_dim,
        "config": dict(config_kwargs),
    }
    if state_dim is not None:
        spec["state_dim"] = state_dim
    if input_shape is not None:
        spec["input_shape"] = input_shape
    if is_continuous:
        spec["is_continuous"] = True
        spec["action_low"] = action_low
        spec["action_high"] = action_high

    if model is not None:
        spec["model"] = model
    else:
        backbone_config: Dict[str, Any] = {"type": backbone}
        if hidden_dims is not None:
            backbone_config["hidden_dims"] = hidden_dims
        spec["model"] = {"backbone": backbone_config}

    return spec
