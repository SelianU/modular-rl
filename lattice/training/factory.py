"""
One-liner factory functions for the most common training setups.

These helpers return RLTrainer instances and delegate assembly to the
config-driven builder. That keeps all experiment construction logic in one
place.
"""
from typing import List, Optional

from .builders import build_trainer
from .logger import BaseLogger
from .rl_trainer import RLTrainer


def _model_spec(backbone: str = "mlp", hidden_dims: Optional[List[int]] = None) -> dict:
    backbone_config = {"type": backbone}
    if hidden_dims is not None:
        backbone_config["hidden_dims"] = hidden_dims
    return {"backbone": backbone_config}


def _trainer_spec(
    algorithm: str,
    env_name: str,
    hidden_dims: Optional[List[int]],
    save_path: Optional[str],
    plot_path: Optional[str],
    logger: Optional[BaseLogger],
    config_kwargs: dict,
    backbone: str = "mlp",
) -> dict:
    trainer_spec = {
        "algorithm": algorithm,
        "env": {"type": "gym", "name": env_name},
        "model": _model_spec(backbone=backbone, hidden_dims=hidden_dims),
        "config": config_kwargs,
        "save_path": save_path,
        "plot_path": plot_path,
    }
    if logger is not None:
        trainer_spec["logger"] = logger
    return trainer_spec


def quick_dqn(
    env_name: str,
    backbone: str = "mlp",
    hidden_dims: Optional[List[int]] = None,
    save_path: Optional[str] = None,
    plot_path: Optional[str] = None,
    logger: Optional[BaseLogger] = None,
    **config_kwargs,
) -> RLTrainer:
    """
    Build and return an RLTrainer for a DQN experiment in one call.
    """
    return build_trainer(
        _trainer_spec(
            "dqn",
            env_name,
            hidden_dims,
            save_path,
            plot_path,
            logger,
            config_kwargs,
            backbone=backbone,
        )
    )


def quick_sac(
    env_name: str,
    hidden_dims: Optional[List[int]] = None,
    save_path: Optional[str] = None,
    plot_path: Optional[str] = None,
    logger: Optional[BaseLogger] = None,
    **config_kwargs,
) -> RLTrainer:
    """Build and return an RLTrainer for a SAC experiment in one call."""
    return build_trainer(
        _trainer_spec("sac", env_name, hidden_dims, save_path, plot_path, logger, config_kwargs)
    )


def quick_ppo(
    env_name: str,
    hidden_dims: Optional[List[int]] = None,
    save_path: Optional[str] = None,
    plot_path: Optional[str] = None,
    logger: Optional[BaseLogger] = None,
    **config_kwargs,
) -> RLTrainer:
    """Build and return an RLTrainer for a PPO experiment in one call."""
    return build_trainer(
        _trainer_spec("ppo", env_name, hidden_dims, save_path, plot_path, logger, config_kwargs)
    )


def quick_td3(
    env_name: str,
    hidden_dims: Optional[List[int]] = None,
    save_path: Optional[str] = None,
    plot_path: Optional[str] = None,
    logger: Optional[BaseLogger] = None,
    **config_kwargs,
) -> RLTrainer:
    """Build and return an RLTrainer for a TD3 experiment in one call."""
    return build_trainer(
        _trainer_spec("td3", env_name, hidden_dims, save_path, plot_path, logger, config_kwargs)
    )
