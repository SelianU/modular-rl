from typing import Iterable, Union

import torch
import torch.nn as nn


LossLike = Union[str, nn.Module]
OptimizerLike = Union[str, type, torch.optim.Optimizer]


def make_loss(loss: LossLike) -> nn.Module:
    """
    Build a loss module from a beginner-friendly name or return an existing loss.

    Supported names:
    - "cross_entropy": classification with class-index targets.
    - "mse": regression or value fitting.
    - "smooth_l1": robust regression, commonly used by DQN-style updates.
    """
    if isinstance(loss, nn.Module):
        return loss

    if loss == "cross_entropy":
        return nn.CrossEntropyLoss()
    if loss == "mse":
        return nn.MSELoss()
    if loss == "smooth_l1":
        return nn.SmoothL1Loss()

    raise ValueError(
        f"Unknown loss '{loss}'. Available: ['cross_entropy', 'mse', 'smooth_l1']"
    )


def make_optimizer(
    optimizer: OptimizerLike,
    parameters: Iterable[torch.nn.Parameter],
    learning_rate: float = 1e-3,
    **optimizer_kwargs,
) -> torch.optim.Optimizer:
    """
    Build an optimizer from a name/class or return an existing optimizer.

    Supported names:
    - "adam"
    - "sgd"
    - "rmsprop"
    """
    if isinstance(optimizer, torch.optim.Optimizer):
        return optimizer

    optimizer_cls = optimizer
    if isinstance(optimizer, str):
        optimizer_cls = _optimizer_classes().get(optimizer)
        if optimizer_cls is None:
            raise ValueError(
                f"Unknown optimizer '{optimizer}'. "
                "Available: ['adam', 'sgd', 'rmsprop']"
            )

    return optimizer_cls(parameters, lr=learning_rate, **optimizer_kwargs)


def _optimizer_classes() -> dict:
    return {
        "adam": torch.optim.Adam,
        "sgd": torch.optim.SGD,
        "rmsprop": torch.optim.RMSprop,
    }
