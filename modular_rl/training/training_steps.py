from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass(frozen=True)
class BatchMetrics:
    """Metrics returned by one model update or evaluation batch."""

    loss: float
    accuracy: Optional[float] = None
    num_samples: int = 0


def run_training_step(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str = "cpu",
    output_mode: str = "auto",
    gradient_clip_norm: Optional[float] = None,
) -> BatchMetrics:
    """
    Run one differentiable update.

    This is intentionally small: supervised trainers can call it directly, and
    RL agents can reuse the same pattern when their update method has already
    computed model inputs and learning targets.
    """
    model.train()
    inputs = inputs.to(device)
    targets = targets.to(device)

    outputs = model(inputs)
    outputs = select_training_outputs(outputs, output_mode=output_mode)
    loss = loss_fn(outputs, targets)

    optimizer.zero_grad()
    loss.backward()
    if gradient_clip_norm is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_norm)
    optimizer.step()

    return BatchMetrics(
        loss=loss.item(),
        accuracy=compute_accuracy(outputs, targets),
        num_samples=_batch_size(targets),
    )


def run_evaluation_step(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    loss_fn: nn.Module,
    device: str = "cpu",
    output_mode: str = "auto",
) -> BatchMetrics:
    """Run one no-gradient evaluation batch."""
    model.eval()
    inputs = inputs.to(device)
    targets = targets.to(device)

    with torch.no_grad():
        outputs = model(inputs)
        outputs = select_training_outputs(outputs, output_mode=output_mode)
        loss = loss_fn(outputs, targets)

    return BatchMetrics(
        loss=loss.item(),
        accuracy=compute_accuracy(outputs, targets),
        num_samples=_batch_size(targets),
    )


def select_training_outputs(
    outputs: torch.Tensor,
    output_mode: str = "auto",
) -> torch.Tensor:
    """
    Pick the tensor that should be compared with targets.

    - "auto": use last sequence step for 3D outputs, otherwise use outputs.
    - "all": use outputs exactly as returned by the model.
    - "last": use `outputs[:, -1, :]`.
    """
    if output_mode == "all":
        return outputs
    if output_mode == "last":
        return outputs[:, -1, :]
    if output_mode == "auto":
        if outputs.dim() == 3:
            return outputs[:, -1, :]
        return outputs

    raise ValueError("output_mode must be one of: 'auto', 'all', 'last'")


def compute_accuracy(outputs: torch.Tensor, targets: torch.Tensor) -> Optional[float]:
    """Return classification accuracy when shapes look like class logits."""
    if outputs.dim() < 2:
        return None
    if targets.dim() != outputs.dim() - 1:
        return None

    predictions = outputs.argmax(dim=-1)
    return (predictions == targets).float().mean().item()


def _batch_size(targets: torch.Tensor) -> int:
    return int(targets.shape[0]) if targets.dim() > 0 else 1
