from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn

from .optimizers import LossLike, OptimizerLike, make_loss, make_optimizer
from .training_steps import BatchMetrics, run_evaluation_step, run_training_step


@dataclass(frozen=True)
class SupervisedTrainingConfig:
    """Configuration for beginner-friendly supervised training."""

    loss: LossLike = "cross_entropy"
    optimizer: OptimizerLike = "adam"
    learning_rate: float = 1e-3
    epochs: int = 10
    device: str = "cpu"
    output_mode: str = "auto"
    gradient_clip_norm: Optional[float] = None
    log_interval: int = 1
    optimizer_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SupervisedTrainingContext:
    """Context passed to custom training steps."""

    loss_fn: nn.Module
    optimizer: torch.optim.Optimizer
    device: str
    output_mode: str
    gradient_clip_norm: Optional[float]
    epoch: int = 0
    batch_index: int = 0


@dataclass(frozen=True)
class EpochMetrics:
    """Metrics collected after one epoch."""

    epoch: int
    train_loss: float
    train_accuracy: Optional[float] = None
    validation_loss: Optional[float] = None
    validation_accuracy: Optional[float] = None


@dataclass
class SupervisedTrainingHistory:
    """Epoch-level training history."""

    epochs: List[EpochMetrics] = field(default_factory=list)

    def as_dict(self) -> Dict[str, List[Optional[float]]]:
        return {
            "epoch": [metrics.epoch for metrics in self.epochs],
            "train_loss": [metrics.train_loss for metrics in self.epochs],
            "train_accuracy": [metrics.train_accuracy for metrics in self.epochs],
            "validation_loss": [metrics.validation_loss for metrics in self.epochs],
            "validation_accuracy": [metrics.validation_accuracy for metrics in self.epochs],
        }


TrainingStepFunction = Callable[[nn.Module, Any, SupervisedTrainingContext], BatchMetrics]
TrainingCallback = Callable[[EpochMetrics, SupervisedTrainingHistory, nn.Module], Optional[bool]]


def train_supervised_model(
    model: nn.Module,
    train_loader,
    validation_loader=None,
    config: Optional[SupervisedTrainingConfig] = None,
    loss: LossLike = "cross_entropy",
    optimizer: OptimizerLike = "adam",
    learning_rate: float = 1e-3,
    epochs: int = 10,
    device: str = "cpu",
    output_mode: str = "auto",
    gradient_clip_norm: Optional[float] = None,
    log_interval: int = 1,
    callbacks: Optional[List[TrainingCallback]] = None,
    training_step_function: Optional[TrainingStepFunction] = None,
    **optimizer_kwargs,
) -> SupervisedTrainingHistory:
    """
    Train a regular PyTorch model with a simple supervised loop.

    Each batch from `train_loader` should yield `(inputs, targets)`. Sequence
    models that return `(batch, sequence, classes)` use the last step by default.
    """
    if config is not None:
        loss = config.loss
        optimizer = config.optimizer
        learning_rate = config.learning_rate
        epochs = config.epochs
        device = config.device
        output_mode = config.output_mode
        gradient_clip_norm = config.gradient_clip_norm
        log_interval = config.log_interval
        merged_optimizer_kwargs = dict(config.optimizer_kwargs)
        merged_optimizer_kwargs.update(optimizer_kwargs)
        optimizer_kwargs = merged_optimizer_kwargs

    callbacks = [] if callbacks is None else callbacks
    model = model.to(device)
    loss_fn = make_loss(loss)
    optimizer_instance = make_optimizer(
        optimizer,
        model.parameters(),
        learning_rate=learning_rate,
        **optimizer_kwargs,
    )
    training_context = SupervisedTrainingContext(
        loss_fn=loss_fn,
        optimizer=optimizer_instance,
        device=device,
        output_mode=output_mode,
        gradient_clip_norm=gradient_clip_norm,
    )
    history = SupervisedTrainingHistory()

    for epoch in range(1, epochs + 1):
        train_metrics = _run_training_epoch(
            model=model,
            data_loader=train_loader,
            context=training_context,
            epoch=epoch,
            training_step_function=training_step_function,
        )
        validation_metrics = None
        if validation_loader is not None:
            validation_metrics = evaluate_supervised_model(
                model=model,
                data_loader=validation_loader,
                loss=loss_fn,
                device=device,
                output_mode=output_mode,
            )

        epoch_metrics = EpochMetrics(
            epoch=epoch,
            train_loss=train_metrics.loss,
            train_accuracy=train_metrics.accuracy,
            validation_loss=validation_metrics.loss if validation_metrics else None,
            validation_accuracy=validation_metrics.accuracy if validation_metrics else None,
        )
        history.epochs.append(epoch_metrics)

        if log_interval > 0 and (epoch == 1 or epoch == epochs or epoch % log_interval == 0):
            _print_epoch_metrics(epoch_metrics, total_epochs=epochs)

        should_stop = _run_callbacks(
            callbacks=callbacks,
            metrics=epoch_metrics,
            history=history,
            model=model,
        )
        if should_stop:
            break

    return history


def evaluate_supervised_model(
    model: nn.Module,
    data_loader,
    loss: LossLike = "cross_entropy",
    device: str = "cpu",
    output_mode: str = "auto",
) -> BatchMetrics:
    """Evaluate a model over a full DataLoader."""
    model = model.to(device)
    loss_fn = make_loss(loss)
    batch_metrics = [
        run_evaluation_step(
            model=model,
            inputs=inputs,
            targets=targets,
            loss_fn=loss_fn,
            device=device,
            output_mode=output_mode,
        )
        for inputs, targets in data_loader
    ]
    return _average_batch_metrics(batch_metrics)


def _run_training_epoch(
    model: nn.Module,
    data_loader,
    context: SupervisedTrainingContext,
    epoch: int,
    training_step_function: Optional[TrainingStepFunction],
) -> BatchMetrics:
    batch_metrics = []
    for batch_index, batch in enumerate(data_loader, start=1):
        context.epoch = epoch
        context.batch_index = batch_index
        if training_step_function is not None:
            metrics = training_step_function(model, batch, context)
        else:
            inputs, targets = batch
            metrics = run_training_step(
                model=model,
                inputs=inputs,
                targets=targets,
                loss_fn=context.loss_fn,
                optimizer=context.optimizer,
                device=context.device,
                output_mode=context.output_mode,
                gradient_clip_norm=context.gradient_clip_norm,
            )
        batch_metrics.append(metrics)
    return _average_batch_metrics(batch_metrics)


def _average_batch_metrics(batch_metrics: List[BatchMetrics]) -> BatchMetrics:
    total_samples = sum(metrics.num_samples for metrics in batch_metrics)
    if total_samples == 0:
        return BatchMetrics(loss=0.0, accuracy=None, num_samples=0)

    loss = sum(metrics.loss * metrics.num_samples for metrics in batch_metrics) / total_samples
    accuracy_values = [
        metrics.accuracy * metrics.num_samples
        for metrics in batch_metrics
        if metrics.accuracy is not None
    ]
    accuracy = sum(accuracy_values) / total_samples if accuracy_values else None
    return BatchMetrics(loss=loss, accuracy=accuracy, num_samples=total_samples)


def _print_epoch_metrics(metrics: EpochMetrics, total_epochs: int) -> None:
    message = (
        f"epoch={metrics.epoch:03d}/{total_epochs:03d} "
        f"train_loss={metrics.train_loss:.4f}"
    )
    if metrics.train_accuracy is not None:
        message += f" train_accuracy={metrics.train_accuracy:.2%}"
    if metrics.validation_loss is not None:
        message += f" validation_loss={metrics.validation_loss:.4f}"
    if metrics.validation_accuracy is not None:
        message += f" validation_accuracy={metrics.validation_accuracy:.2%}"
    print(message)


def _run_callbacks(
    callbacks: List[TrainingCallback],
    metrics: EpochMetrics,
    history: SupervisedTrainingHistory,
    model: nn.Module,
) -> bool:
    should_stop = False
    for callback in callbacks:
        callback_result = callback(metrics, history, model)
        should_stop = should_stop or bool(callback_result)
    return should_stop
