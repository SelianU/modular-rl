from __future__ import annotations

import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .models import build_mnist_model


@dataclass(frozen=True)
class MnistExperimentSpec:
    model_name: str
    params: dict


@dataclass(frozen=True)
class MnistEpochResult:
    epoch: int
    train_loss: float
    train_accuracy: float
    test_loss: float
    test_accuracy: float
    elapsed_seconds: float


@dataclass(frozen=True)
class MnistExperimentResult:
    model_name: str
    params: dict
    epochs: list[MnistEpochResult]
    final_test_accuracy: float


def train_mnist_experiment(
    spec: MnistExperimentSpec,
    train_loader,
    test_loader,
    device: str = "cpu",
) -> MnistExperimentResult:
    torch.manual_seed(spec.params["seed"])
    model = build_mnist_model(spec.model_name, spec.params).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=spec.params["learning_rate"])
    epoch_results = []

    print(f"\n=== MNIST | {spec.model_name.upper()} ===")
    print(f"params={spec.params}")

    for epoch in range(1, spec.params["epochs"] + 1):
        start_time = time.perf_counter()
        train_loss, train_accuracy = _train_one_epoch(model, train_loader, optimizer, device)
        test_loss, test_accuracy = evaluate_mnist_model(model, test_loader, device)
        elapsed_seconds = time.perf_counter() - start_time
        epoch_result = MnistEpochResult(
            epoch=epoch,
            train_loss=train_loss,
            train_accuracy=train_accuracy,
            test_loss=test_loss,
            test_accuracy=test_accuracy,
            elapsed_seconds=elapsed_seconds,
        )
        epoch_results.append(epoch_result)
        print(
            f"epoch={epoch:02d}/{spec.params['epochs']:02d} "
            f"train_loss={train_loss:.4f} "
            f"train_accuracy={train_accuracy:.2%} "
            f"test_loss={test_loss:.4f} "
            f"test_accuracy={test_accuracy:.2%} "
            f"elapsed_seconds={elapsed_seconds:.2f}"
        )

    final_test_accuracy = epoch_results[-1].test_accuracy
    print(f"final_test_accuracy={final_test_accuracy:.2%}")
    return MnistExperimentResult(
        model_name=spec.model_name,
        params=spec.params,
        epochs=epoch_results,
        final_test_accuracy=final_test_accuracy,
    )


def evaluate_mnist_model(model, data_loader, device: str) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = F.cross_entropy(logits, labels, reduction="sum")
            total_loss += loss.item()
            total_correct += (logits.argmax(dim=-1) == labels).sum().item()
            total_count += labels.numel()

    return total_loss / total_count, total_correct / total_count


def _train_one_epoch(model, data_loader, optimizer, device: str) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = F.cross_entropy(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.numel()
        total_correct += (logits.argmax(dim=-1) == labels).sum().item()
        total_count += labels.numel()

    return total_loss / total_count, total_correct / total_count
