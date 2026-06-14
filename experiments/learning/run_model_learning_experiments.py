from __future__ import annotations

import argparse
import time
import warnings
from dataclasses import dataclass
from itertools import product
from typing import Callable

import torch
import torch.nn.functional as F

from modular_rl.networks import (
    make_cnn_mlp,
    make_mini_gpt,
    make_mlp,
    make_rnn,
    make_transformer,
)


@dataclass(frozen=True)
class ExperimentSpec:
    model_name: str
    task_name: str
    params: dict
    train_fn: Callable[[dict, int, int], "ExperimentResult"]


@dataclass(frozen=True)
class ExperimentResult:
    model_name: str
    task_name: str
    params: dict
    initial_loss: float
    final_loss: float
    best_loss: float
    loss_ratio: float
    final_accuracy: float
    elapsed_seconds: float


def build_experiment_specs(quick: bool = False) -> list[ExperimentSpec]:
    specs = []
    specs.extend(_build_mlp_specs(quick))
    specs.extend(_build_cnn_specs(quick))
    specs.extend(_build_rnn_specs(quick))
    specs.extend(_build_transformer_specs(quick))
    specs.extend(_build_mini_gpt_specs(quick))
    return specs


def filter_experiment_specs(
    specs: list[ExperimentSpec],
    model_name: str | None = None,
) -> list[ExperimentSpec]:
    if model_name is None or model_name == "all":
        return specs
    return [spec for spec in specs if spec.model_name == model_name]


def run_experiment_spec(spec: ExperimentSpec, log_every: int = 25) -> ExperimentResult:
    print(f"\n=== {spec.model_name.upper()} | {spec.task_name} ===")
    print(f"params={spec.params}")
    return spec.train_fn(spec.params, spec.params["steps"], log_every)


def run_experiments(
    model_name: str = "all",
    quick: bool = False,
    log_every: int = 25,
) -> list[ExperimentResult]:
    specs = filter_experiment_specs(build_experiment_specs(quick=quick), model_name)
    if not specs:
        available = sorted({spec.model_name for spec in build_experiment_specs(quick=True)})
        raise ValueError(f"Unknown model '{model_name}'. Available models: {available}")

    results = [run_experiment_spec(spec, log_every=log_every) for spec in specs]
    _print_result_table(results)
    return results


def _build_mlp_specs(quick: bool) -> list[ExperimentSpec]:
    hidden_options = [[8], [16, 16]] if quick else [[8], [16, 16], [32, 32]]
    learning_rates = [0.03] if quick else [0.01, 0.03]
    return [
        ExperimentSpec(
            model_name="mlp",
            task_name="two_moons_like_classification",
            params={
                "hidden_dims": hidden_dims,
                "learning_rate": learning_rate,
                "steps": 120 if quick else 240,
                "seed": 10 + index,
            },
            train_fn=_train_mlp_two_moons_like,
        )
        for index, (hidden_dims, learning_rate) in enumerate(product(hidden_options, learning_rates))
    ]


def _build_cnn_specs(quick: bool) -> list[ExperimentSpec]:
    channel_options = [[4, 8], [8, 16]] if quick else [[4, 8], [8, 16], [16, 16]]
    learning_rates = [0.02] if quick else [0.01, 0.02]
    return [
        ExperimentSpec(
            model_name="cnn",
            task_name="bar_orientation_image_classification",
            params={
                "conv_channels": conv_channels,
                "learning_rate": learning_rate,
                "steps": 100 if quick else 200,
                "seed": 20 + index,
            },
            train_fn=_train_cnn_bar_orientation,
        )
        for index, (conv_channels, learning_rate) in enumerate(product(channel_options, learning_rates))
    ]


def _build_rnn_specs(quick: bool) -> list[ExperimentSpec]:
    hidden_options = [8, 16] if quick else [8, 16, 24]
    sequence_lengths = [5] if quick else [5, 7]
    return [
        ExperimentSpec(
            model_name="rnn",
            task_name="first_last_token_sequence_classification",
            params={
                "rnn_hidden_dim": hidden_dim,
                "sequence_length": sequence_length,
                "learning_rate": 0.02,
                "steps": 120 if quick else 220,
                "seed": 30 + index,
            },
            train_fn=_train_rnn_first_last,
        )
        for index, (hidden_dim, sequence_length) in enumerate(product(hidden_options, sequence_lengths))
    ]


def _build_transformer_specs(quick: bool) -> list[ExperimentSpec]:
    embed_options = [16, 32] if quick else [16, 32]
    head_options = [4] if quick else [2, 4]
    return [
        ExperimentSpec(
            model_name="transformer",
            task_name="marked_position_sequence_classification",
            params={
                "embed_dim": embed_dim,
                "num_heads": num_heads,
                "learning_rate": 0.02,
                "steps": 120 if quick else 220,
                "seed": 40 + index,
            },
            train_fn=_train_transformer_marked_position,
        )
        for index, (embed_dim, num_heads) in enumerate(product(embed_options, head_options))
    ]


def _build_mini_gpt_specs(quick: bool) -> list[ExperimentSpec]:
    embed_options = [12, 24] if quick else [12, 24]
    layer_options = [1] if quick else [1, 2]
    return [
        ExperimentSpec(
            model_name="mini_gpt",
            task_name="cyclic_next_token_prediction",
            params={
                "embed_dim": embed_dim,
                "num_layers": num_layers,
                "learning_rate": 0.03,
                "steps": 120 if quick else 220,
                "seed": 50 + index,
            },
            train_fn=_train_mini_gpt_cycle,
        )
        for index, (embed_dim, num_layers) in enumerate(product(embed_options, layer_options))
    ]


def _train_mlp_two_moons_like(params: dict, steps: int, log_every: int) -> ExperimentResult:
    torch.manual_seed(params["seed"])
    inputs, targets = _make_two_moons_like_data()
    model = make_mlp(input_dim=2, output_dim=2, hidden_dims=params["hidden_dims"])
    return _train_classifier(
        model=model,
        inputs=inputs,
        targets=targets,
        model_name="mlp",
        task_name="two_moons_like_classification",
        params=params,
        steps=steps,
        learning_rate=params["learning_rate"],
        log_every=log_every,
    )


def _train_cnn_bar_orientation(params: dict, steps: int, log_every: int) -> ExperimentResult:
    torch.manual_seed(params["seed"])
    inputs, targets = _make_bar_orientation_images()
    model = make_cnn_mlp(
        input_shape=(1, 12, 12),
        output_dim=3,
        conv_channels=params["conv_channels"],
        mlp_hidden_dims=[24],
        cnn_feature_dim=24,
    )
    return _train_classifier(
        model=model,
        inputs=inputs,
        targets=targets,
        model_name="cnn",
        task_name="bar_orientation_image_classification",
        params=params,
        steps=steps,
        learning_rate=params["learning_rate"],
        log_every=log_every,
    )


def _train_rnn_first_last(params: dict, steps: int, log_every: int) -> ExperimentResult:
    torch.manual_seed(params["seed"])
    inputs, targets = _make_first_last_sequence_data(params["sequence_length"])
    model = make_rnn(
        input_dim=2,
        output_dim=2,
        hidden_dims=[16],
        rnn_hidden_dim=params["rnn_hidden_dim"],
    )
    return _train_sequence_classifier(
        model=model,
        inputs=inputs,
        targets=targets,
        model_name="rnn",
        task_name="first_last_token_sequence_classification",
        params=params,
        steps=steps,
        learning_rate=params["learning_rate"],
        log_every=log_every,
    )


def _train_transformer_marked_position(params: dict, steps: int, log_every: int) -> ExperimentResult:
    torch.manual_seed(params["seed"])
    inputs, targets = _make_marked_position_sequence_data()
    model = make_transformer(
        input_dim=1,
        output_dim=6,
        hidden_dims=[16],
        embed_dim=params["embed_dim"],
        num_heads=params["num_heads"],
        num_layers=1,
        max_seq_len=6,
        dropout=0.0,
    )
    return _train_sequence_classifier(
        model=model,
        inputs=inputs,
        targets=targets,
        model_name="transformer",
        task_name="marked_position_sequence_classification",
        params=params,
        steps=steps,
        learning_rate=params["learning_rate"],
        log_every=log_every,
    )


def _train_mini_gpt_cycle(params: dict, steps: int, log_every: int) -> ExperimentResult:
    torch.manual_seed(params["seed"])
    input_ids, targets = _make_cyclic_token_data()
    model = make_mini_gpt(
        vocab_size=5,
        max_seq_len=6,
        embed_dim=params["embed_dim"],
        num_heads=4,
        num_layers=params["num_layers"],
        dim_feedforward=params["embed_dim"] * 2,
        dropout=0.0,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])
    losses = []
    start_time = time.perf_counter()

    for step in range(1, steps + 1):
        logits = model(input_ids)
        loss = F.cross_entropy(logits.reshape(-1, 5), targets.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        _print_step_log(step, steps, loss.item(), log_every)

    elapsed_seconds = time.perf_counter() - start_time
    with torch.no_grad():
        predictions = model(input_ids).argmax(dim=-1)
        accuracy = (predictions == targets).float().mean().item()

    return _make_result("mini_gpt", "cyclic_next_token_prediction", params, losses, accuracy, elapsed_seconds)


def _train_classifier(
    model,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    model_name: str,
    task_name: str,
    params: dict,
    steps: int,
    learning_rate: float,
    log_every: int,
) -> ExperimentResult:
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    losses = []
    start_time = time.perf_counter()

    for step in range(1, steps + 1):
        logits = model(inputs)
        loss = F.cross_entropy(logits, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        _print_step_log(step, steps, loss.item(), log_every)

    elapsed_seconds = time.perf_counter() - start_time
    with torch.no_grad():
        predictions = model(inputs).argmax(dim=-1)
        accuracy = (predictions == targets).float().mean().item()

    return _make_result(model_name, task_name, params, losses, accuracy, elapsed_seconds)


def _train_sequence_classifier(
    model,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    model_name: str,
    task_name: str,
    params: dict,
    steps: int,
    learning_rate: float,
    log_every: int,
) -> ExperimentResult:
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    losses = []
    start_time = time.perf_counter()

    for step in range(1, steps + 1):
        logits = model(inputs)
        loss = F.cross_entropy(logits[:, -1, :], targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        _print_step_log(step, steps, loss.item(), log_every)

    elapsed_seconds = time.perf_counter() - start_time
    with torch.no_grad():
        predictions = model(inputs)[:, -1, :].argmax(dim=-1)
        accuracy = (predictions == targets).float().mean().item()

    return _make_result(model_name, task_name, params, losses, accuracy, elapsed_seconds)


def _make_result(
    model_name: str,
    task_name: str,
    params: dict,
    losses: list[float],
    accuracy: float,
    elapsed_seconds: float,
) -> ExperimentResult:
    result = ExperimentResult(
        model_name=model_name,
        task_name=task_name,
        params=params,
        initial_loss=losses[0],
        final_loss=losses[-1],
        best_loss=min(losses),
        loss_ratio=losses[-1] / losses[0],
        final_accuracy=accuracy,
        elapsed_seconds=elapsed_seconds,
    )
    print(
        f"result initial_loss={result.initial_loss:.4f} "
        f"final_loss={result.final_loss:.4f} "
        f"best_loss={result.best_loss:.4f} "
        f"loss_ratio={result.loss_ratio:.4f} "
        f"final_accuracy={result.final_accuracy:.2%} "
        f"elapsed_seconds={result.elapsed_seconds:.2f}"
    )
    return result


def _print_step_log(step: int, total_steps: int, loss: float, log_every: int) -> None:
    if log_every <= 0:
        return
    if step == 1 or step == total_steps or step % log_every == 0:
        print(f"step={step:04d}/{total_steps:04d} loss={loss:.4f}")


def _print_result_table(results: list[ExperimentResult]) -> None:
    print("\n=== Summary ===")
    for result in results:
        print(
            f"{result.model_name:11s} "
            f"loss {result.initial_loss:.4f}->{result.final_loss:.4f} "
            f"best={result.best_loss:.4f} "
            f"accuracy={result.final_accuracy:.2%} "
            f"time={result.elapsed_seconds:.2f}s "
            f"params={result.params}"
        )


def _make_two_moons_like_data() -> tuple[torch.Tensor, torch.Tensor]:
    angles = torch.linspace(0.0, 3.14159, 16)
    upper = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)
    lower = torch.stack([1.0 - torch.cos(angles), -torch.sin(angles) - 0.3], dim=1)
    inputs = torch.cat([upper, lower], dim=0).float()
    targets = torch.cat(
        [
            torch.zeros(upper.shape[0], dtype=torch.long),
            torch.ones(lower.shape[0], dtype=torch.long),
        ]
    )
    return inputs, targets


def _make_bar_orientation_images() -> tuple[torch.Tensor, torch.Tensor]:
    images = []
    targets = []
    for offset in [2, 4, 6, 8]:
        vertical = torch.zeros(1, 12, 12)
        vertical[:, :, offset] = 1.0
        images.append(vertical)
        targets.append(0)

        horizontal = torch.zeros(1, 12, 12)
        horizontal[:, offset, :] = 1.0
        images.append(horizontal)
        targets.append(1)

        diagonal = torch.zeros(1, 12, 12)
        diagonal[:, torch.arange(12), torch.arange(12)] = 1.0
        diagonal = torch.roll(diagonal, shifts=offset - 2, dims=1)
        images.append(diagonal)
        targets.append(2)
    return torch.stack(images), torch.tensor(targets)


def _make_first_last_sequence_data(sequence_length: int) -> tuple[torch.Tensor, torch.Tensor]:
    rows = []
    targets = []
    for first in [0, 1]:
        for last in [0, 1]:
            for middle_value in [0, 1]:
                token_ids = [first] + [middle_value] * (sequence_length - 2) + [last]
                rows.append(F.one_hot(torch.tensor(token_ids), num_classes=2).float())
                targets.append(1 if first == last else 0)
    return torch.stack(rows), torch.tensor(targets)


def _make_marked_position_sequence_data() -> tuple[torch.Tensor, torch.Tensor]:
    rows = []
    targets = []
    for index in range(6):
        row = torch.zeros(6, 1)
        row[index, 0] = 1.0
        rows.append(row)
        targets.append(index)
    return torch.stack(rows), torch.tensor(targets)


def _make_cyclic_token_data() -> tuple[torch.Tensor, torch.Tensor]:
    rows = []
    targets = []
    for start in range(5):
        row = torch.tensor([(start + offset) % 5 for offset in range(6)], dtype=torch.long)
        target = torch.tensor([(start + offset + 1) % 5 for offset in range(6)], dtype=torch.long)
        rows.append(row)
        targets.append(target)
    return torch.stack(rows), torch.stack(targets)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run model learning experiments.")
    parser.add_argument(
        "--model",
        default="all",
        choices=["all", "mlp", "cnn", "rnn", "transformer", "mini_gpt"],
        help="Model family to run.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run fewer parameter combinations and shorter training.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=25,
        help="Print training loss every N steps. Use 0 to print only final results.",
    )
    return parser.parse_args()


def main() -> None:
    warnings.filterwarnings(
        "ignore",
        message="enable_nested_tensor is True, but self.use_nested_tensor is False.*",
        category=UserWarning,
    )
    args = _parse_args()
    run_experiments(model_name=args.model, quick=args.quick, log_every=args.log_every)


if __name__ == "__main__":
    main()
