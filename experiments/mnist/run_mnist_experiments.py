from __future__ import annotations

import argparse
import warnings
from itertools import product

import torch

from .data import create_mnist_data_loaders
from .train import MnistExperimentResult, MnistExperimentSpec, train_mnist_experiment


def build_mnist_experiment_specs(
    quick: bool = False,
    model_name: str = "all",
    train_limit: int | None = None,
    test_limit: int | None = None,
    epochs: int | None = None,
    learning_rate: float | None = None,
) -> list[MnistExperimentSpec]:
    train_limit = train_limit if train_limit is not None else (1024 if quick else 60000)
    test_limit = test_limit if test_limit is not None else (256 if quick else 10000)
    epochs = epochs if epochs is not None else (1 if quick else 3)
    specs = []
    specs.extend(_build_mlp_specs(quick, train_limit, test_limit, epochs, learning_rate))
    specs.extend(_build_cnn_specs(quick, train_limit, test_limit, epochs, learning_rate))
    specs.extend(_build_rnn_specs(quick, train_limit, test_limit, epochs, learning_rate))
    specs.extend(_build_transformer_specs(quick, train_limit, test_limit, epochs, learning_rate))

    if model_name == "all":
        return specs
    filtered_specs = [spec for spec in specs if spec.model_name == model_name]
    if not filtered_specs:
        available = sorted({spec.model_name for spec in specs})
        raise ValueError(f"Unknown model '{model_name}'. Available models: {available}")
    return filtered_specs


def run_mnist_experiments(args: argparse.Namespace) -> list[MnistExperimentResult]:
    train_limit = args.train_limit if args.train_limit is not None else (1024 if args.quick else None)
    test_limit = args.test_limit if args.test_limit is not None else (256 if args.quick else None)
    train_loader, test_loader = create_mnist_data_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        train_limit=train_limit,
        test_limit=test_limit,
        download=not args.no_download,
    )
    device = _resolve_device(args.device)
    specs = build_mnist_experiment_specs(
        quick=args.quick,
        model_name=args.model,
        train_limit=train_limit,
        test_limit=test_limit,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
    )
    results = [
        train_mnist_experiment(
            spec=spec,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
        )
        for spec in specs
    ]
    _print_summary(results)
    return results


def _build_mlp_specs(
    quick: bool,
    train_limit: int | None,
    test_limit: int | None,
    epochs: int,
    learning_rate: float | None,
) -> list[MnistExperimentSpec]:
    hidden_options = [[128], [256, 128]] if quick else [[128], [256, 128], [512, 256]]
    learning_rates = [learning_rate] if learning_rate is not None else ([0.001] if quick else [0.001, 0.0005])
    return [
        MnistExperimentSpec(
            model_name="mlp",
            params={
                "hidden_dims": hidden_dims,
                "learning_rate": learning_rate,
                "epochs": epochs,
                "train_limit": train_limit,
                "test_limit": test_limit,
                "seed": 100 + index,
            },
        )
        for index, (hidden_dims, learning_rate) in enumerate(product(hidden_options, learning_rates))
    ]


def _build_cnn_specs(
    quick: bool,
    train_limit: int | None,
    test_limit: int | None,
    epochs: int,
    learning_rate: float | None,
) -> list[MnistExperimentSpec]:
    channel_options = [[16, 32], [32, 64]] if quick else [[16, 32], [32, 64]]
    learning_rates = [learning_rate] if learning_rate is not None else ([0.001] if quick else [0.001, 0.0005])
    return [
        MnistExperimentSpec(
            model_name="cnn",
            params={
                "conv_channels": conv_channels,
                "mlp_hidden_dims": [128],
                "cnn_feature_dim": 128,
                "learning_rate": learning_rate,
                "epochs": epochs,
                "train_limit": train_limit,
                "test_limit": test_limit,
                "seed": 200 + index,
            },
        )
        for index, (conv_channels, learning_rate) in enumerate(product(channel_options, learning_rates))
    ]


def _build_rnn_specs(
    quick: bool,
    train_limit: int | None,
    test_limit: int | None,
    epochs: int,
    learning_rate: float | None,
) -> list[MnistExperimentSpec]:
    hidden_options = [64, 128] if quick else [64, 128]
    learning_rates = [learning_rate] if learning_rate is not None else ([0.001] if quick else [0.001, 0.0005])
    return [
        MnistExperimentSpec(
            model_name="rnn",
            params={
                "hidden_dims": [64],
                "rnn_hidden_dim": rnn_hidden_dim,
                "learning_rate": learning_rate,
                "epochs": epochs,
                "train_limit": train_limit,
                "test_limit": test_limit,
                "seed": 300 + index,
            },
        )
        for index, (rnn_hidden_dim, learning_rate) in enumerate(product(hidden_options, learning_rates))
    ]


def _build_transformer_specs(
    quick: bool,
    train_limit: int | None,
    test_limit: int | None,
    epochs: int,
    learning_rate: float | None,
) -> list[MnistExperimentSpec]:
    embed_options = [64, 128] if quick else [64, 128]
    learning_rates = [learning_rate] if learning_rate is not None else ([0.001] if quick else [0.001, 0.0005])
    return [
        MnistExperimentSpec(
            model_name="transformer",
            params={
                "hidden_dims": [64],
                "embed_dim": embed_dim,
                "num_heads": 4,
                "num_layers": 1,
                "dim_feedforward": embed_dim * 2,
                "dropout": 0.0,
                "learning_rate": learning_rate,
                "epochs": epochs,
                "train_limit": train_limit,
                "test_limit": test_limit,
                "seed": 400 + index,
            },
        )
        for index, (embed_dim, learning_rate) in enumerate(product(embed_options, learning_rates))
    ]


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _print_summary(results: list[MnistExperimentResult]) -> None:
    print("\n=== MNIST Summary ===")
    for result in results:
        last_epoch = result.epochs[-1]
        print(
            f"{result.model_name:11s} "
            f"train_accuracy={last_epoch.train_accuracy:.2%} "
            f"test_accuracy={last_epoch.test_accuracy:.2%} "
            f"test_loss={last_epoch.test_loss:.4f} "
            f"params={result.params}"
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MNIST classification experiments.")
    parser.add_argument(
        "--model",
        default="all",
        choices=["all", "mlp", "cnn", "rnn", "transformer"],
        help="Model family to run.",
    )
    parser.add_argument("--quick", action="store_true", help="Use small MNIST subsets and 1 epoch.")
    parser.add_argument("--data-dir", default="data", help="Where MNIST files are stored.")
    parser.add_argument("--batch-size", type=int, default=64, help="Training and test batch size.")
    parser.add_argument("--train-limit", type=int, default=None, help="Optional train sample limit.")
    parser.add_argument("--test-limit", type=int, default=None, help="Optional test sample limit.")
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count for every run.")
    parser.add_argument("--learning-rate", type=float, default=None, help="Override learning rate for every run.")
    parser.add_argument("--device", default="auto", help="Use auto, cpu, cuda, or another torch device.")
    parser.add_argument("--no-download", action="store_true", help="Do not download MNIST if missing.")
    return parser.parse_args()


def main() -> None:
    warnings.filterwarnings(
        "ignore",
        message="enable_nested_tensor is True, but self.use_nested_tensor is False.*",
        category=UserWarning,
    )
    run_mnist_experiments(_parse_args())


if __name__ == "__main__":
    main()
