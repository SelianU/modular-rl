import torch
from torch.utils.data import DataLoader, TensorDataset

from experiments.mnist.run_mnist_experiments import build_mnist_experiment_specs
from experiments.mnist.train import train_mnist_experiment


def _make_synthetic_mnist_loader(num_samples=16, batch_size=8):
    images = torch.zeros(num_samples, 1, 28, 28)
    labels = torch.arange(num_samples) % 10

    for index, label in enumerate(labels):
        row = 2 + int(label) * 2
        images[index, 0, row : row + 2, 4:24] = 1.0

    dataset = TensorDataset(images, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def test_build_mnist_experiment_specs_includes_supported_model_families():
    specs = build_mnist_experiment_specs(quick=True)
    model_names = {spec.model_name for spec in specs}

    assert model_names == {"mlp", "cnn", "rnn", "transformer"}
    assert all(spec.params["epochs"] == 1 for spec in specs)
    assert all(spec.params["train_limit"] > 0 for spec in specs)


def test_build_mnist_experiment_specs_applies_runtime_overrides():
    specs = build_mnist_experiment_specs(
        quick=True,
        model_name="cnn",
        train_limit=128,
        test_limit=64,
        epochs=2,
        learning_rate=0.005,
    )

    assert specs
    assert {spec.model_name for spec in specs} == {"cnn"}
    assert all(spec.params["train_limit"] == 128 for spec in specs)
    assert all(spec.params["test_limit"] == 64 for spec in specs)
    assert all(spec.params["epochs"] == 2 for spec in specs)
    assert all(spec.params["learning_rate"] == 0.005 for spec in specs)


def test_train_mnist_experiment_runs_one_synthetic_epoch_for_mlp():
    train_loader = _make_synthetic_mnist_loader()
    test_loader = _make_synthetic_mnist_loader()
    spec = build_mnist_experiment_specs(quick=True, model_name="mlp")[0]

    result = train_mnist_experiment(
        spec=spec,
        train_loader=train_loader,
        test_loader=test_loader,
        device="cpu",
    )

    assert result.model_name == "mlp"
    assert len(result.epochs) == 1
    assert result.epochs[0].train_loss > 0.0
    assert 0.0 <= result.epochs[0].train_accuracy <= 1.0
    assert 0.0 <= result.final_test_accuracy <= 1.0
