from __future__ import annotations

from pathlib import Path

from torch.utils.data import DataLoader, Subset


def create_mnist_data_loaders(
    data_dir: str | Path = "data",
    batch_size: int = 64,
    train_limit: int | None = None,
    test_limit: int | None = None,
    download: bool = True,
) -> tuple[DataLoader, DataLoader]:
    """
    Create MNIST train/test loaders.

    torchvision is imported inside this function so the rest of the project can
    still be used without installing MNIST-specific dependencies.
    """
    try:
        from torchvision import datasets, transforms
    except ImportError as error:
        raise ImportError(
            "MNIST experiments require torchvision. Install it with "
            "`pip install torchvision` or reinstall this project dependencies."
        ) from error

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    train_dataset = datasets.MNIST(
        root=str(data_dir),
        train=True,
        download=download,
        transform=transform,
    )
    test_dataset = datasets.MNIST(
        root=str(data_dir),
        train=False,
        download=download,
        transform=transform,
    )

    if train_limit is not None:
        train_dataset = Subset(train_dataset, range(min(train_limit, len(train_dataset))))
    if test_limit is not None:
        test_dataset = Subset(test_dataset, range(min(test_limit, len(test_dataset))))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader
