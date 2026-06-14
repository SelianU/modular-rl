from __future__ import annotations

import torch.nn as nn

from modular_rl.networks import make_cnn_mlp, make_mlp, make_rnn, make_transformer


class FlattenImageModel(nn.Module):
    """Convert MNIST images into vectors before running a vector model."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, images):
        return self.model(images.flatten(start_dim=1))


class RowSequenceModel(nn.Module):
    """Convert MNIST images into 28 row tokens before running a sequence model."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, images):
        row_sequence = images.squeeze(1)
        sequence_logits = self.model(row_sequence)
        return sequence_logits[:, -1, :]


def build_mnist_model(model_name: str, params: dict) -> nn.Module:
    if model_name == "mlp":
        return FlattenImageModel(
            make_mlp(
                input_dim=28 * 28,
                output_dim=10,
                hidden_dims=params["hidden_dims"],
            )
        )

    if model_name == "cnn":
        return make_cnn_mlp(
            input_shape=(1, 28, 28),
            output_dim=10,
            conv_channels=params["conv_channels"],
            mlp_hidden_dims=params["mlp_hidden_dims"],
            cnn_feature_dim=params["cnn_feature_dim"],
        )

    if model_name == "rnn":
        return RowSequenceModel(
            make_rnn(
                input_dim=28,
                output_dim=10,
                hidden_dims=params["hidden_dims"],
                rnn_hidden_dim=params["rnn_hidden_dim"],
            )
        )

    if model_name == "transformer":
        return RowSequenceModel(
            make_transformer(
                input_dim=28,
                output_dim=10,
                hidden_dims=params["hidden_dims"],
                embed_dim=params["embed_dim"],
                num_heads=params["num_heads"],
                num_layers=params["num_layers"],
                dim_feedforward=params["dim_feedforward"],
                dropout=params["dropout"],
                max_seq_len=28,
            )
        )

    raise ValueError("Unknown MNIST model "
                     f"'{model_name}'. Available: ['mlp', 'cnn', 'rnn', 'transformer']")
