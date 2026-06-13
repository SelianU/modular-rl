from typing import Dict, List, Tuple

import torch.nn as nn

from .encoders import CNN, MLP


def make_mlp(
    input_dim: int,
    output_dim: int,
    hidden_dims: List[int] = None,
    activation: type = nn.ReLU,
) -> nn.Sequential:
    """
    Build a plain MLP for beginners.

    The final output layer has no activation, so the returned model can be used
    for logits, regression values, or as a feature head.
    """
    hidden_dims = [64, 64] if hidden_dims is None else hidden_dims
    encoder = MLP(input_dim=input_dim, hidden_dims=hidden_dims, activation=activation)
    return nn.Sequential(encoder, nn.Linear(encoder.output_dim, output_dim))


def make_cnn_mlp(
    input_shape: Tuple[int, int, int],
    output_dim: int,
    conv_channels: List[int] = None,
    mlp_hidden_dims: List[int] = None,
    kernel_size: int = 3,
    stride: int = 1,
    padding: int = 1,
    cnn_feature_dim: int = 128,
    activation: type = nn.ReLU,
) -> nn.Sequential:
    """
    Build a CNN followed by an MLP.

    Defaults to two 3x3 convolution layers with padding=1, so image height and
    width stay the same before flattening.
    """
    conv_channels = [32, 64] if conv_channels is None else conv_channels
    mlp_hidden_dims = [] if mlp_hidden_dims is None else mlp_hidden_dims
    n_convs = len(conv_channels)

    cnn = CNN(
        input_shape=input_shape,
        feature_dim=cnn_feature_dim,
        channels=conv_channels,
        kernels=[kernel_size] * n_convs,
        strides=[stride] * n_convs,
        paddings=[padding] * n_convs,
        activation=activation,
    )
    head = make_mlp(
        input_dim=cnn.output_dim,
        output_dim=output_dim,
        hidden_dims=mlp_hidden_dims,
        activation=activation,
    )
    return nn.Sequential(cnn, head)


def build_model(config: Dict) -> nn.Module:
    """
    Build a simple standalone model from a dictionary.

    Supported types:
    - {"type": "mlp", ...}
    - {"type": "cnn_mlp", ...}
    """
    config = dict(config)
    model_type = config.pop("type", "mlp")

    if model_type == "mlp":
        return make_mlp(**config)
    if model_type == "cnn_mlp":
        return make_cnn_mlp(**config)

    raise ValueError("Unknown model type "
                     f"'{model_type}'. Available: ['mlp', 'cnn_mlp']")
