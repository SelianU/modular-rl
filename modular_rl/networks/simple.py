from functools import reduce
from operator import mul
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from .encoders import CNN, MLP, RNN, Transformer


class _SequenceOutputModel(nn.Module):
    """Applies a sequence backbone and maps each feature vector to output_dim."""

    def __init__(self, backbone: nn.Module, output_dim: int):
        super().__init__()
        self.backbone = backbone
        self.output_layer = nn.Linear(backbone.output_dim, output_dim)
        self.output_dim = output_dim

    def forward(self, x):
        features = self.backbone(x)
        if isinstance(features, tuple):
            features = features[0]
        return self.output_layer(features)


class MiniGPT(nn.Module):
    """
    A small GPT-style causal language model.

    This is an educational, randomly initialized model. It does not include a
    tokenizer, pretrained weights, or generation utilities.
    """

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim

        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(embed_dim, vocab_size)
        self.output_dim = vocab_size

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.dim() != 2:
            raise ValueError("MiniGPT expects input_ids with shape (batch, seq_len).")

        batch_size, seq_len = input_ids.shape
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_seq_len {self.max_seq_len}."
            )

        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        hidden_states = self.token_embedding(input_ids) + self.position_embedding(positions)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            seq_len,
            device=input_ids.device,
        )
        hidden_states = self.transformer(hidden_states, mask=causal_mask, is_causal=True)
        return self.output_layer(hidden_states)


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


def make_mlp_classifier(
    input_shape: Tuple[int, ...],
    output_dim: int,
    hidden_dims: List[int] = None,
    flatten_input: bool = True,
    activation: type = nn.ReLU,
) -> nn.Sequential:
    """
    Build an MLP classifier for shaped inputs such as images.

    This is a beginner-friendly wrapper for cases like MNIST, where data arrives
    as (batch, channels, height, width) but a plain MLP expects vectors.
    """
    input_dim = _num_features(input_shape)
    model = make_mlp(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dims=hidden_dims,
        activation=activation,
    )
    if flatten_input:
        return nn.Sequential(nn.Flatten(), model)
    return model


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


def make_rnn(
    input_dim: int,
    output_dim: int,
    hidden_dims: List[int] = None,
    rnn_type: str = "LSTM",
    rnn_hidden_dim: int = 128,
    activation: type = nn.ReLU,
) -> nn.Module:
    """
    Build an MLP + RNN + Linear output model.

    Accepts either (batch, input_dim) or (batch, sequence, input_dim). The
    beginner-friendly wrapper returns only the output tensor; use the lower-level
    RNN backbone directly if hidden states are needed.
    """
    hidden_dims = [64] if hidden_dims is None else hidden_dims
    base_backbone = MLP(input_dim=input_dim, hidden_dims=hidden_dims, activation=activation)
    recurrent_backbone = RNN(
        base_backbone=base_backbone,
        rnn_type=rnn_type,
        rnn_hidden_dim=rnn_hidden_dim,
    )
    return _SequenceOutputModel(recurrent_backbone, output_dim)


def make_transformer(
    input_dim: int,
    output_dim: int,
    hidden_dims: List[int] = None,
    embed_dim: int = 128,
    num_heads: int = 4,
    num_layers: int = 2,
    dim_feedforward: int = 256,
    dropout: float = 0.1,
    max_seq_len: int = 100,
    activation: type = nn.ReLU,
) -> nn.Module:
    """
    Build an MLP + Transformer + Linear output model.

    Accepts either (batch, input_dim) or (batch, sequence, input_dim).
    """
    hidden_dims = [64] if hidden_dims is None else hidden_dims
    base_backbone = MLP(input_dim=input_dim, hidden_dims=hidden_dims, activation=activation)
    transformer_backbone = Transformer(
        base_backbone=base_backbone,
        num_heads=num_heads,
        num_layers=num_layers,
        embed_dim=embed_dim,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        max_seq_len=max_seq_len,
    )
    return _SequenceOutputModel(transformer_backbone, output_dim)


def make_mini_gpt(
    vocab_size: int,
    max_seq_len: int,
    embed_dim: int = 128,
    num_heads: int = 4,
    num_layers: int = 2,
    dim_feedforward: int = 256,
    dropout: float = 0.1,
) -> MiniGPT:
    """Build a small GPT-style causal language model."""
    return MiniGPT(
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
    )


def build_sequential_model(
    input_shape: Tuple[int, ...],
    layers: List[Dict],
) -> nn.Sequential:
    """
    Build a torch-compatible nn.Sequential model from layer dictionaries.

    The builder tracks tensor shape so common fields can be inferred:
    - linear.in_features after vector inputs or flatten
    - conv2d.in_channels after image inputs or previous conv2d
    """
    current_shape = tuple(input_shape)
    modules = []

    for layer_config in layers:
        layer_config = dict(layer_config)
        layer_type = layer_config.pop("type")

        if layer_type == "linear":
            in_features = layer_config.pop("in_features", _num_features(current_shape))
            out_features = layer_config.pop("out_features")
            modules.append(nn.Linear(in_features=in_features, out_features=out_features, **layer_config))
            current_shape = (out_features,)
        elif layer_type == "conv2d":
            in_channels = layer_config.pop("in_channels", current_shape[0])
            out_channels = layer_config["out_channels"]
            kernel_size = layer_config.get("kernel_size", 3)
            stride = layer_config.get("stride", 1)
            padding = layer_config.get("padding", 0)
            dilation = layer_config.get("dilation", 1)
            modules.append(nn.Conv2d(in_channels=in_channels, **layer_config))
            current_shape = (
                out_channels,
                _conv_output_size(current_shape[1], kernel_size, stride, padding, dilation),
                _conv_output_size(current_shape[2], kernel_size, stride, padding, dilation),
            )
        elif layer_type == "flatten":
            modules.append(nn.Flatten(**layer_config))
            current_shape = (_num_features(current_shape),)
        elif layer_type == "relu":
            modules.append(nn.ReLU(**layer_config))
        elif layer_type == "tanh":
            modules.append(nn.Tanh())
        elif layer_type == "sigmoid":
            modules.append(nn.Sigmoid())
        elif layer_type == "dropout":
            modules.append(nn.Dropout(**layer_config))
        elif layer_type == "batch_norm1d":
            num_features = layer_config.pop("num_features", current_shape[0])
            modules.append(nn.BatchNorm1d(num_features=num_features, **layer_config))
        elif layer_type == "batch_norm2d":
            num_features = layer_config.pop("num_features", current_shape[0])
            modules.append(nn.BatchNorm2d(num_features=num_features, **layer_config))
        elif layer_type == "layer_norm":
            normalized_shape = layer_config.pop("normalized_shape", current_shape)
            modules.append(nn.LayerNorm(normalized_shape=normalized_shape, **layer_config))
        else:
            raise ValueError(
                f"Unknown layer type '{layer_type}'. "
                f"Available: {list(_available_sequential_layers())}"
            )

    return nn.Sequential(*modules)


def build_model(config: Dict) -> nn.Module:
    """
    Build a simple standalone model from a dictionary.

    Supported types:
    - {"type": "mlp", ...}
    - {"type": "cnn_mlp", ...}
    - {"type": "rnn", ...}
    - {"type": "transformer", ...}
    - {"type": "mini_gpt", ...}
    - {"type": "sequential", "input_shape": (...), "layers": [...]}
    """
    config = dict(config)
    model_type = config.pop("type", "mlp")

    if model_type == "mlp":
        return make_mlp(**config)
    if model_type == "mlp_classifier":
        return make_mlp_classifier(**config)
    if model_type == "cnn_mlp":
        return make_cnn_mlp(**config)
    if model_type == "rnn":
        return make_rnn(**config)
    if model_type == "transformer":
        return make_transformer(**config)
    if model_type == "mini_gpt":
        return make_mini_gpt(**config)
    if model_type == "sequential":
        return build_sequential_model(**config)

    raise ValueError("Unknown model type "
                     f"'{model_type}'. Available: ['mlp', 'mlp_classifier', 'cnn_mlp', 'rnn', 'transformer', 'mini_gpt', 'sequential']")


def _num_features(shape: Tuple[int, ...]) -> int:
    return int(reduce(mul, shape, 1))


def _pair(value) -> Tuple[int, int]:
    return value if isinstance(value, tuple) else (value, value)


def _conv_output_size(
    input_size: int,
    kernel_size,
    stride,
    padding,
    dilation,
) -> int:
    kernel, _ = _pair(kernel_size)
    stride_value, _ = _pair(stride)
    padding_value, _ = _pair(padding)
    dilation_value, _ = _pair(dilation)
    return ((input_size + 2 * padding_value - dilation_value * (kernel - 1) - 1) // stride_value) + 1


def _available_sequential_layers() -> Tuple[str, ...]:
    return (
        "linear",
        "conv2d",
        "flatten",
        "relu",
        "tanh",
        "sigmoid",
        "dropout",
        "batch_norm1d",
        "batch_norm2d",
        "layer_norm",
    )
