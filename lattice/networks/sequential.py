from functools import reduce
from operator import mul
from typing import Dict, List, Tuple

import torch.nn as nn


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
