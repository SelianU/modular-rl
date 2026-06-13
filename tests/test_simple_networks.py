import pytest
import torch
import torch.nn as nn

from modular_rl.networks import build_model, make_cnn_mlp, make_mlp


def test_make_mlp_maps_input_dim_to_output_dim():
    model = make_mlp(input_dim=5, output_dim=2, hidden_dims=[8])

    y = model(torch.randn(4, 5))

    assert y.shape == (4, 2)


def test_make_mlp_uses_default_hidden_layers():
    model = make_mlp(input_dim=3, output_dim=7)

    y = model(torch.randn(2, 3))

    assert y.shape == (2, 7)


def test_make_cnn_mlp_uses_two_3x3_convs_and_outputs_requested_dim():
    model = make_cnn_mlp(
        input_shape=(1, 12, 12),
        output_dim=4,
        conv_channels=[6, 8],
        mlp_hidden_dims=[10],
    )

    conv_layers = [layer for layer in model.modules() if isinstance(layer, nn.Conv2d)]
    y = model(torch.randn(3, 1, 12, 12))

    assert len(conv_layers) == 2
    assert [layer.kernel_size for layer in conv_layers] == [(3, 3), (3, 3)]
    assert y.shape == (3, 4)


def test_build_model_creates_mlp_from_dict():
    model = build_model({
        "type": "mlp",
        "input_dim": 5,
        "output_dim": 2,
        "hidden_dims": [8],
    })

    y = model(torch.randn(4, 5))

    assert y.shape == (4, 2)


def test_build_model_creates_cnn_mlp_from_dict():
    model = build_model({
        "type": "cnn_mlp",
        "input_shape": (1, 12, 12),
        "output_dim": 4,
        "conv_channels": [6, 8],
        "mlp_hidden_dims": [10],
    })

    y = model(torch.randn(3, 1, 12, 12))

    assert y.shape == (3, 4)


def test_build_model_rejects_unknown_type_with_available_options():
    with pytest.raises(ValueError, match="Available"):
        build_model({"type": "unknown", "input_dim": 5, "output_dim": 2})


def test_build_model_creates_sequential_mlp_from_layer_configs():
    model = build_model({
        "type": "sequential",
        "input_shape": (5,),
        "layers": [
            {"type": "linear", "out_features": 8},
            {"type": "relu"},
            {"type": "linear", "out_features": 2},
        ],
    })

    y = model(torch.randn(4, 5))

    assert y.shape == (4, 2)


def test_build_model_creates_sequential_cnn_with_auto_linear_input_size():
    model = build_model({
        "type": "sequential",
        "input_shape": (1, 12, 12),
        "layers": [
            {"type": "conv2d", "out_channels": 6, "kernel_size": 3, "padding": 1},
            {"type": "relu"},
            {"type": "conv2d", "out_channels": 8, "kernel_size": 3, "padding": 1},
            {"type": "relu"},
            {"type": "flatten"},
            {"type": "linear", "out_features": 4},
        ],
    })

    linear_layers = [layer for layer in model.modules() if isinstance(layer, nn.Linear)]
    y = model(torch.randn(3, 1, 12, 12))

    assert linear_layers[-1].in_features == 8 * 12 * 12
    assert y.shape == (3, 4)


def test_build_model_rejects_unknown_sequential_layer_with_available_options():
    with pytest.raises(ValueError, match="Available"):
        build_model({
            "type": "sequential",
            "input_shape": (5,),
            "layers": [{"type": "mystery"}],
        })
