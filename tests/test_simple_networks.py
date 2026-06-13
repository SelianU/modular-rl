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
