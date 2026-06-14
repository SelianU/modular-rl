from typing import List

import torch.nn as nn

from .encoders import MLP, RNN, Transformer


class _SequenceOutputModel(nn.Module):
    """Applies a sequence backbone and maps each feature vector to output_dim."""

    def __init__(self, backbone: nn.Module, output_dim: int):
        super().__init__()
        self.backbone = backbone
        self.output_layer = nn.Linear(backbone.output_dim, output_dim)
        self.output_dim = output_dim

    def forward(self, inputs):
        features = self.backbone(inputs)
        if isinstance(features, tuple):
            features = features[0]
        return self.output_layer(features)


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
