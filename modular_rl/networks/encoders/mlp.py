import torch
import torch.nn as nn
from typing import List, Type

class MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) backbone.
    Maps a flat input vector to a feature vector.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        activation: Type[nn.Module] = nn.ReLU,
        use_layer_norm: bool = False,
        dropout: float = 0.0
    ):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(h_dim))
            layers.append(activation())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h_dim
        self.network = nn.Sequential(*layers)
        self.output_dim = prev_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
