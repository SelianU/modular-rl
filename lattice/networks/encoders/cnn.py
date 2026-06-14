import torch
import torch.nn as nn
from typing import List, Tuple, Type

class CNN(nn.Module):
    """
    Convolutional Neural Network (CNN) backbone for spatial (image) observations.
    """
    def __init__(
        self,
        input_shape: Tuple[int, int, int],  # (C, H, W)
        feature_dim: int = 256,
        channels: List[int] = [32, 64, 64],
        kernels: List[int] = [8, 4, 3],
        strides: List[int] = [4, 2, 1],
        paddings: List[int] = [0, 0, 0],
        activation: Type[nn.Module] = nn.ReLU
    ):
        super().__init__()
        c, h, w = input_shape
        
        layers = []
        prev_c = c
        for out_c, k, s, p in zip(channels, kernels, strides, paddings):
            layers.append(nn.Conv2d(prev_c, out_c, kernel_size=k, stride=s, padding=p))
            layers.append(activation())
            prev_c = out_c
            
            # Compute spatial output size
            h = (h - k + 2 * p) // s + 1
            w = (w - k + 2 * p) // s + 1
            
        self.conv_net = nn.Sequential(*layers)
        self.conv_output_dim = prev_c * h * w
        self.fc = nn.Linear(self.conv_output_dim, feature_dim)
        self.fc_act = activation()
        self.output_dim = feature_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Check input dimension: normally expect (B, C, H, W)
        if x.dim() == 3:
            x = x.unsqueeze(0)
        # Normalize pixel values if they are integers [0, 255]
        if x.max() > 1.0:
            x = x.float() / 255.0
        features = self.conv_net(x)
        features = features.reshape(features.size(0), -1)
        return self.fc_act(self.fc(features))
