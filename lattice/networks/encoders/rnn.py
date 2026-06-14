import torch
import torch.nn as nn
from typing import Tuple, Optional, Union

from .mlp import MLP
from .cnn import CNN

class RNN(nn.Module):
    """
    Recurrent Neural Network (RNN) backbone wrapper.
    Passes input observations through an underlying base backbone (MLP/CNN),
    then aggregates temporal state history using an LSTM or GRU cell.
    """
    def __init__(
        self,
        base_backbone: nn.Module,
        rnn_type: str = "LSTM",  # "LSTM" or "GRU"
        rnn_hidden_dim: int = 128
    ):
        super().__init__()
        self.base_backbone = base_backbone
        self.rnn_type = rnn_type
        self.rnn_hidden_dim = rnn_hidden_dim
        
        if rnn_type.upper() == "LSTM":
            self.rnn = nn.LSTM(input_size=base_backbone.output_dim, hidden_size=rnn_hidden_dim, batch_first=True)
        elif rnn_type.upper() == "GRU":
            self.rnn = nn.GRU(input_size=base_backbone.output_dim, hidden_size=rnn_hidden_dim, batch_first=True)
        else:
            raise ValueError(f"Unknown rnn_type: {rnn_type}")
            
        self.output_dim = rnn_hidden_dim

    def forward(
        self, 
        x: torch.Tensor, 
        hx: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]] = None
    ) -> Tuple[torch.Tensor, Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            x: Input tensor.
               - During training: (Batch, Sequence, Observations...)
               - During action selection: (Batch, Observations...)
            hx: Hidden/cell states for recurrent cells.
        Returns:
            output: Feature vector.
                    - If input has seq_dim (seq training): (Batch, Sequence, rnn_hidden_dim)
                    - If input has no seq_dim: (Batch, rnn_hidden_dim)
            next_hx: Updated hidden state(s).
        """
        # Distinguish if input has a sequence dimension
        has_seq_dim = False
        if (isinstance(self.base_backbone, MLP) and x.dim() == 3) or \
           (isinstance(self.base_backbone, CNN) and x.dim() == 5):
            has_seq_dim = True
            
        if not has_seq_dim:
            # Temporarily add sequence dimension of size 1
            x = x.unsqueeze(1)
            
        b, seq_len = x.size(0), x.size(1)
        other_dims = x.shape[2:]
        x_flat = x.reshape(b * seq_len, *other_dims)
        
        features_flat = self.base_backbone(x_flat)
        features = features_flat.reshape(b, seq_len, -1)
        
        out, next_hx = self.rnn(features, hx)
        
        if not has_seq_dim:
            # Remove the temporary sequence dimension
            out = out.squeeze(1)
            
        return out, next_hx

    def init_hidden(self, batch_size: int, device: torch.device = torch.device("cpu")) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if self.rnn_type.upper() == "LSTM":
            return (torch.zeros(1, batch_size, self.rnn_hidden_dim, device=device),
                    torch.zeros(1, batch_size, self.rnn_hidden_dim, device=device))
        else:
            return torch.zeros(1, batch_size, self.rnn_hidden_dim, device=device)
