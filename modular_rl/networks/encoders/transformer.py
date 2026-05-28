import torch
import torch.nn as nn
from typing import Optional

from .mlp import MLP
from .cnn import CNN

class Transformer(nn.Module):
    """
    Transformer-based backbone.
    Applies causal (masked) self-attention over a sequence of states/features.
    Useful for sequence-level RL representation.
    """
    def __init__(
        self,
        base_backbone: nn.Module,
        num_heads: int = 4,
        num_layers: int = 2,
        embed_dim: int = 128,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        max_seq_len: int = 100
    ):
        super().__init__()
        self.base_backbone = base_backbone
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        
        # Project base features to transformer embed_dim
        self.projection = nn.Linear(base_backbone.output_dim, embed_dim)
        
        # Learnable temporal positional embedding
        self.pos_emb = nn.Parameter(torch.zeros(1, max_seq_len, embed_dim))
        
        # Transformer Encoder Layer (using Pre-LN setup: norm_first=True for RL stability)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.output_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor.
               - During training: (Batch, Sequence, Observations...)
               - During action selection: (Batch, Observations...)
        """
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
        
        # 1. Feed through base backbone
        features_flat = self.base_backbone(x_flat)
        features = features_flat.reshape(b, seq_len, -1)
        
        # 2. Project features to transformer embedding dimension
        h = self.projection(features)
        
        # 3. Add positional embeddings
        assert seq_len <= self.max_seq_len, f"Sequence length {seq_len} exceeds max_seq_len {self.max_seq_len}"
        h = h + self.pos_emb[:, :seq_len, :]
        
        # 4. Generate causal mask (so each step can only attend to past steps)
        device = x.device
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=device)
        
        # 5. Process through the Transformer Encoder
        out = self.transformer(h, mask=causal_mask, is_causal=True)
        
        if not has_seq_dim:
            # Remove temporary sequence dimension
            out = out.squeeze(1)
            
        return out
