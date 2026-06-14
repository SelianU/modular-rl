import torch
import torch.nn as nn


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
