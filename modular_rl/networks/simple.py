"""
Compatibility exports for beginner-friendly network builders.

New code can import from `modular_rl.networks` directly. This module remains so
older imports such as `from modular_rl.networks.simple import make_mlp` keep
working.
"""

from .builders import build_model, make_cnn_mlp, make_mlp, make_mlp_classifier
from .language_models import MiniGPT, make_mini_gpt
from .sequence_builders import make_rnn, make_transformer
from .sequential import build_sequential_model

__all__ = [
    "MiniGPT",
    "build_model",
    "build_sequential_model",
    "make_cnn_mlp",
    "make_mini_gpt",
    "make_mlp",
    "make_mlp_classifier",
    "make_rnn",
    "make_transformer",
]
