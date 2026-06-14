"""
Lattice: PyTorch tools for models, training loops, and RL agents.
"""

__version__ = "0.1.0"

from .networks import (
    build_model,
    make_cnn_mlp,
    make_mlp,
    make_mlp_classifier,
    make_rnn,
    make_transformer,
)
from .training import (
    SupervisedTrainingConfig,
    quick_dqn,
    quick_ppo,
    quick_sac,
    quick_td3,
    train_supervised_model,
)

__all__ = [
    "__version__",
    "build_model",
    "make_cnn_mlp",
    "make_mlp",
    "make_mlp_classifier",
    "make_rnn",
    "make_transformer",
    "SupervisedTrainingConfig",
    "train_supervised_model",
    "quick_dqn",
    "quick_ppo",
    "quick_sac",
    "quick_td3",
]
