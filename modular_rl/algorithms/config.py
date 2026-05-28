import torch
from dataclasses import dataclass
from typing import Optional


@dataclass
class DQNConfig:
    # Environment
    env_name: str = "CartPole-v1"

    # Optimization
    total_timesteps: int = 50000
    learning_rate: float = 1e-3
    buffer_size: int = 20000
    batch_size: int = 64
    gamma: float = 0.99

    # DQN specific
    target_update_freq: int = 200
    learning_starts: int = 1000
    is_double: bool = False         # Enable Double DQN

    # Exploration (epsilon-greedy)
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 5000

    # Device — "auto" detects CUDA automatically
    device: str = "auto"

    # Logging & saving
    log_interval: int = 10
    eval_interval: int = 0          # 0 = disabled; N = evaluate every N episodes
    save_dir: str = "checkpoints"

    def __post_init__(self):
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if self.batch_size > self.buffer_size:
            raise ValueError(
                f"batch_size ({self.batch_size}) must be <= buffer_size ({self.buffer_size})"
            )
        if self.learning_starts > self.total_timesteps:
            raise ValueError(
                f"learning_starts ({self.learning_starts}) must be <= total_timesteps ({self.total_timesteps})"
            )
        if not (0 < self.gamma <= 1):
            raise ValueError(f"gamma must be in (0, 1], got {self.gamma}")
        if not (0.0 <= self.epsilon_end <= self.epsilon_start <= 1.0):
            raise ValueError(
                f"Epsilon must satisfy 0 <= epsilon_end <= epsilon_start <= 1, "
                f"got end={self.epsilon_end}, start={self.epsilon_start}"
            )


@dataclass
class SACConfig:
    # Environment
    env_name: str = "Pendulum-v1"

    # Optimization
    total_timesteps: int = 50000
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    buffer_size: int = 100000
    batch_size: int = 256
    gamma: float = 0.99

    # SAC specific
    tau: float = 0.005
    alpha: float = 0.2
    auto_alpha: bool = True
    target_entropy: Optional[float] = None
    learning_starts: int = 1000

    # Device — "auto" detects CUDA automatically
    device: str = "auto"

    # Logging & saving
    log_interval: int = 10
    eval_interval: int = 0          # 0 = disabled; N = evaluate every N episodes
    save_dir: str = "checkpoints"

    def __post_init__(self):
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if self.batch_size > self.buffer_size:
            raise ValueError(
                f"batch_size ({self.batch_size}) must be <= buffer_size ({self.buffer_size})"
            )
        if self.learning_starts > self.total_timesteps:
            raise ValueError(
                f"learning_starts ({self.learning_starts}) must be <= total_timesteps ({self.total_timesteps})"
            )
        if not (0 < self.gamma <= 1):
            raise ValueError(f"gamma must be in (0, 1], got {self.gamma}")
        if not (0 < self.tau <= 1):
            raise ValueError(f"tau must be in (0, 1], got {self.tau}")
        if not self.auto_alpha and self.alpha <= 0:
            raise ValueError(f"alpha must be > 0 when auto_alpha=False, got {self.alpha}")


@dataclass
class PPOConfig:
    # Environment
    env_name: str = "CartPole-v1"

    # Rollout & optimization
    total_timesteps: int = 200_000
    n_steps: int = 2048             # Rollout length before each update
    n_epochs: int = 10              # PPO update epochs per rollout
    batch_size: int = 64            # Mini-batch size for PPO updates
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95        # GAE lambda (0=TD(0), 1=Monte-Carlo)

    # PPO clip objective
    clip_eps: float = 0.2
    value_coeff: float = 0.5        # Value loss coefficient
    entropy_coeff: float = 0.01     # Entropy bonus coefficient
    max_grad_norm: float = 0.5

    # Device — "auto" detects CUDA automatically
    device: str = "auto"

    # Logging & saving
    log_interval: int = 10
    eval_interval: int = 0
    save_dir: str = "checkpoints"

    def __post_init__(self):
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.batch_size > self.n_steps:
            raise ValueError(f"batch_size ({self.batch_size}) must be <= n_steps ({self.n_steps})")
        if not (0 < self.gamma <= 1):
            raise ValueError(f"gamma must be in (0, 1], got {self.gamma}")
        if not (0.0 <= self.gae_lambda <= 1.0):
            raise ValueError(f"gae_lambda must be in [0, 1], got {self.gae_lambda}")
        if not (0.0 < self.clip_eps < 1.0):
            raise ValueError(f"clip_eps must be in (0, 1), got {self.clip_eps}")


@dataclass
class TD3Config:
    # Environment
    env_name: str = "Pendulum-v1"

    # Optimization
    total_timesteps: int = 50_000
    learning_rate: float = 3e-4
    buffer_size: int = 100_000
    batch_size: int = 256
    gamma: float = 0.99
    tau: float = 0.005              # Polyak averaging rate

    # TD3 specific
    policy_delay: int = 2           # Update actor every N critic updates
    exploration_noise: float = 0.1  # Std of Gaussian exploration noise
    target_noise: float = 0.2       # Std of target policy smoothing noise
    noise_clip: float = 0.5         # Clip range for target noise
    learning_starts: int = 1000

    # Device — "auto" detects CUDA automatically
    device: str = "auto"

    # Logging & saving
    log_interval: int = 10
    eval_interval: int = 0
    save_dir: str = "checkpoints"

    def __post_init__(self):
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.batch_size > self.buffer_size:
            raise ValueError(f"batch_size ({self.batch_size}) must be <= buffer_size ({self.buffer_size})")
        if self.learning_starts > self.total_timesteps:
            raise ValueError(f"learning_starts ({self.learning_starts}) must be <= total_timesteps ({self.total_timesteps})")
        if not (0 < self.gamma <= 1):
            raise ValueError(f"gamma must be in (0, 1], got {self.gamma}")
        if not (0 < self.tau <= 1):
            raise ValueError(f"tau must be in (0, 1], got {self.tau}")
