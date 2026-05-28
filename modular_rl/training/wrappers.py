"""
Environment wrappers that modify observations or rewards.
All wrappers implement BaseEnv and can be stacked.
"""
import numpy as np
import torch
from typing import Dict, Optional, Tuple, Union

from .env_wrapper import BaseEnv


class RewardNormWrapper(BaseEnv):
    """
    Normalises rewards online using Welford's running mean/variance.

    At each step the raw reward is divided by the running standard deviation
    (clipped to ±clip). Useful for continuous-control tasks (SAC on Pendulum,
    etc.) where reward scales vary widely across environments.

    Usage
    -----
    env = GymEnvWrapper("Pendulum-v1")
    env = RewardNormWrapper(env, clip=10.0)
    trainer = Trainer(agent, env, config, ...)
    """

    def __init__(self, env: BaseEnv, clip: float = 10.0, epsilon: float = 1e-8):
        self._env = env
        self._clip = clip
        self._epsilon = epsilon
        # Welford's online algorithm state
        self._n = 0
        self._mean = 0.0
        self._M2 = 0.0     # accumulated sum of squared deviations from mean

    # ------------------------------------------------------------------ #
    # Welford update                                                        #
    # ------------------------------------------------------------------ #

    def _normalise(self, reward: float) -> float:
        self._n += 1
        delta = reward - self._mean
        self._mean += delta / self._n
        self._M2 += delta * (reward - self._mean)
        if self._n < 2:
            return reward
        std = (self._M2 / (self._n - 1)) ** 0.5
        return float(np.clip(reward / (std + self._epsilon), -self._clip, self._clip))

    # ------------------------------------------------------------------ #
    # BaseEnv interface                                                     #
    # ------------------------------------------------------------------ #

    def reset(self) -> np.ndarray:
        return self._env.reset()

    def step(self, action) -> Tuple[np.ndarray, float, bool, Dict]:
        obs, reward, done, info = self._env.step(action)
        return obs, self._normalise(reward), done, info

    def close(self):
        self._env.close()

    @property
    def state_dim(self) -> Union[int, tuple]:
        return self._env.state_dim

    @property
    def action_dim(self) -> int:
        return self._env.action_dim

    @property
    def is_continuous(self) -> bool:
        return self._env.is_continuous

    # Forward optional properties (action bounds for SAC)
    @property
    def action_low(self):
        return getattr(self._env, "action_low", None)

    @property
    def action_high(self):
        return getattr(self._env, "action_high", None)


class LearnedRewardWrapper(BaseEnv):
    """
    Replaces the environment's reward signal with a learned RewardModel.

    Used in D-REX, CLARE, and any reward-learning pipeline where the true
    environment reward is unavailable at training time.

    Usage
    -----
    env = GymEnvWrapper("CartPole-v1")
    env = LearnedRewardWrapper(env, reward_model, device="cpu")
    trainer = Trainer(agent, env, config)   # agent trains on learned reward

    Parameters
    ----------
    env          : Any BaseEnv implementation.
    reward_model : Trained RewardModel (R(s) or R(s,a)).
    device       : Torch device for reward inference.
    """

    def __init__(self, env: BaseEnv, reward_model, device: str = "cpu"):
        self._env = env
        self._reward_model = reward_model
        self._device = torch.device(device)
        self._reward_model.to(self._device)
        self._reward_model.eval()
        self._last_state: Optional[np.ndarray] = None

    def reset(self) -> np.ndarray:
        obs = self._env.reset()
        self._last_state = obs
        return obs

    def step(self, action) -> Tuple[np.ndarray, float, bool, Dict]:
        obs, _, done, info = self._env.step(action)

        with torch.no_grad():
            state_t = torch.from_numpy(
                np.array(self._last_state, dtype=np.float32)
            ).unsqueeze(0).to(self._device)

            if self._reward_model.uses_action:
                action_t = torch.from_numpy(
                    np.array(action, dtype=np.float32)
                ).unsqueeze(0).to(self._device)
                reward = self._reward_model(state_t, action_t).item()
            else:
                reward = self._reward_model(state_t).item()

        self._last_state = obs
        return obs, reward, done, info

    def close(self):
        self._env.close()

    @property
    def state_dim(self) -> Union[int, tuple]:
        return self._env.state_dim

    @property
    def action_dim(self) -> int:
        return self._env.action_dim

    @property
    def is_continuous(self) -> bool:
        return self._env.is_continuous

    @property
    def action_low(self):
        return getattr(self._env, "action_low", None)

    @property
    def action_high(self):
        return getattr(self._env, "action_high", None)
