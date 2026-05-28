from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, Tuple, Union
import numpy as np
import gymnasium as gym


class BaseEnv(ABC):
    """Abstract environment interface used by Trainer."""

    @abstractmethod
    def reset(self) -> np.ndarray:
        """Reset the environment and return the initial observation."""
        ...

    @abstractmethod
    def step(self, action) -> Tuple[np.ndarray, float, bool, Dict]:
        """Apply action. Returns (next_obs, reward, done, info)."""
        ...

    @abstractmethod
    def close(self):
        ...

    @property
    @abstractmethod
    def state_dim(self) -> Union[int, Tuple[int, ...]]:
        """Flat int for vector envs; (C, H, W) tuple for image envs."""
        ...

    @property
    @abstractmethod
    def action_dim(self) -> int:
        ...

    @property
    @abstractmethod
    def is_continuous(self) -> bool:
        ...


class GymEnvWrapper(BaseEnv):
    """
    Wraps a Gymnasium environment.
    Pass obs_transform to apply per-step observation preprocessing
    (e.g. POMDP masking, normalisation).
    """

    def __init__(
        self,
        env_name: str,
        obs_transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        **gym_kwargs,
    ):
        self._env = gym.make(env_name, **gym_kwargs)
        self._obs_transform = obs_transform

        obs_space = self._env.observation_space
        act_space = self._env.action_space

        if hasattr(obs_space, "shape"):
            self._state_dim: Union[int, Tuple] = (
                obs_space.shape[0] if len(obs_space.shape) == 1 else obs_space.shape
            )
        else:
            self._state_dim = obs_space.n

        if hasattr(act_space, "n"):
            self._action_dim = act_space.n
            self._is_continuous = False
            self._action_low: Optional[float] = None
            self._action_high: Optional[float] = None
        else:
            self._action_dim = act_space.shape[0]
            self._is_continuous = True
            self._action_low = float(act_space.low[0])
            self._action_high = float(act_space.high[0])

    def _transform(self, obs: np.ndarray) -> np.ndarray:
        return self._obs_transform(obs) if self._obs_transform else obs

    def reset(self) -> np.ndarray:
        obs, _ = self._env.reset()
        return self._transform(obs)

    def step(self, action) -> Tuple[np.ndarray, float, bool, Dict]:
        next_obs, reward, terminated, truncated, info = self._env.step(action)
        return self._transform(next_obs), float(reward), terminated or truncated, info

    def close(self):
        self._env.close()

    @property
    def state_dim(self) -> Union[int, Tuple]:
        return self._state_dim

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def is_continuous(self) -> bool:
        return self._is_continuous

    @property
    def action_low(self) -> Optional[float]:
        return self._action_low

    @property
    def action_high(self) -> Optional[float]:
        return self._action_high


class CustomEnvWrapper(BaseEnv):
    """
    Wraps any custom environment that follows the Gymnasium step/reset interface
    (reset → (obs, info), step → (obs, reward, terminated, truncated, info)).
    Used for envs that don't inherit gym.Env (e.g. GridWorldEnv).
    """

    def __init__(
        self,
        env: Any,
        state_dim: Union[int, Tuple[int, ...]],
        action_dim: int,
        is_continuous: bool = False,
        action_low: Optional[float] = None,
        action_high: Optional[float] = None,
        obs_transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ):
        self._env = env
        self._state_dim = state_dim
        self._action_dim = action_dim
        self._is_continuous = is_continuous
        self._action_low = action_low
        self._action_high = action_high
        self._obs_transform = obs_transform

    def _transform(self, obs: np.ndarray) -> np.ndarray:
        return self._obs_transform(obs) if self._obs_transform else obs

    def reset(self) -> np.ndarray:
        result = self._env.reset()
        obs = result[0] if isinstance(result, tuple) else result
        return self._transform(obs)

    def step(self, action) -> Tuple[np.ndarray, float, bool, Dict]:
        next_obs, reward, terminated, truncated, info = self._env.step(action)
        return self._transform(next_obs), float(reward), terminated or truncated, info

    def close(self):
        self._env.close()

    @property
    def state_dim(self) -> Union[int, Tuple]:
        return self._state_dim

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def is_continuous(self) -> bool:
        return self._is_continuous

    @property
    def action_low(self) -> Optional[float]:
        return self._action_low

    @property
    def action_high(self) -> Optional[float]:
        return self._action_high
