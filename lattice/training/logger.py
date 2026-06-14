from abc import ABC, abstractmethod
from typing import Dict, List
import numpy as np


class BaseLogger(ABC):
    """Abstract logger interface. Implement to add WandB, TensorBoard, etc."""

    @abstractmethod
    def log_episode(self, episode: int, step: int, metrics: Dict[str, float]) -> None:
        ...

    def close(self) -> None:
        pass


class ConsoleLogger(BaseLogger):
    """Prints episode metrics to stdout at a configurable interval."""

    def __init__(self, log_interval: int = 10, prefix: str = ""):
        self.log_interval = log_interval
        self.prefix = prefix
        self._recent_rewards: List[float] = []

    def log_episode(self, episode: int, step: int, metrics: Dict[str, float]) -> None:
        reward = metrics.get("reward", 0.0)
        self._recent_rewards.append(reward)
        if len(self._recent_rewards) > 100:
            self._recent_rewards.pop(0)

        if episode % self.log_interval == 0:
            parts = [f"{self.prefix}Ep {episode:4d}", f"Step {step:6d}"]
            parts.append(f"Reward {reward:8.2f}")
            parts.append(f"Mean(100) {np.mean(self._recent_rewards):8.2f}")
            for metric_name, metric_value in metrics.items():
                if metric_name == "reward":
                    continue
                parts.append(f"{metric_name} {metric_value:.4f}")
            print(" | ".join(parts))


class MatplotlibLogger(BaseLogger):
    """Accumulates per-episode metrics and saves a multi-panel PNG on close()."""

    def __init__(self, save_path: str = "results.png", smooth_window: int = 10):
        self.save_path = save_path
        self.smooth_window = smooth_window
        self._history: Dict[str, List[float]] = {}

    def log_episode(self, episode: int, step: int, metrics: Dict[str, float]) -> None:
        for metric_name, metric_value in metrics.items():
            self._history.setdefault(metric_name, []).append(float(metric_value))

    def close(self) -> None:
        import matplotlib.pyplot as plt

        keys = list(self._history.keys())
        if not keys:
            return

        fig, axes = plt.subplots(1, len(keys), figsize=(6 * len(keys), 5))
        if len(keys) == 1:
            axes = [axes]

        for ax, key in zip(axes, keys):
            values = self._history[key]
            ax.plot(values, alpha=0.4, label=key)
            if len(values) >= self.smooth_window:
                smoothed = np.convolve(
                    values,
                    np.ones(self.smooth_window) / self.smooth_window,
                    mode="valid",
                )
                ax.plot(
                    range(self.smooth_window - 1, len(values)),
                    smoothed,
                    linewidth=2,
                    label=f"Smoothed ({self.smooth_window})",
                )
            ax.set_xlabel("Episode")
            ax.set_ylabel(key)
            ax.set_title(key)
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.save_path)
        plt.close()
        print(f"Plot saved to {self.save_path}")


class CompositeLogger(BaseLogger):
    """Fans out log calls to a list of loggers (console + plot, etc.)."""

    def __init__(self, loggers: List[BaseLogger]):
        self._loggers = loggers

    def log_episode(self, episode: int, step: int, metrics: Dict[str, float]) -> None:
        for logger in self._loggers:
            logger.log_episode(episode, step, metrics)

    def close(self) -> None:
        for logger in self._loggers:
            logger.close()
