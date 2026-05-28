import os
from typing import Dict, List, Optional
import numpy as np

from .env_wrapper import BaseEnv
from .logger import BaseLogger, ConsoleLogger


class Trainer:
    """
    Unified training loop for DQN and SAC agents.

    Handles:
    - Episode management and step counting
    - Per-episode metric aggregation and logging
    - Hidden-state reset for recurrent agents (DRQN)
    - Model checkpointing
    - Evaluation rollouts

    The agent must implement:
        observe(state, action, reward, next_state, done)
        select_action(state, evaluation) -> action
        update() -> dict of metrics (may be empty)
        save(filepath)
    Recurrent agents optionally implement reset_hidden_state().
    Set config.eval_interval > 0 to run periodic deterministic evaluation
    during training (logged with an "eval_reward" key).
    """

    def __init__(
        self,
        agent,
        env: BaseEnv,
        config,
        logger: Optional[BaseLogger] = None,
        save_path: Optional[str] = None,
    ):
        self.agent = agent
        self.env = env
        self.config = config
        self.logger = logger or ConsoleLogger(
            log_interval=getattr(config, "log_interval", 10)
        )
        self.save_path = save_path

    # ------------------------------------------------------------------ #
    # Training                                                             #
    # ------------------------------------------------------------------ #

    def train(self) -> Dict[str, List[float]]:
        """
        Run the full training loop.
        Returns a dict of episode-level metric histories, e.g.:
        {"reward": [...], "loss": [...], "epsilon": [...]}
        """
        history: Dict[str, List[float]] = {}
        episode_reward = 0.0
        step_buf: Dict[str, List[float]] = {}
        curr_episode = 0

        state = self.env.reset()
        self._maybe_reset_hidden()

        for global_step in range(1, self.config.total_timesteps + 1):
            action = self.agent.select_action(state, evaluation=False)
            next_state, reward, done, _ = self.env.step(action)
            episode_reward += reward

            self.agent.observe(state, action, reward, next_state, done)
            state = next_state

            for k, v in self.agent.update().items():
                step_buf.setdefault(k, []).append(v)

            if done:
                curr_episode += 1
                log_metrics = {"reward": episode_reward}
                for k, vs in step_buf.items():
                    log_metrics[k] = float(np.mean(vs))

                self.logger.log_episode(curr_episode, global_step, log_metrics)
                for k, v in log_metrics.items():
                    history.setdefault(k, []).append(v)

                eval_interval = getattr(self.config, "eval_interval", 0)
                if eval_interval > 0 and curr_episode % eval_interval == 0:
                    eval_result = self.evaluate(n_episodes=5)
                    eval_metrics = {"eval_reward": eval_result["mean_reward"]}
                    self.logger.log_episode(curr_episode, global_step, eval_metrics)
                    for k, v in eval_metrics.items():
                        history.setdefault(k, []).append(v)

                state = self.env.reset()
                self._maybe_reset_hidden()
                episode_reward = 0.0
                step_buf = {}

        self.env.close()

        if self.save_path:
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
            self.agent.save(self.save_path)
            print(f"Model saved to {self.save_path}")

        self.logger.close()
        return history

    # ------------------------------------------------------------------ #
    # Evaluation                                                           #
    # ------------------------------------------------------------------ #

    def evaluate(self, n_episodes: int = 10) -> Dict[str, float]:
        """
        Run deterministic evaluation episodes.
        Returns mean and std of episode rewards.
        """
        rewards = []
        for _ in range(n_episodes):
            state = self.env.reset()
            self._maybe_reset_hidden()
            ep_reward = 0.0
            done = False
            while not done:
                action = self.agent.select_action(state, evaluation=True)
                state, reward, done, _ = self.env.step(action)
                ep_reward += reward
            rewards.append(ep_reward)
        return {
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
        }

    # ------------------------------------------------------------------ #
    # Internal                                                             #
    # ------------------------------------------------------------------ #

    def _maybe_reset_hidden(self):
        if hasattr(self.agent, "reset_hidden_state"):
            self.agent.reset_hidden_state()
