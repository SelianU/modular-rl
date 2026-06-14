import os
from typing import Dict, List, Optional
import numpy as np

from .env_wrapper import BaseEnv
from .hooks import HookManager, HookSpec, RLHookContext, RLTransition, RL_HOOK_NAMES
from .logger import BaseLogger, ConsoleLogger


class RLTrainer:
    """
    Unified reinforcement learning loop for DQN, SAC, PPO, and TD3 agents.

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
        hooks: HookSpec = None,
    ):
        self.agent = agent
        self.env = env
        self.config = config
        self.logger = logger or ConsoleLogger(
            log_interval=getattr(config, "log_interval", 10)
        )
        self.save_path = save_path
        self.hooks = HookManager(hooks, allowed_names=RL_HOOK_NAMES)

    def add_hook(self, name: str, hook) -> None:
        self.hooks.add(name, hook)

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
        hook_context = RLHookContext(
            agent=self.agent,
            environment=self.env,
            config=self.config,
        )
        self.hooks.run("on_episode_start", state=state, context=hook_context)

        for global_step in range(1, self.config.total_timesteps + 1):
            hook_context.global_step = global_step
            hook_context.episode = curr_episode + 1

            processed_state = self.hooks.transform(
                "process_state",
                state,
                context=hook_context,
            )
            custom_action = self.hooks.first(
                "select_action",
                agent=self.agent,
                state=processed_state,
                context=hook_context,
            )
            action = (
                custom_action
                if custom_action is not None
                else self.agent.select_action(processed_state, evaluation=False)
            )
            action = self.hooks.transform("process_action", action, context=hook_context)

            next_state, reward, done, info = self.env.step(action)
            next_state = self.hooks.transform(
                "process_next_state",
                next_state,
                state=processed_state,
                action=action,
                context=hook_context,
            )
            transition = RLTransition(
                state=processed_state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done,
                info=info,
            )
            reward = self.hooks.transform(
                "process_reward",
                reward,
                transition=transition,
                context=hook_context,
            )
            transition.reward = reward
            episode_reward += reward

            self.hooks.run("on_transition", transition=transition, context=hook_context)
            self.agent.observe(
                transition.state,
                transition.action,
                transition.reward,
                transition.next_state,
                transition.done,
            )
            state = transition.next_state

            self.hooks.run("before_update", agent=self.agent, context=hook_context)
            update_metrics = self.agent.update()
            hook_context.metrics = update_metrics
            self.hooks.run("after_update", metrics=update_metrics, context=hook_context)

            for k, v in update_metrics.items():
                step_buf.setdefault(k, []).append(v)

            if done:
                curr_episode += 1
                log_metrics = {"reward": episode_reward}
                for k, vs in step_buf.items():
                    log_metrics[k] = float(np.mean(vs))

                self.logger.log_episode(curr_episode, global_step, log_metrics)
                for k, v in log_metrics.items():
                    history.setdefault(k, []).append(v)

                hook_context.metrics = log_metrics
                self.hooks.run("on_episode_end", metrics=log_metrics, context=hook_context)

                eval_interval = getattr(self.config, "eval_interval", 0)
                if eval_interval > 0 and curr_episode % eval_interval == 0:
                    eval_result = self.evaluate(n_episodes=5)
                    eval_metrics = {"eval_reward": eval_result["mean_reward"]}
                    self.logger.log_episode(curr_episode, global_step, eval_metrics)
                    for k, v in eval_metrics.items():
                        history.setdefault(k, []).append(v)

                state = self.env.reset()
                self._maybe_reset_hidden()
                hook_context.episode = curr_episode + 1
                self.hooks.run("on_episode_start", state=state, context=hook_context)
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
        hook_context = RLHookContext(
            agent=self.agent,
            environment=self.env,
            config=self.config,
            evaluation=True,
        )
        for episode_index in range(1, n_episodes + 1):
            hook_context.episode = episode_index
            state = self.env.reset()
            self._maybe_reset_hidden()
            self.hooks.run("on_episode_start", state=state, context=hook_context)
            ep_reward = 0.0
            done = False
            while not done:
                processed_state = self.hooks.transform(
                    "process_state",
                    state,
                    context=hook_context,
                )
                custom_action = self.hooks.first(
                    "select_action",
                    agent=self.agent,
                    state=processed_state,
                    context=hook_context,
                )
                action = (
                    custom_action
                    if custom_action is not None
                    else self.agent.select_action(processed_state, evaluation=True)
                )
                action = self.hooks.transform("process_action", action, context=hook_context)
                next_state, reward, done, info = self.env.step(action)
                next_state = self.hooks.transform(
                    "process_next_state",
                    next_state,
                    state=processed_state,
                    action=action,
                    context=hook_context,
                )
                transition = RLTransition(
                    state=processed_state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    done=done,
                    info=info,
                )
                reward = self.hooks.transform(
                    "process_reward",
                    reward,
                    transition=transition,
                    context=hook_context,
                )
                transition.reward = reward
                state = transition.next_state
                ep_reward += reward
            hook_context.metrics = {"eval_reward": ep_reward}
            self.hooks.run("on_episode_end", metrics=hook_context.metrics, context=hook_context)
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
