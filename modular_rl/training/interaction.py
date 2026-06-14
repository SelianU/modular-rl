from dataclasses import dataclass
from typing import Any, Callable, Dict, Tuple

import numpy as np


TransitionFunction = Callable[[Any], Tuple[np.ndarray, float, bool]]


@dataclass(frozen=True)
class InteractionStep:
    """Result from one external interaction plus optional agent update."""

    action: Any
    next_state: np.ndarray
    reward: float
    done: bool
    metrics: Dict[str, float]


def run_interaction_step(
    agent,
    state: np.ndarray,
    transition_function: TransitionFunction,
    evaluation: bool = False,
    update: bool = True,
) -> InteractionStep:
    """
    Run one state -> action -> external transition -> observe/update step.

    The transition_function owns the outside world. It receives the action and
    returns (next_state, reward, done).
    """
    action = agent.select_action(state, evaluation=evaluation)
    next_state, reward, done = transition_function(action)

    metrics = {}
    if not evaluation:
        agent.observe(state, action, reward, next_state, done)
        if update:
            metrics = agent.update()

    return InteractionStep(
        action=action,
        next_state=next_state,
        reward=reward,
        done=done,
        metrics=metrics,
    )
