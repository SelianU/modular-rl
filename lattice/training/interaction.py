from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np

from .hooks import HookManager, HookSpec, RLHookContext, RLTransition, RL_HOOK_NAMES


TransitionResult = Union[
    Tuple[np.ndarray, float, bool],
    Tuple[np.ndarray, float, bool, Dict[str, Any]],
]
TransitionFunction = Callable[[Any], TransitionResult]


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
    hooks: HookSpec = None,
    context: Optional[RLHookContext] = None,
) -> InteractionStep:
    """
    Run one state -> action -> external transition -> observe/update step.

    The transition_function owns the outside world. It receives the action and
    returns (next_state, reward, done).
    """
    hook_manager = HookManager(hooks, allowed_names=RL_HOOK_NAMES)
    hook_context = context or RLHookContext(
        agent=agent,
        environment=None,
        config=getattr(agent, "config", None),
        evaluation=evaluation,
    )

    processed_state = hook_manager.transform("process_state", state, context=hook_context)
    custom_action = hook_manager.first(
        "select_action",
        agent=agent,
        state=processed_state,
        context=hook_context,
    )
    action = custom_action if custom_action is not None else agent.select_action(
        processed_state,
        evaluation=evaluation,
    )
    action = hook_manager.transform("process_action", action, context=hook_context)
    next_state, reward, done, info = _unpack_transition_result(transition_function(action))
    next_state = hook_manager.transform(
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
    reward = hook_manager.transform(
        "process_reward",
        reward,
        transition=transition,
        context=hook_context,
    )
    transition.reward = reward

    metrics = {}
    if not evaluation:
        hook_manager.run("on_transition", transition=transition, context=hook_context)
        agent.observe(
            transition.state,
            transition.action,
            transition.reward,
            transition.next_state,
            transition.done,
        )
        if update:
            hook_manager.run("before_update", agent=agent, context=hook_context)
            metrics = agent.update()
            hook_context.metrics = metrics
            hook_manager.run("after_update", metrics=metrics, context=hook_context)

    return InteractionStep(
        action=action,
        next_state=next_state,
        reward=reward,
        done=done,
        metrics=metrics,
    )


def _unpack_transition_result(result: TransitionResult) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
    if len(result) == 3:
        next_state, reward, done = result
        return next_state, float(reward), bool(done), {}
    if len(result) == 4:
        next_state, reward, done, info = result
        return next_state, float(reward), bool(done), dict(info)
    raise ValueError(
        "transition_function must return (next_state, reward, done) "
        "or (next_state, reward, done, info)."
    )
