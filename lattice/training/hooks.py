from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Union


HookFunction = Callable[..., Any]
HookSpec = Optional[Dict[str, Union[Iterable[HookFunction], HookFunction]]]

RL_HOOK_NAMES = {
    "on_episode_start",
    "process_state",
    "process_next_state",
    "select_action",
    "process_action",
    "process_reward",
    "on_transition",
    "before_update",
    "after_update",
    "on_episode_end",
}

SUPERVISED_HOOK_NAMES = {
    "before_epoch",
    "before_batch",
    "process_batch",
    "compute_loss",
    "before_backward",
    "after_backward",
    "before_optimizer_step",
    "after_optimizer_step",
    "after_batch",
    "after_epoch",
}


@dataclass
class RLTransition:
    """One reinforcement learning transition."""

    state: Any
    action: Any
    reward: float
    next_state: Any
    done: bool
    info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RLHookContext:
    """Runtime information passed to RL hooks."""

    agent: Any
    environment: Any
    config: Any
    global_step: int = 0
    episode: int = 0
    metrics: Dict[str, Any] = field(default_factory=dict)
    evaluation: bool = False


@dataclass
class SupervisedHookContext:
    """Runtime information passed to supervised training hooks."""

    model: Any
    config: Any
    epoch: int = 0
    batch_index: int = 0
    metrics: Dict[str, Any] = field(default_factory=dict)
    validation: bool = False


class HookManager:
    """Named hook registry for optional training customization."""

    def __init__(self, hooks: HookSpec = None, allowed_names: Optional[Iterable[str]] = None):
        self._hooks: Dict[str, List[HookFunction]] = {}
        self._allowed_names: Optional[Set[str]] = set(allowed_names) if allowed_names is not None else None
        if hooks:
            for name, hook_or_hooks in hooks.items():
                self.add(name, hook_or_hooks)

    def add(self, name: str, hook_or_hooks: Union[Iterable[HookFunction], HookFunction]) -> None:
        self._validate_name(name)
        if callable(hook_or_hooks):
            hooks = [hook_or_hooks]
        else:
            hooks = list(hook_or_hooks)
        self._hooks.setdefault(name, []).extend(hooks)

    def run(self, name: str, **kwargs) -> List[Any]:
        return [hook(**kwargs) for hook in self._hooks.get(name, [])]

    def first(self, name: str, **kwargs) -> Any:
        for hook in self._hooks.get(name, []):
            result = hook(**kwargs)
            if result is not None:
                return result
        return None

    def transform(self, name: str, value: Any, **kwargs) -> Any:
        current_value = value
        for hook in self._hooks.get(name, []):
            result = hook(value=current_value, **kwargs)
            if result is not None:
                current_value = result
        return current_value

    def _validate_name(self, name: str) -> None:
        if self._allowed_names is None or name in self._allowed_names:
            return
        available_names = ", ".join(sorted(self._allowed_names))
        raise ValueError(f"Unknown hook name '{name}'. Available hook names: {available_names}")
