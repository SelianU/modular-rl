from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Union


HookFunction = Callable[..., Any]
HookSpec = Optional[Dict[str, Union[Iterable[HookFunction], HookFunction]]]


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

    def __init__(self, hooks: HookSpec = None):
        self._hooks: Dict[str, List[HookFunction]] = {}
        if hooks:
            for name, hook_or_hooks in hooks.items():
                self.add(name, hook_or_hooks)

    def add(self, name: str, hook_or_hooks: Union[Iterable[HookFunction], HookFunction]) -> None:
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
