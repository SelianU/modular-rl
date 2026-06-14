# Adding a New RL Algorithm

This guide shows the smallest path for adding a new algorithm to `lattice`.

The core idea is:

```text
Config -> Agent -> AgentBuilder -> Registry -> build_agent
```

If your agent implements the common interface, the existing `RLTrainer`,
history, hooks, and loggers can reuse it.

## 1. Define a Config

Use a dataclass for algorithm settings.

```python
from dataclasses import dataclass

@dataclass
class A2CConfig:
    total_timesteps: int = 30000
    learning_rate: float = 3e-4
    gamma: float = 0.99
    device: str = "cpu"
```

Register it:

```python
from lattice.training import Registry

Registry.register_config("a2c", A2CConfig)
```

## 2. Implement the Agent Interface

An agent must provide these methods:

```python
class A2CAgent:
    def select_action(self, state, evaluation=False):
        ...

    def observe(self, state, action, reward, next_state, done):
        ...

    def update(self):
        return {"loss": 0.0}

    def save(self, filepath):
        ...
```

`RLTrainer` only relies on that interface. The algorithm can store rollouts,
sample replay, update every step, or update every episode internally.

## 3. Create an AgentBuilder

The builder receives an `AgentBuildContext` and the original spec dictionary.
It should assemble models, buffers, optimizers, losses, and the agent.

```python
from lattice.algorithms import AgentBuildContext

class A2CAgentBuilder:
    def __init__(self, registry=None):
        self.registry = registry

    def build(self, context: AgentBuildContext, spec):
        model_spec = spec.get("model", {})
        hidden_dims = model_spec.get("backbone", {}).get("hidden_dims", [64, 64])

        # Build actor, critic, optimizer, buffer, then return your agent.
        return A2CAgent(
            action_dim=context.action_dim,
            config=context.config,
        )
```

Register it:

```python
Registry.register_agent_builder("a2c", A2CAgentBuilder)
```

## 4. Build the Agent

After registration, `build_agent` can construct the algorithm by name.

```python
from lattice.algorithms import build_agent

agent = build_agent({
    "algorithm": "a2c",
    "state_dim": 4,
    "action_dim": 2,
    "config": {"total_timesteps": 30000},
    "model": {"backbone": {"type": "mlp", "hidden_dims": [64, 64]}},
})
```

## 5. Use the Existing Trainer

If the agent follows the common interface, it can run in `RLTrainer`.

```python
from lattice.training import RLTrainer

trainer = RLTrainer(
    agent=agent,
    env=environment,
    config=agent.config,
)

history = trainer.train()
```

## What to Reuse

Reuse existing pieces where they fit:

| Need | Existing Tool |
|---|---|
| Vector/image/sequence feature extractor | `lattice.networks` |
| Replay data | `ReplayBuffer` or `SequenceReplayBuffer` |
| Rollout data | `RolloutBuffer` |
| Optimizer creation | `Registry.build_optimizer` |
| Logging and histories | `RLTrainer` |
| Reward/state/action customization | RL hooks |

## When to Add a Direct Update Function

If the algorithm has a meaningful gradient update that should be studied or
tested independently, put it under `lattice/algorithms/updates/`.

Example:

```text
lattice/algorithms/updates/a2c_update.py
```

Then the agent can collect data and delegate the gradient math:

```python
metrics = run_a2c_update(...)
```

This keeps algorithm math easier to test than a full environment loop.

## Minimal File Checklist

For a built-in algorithm, add:

```text
lattice/algorithms/config.py              # A2CConfig
lattice/algorithms/agents/a2c.py          # A2CAgent
lattice/algorithms/agent_builders/a2c_builder.py
lattice/algorithms/updates/a2c_update.py  # optional but recommended
```

For an external project, you can keep the files outside `lattice` and only
register the config and builder at runtime.
