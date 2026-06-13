# Config-Driven Builders Design

## Goal

Make `modular_rl` easier to extend by adding a config-driven assembly layer that can build trainers from plain dictionaries while preserving the existing `quick_dqn`, `quick_sac`, `quick_ppo`, and `quick_td3` APIs.

## Architecture

The package keeps the current separation between `networks`, `algorithms`, and `training`. A new `training.builders` module is the assembly boundary: it reads a dictionary, builds the environment, algorithm config, networks, buffers, optimizers, logger, agent, and `Trainer`.

`training.registry.Registry` remains the extensibility point for component lookup. It is expanded so all built-in algorithms, configs, buffers, heads, env wrappers, loggers, and optimizers can be registered and introspected. Existing specialized helpers such as `build_q_network` remain available for compatibility.

## Public API

The new main API is:

```python
from modular_rl.training import build_trainer

trainer = build_trainer({
    "algorithm": "dqn",
    "env": {"type": "gym", "name": "CartPole-v1"},
    "model": {"backbone": {"type": "mlp", "hidden_dims": [64, 64]}},
    "config": {"total_timesteps": 1000, "learning_starts": 10},
    "logger": {"type": "console"},
})
```

The dictionary is intentionally minimal. Defaults match the existing quick factories wherever possible. Users can still pass already-built custom loggers to the existing quick factories.

## Components

- `ExperimentBuilder`: owns the high-level `build_trainer(spec)` flow.
- `BuildContext`: stores derived values from the environment and config, such as state dimension, action dimension, action bounds, and device.
- `Registry`: stores classes and factories by string key.
- `build_trainer(spec)`: convenience function for the common path.

The builder supports DQN, SAC, PPO, and TD3 first. Custom algorithms can be registered later by adding a new builder function without changing `Trainer`.

## Data Flow

1. Normalize the input dictionary and validate required keys.
2. Build an environment from `env.type`.
3. Build the algorithm dataclass from `algorithm` and `config`.
4. Build model parts from `model`.
5. Build replay or rollout buffers from algorithm defaults or explicit `buffer`.
6. Build optimizers from `optimizer` settings or algorithm defaults.
7. Build logger from `logger`.
8. Return `Trainer(agent, env, config, logger, save_path)`.

## Error Handling

Unknown registry keys raise `KeyError` with available options. Invalid algorithm/env combinations raise `ValueError` with a message that names the algorithm and expectation, such as requiring continuous actions for SAC and TD3.

## Testing

Tests cover:

- DQN trainer construction from a minimal dictionary.
- DQN RNN construction selecting `SequenceReplayBuffer` and recurrent mode.
- SAC/PPO/TD3 trainer construction through the same API.
- Clear errors for unknown algorithm keys.
- Registry introspection showing the newly registered component groups.

The tests focus on construction, not long training runs.
