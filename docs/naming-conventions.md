# Naming Conventions

This project favors explicit, beginner-friendly names over short abbreviations.
Names should make the role, shape, or ownership of a value clear without forcing
the reader to inspect the implementation.

## Files and Modules

- Use lowercase `snake_case` file names.
- Use module names that describe one responsibility.
- Prefer nouns for component modules: `buffers.py`, `models.py`, `registry.py`.
- Prefer action-oriented names for builder modules: `builders.py`, `factory.py`.
- Keep examples named by algorithm and model family:
  - `train_dqn_mlp.py`
  - `train_dqn_cnn.py`
  - `train_sac_mlp.py`

## Classes

- Use `PascalCase`.
- Include the component role in the class name.
- Use established suffixes consistently:
  - `*Agent`: algorithm implementations, such as `DQNAgent`.
  - `*Config`: algorithm configuration dataclasses, such as `PPOConfig`.
  - `*Buffer`: replay or rollout storage, such as `ReplayBuffer`.
  - `*Wrapper`: environment or reward wrappers, such as `GymEnvWrapper`.
  - `*Logger`: logging implementations, such as `ConsoleLogger`.
  - `*Head`: neural network output heads, such as `QHead`.

## Public Functions

- Use lowercase `snake_case`.
- Start with a clear verb.
- Use these prefixes consistently:
  - `make_*`: beginner-friendly model constructors.
    - Example: `make_mlp`, `make_cnn_mlp`
  - `build_*`: assemble configurable components.
    - Example: `build_model`, `build_trainer`, `build_q_network`
  - `quick_*`: shortcut helpers for common training setups.
    - Example: `quick_dqn`, `quick_sac`
  - `register_*`: add a component to `Registry`.
    - Example: `register_backbone`, `register_logger`
  - `list_*`: return available registry keys.
    - Example: `list_backbones`, `list_optimizers`

## Variables

- Prefer descriptive names over abbreviations.
- Keep common ML terms only when they are standard and obvious in context.
- Use singular names for one object and plural names for collections.

Preferred names:

| Avoid | Prefer |
|---|---|
| `env` | `environment` |
| `ctx` | `context` |
| `cfg` | `config`, `config_dict`, `backbone_config` |
| `bb` | `backbone` |
| `actor_bb` | `actor_backbone` |
| `critic_bb` | `critic_backbone` |
| `q_net` | `q_network` |
| `target_q_net` | `target_q_network` |
| `s_dim` | `state_dim` |
| `a_dim` | `action_dim` |
| `a_low` | `action_low` |
| `a_high` | `action_high` |
| `buf` | `buffer`, `replay_buffer`, `demonstration_buffer` |
| `ep_r` | `episode_reward` |
| `ep_lens` | `episode_lengths` |

Acceptable short names:

- `x`, `y`: local tensor inputs/outputs in tiny examples.
- `i`, `j`: small loop indices when the loop body is short.
- `idx`: index variables in low-level data structures.
- `rnn`, `cnn`, `mlp`: standard model-family abbreviations.

## Shapes and Dimensions

- Name dimensions by meaning, not by letter:
  - `input_dim`
  - `output_dim`
  - `state_dim`
  - `action_dim`
  - `feature_dim`
  - `hidden_dims`
  - `channels`
  - `height`
  - `width`

Use `input_shape` for image or tensor shapes. For image models, prefer
`(channels, height, width)`.

## Config Dictionaries

- Use `*_config` for structured settings.
- Use `*_spec` for dictionary inputs that describe how to build an object.

Examples:

```python
backbone_config = {"type": "mlp", "hidden_dims": [128, 128]}
trainer_spec = {"algorithm": "dqn", "model": {"backbone": backbone_config}}
algorithm_config = DQNConfig(total_timesteps=30000)
```

## Neural Network Parts

- Use `backbone` for feature extractors.
- Use `head` for the final output module.
- Use `actor` and `critic` for policy/value networks.
- Use `target_*` for target networks.

Examples:

```python
actor_backbone = MLP(input_dim=state_dim, hidden_dims=[256, 256])
critic_backbone = MLP(input_dim=state_dim, hidden_dims=[256, 256])
target_q_network = QNetwork(target_backbone, QHead(target_backbone.output_dim, action_dim))
```

## Examples

Example code should be more explicit than library internals because it teaches
users how to think about the project.

Good:

```python
environment = GymEnvWrapper(config.env_name)
state_dim = environment.state_dim
action_dim = environment.action_dim
q_network = QNetwork(backbone, QHead(backbone.output_dim, action_dim))
```

Avoid:

```python
env = GymEnvWrapper(config.env_name)
s_dim = env.state_dim
a_dim = env.action_dim
q_net = QNetwork(bb, QHead(bb.output_dim, a_dim))
```

## Compatibility

Do not rename public APIs casually. If a public name must change:

1. Add the new name.
2. Keep the old name as a compatibility alias.
3. Document the migration path.
4. Remove the old name only in a planned breaking release.
