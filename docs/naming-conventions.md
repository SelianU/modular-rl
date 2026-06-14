# Naming Conventions

This project favors explicit, beginner-friendly names over short abbreviations.
Names should make the role, shape, or ownership of a value clear without forcing
the reader to inspect the implementation.

## Files and Modules

- Use lowercase `snake_case` file names.
- Use module names that describe one responsibility.
- Prefer nouns for component modules: `buffers.py`, `models.py`, `registry.py`.
- Prefer action-oriented names for builder modules: `builders.py`, `factory.py`.
- Prefer explicit domain names when a generic name becomes unclear:
  - `optimizers.py` instead of `optim.py`
  - `training_steps.py` instead of `steps.py`
  - `supervised_training.py` instead of `supervised.py`
- Keep examples named by algorithm and model family:
  - `train_dqn_mlp.py`
  - `train_dqn_cnn.py`
  - `train_sac_mlp.py`

Current core module layout:

```text
lattice/networks/
├── builders.py            # make_mlp, make_mlp_classifier, make_cnn_mlp, build_model
├── sequence_builders.py   # make_rnn, make_transformer
├── language_models.py     # MiniGPT, make_mini_gpt
└── sequential.py          # build_sequential_model

lattice/training/
├── optimizers.py          # make_loss, make_optimizer
├── hooks.py               # HookManager, RLHookContext, SupervisedHookContext
├── interaction.py         # run_interaction_step
├── training_steps.py      # run_training_step, run_evaluation_step
├── supervised_training.py # SupervisedTrainingConfig, train_supervised_model
├── rl_trainer.py          # RLTrainer
└── builders.py            # build_trainer, ExperimentBuilder

lattice/algorithms/
├── builders.py            # AgentBuildContext, AgentBuilder, build_agent
├── agents/                # DQNAgent, SACAgent, PPOAgent, TD3Agent
├── updates/               # run_dqn_update, run_recurrent_dqn_update
├── buffers.py             # ReplayBuffer, RolloutBuffer
├── config.py              # DQNConfig, SACConfig, PPOConfig, TD3Config
└── models.py              # QNetwork, actor, critic modules
```

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
  - `*Trainer`: training loops, such as `RLTrainer` or `RewardModelTrainer`.
  - `*Builder`: object assembly helpers, such as `AgentBuilder` or `ExperimentBuilder`.
  - `*BuildContext`: explicit build-time shape and space metadata, such as `AgentBuildContext`.
  - `*HookContext`: runtime hook metadata, such as `RLHookContext`.
  - `*UpdateBatch`: immutable inputs for one algorithm update, such as `DQNUpdateBatch`.
  - `*UpdateMetrics`: outputs from one algorithm update, such as `DQNUpdateMetrics`.

## Public Functions

- Use lowercase `snake_case`.
- Start with a clear verb.
- Use these prefixes consistently:
  - `make_*`: beginner-friendly model constructors.
    - Example: `make_mlp`, `make_cnn_mlp`
  - `build_*`: assemble configurable components.
    - Example: `build_model`, `build_agent`, `build_trainer`, `build_q_network`
  - `quick_*`: shortcut helpers for common training setups.
    - Example: `quick_dqn`, `quick_sac`
  - `register_*`: add a component to `Registry`.
    - Example: `register_backbone`, `register_logger`
  - `list_*`: return available registry keys.
    - Example: `list_backbones`, `list_optimizers`
  - `run_*`: execute one explicit operation or step.
    - Example: `run_training_step`, `run_interaction_step`, `run_dqn_update`

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
| `fn` | `function`, or a precise callable role such as `training_step_function` |

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

For RL agents:

- Use `state_dim` for one-dimensional vector observations.
- Use `input_shape` for image or sequence observations:
  - CNN: `(channels, height, width)`
  - Transformer over vectors: `(sequence_length, feature_dim)`
  - Transformer over images: `(sequence_length, channels, height, width)`
- Keep `config.sequence_length` aligned with the sequence length used by RNN or
  Transformer DQN.

## Config Dictionaries

- Use `*_config` for structured settings.
- Use `*_spec` for dictionary inputs that describe how to build an object.
- Use explicit names for prebuilt RL modules:
  - `q_network`
  - `target_q_network`
  - `actor`
  - `critic`
  - `actor_target`
  - `critic_target`

Examples:

```python
backbone_config = {"type": "mlp", "hidden_dims": [128, 128]}
trainer_spec = {"algorithm": "dqn", "model": {"backbone": backbone_config}}
algorithm_config = DQNConfig(total_timesteps=30000)
training_config = SupervisedTrainingConfig(epochs=10, learning_rate=1e-3)
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

## Pre-release API Changes

This project is still pre-release. Prefer one clear public API over old-name
aliases. If a public name must change before release:

1. Move the code to the new name.
2. Update README and naming conventions.
3. Update internal imports.
4. Verify the new public API.
