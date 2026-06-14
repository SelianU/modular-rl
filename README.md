# lattice

`lattice` is a compact PyTorch toolkit for building neural networks, training
supervised models, and assembling reinforcement learning agents.

The main idea is simple:

```text
make a model -> train it directly, or wrap it as an RL agent
```

## Quick Start

Install the package in editable mode:

```bash
python -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e .
```

Create and train a supervised model:

```python
import torch
from torch.utils.data import DataLoader, TensorDataset

from lattice import SupervisedTrainingConfig, make_mlp, train_supervised_model

inputs = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
targets = torch.tensor([0, 1, 1, 0])
train_loader = DataLoader(TensorDataset(inputs, targets), batch_size=4, shuffle=True)

model = make_mlp(input_dim=2, output_dim=2, hidden_dims=[16, 16])
history = train_supervised_model(
    model=model,
    train_loader=train_loader,
    config=SupervisedTrainingConfig(epochs=100, learning_rate=0.05),
)
```

Run a reinforcement learning baseline:

```python
from lattice import quick_dqn

trainer = quick_dqn("CartPole-v1", total_timesteps=30000)
history = trainer.train()
print(history.last("reward"))
```

## Choose Your Level

`lattice` is organized so you can stop at the level you need.

```text
Level 1: Quick APIs
  Use make_mlp, train_supervised_model, quick_dqn.

Level 2: Builder APIs
  Use build_agent or build_trainer when you want to choose models and configs.

Level 3: Prebuilt Models
  Build PyTorch modules yourself, then pass them into RL agents.

Level 4: Hooks
  Customize one training segment without rewriting the full loop.

Level 5: Internals
  Extend registries, buffers, update functions, or agents.
```

Start with Level 1. Move down only when you need more control.
Advanced extension examples live in
[`docs/advanced-customization.md`](docs/advanced-customization.md).

## Neural Networks

Use beginner-friendly builders when you do not want to wire every `torch.nn`
layer by hand.

```python
from lattice import make_cnn_mlp, make_mlp, make_transformer

mlp = make_mlp(input_dim=10, output_dim=3, hidden_dims=[64, 64])

cnn = make_cnn_mlp(
    input_shape=(1, 28, 28),
    output_dim=10,
    conv_channels=[32, 64],
    mlp_hidden_dims=[128],
)

transformer = make_transformer(input_dim=28, output_dim=10, max_seq_len=28)
```

For config-driven construction, use `build_model`:

```python
from lattice import build_model

model = build_model({
    "type": "sequential",
    "input_shape": (1, 28, 28),
    "layers": [
        {"type": "conv2d", "out_channels": 32, "kernel_size": 3, "padding": 1},
        {"type": "relu"},
        {"type": "flatten"},
        {"type": "linear", "out_features": 128},
        {"type": "relu"},
        {"type": "linear", "out_features": 10},
    ],
})
```

## Supervised Training

The default supervised trainer is intentionally small.

```python
from lattice import SupervisedTrainingConfig, train_supervised_model

history = train_supervised_model(
    model=model,
    train_loader=train_loader,
    validation_loader=validation_loader,
    config=SupervisedTrainingConfig(
        loss="cross_entropy",
        optimizer="adam",
        learning_rate=1e-3,
        epochs=10,
    ),
)

print(history.as_dict())
print(history.last("validation_accuracy"))
```

When you need a custom batch update, pass a custom training step:

```python
from lattice.training import BatchMetrics

def run_custom_training_step(model, batch, context):
    inputs, targets = batch
    inputs = inputs.to(context.device)
    targets = targets.to(context.device)

    outputs = model(inputs)
    loss = context.loss_fn(outputs, targets)

    context.optimizer.zero_grad()
    loss.backward()
    context.optimizer.step()

    return BatchMetrics(loss=loss.item(), num_samples=targets.shape[0])

history = train_supervised_model(
    model=model,
    train_loader=train_loader,
    training_step_function=run_custom_training_step,
)
```

## Reinforcement Learning

For a quick Gymnasium experiment:

```python
from lattice import quick_dqn

quick_dqn("CartPole-v1").train()
```

If your environment loop lives outside `lattice`, create only the agent:

```python
from lattice import make_dqn_agent

agent = make_dqn_agent(
    state_dim=4,
    action_dim=2,
    hidden_dims=[128, 128],
    total_timesteps=30000,
    learning_starts=1000,
)

action = agent.select_action(state)
next_state, reward, done, info = external_system_step(action)
agent.observe(state, action, reward, next_state, done)
metrics = agent.update()
```

For explicit agent construction:

```python
from lattice.algorithms import build_agent

agent = build_agent({
    "algorithm": "dqn",
    "state_dim": 4,
    "action_dim": 2,
    "model": {"backbone": {"type": "mlp", "hidden_dims": [128, 128]}},
    "config": {"total_timesteps": 30000, "learning_starts": 1000},
})
```

Then use the agent with any external system:

```python
action = agent.select_action(state)
next_state, reward, done, info = external_system_step(action)
agent.observe(state, action, reward, next_state, done)
metrics = agent.update()
```

Continuous-control agents have the same beginner-friendly shape:

```python
from lattice import make_ppo_agent, make_sac_agent, make_td3_agent

ppo_agent = make_ppo_agent(state_dim=4, action_dim=2)
sac_agent = make_sac_agent(state_dim=3, action_dim=1)
td3_agent = make_td3_agent(state_dim=3, action_dim=1)
```

For one-step convenience:

```python
from lattice.training import run_interaction_step

result = run_interaction_step(
    agent=agent,
    state=state,
    transition_function=external_system_step,
)
```

If you want `lattice` to own the environment loop:

```python
from lattice.training import build_trainer

trainer = build_trainer({
    "algorithm": "dqn",
    "env": {"type": "gym", "name": "CartPole-v1"},
    "model": {"backbone": {"type": "mlp", "hidden_dims": [128, 128]}},
    "config": {"total_timesteps": 30000, "learning_starts": 1000},
})

trainer.train()
```

`trainer.train()` returns an `RLTrainingHistory`, so supervised and RL training
results can both be inspected with `history.as_dict()` and `history.last(name)`.

## Changing DQN Backbones

Use `state_dim` for vector observations.

```python
agent = build_agent({
    "algorithm": "dqn",
    "state_dim": 4,
    "action_dim": 2,
    "model": {"backbone": {"type": "mlp", "hidden_dims": [128, 128]}},
})
```

Use `input_shape` for image observations.

```python
agent = build_agent({
    "algorithm": "dqn",
    "input_shape": (1, 84, 84),
    "action_dim": 4,
    "model": {
        "backbone": {
            "type": "cnn",
            "channels": [32, 64, 64],
            "feature_dim": 256,
        }
    },
})
```

Use `input_shape=(sequence_length, feature_dim)` for sequence observations.

```python
agent = build_agent({
    "algorithm": "dqn",
    "input_shape": (16, 32),
    "action_dim": 5,
    "model": {
        "backbone": {
            "type": "transformer",
            "hidden_dims": [128],
            "embed_dim": 128,
            "num_heads": 4,
            "num_layers": 2,
        }
    },
    "config": {"sequence_length": 16},
})
```

## Prebuilt Models

You can build the neural network first, then pass it into an RL agent.

```python
from lattice import make_mlp
from lattice.algorithms import build_agent

q_network = make_mlp(input_dim=4, output_dim=2, hidden_dims=[128, 128])

agent = build_agent({
    "algorithm": "dqn",
    "action_dim": 2,
    "model": q_network,
})
```

For more control, pass both the online and target Q-networks:

```python
agent = build_agent({
    "algorithm": "dqn",
    "action_dim": 2,
    "model": {
        "q_network": q_network,
        "target_q_network": target_q_network,
    },
})
```

Actor-critic algorithms accept prebuilt actor and critic modules:

```python
agent = build_agent({
    "algorithm": "ppo",
    "action_dim": 2,
    "model": {
        "actor": actor,
        "critic": critic,
    },
})
```

## Customization With Hooks

Hooks let you customize one segment of a training loop without copying the
whole trainer.

RL hook example:

```python
from lattice.training import run_interaction_step

def process_reward(value, transition, context):
    return value + 0.1 * custom_bonus(transition.next_state)

result = run_interaction_step(
    agent=agent,
    state=state,
    transition_function=external_system_step,
    hooks={"process_reward": process_reward},
)
```

Common RL hook names:

```text
on_episode_start
process_state
process_next_state
select_action
process_action
process_reward
on_transition
before_update
after_update
on_episode_end
```

Supervised hook example:

```python
from lattice import train_supervised_model

def compute_loss(model, outputs, targets, loss_fn, context):
    return loss_fn(outputs, targets) + regularization_term(model)

history = train_supervised_model(
    model=model,
    train_loader=train_loader,
    hooks={"compute_loss": compute_loss},
)
```

Common supervised hook names:

```text
before_epoch
before_batch
process_batch
compute_loss
before_backward
after_backward
before_optimizer_step
after_optimizer_step
after_batch
after_epoch
```

Unknown hook names raise a `ValueError`, so typos fail early.

## Advanced Internals

Use lower-level modules when you want to inspect or extend internals.

```python
from lattice.algorithms.updates import DQNUpdateBatch, run_dqn_update
from lattice.training import Registry
```

`run_dqn_update` exposes the DQN update rule directly. `Registry` lets you add
project-specific backbones, loggers, optimizers, buffers, and environments.

## Structure

```text
lattice/
├── networks/      # Model builders, encoders, heads
├── algorithms/    # RL agents, algorithm-specific builders, buffers, updates
└── training/      # Supervised/RL training loops, histories, hooks, loggers
```

Naming and file creation rules live in
[`docs/naming-conventions.md`](docs/naming-conventions.md).
Advanced customization examples live in
[`docs/advanced-customization.md`](docs/advanced-customization.md).

## Requirements

```text
torch>=2.0.0
gymnasium[classic-control]>=0.28.1
numpy>=1.22.0
matplotlib>=3.5.0
```

For development and tests:

```bash
.venv/bin/python -m pip install -e ".[dev]"
.venv/bin/pytest
```

## License

[MIT License](LICENSE)
