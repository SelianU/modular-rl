# Advanced Customization

Start with the top-level helpers first:

```python
from lattice import make_dqn_agent, make_mlp, train_supervised_model
```

Move to the tools below only when you need to replace one part of the default
behavior.

## Which Extension Point Should I Use?

| Goal | Use |
|---|---|
| Change a neural network architecture | Pass a prebuilt PyTorch model |
| Change one step in a training loop | Use hooks |
| Replace the full supervised batch update | Use `training_step_function` |
| Reuse a custom component by name | Register it with `Registry` |
| Add a new RL algorithm | Register a config and agent builder |
| Study or test an algorithm update directly | Use `lattice.algorithms.updates` |

## Custom PyTorch Models

`lattice` agents accept regular `torch.nn.Module` objects.

For DQN, pass a Q-network that maps `state -> action_values`:

```python
import torch.nn as nn

from lattice import make_dqn_agent

q_network = nn.Sequential(
    nn.Linear(4, 128),
    nn.ReLU(),
    nn.Linear(128, 2),
)

agent = make_dqn_agent(
    action_dim=2,
    model=q_network,
    total_timesteps=30000,
)
```

For actor-critic algorithms, pass the expected modules explicitly:

```python
from lattice.algorithms import build_agent

agent = build_agent({
    "algorithm": "ppo",
    "state_dim": 4,
    "action_dim": 2,
    "model": {
        "actor": actor,
        "critic": critic,
    },
})
```

Use this when PyTorch already gives you the cleanest model code.

## Supervised Hooks

Hooks customize one segment of the default supervised training loop.

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

Available supervised hook names:

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

Unknown hook names raise `ValueError`, so typo mistakes fail early.

## RL Hooks

RL hooks customize the interaction loop without replacing the agent.

```python
from lattice.training import run_interaction_step

def process_reward(value, transition, context):
    return value + reward_bonus(transition.next_state)

result = run_interaction_step(
    agent=agent,
    state=state,
    transition_function=external_system_step,
    hooks={"process_reward": process_reward},
)
```

Available RL hook names:

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

Use hooks when the default loop is almost right.

## Custom Supervised Training Step

Use `training_step_function` when the whole batch update is different.

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
```

Then pass it into the trainer:

```python
history = train_supervised_model(
    model=model,
    train_loader=train_loader,
    training_step_function=run_custom_training_step,
)
```

## Registry

Use `Registry` when you want to refer to a project-specific component by name.

```python
import torch
import torch.nn as nn

from lattice import make_dqn_agent
from lattice.training import Registry

class TinyBackbone(nn.Module):
    output_dim = 8

    def __init__(self, input_dim, hidden_dims=None):
        super().__init__()
        self.network = nn.Linear(input_dim, self.output_dim)

    def forward(self, inputs):
        return torch.relu(self.network(inputs))

Registry.register_backbone("tiny", TinyBackbone)

agent = make_dqn_agent(
    state_dim=4,
    action_dim=2,
    backbone="tiny",
)
```

Registry is best for repeated components. For one-off models, passing a
prebuilt PyTorch module is usually simpler.

## New RL Algorithms

Registering an agent builder lets `build_agent` construct algorithms that are
not built into `lattice`.

```python
from lattice.training import Registry

Registry.register_config("my_algo", MyConfig)
Registry.register_agent_builder("my_algo", MyAgentBuilder)
```

Then:

```python
from lattice.algorithms import build_agent

agent = build_agent({
    "algorithm": "my_algo",
    "state_dim": 4,
    "action_dim": 2,
})
```

See [`new-rl-algorithm.md`](new-rl-algorithm.md) for the full checklist.

## Direct Update Functions

Algorithm update functions are exposed for testing and study.

```python
from lattice.algorithms.updates import DQNUpdateBatch, run_dqn_update
```

This is the lowest-level path. Use it when you want to test the Bellman update
itself instead of running a full environment loop.
