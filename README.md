# modular-rl

`modular-rl` is a compact PyTorch-based reinforcement learning and neural
network library. The repository now keeps only the core package code under
`modular_rl/` plus the root files needed for installation and packaging.

## Structure

```text
modular_rl/
├── networks/               # Neural network builders and reusable blocks
│   ├── encoders/           # MLP, CNN, RNN, Transformer backbones
│   ├── heads.py            # Q, policy, critic, and value heads
│   ├── builders.py         # MLP/CNN model builders and build_model
│   ├── sequence_builders.py # RNN/Transformer beginner builders
│   ├── language_models.py   # MiniGPT and language model helpers
│   ├── sequential.py        # Dict-driven nn.Sequential builder
│   └── simple.py            # Compatibility exports for older imports
│
├── algorithms/             # RL agents, buffers, configs, and reward learning
│   ├── agents/             # DQN, SAC, PPO, TD3 agents
│   ├── reward/             # Reward model and D-REX-style trainer
│   ├── buffers.py
│   ├── config.py
│   └── models.py
│
└── training/               # Trainer, registry, builders, loggers, env wrappers
    ├── builders.py
    ├── factory.py
    ├── trainer.py
    ├── supervised_training.py
    ├── training_steps.py
    ├── optimizers.py
    ├── registry.py
    ├── logger.py
    ├── env_wrapper.py
    └── wrappers.py
```

Root files:

```text
README.md
pyproject.toml
requirements.txt
LICENSE
.gitignore
docs/naming-conventions.md
```

Naming and file creation rules live in
[`docs/naming-conventions.md`](docs/naming-conventions.md).

## Installation

```bash
python -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e .
```

Required packages:

```text
torch>=2.0.0
gymnasium[classic-control]>=0.28.1
numpy>=1.22.0
matplotlib>=3.5.0
```

## Beginner-Friendly Model Builders

Use these when you want to create a model quickly without manually wiring every
`torch.nn` layer.

```python
import torch
from modular_rl.networks import (
    make_cnn_mlp,
    make_mlp,
    make_mlp_classifier,
    make_rnn,
    make_transformer,
)

# n-dimensional vector -> m-dimensional output
mlp = make_mlp(input_dim=10, output_dim=3, hidden_dims=[64, 64])
vector_output = mlp(torch.randn(32, 10))
print(vector_output.shape)  # torch.Size([32, 3])

# shaped input -> flatten -> MLP classifier
mlp_classifier = make_mlp_classifier(
    input_shape=(1, 28, 28),
    output_dim=10,
    hidden_dims=[128, 128],
)
classifier_output = mlp_classifier(torch.randn(32, 1, 28, 28))
print(classifier_output.shape)  # torch.Size([32, 10])

# image -> two 3x3 conv layers -> MLP -> k-dimensional output
cnn_mlp = make_cnn_mlp(
    input_shape=(1, 28, 28),
    output_dim=10,
    conv_channels=[32, 64],
    mlp_hidden_dims=[128],
)
image_output = cnn_mlp(torch.randn(32, 1, 28, 28))
print(image_output.shape)  # torch.Size([32, 10])

# sequence models
rnn = make_rnn(input_dim=28, output_dim=10)
rnn_output = rnn(torch.randn(32, 28, 28))
print(rnn_output.shape)  # torch.Size([32, 28, 10])

transformer = make_transformer(input_dim=28, output_dim=10, max_seq_len=28)
transformer_output = transformer(torch.randn(32, 28, 28))
print(transformer_output.shape)  # torch.Size([32, 28, 10])
```

## Dict-Based Model Construction

Use `build_model` when you want config-driven construction.

```python
from modular_rl.networks import build_model

model = build_model({
    "type": "cnn_mlp",
    "input_shape": (1, 28, 28),
    "output_dim": 10,
    "conv_channels": [32, 64],
    "mlp_hidden_dims": [128],
})
```

For PyTorch-style layer-by-layer construction:

```python
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

The same model builders can be trained with a beginner-friendly supervised
training loop. This is useful for classification or regression tasks where you
already have `(inputs, targets)` batches.

```python
import torch
from torch.utils.data import DataLoader, TensorDataset

from modular_rl.networks import make_mlp
from modular_rl.training import SupervisedTrainingConfig, train_supervised_model

inputs = torch.tensor([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0],
])
targets = torch.tensor([0, 1, 1, 0])
train_loader = DataLoader(TensorDataset(inputs, targets), batch_size=4, shuffle=True)

model = make_mlp(input_dim=2, output_dim=2, hidden_dims=[16, 16])
config = SupervisedTrainingConfig(
    loss="cross_entropy",
    optimizer="adam",
    learning_rate=0.05,
    epochs=100,
)
history = train_supervised_model(
    model=model,
    train_loader=train_loader,
    config=config,
)

print(history.as_dict())
```

For MNIST-style image data with an MLP, use `make_mlp_classifier` so the input
is flattened inside the model:

```python
from modular_rl.networks import make_mlp_classifier
from modular_rl.training import SupervisedTrainingConfig, train_supervised_model

model = make_mlp_classifier(
    input_shape=(1, 28, 28),
    output_dim=10,
    hidden_dims=[128, 128],
)
config = SupervisedTrainingConfig(epochs=10, learning_rate=1e-3)

history = train_supervised_model(
    model=model,
    train_loader=train_loader,
    validation_loader=validation_loader,
    config=config,
)
```

Under the hood, supervised learning and RL updates share the same basic pieces:

```text
model(inputs) -> loss(outputs, targets) -> backward() -> optimizer.step()
```

For that reason, the package exposes small reusable training utilities too:

```python
from modular_rl.training import make_loss, make_optimizer, run_training_step

loss_fn = make_loss("smooth_l1")
optimizer = make_optimizer("adam", model.parameters(), learning_rate=1e-3)
metrics = run_training_step(model, inputs, targets, loss_fn, optimizer)
```

`run_training_step` and `run_evaluation_step` are the preferred names for new
code. Older names such as `training_step` remain available as compatibility
aliases.

When the default batch update is not enough, pass a custom step function. The
step receives the model, the raw batch, and a context object containing the
loss, optimizer, device, and current epoch/batch index.

```python
from modular_rl.training import BatchMetrics

def custom_training_step(model, batch, context):
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
    config=config,
    training_step_fn=custom_training_step,
)
```

Callbacks run after each epoch and can return `True` to stop training:

```python
def stop_when_good_enough(metrics, history, model):
    return metrics.validation_accuracy is not None and metrics.validation_accuracy > 0.95

history = train_supervised_model(
    model=model,
    train_loader=train_loader,
    validation_loader=validation_loader,
    config=config,
    callbacks=[stop_when_good_enough],
)
```

`train_supervised_model` uses these utilities directly. RL agents such as DQN
also perform loss/backward/optimizer updates internally, so their update methods
can gradually reuse the same lower-level utilities without changing the public
RL training API.

## RL Training Shortcuts

```python
from modular_rl.training import quick_dqn, quick_sac, quick_ppo, quick_td3

quick_dqn("CartPole-v1").train()
quick_sac("Pendulum-v1", total_timesteps=50000).train()
quick_ppo("CartPole-v1").train()
quick_td3("Pendulum-v1").train()
```

## Config-Driven Trainers

```python
from modular_rl.training import build_trainer

trainer = build_trainer({
    "algorithm": "dqn",
    "env": {"type": "gym", "name": "CartPole-v1"},
    "model": {"backbone": {"type": "mlp", "hidden_dims": [128, 128]}},
    "config": {"total_timesteps": 30000, "learning_starts": 1000},
    "logger": {"type": "console"},
})

trainer.train()
```

## Extending Components

The registry lets you add project-specific components without changing the core
library files.

```python
from modular_rl.training import Registry

Registry.register_backbone("my_encoder", MyEncoder)
Registry.register_logger("my_logger", MyLogger)

print(Registry.list_backbones())
print(Registry.list_loggers())
```

## License

[MIT License](LICENSE)
