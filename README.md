# modular-rl

A modular, extensible Reinforcement Learning library built on PyTorch.  
Network structures (`networks`) are fully decoupled from RL algorithms (`algorithms`),  
and the `training` package provides a unified training loop, environment abstraction, registry, and pluggable logging.

```bash
pip install modular-rl
```

---

## Package Structure

```
modular_rl/
├── networks/               # Neural network building blocks
│   ├── encoders/           # MLP, CNN, RNN, Transformer
│   ├── heads.py            # QHead, DuelingQHead, GaussianPolicyHead, ...
│   └── simple.py           # make_mlp, make_cnn_mlp, build_model
│
├── algorithms/             # RL algorithms
│   ├── agents/             # DQNAgent, SACAgent, PPOAgent, TD3Agent
│   ├── reward/             # Reward learning (D-REX, CLARE)
│   ├── buffers.py          # ReplayBuffer, PrioritizedReplayBuffer, RolloutBuffer
│   ├── config.py           # DQNConfig, SACConfig, PPOConfig, TD3Config
│   └── models.py           # QNetwork, SACActor/Critic, PPOActor/Critic, TD3Actor
│
└── training/               # Training infrastructure
    ├── builders.py         # build_trainer, ExperimentBuilder
    ├── factory.py          # quick_dqn, quick_sac, quick_ppo, quick_td3
    ├── trainer.py          # Trainer — unified training loop
    ├── registry.py         # Registry — string-keyed component factory
    ├── env_wrapper.py      # BaseEnv, GymEnvWrapper, CustomEnvWrapper
    └── wrappers.py         # RewardNormWrapper, LearnedRewardWrapper

docs/
├── naming-conventions.md   # File, function, class, and variable naming rules
└── superpowers/            # Design and implementation notes

experiments/
├── learning/               # Small synthetic learning experiments
└── mnist/                  # MNIST classification experiments

scripts/                    # Shell wrappers for runnable experiments
```

---

## Quickstart

### Beginner-Friendly Standalone Models

If you only want to build a neural network, start here:

```python
import torch
from modular_rl.networks import make_mlp, make_cnn_mlp, make_rnn, make_transformer, make_mini_gpt

# 1) n-dimensional vector -> m-dimensional output
n = 10
m = 3
mlp = make_mlp(input_dim=n, output_dim=m)

x = torch.randn(32, n)
y = mlp(x)
print(y.shape)  # torch.Size([32, 3])

# 2) channels x height x width image -> two 3x3 conv layers -> MLP -> output_dim
channels, height, width = 1, 28, 28
output_dim = 5
cnn_mlp = make_cnn_mlp(
    input_shape=(channels, height, width),
    output_dim=output_dim,
    conv_channels=[32, 64],   # two convolution layers
    mlp_hidden_dims=[128],
)

images = torch.randn(32, channels, height, width)
out = cnn_mlp(images)
print(out.shape)  # torch.Size([32, 5])

# 3) sequence model: (batch, input_dim) or (batch, sequence, input_dim)
rnn = make_rnn(input_dim=10, output_dim=3)
single_step_out = rnn(torch.randn(32, 10))
sequence_out = rnn(torch.randn(32, 6, 10))
print(single_step_out.shape)  # torch.Size([32, 3])
print(sequence_out.shape)     # torch.Size([32, 6, 3])

transformer = make_transformer(input_dim=10, output_dim=3)
sequence_out = transformer(torch.randn(32, 6, 10))
print(sequence_out.shape)     # torch.Size([32, 6, 3])

# 4) small GPT-style language model: token ids -> vocabulary logits
mini_gpt = make_mini_gpt(vocab_size=1000, max_seq_len=64)
input_ids = torch.randint(0, 1000, (32, 16))
logits = mini_gpt(input_ids)
print(logits.shape)           # torch.Size([32, 16, 1000])
```

You can also build the same models from dictionaries:

```python
from modular_rl.networks import build_model

model = build_model({
    "type": "cnn_mlp",
    "input_shape": (1, 28, 28),
    "output_dim": 5,
    "conv_channels": [32, 64],
    "mlp_hidden_dims": [128],
})

sequence_model = build_model({
    "type": "transformer",
    "input_dim": 10,
    "output_dim": 3,
    "hidden_dims": [64],
    "embed_dim": 128,
    "num_heads": 4,
    "num_layers": 2,
})

language_model = build_model({
    "type": "mini_gpt",
    "vocab_size": 1000,
    "max_seq_len": 64,
    "embed_dim": 128,
    "num_heads": 4,
    "num_layers": 2,
})
```

The beginner sequence wrappers return only the output tensor. If you need hidden
states from an RNN, use the lower-level `RNN` backbone directly.

`MiniGPT` is a small, randomly initialized GPT-style causal language model. It
does not include a tokenizer, pretrained weights, text generation helpers, or
Hugging Face compatibility. Its purpose is to make the architecture easy to
inspect and train from scratch.

For PyTorch-style layer-by-layer construction, use `type="sequential"`.
The result is still a normal `torch.nn.Module`, so `.parameters()`, `.to(device)`,
`.train()`, `.eval()`, and PyTorch optimizers all work normally.

```python
import torch
from modular_rl.networks import build_model

model = build_model({
    "type": "sequential",
    "input_shape": (1, 28, 28),
    "layers": [
        {"type": "conv2d", "out_channels": 32, "kernel_size": 3, "padding": 1},
        {"type": "relu"},
        {"type": "conv2d", "out_channels": 64, "kernel_size": 3, "padding": 1},
        {"type": "relu"},
        {"type": "flatten"},
        {"type": "linear", "out_features": 128},
        {"type": "relu"},
        {"type": "linear", "out_features": 5},
    ],
})

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
out = model(torch.randn(32, 1, 28, 28))
print(out.shape)  # torch.Size([32, 5])
```

Supported sequential layer types include `linear`, `conv2d`, `flatten`, `relu`,
`tanh`, `sigmoid`, `dropout`, `batch_norm1d`, `batch_norm2d`, and `layer_norm`.
For `linear` and `conv2d`, common input sizes are inferred from `input_shape`
and previous layers.

---

### RL Training Shortcuts

```python
from modular_rl.training import quick_dqn, quick_sac, quick_ppo, quick_td3

quick_dqn("CartPole-v1").train()
quick_sac("Pendulum-v1", total_timesteps=50000).train()
quick_ppo("CartPole-v1").train()
quick_td3("Pendulum-v1").train()
```

### Config-Driven Experiments

For larger experiment grids, build trainers from plain dictionaries instead of
writing a new factory function for every combination:

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

The builder supports the built-in `dqn`, `sac`, `ppo`, and `td3` algorithms.
The same registry used by the builder can be extended with custom backbones,
heads, agents, configs, buffers, environment wrappers, loggers, and optimizers:

```python
from modular_rl.training import Registry

Registry.register_backbone("my_encoder", MyEncoder)
Registry.register_logger("wandb", WandBLogger)
print(Registry.list_backbones())
print(Registry.list_loggers())
```

---

## Naming Conventions

This project keeps public APIs stable and uses explicit names in examples and
internal assembly code. See [docs/naming-conventions.md](docs/naming-conventions.md)
for the full rules.

Common patterns:

- `make_*`: beginner-friendly neural network constructors.
- `build_*`: configurable component builders.
- `quick_*`: common training shortcuts.
- `*Agent`, `*Config`, `*Buffer`, `*Wrapper`, `*Logger`, `*Head`: class role suffixes.
- Prefer `q_network`, `target_q_network`, `state_dim`, `action_dim`, and `environment`
  over short abbreviations.

---

## Examples

```bash
cd examples/

python train_dqn_mlp.py          # DQN  + MLP         → CartPole-v1
python train_sac_mlp.py          # SAC  + MLP         → Pendulum-v1
python train_ppo_mlp.py          # PPO  + MLP         → CartPole-v1
python train_td3_mlp.py          # TD3  + MLP         → Pendulum-v1
python train_dqn_rnn.py          # DRQN + LSTM        → POMDP CartPole-v1
python train_dqn_cnn.py          # DQN  + CNN         → GridWorld (image)
python train_dqn_transformer.py  # DQN  + Transformer → CartPole-v1
python train_drex_long.py        # D-REX reward learning pipeline
```

---

## Supported Algorithms

| Algorithm | Type | Action Space |
|-----------|------|-------------|
| DQN / Double DQN | Value-based | Discrete |
| Dueling DQN | Value-based | Discrete |
| DQN + PER | Value-based | Discrete |
| DRQN (RNN/Transformer) | Value-based (recurrent) | Discrete |
| SAC | Actor-critic | Continuous |
| PPO | On-policy | Both |
| TD3 | Actor-critic | Continuous |
| D-REX | Reward learning | Any |

---

## Key Features

### Swappable Encoders

```python
from modular_rl.networks.encoders import MLP, CNN, RNN, Transformer
from modular_rl.networks import QHead, DuelingQHead
from modular_rl.algorithms import QNetwork

# Drop-in encoder swap
q_network = QNetwork(MLP(input_dim=4, hidden_dims=[128, 128]), QHead(128, 2))
q_network = QNetwork(MLP(input_dim=4, hidden_dims=[128, 128]), DuelingQHead(128, 2))

# Recurrent encoder
base_backbone = MLP(input_dim=4, hidden_dims=[64])
q_network = QNetwork(RNN(base_backbone=base_backbone, rnn_hidden_dim=64), QHead(64, 2))
```

### Prioritized Experience Replay

```python
from modular_rl.algorithms import PrioritizedReplayBuffer

# DQNAgent detects PER automatically and applies IS weights
agent = DQNAgent(..., replay_buffer=PrioritizedReplayBuffer(capacity=10000))
```

### Double DQN

```python
from modular_rl.algorithms import DQNConfig

config = DQNConfig(is_double=True)
```

### Config Validation & Auto Device

```python
from modular_rl.algorithms import DQNConfig

config = DQNConfig()                        # device auto-detected (CUDA if available)
config = DQNConfig(batch_size=512,
                   buffer_size=100)         # ValueError: batch_size > buffer_size
```

### Registry — Dict-Driven Construction

```python
from modular_rl.training import Registry

q_network = Registry.build_q_network(
    backbone_cfg={"type": "mlp", "input_dim": 4, "hidden_dims": [128, 128]},
    head_cfg={"action_dim": 2},
)

# Register custom components
Registry.register_backbone("my_net", MyEncoder)
print(Registry.list_backbones())  # ["mlp", "cnn", "rnn", "transformer", "my_net"]
```

### Environment Abstraction

```python
from modular_rl.training import GymEnvWrapper, CustomEnvWrapper, BaseEnv

# Gymnasium env with optional obs transform
environment = GymEnvWrapper("CartPole-v1", obs_transform=lambda obs: obs * mask)

# Custom env (non-Gymnasium)
environment = CustomEnvWrapper(MyEnv(), state_dim=(1, 16, 16), action_dim=4)

# Implement BaseEnv for any simulator
class MyEnv(BaseEnv):
    def reset(self): ...
    def step(self, action): ...    # → (obs, reward, done, info)
    def close(self): ...
    @property
    def state_dim(self): ...
    @property
    def action_dim(self): ...
    @property
    def is_continuous(self): ...
```

### Stackable Wrappers

```python
from modular_rl.training import RewardNormWrapper, LearnedRewardWrapper

environment = GymEnvWrapper("Pendulum-v1")
environment = RewardNormWrapper(environment, clip=10.0)        # online reward normalisation
environment = LearnedRewardWrapper(environment, reward_model)  # replace reward with learned R(s)
```

### Unified Training Loop

```python
from modular_rl.training import Trainer, ConsoleLogger, MatplotlibLogger, CompositeLogger
from modular_rl.algorithms import DQNConfig

config = DQNConfig(total_timesteps=30000, eval_interval=50)

logger = CompositeLogger([
    ConsoleLogger(log_interval=10),
    MatplotlibLogger(save_path="results.png"),
])

trainer = Trainer(agent, environment, config, logger, save_path="model.pt")
history = trainer.train()        # {"reward": [...], "loss": [...], "eval_reward": [...]}
result  = trainer.evaluate(n_episodes=10)
```

### Pluggable Logging

```python
import wandb
from modular_rl.training import BaseLogger

class WandBLogger(BaseLogger):
    def __init__(self, project):
        wandb.init(project=project)
    def log_episode(self, episode, step, metrics):
        wandb.log({"episode": episode, **metrics})
    def close(self):
        wandb.finish()
```

### D-REX — Reward Learning from Demonstrations

```python
from modular_rl.networks.encoders import MLP
from modular_rl.algorithms.reward import RewardModel, DemonstrationBuffer, RewardModelTrainer
from modular_rl.training import LearnedRewardWrapper, GymEnvWrapper

# 1. Collect ranked demonstrations (higher rank = better)
demonstration_buffer = DemonstrationBuffer()
demonstration_buffer.add_trajectory(states, actions, rank=0)   # worst
demonstration_buffer.add_trajectory(states, actions, rank=5)   # best

# 2. Train reward model (Bradley-Terry preference loss)
backbone = MLP(input_dim=4, hidden_dims=[64, 64])
reward_model = RewardModel(backbone, action_dim=0)   # R(s)
trainer = RewardModelTrainer(reward_model, optimizer)
trainer.train(demonstration_buffer, n_epochs=300)

# 3. Train any agent with learned reward — no true env reward needed
environment = LearnedRewardWrapper(GymEnvWrapper("CartPole-v1"), reward_model)
```

---

## Installation

```bash
# Editable install (development)
git clone https://github.com/yourname/modular-rl
cd modular-rl
pip install -e .

# From PyPI (when published)
pip install modular-rl
```

**Requirements:** `torch>=2.0.0`, `torchvision>=0.15.0`, `gymnasium[classic-control]>=0.28.1`, `numpy>=1.22.0`, `matplotlib>=3.5.0`

---

## Development

Create a local virtual environment and install the project in editable mode:

```bash
python -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e .
```

Run parameterized learning experiments with detailed logs:

```bash
./scripts/run_learning_experiments.sh --quick
./scripts/run_learning_experiments.sh --model mlp --quick
./scripts/run_learning_experiments.sh --model cnn --quick
./scripts/run_learning_experiments.sh --model rnn --quick
./scripts/run_learning_experiments.sh --model transformer --quick
./scripts/run_learning_experiments.sh --model mini_gpt --quick
```

The synthetic experiment runner is meant for watching how different model
settings learn on small but non-trivial tasks:

- MLP: two-moons-like nonlinear classification.
- CNN + MLP: image bar-orientation classification.
- RNN: first/last token sequence classification.
- Transformer: marked-position sequence classification.
- MiniGPT: cyclic next-token prediction.

Each run prints the model, task, parameters, step losses, final loss, best loss,
loss ratio, final accuracy, and elapsed time. Example:

```text
=== MLP | two_moons_like_classification ===
params={'hidden_dims': [8], 'learning_rate': 0.03, 'steps': 120, 'seed': 10}
step=0001/0120 loss=0.6880
step=0025/0120 loss=0.4321
result initial_loss=0.6880 final_loss=0.0053 best_loss=0.0053 loss_ratio=0.0077 final_accuracy=100.00% elapsed_seconds=0.10
```

Run MNIST classification experiments:

```bash
./scripts/run_mnist_experiments.sh --model mlp --quick
./scripts/run_mnist_experiments.sh --model cnn --quick
./scripts/run_mnist_experiments.sh --model rnn --quick
./scripts/run_mnist_experiments.sh --model transformer --quick
./scripts/run_mnist_experiments.sh --model all --quick
```

You can also change common training parameters from the command line:

```bash
./scripts/run_mnist_experiments.sh --model cnn --quick --epochs 3 --learning-rate 0.001
./scripts/run_mnist_experiments.sh --model all --quick --train-limit 2048 --test-limit 512
./scripts/run_mnist_experiments.sh --model mlp --quick --device cpu --no-download
```

The MNIST experiments live under `experiments/mnist/` and are split by role:

- `data.py`: downloads/loads MNIST with torchvision.
- `models.py`: adapts each modular_rl model family to MNIST input shapes.
- `train.py`: trains one experiment and prints epoch metrics.
- `run_mnist_experiments.py`: CLI entry point and parameter grids.

Model input handling:

- MLP flattens each image from `(1, 28, 28)` to `784`.
- CNN keeps the image as `(1, 28, 28)`.
- RNN reads the image as 28 row tokens with 28 features each.
- Transformer reads the same 28 row tokens with positional encoding.

The first MNIST run downloads data into `data/`. Use `--no-download` when you
want the command to fail instead of downloading missing files. Example log:

```text
=== MNIST | CNN ===
params={'conv_channels': [16, 32], 'learning_rate': 0.001, 'epochs': 1, ...}
epoch=01/01 train_loss=0.9123 train_accuracy=72.56% test_loss=0.4211 test_accuracy=88.67% elapsed_seconds=2.34
final_test_accuracy=88.67%
```

Generated artifacts such as `.venv/`, `data/`, `checkpoints/`, training plots,
logs, and model files are ignored by `.gitignore`.

---

## License

[MIT License](LICENSE)
