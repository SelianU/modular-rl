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
│   └── heads.py            # QHead, DuelingQHead, GaussianPolicyHead, ...
│
├── algorithms/             # RL algorithms
│   ├── agents/             # DQNAgent, SACAgent, PPOAgent, TD3Agent
│   ├── reward/             # Reward learning (D-REX, CLARE)
│   ├── buffers.py          # ReplayBuffer, PrioritizedReplayBuffer, RolloutBuffer
│   ├── config.py           # DQNConfig, SACConfig, PPOConfig, TD3Config
│   └── models.py           # QNetwork, SACActor/Critic, PPOActor/Critic, TD3Actor
│
└── training/               # Training infrastructure
    ├── factory.py          # quick_dqn, quick_sac, quick_ppo, quick_td3
    ├── trainer.py          # Trainer — unified training loop
    ├── registry.py         # Registry — string-keyed component factory
    ├── env_wrapper.py      # BaseEnv, GymEnvWrapper, CustomEnvWrapper
    └── wrappers.py         # RewardNormWrapper, LearnedRewardWrapper
```

---

## Quickstart

```python
from modular_rl.training import quick_dqn, quick_sac, quick_ppo, quick_td3

quick_dqn("CartPole-v1").train()
quick_sac("Pendulum-v1", total_timesteps=50000).train()
quick_ppo("CartPole-v1").train()
quick_td3("Pendulum-v1").train()
```

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
q_net = QNetwork(MLP(input_dim=4, hidden_dims=[128, 128]), QHead(128, 2))
q_net = QNetwork(MLP(input_dim=4, hidden_dims=[128, 128]), DuelingQHead(128, 2))

# Recurrent encoder
base = MLP(input_dim=4, hidden_dims=[64])
q_net = QNetwork(RNN(base_backbone=base, rnn_hidden_dim=64), QHead(64, 2))
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

q_net = Registry.build_q_network(
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
env = GymEnvWrapper("CartPole-v1", obs_transform=lambda o: o * mask)

# Custom env (non-Gymnasium)
env = CustomEnvWrapper(MyEnv(), state_dim=(1, 16, 16), action_dim=4)

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

env = GymEnvWrapper("Pendulum-v1")
env = RewardNormWrapper(env, clip=10.0)           # online reward normalisation
env = LearnedRewardWrapper(env, reward_model)     # replace reward with learned R(s)
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

trainer = Trainer(agent, env, config, logger, save_path="model.pt")
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
buf = DemonstrationBuffer()
buf.add_trajectory(states, actions, rank=0)   # worst
buf.add_trajectory(states, actions, rank=5)   # best

# 2. Train reward model (Bradley-Terry preference loss)
backbone = MLP(input_dim=4, hidden_dims=[64, 64])
reward_model = RewardModel(backbone, action_dim=0)   # R(s)
trainer = RewardModelTrainer(reward_model, optimizer)
trainer.train(buf, n_epochs=300)

# 3. Train any agent with learned reward — no true env reward needed
env = LearnedRewardWrapper(GymEnvWrapper("CartPole-v1"), reward_model)
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

**Requirements:** `torch>=2.0.0`, `gymnasium[classic-control]>=0.28.1`, `numpy>=1.22.0`, `matplotlib>=3.5.0`

---

## License

[MIT License](LICENSE)
