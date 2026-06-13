import copy
from typing import Any, Dict, Type
import torch
import torch.nn as nn

from modular_rl.networks.encoders import MLP, CNN, RNN, Transformer
from modular_rl.networks import (
    QHead, DuelingQHead,
    GaussianPolicyHead, DiagonalGaussianHead, DeterministicPolicyHead,
    CategoricalPolicyHead,
    DoubleQCriticHead, ValueHead,
)
from modular_rl.algorithms.config import DQNConfig, SACConfig, PPOConfig, TD3Config
from modular_rl.algorithms.buffers import ReplayBuffer, SequenceReplayBuffer, PrioritizedReplayBuffer, RolloutBuffer
from modular_rl.algorithms.models import QNetwork, SACActor, SACCritic
from modular_rl.algorithms.agents import DQNAgent, SACAgent, PPOAgent, TD3Agent
from modular_rl.training.env_wrapper import GymEnvWrapper, CustomEnvWrapper
from modular_rl.training.logger import ConsoleLogger, MatplotlibLogger, CompositeLogger


class Registry:
    """
    Central string-keyed registry for backbones, heads, and agents.

    Built-in keys
    -------------
    Backbones : "mlp", "cnn", "rnn", "transformer"
    Heads     : "q_head", "dueling_q_head", "gaussian_policy", ...
    Agents    : "dqn", "sac", "ppo", "td3"

    Extend at runtime
    -----------------
    Registry.register_backbone("my_net", MyNet)
    Registry.register_agent("my_algo", MyAgent)

    Dict-driven construction
    ------------------------
    q_network = Registry.build_q_network(
        backbone_cfg={"type": "mlp", "input_dim": 4, "hidden_dims": [128, 128]},
        head_cfg={"action_dim": 2},
    )
    """

    _backbones: Dict[str, Type[nn.Module]] = {
        "mlp": MLP,
        "cnn": CNN,
        "rnn": RNN,
        "transformer": Transformer,
    }

    _heads: Dict[str, Type[nn.Module]] = {
        "q_head": QHead,
        "dueling_q_head": DuelingQHead,
        "gaussian_policy": GaussianPolicyHead,
        "diagonal_gaussian": DiagonalGaussianHead,
        "deterministic_policy": DeterministicPolicyHead,
        "categorical_policy": CategoricalPolicyHead,
        "double_q_critic": DoubleQCriticHead,
        "value": ValueHead,
    }

    _agents: Dict[str, type] = {
        "dqn": DQNAgent,
        "sac": SACAgent,
        "ppo": PPOAgent,
        "td3": TD3Agent,
    }

    _configs: Dict[str, type] = {
        "dqn": DQNConfig,
        "sac": SACConfig,
        "ppo": PPOConfig,
        "td3": TD3Config,
    }

    _buffers: Dict[str, type] = {
        "replay": ReplayBuffer,
        "prioritized_replay": PrioritizedReplayBuffer,
        "sequence_replay": SequenceReplayBuffer,
        "rollout": RolloutBuffer,
    }

    _envs: Dict[str, type] = {
        "gym": GymEnvWrapper,
        "custom": CustomEnvWrapper,
    }

    _loggers: Dict[str, type] = {
        "console": ConsoleLogger,
        "matplotlib": MatplotlibLogger,
        "composite": CompositeLogger,
    }

    _optimizers: Dict[str, type] = {
        "adam": torch.optim.Adam,
        "sgd": torch.optim.SGD,
        "rmsprop": torch.optim.RMSprop,
    }

    # ------------------------------------------------------------------ #
    # Registration                                                         #
    # ------------------------------------------------------------------ #

    @classmethod
    def register_backbone(cls, name: str, backbone_cls: Type[nn.Module]) -> None:
        cls._backbones[name] = backbone_cls

    @classmethod
    def register_head(cls, name: str, head_cls: Type[nn.Module]) -> None:
        cls._heads[name] = head_cls

    @classmethod
    def register_agent(cls, name: str, agent_cls: type) -> None:
        cls._agents[name] = agent_cls

    @classmethod
    def register_config(cls, name: str, config_cls: type) -> None:
        cls._configs[name] = config_cls

    @classmethod
    def register_buffer(cls, name: str, buffer_cls: type) -> None:
        cls._buffers[name] = buffer_cls

    @classmethod
    def register_env(cls, name: str, env_cls: type) -> None:
        cls._envs[name] = env_cls

    @classmethod
    def register_logger(cls, name: str, logger_cls: type) -> None:
        cls._loggers[name] = logger_cls

    @classmethod
    def register_optimizer(cls, name: str, optimizer_cls: type) -> None:
        cls._optimizers[name] = optimizer_cls

    # ------------------------------------------------------------------ #
    # Instantiation helpers                                                #
    # ------------------------------------------------------------------ #

    @classmethod
    def build_backbone(cls, name: str, **kwargs) -> nn.Module:
        if name not in cls._backbones:
            raise KeyError(f"Backbone '{name}' not found. Available: {list(cls._backbones)}")
        return cls._backbones[name](**kwargs)

    @classmethod
    def build_head(cls, name: str, **kwargs) -> nn.Module:
        if name not in cls._heads:
            raise KeyError(f"Head '{name}' not found. Available: {list(cls._heads)}")
        return cls._heads[name](**kwargs)

    @classmethod
    def build_agent(cls, name: str, **kwargs):
        if name not in cls._agents:
            raise KeyError(f"Agent '{name}' not found. Available: {list(cls._agents)}")
        return cls._agents[name](**kwargs)

    @classmethod
    def build_config(cls, name: str, **kwargs):
        if name not in cls._configs:
            raise KeyError(f"Config '{name}' not found. Available: {list(cls._configs)}")
        return cls._configs[name](**kwargs)

    @classmethod
    def build_buffer(cls, name: str, **kwargs):
        if name not in cls._buffers:
            raise KeyError(f"Buffer '{name}' not found. Available: {list(cls._buffers)}")
        return cls._buffers[name](**kwargs)

    @classmethod
    def build_env(cls, name: str, **kwargs):
        if name not in cls._envs:
            raise KeyError(f"Env '{name}' not found. Available: {list(cls._envs)}")
        return cls._envs[name](**kwargs)

    @classmethod
    def build_logger(cls, name: str, **kwargs):
        if name not in cls._loggers:
            raise KeyError(f"Logger '{name}' not found. Available: {list(cls._loggers)}")
        return cls._loggers[name](**kwargs)

    @classmethod
    def build_optimizer(cls, name: str, params, **kwargs):
        if name not in cls._optimizers:
            raise KeyError(f"Optimizer '{name}' not found. Available: {list(cls._optimizers)}")
        return cls._optimizers[name](params, **kwargs)

    # ------------------------------------------------------------------ #
    # Dict-driven network builders                                         #
    # ------------------------------------------------------------------ #

    @classmethod
    def build_q_network(
        cls,
        backbone_cfg: Dict[str, Any],
        head_cfg: Dict[str, Any],
    ) -> QNetwork:
        """
        Build a QNetwork from plain dicts (e.g. loaded from YAML).

        backbone_cfg: {"type": "mlp", "input_dim": 4, "hidden_dims": [128, 128]}
        head_cfg:     {"action_dim": 2}
        """
        bb_cfg = copy.deepcopy(backbone_cfg)
        backbone = cls.build_backbone(bb_cfg.pop("type"), **bb_cfg)
        return QNetwork(backbone, QHead(backbone.output_dim, **head_cfg))

    @classmethod
    def build_sac_actor(
        cls,
        backbone_cfg: Dict[str, Any],
        head_cfg: Dict[str, Any],
    ) -> SACActor:
        bb_cfg = copy.deepcopy(backbone_cfg)
        backbone = cls.build_backbone(bb_cfg.pop("type"), **bb_cfg)
        return SACActor(backbone, GaussianPolicyHead(backbone.output_dim, **head_cfg))

    @classmethod
    def build_sac_critic(
        cls,
        backbone_cfg: Dict[str, Any],
        head_cfg: Dict[str, Any],
    ) -> SACCritic:
        bb_cfg = copy.deepcopy(backbone_cfg)
        backbone = cls.build_backbone(bb_cfg.pop("type"), **bb_cfg)
        return SACCritic(backbone, DoubleQCriticHead(backbone.output_dim, **head_cfg))

    # ------------------------------------------------------------------ #
    # Introspection                                                        #
    # ------------------------------------------------------------------ #

    @classmethod
    def list_backbones(cls) -> list:
        return list(cls._backbones)

    @classmethod
    def list_heads(cls) -> list:
        return list(cls._heads)

    @classmethod
    def list_agents(cls) -> list:
        return list(cls._agents)

    @classmethod
    def list_configs(cls) -> list:
        return list(cls._configs)

    @classmethod
    def list_buffers(cls) -> list:
        return list(cls._buffers)

    @classmethod
    def list_envs(cls) -> list:
        return list(cls._envs)

    @classmethod
    def list_loggers(cls) -> list:
        return list(cls._loggers)

    @classmethod
    def list_optimizers(cls) -> list:
        return list(cls._optimizers)
