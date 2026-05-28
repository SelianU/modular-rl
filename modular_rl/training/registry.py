import copy
from typing import Any, Dict, Type
import torch.nn as nn

from modular_rl.networks.encoders import MLP, CNN, RNN, Transformer
from modular_rl.networks import QHead, GaussianPolicyHead, DoubleQCriticHead
from modular_rl.algorithms.models import QNetwork, SACActor, SACCritic
from modular_rl.algorithms.agents import DQNAgent, SACAgent


class Registry:
    """
    Central string-keyed registry for backbones, heads, and agents.

    Built-in keys
    -------------
    Backbones : "mlp", "cnn", "rnn", "transformer"
    Heads     : "q_head", "gaussian_policy", "double_q_critic"
    Agents    : "dqn", "sac"

    Extend at runtime
    -----------------
    Registry.register_backbone("my_net", MyNet)
    Registry.register_agent("my_algo", MyAgent)

    Dict-driven construction
    ------------------------
    q_net = Registry.build_q_network(
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
        "gaussian_policy": GaussianPolicyHead,
        "double_q_critic": DoubleQCriticHead,
    }

    _agents: Dict[str, type] = {
        "dqn": DQNAgent,
        "sac": SACAgent,
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
