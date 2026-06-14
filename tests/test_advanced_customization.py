import pytest
import torch
import torch.nn as nn

from lattice import make_dqn_agent
from lattice.training import HookManager, RL_HOOK_NAMES, Registry


class TinyBackbone(nn.Module):
    output_dim = 8

    def __init__(self, input_dim, hidden_dims=None):
        super().__init__()
        self.network = nn.Linear(input_dim, self.output_dim)

    def forward(self, inputs):
        return torch.relu(self.network(inputs))


def test_registered_backbone_can_be_used_by_beginner_agent_factory():
    Registry.register_backbone("tiny_test_backbone", TinyBackbone)

    agent = make_dqn_agent(
        state_dim=4,
        action_dim=2,
        backbone="tiny_test_backbone",
        total_timesteps=10,
        learning_starts=1,
        batch_size=2,
        buffer_size=8,
        device="cpu",
    )

    assert isinstance(agent.q_network.backbone, TinyBackbone)


def test_unknown_hook_names_fail_early():
    with pytest.raises(ValueError, match="Unknown hook name 'process_rewrad'"):
        HookManager({"process_rewrad": lambda value, context: value}, allowed_names=RL_HOOK_NAMES)
