import torch.nn as nn

from lattice.algorithms import build_agent
from lattice.algorithms.agent_context import AgentBuildContext
from lattice.algorithms.agent_builders import DQNAgentBuilder
from lattice.algorithms.config import DQNConfig
from lattice.algorithms.agents import DQNAgent


def test_dqn_builder_is_importable_and_builds_agent_from_context():
    context = AgentBuildContext(
        algorithm="dqn",
        action_dim=2,
        config=DQNConfig(
            total_timesteps=10,
            learning_starts=1,
            batch_size=2,
            buffer_size=8,
            device="cpu",
        ),
        state_dim=4,
        input_shape=(4,),
    )

    agent = DQNAgentBuilder().build(context, {"algorithm": "dqn"})

    assert isinstance(agent, DQNAgent)
    assert agent.action_dim == 2


def test_build_agent_still_accepts_prebuilt_dqn_model():
    q_network = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))

    agent = build_agent(
        {
            "algorithm": "dqn",
            "action_dim": 2,
            "model": q_network,
            "config": {
                "total_timesteps": 10,
                "learning_starts": 1,
                "batch_size": 2,
                "buffer_size": 8,
                "device": "cpu",
            },
        }
    )

    assert isinstance(agent, DQNAgent)
    assert agent.q_network is q_network
