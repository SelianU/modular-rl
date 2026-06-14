from dataclasses import dataclass

from lattice.algorithms import AgentBuildContext, build_agent
from lattice.training import Registry


@dataclass
class TinyConfig:
    total_timesteps: int = 10
    device: str = "cpu"


class TinyAgent:
    def __init__(self, action_dim, config):
        self.action_dim = action_dim
        self.config = config

    def select_action(self, state, evaluation=False):
        return 0

    def observe(self, state, action, reward, next_state, done):
        return None

    def update(self):
        return {"tiny_loss": 0.0}

    def save(self, filepath):
        return None


class TinyAgentBuilder:
    def __init__(self, registry=None):
        self.registry = registry

    def build(self, context: AgentBuildContext, spec):
        return TinyAgent(action_dim=context.action_dim, config=context.config)


def test_custom_algorithm_can_register_config_and_agent_builder():
    Registry.register_config("tiny_algo", TinyConfig)
    Registry.register_agent_builder("tiny_algo", TinyAgentBuilder)

    agent = build_agent(
        {
            "algorithm": "tiny_algo",
            "state_dim": 3,
            "action_dim": 2,
            "config": {"total_timesteps": 5},
        }
    )

    assert isinstance(agent, TinyAgent)
    assert agent.action_dim == 2
    assert agent.config.total_timesteps == 5


def test_registered_agent_builders_are_listed():
    Registry.register_agent_builder("tiny_algo_listed", TinyAgentBuilder)

    assert "tiny_algo_listed" in Registry.list_agent_builders()
