from lattice import make_dqn_agent, make_ppo_agent, make_sac_agent, make_td3_agent
from lattice.algorithms import DQNAgent, PPOAgent, SACAgent, TD3Agent


def test_make_dqn_agent_builds_vector_agent_without_spec_dict():
    agent = make_dqn_agent(
        state_dim=4,
        action_dim=2,
        hidden_dims=[16],
        total_timesteps=10,
        learning_starts=1,
        batch_size=2,
        buffer_size=8,
        device="cpu",
    )

    assert isinstance(agent, DQNAgent)
    assert agent.action_dim == 2
    assert agent.config.total_timesteps == 10


def test_make_ppo_agent_defaults_to_discrete_action_space():
    agent = make_ppo_agent(
        state_dim=4,
        action_dim=2,
        hidden_dims=[16],
        total_timesteps=10,
        n_steps=4,
        batch_size=2,
        device="cpu",
    )

    assert isinstance(agent, PPOAgent)
    assert agent.config.n_steps == 4


def test_make_sac_and_td3_agents_default_to_continuous_action_space():
    sac_agent = make_sac_agent(
        state_dim=3,
        action_dim=1,
        hidden_dims=[16],
        total_timesteps=10,
        learning_starts=1,
        batch_size=2,
        buffer_size=8,
        device="cpu",
    )
    td3_agent = make_td3_agent(
        state_dim=3,
        action_dim=1,
        hidden_dims=[16],
        total_timesteps=10,
        learning_starts=1,
        batch_size=2,
        buffer_size=8,
        device="cpu",
    )

    assert isinstance(sac_agent, SACAgent)
    assert isinstance(td3_agent, TD3Agent)
    assert sac_agent.action_dim == 1
    assert td3_agent.action_low == -1.0
    assert td3_agent.action_high == 1.0
