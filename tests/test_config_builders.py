import pytest

from modular_rl.algorithms import (
    DQNAgent,
    PPOAgent,
    ReplayBuffer,
    SACAgent,
    SequenceReplayBuffer,
    TD3Agent,
)
from modular_rl.training import Registry, Trainer, build_trainer


def test_build_trainer_creates_dqn_from_minimal_spec():
    trainer = build_trainer({
        "algorithm": "dqn",
        "env": {"type": "gym", "name": "CartPole-v1"},
        "config": {"total_timesteps": 10, "learning_starts": 1},
    })

    assert isinstance(trainer, Trainer)
    assert isinstance(trainer.agent, DQNAgent)
    assert isinstance(trainer.agent.replay_buffer, ReplayBuffer)
    assert trainer.config.env_name == "CartPole-v1"


def test_build_trainer_creates_recurrent_dqn_from_model_spec():
    trainer = build_trainer({
        "algorithm": "dqn",
        "env": {"type": "gym", "name": "CartPole-v1"},
        "model": {"backbone": {"type": "rnn", "hidden_dims": [32]}},
        "config": {"total_timesteps": 10, "learning_starts": 1},
    })

    assert isinstance(trainer.agent.replay_buffer, SequenceReplayBuffer)
    assert trainer.agent.is_recurrent is True


@pytest.mark.parametrize(
    ("algorithm", "env_name", "agent_type"),
    [
        ("sac", "Pendulum-v1", SACAgent),
        ("ppo", "CartPole-v1", PPOAgent),
        ("td3", "Pendulum-v1", TD3Agent),
    ],
)
def test_build_trainer_supports_builtin_algorithms(algorithm, env_name, agent_type):
    config = {"total_timesteps": 10}
    if algorithm in {"sac", "td3"}:
        config["learning_starts"] = 1

    trainer = build_trainer({
        "algorithm": algorithm,
        "env": {"type": "gym", "name": env_name},
        "config": config,
    })

    assert isinstance(trainer.agent, agent_type)


def test_build_trainer_rejects_unknown_algorithm_with_available_keys():
    with pytest.raises(KeyError, match="Available"):
        build_trainer({
            "algorithm": "missing",
            "env": {"type": "gym", "name": "CartPole-v1"},
        })


def test_registry_lists_new_component_groups():
    assert "dqn" in Registry.list_configs()
    assert "gym" in Registry.list_envs()
    assert "replay" in Registry.list_buffers()
    assert "adam" in Registry.list_optimizers()
