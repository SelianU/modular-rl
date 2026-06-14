import numpy as np

from lattice.training import RLTrainer, SupervisedTrainingHistory
from lattice.training.history import RLTrainingHistory


class DummyAgent:
    def __init__(self):
        self.config = None
        self.observed = []

    def select_action(self, state, evaluation=False):
        return 0

    def observe(self, state, action, reward, next_state, done):
        self.observed.append((state, action, reward, next_state, done))

    def update(self):
        return {"loss": 0.25}

    def save(self, filepath):
        return None


class DummyEnvironment:
    def reset(self):
        return np.array([0.0], dtype=np.float32)

    def step(self, action):
        return np.array([1.0], dtype=np.float32), 1.0, True, {}

    def close(self):
        return None


class DummyConfig:
    total_timesteps = 1
    log_interval = 0
    eval_interval = 0


class DummyLogger:
    def __init__(self):
        self.records = []

    def log_episode(self, episode, global_step, metrics):
        self.records.append((episode, global_step, dict(metrics)))

    def close(self):
        return None


def test_rl_trainer_returns_history_object_with_common_helpers():
    logger = DummyLogger()
    trainer = RLTrainer(
        agent=DummyAgent(),
        env=DummyEnvironment(),
        config=DummyConfig(),
        logger=logger,
    )

    history = trainer.train()

    assert isinstance(history, RLTrainingHistory)
    assert history.as_dict() == {"reward": [1.0], "loss": [0.25]}
    assert history.last("reward") == 1.0
    assert logger.records == [(1, 1, {"reward": 1.0, "loss": 0.25})]


def test_supervised_history_exposes_same_last_helper():
    history = SupervisedTrainingHistory()
    assert history.as_dict() == {
        "epoch": [],
        "train_loss": [],
        "train_accuracy": [],
        "validation_loss": [],
        "validation_accuracy": [],
    }
    assert history.last("train_loss") is None
