from typing import Any, Dict, Optional

from lattice.algorithms import (
    AgentBuildContext,
    build_agent,
)

from .logger import BaseLogger, CompositeLogger
from .registry import Registry
from .rl_trainer import RLTrainer


class ExperimentBuilder:
    """Build RLTrainer instances by adding an environment loop around an agent."""

    def __init__(self, registry: type = Registry):
        self.registry = registry

    def build_trainer(self, spec: Dict[str, Any]) -> RLTrainer:
        spec = dict(spec)
        algorithm = spec.get("algorithm", "dqn")
        if algorithm not in self.registry.list_configs():
            raise KeyError(
                f"Algorithm '{algorithm}' not found. Available: {self.registry.list_configs()}"
            )

        environment = self._build_env(spec.get("env", {}), algorithm)
        algorithm_config = self._build_config(algorithm, environment, spec.get("config", {}))
        context = AgentBuildContext(
            algorithm=algorithm,
            state_dim=environment.state_dim,
            action_dim=environment.action_dim,
            config=algorithm_config,
            is_continuous=environment.is_continuous,
            action_low=getattr(environment, "action_low", -1.0) or -1.0,
            action_high=getattr(environment, "action_high", 1.0) or 1.0,
        )
        logger = self._build_logger(spec.get("logger"), algorithm_config, spec.get("plot_path"))

        agent = build_agent(spec, context=context, registry=self.registry)
        return RLTrainer(agent, environment, algorithm_config, logger, save_path=spec.get("save_path"))

    def _build_env(self, env_spec: Dict[str, Any], algorithm: str):
        env_spec = dict(env_spec)
        env_type = env_spec.pop("type", "gym")
        env_name = env_spec.pop("name", None) or self._default_env_name(algorithm)

        if env_type == "gym":
            return self.registry.build_env(env_type, env_name=env_name, **env_spec)
        return self.registry.build_env(env_type, **env_spec)

    def _build_config(self, algorithm: str, env, config_spec: Dict[str, Any]):
        config_spec = dict(config_spec)
        gym_env = getattr(env, "_env", None)
        default_name = getattr(getattr(gym_env, "spec", None), "id", None)
        config_spec.setdefault("env_name", default_name or self._default_env_name(algorithm))
        return self.registry.build_config(algorithm, **config_spec)

    def _build_logger(
        self,
        logger_spec: Optional[Any],
        config,
        plot_path: Optional[str] = None,
    ) -> BaseLogger:
        if isinstance(logger_spec, BaseLogger):
            return logger_spec

        if logger_spec is None:
            loggers = [self.registry.build_logger("console", log_interval=config.log_interval)]
            if plot_path:
                loggers.append(self.registry.build_logger("matplotlib", save_path=plot_path))
            return CompositeLogger(loggers) if len(loggers) > 1 else loggers[0]

        if isinstance(logger_spec, list):
            return CompositeLogger([self._build_logger(item, config) for item in logger_spec])

        logger_spec = dict(logger_spec)
        logger_type = logger_spec.pop("type", "console")
        if logger_type == "console":
            logger_spec.setdefault("log_interval", config.log_interval)
        return self.registry.build_logger(logger_type, **logger_spec)

    @staticmethod
    def _default_env_name(algorithm: str) -> str:
        return {
            "dqn": "CartPole-v1",
            "ppo": "CartPole-v1",
            "sac": "Pendulum-v1",
            "td3": "Pendulum-v1",
        }[algorithm]


def build_trainer(spec: Dict[str, Any]) -> RLTrainer:
    return ExperimentBuilder().build_trainer(spec)
