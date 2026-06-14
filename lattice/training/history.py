from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class MetricHistory:
    """Metric history with common helper methods."""

    records: List[Dict[str, float]] = field(default_factory=list)

    def append(self, metrics: Dict[str, float]) -> None:
        self.records.append(dict(metrics))

    def as_dict(self) -> Dict[str, List[float]]:
        history: Dict[str, List[float]] = {}
        for record in self.records:
            for name, value in record.items():
                history.setdefault(name, []).append(value)
        return history

    def last(self, name: str) -> Optional[float]:
        for record in reversed(self.records):
            if name in record:
                return record[name]
        return None


class RLTrainingHistory(MetricHistory):
    """Episode-level history returned by reinforcement learning trainers."""
