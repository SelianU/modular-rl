import random
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class Trajectory:
    """A single demonstration trajectory with a quality rank."""
    states: np.ndarray      # (T, state_dim)  float32
    actions: np.ndarray     # (T,) int64  or  (T, action_dim) float32
    rank: int               # Higher = better quality

    @property
    def length(self) -> int:
        return len(self.states)


class DemonstrationBuffer:
    """
    Stores ranked demonstration trajectories for preference-based reward learning.

    Usage
    -----
    buf = DemonstrationBuffer()
    buf.add_trajectory(states, actions, rank=0)   # worst
    buf.add_trajectory(states, actions, rank=5)   # best

    pairs = buf.sample_preference_pairs(32)
    # → list of (better_traj, worse_traj) with better.rank > worse.rank
    """

    def __init__(self):
        self._trajectories: List[Trajectory] = []

    def add_trajectory(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rank: int,
    ) -> None:
        self._trajectories.append(
            Trajectory(
                states=np.asarray(states, dtype=np.float32),
                actions=np.asarray(actions),
                rank=rank,
            )
        )

    def sample_preference_pairs(
        self, n_pairs: int
    ) -> List[Tuple[Trajectory, Trajectory]]:
        """
        Sample (better, worse) pairs with strictly different ranks.
        Returns up to n_pairs pairs (may return fewer if data is insufficient).
        """
        trajs = self._trajectories
        if len(trajs) < 2:
            raise ValueError("Need at least 2 trajectories to form preference pairs.")

        pairs: List[Tuple[Trajectory, Trajectory]] = []
        max_attempts = n_pairs * 20
        attempts = 0

        while len(pairs) < n_pairs and attempts < max_attempts:
            attempts += 1
            i, j = random.sample(range(len(trajs)), 2)
            if trajs[i].rank == trajs[j].rank:
                continue
            better, worse = (trajs[i], trajs[j]) if trajs[i].rank > trajs[j].rank else (trajs[j], trajs[i])
            pairs.append((better, worse))

        return pairs

    def rank_summary(self) -> Dict[int, int]:
        """Count of trajectories per rank level."""
        from collections import Counter
        return dict(sorted(Counter(t.rank for t in self._trajectories).items()))

    def __len__(self) -> int:
        return len(self._trajectories)
