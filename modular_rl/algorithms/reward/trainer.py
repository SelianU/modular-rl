import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Tuple

from .models import RewardModel
from .datasets import DemonstrationBuffer, Trajectory


class RewardModelTrainer:
    """
    Trains a RewardModel using the Bradley-Terry pairwise preference model.

    Given a better trajectory τ+ and a worse trajectory τ−, the loss is:

        L = −log σ( R(τ+) − R(τ−) )

    where R(τ) = Σ_t r(s_t, a_t) is the total trajectory return under the
    reward model. L2 regularisation on the reward weights is applied to
    prevent reward magnitude from growing unboundedly.

    Parameters
    ----------
    reward_model : RewardModel to train.
    optimizer    : PyTorch optimizer (e.g. Adam).
    l2_coeff     : L2 regularisation coefficient on reward weights.
    device       : Torch device string.
    """

    def __init__(
        self,
        reward_model: RewardModel,
        optimizer: torch.optim.Optimizer,
        l2_coeff: float = 1e-4,
        device: str = "cpu",
    ):
        self.reward_model = reward_model.to(device)
        self.optimizer = optimizer
        self.l2_coeff = l2_coeff
        self.device = torch.device(device)

    # ------------------------------------------------------------------ #
    # Core loss                                                            #
    # ------------------------------------------------------------------ #

    def _to_tensors(self, traj: Trajectory) -> Tuple[torch.Tensor, torch.Tensor]:
        states = torch.from_numpy(traj.states).to(self.device)
        if traj.actions.dtype in (np.int32, np.int64):
            actions = torch.from_numpy(traj.actions.astype(np.float32)).to(self.device)
        else:
            actions = torch.from_numpy(traj.actions).to(self.device)
        return states, actions

    def preference_loss(
        self, better: Trajectory, worse: Trajectory
    ) -> torch.Tensor:
        """Bradley-Terry loss for one (better, worse) pair."""
        s1, a1 = self._to_tensors(better)
        s2, a2 = self._to_tensors(worse)

        r1 = self.reward_model.trajectory_return(
            s1, a1 if self.reward_model.uses_action else None
        )
        r2 = self.reward_model.trajectory_return(
            s2, a2 if self.reward_model.uses_action else None
        )
        return -F.logsigmoid(r1 - r2)

    # ------------------------------------------------------------------ #
    # Training loop                                                        #
    # ------------------------------------------------------------------ #

    def train_epoch(
        self,
        demo_buffer: DemonstrationBuffer,
        batch_size: int = 32,
    ) -> float:
        """One epoch: sample batch_size pairs, compute loss, update weights."""
        self.reward_model.train()
        pairs = demo_buffer.sample_preference_pairs(batch_size)

        total_loss = torch.zeros(1, device=self.device)
        for better, worse in pairs:
            total_loss = total_loss + self.preference_loss(better, worse)

        # L2 regularisation
        l2 = sum(p.pow(2).sum() for p in self.reward_model.parameters())
        loss = total_loss / len(pairs) + self.l2_coeff * l2

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train(
        self,
        demo_buffer: DemonstrationBuffer,
        n_epochs: int = 300,
        batch_size: int = 32,
        log_interval: int = 50,
    ) -> List[float]:
        """Full training loop. Returns per-epoch loss history."""
        losses = []
        for epoch in range(1, n_epochs + 1):
            loss = self.train_epoch(demo_buffer, batch_size)
            losses.append(loss)
            if epoch % log_interval == 0:
                print(f"  Reward epoch {epoch:4d}/{n_epochs} | Loss: {loss:.4f}")
        self.reward_model.eval()
        return losses
