import torch
import torch.nn as nn
from typing import Optional


class RewardModel(nn.Module):
    """
    Learned reward network: R(s) or R(s, a) → scalar.

    Used in D-REX and CLARE for reward learning from demonstrations.
    Plug into LearnedRewardWrapper to replace the environment reward.

    Parameters
    ----------
    backbone   : Feature extractor (any dl_modules backbone).
    action_dim : If > 0, concatenate action features before the reward head.
                 Set to 0 for a state-only reward R(s).
    """

    def __init__(self, backbone: nn.Module, action_dim: int = 0):
        super().__init__()
        self.backbone = backbone
        self.action_dim = action_dim
        self.reward_head = nn.Linear(backbone.output_dim + action_dim, 1)

    @property
    def uses_action(self) -> bool:
        return self.action_dim > 0

    def forward(
        self,
        state: torch.Tensor,
        action: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Per-step reward.

        Parameters
        ----------
        state  : (..., state_dim)
        action : (..., action_dim) or (...,) — required if uses_action

        Returns
        -------
        reward : (...,) scalar per sample
        """
        features = self.backbone(state)
        if self.uses_action and action is not None:
            action = action.float()
            if action.dim() < features.dim():
                action = action.unsqueeze(-1)
            features = torch.cat([features, action], dim=-1)
        return self.reward_head(features).squeeze(-1)

    def trajectory_return(
        self,
        states: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Sum of per-step rewards along a trajectory.

        Parameters
        ----------
        states  : (T, state_dim)
        actions : (T,) or (T, action_dim) — optional

        Returns
        -------
        return_ : scalar tensor
        """
        return self.forward(states, actions).sum()
