"""RewardSignal — DREX-UNIFIED reward computation.

Computes reward as negative cross-entropy between predicted logits and target
class IDs.  Returns a NaN tensor if either input contains NaN or Inf values so
the caller (DREXController.update) can activate its NaN guard.

Ref: DREX_UNIFIED_SPEC.md § COMPONENT 4: DREX CONTROLLER — Reward signal
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


class RewardSignal:
    """Computes routing reward from prediction quality.

    reward = -F.cross_entropy(output_logits, target_ids)

    Returns a NaN float32 scalar if either input contains NaN or Inf.
    The caller is responsible for handling NaN rewards via the NaN guard
    in DREXController.update().
    """

    @staticmethod
    def compute(
        output_logits: torch.Tensor,
        target_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Compute reward as negative cross-entropy.

        Args:
            output_logits: (N, C) float32 logits over C classes.
            target_ids:    (N,) integer class indices (int32 or int64).

        Returns:
            Scalar float32 reward tensor.
            Returns ``torch.tensor(float('nan'))`` if any input is NaN/Inf.
        """
        _nan = torch.tensor(float("nan"), dtype=torch.float32)

        if torch.isnan(output_logits).any() or torch.isinf(output_logits).any():
            return _nan

        target_f = target_ids.float()
        if torch.isnan(target_f).any() or torch.isinf(target_f).any():
            return _nan

        return -F.cross_entropy(output_logits.float(), target_ids.long()).to(
            torch.float32
        )
