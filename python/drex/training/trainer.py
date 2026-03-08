"""
drex.training.trainer — DrexTrainer with Truncated BPTT.

Processes long sequences as consecutive segments. State (L2 memory) is
threaded across segments. Gradients are detached at segment boundaries
(TBPTT), but the state carries information forward.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from drex.models.memory import LayerState
from drex.models.transformer import DrexConfig, DrexTransformer


class DrexTrainer:
    """
    Minimal TBPTT trainer for DrexTransformer.

    Usage::

        trainer = DrexTrainer(model, config)
        for batch in dataloader:          # batch: (B, total_len) token ids
            loss = trainer.train_step(batch)

    The trainer handles segmentation internally. Each call to train_step
    accumulates gradients across n_segments_per_step consecutive segments,
    then updates parameters once.
    """

    def __init__(
        self,
        model: DrexTransformer,
        config: DrexConfig,
        lr: float = 3e-4,
        weight_decay: float = 0.1,
        grad_clip: float = 1.0,
        n_segments_per_step: int = 4,
        segment_len: Optional[int] = None,  # defaults to config.window_size
    ) -> None:
        self.model = model
        self.config = config
        self.grad_clip = grad_clip
        self.n_segments_per_step = n_segments_per_step
        self.segment_len = segment_len or config.window_size

        self.optim = AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.95),
        )

        self._global_step = 0
        self._states: Optional[list[LayerState]] = None

    # ------------------------------------------------------------------

    def train_step(self, token_ids: torch.Tensor) -> float:
        """
        token_ids: (B, T) — full sequence, T >= segment_len.
        Returns average cross-entropy loss for the step.
        """
        B, T = token_ids.shape
        device = token_ids.device

        # Initialise or re-use states
        if self._states is None:
            self._states = self.model.init_states(B, device)

        self.model.train()
        self.optim.zero_grad()

        total_loss = torch.tensor(0.0, device=device)
        n_tokens = 0

        for seg_idx in range(self.n_segments_per_step):
            start = seg_idx * self.segment_len
            end = start + self.segment_len + 1  # +1 for targets
            if end > T:
                break

            src = token_ids[:, start : start + self.segment_len]
            tgt = token_ids[:, start + 1 : start + self.segment_len + 1]

            logits, self._states = self.model(src, self._states)

            # Cross-entropy over the segment
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, self.config.vocab_size),
                tgt.reshape(-1),
            )
            total_loss = total_loss + loss
            n_tokens += tgt.numel()

            # Detach states at segment boundaries to limit BPTT graph depth
            self._states = [s.detach() for s in self._states]

        if n_tokens == 0:
            return 0.0

        avg_loss = total_loss / self.n_segments_per_step
        avg_loss.backward()

        if self.grad_clip > 0.0:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

        self.optim.step()
        self._global_step += 1

        return avg_loss.item()

    def reset_states(self) -> None:
        """Force-reset recurrent states (e.g. at document boundaries)."""
        self._states = None
