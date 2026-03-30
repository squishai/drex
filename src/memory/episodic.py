"""Episodic Memory — DREX-UNIFIED COMPONENT 6, OBJECTIVE 2b.

EMA-based short-to-medium term memory with hard overwrite gating.
No learned parameters — all state manipulation is deterministic.

Interface (per DREX_UNIFIED_SPEC.md v0.2 § COMPONENT 6):
  Input:  write_signal (B, d_model) float32
  Output: (new_state (B, d_model), read_output (B, d_model)) float32

Dtype contract:
  ALL tensors in this module are float32.

Write dynamics:
  EMA delta:   new_state = alpha * state + (1 - alpha) * (write_signal - state)
             = (2*alpha - 1) * state + (1 - alpha) * write_signal
  Hard gate:   if ||ws - state||₂  >=  write_thresh * ||ws||₂
               then new_state = write_signal  (hard overwrite)

alpha(L) formula for length-dependent decay (Phase 11):
  alpha(L) = 0.95 ^ (96 / L)

Ref: DREX_UNIFIED_SPEC.md § COMPONENT 6: SHORT-TERM MEMORY — L2 EPISODIC
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


def compute_alpha(seq_len: int, base: float = 0.95, ref_len: int = 96) -> float:
    """Compute length-dependent EMA alpha.

    alpha(L) = base ^ (ref_len / L)

    Args:
        seq_len: Actual sequence length L.
        base:    Base coefficient (default 0.95 per spec).
        ref_len: Reference length (default 96 per spec Phase 11).

    Returns:
        alpha in (0, 1).
    """
    return base ** (ref_len / seq_len)


class EpisodicMemory(nn.Module):
    """EMA-based episodic memory cell.

    No trainable parameters.  State is passed in and returned — the caller
    owns the tensor and decides whether to keep it between timesteps.

    Args:
        d_model:      State/write-signal dimension.
        alpha:        EMA coefficient (smoothing factor, default 0.90).
        write_thresh: Hard-overwrite gate threshold (default 0.70).
        seed:         Unused for now; reserved for future stochastic extensions.
    """

    def __init__(
        self,
        d_model: int,
        alpha: float = 0.90,
        write_thresh: float = 0.70,
        seed: int = 42,
    ) -> None:
        super().__init__()
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha must be in (0, 1); got {alpha}")
        if not (0.0 < write_thresh <= 1.0):
            raise ValueError(f"write_thresh must be in (0, 1]; got {write_thresh}")

        self.d_model = d_model
        self.alpha = alpha
        self.write_thresh = write_thresh

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def reset_state(self, batch_size: int) -> torch.Tensor:
        """Return zeroed initial episodic state (B, d_model) float32."""
        return torch.zeros(batch_size, self.d_model, dtype=torch.float32)

    # ------------------------------------------------------------------
    # Core write operation
    # ------------------------------------------------------------------

    def write(
        self,
        write_signal: torch.Tensor,
        state: torch.Tensor | None = None,
        force_overwrite: bool = False,
    ) -> torch.Tensor:
        """Write new signal into episodic state.

        Args:
            write_signal:    (B, d_model) float32 — incoming signal.
            state:           (B, d_model) float32 or None (zeros if None).
            force_overwrite: If True, bypass EMA and hard-overwrite with write_signal.

        Returns:
            new_state: (B, d_model) float32
        """
        assert write_signal.dtype == torch.float32, (
            f"EpisodicMemory.write expects float32, got {write_signal.dtype}"
        )
        B = write_signal.shape[0]
        if state is None:
            state = self.reset_state(B).to(write_signal.device)

        if force_overwrite:
            return write_signal.clone()

        # Hard overwrite gate:
        #   ||ws - state||₂  >=  write_thresh * ||ws||₂  → hard overwrite
        delta = write_signal - state  # (B, d_model)
        delta_norm = delta.norm(dim=-1, keepdim=True)     # (B, 1)
        ws_norm = write_signal.norm(dim=-1, keepdim=True) # (B, 1)
        overwrite_mask = (delta_norm >= self.write_thresh * ws_norm).float()

        # EMA update: new = alpha*state + (1-alpha)*delta = (2α-1)*state + (1-α)*ws
        ema_new = self.alpha * state + (1.0 - self.alpha) * delta

        # Merge: use hard overwrite where gate fires, EMA otherwise
        new_state = overwrite_mask * write_signal + (1.0 - overwrite_mask) * ema_new
        return new_state.float()  # (B, d_model) float32

    # ------------------------------------------------------------------
    # Read (passthrough — episodic memory is read by inspecting state)
    # ------------------------------------------------------------------

    def read(self, state: torch.Tensor) -> torch.Tensor:
        """Return the current episodic state (passthrough).

        Args:
            state: (B, d_model) float32

        Returns:
            (B, d_model) float32
        """
        return state

    # ------------------------------------------------------------------
    # Combined forward for sequential use
    # ------------------------------------------------------------------

    def forward(
        self,
        write_signal: torch.Tensor,
        state: torch.Tensor | None = None,
        force_overwrite: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Write and then read the episodic state.

        Args:
            write_signal:    (B, d_model) float32
            state:           (B, d_model) float32 or None
            force_overwrite: If True, hard-overwrite state with write_signal.

        Returns:
            (new_state, read_output) — both (B, d_model) float32
        """
        new_state = self.write(write_signal, state, force_overwrite=force_overwrite)
        read_output = self.read(new_state)
        return new_state, read_output
