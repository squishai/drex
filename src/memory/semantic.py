"""NoProp Semantic Memory — DREX-UNIFIED COMPONENT 3, OBJECTIVE 2c.

Small trained SSM storing compressed world knowledge.  Each block denoises
independently — no global backpropagation, no cross-block gradients.

Architecture:
    NoPropBlock          — single denoising block with local Adam optimizer.
    NoPropSemanticMemory — stack of n_blocks NoPropBlocks.

Dtype contract (DREX_UNIFIED_SPEC.md §dtype):
    Input to NoPropBlock:   float32  (asserted in forward)
    After input projection: cast → bfloat16  (ONLY place this cast happens)
    NoPropBlock output:     bfloat16
    NoPropSemanticMemory output: bfloat16

NoProp reference: arXiv 2503.24322 — per-block local denoising objective,
    no inter-block gradient flow, parallel block training.

Ref: DREX_UNIFIED_SPEC.md § COMPONENT 3: NOPROP SEMANTIC MEMORY
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# NoPropBlock — single denoising block
# ---------------------------------------------------------------------------

class NoPropBlock(nn.Module):
    """Single NoProp denoising block.

    During training, injects Gaussian noise into the input and learns to
    reconstruct the clean signal (local MSE loss).  During evaluation,
    acts as a pass-through projection (no noise, no loss).

    Each block owns its own Adam optimizer — never shares with other blocks.

    Args:
        d_model:   Input/output dimension.
        noise_std: Standard deviation of the Gaussian denoising noise.
                   Injected only in train mode.
    """

    def __init__(self, d_model: int, noise_std: float = 0.1) -> None:
        super().__init__()
        self.d_model = d_model
        self.noise_std = noise_std

        # Input projection: float32 → bfloat16  (ONLY dtype cast in this module)
        self.in_proj = nn.Linear(d_model, d_model, bias=False)
        # Hidden layer + output projection
        self.hidden = nn.Linear(d_model, d_model * 2, bias=True)
        self.out_proj = nn.Linear(d_model * 2, d_model, bias=False)

        # Cast all weights to bfloat16 once at init.
        self.to(torch.bfloat16)

        # Per-block optimizer — owns ONLY this block's parameters.
        self.optimizer = torch.optim.Adam(list(self.parameters()), lr=1e-3)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Run the denoising block.

        Args:
            x: (B, S, d_model) float32 — asserted by contract.

        Returns:
            out:        (B, S, d_model) bfloat16
            local_loss: scalar Tensor with grad_fn (train mode) or None (eval).

        Raises:
            AssertionError: if x.dtype != torch.float32.
        """
        assert x.dtype == torch.float32, (
            f"NoPropBlock expects float32 input, got {x.dtype}. "
            "Cast to float32 before passing to this block."
        )

        # --- Cast to bfloat16 at the in_proj boundary (only here) ---
        x_bf = x.to(torch.bfloat16)

        if self.training:
            # Inject Gaussian noise for denoising objective
            noise = torch.randn_like(x_bf) * self.noise_std
            x_noisy = x_bf + noise
        else:
            x_noisy = x_bf

        # Denoising projection
        h = self.in_proj(x_noisy)       # (B, S, d_model) bfloat16
        h = F.silu(self.hidden(h))      # (B, S, d_model * 2) bfloat16
        out = self.out_proj(h)          # (B, S, d_model) bfloat16

        if self.training:
            # Local denoising loss: push output toward the clean input.
            # x_bf is the clean target; out must not depend on x_bf through
            # the noise path — stop_gradient on target is implicit (x_bf is
            # a tensor computed outside this graph node).
            local_loss = F.mse_loss(out.float(), x_bf.detach().float())
        else:
            local_loss = None

        return out, local_loss


# ---------------------------------------------------------------------------
# NoPropSemanticMemory — stacked NoPropBlocks
# ---------------------------------------------------------------------------

class NoPropSemanticMemory(nn.Module):
    """Stack of NoPropBlocks for semantic memory storage and retrieval.

    Each block is trained on its own local denoising loss — no cross-block
    gradient flow.  Blocks are independent: each has its own Adam optimizer
    and parameters.

    Args:
        d_model:   Model dimension.
        n_blocks:  Number of denoising blocks.  Default 4.
        noise_std: Denoising noise level.  Default 0.1.
        lr:        Per-block Adam learning rate.  Default 1e-3.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_blocks: int = 4,
        noise_std: float = 0.1,
        lr: float = 1e-3,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_blocks = n_blocks

        self.blocks = nn.ModuleList([
            NoPropBlock(d_model=d_model, noise_std=noise_std)
            for _ in range(n_blocks)
        ])

        # Override each block's internal lr with the module-level lr.
        # Per-block optimizers own ONLY that block's parameters — never shared.
        for block in self.blocks:
            block.optimizer = torch.optim.Adam(list(block.parameters()), lr=lr)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, list[Optional[torch.Tensor]]]:
        """Run all blocks sequentially with stop_gradient between them.

        Args:
            x: (B, S, d_model) float32

        Returns:
            out:          (B, S, d_model) bfloat16 — output of final block.
            local_losses: list[Optional[Tensor]] of length n_blocks.
                          local_losses[i] is the MSE for block i (train) or None (eval).
        """
        local_losses: list[Optional[torch.Tensor]] = []
        current = x  # float32

        for block in self.blocks:
            out, loss = block(current)
            local_losses.append(loss)
            # Stop gradient between blocks: detach and cast back to float32
            # so next block receives a clean float32 input.
            current = out.detach().float()

        return out, local_losses  # type: ignore[return-value]  # out set by last iteration

    def train_step(self, x: torch.Tensor) -> list[float]:
        """Perform one NoProp training step.

        Each block independently back-props its own local denoising loss.
        Graphs are isolated: each block re-runs from its detached predecessor
        output, so no block's backward can touch another block's parameters.

        Args:
            x: (B, S, d_model) float32

        Returns:
            loss_values: list[float] of length n_blocks.
        """
        # First pass: collect detached outputs for each block's input anchor.
        detached_inputs: list[torch.Tensor] = [x]
        current = x
        with torch.no_grad():
            for block in self.blocks:
                out, _ = block(current)
                next_inp = out.detach().float()
                detached_inputs.append(next_inp)
                current = next_inp

        loss_values: list[float] = []
        for i, block in enumerate(self.blocks):
            # Re-run this block from its detached input to build an isolated graph.
            inp = detached_inputs[i]  # float32, no grad history
            out_i, loss_i = block(inp)
            assert loss_i is not None, "train_step requires self.training=True"

            block.optimizer.zero_grad()
            loss_i.backward()
            block.optimizer.step()
            loss_values.append(loss_i.item())

        return loss_values
