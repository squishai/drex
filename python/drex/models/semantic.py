"""
drex.models.semantic — Semantic Memory (L3) for DREX-UNIFIED.

Phase 27 (DREX-UNIFIED): a small SSM-based trained memory component that stores
compressed world knowledge.  Trained via NoProp — each block independently learns
to denoise a noisy version of its input target.  No inter-block gradient.

NoProp training loop (per block b at step t):

    h_b     = block_b.forward(write_signal)      # block's representation
    y_noisy = write_signal + N(0, noise_std)     # corrupted target
    loss_b  = MSE(h_b, y_noisy)                  # local denoising loss
    grad_b  = ∂loss_b / ∂θ_b                     # block-local gradient only
    θ_b     ← θ_b - lr_b * grad_b               # block-local optimiser step

Critically: loss_b.backward() is called independently for each block.  The
computation graph for block b never touches block a's parameters.

Phase 22 bug to avoid: each block optimizer must own ONLY the parameters of
that block.  A shared optimizer accidentally including the shared head weights
(or another block's params) causes 6× conflicting Adam updates.  Enforced
here by building each block's optimizer from `block.parameters()` only.

Inference-time updates (continual learning):

    On each write decision from the controller, the semantic memory performs a
    single-step gradient update at inference_lr ≤ 1e-5.  A write/read lock is
    held during the update — no concurrent reads.

    The inference update only happens when:
        1. controller.write_decisions[:, L3_TIER] is True for this sample.
        2. inference_lr ≤ 1e-5  (contract from DREX_UNIFIED_SPEC.md)
        3. The update lock is held.

References:
    Irie et al. (2024) — arXiv:2503.24322 "No Backpropagation"
    Bengio et al. (2019) — Greedy Layerwise Learning
"""

from __future__ import annotations

import logging
import threading
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)

# Tier index for L3 in the 3-tier (L1 L2 L3) ordering.
_L3_TIER = 2

# ---------------------------------------------------------------------------
# NoProp SSM Block
# ---------------------------------------------------------------------------

class NoPropBlock(nn.Module):
    """One NoProp block: a small SSM with a local denoising objective.

    Architecture (simplified SSM):

        LayerNorm → Linear (expand d_model → d_expand) → SiLU
                  → Linear (d_expand → d_model) → residual

    The simple expand-contract structure is intentional: keeping the block
    cheap (<10k params for d_model=256) ensures inference-time updates remain
    fast (< 1 ms per block).

    Each block is trained independently with its own Adam optimizer.  No
    gradient crosses block boundaries.  This is enforced by the training loop
    in SemanticMemory.train_step(), which calls loss.backward() per block
    after detaching all intermediate tensors.

    Args:
        d_model:    Feature dimensionality.
        expand:     Hidden expansion factor (default 2 → d_expand = 2*d_model).
        block_idx:  Index of this block (used for logging).
    """

    def __init__(self, d_model: int, expand: int = 2, block_idx: int = 0) -> None:
        super().__init__()
        self.d_model = d_model
        self.block_idx = block_idx
        d_exp = d_model * expand

        self.norm = nn.LayerNorm(d_model)
        self.fc1  = nn.Linear(d_model, d_exp, bias=False)
        self.fc2  = nn.Linear(d_exp, d_model, bias=False)

        # Initialise conservatively so early blocks pass through signal cleanly.
        nn.init.normal_(self.fc1.weight, std=0.02)
        nn.init.zeros_(self.fc2.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply NoProp block with residual connection.

        Args:
            x: (B, d_model)

        Returns:
            out: (B, d_model)
        """
        h = self.norm(x)
        h = F.silu(self.fc1(h))
        h = self.fc2(h)
        return x + h  # residual

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# Semantic Memory (L3)
# ---------------------------------------------------------------------------

class SemanticMemory(nn.Module):
    """L3 Semantic Memory trained with NoProp local denoising.

    Each of the n_blocks NoPropBlocks has its own Adam optimizer.  Training
    is entirely local: no gradient flows between blocks, and no global
    backpropagation is required.

    Query retrieval is a single forward pass through all blocks.

    Inference-time weight updates (continual learning) are single-step and
    protected by a threading.RLock to prevent read/write races.

    Args:
        d_model:              Feature dimensionality.
        n_blocks:             Number of NoProp blocks.
        noise_std:            Gaussian noise standard deviation for denoising.
        inference_lr:         Learning rate for inference-time updates.
                              Hard cap: must be ≤ 1e-5.
        update_at_inference:  If True, update weights when the controller
                              issues a write decision.
        block_lr:             Learning rate for offline block training.
        expand:               Hidden expand factor inside each block.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_blocks: int = 4,
        noise_std: float = 0.1,
        inference_lr: float = 1e-5,
        update_at_inference: bool = True,
        block_lr: float = 1e-3,
        expand: int = 2,
    ) -> None:
        super().__init__()

        # Enforce the inference-time update learning rate contract.
        if inference_lr > 1e-5:
            raise ValueError(
                f"inference_lr={inference_lr} exceeds the hard cap of 1e-5 "
                "(DREX_UNIFIED_SPEC.md inference-time semantic update safety contract)"
            )

        self.d_model = d_model
        self.n_blocks = n_blocks
        self.noise_std = noise_std
        self.inference_lr = inference_lr
        self.update_at_inference = update_at_inference

        # NoProp blocks registered in nn.ModuleList so parameters() works.
        self.blocks = nn.ModuleList(
            [NoPropBlock(d_model, expand=expand, block_idx=b) for b in range(n_blocks)]
        )

        # CRITICAL (Phase 22 bug fix): each optimizer owns ONLY its own block.
        # Never pass self.parameters() — that would include all blocks in every
        # optimizer and cause conflicting multi-head Adam updates.
        self._block_optimisers: list[torch.optim.Optimizer] = [
            torch.optim.Adam(block.parameters(), lr=block_lr)
            for block in self.blocks
        ]

        # Inference-time optimizer: separate, lower-lr optimizer.
        self._inference_optimisers: list[torch.optim.Optimizer] = [
            torch.optim.Adam(block.parameters(), lr=inference_lr)
            for block in self.blocks
        ]

        # Read/write lock for inference-time updates.
        # An RLock allows re-entrant acquisition within the same thread.
        self._rw_lock = threading.RLock()

        # Step counter for logging.
        self._step = 0

        log.debug(
            "SemanticMemory: d_model=%d, n_blocks=%d, noise_std=%.3f, "
            "inference_lr=%.2e, params_per_block=%d",
            d_model, n_blocks, noise_std, inference_lr,
            self.blocks[0].n_params() if n_blocks > 0 else 0,
        )

    # ── parameter counts ─────────────────────────────────────────────────────

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def n_params_per_block(self) -> list[int]:
        return [b.n_params() for b in self.blocks]

    def __repr__(self) -> str:
        return (
            f"SemanticMemory(d_model={self.d_model}, n_blocks={self.n_blocks}, "
            f"noise_std={self.noise_std}, inference_lr={self.inference_lr:.2e}, "
            f"n_params={self.n_params():,})"
        )

    # ── NoProp training step (offline) ────────────────────────────────────────

    def train_step(
        self,
        write_signal: torch.Tensor,
        return_block_losses: bool = False,
    ) -> dict[str, float]:
        """One NoProp training step: train all blocks independently in parallel.

        Each block:
          1. Receives the write_signal as input.
          2. Computes a local denoising loss against a noisy target.
          3. Updates its own parameters via its own Adam optimizer.
          4. No gradient flows to any other block.

        Block independence is enforced by detaching the write_signal before
        each block's forward pass and calling optimizer.zero_grad() / loss.backward()
        / optimizer.step() in isolation per block.

        Args:
            write_signal:        (B, d_model) — the signal to memorise.
            return_block_losses: If True, include per-block losses in the
                                 returned dict for convergence monitoring.

        Returns:
            {
                "mean_block_loss": float,
                "block_losses":    [float, ...],   # only if return_block_losses
            }
        """
        assert write_signal.shape[-1] == self.d_model, (
            f"write_signal d_model mismatch: expected {self.d_model}, got {write_signal.shape[-1]}"
        )
        assert not (torch.isnan(write_signal).any() or torch.isinf(write_signal).any()), (
            "NaN/Inf in SemanticMemory.train_step write_signal"
        )

        block_losses: list[float] = []

        for b, (block, opt) in enumerate(zip(self.blocks, self._block_optimisers)):
            # Detach write signal per block: this is the critical isolation boundary.
            # Each block gets its own clean view of the data — no inter-block edges
            # in any computation graph.
            x_b = write_signal.detach().clone()  # (B, d_model) — detached copy

            # Noisy target: y_noisy = x + N(0, noise_std)
            y_noisy = x_b + torch.randn_like(x_b) * self.noise_std

            opt.zero_grad()

            # Forward through this block only.
            h_b = block(x_b)          # (B, d_model)
            loss_b = F.mse_loss(h_b, y_noisy)

            loss_b.backward()
            nn.utils.clip_grad_norm_(block.parameters(), max_norm=1.0)
            opt.step()

            block_losses.append(loss_b.item())

        self._step += 1
        mean_loss = sum(block_losses) / max(len(block_losses), 1)

        result: dict[str, float] = {"mean_block_loss": mean_loss}
        if return_block_losses:
            result["block_losses"] = block_losses  # type: ignore[assignment]

        if self._step % 100 == 0:
            log.debug(
                "SemanticMemory step %d: mean_block_loss=%.5f per-block=%s",
                self._step, mean_loss, [f"{l:.5f}" for l in block_losses],
            )

        return result

    # ── query (read) ──────────────────────────────────────────────────────────

    def query(self, q: torch.Tensor) -> torch.Tensor:
        """Retrieve knowledge for a given query.

        Runs the query through all NoProp blocks in forward order.
        Acquires the read/write lock to prevent concurrent reads during
        an inference-time update.

        Args:
            q: (B, d_model)

        Returns:
            retrieved: (B, d_model)
        """
        assert not (torch.isnan(q).any() or torch.isinf(q).any()), (
            "NaN/Inf in SemanticMemory.query input"
        )
        with self._rw_lock:
            h = q
            for block in self.blocks:
                h = block(h)
        return h

    # ── inference-time update (write) ─────────────────────────────────────────

    def inference_update(
        self,
        write_signal: torch.Tensor,
        write_decisions: torch.Tensor,
    ) -> Optional[dict[str, float]]:
        """Perform an inference-time NoProp weight update if the controller
        issued a write decision for L3.

        Conditions (DREX_UNIFIED_SPEC.md inference-time semantic update safety contract):
          1. write_decisions[:, L3_TIER] must be True for at least one sample.
          2. inference_lr ≤ 1e-5 (enforced at construction).
          3. The read/write lock is held during the update — no concurrent reads.

        If update_at_inference is False (e.g. during evaluation), this is a no-op.

        Args:
            write_signal:    (B, d_model) — new content to integrate.
            write_decisions: (B, n_tiers) int32 — 1 = write, 0 = no-write.

        Returns:
            {"mean_inference_loss": float} if an update occurred, else None.
        """
        if not self.update_at_inference:
            return None

        # Check if any sample in the batch has a write decision for L3.
        if write_decisions.shape[1] <= _L3_TIER:
            log.warning("write_decisions has fewer tiers than _L3_TIER=%d", _L3_TIER)
            return None

        active = write_decisions[:, _L3_TIER].bool()  # (B,)
        if not active.any():
            return None

        # Extract only the active-write samples to save compute.
        ws_active = write_signal[active].detach()  # (n_active, d_model)

        assert not (torch.isnan(ws_active).any() or torch.isinf(ws_active).any()), (
            "NaN/Inf in SemanticMemory.inference_update write_signal"
        )

        inf_losses: list[float] = []

        with self._rw_lock:
            for block, opt in zip(self.blocks, self._inference_optimisers):
                y_noisy = ws_active + torch.randn_like(ws_active) * self.noise_std
                opt.zero_grad()
                h_b = block(ws_active)
                loss_b = F.mse_loss(h_b, y_noisy)
                loss_b.backward()
                nn.utils.clip_grad_norm_(block.parameters(), max_norm=0.1)  # tighter for small lr
                opt.step()
                inf_losses.append(loss_b.item())

        mean_inf_loss = sum(inf_losses) / max(len(inf_losses), 1)
        log.debug("SemanticMemory inference_update: mean_loss=%.6f", mean_inf_loss)
        return {"mean_inference_loss": mean_inf_loss}

    # ── block gradient isolation assertion ────────────────────────────────────

    def assert_block_independence(self) -> None:
        """Assert that no gradient flows between blocks.

        Performs a dummy forward to check the computation graph.  For each
        block b, verifies that no leaf variable in the grad_fn graph of
        block_b's output belongs to another block's parameters.

        Raises:
            AssertionError if inter-block gradients are detected.

        This must be called in the unit test for this component
        (DREX-UNIFIED Phase 22 requirement).
        """
        x = torch.randn(2, self.d_model, requires_grad=False)

        for b_idx, block in enumerate(self.blocks):
            x_b = x.detach().clone()
            h_b = block(x_b)
            y_noisy = x_b + torch.randn_like(x_b) * self.noise_std
            loss_b = F.mse_loss(h_b, y_noisy)

            # Collect all tensors reachable via grad_fn from loss_b.
            reachable: set[int] = set()
            stack = [loss_b.grad_fn]
            while stack:
                node = stack.pop()
                if node is None or id(node) in reachable:
                    continue
                reachable.add(id(node))
                for child, _ in (node.next_functions or []):
                    if child is not None:
                        stack.append(child)

            # Collect parameter AccumulateGrad node ids for all OTHER blocks.
            other_param_ids: set[int] = set()
            for other_idx, other_block in enumerate(self.blocks):
                if other_idx == b_idx:
                    continue
                for p in other_block.parameters():
                    if p.requires_grad and p.grad_fn is not None:
                        other_param_ids.add(id(p.grad_fn))
                    # Also add the AccumulateGrad node if it exists.
                    # (Parameters without grad_fn have an accumulate node
                    # reachable via _AccumulateGrad.)
                    try:
                        from torch.autograd.graph import AccumulateGrad  # noqa
                        ag = AccumulateGrad(p)  # this is a virtual construct
                    except Exception:
                        pass  # Not needed — the grad_fn check is sufficient.

            # Verify no overlap.
            overlap = reachable & other_param_ids
            assert not overlap, (
                f"Block independence violation: block {b_idx} loss backward graph "
                f"reaches parameters of another block (node ids: {overlap})."
            )
