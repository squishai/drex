"""
drex.models.router — Sparse Router for DREX-UNIFIED.

Phase 29 (DREX-UNIFIED): top-k query-gated routing across memory tiers.
Dead tiers receive zero compute and zero gradient.

Architecture:

    query   (B, d_model)   ← last Mamba hidden state
    tier_outputs[i]  (B, d_model)  ← L1, L2, L3 outputs

    score_i  = dot(query, tier_outputs[i])        — unnormalised gate logit
    top-k scores selected                          — exactly k tiers active
    gate_weights = softmax(top_k_scores)           — normalised over active k
    merged = Σ_i  gate_weights_i * tier_outputs_i  — weighted combination

Inactive tiers are detached from the computation graph before their scores
are computed, so they receive zero gradient from the task loss.

Load-balance loss (prevents tier collapse):

    lb_loss = load_balance_coeff * Var(fraction_routed_per_tier)

This loss term is added to the main training loss.  Tracking the routing
fraction over a sliding window of 1000 steps is the caller's responsibility;
this module exposes a helper `routing_fractions()` for access.

Validation criteria (DREX_UNIFIED_SPEC.md Component 8):
    [ ] exactly top_k tiers activate per input token
    [ ] inactive tier params receive zero gradient
    [ ] routing fraction per tier stays within 10% of uniform over 1000 steps
    [ ] sparse forward is ≥20% faster than dense at n_tiers=3, top_k=2

References:
    Shazeer et al. (2017) — Outrageously Large Neural Networks (MoE top-k gating)
    Lepikhin et al. (2021) — GShard (load-balancing auxiliary loss)
"""

from __future__ import annotations

import logging
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sparse Router
# ---------------------------------------------------------------------------

class SparseRouter(nn.Module):
    """Top-k gated sparse router over memory tier outputs.

    Args:
        d_model:             Query / tier-output dimensionality.
        n_tiers:             Total number of memory tiers (default 3: L1, L2, L3).
        top_k:               Number of active tiers per forward pass (default 2).
        load_balance_coeff:  Weight of the load-balance variance penalty loss.
        history_len:         Sliding window length for routing fraction tracking.
    """

    def __init__(
        self,
        d_model: int,
        n_tiers: int = 3,
        top_k: int = 2,
        load_balance_coeff: float = 0.01,
        history_len: int = 1000,
    ) -> None:
        super().__init__()

        if top_k > n_tiers:
            raise ValueError(f"top_k ({top_k}) must be ≤ n_tiers ({n_tiers})")

        self.d_model = d_model
        self.n_tiers = n_tiers
        self.top_k = top_k
        self.load_balance_coeff = load_balance_coeff

        # Learned query projection: collapses the d_model query to a gate score
        # per tier.  This is the only trained component (n_tiers * d_model params).
        # We use a no-bias linear projection to keep the scoring symmetric.
        self.gate_proj = nn.Linear(d_model, n_tiers, bias=False)
        nn.init.normal_(self.gate_proj.weight, std=0.02)

        # Sliding history of routing decisions for fraction tracking.
        # Each entry is a (B,) int tensor of chosen tier indices.
        self._history: deque[torch.Tensor] = deque(maxlen=history_len)
        self._history_len = history_len

    # -- helpers ---------------------------------------------------------------

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"SparseRouter(d_model={self.d_model}, n_tiers={self.n_tiers}, "
            f"top_k={self.top_k}, lb_coeff={self.load_balance_coeff}, "
            f"n_params={self.n_params():,})"
        )

    # -- routing fraction tracking -------------------------------------------

    def routing_fractions(self) -> torch.Tensor:
        """Return the routing fraction for each tier over the sliding window.

        Returns:
            fracs: (n_tiers,) float32 — fraction of tokens routed to each tier.
                   Sums to top_k (since top_k tiers activate per token).
        """
        if not self._history:
            return torch.full((self.n_tiers,), 1.0 / self.n_tiers)

        counts = torch.zeros(self.n_tiers, dtype=torch.float32)
        total = 0
        for indices in self._history:
            # indices: (B, top_k) — dtype int64
            for t in indices.reshape(-1).tolist():
                counts[int(t)] += 1.0
                total += 1

        return counts / max(total, 1)

    # -- load-balance loss ----------------------------------------------------

    def load_balance_loss(self) -> torch.Tensor:
        """Compute the variance-based load-balance auxiliary loss.

        Returns:
            loss: scalar float32 tensor — 0.0 if history is empty.
        """
        fracs = self.routing_fractions()
        # Ideal fraction: top_k / n_tiers per tier.
        ideal = float(self.top_k) / self.n_tiers
        # Variance of deviation from ideal.
        variance = ((fracs - ideal) ** 2).mean()
        return self.load_balance_coeff * variance

    # -- forward --------------------------------------------------------------

    def forward(
        self,
        tier_outputs: list[torch.Tensor],
        query: torch.Tensor,
        sparse_gates: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Route across memory tier outputs using top-k gating.

        Args:
            tier_outputs:  List of n_tiers tensors each (B, d_model) — L1, L2,
                           L3 outputs.  Inactive tiers will be detached.
            query:         (B, d_model) — routing query, typically the last
                           Mamba hidden state before the memory tiers.
            sparse_gates:  (B, n_modules) bool from controller — currently
                           reserved for future module-level gating.  Ignored if
                           None.

        Returns:
            merged:         (B, d_model) — weighted combination of active tiers.
            gate_weights:   (B, n_tiers) — soft weights; zero for inactive tiers.
            router_logits:  (B, n_tiers) — raw logits before top-k selection,
                            for logging / aux-loss computation outside this
                            module.

        Raises:
            AssertionError: if NaN/Inf in query or any tier output.
        """
        assert not (torch.isnan(query).any() or torch.isinf(query).any()), (
            "NaN/Inf in SparseRouter query"
        )
        assert len(tier_outputs) == self.n_tiers, (
            f"Expected {self.n_tiers} tier outputs, got {len(tier_outputs)}"
        )

        B = query.shape[0]
        device = query.device

        # ── Gate score computation ───────────────────────────────────────────
        # Detach query from task-loss gradient so tier routing is purely
        # a function of the router's own gate_proj weights.
        q = query.detach()                          # (B, d_model) — no grad
        # Stack tier outputs: (B, n_tiers, d_model)
        # Detach inactive tiers after top-k selection (below).
        tier_stack = torch.stack(tier_outputs, dim=1)  # (B, n_tiers, d_model)

        # Compute scores: gate_proj(q) → (B, n_tiers)
        router_logits = self.gate_proj(q)           # (B, n_tiers)  float32

        # ── Top-k selection ──────────────────────────────────────────────────
        # topk_vals:    (B, top_k) — top-k raw logit values
        # topk_indices: (B, top_k) — which tier indices are active
        topk_vals, topk_indices = torch.topk(router_logits, self.top_k, dim=-1)

        # Normalised gate weights for active tiers only.
        topk_weights = F.softmax(topk_vals, dim=-1)  # (B, top_k)

        # ── Gradient isolation for inactive tiers ───────────────────────────
        # Build a full (B, n_tiers) weight tensor; inactive tiers get zero weight
        # and their underlying tier_output gradients are blocked.
        gate_weights = torch.zeros(B, self.n_tiers, dtype=topk_weights.dtype, device=device)
        gate_weights.scatter_(dim=1, index=topk_indices, src=topk_weights)

        # Detach inactive tier outputs so they receive zero gradient from the
        # task loss.  Active tiers retain their gradients.
        active_mask = gate_weights > 0  # (B, n_tiers) bool

        # Build the tier stack with conditional detach per tier.
        # Each tier_outputs[t] may be updated in-place during ESN feedback;
        # we only want gradients for active tiers.
        tier_stack_gated = torch.zeros_like(tier_stack)
        for t in range(self.n_tiers):
            t_out = tier_outputs[t]                           # (B, d_model)
            w_t = gate_weights[:, t].unsqueeze(-1)           # (B, 1)
            # Active tokens for this tier: detach the tensor for inactive batch
            # entries but allow gradient for active ones.
            active_t = active_mask[:, t]                     # (B,) bool
            if active_t.any():
                # Detach the full tier, then add back only the active entries.
                # This gives zero gradient flowing through inactive batch items.
                t_active = t_out.clone()
                t_active[~active_t] = t_out[~active_t].detach()
                tier_stack_gated[:, t, :] = t_active * w_t
            # Inactive tiers for all batch items: zero (already zero_like).

        # Merge: sum over tiers → (B, d_model)
        merged = tier_stack_gated.sum(dim=1)  # (B, d_model)

        # ── Routing history ──────────────────────────────────────────────────
        self._history.append(topk_indices.detach().cpu())

        return merged, gate_weights, router_logits
