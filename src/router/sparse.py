"""DREX-UNIFIED COMPONENT 8: Sparse Router.

Top-k gating over memory tier outputs with load-balance auxiliary loss.

Gradient isolation contract
---------------------------
Inactive tier tensors receive EXACTLY zero gradient — not a zero-tensor grad,
but ``t.grad is None`` after backward.  This is achieved by never entering
inactive tier tensors into the computation graph at all.  A multiply-by-zero
mask does NOT satisfy this contract: it produces ``t.grad = zeros`` and still
runs backward through the entire inactive tier path.

The implementation uses an explicit ``(rank, tier)`` double-loop with
``mask.any()`` guards.  When a tier is inactive for all batch elements, its
tensor is never indexed and never participates in any autograd node.

Dtype contract
--------------
query input may be bfloat16 (Mamba backbone output at integration time).
It is cast to float32 at the top of forward().  All outputs are float32.

Gating algorithm (DREX_UNIFIED_SPEC.md § COMPONENT 8)
------------------------------------------------------
1. Detach tier outputs before scoring — score-path gradients must not bleed
   into tier tensors (only the merge path carries tier gradients).
2. score_i = dot(query, tier_i)                   → (B, n_tiers)
3. top_k by score                                 → topk_idx (B, top_k)
4. softmax over top_k scores                      → weights (B, top_k)
5. merged = Σ_(k,t_i) weights[b, k] * tier_i[b]  via index_add over active pairs
6. routing_weights = scatter(weights → (B, n_tiers)) for logging + lb_loss
7. lb_loss = load_balance_coeff * Var(mean routing fraction per tier)

Ref: DREX_UNIFIED_SPEC.md § COMPONENT 8: SPARSE ROUTER
"""
from __future__ import annotations

import torch
import torch.nn as nn


class SparseRouter(nn.Module):
    """Gates memory tier outputs via score-based top-k selection.

    No learnable parameters — scoring is purely via dot product between the
    query vector and each (detached) tier output.  This keeps the router at
    ~0 parameter overhead while still routing adaptively per input.

    Args:
        d_model:             Dimension of tier outputs and query vector.
        n_tiers:             Number of memory tiers.  Default 3.
        top_k:               Number of tiers to activate per token.  Default 2.
                             Must be ≤ n_tiers.
        load_balance_coeff:  Coefficient for load-balance auxiliary loss.
                             Default 0.01.

    Raises:
        ValueError: if top_k > n_tiers.
    """

    def __init__(
        self,
        d_model: int,
        n_tiers: int = 3,
        top_k: int = 2,
        load_balance_coeff: float = 0.01,
    ) -> None:
        super().__init__()
        if top_k > n_tiers:
            raise ValueError(
                f"top_k ({top_k}) must be ≤ n_tiers ({n_tiers})"
            )
        self.d_model = d_model
        self.n_tiers = n_tiers
        self.top_k = top_k
        self.load_balance_coeff = load_balance_coeff

    def forward(
        self,
        tier_outputs: list[torch.Tensor],
        query: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Route query across memory tiers via top-k gating.

        Args:
            tier_outputs: list of n_tiers tensors, each (B, d_model) float32.
            query:        (B, d_model), may be bfloat16 (cast to float32 here).

        Returns:
            merged:          (B, d_model) float32 — weighted sum of active tiers.
            routing_weights: (B, n_tiers) float32 — softmax weights, 0 inactive.
            lb_loss:         scalar float32 — load-balance auxiliary loss.
        """
        # Explicit cast — bfloat16 from Mamba must become float32 here.
        query = query.float()
        B = query.shape[0]

        # --- Scoring: detach tier outputs so score path carries no tier grads ---
        tier_stack_det = torch.stack(
            [t.detach().float() for t in tier_outputs], dim=1
        )  # (B, n_tiers, d_model) — fully detached from tier computation graphs

        scores = torch.bmm(
            tier_stack_det, query.unsqueeze(-1)
        ).squeeze(-1)  # (B, n_tiers)

        topk_vals, topk_idx = scores.topk(self.top_k, dim=-1)  # (B, top_k)
        weights = torch.softmax(topk_vals, dim=-1)  # (B, top_k)

        # --- Merge: only active (rank, tier) pairs enter the computation graph ---
        # The double loop runs top_k × n_tiers = 6 iterations at default config.
        # Inactive tiers pass the mask.any() guard and are never indexed —
        # guaranteeing t.grad is None (not zeros) after backward.
        merged = torch.zeros(B, self.d_model, dtype=torch.float32, device=query.device)
        for k in range(self.top_k):
            for t_i, t_out in enumerate(tier_outputs):
                # Which batch elements activate tier t_i at rank k?
                mask = topk_idx[:, k] == t_i  # (B,) bool
                if not mask.any():
                    continue  # tier t_i never selected at this rank — skip entirely

                b_idx = mask.nonzero(as_tuple=True)[0]  # 1-D active batch indices
                contrib = (
                    weights[b_idx, k].unsqueeze(-1) * t_out[b_idx].float()
                )  # (|b_idx|, d_model) — gradient flows to t_out and query
                merged = merged.index_add(0, b_idx, contrib)

        # --- routing_weights full tensor for logging / load-balance loss ---
        # Built from live weights (differentiable path to query via scores).
        routing_weights = torch.zeros(
            B, self.n_tiers, dtype=torch.float32, device=query.device
        ).scatter(1, topk_idx, weights)  # (B, n_tiers)

        lb_loss = self.load_balance_loss(routing_weights)
        return merged, routing_weights, lb_loss

    def load_balance_loss(self, routing_weights: torch.Tensor) -> torch.Tensor:
        """Load-balance auxiliary loss to prevent tier routing collapse.

        Minimising this loss encourages all tiers to receive approximately
        equal routing weight across batch elements.

        Args:
            routing_weights: (B, n_tiers) float32 — full routing weight tensor
                             as returned by forward().

        Returns:
            Scalar float32: load_balance_coeff * Var(mean fraction per tier).
        """
        fraction_per_tier = routing_weights.mean(dim=0)  # (n_tiers,)
        return self.load_balance_coeff * fraction_per_tier.var()
