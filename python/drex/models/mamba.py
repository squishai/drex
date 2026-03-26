"""
drex.models.mamba — Mamba-1 Selective SSM Layer (Gu & Dao 2023).

Pure-PyTorch implementation that runs on any device (CPU, MPS, CUDA).
No dependency on the mamba-ssm package, which requires CUDA.

Architecture:
    MambaSSM   — standalone Mamba-1 block; maps (B, S, D) → (B, S, D).
    MambaLayer — MambaSSM with optional L2 InfiniAttention cross-segment
                 gating.  Drop-in replacement for HybridAttention with the
                 same forward signature:
                     (x: Tensor, state: MemoryState) → (Tensor, MemoryState)

Phase 25 integration:
    - MambaSSM replaces SlidingWindowAttention (L1 local context).
    - L2 InfiniAttention cross-segment memory is preserved when use_l2=True.
    - DrexLayer.attn is assigned a MambaLayer when DrexConfig.use_mamba=True.
    - DrexTransformer.forward() and LayerState are unchanged — Mamba is
      stateless within a segment (h=0 at each segment boundary), consistent
      with how SlidingWindowAttention was also stateless across segments.

Selective SSM (Mamba-1):
    Given input u (B, S, d_inner) and parameters A, B, C, Δ (input-dependent):
        1. Discretise:  Ā = exp(Δ ⊗ A),   B̄ = Δ ⊗ B  (approximate ZOH)
        2. Scan:        h_t = Ā_t ⊙ h_{t-1} + B̄_t ⊙ u_t
                        y_t = ⟨C_t, h_t⟩
        3. D skip:      y  += D ⊙ u

Hard constraints (carry-over from Phase 1-16 research):
    - spectral_radius of Ā must be < 1 for stable SSM (log_A > 0 ensures A_neg < 0).
    - dt_min=0.001, dt_max=0.1 gives a good prior over timescales (Mamba paper §A.3).

References:
    Gu & Dao 2023 — Mamba: Linear-Time Sequence Modeling with Selective State Spaces.
    Dao & Gu 2024 — Transformers are SSMs: Generalized Models and Efficient Algorithms.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from drex.models.memory import MemoryState

# ---------------------------------------------------------------------------
# Δ (timescale) initialisation helper
# ---------------------------------------------------------------------------

_DT_MIN: float = 0.001
_DT_MAX: float = 0.100


def _dt_init(
    dt_proj: nn.Linear,
    dt_min: float = _DT_MIN,
    dt_max: float = _DT_MAX,
) -> None:
    """
    Initialise the bias of dt_proj so softplus(bias) is uniform in
    [dt_min, dt_max], giving a good prior over SSM timescales.

    Derivation (Mamba paper §A.3):
        softplus_inv(y) = y + log(1 − exp(−y))  for y > 0.
    We draw dt ~ Uniform(log dt_min, log dt_max) in log-space, then set
    bias = softplus_inv(dt), so softplus(bias) = dt.
    """
    dt = torch.exp(
        torch.rand(dt_proj.out_features)
        * (math.log(dt_max) - math.log(dt_min))
        + math.log(dt_min)
    )
    # Numerically stable inverse softplus: log(exp(dt) − 1) = dt + log(1 − exp(−dt))
    inv_sp = dt + torch.log(-torch.expm1(-dt))
    with torch.no_grad():
        dt_proj.bias.copy_(inv_sp)
    dt_proj.bias._no_reinit = True  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Core SSM block
# ---------------------------------------------------------------------------


class MambaSSM(nn.Module):
    """
    Mamba-1 SSM block (pure PyTorch, device-agnostic).

    Args:
        d_model: input / output dimension.
        d_state: SSM state expansion N (default 16).
        d_conv:  causal depthwise-conv kernel width (default 4).
        expand:  inner-dimension expansion factor E (default 2).

    Derived dims:
        d_inner  = expand × d_model
        dt_rank  = max(1, d_model // 16)   (low-rank Δ projection)

    Parameter layout:
        in_proj   (d_model → 2·d_inner)                — split into x, z branches
        conv1d    (d_inner depthwise, width d_conv)     — causal local context
        x_proj    (d_inner → dt_rank + 2·d_state)       — Δ_raw, B_ss, C_ss
        dt_proj   (dt_rank → d_inner)                   — expand Δ to d_inner
        log_A     (d_inner, d_state)                    — log(-A), always > 0
        D         (d_inner,)                             — skip connection
        out_proj  (d_inner → d_model)                   — output projection
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner: int = expand * d_model
        self.dt_rank: int = max(1, d_model // 16)

        # Dual-branch input projection
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)

        # Causal depthwise conv on x branch (groups = d_inner for depthwise)
        self.conv1d = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,   # left-only causal padding; right trimmed in forward
            bias=True,
        )

        # Low-rank projections: [Δ_raw | B_ss | C_ss]
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)

        # Δ expansion: dt_rank → d_inner (bias carries the timescale prior)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # A eigenvalues: parameterised as log(-A) (A is always negative = stable SSM)
        # Init: A[i, n] = i + 1  (same index as HiPPO / S4 convention)
        A_init = torch.arange(1, self.d_inner + 1, dtype=torch.float32).unsqueeze(1).expand(
            self.d_inner, d_state
        )
        self.log_A = nn.Parameter(torch.log(A_init.clone()))

        # D skip (learnable, init to ones)
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.in_proj.weight, std=0.02)
        nn.init.trunc_normal_(self.x_proj.weight, std=0.02)
        nn.init.trunc_normal_(self.out_proj.weight, std=0.02)
        # Depthwise conv: near-identity init
        nn.init.ones_(self.conv1d.weight)
        nn.init.zeros_(self.conv1d.bias)
        # dt_proj weight: small normal; bias: timescale prior (done after)
        nn.init.trunc_normal_(self.dt_proj.weight, std=0.02)
        _dt_init(self.dt_proj)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, S, d_model)
        Returns: (B, S, d_model)
        """
        B, S, _ = x.shape
        d_inner, d_state = self.d_inner, self.d_state

        # ── 1. Dual-branch input split ──────────────────────────────────────
        xz = self.in_proj(x)                          # (B, S, 2·d_inner)
        x_branch, z_branch = xz.chunk(2, dim=-1)      # each (B, S, d_inner)

        # ── 2. Causal depthwise conv ────────────────────────────────────────
        # conv1d expects (B, C, L); padding=d_conv-1 on both sides, trim right
        x_t = self.conv1d(x_branch.transpose(1, 2))[:, :, :S]  # (B, d_inner, S)
        x_branch = F.silu(x_t.transpose(1, 2))        # (B, S, d_inner)

        # ── 3. Input-dependent SSM parameters ──────────────────────────────
        x_dbc = self.x_proj(x_branch)                 # (B, S, dt_rank + 2·d_state)
        delta_raw = x_dbc[:, :, : self.dt_rank]           # (B, S, dt_rank)
        B_ss = x_dbc[:, :, self.dt_rank : self.dt_rank + d_state]  # (B, S, d_state)
        C_ss = x_dbc[:, :, self.dt_rank + d_state :]               # (B, S, d_state)

        # Δ: expand via dt_proj, then softplus to ensure Δ > 0
        delta = F.softplus(self.dt_proj(delta_raw))    # (B, S, d_inner)

        # A_neg: stable negative eigenvalues  (d_inner, d_state)
        A_neg = -torch.exp(self.log_A)

        # ── 4+5. Discretise on-the-fly and selective scan ────────────────────
        # Computing Ā and B̄ per timestep avoids pre-allocating
        # (B, S, d_inner, d_state) tensors — at Sprint-5 scale (B=32, S=512,
        # d_inner=512, d_state=16) that would be 2×536 MB per layer × 8 layers
        # = >8 GB, overflowing a 14-16 GB T4/P100.  Instead we compute each
        # (B, d_inner, d_state) slice (~1 MB) inside the loop.
        #
        #   Ā_t = exp(Δ_t · A_neg)       (B, d_inner, d_state)
        #   B̄_t = Δ_t · B_t             (B, d_inner, d_state)   [broadcast]
        #   h_t = Ā_t ⊙ h_{t-1} + B̄_t ⊙ u_t
        #   y_t = ⟨C_t, h_t⟩            (B, d_inner)
        h = x.new_zeros(B, d_inner, d_state)
        ys: list[torch.Tensor] = []
        for t in range(S):
            d_t = delta[:, t].unsqueeze(-1)                         # (B, d_inner, 1)
            a_bar_t = torch.exp(d_t * A_neg)                        # (B, d_inner, d_state)
            b_bar_t = d_t * B_ss[:, t].unsqueeze(1)                 # (B, d_inner, d_state)
            h = a_bar_t * h + b_bar_t * x_branch[:, t].unsqueeze(-1)
            ys.append((h * C_ss[:, t].unsqueeze(1)).sum(-1))        # (B, d_inner)
        y = torch.stack(ys, dim=1)                                  # (B, S, d_inner)

        # ── 6. D skip + output gate ──────────────────────────────────────────
        y = y + self.D * x_branch                     # learnable skip connection
        y = y * F.silu(z_branch)                      # gating branch

        # ── 7. Output projection ─────────────────────────────────────────────
        return self.out_proj(y)                        # (B, S, d_model)


# ---------------------------------------------------------------------------
# Drop-in replacement for HybridAttention
# ---------------------------------------------------------------------------


class MambaLayer(nn.Module):
    """
    MambaLayer — Mamba-1 SSM with optional L2 InfiniAttention cross-segment gating.

    Drop-in replacement for HybridAttention in DrexLayer when
    DrexConfig.use_mamba=True.

    Forward signature (identical to HybridAttention):
        x:     (B, S, d_model)        — current segment hidden states
        state: MemoryState            — L2 cross-segment memory (M, z)
        → output:   (B, S, d_model)
        → new_state: MemoryState      — updated L2 memory, or unchanged if use_l2=False

    When use_l2=True:
        output = sigmoid(gate) · L2(x) + (1 − sigmoid(gate)) · Mamba(x)
        gate is a single learnable scalar, initialised to 0 (equal mix at start).

    When use_l2=False:
        output = Mamba(x)
        state is passed through unchanged.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        use_l2: bool = True,
    ) -> None:
        super().__init__()
        self.ssm = MambaSSM(d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self._use_l2 = use_l2

        if use_l2:
            from drex.models.attention import InfiniAttention
            self.l2 = InfiniAttention(d_model, n_heads)
            # Single learnable merge gate (init=0 → equal mix at start)
            self.gate = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        x: torch.Tensor,
        state: MemoryState,
    ) -> tuple[torch.Tensor, MemoryState]:
        ssm_out = self.ssm(x)                         # (B, S, d_model)

        if self._use_l2:
            l2_out, new_state = self.l2(x, state)     # (B, S, d_model), MemoryState
            g = torch.sigmoid(self.gate)
            out = g * l2_out + (1.0 - g) * ssm_out
        else:
            out = ssm_out
            new_state = state

        return out, new_state
