"""
drex.models.memory_esn — Echo State Network (reservoir computing) memory module.

EchoStateMemory is a drop-in replacement for MemoryModule (Phase 13) that replaces
the trained associative delta-rule matrices with a fixed random reservoir (ESN).

Architecture (Phase 23, DREX-UNIFIED roadmap):
  - Two fixed random reservoirs: semantic (N_sem neurons) and episodic (N_epi).
  - N_sem = N_epi = reservoir_mult × d_model (default: 4×).  Registered as buffers
    (persistent across saves/loads but excluded from optimizer).
  - OR relative-norm write gate using EMA tracking vectors (same validated gate
    criterion as MemoryModule; same α(L) = 0.95^(96/L) formula, Phase 11).
  - Gate-controlled ESN state update: h_t = tanh(W_res @ h_{t-1} + W_in @ k_t)
    only when gate fires; h_t = h_{t-1} otherwise (selective state evolution).
  - Readout: learned linear projections on final reservoir states (the ONLY trained
    components in the reservoir path).
  - Null retrieval gate, output projection, and LayerNorm — identical to MemoryModule.

Forward contract (identical to MemoryModule):
    x      : (B, L, d_model)
    returns: (B, d_model)

Hard constraints (carry-over from Phase 1-16 research):
    - gate_thresh must not be set below 0.40 (exp_43_1).
    - Use α(L) = 0.95^(96/L), never fixed α (EMA bootstrap, Phase 11).
    - Validate write rate ∈ [0.10, 0.85] after any write-mechanism change.
    - spectral_radius < 1 required for echo state property.

Key difference from MemoryModule:
    MemoryModule: delta-rule writes accumulate in trained H/2×H/2 matrices.
    EchoStateMemory: ESN state evolves through fixed random dynamics; only the
    linear readout (N → d_half) is trained.  Training cost for the reservoir
    component is zero — only the readout parameters require gradient updates.

Relevant literature:
    Jaeger & Haas 2004 (Science) — original practical ESN demonstration.
    BabyLM 2025 — ESN matches Transformer on grammaticality at ~100M-word scale.
    Nature 2025 — Attention-enhanced reservoirs close to Transformer performance.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Constants — shared with MemoryModule for API consistency
# ---------------------------------------------------------------------------

# EMA decay formula: α(L) = ALPHA_REF^(L_REF / L)
# Keeps τ/L ≈ 0.21 constant across L=32-128 (exp_47_3, Phase 11).
_ALPHA_REF: float = 0.95
_L_REF: int = 96

# Acceptable write-rate window during training (exp_45, Phase 9).
WRITE_RATE_LO: float = 0.10
WRITE_RATE_HI: float = 0.85

# Number of power-iteration steps used to estimate the spectral norm at init.
_SPECTRAL_ITER: int = 20


# ---------------------------------------------------------------------------
# Reservoir construction
# ---------------------------------------------------------------------------


def _make_reservoir(
    N: int,
    connectivity: float,
    spectral_radius: float,
    generator: torch.Generator,
) -> torch.Tensor:
    """
    Create a sparse random reservoir matrix scaled to the target spectral radius.

    Procedure:
        1. Sample N×N Gaussian weights.
        2. Apply sparsity mask (retain ``connectivity`` fraction of entries).
        3. Use power iteration (_SPECTRAL_ITER steps) to estimate the dominant
           singular value (spectral norm), which upper-bounds the spectral radius.
        4. Scale so the spectral norm equals ``spectral_radius``.

    The resulting matrix satisfies spectral_norm(W) == spectral_radius, which
    guarantees the echo state property (spectral_radius < 1 ⇒ fading memory).

    Args:
        N            : Reservoir size (side length of square weight matrix).
        connectivity : Fraction of non-zero entries (typical: 0.01).
        spectral_radius: Target spectral norm (must be < 1 for echo state property).
        generator    : Seeded RNG for reproducible reservoir construction.

    Returns:
        Float32 tensor of shape (N, N).
    """
    if not (0.0 < spectral_radius < 1.0):
        raise ValueError(
            f"spectral_radius must be in (0, 1) for echo state property, got {spectral_radius}"
        )

    W = torch.randn(N, N, generator=generator)

    # Sparsity mask: retain ~connectivity fraction of connections.
    mask = torch.rand(N, N, generator=generator) < connectivity
    W = W * mask.float()

    # Power iteration for dominant singular value.
    # Initialize with a random unit vector; alternate W and W^T applications.
    v = F.normalize(torch.randn(N, 1, generator=generator), dim=0)
    sigma = 1.0
    for _ in range(_SPECTRAL_ITER):
        u = F.normalize(W @ v, dim=0)
        v = F.normalize(W.t() @ u, dim=0)
        # Rayleigh quotient: σ ≈ u^T W v
        sigma = (u.t() @ W @ v).item()

    if abs(sigma) > 1e-10:
        W = W * (spectral_radius / abs(sigma))

    return W.float()


# ---------------------------------------------------------------------------
# EchoStateMemory
# ---------------------------------------------------------------------------


class EchoStateMemory(nn.Module):
    """
    Echo State Network memory module — drop-in replacement for MemoryModule.

    Replaces the trained associative delta-rule matrices with a pair of fixed
    random reservoirs (semantic and episodic branches).  The reservoir weights
    are registered as non-parameter buffers: they are persisted in checkpoints
    and moved by ``.to(device)`` but excluded from the optimizer.

    The ONLY trained parameters are:
        sem_proj    : Linear(d_model, d_half)  — semantic key projection
        epi_proj    : Linear(d_model, d_half)  — episodic key projection
        sem_readout : Linear(N, d_half)        — semantic reservoir readout
        epi_readout : Linear(N, d_half)        — episodic reservoir readout
        null_gate   : Linear(d_model, 1)       — null retrieval gate (optional)
        out_proj    : Linear(d_model, d_model) — output projection
        norm_out    : LayerNorm(d_model)        — output normalization

    Write gate:
        Identical criterion to MemoryModule (OR relative-norm gate):
            fire iff ||k_s − ema_ks|| ≥ thresh·||k_s||
                  OR  ||k_e − ema_ke|| ≥ thresh·||k_e||
        Where ema_ks / ema_ke are per-sequence EMA tracking vectors (not trained;
        reset to zero at each forward call, like M_sem / M_epi in MemoryModule).
        α(L) = 0.95^(96/L) — same validated length-adaptive formula (Phase 11).

    Reservoir update (gate-controlled):
        h_sem_t = tanh(h_sem_{t-1} @ W_res_sem^T + k̂_s_t @ W_in_sem^T)  if fire
        h_sem_t = h_sem_{t-1}                                              otherwise
        (episodic branch analogous, with optional recency weighting on W_in input)

    Readout:
        r_sem = sem_readout(h_sem_{L-1})     # (B, d_half)
        r_epi = epi_readout(h_epi_{L-1})    # (B, d_half)
        r     = concat(r_sem, r_epi)        # (B, d_model) — same as MemoryModule

    See DREX_UNIFIED_PLAN.md Phase 23 for experiment specifications (exp_53 / exp_54).
    """

    def __init__(
        self,
        d_model: int,
        reservoir_mult: int = 4,
        spectral_radius: float = 0.95,
        connectivity: float = 0.01,
        gate_thresh: float = 0.70,
        use_null_gate: bool = True,
        use_recency_weight: bool = True,
        reservoir_seed: int = 42,
    ) -> None:
        """
        Args:
            d_model         : Model hidden dimension (must be even).
            reservoir_mult  : Reservoir size N = reservoir_mult × d_model per branch.
            spectral_radius : ESN spectral radius (must be < 1; default 0.95).
            connectivity    : Fraction of non-zero reservoir weights (~0.01 = 1%).
            gate_thresh     : OR-gate threshold (must be ≥ 0.40; default 0.70).
            use_null_gate   : Include learned null retrieval gate (default True).
            use_recency_weight: Apply recency weight w_t=(t+1)/L to episodic branch.
            reservoir_seed  : RNG seed for reproducible reservoir construction.
        """
        super().__init__()

        if d_model % 2 != 0:
            raise ValueError(
                f"d_model must be even for episodic/semantic split, got {d_model}"
            )
        if gate_thresh < 0.40:
            raise ValueError(
                f"gate_thresh must be ≥ 0.40 (exp_43_1 hard constraint), got {gate_thresh}"
            )
        if not (0.0 < spectral_radius < 1.0):
            raise ValueError(
                f"spectral_radius must be in (0, 1) for echo state property, got {spectral_radius}"
            )

        self.d_model = d_model
        self.gate_thresh = gate_thresh
        self.use_null_gate = use_null_gate
        self.use_recency_weight = use_recency_weight

        d_half = d_model // 2
        self._d_half = d_half
        N = d_model * reservoir_mult
        self._N = N

        # ── Trained key projections (same as MemoryModule) ──────────────────
        self.sem_proj = nn.Linear(d_model, d_half, bias=False)
        self.epi_proj = nn.Linear(d_model, d_half, bias=False)

        # ── Trained readout layers (only new trained parameters vs MemoryModule) ─
        # Map final reservoir state → branch output (N → d_half each).
        self.sem_readout = nn.Linear(N, d_half, bias=False)
        self.epi_readout = nn.Linear(N, d_half, bias=False)

        # ── Optional null retrieval gate ─────────────────────────────────────
        self.null_gate: Optional[nn.Linear] = (
            nn.Linear(d_model, 1) if use_null_gate else None
        )

        # ── Output path (identical to MemoryModule) ──────────────────────────
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm_out = nn.LayerNorm(d_model)

        # ── Fixed random reservoir weights (registered as buffers) ───────────
        # Two independent generators seeded from reservoir_seed so that semantic
        # and episodic reservoirs are distinct but reproducible.
        rng_sem = torch.Generator().manual_seed(reservoir_seed)
        rng_epi = torch.Generator().manual_seed(reservoir_seed + 1)

        # W_res: (N, N) — recurrent reservoir weights, fixed.
        # W_in:  (N, d_half) — input projection into reservoir, fixed.
        self.register_buffer(
            "W_res_sem",
            _make_reservoir(N, connectivity, spectral_radius, rng_sem),
        )
        self.register_buffer(
            "W_in_sem",
            (torch.randn(N, d_half, generator=rng_sem) * 0.1).float(),
        )
        self.register_buffer(
            "W_res_epi",
            _make_reservoir(N, connectivity, spectral_radius, rng_epi),
        )
        self.register_buffer(
            "W_in_epi",
            (torch.randn(N, d_half, generator=rng_epi) * 0.1).float(),
        )

        # Write-rate from the most recent forward pass.
        self._last_write_rate: float = 0.0

    # ── Class-level constants for convenience ────────────────────────────────

    @staticmethod
    def alpha(L: int) -> float:
        """Length-adaptive EMA coefficient: α(L) = 0.95^(96/L)."""
        return _ALPHA_REF ** (_L_REF / L)

    def last_write_rate(self) -> float:
        """Fraction of (batch × step) positions where the OR gate fired."""
        return self._last_write_rate

    def assert_write_rate_valid(self) -> None:
        """Raise AssertionError if the last write rate is outside [0.10, 0.85]."""
        wr = self._last_write_rate
        assert WRITE_RATE_LO <= wr <= WRITE_RATE_HI, (
            f"Write rate {wr:.3f} outside valid range "
            f"[{WRITE_RATE_LO}, {WRITE_RATE_HI}]. "
            "Check gate_thresh or sequence-length configuration."
        )

    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process the full context x and return the memory retrieval for the
        query token at position L-1.

        Write phase (positions 0..L-2):
            For each position t, compute semantic and episodic keys from x_t.
            Use the OR relative-norm gate (EMA tracking) to decide whether to
            advance the reservoir state.  The EMA tracks the running average of
            seen keys and fires when the current key is sufficiently novel.
            When the gate fires: h_t = tanh(W_res @ h_{t-1} + W_in @ k̂_t).
            When silent:         h_t = h_{t-1}.

        Read phase (position L-1):
            r_sem = sem_readout(h_sem)   — learned linear on final sem state
            r_epi = epi_readout(h_epi)  — learned linear on final epi state
            r     = concat(r_sem, r_epi) — (B, d_model)

        Args:
            x: (B, L, d_model) — token representations.

        Returns:
            (B, d_model) — bounded memory retrieval (after norm_out).
        """
        B, L, _ = x.shape
        a = self.alpha(L)
        d_half = self._d_half
        N = self._N
        device = x.device

        # Initialise reservoir hidden states and EMA policy vectors.
        # All reset to zero at each forward call (same as M_sem/M_epi in MemoryModule).
        h_sem = torch.zeros(B, N, device=device)
        h_epi = torch.zeros(B, N, device=device)

        if L > 1:
            # ── Vectorised pre-computation (GPU) ─────────────────────────────
            ks_all = self.sem_proj(x[:, :-1, :])             # (B, L-1, d_half)
            ke_all = self.epi_proj(x[:, :-1, :])
            kns_all = F.normalize(ks_all, dim=-1, eps=1e-6)  # unit keys, semantic
            kne_all = F.normalize(ke_all, dim=-1, eps=1e-6)  # unit keys, episodic

            # Gate reference norm (thresh × ||k||): (B, L-1)
            ref_s_all = self.gate_thresh * ks_all.norm(dim=-1)
            ref_e_all = self.gate_thresh * ke_all.norm(dim=-1)

            # ── Sequential reservoir loop on CPU (detached) ──────────────────
            # Same CPU-offloading strategy as MemoryModule: move matrices to CPU,
            # run the sequential recurrence inside torch.no_grad(), then move
            # final states back to the original device for the read phase.
            # This avoids both MPS per-kernel-launch overhead and O(L) autograd
            # graph construction for sequential state updates.
            cpu = torch.device("cpu")
            h_sem = h_sem.to(cpu)
            h_epi = h_epi.to(cpu)

            # Load reservoir buffers on CPU (already float32; .to(cpu) is no-op if
            # the model lives on CPU, cheap if on MPS/CUDA).
            W_res_sem_c = self.W_res_sem.to(cpu)
            W_in_sem_c = self.W_in_sem.to(cpu)
            W_res_epi_c = self.W_res_epi.to(cpu)
            W_in_epi_c = self.W_in_epi.to(cpu)

            ks_c = ks_all.detach().to(cpu)      # (B, L-1, d_half)
            ke_c = ke_all.detach().to(cpu)
            kns_c = kns_all.detach().to(cpu)
            kne_c = kne_all.detach().to(cpu)
            ref_s_c = ref_s_all.detach().to(cpu)
            ref_e_c = ref_e_all.detach().to(cpu)

            # EMA policy vectors for gate decisions.
            # These track a running average of seen keys and serve as gate
            # "predictions" — analogous to M_sem @ k̂ in MemoryModule.
            ema_ks = torch.zeros(B, d_half, device=cpu)
            ema_ke = torch.zeros(B, d_half, device=cpu)

            fires: list[torch.Tensor] = []

            with torch.no_grad():
                for t in range(L - 1):
                    kns_t = kns_c[:, t]   # (B, d_half) — unit semantic key
                    kne_t = kne_c[:, t]   # (B, d_half) — unit episodic key
                    ks_t = ks_c[:, t]     # (B, d_half) — unnormalised semantic key
                    ke_t = ke_c[:, t]     # (B, d_half) — unnormalised episodic key

                    # OR write gate: fire when prediction error exceeds threshold.
                    # ema_ks is the running average of seen semantic keys (initialised
                    # to zero, so the first token always triggers).
                    err_s = (ks_t - ema_ks).norm(dim=-1)   # (B,)
                    err_e = (ke_t - ema_ke).norm(dim=-1)

                    fire = (
                        (err_s >= ref_s_c[:, t]) | (err_e >= ref_e_c[:, t])
                    ).float()  # (B,)
                    fires.append(fire)

                    gate = fire[:, None]   # (B, 1) — broadcast mask

                    # Recency weight (episodic branch only, same as MemoryModule).
                    w_t = (t + 1) / L if self.use_recency_weight else 1.0

                    # EMA update (gate-controlled, same α(L) as MemoryModule).
                    # The EMA only advances when the gate fires — same semantics as
                    # the delta-rule write in MemoryModule.
                    ema_ks = ema_ks + (1.0 - a) * gate * (ks_t - ema_ks)
                    ema_ke = ema_ke + (1.0 - a) * w_t * gate * (ke_t - ema_ke)

                    # ESN reservoir update.
                    # pre = W_res @ h_{t-1} + W_in @ k̂_t
                    # Shapes: (B,N) @ (N,N).T = (B,N); (B,d_half) @ (N,d_half).T = (B,N)
                    pre_sem = (
                        torch.mm(h_sem, W_res_sem_c.t())
                        + torch.mm(kns_t, W_in_sem_c.t())
                    )
                    pre_epi = (
                        torch.mm(h_epi, W_res_epi_c.t())
                        + w_t * torch.mm(kne_t, W_in_epi_c.t())
                    )

                    h_sem_new = torch.tanh(pre_sem)   # (B, N)
                    h_epi_new = torch.tanh(pre_epi)

                    # Gate-controlled state update: h_t = h_{t-1} + fire·(h_new − h_{t-1})
                    # When fire=0, h stays unchanged.  When fire=1, h = h_new.
                    h_sem = h_sem + gate * (h_sem_new - h_sem)
                    h_epi = h_epi + gate * (h_epi_new - h_epi)

            # Move final reservoir states back to the original device for the read phase.
            h_sem = h_sem.to(device)
            h_epi = h_epi.to(device)

            fires_t = torch.stack(fires, dim=1)   # (B, L-1) on CPU
            self._last_write_rate = fires_t.sum().item() / max(B * (L - 1), 1)

        else:
            self._last_write_rate = 0.0

        # ── Read phase (query at position L-1) ───────────────────────────────
        q = x[:, -1, :]                                              # (B, d_model)
        qns = F.normalize(self.sem_proj(q), dim=-1, eps=1e-6)       # (B, d_half)
        qne = F.normalize(self.epi_proj(q), dim=-1, eps=1e-6)

        # Query-conditioned readout: element-wise multiplication of the reservoir
        # output with the projected query key.  Analogous to M_sem @ qns in
        # MemoryModule — content-addressed filtering of the reservoir state.
        # Gradient flows through sem_readout.weight (via h_sem) AND through
        # sem_proj.weight (via qns), both of which are trained parameters.
        r_sem = self.sem_readout(h_sem) * qns   # (B, d_half)
        r_epi = self.epi_readout(h_epi) * qne  # (B, d_half)
        r = torch.cat([r_sem, r_epi], dim=-1)  # (B, d_model)

        # Null retrieval gate: suppress output when reservoir state is uninformative.
        if self.use_null_gate and self.null_gate is not None:
            g_null = torch.sigmoid(self.null_gate(q))   # (B, 1)
            r = g_null * r

        return self.norm_out(self.out_proj(r))   # (B, d_model) — bounded output
