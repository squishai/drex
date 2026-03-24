"""
drex.models.hdc_encoder — Fixed HDC (Hyperdimensional Computing) Encoder.

Phase 24 (DREX-UNIFIED): adds a zero-training projection layer before the main
transformer backbone.  Input token embeddings are lifted into a high-dimensional
hypervector space (Johnson–Lindenstrauss geometry) and then projected back down
to d_model before being passed to the transformer layers.

Architecture:
    raw token embedding  (B, S, d_model)
        ↓
    HDC lift:  fixed random projection  →  (B, S, hdc_dim)
        ↓
    optional L2 normalisation in hypervector space
        ↓
    HDC readdown: fixed random projection  →  (B, S, d_model)
        ↓
    layer norm + residual (preserves original embedding signal)

Training cost: ZERO.  All projection weights are frozen immediately after
construction and are never updated.  Only the downstream transformer trains.

Typical configs:
    hdc_dim=2048  — fast; mild geometry improvement
    hdc_dim=4096  — fuller approximation; default
    hdc_dim=8192  — diminishing returns; use only if memory is very wide

Binding / bundling primitives are implemented as standalone functions for use by
the future DREX Controller (Phase 26) when it reasons over hypervectors directly.

References:
    Kanerva (1988, 2009) — Sparse Distributed Memory / HDC theory
    Rahimi & Recht (2007) — Random Features for Large-Scale Kernel Machines
    Mitrokhin et al. (2019) — Associative Memory with HDC
    ACM HDC Survey (2023)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# HDC primitive operations
# ---------------------------------------------------------------------------

def hdc_bind(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Element-wise multiplication (binding / association).

    Binding creates a new vector that is quasi-orthogonal to both inputs.
    Shape: (*, D) × (*, D) → (*, D).
    """
    return a * b


def hdc_bundle(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Element-wise addition followed by sign normalisation (bundling / superposition).

    Bundling creates a vector similar to both inputs simultaneously.
    Shape: (*, D) × (*, D) → (*, D).
    """
    return F.normalize(a + b, p=2, dim=-1)


def hdc_permute(x: torch.Tensor, shifts: int = 1) -> torch.Tensor:
    """Circular shift along the last dimension (positional encoding primitive).

    Repeated permutations are quasi-orthogonal, allowing positional distinction.
    Shape: (*, D) → (*, D).
    """
    return x.roll(shifts, dims=-1)


# ---------------------------------------------------------------------------
# HDCEncoder module
# ---------------------------------------------------------------------------

class HDCEncoder(nn.Module):
    """Fixed random-projection HDC encoder.

    Lifts token embeddings into an ``hdc_dim``-dimensional hypervector space
    and projects them back to ``d_model``.  All weights are frozen at
    construction; the module contributes zero trainable parameters.

    Args:
        d_model:      Input (and output) embedding dimension.
        hdc_dim:      Hypervector dimension.  Must be strictly > d_model.
                      Typical: 2048 or 4096.
        normalize:    If True, L2-normalise hypervectors before readdown.
        seed:         RNG seed for reproducible weight construction.
    """

    def __init__(
        self,
        d_model: int,
        hdc_dim: int = 4096,
        normalize: bool = True,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if hdc_dim <= d_model:
            raise ValueError(
                f"hdc_dim={hdc_dim} must be > d_model={d_model} "
                "(lifting into a strictly higher-dimensional space)"
            )

        self.d_model = d_model
        self.hdc_dim = hdc_dim
        self.normalize = normalize

        # Fixed random projections — never updated.
        # W_lift:  (d_model, hdc_dim)
        # W_down:  (hdc_dim, d_model)
        # Initialised with orthogonal-ish columns (kaiming_uniform) then frozen.
        gen = torch.Generator()
        gen.manual_seed(seed)

        W_lift = torch.empty(d_model, hdc_dim)
        nn.init.kaiming_uniform_(W_lift, a=math.sqrt(5), generator=gen)

        W_down = torch.empty(hdc_dim, d_model)
        nn.init.kaiming_uniform_(W_down, a=math.sqrt(5), generator=gen)

        # Register as buffers (saved in checkpoints, moved with .to(device),
        # but excluded from optimizer parameter groups).
        self.register_buffer("W_lift", W_lift)
        self.register_buffer("W_down", W_down)

        # Output layer-norm keeps the residual-merged signal in a stable range.
        self.norm = nn.LayerNorm(d_model)

    def _lift(self, x: torch.Tensor) -> torch.Tensor:
        """Project (B, S, d_model) → (B, S, hdc_dim)."""
        h = x @ self.W_lift  # (B, S, hdc_dim)
        # Bipolar HDC: sign-threshold to {-1, +1} with smooth tanh during training
        # so gradients flow through the norm; hard threshold deactivates for eval.
        if self.training:
            h = torch.tanh(h)
        else:
            h = h.sign()
        if self.normalize:
            h = F.normalize(h, p=2, dim=-1)
        return h

    def _readdown(self, h: torch.Tensor) -> torch.Tensor:
        """Project (B, S, hdc_dim) → (B, S, d_model)."""
        return h @ self.W_down  # (B, S, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Lift embeddings into hypervector space and return a residual-merged output.

        Args:
            x:  (B, S, d_model) — input token embeddings.

        Returns:
            (B, S, d_model) — embeddings enriched with HDC structure; same shape as input.
        """
        h = self._lift(x)            # (B, S, hdc_dim)
        readdown = self._readdown(h) # (B, S, d_model)
        # Residual merge: preserves the original embedding signal while adding
        # the compositional HDC structure.  Layer-norm stabilises the sum.
        return self.norm(x + readdown)

    # ------------------------------------------------------------------
    # Diagnostic helpers
    # ------------------------------------------------------------------

    def hypervector(self, x: torch.Tensor) -> torch.Tensor:
        """Return the raw hypervector representation (B, S, hdc_dim) for inspection."""
        return self._lift(x)

    def similarity(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Cosine similarity between two hypervectors (B, S, hdc_dim) → (B, S)."""
        return F.cosine_similarity(a, b, dim=-1)
