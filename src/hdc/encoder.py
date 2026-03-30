"""HDC Token Encoder — DREX-UNIFIED COMPONENT 2, OBJECTIVE 0.

Projects integer token IDs into a high-dimensional hypervector space using
three classical HDC operations:
  - Binding   (element-wise multiplication)  — encodes associations
  - Bundling  (element-wise sum + sign norm) — encodes composition
  - Permutation (cyclic roll by position)    — encodes sequence order

All matrices are fixed at initialization via seed=42 and are NEVER trained.
Training cost: zero.

Interface (per DREX_UNIFIED_SPEC.md v0.2 § COMPONENT 2):
  Input:  (B, S) int32  token IDs
  Output: (B, S, D_hdc) float32  hypervectors

Dtype contract:
  Output is always float32.  The Mamba input projection is the ONLY place
  where a float32 → bfloat16 cast is permitted (not here).

Ref: DREX_UNIFIED_SPEC.md § COMPONENT 2: HDC ENCODER — OBJECTIVE 0, PHASE 1
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class HDCTokenEncoder(nn.Module):
    """Token-ID HDC encoder.

    Maps discrete token IDs to compositional hypervectors using item memory
    and positional cyclic-roll permutations.

    Args:
        d_hdc:      Hypervector dimension.  Default 10000.
                    The orthogonality contract (mean cosine sim < 0.05) holds
                    reliably at ≥1024 for VOCAB_SIZE ≤ 32768.
                    Minimum validated D_hdc is determined empirically during
                    Phase 1 Wave 1 and recorded in the commit message.
        vocab_size: Number of distinct token IDs.  Default 256 (byte-mode).
        normalize:  L2-normalize each output hypervector.  Default True.
                    Required for cosine-similarity contracts to hold.
        seed:       RNG seed for item_memory and permutation tables.
                    Default 42 — never change without re-running Phase 1 gate.
    """

    def __init__(
        self,
        d_hdc: int = 10_000,
        vocab_size: int = 256,
        normalize: bool = True,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.d_hdc = d_hdc
        self.vocab_size = vocab_size
        self.normalize = normalize
        self.seed = seed

        # item_memory: (vocab_size, D_hdc) — one bipolar random hypervector per token.
        # Stored as a non-parameter buffer (never trained, survives state_dict).
        gen = torch.Generator()
        gen.manual_seed(seed)
        item_mem = torch.bernoulli(
            torch.full((vocab_size, d_hdc), 0.5), generator=gen
        ).float() * 2.0 - 1.0  # bipolar {-1, +1}
        self.register_buffer("item_memory", item_mem)  # (vocab_size, D_hdc) float32

    # ------------------------------------------------------------------
    # HDC primitive operations (module-level, no state)
    # ------------------------------------------------------------------

    @staticmethod
    def bind(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Element-wise multiplication — binding / association.

        Result is quasi-orthogonal to both inputs.
        Shape: (*, D) × (*, D) → (*, D).
        """
        return a * b

    @staticmethod
    def bundle(hvs: torch.Tensor) -> torch.Tensor:
        """Element-wise addition over a stack, then sign normalization — bundling.

        Args:
            hvs: (..., N, D) — N hypervectors to bundle.
        Returns:
            (..., D) — superposition of all N vectors, dtype float32.
        """
        summed = hvs.sum(dim=-2)  # (..., D)
        return summed.sign().float()

    @staticmethod
    def permute(hv: torch.Tensor, n: int) -> torch.Tensor:
        """Cyclic roll by n positions along the last dimension.

        Distinct n values produce quasi-orthogonal permuted copies,
        enabling positional encoding without learned parameters.

        Args:
            hv: (*, D)
            n:  number of positions to roll
        Returns:
            (*, D) — same dtype as input
        """
        return hv.roll(n, dims=-1)

    # ------------------------------------------------------------------
    # Encoding API
    # ------------------------------------------------------------------

    def encode_token(self, token_id: torch.Tensor) -> torch.Tensor:
        """Look up item memory for one or more token IDs.

        Args:
            token_id: integer tensor of arbitrary shape, dtype int32 or int64.
        Returns:
            float32 tensor of shape (*token_id.shape, D_hdc).
        """
        return self.item_memory[token_id.long()]  # (*, D_hdc) float32

    def encode_sequence(self, tokens: torch.Tensor) -> torch.Tensor:
        """Encode a single sequence of token IDs with positional HDC.

        Bundles position-permuted token hypervectors:
            h_i = permute(item_memory[t_i], i)
            output = bundle([h_0, h_1, ..., h_{S-1}])

        Args:
            tokens: (S,) int32 token IDs.
        Returns:
            (D_hdc,) float32 — sequence hypervector.
        """
        S = tokens.shape[0]
        hvs = []
        for i in range(S):
            hv = self.encode_token(tokens[i])  # (D_hdc,)
            hvs.append(self.permute(hv, i))
        stacked = torch.stack(hvs, dim=0)  # (S, D_hdc)
        return self.bundle(stacked)         # (D_hdc,)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Encode a batch of token-ID sequences into hypervectors.

        Args:
            token_ids: (B, S) int32 — batch of token ID sequences.

        Returns:
            (B, S, D_hdc) float32 — per-token positional hypervectors.
            Each position i in the output is permute(item_memory[t_i], i),
            optionally L2-normalized.

        Note:
            This returns per-token (not per-sequence) hypervectors so that
            downstream components (Mamba SSM) can process the sequence
            step-by-step.  For a full sequence-level compositional vector,
            use encode_sequence().
        """
        assert token_ids.dtype in (torch.int32, torch.int64), (
            f"HDCTokenEncoder expects int32/int64 token IDs, got {token_ids.dtype}"
        )
        B, S = token_ids.shape
        # Look up: (B, S, D_hdc)
        hvs = self.item_memory[token_ids.long()]  # float32

        # Apply positional cyclic-roll permutation to each position.
        # roll by position i along the last dimension.
        # Vectorized: build position offsets and roll in one pass.
        out = hvs.clone()
        for i in range(S):
            out[:, i, :] = torch.roll(hvs[:, i, :], shifts=i, dims=-1)

        if self.normalize:
            out = F.normalize(out, p=2, dim=-1)

        return out  # (B, S, D_hdc) float32
