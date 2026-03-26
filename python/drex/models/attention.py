"""
drex.models.attention — L1 and L2 attention modules.

L1: SlidingWindowAttention — causal SDPA within a fixed window.
L2: InfiniAttention — delta-rule write to matrix memory + linear attention read.
Hybrid: HybridAttention — L1 + L2 combined with a sigmoid gate β.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from drex.models.memory import DeltaRuleUpdate, MemoryState, _elu1


class SlidingWindowAttention(nn.Module):
    """
    L1 sliding-window causal multi-head attention.

    Attends only to the most recent `window_size` tokens using
    PyTorch's scaled_dot_product_attention with a causal mask.
    """

    def __init__(self, d_model: int, n_heads: int, window_size: int = 2048) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.window_size = window_size

        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, S, d_model) — assume S <= window_size (caller is responsible
           for chunking longer sequences into segments).
        Returns: (B, S, d_model)
        """
        B, S, D = x.shape
        QKV = self.qkv_proj(x)  # (B, S, 3*D)
        Q, K, V = QKV.split(D, dim=-1)

        # Reshape to (B, H, S, d_k)
        def _split_heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(B, S, self.n_heads, self.d_k).transpose(1, 2)

        Q, K, V = _split_heads(Q), _split_heads(K), _split_heads(V)

        # Causal attention with window mask
        # For simplicity: standard causal for S <= window_size.
        # scaled_dot_product_attention handles the causal mask via is_causal=True.
        attn_out = F.scaled_dot_product_attention(Q, K, V, is_causal=True)

        # Merge heads: (B, H, S, d_k) → (B, S, D)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, D)
        return self.out_proj(attn_out)


class InfiniAttention(nn.Module):
    """
    L2 Infini-Attention (Munkhdalai et al., Google 2024).

    At each segment:
      1. Project x → Q, K, V
      2. Read from L2 memory: A_mem = φ(Q) M / (φ(Q) z + ε)
      3. Write new KVs to L2 memory via delta rule
      4. Gate: output = σ(β) A_mem + (1 − σ(β)) A_sdpa

    The gate β is a learnable scalar per head.
    """

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.d_v = d_model // n_heads

        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # Learnable gate — one scalar per head, broadcast over (B, S)
        self.beta = nn.Parameter(torch.zeros(n_heads))

        self._delta_update = DeltaRuleUpdate()

    def forward(
        self,
        x: torch.Tensor,     # (B, S, d_model)
        state: MemoryState,  # in-place updated
    ) -> tuple[torch.Tensor, MemoryState]:
        """
        Returns (output, new_state).
        output: (B, S, d_model)
        """
        B, S, D = x.shape
        H, d_k, d_v = self.n_heads, self.d_k, self.d_v

        QKV = self.qkv_proj(x)
        Q, K, V = QKV.split(D, dim=-1)

        def _split_heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(B, S, H, d_k).transpose(1, 2)  # (B, H, S, d_k)

        Q, K, V = _split_heads(Q), _split_heads(K), _split_heads(V)

        # ----- L2 memory read -----
        phi_Q = _elu1(Q)  # (B, H, S, d_k)
        # A_mem = φ(Q) M / (φ(Q) z + ε)
        # state.M/z are float32 (DREX dtype contract: episodic state is float32).
        # Cast to phi_Q.dtype for mixed-precision compatibility when Mamba is bfloat16.
        # (B, H, S, d_k) @ (B, H, d_k, d_v) → (B, H, S, d_v)
        A_mem = torch.matmul(phi_Q, state.M.to(phi_Q.dtype))
        # denom: (B, H, S, 1) — prevent divide by zero
        denom = (phi_Q * state.z.to(phi_Q.dtype).unsqueeze(-2)).sum(dim=-1, keepdim=True).clamp(min=1e-6)
        A_mem = A_mem / denom

        # ----- L2 memory write (delta rule) -----
        new_state = self._delta_update(K, V, state)

        # ----- Local dot-product attention -----
        A_sdpa = F.scaled_dot_product_attention(Q, K, V, is_causal=True)

        # ----- Sigmoid gate β per head -----
        # beta: (H,) → (1, H, 1, 1)
        gate = torch.sigmoid(self.beta).view(1, H, 1, 1)
        A_combined = gate * A_mem + (1.0 - gate) * A_sdpa  # (B, H, S, d_v)

        # Merge heads: (B, H, S, d_v) → (B, S, D)
        out = A_combined.transpose(1, 2).contiguous().view(B, S, D)
        return self.out_proj(out), new_state


class HybridAttention(nn.Module):
    """
    Combines SlidingWindowAttention (L1) with InfiniAttention (L2).

    Both branches share the same QKV projection to avoid doubling parameters.
    The InfiniAttention gate β determines the L1/L2 split.
    """

    def __init__(self, d_model: int, n_heads: int, window_size: int = 2048, use_l2: bool = True) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.window_size = window_size
        self._use_l2 = use_l2

        # Shared projections
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # Gate: per-head learnable scalar
        self.beta = nn.Parameter(torch.zeros(n_heads))

        self._delta_update = DeltaRuleUpdate()

    def forward(
        self,
        x: torch.Tensor,     # (B, S, d_model)
        state: MemoryState,
    ) -> tuple[torch.Tensor, MemoryState]:
        B, S, D = x.shape
        H, d_k = self.n_heads, self.d_k

        QKV = self.qkv_proj(x)
        Q, K, V = QKV.split(D, dim=-1)

        def _split_heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(B, S, H, d_k).transpose(1, 2)

        Q, K, V = _split_heads(Q), _split_heads(K), _split_heads(V)

        # L1: sliding window causal attention over the current segment
        A_local = F.scaled_dot_product_attention(Q, K, V, is_causal=True)

        if self._use_l2:
            # L2: read from memory
            phi_Q = _elu1(Q)
            A_mem = torch.matmul(phi_Q, state.M)
            denom = (phi_Q * state.z.unsqueeze(-2)).sum(dim=-1, keepdim=True).clamp(min=1e-6)
            A_mem = A_mem / denom

            # L2: write to memory (delta rule)
            new_state = self._delta_update(K, V, state)

            # Gate and combine
            gate = torch.sigmoid(self.beta).view(1, H, 1, 1)
            A_combined = gate * A_mem + (1.0 - gate) * A_local
        else:
            # L2 disabled: use only L1 sliding-window attention; pass state unchanged
            A_combined = A_local
            new_state = state

        out = A_combined.transpose(1, 2).contiguous().view(B, S, D)
        return self.out_proj(out), new_state
