"""
drex.models.transformer — DrexConfig, DrexLayer, DrexTransformer.

Each layer has HybridAttention (L1+L2) + FeedForward.
Model carries a list of LayerState across segment boundaries.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from drex.models.attention import HybridAttention
from drex.models.memory import LayerState


@dataclass
class DrexConfig:
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 4
    window_size: int = 2048
    ff_mult: int = 4           # feed-forward hidden = d_model * ff_mult
    vocab_size: int = 32_000
    max_seq_len: int = 4096
    dropout: float = 0.0
    use_l3: bool = False       # enable L3 disk cache (requires Rust extension)
    l3_base_path: str = "/tmp/drex_l3"
    l3_compress: bool = False


class FeedForward(nn.Module):
    def __init__(self, d_model: int, ff_mult: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        d_ff = d_model * ff_mult
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model, bias=False),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DrexLayer(nn.Module):
    def __init__(self, config: DrexConfig) -> None:
        super().__init__()
        self.attn = HybridAttention(
            d_model=config.d_model,
            n_heads=config.n_heads,
            window_size=config.window_size,
        )
        self.ff = FeedForward(config.d_model, config.ff_mult, config.dropout)
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)

    def forward(
        self,
        x: torch.Tensor,       # (B, S, d_model)
        layer_state: LayerState,
    ) -> tuple[torch.Tensor, LayerState]:
        # Pre-norm + residual for attention
        normed = self.norm1(x)
        attn_out, new_memory = self.attn(normed, layer_state.memory)
        x = x + attn_out

        # Pre-norm + residual for feed-forward
        x = x + self.ff(self.norm2(x))

        new_state = LayerState(memory=new_memory, step=layer_state.step + 1)
        return x, new_state


class DrexTransformer(nn.Module):
    """
    Full Drex model.

    For training over long sequences, call forward() on each segment and
    thread states through. Use state.detach() at segment boundaries for TBPTT.
    """

    def __init__(self, config: DrexConfig) -> None:
        super().__init__()
        self.config = config

        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)
        self.drop = nn.Dropout(config.dropout)

        self.layers = nn.ModuleList([DrexLayer(config) for _ in range(config.n_layers)])
        self.norm_out = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying: share token embedding and LM head weights
        self.lm_head.weight = self.token_emb.weight

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
            elif isinstance(m, nn.Embedding):
                nn.init.trunc_normal_(m.weight, std=0.02)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def init_states(self, batch: int, device: torch.device) -> list[LayerState]:
        """Create fresh zero states for all layers."""
        cfg = self.config
        d_k = cfg.d_model // cfg.n_heads
        return [
            LayerState.zeros(batch, cfg.n_heads, d_k, d_k, device)
            for _ in range(cfg.n_layers)
        ]

    def forward(
        self,
        input_ids: torch.Tensor,         # (B, S)
        states: Optional[list[LayerState]] = None,
    ) -> tuple[torch.Tensor, list[LayerState]]:
        """
        Returns:
            logits: (B, S, vocab_size)
            new_states: list of LayerState, one per layer
        """
        B, S = input_ids.shape
        device = input_ids.device

        if states is None:
            states = self.init_states(B, device)

        # Positions: 0..S-1 relative to segment start
        pos = torch.arange(S, device=device).unsqueeze(0)  # (1, S)

        x = self.drop(self.token_emb(input_ids) + self.pos_emb(pos))

        new_states: list[LayerState] = []
        for layer, state in zip(self.layers, states):
            x, new_state = layer(x, state)
            new_states.append(new_state)

        logits = self.lm_head(self.norm_out(x))  # (B, S, vocab_size)
        return logits, new_states
