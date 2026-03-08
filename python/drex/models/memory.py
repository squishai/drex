"""
drex.models.memory — L2 and L3 memory components.

L2: Infini-Attention matrix memory (per-head M and z vectors).
L3: Titans-style MLP weight snapshots; L3MemoryBridge coordinates with the Rust layer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# L2 state containers
# ---------------------------------------------------------------------------


@dataclass
class MemoryState:
    """
    Per-layer Infini-Attention matrix memory.

    Shapes (all float32):
        M  : (B, H, d_k, d_v)  — memory matrix
        z  : (B, H, d_k)       — normalisation accumulator
    """

    M: torch.Tensor  # (B, H, d_k, d_v)
    z: torch.Tensor  # (B, H, d_k)

    @staticmethod
    def zeros(batch: int, n_heads: int, d_k: int, d_v: int, device: torch.device) -> "MemoryState":
        return MemoryState(
            M=torch.zeros(batch, n_heads, d_k, d_v, device=device, dtype=torch.float32),
            z=torch.zeros(batch, n_heads, d_k, device=device, dtype=torch.float32),
        )

    def detach(self) -> "MemoryState":
        """Return a copy with gradients detached (for TBPTT segment boundaries)."""
        return MemoryState(M=self.M.detach(), z=self.z.detach())

    def to(self, device: torch.device) -> "MemoryState":
        return MemoryState(M=self.M.to(device), z=self.z.to(device))


@dataclass
class LayerState:
    """All recurrent state for one DrexLayer."""

    memory: MemoryState  # L2 Infini-Attention state
    step: int = 0        # global training step (used for L3 snapshot keys)

    @staticmethod
    def zeros(batch: int, n_heads: int, d_k: int, d_v: int, device: torch.device) -> "LayerState":
        return LayerState(
            memory=MemoryState.zeros(batch, n_heads, d_k, d_v, device),
            step=0,
        )

    def detach(self) -> "LayerState":
        return LayerState(memory=self.memory.detach(), step=self.step)

    def to(self, device: torch.device) -> "LayerState":
        return LayerState(memory=self.memory.to(device), step=self.step)


# ---------------------------------------------------------------------------
# L2 delta-rule write
# ---------------------------------------------------------------------------


def _elu1(x: torch.Tensor) -> torch.Tensor:
    """ELU + 1 feature map — positive-valued, used as φ in linear attention."""
    return F.elu(x) + 1.0


class DeltaRuleUpdate(nn.Module):
    """
    Stateless module implementing one Infini-Attention delta-rule write step.

    For each token position the update is:
        V_existing = φ(K) @ M          (B, H, S, d_v)
        M  +=  φ(K)ᵀ @ (V - V_existing)
        z  +=  sum(φ(K), dim=S)

    This is the "associative delta rule" from:
        Munkhdalai et al., "Leave No Context Behind", Google 2024.
    """

    def forward(
        self,
        K: torch.Tensor,      # (B, H, S, d_k)
        V: torch.Tensor,      # (B, H, S, d_v)
        state: MemoryState,
    ) -> MemoryState:
        phi_K = _elu1(K)      # (B, H, S, d_k)

        # V_existing = φ(K) M — what memory currently "says" for these keys
        # (B, H, S, d_k) @ (B, H, d_k, d_v) → (B, H, S, d_v)
        V_existing = torch.matmul(phi_K, state.M)

        # Associative delta update: ΔM = φ(K)ᵀ @ (V - V_existing)
        # (B, H, d_k, S) @ (B, H, S, d_v) → (B, H, d_k, d_v)
        delta_M = torch.matmul(phi_K.transpose(-2, -1), V - V_existing)

        new_M = state.M + delta_M
        new_z = state.z + phi_K.sum(dim=-2)  # (B, H, d_k)

        return MemoryState(M=new_M, z=new_z)


# ---------------------------------------------------------------------------
# L3 Titan MLP memory
# ---------------------------------------------------------------------------


class TitanMemory(nn.Module):
    """
    Titans-style memory MLP (Behrouz et al., Google Jan 2025).

    Memory is encoded as the weights of a small 2-layer MLP with no
    layer normalisation. Writing to memory = one gradient descent step
    on a surprise loss: ‖net(k) − v‖².

    The MLP has its own internal Adam optimiser so gradient steps are
    independent of the outer training optimiser.
    """

    def __init__(self, d_model: int, d_hidden: int, lr: float = 1e-3) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_hidden = d_hidden

        # Two-layer MLP, no bias, no normalisation
        self.fc1 = nn.Linear(d_model, d_hidden, bias=False)
        self.fc2 = nn.Linear(d_hidden, d_model, bias=False)

        # Internal optimiser — updated only during write(), not outer backward
        self._optim = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Read from memory: x → hidden → output."""
        return self.fc2(F.gelu(self.fc1(x)))

    def write(self, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """
        One gradient step on surprise loss: ‖net(key) − value‖².

        key, value: (B, d_model) — mean-pooled or representative token vecs.
        Returns the loss scalar (detached) for logging.
        """
        self._optim.zero_grad()
        pred = self(key)
        loss = F.mse_loss(pred, value.detach())
        loss.backward()
        self._optim.step()
        return loss.detach()

    def snapshot_weights(self) -> list[float]:
        """Return all parameters as a flat list of Python floats (f32)."""
        parts: list[torch.Tensor] = []
        for p in self.parameters():
            parts.append(p.detach().cpu().float().flatten())
        return torch.cat(parts).tolist()

    def load_weights(self, weights: list[float]) -> None:
        """Load weights from a flat list (must match weight_vector_size())."""
        buf = torch.tensor(weights, dtype=torch.float32)
        offset = 0
        with torch.no_grad():
            for p in self.parameters():
                n = p.numel()
                p.copy_(buf[offset : offset + n].view(p.shape))
                offset += n

    def weight_vector_size(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# L3 bridge — Python coordinator between TitanMemory and Rust _sys
# ---------------------------------------------------------------------------


class L3MemoryBridge:
    """
    Coordinates L3 (disk) memory I/O between Python TitanMemory and Rust _sys.

    Requires drex._RUST_AVAILABLE = True and a SnapshotStore + PrefetchEngine.
    When _RUST_AVAILABLE is False the bridge is a no-op, allowing the model to
    run with only L1+L2 memory.
    """

    def __init__(
        self,
        titan_memories: list[TitanMemory],  # one per layer
        base_path: str = "/tmp/drex_l3",
        compress: bool = False,
        max_prefetch_cache: int = 64,
        sketch_rank: int = 16,
    ) -> None:
        self._titans = titan_memories
        self._prefetch_hits = 0
        self._prefetch_calls = 0

        try:
            import drex._sys as _sys

            self._store = _sys.SnapshotStore(base_path, compress)
            self._engine = _sys.PrefetchEngine(
                self._store,
                d_model=titan_memories[0].d_model if titan_memories else 1,
                max_cache_entries=max_prefetch_cache,
                sketch_rank=sketch_rank,
            )
            self._available = True
        except ImportError:
            self._store = None
            self._engine = None
            self._available = False

    # ------------------------------------------------------------------

    def write_and_snapshot(
        self,
        layer: int,
        head: int,
        step: int,
        key_vec: torch.Tensor,  # (d_model,) representative key
        value_vec: torch.Tensor,  # (d_model,) representative value
    ) -> None:
        """Write key/value to TitanMemory then snapshot weights to Rust store."""
        titan = self._titans[layer]
        titan.write(key_vec.unsqueeze(0).float(), value_vec.unsqueeze(0).float())

        if not self._available:
            return

        weights = titan.snapshot_weights()
        self._store.write(layer, head, step, weights)
        # Register with prefetch engine for future similarity lookups
        self._engine.register_snapshot(layer, head, step, key_vec.float().tolist())

    def retrieve_and_load(self, layer: int, head: int, step: int) -> bool:
        """
        Try prefetch cache first, then disk. Loads weights into TitanMemory.
        Returns True on hit, False on miss.
        """
        if not self._available:
            return False

        self._prefetch_calls += 1

        # Check prefetch cache first
        weights = self._engine.consume_prefetched(layer, head, step)
        if weights is not None:
            self._prefetch_hits += 1
            self._titans[layer].load_weights(weights)
            return True

        # Fall back to synchronous disk read
        if self._store.exists(layer, head, step):
            weights = self._store.read(layer, head, step)
            self._titans[layer].load_weights(weights)
            return True

        return False

    def trigger_prefetch(self, layer: int, query_vec: torch.Tensor, k: int = 4) -> None:
        """Fire async prefetch using the sketch index. Non-blocking."""
        if not self._available:
            return
        self._engine.prefetch(layer, query_vec.float().tolist(), k)

    @property
    def prefetch_hit_rate(self) -> float:
        if self._prefetch_calls == 0:
            return 0.0
        return self._prefetch_hits / self._prefetch_calls
