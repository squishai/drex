"""drex.models.controller — DREX Routing Controller (Phase 26).

REINFORCE-based 2-layer MLP that routes context representations to memory
tiers.  All inputs are DETACHED — the controller learns via policy gradient
only, never via backpropagation through the main computation graph.

Architecture:
    ctx (B, d_input)
      → Linear(d_input, hidden) → GELU → LayerNorm
      → Linear(hidden, hidden)  → GELU → LayerNorm
      → write_head  (hidden → n_tiers)   → Bernoulli sample   → write_decisions
      → read_head   (hidden → n_tiers)   → softmax            → read_weights
      → gate_head   (hidden → n_modules) → Bernoulli sample   → sparse_gates
      log_probs = Bernoulli.log_prob(write_decisions) ++ Bernoulli.log_prob(sparse_gates)

Routing collapse contract (DREX_UNIFIED_SPEC.md §3.5):
    Over a sliding window of 100 steps, if any single tier receives >95% of
    write decisions:
        1. Log WARNING with tier index and percentage.
        2. Set self._collapse_penalty = True (caller can inject load-balance reward penalty).
        3. If collapse persists > 200 steps, raise RuntimeError.

Reward NaN contract:
    If any reward is NaN/Inf: skip REINFORCE update, log ERROR.
    If NaN persists for > 10 consecutive update() calls: raise RuntimeError.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)

# Fraction of writes to a single tier that triggers routing-collapse warning.
_COLLAPSE_THRESH: float = 0.95

# Sliding window length for routing health check.
_WINDOW: int = 100

# Maximum consecutive NaN rewards before hard stop.
_NAN_HALT: int = 10

# Maximum consecutive collapse steps before hard stop.
_COLLAPSE_HALT: int = 200


# ---------------------------------------------------------------------------
# Output container
# ---------------------------------------------------------------------------


@dataclass
class ControllerOutput:
    """Outputs from one DREXController.forward() call.

    write_decisions : (B, n_tiers) int32 {0, 1} — Bernoulli-sampled write gates.
    read_weights    : (B, n_tiers) float32 — softmax read-attention weights.
    sparse_gates    : (B, n_modules) bool   — module enable masks.
    log_probs       : (B, n_tiers + n_modules) float32 — log P(action) for REINFORCE.
    """

    write_decisions: torch.Tensor
    read_weights: torch.Tensor
    sparse_gates: torch.Tensor
    log_probs: torch.Tensor


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------


class DREXController(nn.Module):
    """REINFORCE-based routing controller for DREX-UNIFIED.

    Args:
        d_input:    Input dimension (size of context vector, typically d_model
                    or d_model + d_hdc after concatenation).
        n_tiers:    Number of memory tiers (default 3: L1 ESN, L2 Episodic, L3 Semantic).
        n_modules:  Number of downstream modules gated by sparse_gates (default 4).
        hidden_dim: Hidden dimension of the 2-layer MLP (default 128).
        gamma:      REINFORCE discount factor (default 0.99).
        lr:         Policy optimizer learning rate (default 1e-4).
    """

    def __init__(
        self,
        d_input: int,
        n_tiers: int = 3,
        n_modules: int = 4,
        hidden_dim: int = 128,
        gamma: float = 0.99,
        lr: float = 1e-4,
    ) -> None:
        super().__init__()

        self.d_input = d_input
        self.n_tiers = n_tiers
        self.n_modules = n_modules
        self.gamma = gamma

        # ── MLP backbone ──────────────────────────────────────────────────────
        self.fc1 = nn.Linear(d_input, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        # ── Policy heads ──────────────────────────────────────────────────────
        # write_head: log P(write_i = 1) per tier — Bernoulli policy.
        self.write_head = nn.Linear(hidden_dim, n_tiers)
        # read_head: unnormalised read-attention scores — softmax (deterministic).
        self.read_head = nn.Linear(hidden_dim, n_tiers)
        # gate_head: log P(gate_j = 1) per downstream module — Bernoulli policy.
        self.gate_head = nn.Linear(hidden_dim, n_modules)

        self._init_weights()

        # ── REINFORCE state ───────────────────────────────────────────────────
        self._optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self._lp_buf: list[torch.Tensor] = []
        self._nan_count: int = 0

        # ── Routing collapse detection ────────────────────────────────────────
        self._write_history: deque[torch.Tensor] = deque(maxlen=_WINDOW)
        self.collapse_penalty: bool = False   # readable by caller for reward injection
        self._collapse_tier: int = -1
        self._collapse_steps: int = 0

    def _init_weights(self) -> None:
        for m in (self.fc1, self.fc2):
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.zeros_(m.bias)
        # Zero-init policy heads → uniform distribution at start of training.
        for m in (self.write_head, self.read_head, self.gate_head):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, ctx: torch.Tensor) -> ControllerOutput:
        """Route a context vector to memory tiers.

        Args:
            ctx: (B, d_input) — MUST be detached before calling.

        Returns:
            ControllerOutput.  write_decisions and sparse_gates are sampled
            from Bernoulli distributions and are not differentiable.
            Gradient signal flows exclusively through log_probs via REINFORCE.
        """
        # Hard contract: no gradient from backbone into controller.
        # Explicit cast: input arrives as bfloat16 from Mamba; controller operates in float32.
        ctx = ctx.detach().float()

        # ── Feature extraction ────────────────────────────────────────────────
        h = F.gelu(self.norm1(self.fc1(ctx)))   # (B, hidden)
        h = F.gelu(self.norm2(self.fc2(h)))     # (B, hidden)

        # ── Write decisions ───────────────────────────────────────────────────
        write_probs = torch.sigmoid(self.write_head(h))           # (B, n_tiers)
        write_dist = torch.distributions.Bernoulli(probs=write_probs)
        write_samp = write_dist.sample()                           # (B, n_tiers)
        write_decisions = write_samp.to(torch.int32)
        write_lp = write_dist.log_prob(write_samp)                 # (B, n_tiers)

        # ── Read weights ──────────────────────────────────────────────────────
        read_weights = F.softmax(self.read_head(h), dim=-1)        # (B, n_tiers)

        # ── Sparse gates ──────────────────────────────────────────────────────
        gate_probs = torch.sigmoid(self.gate_head(h))              # (B, n_modules)
        gate_dist = torch.distributions.Bernoulli(probs=gate_probs)
        gate_samp = gate_dist.sample()                             # (B, n_modules)
        sparse_gates = gate_samp.bool()
        gate_lp = gate_dist.log_prob(gate_samp)                    # (B, n_modules)

        # Combined log-probs for REINFORCE.
        log_probs = torch.cat([write_lp, gate_lp], dim=-1)        # (B, n_tiers + n_modules)

        # ── Routing collapse detection ─────────────────────────────────────────
        with torch.no_grad():
            tier_means = write_decisions.float().mean(dim=0).cpu()  # (n_tiers,)
            self._write_history.append(tier_means)

            if len(self._write_history) == _WINDOW:
                window_avg = torch.stack(list(self._write_history)).mean(0)  # (n_tiers,)
                total = window_avg.sum().item()
                if total > 1e-6:
                    fracs = window_avg / total
                    worst = int(fracs.argmax().item())
                    worst_frac = fracs[worst].item()
                    if worst_frac > _COLLAPSE_THRESH:
                        self._collapse_steps += 1
                        log.warning(
                            "controller routing collapse detected — tier %d receiving %.1f%% of writes",
                            worst,
                            worst_frac * 100.0,
                        )
                        self.collapse_penalty = True
                        self._collapse_tier = worst
                        if self._collapse_steps > _COLLAPSE_HALT:
                            raise RuntimeError(
                                f"Controller routing collapse on tier {worst} persisted "
                                f"for {self._collapse_steps} steps. Halt training and "
                                "investigate controller reward signal."
                            )
                    else:
                        self.collapse_penalty = False
                        self._collapse_steps = 0

        return ControllerOutput(
            write_decisions=write_decisions,
            read_weights=read_weights,
            sparse_gates=sparse_gates,
            log_probs=log_probs,
        )

    # ── REINFORCE accumulate / update ─────────────────────────────────────────

    def store(self, log_probs: torch.Tensor) -> None:
        """Buffer log_probs from one forward call for the next REINFORCE update.

        Args:
            log_probs: (B, n_tiers + n_modules) — from ControllerOutput.log_probs.
        """
        self._lp_buf.append(log_probs)

    def update(self, rewards: list[float]) -> dict[str, float]:
        """Perform a REINFORCE policy-gradient update.

        Args:
            rewards: Per-step scalar rewards, length matching the stored log_probs.
                     Must align one-to-one with prior store() calls.

        Returns:
            {"controller_loss": float} on success.
            {"controller_loss": nan} if rewards contained NaN/Inf (update skipped).
            {} if the buffer was empty.

        Raises:
            RuntimeError if NaN rewards persist for more than _NAN_HALT updates.
        """
        if not self._lp_buf:
            return {}

        T = len(rewards)
        if T != len(self._lp_buf):
            log.warning(
                "update: rewards length %d ≠ buffered log_probs %d — clearing buffer",
                T,
                len(self._lp_buf),
            )
            self._lp_buf.clear()
            return {}

        # NaN/Inf reward guard (DREX_UNIFIED_SPEC.md reward loop NaN contract).
        for i, r in enumerate(rewards):
            if not math.isfinite(r):
                log.error("NaN/Inf reward at step %d — skipping REINFORCE update", i)
                self._nan_count += 1
                if self._nan_count > _NAN_HALT:
                    raise RuntimeError(
                        f"NaN/Inf reward persisted for {self._nan_count} consecutive "
                        "controller update() calls. Check upstream computation."
                    )
                self._lp_buf.clear()
                return {"controller_loss": float("nan")}

        self._nan_count = 0

        # Discounted returns G_t = r_t + γ·r_{t+1} + γ²·r_{t+2} + …
        returns: list[float] = []
        G = 0.0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)

        returns_t = torch.tensor(returns, dtype=torch.float32)
        # Variance-reduction baseline: standardise.
        if returns_t.numel() > 1 and returns_t.std() > 1e-8:
            returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

        # REINFORCE loss: -E[log π(a|s) · G_t], averaged over steps, batch, and actions.
        total_loss = torch.zeros(1, requires_grad=False)
        for lp, G_t in zip(self._lp_buf, returns_t.tolist()):
            # lp: (B, n_actions); G_t: scalar.
            # Mean over B and actions, then scale by return.
            total_loss = total_loss + (-lp.mean() * G_t)

        loss = total_loss / T

        self._optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        self._optimizer.step()

        self._lp_buf.clear()
        return {"controller_loss": loss.item()}
