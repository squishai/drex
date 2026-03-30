"""DREXController — DREX-UNIFIED COMPONENT 4, OBJECTIVE 3.

Discrete RL routing policy that decides what to write to each memory tier,
what to read, and which sparse execution paths to activate.

Architecture: 2-layer MLP (~50K params), CPU-trainable.
Training: REINFORCE with EMA baseline subtraction for variance reduction.

Dtype contract (DREX_UNIFIED_SPEC.md §dtype):
    Input:           bfloat16  (asserted in forward)
    Internal:        float32
    write_decisions: int32     (one-hot of selected tier per position)
    read_weights:    float32   (softmax over tier logits)
    sparse_gates:    bool      (top-k=1 gate, identical structure to write_decisions)

NaN guard: >10 consecutive NaN/Inf rewards → RuntimeError halts training.

Routing collapse contract (DREX_UNIFIED_SPEC.md §controller):
    If any single tier receives >95% of writes over a sliding window of
    100 consecutive update() calls:
        - WARNING logged: "controller routing collapse detected — tier {i} receiving {pct:.1f}% of writes"
        - 0.1 load-balance penalty subtracted from reward before REINFORCE update.

Ref: DREX_UNIFIED_SPEC.md § COMPONENT 4: DREX CONTROLLER
"""
from __future__ import annotations

import logging
from collections import deque
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

logger = logging.getLogger(__name__)

_NAN_HALT_THRESHOLD: int = 10


class DREXController(nn.Module):
    """REINFORCE routing policy (2-layer MLP, ~50K parameters, CPU-trainable).

    Args:
        d_model:    Input hidden dimension (must match Mamba backbone d_model).
        n_tiers:    Number of memory routing tiers. Default 3.
        hidden_dim: MLP hidden layer width. Default 128.
        lr:         Adam optimizer learning rate. Default 1e-4.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_tiers: int = 3,
        hidden_dim: int = 128,
        lr: float = 1e-4,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_tiers = n_tiers

        # 2-layer MLP: all weights stay in float32 (policy operates in float32)
        self.policy = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_tiers),
        )
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

        # REINFORCE state — populated in forward(), consumed in update()
        self._log_prob: Optional[torch.Tensor] = None
        self._last_actions: Optional[torch.Tensor] = None  # (B, S) long

        # NaN guard: count consecutive bad rewards
        self._consecutive_nan: int = 0

        # Routing collapse window: last 100 update() calls, each recorded as
        # the modal tier selected in that call.
        self._routing_history: deque[int] = deque(maxlen=100)

        # REINFORCE baseline (EMA of rewards — reduces variance)
        self._reward_baseline: float = 0.0

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Route each (batch, sequence) position to a memory tier.

        In training mode, samples from Categorical(logits) and stores log_prob
        for the REINFORCE update.  In eval mode, uses argmax (deterministic).

        Args:
            x: (B, S, d_model) bfloat16 context tensor.

        Returns:
            write_decisions: (B, S, n_tiers) int32 — one-hot of selected tier.
            read_weights:    (B, S, n_tiers) float32 — softmax over tiers.
            sparse_gates:    (B, S, n_tiers) bool — top-k=1 gate.

        Raises:
            AssertionError: if ``x.dtype != torch.bfloat16``.
        """
        assert x.dtype == torch.bfloat16, (
            f"DREXController input must be bfloat16, got {x.dtype}"
        )
        B, S, _ = x.shape
        x_f32 = x.float()  # float32 internal computation

        logits = self.policy(x_f32)  # (B, S, n_tiers) float32

        flat_logits = logits.view(-1, self.n_tiers)  # (B*S, n_tiers)

        if self.training:
            dist = Categorical(logits=flat_logits)
            actions = dist.sample()  # (B*S,) — sampled tier indices
            self._log_prob = dist.log_prob(actions).sum()  # scalar with grad_fn
        else:
            actions = flat_logits.argmax(dim=-1)  # (B*S,) — deterministic
            self._log_prob = None

        actions = actions.view(B, S)  # (B, S)
        self._last_actions = actions.detach().clone()

        # one-hot of selected tier → int32: shape (B, S, n_tiers)
        write_decisions = F.one_hot(actions.long(), self.n_tiers).to(torch.int32)
        # softmax over tier logits → float32
        read_weights = torch.softmax(logits, dim=-1).float()
        # top-k=1 gate == one-hot → bool
        sparse_gates = write_decisions.bool()

        return write_decisions, read_weights, sparse_gates

    # ------------------------------------------------------------------
    # REINFORCE update
    # ------------------------------------------------------------------

    def update(self, reward: float) -> None:
        """REINFORCE weight update with NaN guard and collapse detection.

        Args:
            reward: Scalar reward signal.  NaN/Inf is caught and handled.

        Raises:
            RuntimeError: if NaN/Inf reward has occurred more than
                ``_NAN_HALT_THRESHOLD`` consecutive times.
        """
        reward_t = torch.tensor(float(reward), dtype=torch.float32)

        # ---- NaN / Inf guard -------------------------------------------
        if torch.isnan(reward_t) or torch.isinf(reward_t):
            self._consecutive_nan += 1
            logger.warning(
                "NaN/Inf reward at consecutive step %d — skipping update",
                self._consecutive_nan,
            )
            if self._consecutive_nan > _NAN_HALT_THRESHOLD:
                raise RuntimeError(
                    f"NaN/Inf reward for >{_NAN_HALT_THRESHOLD} consecutive steps"
                )
            return

        self._consecutive_nan = 0

        # ---- Routing collapse detection ---------------------------------
        if self._last_actions is not None:
            tiers = self._last_actions.flatten().tolist()
            # Summarise this update call as the most-written-to tier.
            modal_tier = max(range(self.n_tiers), key=lambda t: tiers.count(t))
            self._routing_history.append(modal_tier)

        if len(self._routing_history) == self._routing_history.maxlen:
            for t in range(self.n_tiers):
                count = sum(1 for r in self._routing_history if r == t)
                pct = count / len(self._routing_history) * 100.0
                if pct > 95.0:
                    logger.warning(
                        "controller routing collapse detected — tier %d receiving %.1f%% of writes",
                        t,
                        pct,
                    )
                    reward_t = reward_t - 0.1  # load-balance penalty

        # ---- REINFORCE update -------------------------------------------
        if self._log_prob is not None:
            # EMA baseline subtraction (reduces gradient variance)
            self._reward_baseline = (
                0.9 * self._reward_baseline + 0.1 * reward_t.item()
            )
            advantage = reward_t.item() - self._reward_baseline
            loss = -self._log_prob * advantage
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self._log_prob = None
