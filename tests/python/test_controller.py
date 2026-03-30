"""Wave 5 — DREXController + RewardSignal tests.

Validates DREXController and RewardSignal against the contracts defined in
DREX_UNIFIED_SPEC.md § COMPONENT 4, using the REAL pipeline (no mocking).

Tests:
    TestDREXControllerShapeAndDtype  — shape, dtype, bfloat16-only guard,
                                       softmax sums, NaN/Inf check
    TestControllerNaNGuard           — NaN reward counter + RuntimeError halt
    TestControllerRoutingCollapse    — collapse WARNING logged / not logged
    TestControllerLearning           — REINFORCE accuracy > 50% on synthetic task
"""
from __future__ import annotations

import logging

import pytest
import torch

from controller.policy import DREXController
from controller.reward import RewardSignal
from tests.python.conftest import assert_no_nan_inf

# ---------------------------------------------------------------------------
# Module-level constants (mirror conftest dims)
# ---------------------------------------------------------------------------
B = 2
S = 16
D_MODEL = 256
N_TIERS = 3


# ===========================================================================
# 1. Shape and dtype contracts
# ===========================================================================

class TestDREXControllerShapeAndDtype:

    def _make_ctrl(self) -> DREXController:
        ctrl = DREXController(d_model=D_MODEL, n_tiers=N_TIERS)
        ctrl.train()
        return ctrl

    def _x(self) -> torch.Tensor:
        return torch.randn(B, S, D_MODEL, dtype=torch.bfloat16)

    # ---- write_decisions ---------------------------------------------------

    def test_write_decisions_shape_and_dtype(self):
        ctrl = self._make_ctrl()
        wd, _, _ = ctrl(self._x())
        assert wd.shape == (B, S, N_TIERS), f"Expected {(B, S, N_TIERS)}, got {wd.shape}"
        assert wd.dtype == torch.int32, f"Expected int32, got {wd.dtype}"

    # ---- read_weights -------------------------------------------------------

    def test_read_weights_shape_dtype_and_sums_to_one(self):
        ctrl = self._make_ctrl()
        _, rw, _ = ctrl(self._x())
        assert rw.shape == (B, S, N_TIERS), f"Expected {(B, S, N_TIERS)}, got {rw.shape}"
        assert rw.dtype == torch.float32, f"Expected float32, got {rw.dtype}"
        sums = rw.sum(dim=-1)  # (B, S)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5), (
            f"read_weights should sum to 1 per position, got range "
            f"[{sums.min():.6f}, {sums.max():.6f}]"
        )

    # ---- sparse_gates -------------------------------------------------------

    def test_sparse_gates_shape_dtype_and_exactly_one_true(self):
        ctrl = self._make_ctrl()
        _, _, sg = ctrl(self._x())
        assert sg.shape == (B, S, N_TIERS), f"Expected {(B, S, N_TIERS)}, got {sg.shape}"
        assert sg.dtype == torch.bool, f"Expected bool, got {sg.dtype}"
        # Exactly one True per (batch, sequence) position
        n_true = sg.long().sum(dim=-1)  # (B, S)
        assert (n_true == 1).all(), (
            f"Each position should have exactly 1 True gate, "
            f"got counts: {n_true.unique().tolist()}"
        )

    # ---- dtype guard --------------------------------------------------------

    def test_float32_input_raises(self):
        ctrl = self._make_ctrl()
        x_wrong_dtype = torch.randn(B, S, D_MODEL, dtype=torch.float32)
        with pytest.raises(AssertionError):
            ctrl(x_wrong_dtype)

    # ---- no NaN/Inf in read_weights -----------------------------------------

    def test_no_nan_inf_in_read_weights(self):
        ctrl = self._make_ctrl()
        _, rw, _ = ctrl(self._x())
        assert_no_nan_inf(rw, "read_weights")


# ===========================================================================
# 2. NaN guard
# ===========================================================================

class TestControllerNaNGuard:

    def _make_ctrl(self) -> DREXController:
        ctrl = DREXController(d_model=D_MODEL, n_tiers=N_TIERS)
        ctrl.train()
        return ctrl

    def test_ten_consecutive_nan_rewards_do_not_raise(self):
        """10 consecutive NaN rewards must be swallowed silently."""
        ctrl = self._make_ctrl()
        # Must not raise for exactly _NAN_HALT_THRESHOLD (10) consecutive NaNs.
        for _ in range(10):
            ctrl.update(float("nan"))  # no exception expected

    def test_eleventh_nan_reward_raises_runtime_error(self):
        """The 11th consecutive NaN reward must raise RuntimeError."""
        ctrl = self._make_ctrl()
        for _ in range(10):
            ctrl.update(float("nan"))
        with pytest.raises(RuntimeError):
            ctrl.update(float("nan"))  # 11th → RuntimeError


# ===========================================================================
# 3. Routing collapse detection
# ===========================================================================

class TestControllerRoutingCollapse:

    def _make_ctrl_biased_to_tier(self, tier: int) -> DREXController:
        """Return a controller whose policy strongly prefers the given tier."""
        ctrl = DREXController(d_model=D_MODEL, n_tiers=N_TIERS)
        ctrl.train()
        with torch.no_grad():
            bias = torch.full((N_TIERS,), -100.0)
            bias[tier] = 100.0
            ctrl.policy[2].bias.data.copy_(bias)
        return ctrl

    def test_all_tier0_routing_logs_collapse_warning(self, caplog):
        """100 updates all routing to tier 0 must trigger the collapse WARNING."""
        ctrl = self._make_ctrl_biased_to_tier(0)
        x = torch.randn(1, 1, D_MODEL, dtype=torch.bfloat16)

        with caplog.at_level(logging.WARNING):
            for _ in range(100):
                ctrl(x)
                ctrl.update(1.0)

        assert any(
            "routing collapse" in r.message.lower() for r in caplog.records
        ), "Expected routing collapse WARNING after 100 all-tier-0 updates"

    def test_balanced_routing_does_not_log_collapse_warning(self, caplog):
        """Balanced routing (33/33/34) over 100 updates must NOT warn."""
        ctrl = DREXController(d_model=D_MODEL, n_tiers=N_TIERS)
        ctrl.train()
        x = torch.randn(1, 1, D_MODEL, dtype=torch.bfloat16)

        with caplog.at_level(logging.WARNING):
            for i in range(100):
                # Rotate tier bias: 0, 1, 2, 0, 1, 2, ...
                target_tier = i % N_TIERS
                with torch.no_grad():
                    bias = torch.full((N_TIERS,), -100.0)
                    bias[target_tier] = 100.0
                    ctrl.policy[2].bias.data.copy_(bias)
                ctrl(x)
                ctrl.update(1.0)

        assert not any(
            "routing collapse" in r.message.lower() for r in caplog.records
        ), "Balanced routing should not trigger a collapse WARNING"


# ===========================================================================
# 4. Learning — REINFORCE achieves better-than-random routing
# ===========================================================================

class TestControllerLearning:

    def test_reinforce_beats_random_on_synthetic_routing_task(self):
        """REINFORCE must achieve >50% routing accuracy on the last 100 of
        500 episodes (random baseline ≈ 33% for 3 tiers).

        SyntheticRoutingEnv (inline):
            scale=2.0 → target tier 2  (high-surprise input)
            scale=0.1 → target tier 0  (low-surprise input)
        """
        # Small model for speed; lr higher than default to converge faster.
        D_MODEL_SMALL = 32
        ctrl = DREXController(
            d_model=D_MODEL_SMALL, n_tiers=N_TIERS, hidden_dim=128, lr=5e-3
        )
        ctrl.train()

        recent_correct: list[bool] = []

        for ep in range(500):
            # Alternate high-surprise / low-surprise inputs
            if ep % 2 == 0:
                scale, target_tier = 2.0, 2  # high magnitude → tier 2
            else:
                scale, target_tier = 0.1, 0  # low magnitude → tier 0

            x = torch.randn(1, 1, D_MODEL_SMALL, dtype=torch.bfloat16) * scale
            write_decisions, _, _ = ctrl(x)

            # Selected tier: argmax of the one-hot write_decisions (shape 1,1,N_TIERS)
            selected_tier = write_decisions[0, 0].argmax().item()

            reward = 1.0 if selected_tier == target_tier else -1.0
            ctrl.update(reward)

            if ep >= 400:  # evaluate on last 100 episodes
                recent_correct.append(selected_tier == target_tier)

        accuracy = sum(recent_correct) / len(recent_correct)
        assert accuracy > 0.50, (
            f"Expected REINFORCE accuracy > 50% on last 100 episodes, "
            f"got {accuracy:.2%} (random baseline ≈ 33%)"
        )

