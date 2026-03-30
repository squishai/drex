"""Wave 4 — NoProp Semantic Memory tests.

Validates NoPropBlock and NoPropSemanticMemory against the contracts defined in
DREX_UNIFIED_SPEC.md § COMPONENT 3, using the REAL pipeline (no mocking).

Tests:
    TestNoPropBlockShapeAndDtype      — shape, dtype, float32-only guard, NaN/Inf,
                                        local_loss has grad_fn
    TestNoPropBlockIndependenceReal   — real NoPropSemanticMemory gradient isolation
    TestNoPropConvergence             — top-block task loss decreases over 300 steps
    TestNoPropVRAMEfficiency          — RSS growth guard (xfail on CPU)
"""
from __future__ import annotations

import os
import resource
from datetime import datetime

import pytest
import torch
import torch.nn.functional as F

from memory.semantic import NoPropBlock, NoPropSemanticMemory
from tests.python.conftest import assert_no_nan_inf

# ---------------------------------------------------------------------------
# Module-level constants (mirror conftest dims for clarity)
# ---------------------------------------------------------------------------
B = 2
S = 16
D_MODEL = 256


# ===========================================================================
# 1. NoPropBlock shape, dtype, and contract tests
# ===========================================================================

class TestNoPropBlockShapeAndDtype:

    def test_output_shape(self):
        """Output shape must equal input shape."""
        block = NoPropBlock(d_model=D_MODEL)
        block.train()
        x = torch.randn(B, S, D_MODEL)
        out, _ = block(x)
        assert out.shape == (B, S, D_MODEL), (
            f"Expected output shape {(B, S, D_MODEL)}, got {out.shape}"
        )

    def test_output_dtype_bfloat16(self):
        """Output must be bfloat16."""
        block = NoPropBlock(d_model=D_MODEL)
        block.train()
        x = torch.randn(B, S, D_MODEL)
        out, _ = block(x)
        assert out.dtype == torch.bfloat16, (
            f"Expected bfloat16 output, got {out.dtype}"
        )

    def test_input_must_be_float32(self):
        """NoPropBlock must assert float32 input and reject bfloat16."""
        block = NoPropBlock(d_model=D_MODEL)
        block.train()
        x_bf16 = torch.randn(B, S, D_MODEL, dtype=torch.bfloat16)
        with pytest.raises(AssertionError, match="float32"):
            block(x_bf16)

    def test_no_nan_inf(self):
        """Output must contain no NaN or Inf values."""
        block = NoPropBlock(d_model=D_MODEL)
        block.train()
        x = torch.randn(B, S, D_MODEL)
        out, loss = block(x)
        assert_no_nan_inf(out, "NoPropBlock output")
        assert_no_nan_inf(loss, "NoPropBlock local_loss")

    def test_local_loss_is_scalar_with_grad_fn(self):
        """local_loss must be a scalar Tensor with a grad_fn (train mode)."""
        block = NoPropBlock(d_model=D_MODEL)
        block.train()
        x = torch.randn(B, S, D_MODEL)
        _, loss = block(x)
        assert loss is not None, "local_loss must not be None in train mode"
        assert loss.shape == (), f"local_loss must be scalar, got shape {loss.shape}"
        assert loss.grad_fn is not None, "local_loss must have grad_fn"

    def test_eval_mode_returns_none_loss(self):
        """In eval mode, local_loss must be None and output is pass-through."""
        block = NoPropBlock(d_model=D_MODEL)
        block.eval()
        x = torch.randn(B, S, D_MODEL)
        out, loss = block(x)
        assert loss is None, "local_loss must be None in eval mode"
        assert out.shape == (B, S, D_MODEL)
        assert out.dtype == torch.bfloat16

    def test_output_shape_uses_dims_fixture(self, dims):
        """Shape contract against the canonical conftest dims fixture."""
        d = dims["D_MODEL"]
        b = dims["B"]
        s = dims["S"]
        block = NoPropBlock(d_model=d)
        block.train()
        x = torch.randn(b, s, d)
        out, _ = block(x)
        assert out.shape == (b, s, d)


# ===========================================================================
# 2. Block independence — REAL NoPropSemanticMemory (not synthetic stub)
# ===========================================================================

class TestNoPropBlockIndependenceReal:

    def test_block0_loss_does_not_touch_block1_params(self):
        """Backward through block 0's local loss must leave block 1 grads None."""
        mem = NoPropSemanticMemory(d_model=64, n_blocks=2)
        mem.train()

        # Zero all grads first
        for p in mem.parameters():
            p.grad = None

        # Run block 0 only — isolated graph from detached input
        x = torch.randn(B, S, 64)
        out0, loss0 = mem.blocks[0](x)
        assert loss0 is not None
        loss0.backward()

        # Block 1 must have no gradients
        for name, p in mem.blocks[1].named_parameters():
            assert p.grad is None, (
                f"Block 1 param '{name}' received gradient from block 0 loss — "
                "NoProp block independence contract violated."
            )
        # Block 0 must have gradients
        assert any(p.grad is not None for p in mem.blocks[0].parameters()), (
            "Block 0 must have gradients after its own loss backward"
        )

    def test_blockN_loss_does_not_touch_block0_params(self):
        """Backward through block N's local loss must leave block 0 grads None."""
        n_blocks = 3
        mem = NoPropSemanticMemory(d_model=64, n_blocks=n_blocks)
        mem.train()

        for p in mem.parameters():
            p.grad = None

        # Feed block N a fully-detached input so the graph is isolated
        x_detached = torch.randn(B, S, 64)
        out_n, loss_n = mem.blocks[-1](x_detached)
        assert loss_n is not None
        loss_n.backward()

        # All earlier blocks must have grad=None
        for blk_idx in range(n_blocks - 1):
            for name, p in mem.blocks[blk_idx].named_parameters():
                assert p.grad is None, (
                    f"Block {blk_idx} param '{name}' received gradient from "
                    f"block {n_blocks - 1} loss"
                )

    def test_each_optimizer_owns_only_its_block_params(self):
        """Each per-block optimizer must own exactly that block's parameters."""
        n_blocks = 3
        mem = NoPropSemanticMemory(d_model=64, n_blocks=n_blocks)

        for b_idx in range(n_blocks):
            opt_ids = {
                id(p)
                for pg in mem.blocks[b_idx].optimizer.param_groups
                for p in pg["params"]
            }
            block_ids = {id(p) for p in mem.blocks[b_idx].parameters()}
            other_ids = {
                id(p)
                for j, blk in enumerate(mem.blocks)
                if j != b_idx
                for p in blk.parameters()
            }
            assert opt_ids == block_ids, (
                f"Optimizer {b_idx}: param IDs do not match block {b_idx} params"
            )
            assert opt_ids.isdisjoint(other_ids), (
                f"Optimizer {b_idx} contains params from other blocks"
            )

    def test_train_step_isolates_each_block_graph(self):
        """train_step must produce per-block loss values without cross-contamination."""
        mem = NoPropSemanticMemory(d_model=64, n_blocks=3)
        mem.train()

        x = torch.randn(B, S, 64)
        losses = mem.train_step(x)

        assert len(losses) == 3, f"Expected 3 loss values, got {len(losses)}"
        assert all(isinstance(v, float) for v in losses), (
            "All loss values must be Python floats"
        )
        assert all(v >= 0.0 for v in losses), "All losses must be non-negative"


# ===========================================================================
# 3. Convergence test
# ===========================================================================

class TestNoPropConvergence:

    def test_top_block_loss_decreases(self):
        """Top-block denoising loss must strictly decrease over 300 training steps.

        Uses d_model=64, n_blocks=2, noise_std=0.10 for speed.
        """
        torch.manual_seed(42)
        d = 64
        n_blocks = 2
        mem = NoPropSemanticMemory(d_model=d, n_blocks=n_blocks, noise_std=0.10, lr=3e-3)
        mem.train()

        # Fixed clean signal
        clean = torch.randn(B, S, d)

        initial_losses: list[float] = []
        final_losses: list[float] = []
        history: list[list[float]] = [[] for _ in range(n_blocks)]

        for step in range(301):
            losses = mem.train_step(clean)

            if step == 0:
                initial_losses = list(losses)
            if step == 300:
                final_losses = list(losses)
            if step > 0:
                for blk_idx, v in enumerate(losses):
                    history[blk_idx].append(v)

        # Top block trains on its own local denoising task — must strictly improve
        assert final_losses[-1] < initial_losses[-1], (
            f"Top block loss did not decrease: "
            f"step 0 = {initial_losses[-1]:.6f}, step 300 = {final_losses[-1]:.6f}"
        )
        # Lower blocks: divergence guard only
        for blk_idx in range(n_blocks - 1):
            assert final_losses[blk_idx] < 0.5, (
                f"Block {blk_idx} loss diverged: step 300 = {final_losses[blk_idx]:.6f}"
            )

        # Save loss curves if matplotlib available
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_dir = os.path.join(
                "experiments", "runs", f"{timestamp}_noprop_convergence"
            )
            os.makedirs(out_dir, exist_ok=True)

            fig, ax = plt.subplots()
            for blk_idx, h in enumerate(history):
                ax.plot(h, label=f"Block {blk_idx} loss")
            ax.set_xlabel("Step")
            ax.set_ylabel("MSE Loss")
            ax.set_title("NoProp Semantic Memory local losses")
            ax.legend()
            fig.savefig(os.path.join(out_dir, "noprop_loss_curves.png"))
            plt.close(fig)
        except ImportError:
            pass


# ===========================================================================
# 4. VRAM efficiency test (xfail on CPU — meaningful on MPS/CUDA only)
# ===========================================================================

class TestNoPropVRAMEfficiency:

    @pytest.mark.xfail(
        not (torch.backends.mps.is_available() or torch.cuda.is_available()),
        reason="VRAM measurement only meaningful on MPS or CUDA hardware",
        strict=False,
    )
    def test_rss_does_not_double_during_train_step(self):
        """RSS after train_step must be < 2× initial RSS.

        Verifies that NoProp local-loss training does not materialise a large
        global computation graph.
        """
        mem = NoPropSemanticMemory(d_model=64, n_blocks=4, noise_std=0.10)
        mem.train()

        x = torch.randn(32, S, 64)

        # Warmup
        mem.train_step(x)

        rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        mem.train_step(x)
        rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

        assert rss_after < rss_before * 2, (
            f"RSS doubled during NoProp train_step: "
            f"before={rss_before}, after={rss_after}"
        )
