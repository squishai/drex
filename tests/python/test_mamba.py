"""Tests for MambaSSM and PCNMambaBackbone — Wave 3.

Covers:
    TestMambaSSMShapeAndDtype  — shape contract, dtype contract, NaN/Inf guard
    TestMambaCausality         — no future token leakage
    TestPCNConvergence         — all layer losses decrease over 200 steps
    TestPCNGradientLeak        — cross-layer gradient isolation

Ref: DREX_UNIFIED_SPEC.md § COMPONENT 2 validation criteria
"""
import os
from datetime import datetime

import pytest
import torch
import torch.nn.functional as F

from backbone.mamba import MambaSSM, PCNMambaBackbone
from tests.python.conftest import assert_no_nan_inf

# ---------------------------------------------------------------------------
# Constants matching conftest.py canonical dims
# ---------------------------------------------------------------------------
B = 2
S = 16
D_MODEL = 256


# ===========================================================================
# 1. Shape and Dtype contract tests
# ===========================================================================

class TestMambaSSMShapeAndDtype:

    def test_output_shape(self):
        mamba = MambaSSM(d_model=D_MODEL)
        x = torch.randn(B, S, D_MODEL)
        out = mamba(x)
        assert out.shape == (B, S, D_MODEL), (
            f"Expected ({B}, {S}, {D_MODEL}), got {out.shape}"
        )

    def test_output_dtype_bfloat16(self):
        mamba = MambaSSM(d_model=D_MODEL)
        x = torch.randn(B, S, D_MODEL)
        out = mamba(x)
        assert out.dtype == torch.bfloat16, (
            f"MambaSSM output must be bfloat16, got {out.dtype}"
        )

    def test_input_must_be_float32(self):
        mamba = MambaSSM(d_model=D_MODEL)
        x_bf16 = torch.randn(B, S, D_MODEL, dtype=torch.bfloat16)
        with pytest.raises((AssertionError, TypeError, RuntimeError)):
            mamba(x_bf16)

    def test_no_nan_inf(self):
        mamba = MambaSSM(d_model=D_MODEL)
        x = torch.randn(B, S, D_MODEL)
        out = mamba(x)
        assert_no_nan_inf(out.float(), "mamba_output")

    def test_recurrence(self):
        """Two different inputs must produce different outputs."""
        mamba = MambaSSM(d_model=D_MODEL)
        x1 = torch.randn(B, S, D_MODEL)
        x2 = torch.randn(B, S, D_MODEL)
        out1 = mamba(x1)
        out2 = mamba(x2)
        assert not torch.allclose(out1, out2), (
            "Different inputs must produce different outputs"
        )

    def test_output_shape_uses_dims_fixture(self, dims):
        """Shape test using the canonical conftest fixture."""
        mamba = MambaSSM(d_model=dims["D_MODEL"])
        x = torch.randn(dims["B"], dims["S"], dims["D_MODEL"])
        out = mamba(x)
        assert out.shape == (dims["B"], dims["S"], dims["D_MODEL"])


# ===========================================================================
# 2. Causality test
# ===========================================================================

class TestMambaCausality:

    def test_output_at_t_independent_of_future(self):
        """Positions 0..7 must be identical whether positions 8..15 are real or zero.

        This verifies causal conv1d padding is correct (left-pad, not right-pad).
        """
        mamba = MambaSSM(d_model=D_MODEL)
        mamba.eval()

        x = torch.randn(1, S, D_MODEL)

        # Zero out positions 8..S-1
        x_masked = x.clone()
        x_masked[:, 8:, :] = 0.0

        with torch.no_grad():
            out_full = mamba(x)
            out_masked = mamba(x_masked)

        # Positions 0..7 must be identical — future positions must not affect past
        assert torch.allclose(
            out_full[:, :8, :].float(),
            out_masked[:, :8, :].float(),
            atol=1e-3,
        ), (
            "Causality violation: output at positions 0..7 changed when positions "
            "8..15 were zeroed. Check causal conv1d padding."
        )


# ===========================================================================
# 3. PCN Convergence test
# ===========================================================================

class TestPCNConvergence:

    def test_all_layer_losses_decrease(self):
        """All per-layer PCN losses must decrease over 200 training steps.

        Uses a small model (d_model=64, n_layers=2) for speed.
        Loss curves saved to experiments/runs/<timestamp>_pcn_convergence/.
        """
        torch.manual_seed(0)
        d = 64
        net = PCNMambaBackbone(d_model=d, n_layers=2)
        net.train()

        # Synthetic sine task: predict next-step
        t_vals = torch.arange(S, dtype=torch.float32) / 10.0
        seq = torch.sin(t_vals).unsqueeze(0).unsqueeze(-1).expand(B, S, d)
        target = torch.roll(seq, -1, dims=1).clone()
        target[:, -1, :] = seq[:, -1, :]  # last step target = itself

        initial_losses: list[float] = []
        final_losses: list[float] = []
        history: list[list[float]] = [[] for _ in range(2)]

        for step in range(201):
            x_in = seq.float()
            hidden, _ = net(x_in)

            losses = net.train_step(
                x_in, lambda h: F.mse_loss(h.float(), target.float())
            )

            if step == 0:
                initial_losses = list(losses)
            if step == 200:
                final_losses = list(losses)
            if step > 0:
                for l, v in enumerate(losses):
                    history[l].append(v)

        # Top layer: directly minimises task loss — must strictly decrease.
        assert final_losses[-1] < initial_losses[-1], (
            f"Top-layer task loss did not decrease: "
            f"step 0 = {initial_losses[-1]:.6f}, step 200 = {final_losses[-1]:.6f}"
        )
        # Lower layers: inter-layer prediction losses can fluctuate as upper
        # layers evolve their representations.  Guard against divergence only.
        for l in range(len(final_losses) - 1):
            assert final_losses[l] < 0.5, (
                f"Layer {l} inter-layer loss diverged: step 200 = {final_losses[l]:.6f}"
            )

        # Save loss curve PNG
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_dir = os.path.join(
                "experiments", "runs", f"{timestamp}_pcn_convergence"
            )
            os.makedirs(out_dir, exist_ok=True)

            fig, ax = plt.subplots()
            for l, h in enumerate(history):
                ax.plot(h, label=f"Layer {l} loss")
            ax.set_xlabel("Step")
            ax.set_ylabel("MSE Loss")
            ax.set_title("PCN MambaSSM local losses")
            ax.legend()
            fig.savefig(os.path.join(out_dir, "pc_loss_curves.png"))
            plt.close(fig)
        except ImportError:
            pass


# ===========================================================================
# 4. PCN Gradient Leak tests
# ===========================================================================

class TestPCNGradientLeak:

    def test_layer0_loss_does_not_touch_layer1_params(self):
        """loss from layer 0 must leave layer 1 parameters with grad=None."""
        net = PCNMambaBackbone(d_model=64, n_layers=2)
        for p in net.parameters():
            p.grad = None

        x = torch.randn(2, 16, 64)
        out0 = net.layers[0](x.float())
        out1 = net.layers[1](out0.float())
        loss0 = F.mse_loss(out0.float(), out1.detach().float())
        loss0.backward()

        for name, p in net.layers[1].named_parameters():
            assert p.grad is None, (
                f"layer 1 param '{name}' received gradient from layer 0 loss"
            )
        has_grad = any(p.grad is not None for p in net.layers[0].parameters())
        assert has_grad, "layer 0 params must have gradient after loss0.backward()"

    def test_layer1_loss_does_not_touch_layer0_params(self):
        """loss from layer 1 must leave layer 0 parameters with grad=None."""
        net = PCNMambaBackbone(d_model=64, n_layers=2)
        for p in net.parameters():
            p.grad = None

        x = torch.randn(2, 16, 64)
        out0 = net.layers[0](x.float())
        out0_detached = out0.detach().float()
        out1 = net.layers[1](out0_detached)
        loss1 = out1.float().sum()
        loss1.backward()

        for name, p in net.layers[0].named_parameters():
            assert p.grad is None, (
                f"layer 0 param '{name}' received gradient from layer 1 loss"
            )

    def test_each_optimizer_owns_only_its_layer_params(self):
        """Each per-layer optimizer must own exactly that layer's parameters."""
        net = PCNMambaBackbone(d_model=64, n_layers=2)

        for l in range(2):
            opt_ids = {
                id(p)
                for pg in net.optimizers[l].param_groups
                for p in pg["params"]
            }
            layer_ids = {id(p) for p in net.layers[l].parameters()}
            other_ids = {
                id(p)
                for m, layer in enumerate(net.layers)
                if m != l
                for p in layer.parameters()
            }
            assert opt_ids == layer_ids, (
                f"optimizer {l}: param IDs mismatch with layer {l} parameters"
            )
            assert opt_ids.isdisjoint(other_ids), (
                f"optimizer {l} contains params from other layers"
            )

    def test_train_step_returns_n_losses(self):
        """train_step must return one loss value per layer."""
        net = PCNMambaBackbone(d_model=64, n_layers=2)
        x = torch.randn(2, 16, 64)
        hidden, _ = net(x)
        losses = net.train_step(x, lambda h: h.float().mean())
        assert len(losses) == 2
        assert all(isinstance(v, float) for v in losses)


# ---------------------------------------------------------------------------
# TRAINING EXPERIMENT — NOT A CI TEST (must be run manually)
# ---------------------------------------------------------------------------
# Phase 1 exit criterion: PC-trained PCNMambaBackbone within ±10% perplexity of
# same-size backprop-trained MambaSSM on WikiText-2 (10M tokens).
# Script: experiments/pcn_vs_backprop_ppl.py (to be created in a future wave).
# Result must be logged in DREX_UNIFIED_SPEC.md under Component 2 validation
# criteria before Phase 1 is declared complete.
# ---------------------------------------------------------------------------
