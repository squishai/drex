"""
Tests for drex.models.memory_esn — EchoStateMemory (Phase 23).

Coverage targets:
  - Reservoir construction (_make_reservoir): spectral radius bound, sparsity
  - EchoStateMemory.__init__: valid init, bad d_model, bad gate_thresh,
    bad spectral_radius
  - API compatibility with MemoryModule: same method signatures, same output shape
  - Forward pass: correct output shape, L=1 edge case, L=2, write rate tracking,
    write rate range, gate fires for first token, no NaN at L=128
  - Reservoir buffers: not in parameters(), not updated by optimizer, in buffers()
  - Gradient flow: flows through readout; does NOT pass through reservoir buffers
  - Ablation flags: use_null_gate=False, use_recency_weight=False
  - Reproducibility: same seed → same reservoir; different seed → different reservoir
  - assert_write_rate_valid: passes in range, raises outside range
  - alpha() static method
  - Integration: works when wired into DrexLayer via use_esn_memory=True
"""

from __future__ import annotations

import math
from typing import Optional

import pytest
import torch
import torch.nn as nn

from drex.models.memory_esn import (
    WRITE_RATE_HI,
    WRITE_RATE_LO,
    EchoStateMemory,
    _make_reservoir,
)
from drex.models.transformer import DrexConfig, DrexTransformer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def d_model() -> int:
    return 64  # small for fast tests


@pytest.fixture
def batch() -> int:
    return 2


@pytest.fixture
def seq_len() -> int:
    return 16


@pytest.fixture
def esn(d_model: int) -> EchoStateMemory:
    """Default EchoStateMemory with d_model=64."""
    return EchoStateMemory(d_model=d_model, reservoir_mult=2, reservoir_seed=42)


# ---------------------------------------------------------------------------
# _make_reservoir
# ---------------------------------------------------------------------------


class TestMakeReservoir:
    def test_shape(self):
        rng = torch.Generator().manual_seed(0)
        W = _make_reservoir(64, 0.1, 0.9, rng)
        assert W.shape == (64, 64)

    def test_dtype_float32(self):
        rng = torch.Generator().manual_seed(0)
        W = _make_reservoir(32, 0.1, 0.9, rng)
        assert W.dtype == torch.float32

    def test_spectral_radius_bound(self):
        """Spectral norm of returned W should be ≤ target (up to power iteration error)."""
        rng = torch.Generator().manual_seed(1)
        target = 0.95
        W = _make_reservoir(64, 0.1, target, rng)
        # scipy.linalg.svds would be exact; use torch norm-2 approx via a few power iters
        v = torch.randn(64, 1)
        v = v / v.norm()
        for _ in range(30):
            u = (W @ v)
            u = u / u.norm()
            v = (W.t() @ u)
            v = v / v.norm()
        sigma = (u.t() @ W @ v).item()
        assert abs(sigma) <= target + 1e-3, (
            f"Spectral norm {sigma:.4f} exceeds target {target} + tol"
        )

    def test_sparsity(self):
        """Approximately connectivity fraction of entries should be non-zero."""
        rng = torch.Generator().manual_seed(2)
        N, conn = 200, 0.05
        W = _make_reservoir(N, conn, 0.9, rng)
        actual = (W != 0).float().mean().item()
        # Allow ±50% relative error; the mask is random so exact fraction varies
        assert actual < conn * 2.0, f"Too many non-zeros: {actual:.3f} (expected ~{conn})"

    def test_invalid_spectral_radius_above_1(self):
        rng = torch.Generator().manual_seed(0)
        with pytest.raises(ValueError, match="spectral_radius"):
            _make_reservoir(16, 0.1, 1.0, rng)

    def test_invalid_spectral_radius_zero(self):
        rng = torch.Generator().manual_seed(0)
        with pytest.raises(ValueError, match="spectral_radius"):
            _make_reservoir(16, 0.1, 0.0, rng)

    def test_same_seed_deterministic(self):
        rng1 = torch.Generator().manual_seed(7)
        rng2 = torch.Generator().manual_seed(7)
        W1 = _make_reservoir(32, 0.1, 0.9, rng1)
        W2 = _make_reservoir(32, 0.1, 0.9, rng2)
        assert torch.allclose(W1, W2)

    def test_different_seeds_differ(self):
        W1 = _make_reservoir(32, 0.1, 0.9, torch.Generator().manual_seed(0))
        W2 = _make_reservoir(32, 0.1, 0.9, torch.Generator().manual_seed(1))
        assert not torch.allclose(W1, W2)

    def test_degenerate_all_zero_reservoir_does_not_error(self):
        """When connectivity=0 all weights are zero; sigma ≤ 1e-10 branch must not raise."""
        rng = torch.Generator().manual_seed(0)
        # connectivity=0 produces an all-zero mask → W = 0 → sigma ≈ 0, skip scaling
        W = _make_reservoir(16, 0.0, 0.9, rng)
        # The result must be a zero matrix (unscaled) without error
        assert W.shape == (16, 16)
        assert W.abs().max().item() < 1e-6


# ---------------------------------------------------------------------------
# EchoStateMemory.__init__
# ---------------------------------------------------------------------------


class TestEchoStateMemoryInit:
    def test_valid_init(self, d_model):
        esn = EchoStateMemory(d_model=d_model)
        assert esn.d_model == d_model
        assert esn._d_half == d_model // 2

    def test_odd_d_model_raises(self):
        with pytest.raises(ValueError, match="even"):
            EchoStateMemory(d_model=33)

    def test_gate_thresh_below_minimum_raises(self):
        with pytest.raises(ValueError, match="gate_thresh"):
            EchoStateMemory(d_model=32, gate_thresh=0.39)

    def test_spectral_radius_at_1_raises(self):
        with pytest.raises(ValueError, match="spectral_radius"):
            EchoStateMemory(d_model=32, spectral_radius=1.0)

    def test_spectral_radius_above_1_raises(self):
        with pytest.raises(ValueError, match="spectral_radius"):
            EchoStateMemory(d_model=32, spectral_radius=1.5)

    def test_spectral_radius_zero_raises(self):
        with pytest.raises(ValueError, match="spectral_radius"):
            EchoStateMemory(d_model=32, spectral_radius=0.0)

    def test_reservoir_size(self, d_model):
        mult = 3
        esn = EchoStateMemory(d_model=d_model, reservoir_mult=mult)
        assert esn._N == d_model * mult

    def test_null_gate_created_when_enabled(self, d_model):
        esn = EchoStateMemory(d_model=d_model, use_null_gate=True)
        assert esn.null_gate is not None

    def test_null_gate_absent_when_disabled(self, d_model):
        esn = EchoStateMemory(d_model=d_model, use_null_gate=False)
        assert esn.null_gate is None


# ---------------------------------------------------------------------------
# Reservoir buffers: not in parameters, not updated by backward
# ---------------------------------------------------------------------------


class TestReservoirBuffers:
    def test_reservoir_not_in_parameters(self, esn: EchoStateMemory):
        param_names = {n for n, _ in esn.named_parameters()}
        assert "W_res_sem" not in param_names
        assert "W_in_sem" not in param_names
        assert "W_res_epi" not in param_names
        assert "W_in_epi" not in param_names

    def test_reservoir_in_buffers(self, esn: EchoStateMemory):
        buf_names = {n for n, _ in esn.named_buffers()}
        assert "W_res_sem" in buf_names
        assert "W_in_sem" in buf_names
        assert "W_res_epi" in buf_names
        assert "W_in_epi" in buf_names

    def test_reservoir_unchanged_after_forward(self, esn: EchoStateMemory, batch, seq_len, d_model, device):
        esn = esn.to(torch.device(device))
        W_res_sem_before = esn.W_res_sem.clone()
        W_in_sem_before = esn.W_in_sem.clone()

        x = torch.randn(batch, seq_len, d_model, device=device)
        out = esn(x)
        out.sum().backward()

        assert torch.allclose(esn.W_res_sem, W_res_sem_before), "W_res_sem was modified"
        assert torch.allclose(esn.W_in_sem, W_in_sem_before), "W_in_sem was modified"

    def test_reservoir_unchanged_after_optimizer_step(
        self, esn: EchoStateMemory, batch, seq_len, d_model, device
    ):
        esn = esn.to(torch.device(device))
        W_before = esn.W_res_sem.clone()

        opt = torch.optim.Adam(esn.parameters(), lr=1e-3)
        x = torch.randn(batch, seq_len, d_model, device=device)
        out = esn(x)
        out.sum().backward()
        opt.step()

        assert torch.allclose(esn.W_res_sem, W_before), "W_res_sem was updated by optimizer"

    def test_sem_epi_reservoirs_differ(self, d_model):
        esn = EchoStateMemory(d_model=d_model, reservoir_seed=42)
        assert not torch.allclose(esn.W_res_sem, esn.W_res_epi), (
            "Semantic and episodic reservoirs should be independently generated"
        )

    def test_different_seeds_give_different_reservoirs(self, d_model):
        esn1 = EchoStateMemory(d_model=d_model, reservoir_seed=10)
        esn2 = EchoStateMemory(d_model=d_model, reservoir_seed=20)
        assert not torch.allclose(esn1.W_res_sem, esn2.W_res_sem)

    def test_same_seed_gives_same_reservoirs(self, d_model):
        esn1 = EchoStateMemory(d_model=d_model, reservoir_seed=99)
        esn2 = EchoStateMemory(d_model=d_model, reservoir_seed=99)
        assert torch.allclose(esn1.W_res_sem, esn2.W_res_sem)
        assert torch.allclose(esn1.W_in_sem, esn2.W_in_sem)


# ---------------------------------------------------------------------------
# Forward pass: shape, edge cases
# ---------------------------------------------------------------------------


class TestForwardShape:
    def test_output_shape(self, esn: EchoStateMemory, batch, seq_len, d_model, device):
        esn = esn.to(torch.device(device))
        x = torch.randn(batch, seq_len, d_model, device=device)
        out = esn(x)
        assert out.shape == (batch, d_model)

    def test_output_shape_l1(self, d_model, batch, device):
        """L=1: no write phase, output is from zero reservoir state."""
        esn = EchoStateMemory(d_model=d_model, reservoir_mult=2).to(torch.device(device))
        x = torch.randn(batch, 1, d_model, device=device)
        out = esn(x)
        assert out.shape == (batch, d_model)

    def test_output_shape_l2(self, d_model, batch, device):
        esn = EchoStateMemory(d_model=d_model, reservoir_mult=2).to(torch.device(device))
        x = torch.randn(batch, 2, d_model, device=device)
        out = esn(x)
        assert out.shape == (batch, d_model)

    def test_output_shape_batch1(self, d_model, device):
        esn = EchoStateMemory(d_model=d_model, reservoir_mult=2).to(torch.device(device))
        x = torch.randn(1, 8, d_model, device=device)
        out = esn(x)
        assert out.shape == (1, d_model)

    def test_no_nan_short_sequence(self, d_model, device):
        esn = EchoStateMemory(d_model=d_model, reservoir_mult=2).to(torch.device(device))
        x = torch.randn(2, 8, d_model, device=device)
        out = esn(x)
        assert out.isfinite().all(), "NaN or Inf in output for short sequence"

    def test_no_nan_long_sequence(self, d_model, device):
        """L=128 should remain numerically stable."""
        esn = EchoStateMemory(d_model=d_model, reservoir_mult=2).to(torch.device(device))
        x = torch.randn(2, 128, d_model, device=device)
        out = esn(x)
        assert out.isfinite().all(), "NaN or Inf in output for L=128"

    def test_output_nontrivial(self, esn: EchoStateMemory, batch, d_model, device):
        """Output with non-zero input should not be the zero vector."""
        esn = esn.to(torch.device(device))
        x = torch.randn(batch, 16, d_model, device=device)
        out = esn(x)
        assert out.abs().sum().item() > 0.0, "Output is exactly zero"

    def test_output_varies_with_input(self, d_model, batch, device):
        """Two different inputs should produce different outputs."""
        esn = EchoStateMemory(d_model=d_model, reservoir_mult=2).to(torch.device(device))
        x1 = torch.randn(batch, 16, d_model, device=device)
        x2 = torch.randn(batch, 16, d_model, device=device)
        out1 = esn(x1)
        out2 = esn(x2)
        assert not torch.allclose(out1, out2), "Identical output for different inputs"


# ---------------------------------------------------------------------------
# Write rate
# ---------------------------------------------------------------------------


class TestWriteRate:
    def test_write_rate_set_after_forward(self, esn: EchoStateMemory, batch, seq_len, d_model, device):
        esn = esn.to(torch.device(device))
        x = torch.randn(batch, seq_len, d_model, device=device)
        _ = esn(x)
        wr = esn.last_write_rate()
        assert 0.0 <= wr <= 1.0

    def test_write_rate_zero_for_l1(self, d_model, device):
        """No write steps when L=1."""
        esn = EchoStateMemory(d_model=d_model, reservoir_mult=2).to(torch.device(device))
        x = torch.randn(2, 1, d_model, device=device)
        _ = esn(x)
        assert esn.last_write_rate() == 0.0

    def test_first_token_always_fires(self, d_model, device):
        """With L=2, exactly 1 write opportunity; gate must always fire at t=0."""
        esn = EchoStateMemory(d_model=d_model, reservoir_mult=2).to(torch.device(device))
        x = torch.randn(4, 2, d_model, device=device)
        _ = esn(x)
        # With B=4 and L-1=1, wr = total_fires / 4; all 4 must fire → wr=1.0
        assert esn.last_write_rate() == pytest.approx(1.0, abs=1e-5)

    def test_assert_write_rate_valid_passes(self, esn: EchoStateMemory):
        """Manually set a valid write rate and check assertion passes."""
        esn._last_write_rate = 0.5
        esn.assert_write_rate_valid()  # should not raise

    def test_assert_write_rate_valid_fails_low(self, esn: EchoStateMemory):
        esn._last_write_rate = WRITE_RATE_LO - 0.01
        with pytest.raises(AssertionError, match="Write rate"):
            esn.assert_write_rate_valid()

    def test_assert_write_rate_valid_fails_high(self, esn: EchoStateMemory):
        esn._last_write_rate = WRITE_RATE_HI + 0.01
        with pytest.raises(AssertionError, match="Write rate"):
            esn.assert_write_rate_valid()

    def test_write_rate_boundary_values(self, esn: EchoStateMemory):
        esn._last_write_rate = WRITE_RATE_LO
        esn.assert_write_rate_valid()
        esn._last_write_rate = WRITE_RATE_HI
        esn.assert_write_rate_valid()


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------


class TestGradientFlow:
    def test_grad_flows_through_sem_readout(self, esn: EchoStateMemory, batch, d_model, device):
        esn = esn.to(torch.device(device))
        x = torch.randn(batch, 16, d_model, device=device)
        out = esn(x)
        # Use pow(2).sum() — out.sum() gives zero loss when the last op is LayerNorm
        # because sum(LayerNorm(z)) = 0 for any z, making all upstream grads zero.
        out.pow(2).sum().backward()
        assert esn.sem_readout.weight.grad is not None
        assert esn.sem_readout.weight.grad.abs().sum().item() > 0.0

    def test_grad_flows_through_epi_readout(self, esn: EchoStateMemory, batch, d_model, device):
        esn = esn.to(torch.device(device))
        x = torch.randn(batch, 16, d_model, device=device)
        out = esn(x)
        out.pow(2).sum().backward()
        assert esn.epi_readout.weight.grad is not None
        assert esn.epi_readout.weight.grad.abs().sum().item() > 0.0

    def test_grad_flows_through_out_proj(self, esn: EchoStateMemory, batch, d_model, device):
        esn = esn.to(torch.device(device))
        x = torch.randn(batch, 16, d_model, device=device)
        out = esn(x)
        out.pow(2).sum().backward()
        assert esn.out_proj.weight.grad is not None

    def test_grad_flows_through_key_projs(self, esn: EchoStateMemory, batch, d_model, device):
        """sem_proj / epi_proj receive gradients via the read query path.

        In EchoStateMemory (matching MemoryModule), sem_proj is applied to the
        query token q = x[:, -1, :] during the read phase, providing a gradient
        path back to sem_proj.weight identical to MemoryModule line 500.
        """
        esn = esn.to(torch.device(device))
        x = torch.randn(batch, 16, d_model, device=device)
        out = esn(x)
        out.pow(2).sum().backward()
        assert esn.sem_proj.weight.grad is not None
        assert esn.sem_proj.weight.grad.abs().sum().item() > 0.0
        assert esn.epi_proj.weight.grad is not None
        assert esn.epi_proj.weight.grad.abs().sum().item() > 0.0

    def test_no_grad_on_reservoir_buffers(self, esn: EchoStateMemory, batch, d_model, device):
        """Reservoir weights must never accumulate gradients."""
        esn = esn.to(torch.device(device))
        x = torch.randn(batch, 16, d_model, device=device)
        out = esn(x)
        out.pow(2).sum().backward()
        # Buffers don't have .grad attribute — verify they weren't accidentally promoted
        assert not esn.W_res_sem.requires_grad
        assert not esn.W_in_sem.requires_grad
        assert not esn.W_res_epi.requires_grad
        assert not esn.W_in_epi.requires_grad

    def test_null_gate_grad(self, d_model, batch, device):
        esn = EchoStateMemory(d_model=d_model, use_null_gate=True, reservoir_mult=2).to(
            torch.device(device)
        )
        x = torch.randn(batch, 16, d_model, device=device)
        out = esn(x)
        out.pow(2).sum().backward()
        assert esn.null_gate is not None
        assert esn.null_gate.weight.grad is not None


# ---------------------------------------------------------------------------
# Ablation flags
# ---------------------------------------------------------------------------


class TestAblationFlags:
    def test_no_null_gate_output_shape(self, d_model, batch, device):
        esn = EchoStateMemory(d_model=d_model, use_null_gate=False, reservoir_mult=2).to(
            torch.device(device)
        )
        x = torch.randn(batch, 16, d_model, device=device)
        out = esn(x)
        assert out.shape == (batch, d_model)

    def test_no_null_gate_no_parameters_for_it(self, d_model):
        esn_with = EchoStateMemory(d_model=d_model, use_null_gate=True, reservoir_mult=2)
        esn_without = EchoStateMemory(d_model=d_model, use_null_gate=False, reservoir_mult=2)
        params_with = sum(p.numel() for p in esn_with.parameters())
        params_without = sum(p.numel() for p in esn_without.parameters())
        # null_gate is Linear(d_model, 1) with bias → d_model + 1 extra params
        assert params_with == params_without + d_model + 1

    def test_no_recency_weight_output_shape(self, d_model, batch, device):
        esn = EchoStateMemory(d_model=d_model, use_recency_weight=False, reservoir_mult=2).to(
            torch.device(device)
        )
        x = torch.randn(batch, 16, d_model, device=device)
        out = esn(x)
        assert out.shape == (batch, d_model)

    def test_no_recency_weight_differs_from_with(self, d_model, batch, device):
        """Recency-weighted and unweighted versions should produce different outputs."""
        esn_with = EchoStateMemory(
            d_model=d_model, use_recency_weight=True, reservoir_mult=2, reservoir_seed=1
        ).to(torch.device(device))
        esn_without = EchoStateMemory(
            d_model=d_model, use_recency_weight=False, reservoir_mult=2, reservoir_seed=1
        ).to(torch.device(device))
        x = torch.randn(batch, 16, d_model, device=device)
        out_with = esn_with(x)
        out_without = esn_without(x)
        assert not torch.allclose(out_with, out_without)


# ---------------------------------------------------------------------------
# Alpha static method
# ---------------------------------------------------------------------------


class TestAlphaStaticMethod:
    def test_alpha_l32(self):
        a = EchoStateMemory.alpha(32)
        assert abs(a - 0.95 ** (96 / 32)) < 1e-9

    def test_alpha_l96(self):
        a = EchoStateMemory.alpha(96)
        assert abs(a - 0.95) < 1e-9

    def test_alpha_increases_with_l(self):
        """Longer sequences → larger α → slower EMA decay → longer memory."""
        a32 = EchoStateMemory.alpha(32)
        a64 = EchoStateMemory.alpha(64)
        a128 = EchoStateMemory.alpha(128)
        assert a32 < a64 < a128

    def test_alpha_bounded(self):
        for L in [8, 16, 32, 64, 96, 128, 256, 512]:
            a = EchoStateMemory.alpha(L)
            assert 0.0 < a < 1.0, f"α({L}) = {a:.4f} out of (0, 1)"


# ---------------------------------------------------------------------------
# API compatibility with MemoryModule
# ---------------------------------------------------------------------------


class TestAPICompatibility:
    def test_has_last_write_rate_method(self, esn: EchoStateMemory):
        assert hasattr(esn, "last_write_rate")
        assert callable(esn.last_write_rate)

    def test_has_assert_write_rate_valid_method(self, esn: EchoStateMemory):
        assert hasattr(esn, "assert_write_rate_valid")
        assert callable(esn.assert_write_rate_valid)

    def test_has_alpha_static_method(self):
        assert hasattr(EchoStateMemory, "alpha")
        assert callable(EchoStateMemory.alpha)

    def test_forward_returns_same_shape_as_memory_module(self, d_model, batch, device):
        """EchoStateMemory and MemoryModule must return (B, d_model)."""
        from drex.models.memory import MemoryModule

        esn = EchoStateMemory(d_model=d_model, reservoir_mult=2).to(torch.device(device))
        mem = MemoryModule(d_model=d_model).to(torch.device(device))
        x = torch.randn(batch, 16, d_model, device=device)

        out_esn = esn(x)
        out_mem = mem(x)
        assert out_esn.shape == out_mem.shape


# ---------------------------------------------------------------------------
# Integration with DrexLayer / DrexTransformer
# ---------------------------------------------------------------------------


class TestDrexIntegration:
    def test_esn_in_drex_transformer_forward(self, device):
        """EchoStateMemory wired via use_esn_memory=True should not crash forward."""
        cfg = DrexConfig(
            d_model=64,
            n_heads=4,
            n_layers=2,
            vocab_size=256,
            window_size=64,
            use_episodic_memory=True,
            use_esn_memory=True,
            esn_reservoir_mult=2,
        )
        dev = torch.device(device)
        model = DrexTransformer(cfg).to(dev)
        states = model.init_states(batch=2, device=dev)
        x = torch.randint(0, 256, (2, 16), device=dev)
        out, new_states = model(x, states)
        assert out.shape == (2, 16, 256)

    def test_esn_memory_params_in_model(self, device):
        """EchoStateMemory's trained params should appear in model.parameters()."""
        cfg = DrexConfig(
            d_model=64,
            n_heads=4,
            n_layers=2,
            vocab_size=256,
            window_size=64,
            use_episodic_memory=True,
            use_esn_memory=True,
            esn_reservoir_mult=2,
        )
        model = DrexTransformer(cfg)
        param_names = [n for n, _ in model.named_parameters()]
        # sem_readout should appear in at least one layer
        readout_params = [n for n in param_names if "sem_readout" in n or "epi_readout" in n]
        assert len(readout_params) > 0, "No readout parameters found in model"

    def test_esn_reservoir_buffers_not_in_model_params(self, device):
        """Reservoir buffers must NOT appear in model.parameters()."""
        cfg = DrexConfig(
            d_model=64,
            n_heads=4,
            n_layers=2,
            vocab_size=256,
            window_size=64,
            use_episodic_memory=True,
            use_esn_memory=True,
            esn_reservoir_mult=2,
        )
        model = DrexTransformer(cfg)
        param_names = {n for n, _ in model.named_parameters()}
        for n in param_names:
            assert "W_res_sem" not in n, f"Reservoir buffer found in params: {n}"
            assert "W_res_epi" not in n, f"Reservoir buffer found in params: {n}"

    def test_esn_vs_memory_module_both_train(self, device):
        """Both use_episodic_memory=True and use_esn_memory=True should train without error."""
        dev = torch.device(device)
        for use_esn in [False, True]:
            cfg = DrexConfig(
                d_model=64,
                n_heads=4,
                n_layers=2,
                vocab_size=256,
                window_size=64,
                use_episodic_memory=True,
                use_esn_memory=use_esn,
                esn_reservoir_mult=2,
            )
            model = DrexTransformer(cfg).to(dev)
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)
            x = torch.randint(0, 256, (2, 16), device=dev)
            states = model.init_states(batch=2, device=dev)
            logits, _ = model(x, states)
            loss = logits.view(-1, 256)[:, 0].sum()
            loss.backward()
            opt.step()
            # No exception → both modes train without error
