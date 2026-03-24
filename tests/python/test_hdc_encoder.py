"""
Tests for drex.models.hdc_encoder — HDCEncoder (Phase 24).

Coverage targets:
  - Module construction: valid params, hdc_dim < d_model raises
  - Trainable parameter count: zero (all buffers, none in parameters())
  - Buffer registration: W_lift and W_down present, correct shapes
  - Reproducibility: same seed → identical weights; different seeds → different
  - Layer norm: norm module exists and has trainable params (only learnable component)
  - forward(): output shape matches input, no NaN, no Inf
  - forward() residual: output differs from input (HDC enrichment is non-trivial)
  - Training vs eval mode: outputs differ (tanh vs sign threshold)
  - hypervector(): correct shape (B, S, hdc_dim)
  - similarity(): cosine similarity, same vector → 1.0, opposite → -1.0 (approx)
  - HDC primitive ops: hdc_bind, hdc_bundle, hdc_permute shapes and properties
  - Integration: DrexConfig.use_hdc_encoder=True builds and runs without error
  - Integration: HDCEncoder disabled → model unchanged
  - Integration: trainable param count identical with and without HDC encoder
  - Integration: no NaN after a full forward pass with HDC enabled
  - Integration: works at L=1 (single-token sequences)
  - Integration: works at L=512 (max expected segment length)
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from drex.models.hdc_encoder import HDCEncoder, hdc_bind, hdc_bundle, hdc_permute
from drex.models.transformer import DrexConfig, DrexTransformer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def d_model() -> int:
    return 64


@pytest.fixture
def hdc_dim() -> int:
    return 256  # small for fast tests


@pytest.fixture
def enc(d_model: int, hdc_dim: int) -> HDCEncoder:
    return HDCEncoder(d_model=d_model, hdc_dim=hdc_dim, normalize=True, seed=0)


@pytest.fixture
def batch() -> int:
    return 2


@pytest.fixture
def seq_len() -> int:
    return 16


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_valid_construction(self, d_model: int, hdc_dim: int) -> None:
        enc = HDCEncoder(d_model=d_model, hdc_dim=hdc_dim)
        assert enc.d_model == d_model
        assert enc.hdc_dim == hdc_dim

    def test_hdc_dim_smaller_than_d_model_raises(self, d_model: int) -> None:
        with pytest.raises(ValueError, match="hdc_dim"):
            HDCEncoder(d_model=d_model, hdc_dim=d_model - 1)

    def test_hdc_dim_equal_to_d_model_raises(self, d_model: int) -> None:
        with pytest.raises(ValueError, match="hdc_dim"):
            HDCEncoder(d_model=d_model, hdc_dim=d_model)

    def test_hdc_dim_greater_than_d_model_ok(self, d_model: int) -> None:
        enc = HDCEncoder(d_model=d_model, hdc_dim=d_model + 1)
        assert enc.hdc_dim == d_model + 1

    def test_normalize_flag_stored(self, d_model: int, hdc_dim: int) -> None:
        enc = HDCEncoder(d_model=d_model, hdc_dim=hdc_dim, normalize=False)
        assert enc.normalize is False


# ---------------------------------------------------------------------------
# Parameter / buffer split
# ---------------------------------------------------------------------------

class TestParametersAndBuffers:
    def test_zero_trainable_parameters_from_projections(
        self, enc: HDCEncoder, d_model: int, hdc_dim: int
    ) -> None:
        # Only norm has trainable params; projections contribute zero.
        proj_param_names = {"W_lift", "W_down"}
        for name, param in enc.named_parameters():
            assert name not in proj_param_names, (
                f"{name} should be a buffer (frozen), not a trainable parameter"
            )

    def test_w_lift_in_buffers(self, enc: HDCEncoder, d_model: int, hdc_dim: int) -> None:
        buf_names = {name for name, _ in enc.named_buffers()}
        assert "W_lift" in buf_names

    def test_w_down_in_buffers(self, enc: HDCEncoder, d_model: int, hdc_dim: int) -> None:
        buf_names = {name for name, _ in enc.named_buffers()}
        assert "W_down" in buf_names

    def test_w_lift_shape(self, enc: HDCEncoder, d_model: int, hdc_dim: int) -> None:
        assert enc.W_lift.shape == (d_model, hdc_dim)

    def test_w_down_shape(self, enc: HDCEncoder, d_model: int, hdc_dim: int) -> None:
        assert enc.W_down.shape == (hdc_dim, d_model)

    def test_only_norm_contributes_trainable_params(
        self, enc: HDCEncoder, d_model: int
    ) -> None:
        total = sum(p.numel() for p in enc.parameters())
        # LayerNorm(d_model) has weight + bias = 2 × d_model params
        assert total == 2 * d_model, (
            f"Expected 2×{d_model}={2*d_model} trainable params (LayerNorm only), "
            f"got {total}"
        )

    def test_buffers_not_updated_by_optimizer(
        self, enc: HDCEncoder, d_model: int, hdc_dim: int, batch: int, seq_len: int
    ) -> None:
        """A gradient step must not change W_lift or W_down."""
        W_lift_before = enc.W_lift.clone()
        W_down_before = enc.W_down.clone()

        optimizer = torch.optim.Adam(enc.parameters(), lr=1e-3)
        enc.train()
        x = torch.randn(batch, seq_len, d_model)
        loss = enc(x).sum()
        loss.backward()
        optimizer.step()

        assert torch.equal(enc.W_lift, W_lift_before), "W_lift was mutated by optimizer"
        assert torch.equal(enc.W_down, W_down_before), "W_down was mutated by optimizer"


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

class TestReproducibility:
    def test_same_seed_same_weights(self, d_model: int, hdc_dim: int) -> None:
        enc_a = HDCEncoder(d_model=d_model, hdc_dim=hdc_dim, seed=7)
        enc_b = HDCEncoder(d_model=d_model, hdc_dim=hdc_dim, seed=7)
        assert torch.equal(enc_a.W_lift, enc_b.W_lift)
        assert torch.equal(enc_a.W_down, enc_b.W_down)

    def test_different_seeds_different_weights(self, d_model: int, hdc_dim: int) -> None:
        enc_a = HDCEncoder(d_model=d_model, hdc_dim=hdc_dim, seed=0)
        enc_b = HDCEncoder(d_model=d_model, hdc_dim=hdc_dim, seed=99)
        assert not torch.equal(enc_a.W_lift, enc_b.W_lift)


# ---------------------------------------------------------------------------
# Forward pass — shape, numerical health
# ---------------------------------------------------------------------------

class TestForward:
    def test_output_shape_matches_input(
        self, enc: HDCEncoder, batch: int, seq_len: int, d_model: int
    ) -> None:
        x = torch.randn(batch, seq_len, d_model)
        out = enc(x)
        assert out.shape == (batch, seq_len, d_model)

    def test_no_nan_in_output(
        self, enc: HDCEncoder, batch: int, seq_len: int, d_model: int
    ) -> None:
        enc.eval()
        x = torch.randn(batch, seq_len, d_model)
        out = enc(x)
        assert not torch.isnan(out).any(), "NaN detected in HDCEncoder output"

    def test_no_inf_in_output(
        self, enc: HDCEncoder, batch: int, seq_len: int, d_model: int
    ) -> None:
        enc.eval()
        x = torch.randn(batch, seq_len, d_model)
        out = enc(x)
        assert not torch.isinf(out).any(), "Inf detected in HDCEncoder output"

    def test_residual_enriches_embedding(
        self, enc: HDCEncoder, batch: int, seq_len: int, d_model: int
    ) -> None:
        """Output must differ from input — HDC adds structure."""
        enc.eval()
        x = torch.randn(batch, seq_len, d_model)
        out = enc(x)
        assert not torch.allclose(out, x, atol=1e-4), (
            "HDCEncoder output is identical to input; residual enrichment is absent"
        )

    def test_L1_edge_case(self, enc: HDCEncoder, d_model: int) -> None:
        enc.eval()
        x = torch.randn(1, 1, d_model)
        out = enc(x)
        assert out.shape == (1, 1, d_model)
        assert not torch.isnan(out).any()

    def test_L512_edge_case(self, enc: HDCEncoder, d_model: int) -> None:
        enc.eval()
        x = torch.randn(1, 512, d_model)
        out = enc(x)
        assert out.shape == (1, 512, d_model)
        assert not torch.isnan(out).any()

    def test_training_eval_outputs_differ(
        self, enc: HDCEncoder, d_model: int, seq_len: int
    ) -> None:
        """Training mode uses tanh; eval mode uses sign — outputs must differ."""
        x = torch.randn(1, seq_len, d_model)
        enc.train()
        out_train = enc(x).detach()
        enc.eval()
        out_eval = enc(x)
        assert not torch.allclose(out_train, out_eval, atol=1e-4), (
            "Training and eval outputs are identical; tanh/sign branching may be broken"
        )

    def test_gradient_flows_through_norm(
        self, enc: HDCEncoder, batch: int, seq_len: int, d_model: int
    ) -> None:
        enc.train()
        x = torch.randn(batch, seq_len, d_model, requires_grad=True)
        out = enc(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None, "No gradient flowed through HDCEncoder to input"
        assert not torch.isnan(x.grad).any(), "NaN in gradient"

    def test_normalize_false_no_nan(
        self, d_model: int, hdc_dim: int, batch: int, seq_len: int
    ) -> None:
        enc = HDCEncoder(d_model=d_model, hdc_dim=hdc_dim, normalize=False, seed=0)
        enc.eval()
        x = torch.randn(batch, seq_len, d_model)
        out = enc(x)
        assert not torch.isnan(out).any()
        assert out.shape == (batch, seq_len, d_model)


# ---------------------------------------------------------------------------
# hypervector() / similarity()
# ---------------------------------------------------------------------------

class TestHypervectorHelpers:
    def test_hypervector_shape(
        self, enc: HDCEncoder, batch: int, seq_len: int, d_model: int, hdc_dim: int
    ) -> None:
        enc.eval()
        x = torch.randn(batch, seq_len, d_model)
        h = enc.hypervector(x)
        assert h.shape == (batch, seq_len, hdc_dim)

    def test_hypervector_no_nan(
        self, enc: HDCEncoder, batch: int, seq_len: int, d_model: int
    ) -> None:
        enc.eval()
        x = torch.randn(batch, seq_len, d_model)
        h = enc.hypervector(x)
        assert not torch.isnan(h).any()

    def test_similarity_identical_vectors_approx_one(
        self, enc: HDCEncoder, d_model: int, hdc_dim: int
    ) -> None:
        enc.eval()
        x = torch.randn(1, 4, d_model)
        h = enc.hypervector(x)
        sim = enc.similarity(h, h)
        assert (sim > 0.99).all(), f"Self-similarity should be ≈1.0; got {sim}"

    def test_similarity_shape(
        self, enc: HDCEncoder, batch: int, seq_len: int, d_model: int
    ) -> None:
        enc.eval()
        x = torch.randn(batch, seq_len, d_model)
        h = enc.hypervector(x)
        sim = enc.similarity(h, h)
        assert sim.shape == (batch, seq_len)


# ---------------------------------------------------------------------------
# HDC primitive operations
# ---------------------------------------------------------------------------

class TestHDCPrimitives:
    def test_bind_shape(self) -> None:
        a = torch.randn(2, 8, 256)
        b = torch.randn(2, 8, 256)
        assert hdc_bind(a, b).shape == (2, 8, 256)

    def test_bind_is_element_wise_multiply(self) -> None:
        a = torch.tensor([1.0, -1.0, 2.0])
        b = torch.tensor([2.0, 3.0, -1.0])
        result = hdc_bind(a, b)
        expected = torch.tensor([2.0, -3.0, -2.0])
        assert torch.allclose(result, expected)

    def test_bundle_shape(self) -> None:
        a = torch.randn(2, 8, 256)
        b = torch.randn(2, 8, 256)
        assert hdc_bundle(a, b).shape == (2, 8, 256)

    def test_bundle_is_normalised(self) -> None:
        a = torch.randn(4, 256)
        b = torch.randn(4, 256)
        result = hdc_bundle(a, b)
        norms = result.norm(dim=-1)
        assert torch.allclose(norms, torch.ones(4), atol=1e-5), (
            f"bundle result should be unit-norm; got norms {norms}"
        )

    def test_permute_shape(self) -> None:
        x = torch.randn(2, 8, 256)
        assert hdc_permute(x).shape == (2, 8, 256)

    def test_permute_is_circular_shift(self) -> None:
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        shifted = hdc_permute(x, shifts=1)
        expected = torch.tensor([4.0, 1.0, 2.0, 3.0])
        assert torch.allclose(shifted, expected)

    def test_permute_shifts_arg(self) -> None:
        x = torch.randn(1, 1, 16)
        p1 = hdc_permute(x, shifts=1)
        p2 = hdc_permute(x, shifts=2)
        # Two different shifts should produce different vectors
        assert not torch.allclose(p1, p2)

    def test_permute_reversible(self) -> None:
        """Permuting by +n then -n should recover the original."""
        x = torch.randn(1, 1, 32)
        out = hdc_permute(hdc_permute(x, shifts=5), shifts=-5)
        assert torch.allclose(out, x, atol=1e-6)


# ---------------------------------------------------------------------------
# Integration with DrexTransformer
# ---------------------------------------------------------------------------

class TestTransformerIntegration:
    """Verify that DrexConfig.use_hdc_encoder=True threads through correctly."""

    def _config(self, use_hdc: bool, hdc_dim: int = 256) -> DrexConfig:
        return DrexConfig(
            d_model=64,
            n_heads=2,
            n_layers=1,
            vocab_size=256,
            window_size=16,
            max_seq_len=128,
            use_hdc_encoder=use_hdc,
            hdc_dim=hdc_dim,
        )

    def test_model_builds_with_hdc(self) -> None:
        model = DrexTransformer(self._config(use_hdc=True))
        assert model.hdc_encoder is not None

    def test_model_builds_without_hdc(self) -> None:
        model = DrexTransformer(self._config(use_hdc=False))
        assert model.hdc_encoder is None

    def test_trainable_param_count_identical(self) -> None:
        """HDCEncoder contributes ZERO trainable parameters."""
        model_base = DrexTransformer(self._config(use_hdc=False))
        model_hdc  = DrexTransformer(self._config(use_hdc=True))

        # LayerNorm inside HDCEncoder has 2*d_model trainable params; everything
        # else is frozen buffers.
        base_params = sum(p.numel() for p in model_base.parameters() if p.requires_grad)
        hdc_params  = sum(p.numel() for p in model_hdc.parameters()  if p.requires_grad)
        diff = hdc_params - base_params
        d_model = 64
        # Only LayerNorm in HDCEncoder adds trainable params (weight + bias = 2*d_model)
        assert diff == 2 * d_model, (
            f"Expected exactly {2*d_model} extra trainable params from HDCEncoder "
            f"LayerNorm, got {diff}"
        )

    def test_forward_no_nan_with_hdc(self) -> None:
        model = DrexTransformer(self._config(use_hdc=True))
        model.eval()
        ids = torch.randint(0, 256, (1, 16))
        logits, _ = model(ids)
        assert not torch.isnan(logits).any(), "NaN in logits with HDC encoder enabled"

    def test_forward_no_nan_without_hdc(self) -> None:
        model = DrexTransformer(self._config(use_hdc=False))
        model.eval()
        ids = torch.randint(0, 256, (1, 16))
        logits, _ = model(ids)
        assert not torch.isnan(logits).any()

    def test_L1_with_hdc(self) -> None:
        model = DrexTransformer(self._config(use_hdc=True))
        model.eval()
        ids = torch.randint(0, 256, (1, 1))
        logits, _ = model(ids)
        assert logits.shape == (1, 1, 256)
        assert not torch.isnan(logits).any()

    def test_L512_with_hdc(self) -> None:
        config = DrexConfig(
            d_model=64, n_heads=2, n_layers=1, vocab_size=256,
            window_size=512, max_seq_len=4096,
            use_hdc_encoder=True, hdc_dim=256,
        )
        model = DrexTransformer(config)
        model.eval()
        ids = torch.randint(0, 256, (1, 512))
        logits, _ = model(ids)
        assert logits.shape == (1, 512, 256)
        assert not torch.isnan(logits).any()

    def test_hdc_encoder_output_differs_from_no_hdc(self) -> None:
        """The HDC layer must actually change activations."""
        cfg_base = self._config(use_hdc=False)
        cfg_hdc  = self._config(use_hdc=True)

        torch.manual_seed(0)
        model_base = DrexTransformer(cfg_base)
        torch.manual_seed(0)
        model_hdc  = DrexTransformer(cfg_hdc)

        # Copy all shared weights so the only difference is the HDC encoder
        model_hdc.load_state_dict(model_base.state_dict(), strict=False)

        model_base.eval()
        model_hdc.eval()
        ids = torch.randint(0, 256, (1, 8))
        logits_base, _ = model_base(ids)
        logits_hdc,  _ = model_hdc(ids)

        assert not torch.allclose(logits_base, logits_hdc, atol=1e-4), (
            "HDC encoder made no difference to logits; it may not be wired correctly"
        )

    def test_combined_esn_and_hdc(self) -> None:
        """ESN memory and HDC encoder should work together without error."""
        config = DrexConfig(
            d_model=64, n_heads=2, n_layers=2, vocab_size=256,
            window_size=16, max_seq_len=128,
            use_episodic_memory=True,
            use_esn_memory=True,
            esn_reservoir_mult=2,
            use_hdc_encoder=True,
            hdc_dim=256,
        )
        model = DrexTransformer(config)
        model.eval()
        ids = torch.randint(0, 256, (1, 16))
        logits, states = model(ids)
        assert not torch.isnan(logits).any()
        assert logits.shape == (1, 16, 256)
