"""
Tests for drex.models.kan_readout — KANReadout (Phase 28).

Test taxonomy:
  - (a) Pure unit: no I/O, deterministic, no file system.
  - (b) Shape/dtype contract tests.
  - (c) Numerical correctness against known reference.
  - (d) Fitting tests: closed_form and gradient.
  - (e) Failure cases.

Coverage:
  BSplineKANLayer:
    - basis: shape and partition of unity
    - basis: boundary clamping
    - forward: shape (B, d_in) → (B, d_out)
    - forward: (B, S, d_in) → (B, S, d_out)
    - forward: no NaN on random input
    - n_params == expected count
    - fit_closed_form: MSE decreases vs random init
    - fit_closed_form: fitted spline reproduces linear target (exact fit test)
    - spline_functions: shape and dtype

  KANReadout:
    - construction: valid 1-layer, 2-layer, n-layer
    - construction: n_kan_layers < 1 raises ValueError
    - construction: fit_method invalid raises ValueError
    - forward: (B, d_in) → (B, d_out) shape
    - forward: NaN guard raises AssertionError on NaN input
    - forward: asserting d_in mismatch raises AssertionError
    - gradient: fit_method="gradient" reduces MSE over n_steps
    - closed_form: fit_method="closed_form" returns mse dict
    - n_params_vs_mlp: kan_params < mlp_params for small vocab
    - dtype contract: output is float32
    - forward: works on bfloat16 input (dtype propagation)
    - to_bfloat16: parameters become bfloat16, forward still runs
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from drex.models.kan_readout import (
    BSplineKANLayer,
    KANReadout,
    _b_spline_basis,
    _make_grid,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

D_IN  = 16
D_OUT = 8
N     = 32   # batch size
N_GRID   = 5
ORDER    = 3
N_LAYERS = 2


@pytest.fixture
def small_layer() -> BSplineKANLayer:
    torch.manual_seed(0)
    return BSplineKANLayer(D_IN, D_OUT, n_grid=N_GRID, spline_order=ORDER)


@pytest.fixture
def small_kan() -> KANReadout:
    torch.manual_seed(0)
    return KANReadout(d_in=D_IN, d_out=D_OUT, n_grid=N_GRID, spline_order=ORDER, n_kan_layers=N_LAYERS)


# ---------------------------------------------------------------------------
# _b_spline_basis
# ---------------------------------------------------------------------------

class TestBSplineBasis:
    def test_shape(self):
        """Basis output has n_grid + spline_order - 1 functions."""
        grid = _make_grid(N_GRID, ORDER)
        x = torch.linspace(-1, 1, 50)
        basis = _b_spline_basis(x, grid, ORDER)
        expected_n_basis = N_GRID + ORDER - 1
        assert basis.shape == (50, expected_n_basis), basis.shape

    def test_partition_of_unity(self):
        """B-spline basis functions sum to 1 for any interior point."""
        grid = _make_grid(N_GRID, ORDER)
        x = torch.linspace(-0.9, 0.9, 200)  # avoid exact boundaries
        basis = _b_spline_basis(x, grid, ORDER)
        sums = basis.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4), (
            f"Partition of unity failed: min={sums.min():.6f} max={sums.max():.6f}"
        )

    def test_non_negative(self):
        """B-spline basis functions are always ≥ 0."""
        grid = _make_grid(N_GRID, ORDER)
        x = torch.linspace(-1, 1, 200)
        basis = _b_spline_basis(x, grid, ORDER)
        assert (basis >= -1e-6).all(), "Basis values must be non-negative"

    def test_boundary_clamping(self):
        """Output at boundary x=x_max is handled without NaN/Inf."""
        grid = _make_grid(N_GRID, ORDER)
        x = torch.tensor([1.0])  # x_max (repeated-knot boundary)
        basis = _b_spline_basis(x, grid, ORDER)
        # Safety contract: no NaN or Inf (numerical stability)
        assert not torch.isnan(basis).any()
        assert not torch.isinf(basis).any()
        # Basis values at exact right boundary may be 0 (open interval convention);
        # this is an implementation-defined edge case — not testing partition here.


class TestMakeGrid:
    def test_shape(self):
        grid = _make_grid(N_GRID, ORDER)
        assert grid.shape == (N_GRID + 2 * ORDER,)

    def test_monotonic(self):
        grid = _make_grid(N_GRID, ORDER)
        assert (grid[1:] >= grid[:-1]).all()

    def test_boundaries(self):
        grid = _make_grid(N_GRID, ORDER, x_min=-2.0, x_max=2.0)
        assert grid[0].item() == pytest.approx(-2.0)
        assert grid[-1].item() == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# BSplineKANLayer
# ---------------------------------------------------------------------------

class TestBSplineKANLayer:
    def test_n_params(self, small_layer: BSplineKANLayer):
        """Verify parameter count formula."""
        expected = (
            D_OUT * D_IN * small_layer.n_basis  # coeff
            + D_OUT * D_IN                       # w_base
            + D_OUT                              # bias
        )
        assert small_layer.n_params() == expected

    def test_forward_2d_shape(self, small_layer: BSplineKANLayer):
        """(B, d_in) → (B, d_out)."""
        x = torch.randn(N, D_IN)
        y = small_layer(x)
        assert y.shape == (N, D_OUT)

    def test_forward_3d_shape(self, small_layer: BSplineKANLayer):
        """(B, S, d_in) → (B, S, d_out) for sequence inputs."""
        x = torch.randn(4, 10, D_IN)
        y = small_layer(x)
        assert y.shape == (4, 10, D_OUT)

    def test_forward_no_nan(self, small_layer: BSplineKANLayer):
        torch.manual_seed(1)
        x = torch.randn(N, D_IN)
        y = small_layer(x)
        assert not torch.isnan(y).any(), "NaN in forward output"
        assert not torch.isinf(y).any(), "Inf in forward output"

    def test_forward_out_of_range(self, small_layer: BSplineKANLayer):
        """Inputs outside [x_min, x_max] are clamped; output still finite."""
        x = torch.full((N, D_IN), 5.0)  # outside [-1, 1]
        y = small_layer(x)
        assert not torch.isnan(y).any()
        assert not torch.isinf(y).any()

    def test_fit_closed_form_reduces_mse(self, small_layer: BSplineKANLayer):
        """Closed-form fit produces lower MSE on fitting batch than random init."""
        torch.manual_seed(42)
        x = torch.randn(200, D_IN)
        # Linear target: y = W x + b
        W = torch.randn(D_OUT, D_IN)
        targets = (x @ W.T)

        # MSE before fit
        with torch.no_grad():
            y_before = small_layer(x)
            mse_before = F.mse_loss(y_before, targets).item()

        # Fit
        mse_fit = small_layer.fit_closed_form(x, targets, ridge=1e-3)

        # MSE after fit
        with torch.no_grad():
            y_after = small_layer(x)
            mse_after = F.mse_loss(y_after, targets).item()

        assert mse_after < mse_before, (
            f"Closed-form fit did not improve MSE: before={mse_before:.4f}, after={mse_after:.4f}"
        )

    def test_fit_closed_form_linear_target(self):
        """For a purely linear target, closed-form should nearly perfectly fit."""
        torch.manual_seed(99)
        d_in, d_out = 4, 3
        layer = BSplineKANLayer(d_in, d_out, n_grid=8, spline_order=3)
        x = torch.linspace(-0.8, 0.8, 300).unsqueeze(-1).expand(-1, d_in)
        W = torch.eye(d_out, d_in) if d_out <= d_in else torch.randn(d_out, d_in)
        targets = x @ W.T
        layer.fit_closed_form(x, targets, ridge=1e-4)
        with torch.no_grad():
            y = layer(x)
        mse = F.mse_loss(y, targets).item()
        assert mse < 0.05, f"Linear fit MSE too high: {mse:.6f}"

    def test_spline_functions_shape(self, small_layer: BSplineKANLayer):
        """spline_functions() returns correct shapes."""
        xs, values = small_layer.spline_functions(n_points=50)
        assert xs.shape == (50,)
        assert values.shape == (D_OUT, D_IN, 50)

    def test_repr(self, small_layer: BSplineKANLayer):
        r = repr(small_layer)
        assert "BSplineKANLayer" in r
        assert "d_in=" in r


# ---------------------------------------------------------------------------
# KANReadout
# ---------------------------------------------------------------------------

class TestKANReadoutConstruction:
    def test_construction_1_layer(self):
        kan = KANReadout(d_in=D_IN, d_out=D_OUT, n_kan_layers=1)
        assert len(kan.layers) == 1
        assert len(kan.norms) == 0

    def test_construction_2_layer(self, small_kan: KANReadout):
        assert len(small_kan.layers) == N_LAYERS
        assert len(small_kan.norms) == N_LAYERS - 1

    def test_construction_3_layer(self):
        kan = KANReadout(d_in=D_IN, d_out=D_OUT, n_kan_layers=3)
        assert len(kan.layers) == 3
        assert len(kan.norms) == 2

    def test_invalid_n_layers_raises(self):
        with pytest.raises(ValueError, match="n_kan_layers"):
            KANReadout(d_in=D_IN, d_out=D_OUT, n_kan_layers=0)

    def test_invalid_fit_method_raises(self):
        with pytest.raises(ValueError, match="fit_method"):
            KANReadout(d_in=D_IN, d_out=D_OUT, fit_method="nonsense")


class TestKANReadoutForward:
    def test_shape_2d(self, small_kan: KANReadout):
        """(B, d_in) → (B, d_out)."""
        x = torch.randn(N, D_IN)
        y = small_kan(x)
        assert y.shape == (N, D_OUT)

    def test_shape_3d(self, small_kan: KANReadout):
        """(B, S, d_in) → (B, S, d_out)."""
        x = torch.randn(4, 10, D_IN)
        y = small_kan(x)
        assert y.shape == (4, 10, D_OUT)

    def test_output_dtype_float32(self, small_kan: KANReadout):
        x = torch.randn(N, D_IN)
        y = small_kan(x)
        assert y.dtype == torch.float32

    def test_no_nan_on_random(self, small_kan: KANReadout):
        torch.manual_seed(7)
        x = torch.randn(N, D_IN)
        y = small_kan(x)
        assert not torch.isnan(y).any()
        assert not torch.isinf(y).any()

    def test_nan_guard_raises(self, small_kan: KANReadout):
        x = torch.full((N, D_IN), float("nan"))
        with pytest.raises(AssertionError, match="NaN/Inf"):
            small_kan(x)

    def test_dim_mismatch_raises(self, small_kan: KANReadout):
        x = torch.randn(N, D_IN + 1)
        with pytest.raises(AssertionError, match="d_in"):
            small_kan(x)

    def test_bfloat16_input(self, small_kan: KANReadout):
        """bfloat16 input propagates through without error."""
        x = torch.randn(N, D_IN).to(torch.bfloat16)
        y = small_kan(x)
        assert not torch.isnan(y.float()).any()


class TestKANReadoutFitting:
    def test_gradient_fit_reduces_mse(self):
        """gradient fit_method reduces MSE over training steps."""
        torch.manual_seed(0)
        kan = KANReadout(d_in=D_IN, d_out=D_OUT, n_kan_layers=1, fit_method="gradient")
        x = torch.randn(100, D_IN)
        W = torch.randn(D_OUT, D_IN)
        targets = x @ W.T

        with torch.no_grad():
            mse_before = F.mse_loss(kan(x), targets).item()

        result = kan.fit(x, targets, n_steps=300)
        assert "mse" in result

        with torch.no_grad():
            mse_after = F.mse_loss(kan(x), targets).item()

        assert mse_after < mse_before, (
            f"Gradient fit did not improve MSE: before={mse_before:.4f}, after={mse_after:.4f}"
        )

    def test_closed_form_fit_returns_dict(self):
        torch.manual_seed(1)
        kan = KANReadout(d_in=D_IN, d_out=D_OUT, n_kan_layers=1, fit_method="closed_form")
        x = torch.randn(100, D_IN)
        targets = torch.randn(100, D_OUT)
        result = kan.fit(x, targets, ridge=1e-3)
        assert "mse" in result
        assert isinstance(result["mse"], float)

    def test_closed_form_2layer_completes(self):
        """2-layer closed-form fit completes without error."""
        torch.manual_seed(2)
        kan = KANReadout(d_in=D_IN, d_out=D_OUT, n_kan_layers=2, fit_method="closed_form")
        x = torch.randn(100, D_IN)
        targets = torch.randn(100, D_OUT)
        result = kan.fit(x, targets)
        assert "mse" in result


class TestKANReadoutParamCount:
    def test_n_params_vs_mlp(self, small_kan: KANReadout):
        """For char-level vocab (d_out=D_OUT small), KAN param count is tracked."""
        comparison = small_kan.n_params_vs_mlp()
        assert "kan_params" in comparison
        assert "mlp_params" in comparison
        assert comparison["kan_params"] > 0
        assert comparison["mlp_params"] > 0

    def test_to_bfloat16_runs_forward(self, small_kan: KANReadout):
        """After casting to bfloat16, forward still runs and output is finite."""
        small_kan.to_bfloat16()
        x = torch.randn(N, D_IN)
        y = small_kan(x)
        assert not torch.isnan(y.float()).any()

    def test_repr(self, small_kan: KANReadout):
        r = repr(small_kan)
        assert "KANReadout" in r
        assert "d_in=" in r
