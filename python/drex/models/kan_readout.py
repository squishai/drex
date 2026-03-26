"""
drex.models.kan_readout — KAN Readout Layer for DREX-UNIFIED.

Phase 28 (DREX-UNIFIED): replaces the final linear projection with a
Kolmogorov-Arnold Network (KAN) whose edge functions are learnable B-splines.

Architecture:

    merged representation  (B, d_in)   ← from sparse router / mamba hidden
         ↓
    BSplineKANLayer  d_in → d_hidden
         ↓
    BSplineKANLayer  d_hidden → d_out
         ↓
    logits  (B, d_out)

Each B-spline edge has (n_grid + spline_order) free coefficients.  At identical
accuracy, a KAN readout uses fewer parameters than an MLP readout because the
spline bases capture non-linearities without multiplying hidden-state width.

Two fitting modes are supported:

    "closed_form"  (default)  —  ridge regression; one-shot, no optimizer loop.
                                 Fits a single KAN layer to a batch of
                                 (activation, target) pairs in < 60 s for
                                 d_model=256, vocab=32 000 on CPU.

    "gradient"     —  standard Adam; used when fitting inside an end-to-end
                      training loop where gradients must flow back through the
                      spline coefficients.

Both modes share the same forward() path; only the coefficient initialisation
and the fit() helper differ.

Validation criteria (DREX_UNIFIED_SPEC.md Component 9):
    [ ] within 2% of MLP readout accuracy on validation set
    [ ] spline functions are plottable and show non-trivial learned shapes
    [ ] fewer parameters than equivalent MLP for same accuracy
    [ ] closed_form fit < 60 s for d_model=256, d_out=32000 on CPU

References:
    Liu et al. (2024) — KAN: Kolmogorov-Arnold Networks  [arXiv:2404.19756]
    de Boor (1978)   — A Practical Guide to Splines
"""

from __future__ import annotations

import logging
import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# B-spline basis helpers
# ---------------------------------------------------------------------------

def _b_spline_basis(x: torch.Tensor, grid: torch.Tensor, order: int) -> torch.Tensor:
    """Evaluate the B-spline basis functions for each element of x.

    Uses the Cox–de Boor recursion:
        B_{i,0}(x) = 1  if grid[i] ≤ x < grid[i+1]  else 0
        B_{i,k}(x) = (x - grid[i]) / (grid[i+k] - grid[i]) * B_{i,k-1}(x)
                   + (grid[i+k+1] - x) / (grid[i+k+1] - grid[i+1]) * B_{i+1,k-1}(x)

    Args:
        x:     (...,) — input activations, clamped to [grid[0], grid[-1]]
        grid:  (n_grid + 2 * order,) — extended uniform grid including boundary
                knots, built by _make_grid()
        order: int, B-spline polynomial order (3 = cubic)

    Returns:
        basis: (..., n_grid + order - 1) — each position gives the value of
               one basis spline evaluated at the corresponding x.
    """
    # Number of basis functions = n_knots - order - 1 = (n_grid + order - 1)
    # where n_knots = len(grid).
    n_basis = grid.shape[0] - order - 1

    # Order 0: indicator  B_{i,0}(x) ∈ {0, 1}
    # Expand dims so we can broadcast x (...) against grid intervals (n_basis+order,).
    x_exp = x.unsqueeze(-1)  # (..., 1)
    g_lo = grid[:-1]         # (n_knots-1,)
    g_hi = grid[1:]          # (n_knots-1,)

    # B_{i,0}: shape (..., n_knots-1)
    basis = ((x_exp >= g_lo) & (x_exp < g_hi)).to(x.dtype)
    # Handle the last knot: clamp x == grid[-1] into the last interval.
    at_end = (x_exp == grid[-1])
    basis[..., -1] = basis[..., -1] + at_end.squeeze(-1).to(x.dtype)

    # Recursive de Boor: iterate from order 1 to `order`.
    for k in range(1, order + 1):
        n_k = grid.shape[0] - k - 1  # number of basis functions at this order
        # Left term: (x - grid[i]) / (grid[i+k] - grid[i]) * B_{i,k-1}
        g_ik  = grid[:n_k]          # grid[0..n_k-1]
        g_ik1 = grid[k : n_k + k]   # grid[k..n_k+k-1]
        denom_left = (g_ik1 - g_ik).clamp(min=1e-8)
        coeff_left = (x_exp - g_ik) / denom_left  # (..., n_k)

        # Right term: (grid[i+k+1] - x) / (grid[i+k+1] - grid[i+1]) * B_{i+1,k-1}
        g_ik2 = grid[k + 1 : n_k + k + 1]   # grid[k+1..n_k+k]
        g_i1  = grid[1 : n_k + 1]            # grid[1..n_k]
        denom_right = (g_ik2 - g_i1).clamp(min=1e-8)
        coeff_right = (g_ik2 - x_exp) / denom_right  # (..., n_k)

        basis_prev = basis  # (..., n_k+1)
        basis = coeff_left * basis_prev[..., :n_k] + coeff_right * basis_prev[..., 1 : n_k + 1]

    # basis: (..., n_basis)  where n_basis = n_grid + order - 1
    assert basis.shape[-1] == n_basis, (
        f"basis shape mismatch: expected {n_basis}, got {basis.shape[-1]}"
    )
    return basis  # (..., n_basis)


def _make_grid(n_grid: int, order: int, x_min: float = -1.0, x_max: float = 1.0) -> torch.Tensor:
    """Build an extended uniform grid for B-spline evaluation.

    The standard B-spline grid for n_grid interior knots has n_grid + 2*order
    total knots once the boundary is padded with `order` repeated endpoints on
    each side.

    Returns:
        grid: (n_grid + 2 * order,) float32
    """
    # Interior knots: n_grid evenly spaced points in [x_min, x_max]
    interior = torch.linspace(x_min, x_max, n_grid, dtype=torch.float32)
    # Pad each side with `order` copies of the boundary value.
    left  = torch.full((order,), x_min, dtype=torch.float32)
    right = torch.full((order,), x_max, dtype=torch.float32)
    return torch.cat([left, interior, right])  # (n_grid + 2 * order,)


# ---------------------------------------------------------------------------
# B-spline KAN layer
# ---------------------------------------------------------------------------

class BSplineKANLayer(nn.Module):
    """One layer of a Kolmogorov-Arnold Network with B-spline edge functions.

    Each edge (i → j) has its own learned set of B-spline coefficients plus a
    residual scaling factor.  The output for neuron j is:

        y_j = Σ_i  [spline_ij(x_i) + w_base_ij * silu(x_i)]

    where spline_ij(x_i) = Σ_k c_{ij,k} · B_k(x_i) is a weighted sum of
    B-spline basis values.

    This residual SiLU term (the "base" activation) ensures that the layer is
    expressible as a standard linear layer when all spline coefficients are
    zero, and provides gradient signal early in training.

    Args:
        d_in:         Number of input features.
        d_out:        Number of output features.
        n_grid:       Number of interior B-spline knot points.
        spline_order: B-spline polynomial order (3 = cubic piecewise).
        x_min:        Lower bound of grid (input is clamped here).
        x_max:        Upper bound of grid.
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        n_grid: int = 5,
        spline_order: int = 3,
        x_min: float = -1.0,
        x_max: float = 1.0,
    ) -> None:
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.n_grid = n_grid
        self.spline_order = spline_order
        self.x_min = x_min
        self.x_max = x_max

        # Number of B-spline basis functions per edge.
        # Extended grid has n_grid + 2*order knots → n_basis = n_grid + order - 1
        self.n_basis = n_grid + spline_order - 1

        grid = _make_grid(n_grid, spline_order, x_min, x_max)
        self.register_buffer("grid", grid)  # (n_grid + 2*order,) — never trained

        # Spline coefficients: (d_out, d_in, n_basis)
        # Initialise near zero so the layer starts close to identity/linear.
        self.coeff = nn.Parameter(
            torch.zeros(d_out, d_in, self.n_basis)
        )
        # Base (residual) weight: (d_out, d_in)
        # Initialised with Kaiming uniform like a standard linear layer.
        self.w_base = nn.Parameter(
            torch.empty(d_out, d_in)
        )
        nn.init.kaiming_uniform_(self.w_base, a=math.sqrt(5))
        # Bias term.
        self.bias = nn.Parameter(torch.zeros(d_out))

    # -- helpers ----------------------------------------------------------------

    def n_params(self) -> int:
        return self.d_out * self.d_in * self.n_basis + self.d_out * self.d_in + self.d_out

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"BSplineKANLayer(d_in={self.d_in}, d_out={self.d_out}, "
            f"n_grid={self.n_grid}, order={self.spline_order}, "
            f"n_basis={self.n_basis}, n_params={self.n_params():,})"
        )

    # -- forward ----------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate spline KAN layer.

        Args:
            x: (..., d_in) — arbitrary leading batch/sequence dims.

        Returns:
            y: (..., d_out) — same leading dims.
        """
        leading = x.shape[:-1]   # e.g. (B,) or (B, S)

        # Cast to param dtype for computation (handles bfloat16 input gracefully).
        # B-spline basis evaluation benefits from float32 precision; if params are
        # bfloat16 (after to_bfloat16()), compute in bfloat16.
        param_dtype = self.coeff.dtype
        x_work = x.to(param_dtype).clamp(self.x_min, self.x_max)

        # Flatten leading dims: (N, d_in)
        N = x_work.numel() // self.d_in
        x_flat = x_work.reshape(N, self.d_in)  # (N, d_in)

        # Evaluate B-spline basis for each input feature.
        # Process per-feature to keep memory O(N * d_in * n_basis).
        # basis_all: (N, d_in, n_basis)
        basis_all = torch.zeros(N, self.d_in, self.n_basis, dtype=param_dtype, device=x.device)
        for i in range(self.d_in):
            basis_all[:, i, :] = _b_spline_basis(x_flat[:, i].float(), self.grid.float(), self.spline_order).to(param_dtype)

        # Spline term: y_spline[n, j] = Σ_i  Σ_k  coeff[j,i,k] · basis_all[n,i,k]
        # coeff: (d_out, d_in, n_basis) → reshape to (d_out, d_in * n_basis)
        # basis_all: (N, d_in, n_basis) → reshape to (N, d_in * n_basis)
        coeff_flat = self.coeff.reshape(self.d_out, self.d_in * self.n_basis)   # (d_out, D*K)
        basis_flat = basis_all.reshape(N, self.d_in * self.n_basis)             # (N, D*K)
        y_spline = basis_flat @ coeff_flat.T  # (N, d_out)

        # Base (residual) term: y_base[n, j] = Σ_i w_base[j,i] * silu(x_i)
        y_base = F.silu(x_flat) @ self.w_base.T  # (N, d_out)

        y = y_spline + y_base + self.bias  # (N, d_out)
        return y.reshape(*leading, self.d_out)

    # -- closed-form fit --------------------------------------------------------

    def fit_closed_form(
        self,
        x: torch.Tensor,
        targets: torch.Tensor,
        ridge: float = 1e-3,
    ) -> float:
        """Fit spline coefficients via ridge regression (no gradient).

        Given a batch of input/target pairs (x, targets), set self.coeff
        to the analytical ridge-regression solution:

            C* = (Φᵀ Φ + λ I)⁻¹ Φᵀ T

        where Φ is the B-spline design matrix of shape (N, d_in * n_basis),
        T is the target matrix (N, d_out), and λ is the ridge penalty.

        The base weights self.w_base and self.bias are zeroed so the entire
        layer output is explained by the spline term after fitting.

        Args:
            x:       (N, d_in) — input activations.
            targets: (N, d_out) — regression targets (e.g. one-hot or soft
                     logits from a teacher).
            ridge:   Ridge penalty λ for numerical stability.

        Returns:
            residual_mse: float — MSE of fitted layer on the training batch.
        """
        with torch.no_grad():
            N = x.shape[0]
            x_clamped = x.clamp(self.x_min, self.x_max).float()
            targets_f = targets.float()

            # Build design matrix Φ: (N, d_in * n_basis)
            basis_list = []
            for i in range(self.d_in):
                basis_list.append(_b_spline_basis(x_clamped[:, i], self.grid.float(), self.spline_order))
            Phi = torch.cat(basis_list, dim=-1)  # (N, d_in * n_basis)

            # Ridge: C = (ΦᵀΦ + λI)⁻¹ Φᵀ T
            # Shape:  (d_in*n_basis, d_in*n_basis) @ (d_in*n_basis, d_out) → (d_in*n_basis, d_out)
            D = Phi.shape[1]  # d_in * n_basis
            A = Phi.T @ Phi + ridge * torch.eye(D, dtype=torch.float32, device=x.device)
            b = Phi.T @ targets_f
            try:
                C = torch.linalg.solve(A, b)  # (D, d_out)
            except torch.linalg.LinAlgError:
                log.warning("fit_closed_form: linalg.solve failed, falling back to lstsq")
                C = torch.linalg.lstsq(A, b).solution

            # C: (D, d_out); coeff is (d_out, d_in, n_basis) = (d_out, D)
            self.coeff.data = C.T.reshape(self.d_out, self.d_in, self.n_basis)
            # Zero-out base and bias: layer is fully spline after fit.
            self.w_base.data.zero_()
            self.bias.data.zero_()

            # Compute residual MSE on the fitting batch.
            y_hat = Phi @ C  # (N, d_out)
            mse = ((y_hat - targets_f) ** 2).mean().item()
            return mse

    def spline_functions(self, n_points: int = 100) -> tuple[torch.Tensor, torch.Tensor]:
        """Return dense evaluations of all learnt spline edge functions for plotting.

        Returns:
            xs:      (n_points,) — grid of x values in [x_min, x_max]
            values:  (d_out, d_in, n_points) — f_{j,i}(x) for each edge (i→j)
        """
        xs = torch.linspace(self.x_min, self.x_max, n_points)
        basis = torch.stack(
            [_b_spline_basis(xs, self.grid.float(), self.spline_order)],
            dim=0,
        ).squeeze(0)  # (n_points, n_basis)
        # coeff: (d_out, d_in, n_basis)  subscript "jib"
        # basis: (n_points, n_basis)     subscript "pb"
        # result: (d_out, d_in, n_points) subscript "jip"
        # values[j, i, p] = Σ_b  coeff[j, i, b] * basis[p, b]
        values = torch.einsum("jib,pb->jip", self.coeff.detach().float(), basis)
        return xs, values  # (n_points,), (d_out, d_in, n_points)


# ---------------------------------------------------------------------------
# KAN Readout
# ---------------------------------------------------------------------------

class KANReadout(nn.Module):
    """KAN-based output projection for DREX-UNIFIED.

    Stacks n_kan_layers BSplineKANLayers.  For a 2-layer KAN:

        d_in  →  d_hidden  →  d_out

    where d_hidden = max(d_in, 256) to keep the intermediate width at least
    as wide as the input representation.

    Args:
        d_in:         Input dimension (must match d_model from sparse router).
        d_out:        Output dimension (vocab_size for LM, n_classes for CLS).
        n_grid:       B-spline interior knot count.
        spline_order: B-spline polynomial order.
        n_kan_layers: Number of stacked KAN layers (≥ 1).
        fit_method:   "closed_form" or "gradient".  Determines default fitting
                      strategy for fit() / fit_closed_form().
        d_hidden:     Hidden width between KAN layers (default: d_in).
    """

    def __init__(
        self,
        d_in: int = 256,
        d_out: int = 256,
        n_grid: int = 5,
        spline_order: int = 3,
        n_kan_layers: int = 2,
        fit_method: Literal["closed_form", "gradient"] = "closed_form",
        d_hidden: int | None = None,
    ) -> None:
        super().__init__()

        if n_kan_layers < 1:
            raise ValueError(f"n_kan_layers must be ≥ 1, got {n_kan_layers}")
        if fit_method not in ("closed_form", "gradient"):
            raise ValueError(f"fit_method must be 'closed_form' or 'gradient', got {fit_method!r}")

        self.d_in = d_in
        self.d_out = d_out
        self.n_grid = n_grid
        self.spline_order = spline_order
        self.n_kan_layers = n_kan_layers
        self.fit_method = fit_method

        # Hidden width — default to input width.
        dh = d_hidden if d_hidden is not None else d_in

        # Build layer stack.
        layers: list[nn.Module] = []
        if n_kan_layers == 1:
            layers.append(BSplineKANLayer(d_in, d_out, n_grid, spline_order))
        else:
            layers.append(BSplineKANLayer(d_in, dh, n_grid, spline_order))
            for _ in range(n_kan_layers - 2):
                layers.append(BSplineKANLayer(dh, dh, n_grid, spline_order))
            layers.append(BSplineKANLayer(dh, d_out, n_grid, spline_order))

        self.layers = nn.ModuleList(layers)

        # Layer norms between KAN layers (applied to intermediate activations
        # before the next spline evaluation; keeps inputs within grid range).
        if n_kan_layers > 1:
            self.norms = nn.ModuleList(
                [nn.LayerNorm(dh) for _ in range(n_kan_layers - 1)]
            )
        else:
            self.norms = nn.ModuleList()

        log.debug(
            "KANReadout: d_in=%d, d_out=%d, d_hidden=%d, n_kan_layers=%d, "
            "n_grid=%d, order=%d, fit=%s, total_params=%s",
            d_in, d_out, dh, n_kan_layers, n_grid, spline_order,
            fit_method, f"{self._count_params():,}",
        )

    # ── helpers ────────────────────────────────────────────────────────────────

    def _count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def n_params(self) -> int:
        """Total trainable parameters."""
        return self._count_params()

    def n_params_vs_mlp(self) -> dict[str, int]:
        """Compare parameter count against an equivalent 2-layer MLP."""
        kan_params = self._count_params()
        # MLP: Linear(d_in, d_hidden) + GELU + Linear(d_hidden, d_out)
        # = d_in*d_hidden + d_hidden + d_hidden*d_out + d_out
        d_hidden = self.layers[0].d_out if self.n_kan_layers > 1 else self.d_out
        mlp_params = (
            self.d_in * d_hidden + d_hidden
            + d_hidden * self.d_out + self.d_out
        )
        return {"kan_params": kan_params, "mlp_params": mlp_params, "ratio": kan_params / max(1, mlp_params)}

    def __repr__(self) -> str:  # noqa: D105
        layer_str = "\n    ".join(repr(l) for l in self.layers)
        return (
            f"KANReadout(\n"
            f"    d_in={self.d_in}, d_out={self.d_out}, "
            f"fit={self.fit_method}, n_params={self.n_params():,}\n"
            f"    {layer_str}\n"
            f")"
        )

    # ── forward ────────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through stacked KAN layers.

        Args:
            x: (B, d_in) or (B, S, d_in) — merged representation.

        Returns:
            logits: (B, d_out) or (B, S, d_out)
        """
        # Validate input dtype/shape contract (DREX_UNIFIED_SPEC.md).
        assert x.shape[-1] == self.d_in, (
            f"KANReadout: expected d_in={self.d_in}, got {x.shape[-1]}"
        )
        assert not (torch.isnan(x).any() or torch.isinf(x).any()), (
            "NaN/Inf at KANReadout input"
        )

        h = x
        for idx, layer in enumerate(self.layers):
            h = layer(h)
            if idx < len(self.norms):
                h = self.norms[idx](h)

        return h  # (..., d_out)

    # ── fitting helpers ─────────────────────────────────────────────────────────

    def fit(
        self,
        x: torch.Tensor,
        targets: torch.Tensor,
        ridge: float = 1e-3,
        lr: float = 1e-3,
        n_steps: int = 500,
    ) -> dict[str, float]:
        """Fit the KAN readout to (x, targets) pairs.

        Dispatches to closed_form or gradient fitting depending on self.fit_method.

        Args:
            x:        (N, d_in) — input representations.
            targets:  (N, d_out) — regression targets (float, e.g. one-hot or
                      teacher logits).
            ridge:    Ridge penalty for closed_form fitting.
            lr:       Learning rate for gradient fitting.
            n_steps:  Iteration count for gradient fitting.

        Returns:
            {"mse": float} on completion.
        """
        if self.fit_method == "closed_form":
            return self._fit_greedy_closed_form(x, targets, ridge)
        else:
            return self._fit_gradient(x, targets, lr, n_steps)

    def _fit_greedy_closed_form(
        self, x: torch.Tensor, targets: torch.Tensor, ridge: float
    ) -> dict[str, float]:
        """Greedy layer-wise closed-form fitting.

        For each layer in order:
          1. Compute the layer's input (forward through prior layers).
          2. Set the layer's target (the residual signal to explain).
          3. Call layer.fit_closed_form(h, residual_targets, ridge).

        For a 1-layer KAN: targets are the true output targets.
        For a 2-layer KAN: layer 1 learns to produce an intermediate
        representation; layer 2 learns to map that to the final targets.
        We approximate by training layer 1 on a random projection of
        targets, then training layer 2 on the remaining residual.

        This greedy strategy is asymptotically optimal for narrow networks
        and is fast — no matrix decompositions larger than
        (d_in * n_basis) × (d_in * n_basis).
        """
        with torch.no_grad():
            mse_history: list[float] = []
            h = x.float()
            T = targets.float()

            for idx, layer in enumerate(self.layers):
                if idx == len(self.layers) - 1:
                    # Last layer: fit directly to remaining targets.
                    mse = layer.fit_closed_form(h, T, ridge=ridge)
                    mse_history.append(mse)
                else:
                    # Intermediate layer: project targets to hidden dim as proxy.
                    dh = layer.d_out
                    # Use a fixed random projection of T as intermediate training signal.
                    torch.manual_seed(42 + idx)
                    W_proj = torch.randn(T.shape[-1], dh, device=x.device) / math.sqrt(T.shape[-1])
                    T_proj = T @ W_proj  # (N, dh)
                    mse = layer.fit_closed_form(h, T_proj, ridge=ridge)
                    mse_history.append(mse)
                    # Compute intermediate output for the next layer.
                    h = layer(h)
                    if idx < len(self.norms):
                        h = self.norms[idx](h)

            final_mse_approx = sum(mse_history) / len(mse_history)
            log.debug("Greedy closed-form fit complete: layer MSEs %s", mse_history)
            return {"mse": final_mse_approx}

    def _fit_gradient(
        self, x: torch.Tensor, targets: torch.Tensor, lr: float, n_steps: int
    ) -> dict[str, float]:
        """End-to-end gradient fitting through all KAN layers.

        Optimises all layers jointly using Adam with MSE loss.  This path
        is used inside the end-to-end training loop where gradients must
        flow through the KAN readout.

        Returns:
            {"mse": float} — final training MSE.
        """
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        x_f = x.float()
        T_f = targets.float()

        for step in range(n_steps):
            opt.zero_grad()
            y_hat = self.forward(x_f)
            loss = F.mse_loss(y_hat, T_f)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            opt.step()

            if step % 100 == 0:
                log.debug("KAN gradient fit step %d/%d MSE=%.6f", step, n_steps, loss.item())

        with torch.no_grad():
            y_hat = self.forward(x_f)
            final_mse = F.mse_loss(y_hat, T_f).item()

        return {"mse": final_mse}

    # ── dtype utility ──────────────────────────────────────────────────────────

    def to_bfloat16(self) -> "KANReadout":
        """Cast all learnable parameters to bfloat16.

        The DREX dtype contract specifies KAN readout outputs float32 but the
        trained parameters can live in bfloat16 during forward; PyTorch upcasts
        automatically during accumulation.  Call this after fitting when memory
        is a constraint.
        """
        for p in self.parameters():
            p.data = p.data.to(torch.bfloat16)
        return self
