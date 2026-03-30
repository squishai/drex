"""KAN Readout — Kolmogorov-Arnold Network output projection.

Replaces a final linear projection with learnable B-spline functions on edges.
Each edge (i → j) is parameterised by a B-spline with (n_grid + spline_order)
coefficients, making the projection interpretable and parameter-efficient.

Dtype contract:
    input  : float32  (B, d_in)
    output : float32  (B, d_out)

Reference: Liu et al., "KAN: Kolmogorov-Arnold Networks" (arXiv 2404.19756)
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class BSplineKANLayer(nn.Module):
    """Single KAN layer: n_in inputs × n_out outputs, B-spline edges.

    Each edge i → j is a B-spline with (n_grid + spline_order) learnable
    coefficients stored in ``self.coeff``.

    B-spline evaluation:
        1. Extend the uniform grid to (n_grid + 2*spline_order) knot points
           using the extended-knot formula so every input maps to a valid
           de-Boor segment.
        2. Evaluate B-spline basis functions for each input value.
        3. Multiply basis by coefficients and sum to get the edge output.
        4. Sum edge outputs across input dimension → (batch, n_out).

    Parameters
    ----------
    n_in : int
        Number of input features.
    n_out : int
        Number of output features.
    n_grid : int
        Number of grid intervals. Basis has (n_grid + spline_order) functions.
    spline_order : int
        B-spline order (cubic = 3, recommended).
    """

    def __init__(
        self,
        n_in: int,
        n_out: int,
        n_grid: int = 5,
        spline_order: int = 3,
    ) -> None:
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.n_grid = n_grid
        self.spline_order = spline_order
        self.n_basis = n_grid + spline_order  # number of B-spline basis functions

        # Uniform grid over [-1, 1] (n_grid + 1 points = n_grid intervals).
        grid = torch.linspace(-1.0, 1.0, n_grid + 1)  # (n_grid+1,)
        self.register_buffer("grid", grid)

        # Extended knots for B-spline recursion: replicate boundary k times.
        h = 2.0 / n_grid
        extended = torch.cat([
            grid[0] - h * torch.arange(spline_order, 0, -1),
            grid,
            grid[-1] + h * torch.arange(1, spline_order + 1),
        ])  # (n_grid + 1 + 2*spline_order,)
        self.register_buffer("knots", extended)

        # Learnable coefficients: one set per edge.
        # Shape: (n_in, n_out, n_basis)
        self.coeff = nn.Parameter(
            torch.zeros(n_in, n_out, self.n_basis)
        )
        self._init_weights()

        # Optional base linear for residual connection (SiLU(Wx) + spline(x)).
        # Single scalar weight per i → j pair (acts as residual gate).
        self.base_weight = nn.Parameter(torch.empty(n_in, n_out))
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))

    def _init_weights(self) -> None:
        """Initialise spline coefficients with small random values."""
        nn.init.normal_(self.coeff, std=0.1)

    def _b_spline_basis(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate B-spline basis via Cox-de Boor recursion.

        Parameters
        ----------
        x : float32 Tensor, shape (batch, n_in)

        Returns
        -------
        basis : float32 Tensor, shape (batch, n_in, n_basis)
        """
        x_c = x.clamp(self.grid[0], self.grid[-1]).unsqueeze(-1)  # (B, n_in, 1)

        t = self.knots                 # (n_knots,)
        n_intervals = t.shape[0] - 1  # = n_grid + 2*spline_order

        # Order-0: indicator -- 1 iff t[i] <= x < t[i+1].
        # x_c (B, n_in, 1) broadcasts with t slices (n_intervals,).
        B = ((x_c >= t[:-1]) & (x_c < t[1:])).float()  # (B, n_in, n_intervals)

        # Cox-de Boor for orders 1 ... spline_order.
        # After j steps, B has n_intervals - j columns.
        # Final: n_intervals - spline_order = n_grid + spline_order = n_basis.
        for j in range(1, self.spline_order + 1):
            n_valid = n_intervals - j

            ti    = t[:n_valid]                # t[i]
            ti_j  = t[j:n_valid + j]           # t[i+j]
            ti1   = t[1:n_valid + 1]           # t[i+1]
            ti_j1 = t[j + 1:n_valid + j + 1]  # t[i+j+1]

            denom1 = (ti_j  - ti ).clamp(min=1e-8)
            denom2 = (ti_j1 - ti1).clamp(min=1e-8)

            alpha1 = (x_c - ti)    / denom1   # (B, n_in, n_valid)
            alpha2 = (ti_j1 - x_c) / denom2  # (B, n_in, n_valid)

            B = alpha1 * B[..., :n_valid] + alpha2 * B[..., 1:n_valid + 1]

        assert B.shape[-1] == self.n_basis, (
            f"Basis shape error: expected {self.n_basis}, got {B.shape[-1]}"
        )
        return B  # (batch, n_in, n_basis)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the KAN layer.

        Parameters
        ----------
        x : float32 Tensor, shape (B, n_in)

        Returns
        -------
        y : float32 Tensor, shape (B, n_out)
        """
        assert x.dtype == torch.float32, f"Expected float32, got {x.dtype}"
        base = F.silu(x @ self.base_weight)                           # (B, n_out)
        basis = self._b_spline_basis(x)                               # (B, n_in, n_basis)
        spline_out = torch.einsum("bir,ior->bo", basis, self.coeff)   # (B, n_out)
        return base + spline_out


class KANReadout(nn.Module):
    """Two-layer KAN readout: d_in → hidden → d_out.

    Replaces the final linear projection layer in an LM head or classifier
    with a learnable B-spline network. Parameter-efficient relative to MLP
    at equivalent expressivity.

    Dtype contract:
        input  : float32  (B, d_in)
        output : float32  (B, d_out)

    Parameters
    ----------
    d_in : int
        Input dimension (typically d_model = 256).
    d_out : int
        Output dimension (vocab_size or n_classes).
    n_grid : int
        Grid points per spline, default 5.
    spline_order : int
        B-spline order, default 3.
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        n_grid: int = 5,
        spline_order: int = 3,
    ) -> None:
        super().__init__()
        hidden = max(int(math.sqrt(d_in * d_out)), 32)
        self.layer1 = BSplineKANLayer(d_in, hidden, n_grid, spline_order)
        self.layer2 = BSplineKANLayer(hidden, d_out, n_grid, spline_order)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : float32 Tensor, shape (B, d_in)

        Returns
        -------
        logits : float32 Tensor, shape (B, d_out)
        """
        assert x.dtype == torch.float32, f"KANReadout expects float32, got {x.dtype}"
        h = F.gelu(self.layer1(x))
        return self.layer2(h)
