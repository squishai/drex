"""Mamba PCN Backbone — DREX-UNIFIED COMPONENT 2, OBJECTIVE 1.

Minimal pure-PyTorch selective SSM (S6) implementation with Predictive Coding
training wrapper.  No external mamba_ssm dependency required.

Architecture:
    MambaSSM    — single Mamba block (input proj → conv1d → S6 scan → output proj)
    PCNMambaBackbone — stacks N MambaSSM layers, each with its own optimizer.

Dtype contract (DREX_UNIFIED_SPEC.md §dtype):
    Input to MambaSSM:   float32
    After input_proj:    cast → bfloat16 (ONLY place this cast happens)
    MambaSSM output:     bfloat16

Ref: DREX_UNIFIED_SPEC.md § COMPONENT 2: MAMBA SSM BACKBONE
"""
from __future__ import annotations

import math
from collections.abc import Callable
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# MambaSSM — single selective-SSM block
# ---------------------------------------------------------------------------

class MambaSSM(nn.Module):
    """Single Mamba block with causal conv1d and selective S6 scan.

    Args:
        d_model:   Input/output dimension.
        d_state:   SSM state dimension N.  Default 16.
        d_conv:    Causal convolution width.  Default 4.
        expand:    Inner dimension multiplier.  d_inner = expand * d_model.
        dt_rank:   Rank of delta projection.  "auto" → ceil(d_model / 16).
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: int | str = "auto",
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = expand * d_model
        self.dt_rank: int = (
            math.ceil(d_model / 16) if dt_rank == "auto" else int(dt_rank)
        )

        # --- Projections (bfloat16 after cast at input_proj) ---
        # in_proj doubles for z gating: [x_proj | z_proj]
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        # Causal depthwise conv on d_inner channels
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            bias=True,
            padding=0,  # handled manually for causality
        )

        # SSM parameter projections
        # dt: (d_inner → dt_rank) then broadcasted to d_inner
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        # x_proj maps d_inner → dt_rank + 2*d_state (B, C projections)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * self.d_state, bias=False)

        # Fixed SSM log-A initialisation (diagonal, HiPPO-inspired)
        # A is fixed as -exp(log_A) so A < 0, ensuring stability.
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).expand(
            self.d_inner, -1
        )  # (d_inner, d_state)
        self.register_buffer("A_log", torch.log(A))  # fixed, not a Parameter

        # D skip-connection (trainable scalar per channel)
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Cast all trainable weights to bfloat16 after definition.
        # The cast is performed once at init so the module stays in bf16 throughout.
        self.to(torch.bfloat16)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _causal_conv1d(self, x: torch.Tensor) -> torch.Tensor:
        """Apply causal conv1d: left-pad by (d_conv-1) zeros, trim right.

        Args:
            x: (B, d_inner, S) bfloat16

        Returns:
            (B, d_inner, S) bfloat16 — no future leakage
        """
        S = x.shape[-1]
        # Left-pad with (d_conv - 1) zeros
        x_padded = F.pad(x, (self.d_conv - 1, 0))
        # Conv without internal padding, then slice to original length
        return self.conv1d(x_padded)[..., :S]

    def _ssm_scan(
        self,
        u: torch.Tensor,          # (B, S, d_inner) bfloat16
        delta: torch.Tensor,      # (B, S, d_inner) bfloat16
        A: torch.Tensor,          # (d_inner, d_state) float32 (negative)
        B: torch.Tensor,          # (B, S, d_state) bfloat16
        C: torch.Tensor,          # (B, S, d_state) bfloat16
    ) -> torch.Tensor:
        """Selective state-space scan (ZOH discretisation, explicit loop).

        Ā[t] = exp(Δ[t] * A)        — (B, d_inner, d_state)
        B̄[t] ≈ Δ[t] * B[t]          — simplified ZOH (A⁻¹ factor absorbed into init)
        h[t] = Ā[t] * h[t-1] + B̄[t] * u[t]
        y[t] = C[t] · h[t]

        Returns:
            y: (B, S, d_inner) bfloat16
        """
        B_batch, S, d_inner = u.shape
        d_state = self.d_state

        # Work in float32 for numerical stability; cast back at end
        u_f = u.float()
        delta_f = delta.float()
        B_f = B.float()   # (B_batch, S, d_state)
        C_f = C.float()   # (B_batch, S, d_state)
        # A is already float32, negative
        A_f = A           # (d_inner, d_state)

        h = torch.zeros(B_batch, d_inner, d_state, device=u.device, dtype=torch.float32)
        ys = []

        for t in range(S):
            dt_t = delta_f[:, t, :]          # (B, d_inner)
            u_t = u_f[:, t, :]               # (B, d_inner)
            B_t = B_f[:, t, :]               # (B, d_state)
            C_t = C_f[:, t, :]               # (B, d_state)

            # Ā = exp(Δ · A): broadcast (B, d_inner, 1) * (1, d_inner, d_state)
            dA = torch.exp(
                dt_t.unsqueeze(-1) * A_f.unsqueeze(0)
            )  # (B, d_inner, d_state)

            # B̄u = Δ * u * B (ZOH approx)
            # (B, d_inner, 1) * (B, 1, d_state) → (B, d_inner, d_state)
            dBu = (dt_t * u_t).unsqueeze(-1) * B_t.unsqueeze(1)

            h = dA * h + dBu   # (B, d_inner, d_state)

            # y_t = C · h summed over state dim: (B, d_inner)
            y_t = (h * C_t.unsqueeze(1)).sum(-1)  # (B, d_inner)
            ys.append(y_t)

        y = torch.stack(ys, dim=1)  # (B, S, d_inner) float32
        return y.to(torch.bfloat16)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, S, d_model) float32

        Returns:
            out: (B, S, d_model) bfloat16

        Raises:
            AssertionError: if x.dtype != torch.float32
        """
        assert x.dtype == torch.float32, (
            f"MambaSSM expects float32 input, got {x.dtype}. "
            "dtype cast is internal to this module."
        )
        B_batch, S, _ = x.shape

        # --- Cast to bfloat16 at the input projection boundary ---
        x_bf = x.to(torch.bfloat16)

        # in_proj → split into content x and gate z
        xz = self.in_proj(x_bf)                          # (B, S, 2*d_inner)
        x_in, z = xz.chunk(2, dim=-1)                    # each (B, S, d_inner)

        # Causal conv1d on the content path  [S → channel last → S]
        x_conv = self._causal_conv1d(x_in.transpose(1, 2)).transpose(1, 2)
        x_conv = F.silu(x_conv)                          # (B, S, d_inner)

        # SSM parameter projections from x_conv
        x_proj_out = self.x_proj(x_conv)                  # (B, S, dt_rank + 2*d_state)
        dt_raw, B_ssm, C_ssm = x_proj_out.split(
            [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        delta = F.softplus(self.dt_proj(dt_raw))          # (B, S, d_inner)

        # Fixed A (negative, float32 for scan stability)
        A = -torch.exp(self.A_log.float())                # (d_inner, d_state)

        # Selective SSM scan
        y = self._ssm_scan(x_conv, delta, A, B_ssm, C_ssm)  # (B, S, d_inner) bf16

        # D skip connection
        y = y + x_conv * self.D.to(torch.bfloat16)

        # Gate with z
        y = y * F.silu(z)

        # Output projection
        out = self.out_proj(y)                             # (B, S, d_model) bf16
        return out


# ---------------------------------------------------------------------------
# PCNMambaBackbone — stacked MambaSSM with Predictive Coding training
# ---------------------------------------------------------------------------

class PCNMambaBackbone(nn.Module):
    """N MambaSSM layers trained with Predictive Coding (no cross-layer backprop).

    Each layer l has its own Adam optimizer that owns exclusively layer l's
    parameters.  local_loss[l] = MSE(out_l, sg(out_{l+1})) for l < n_layers-1.
    The top layer's loss is supplied by the caller during train_step().

    Args:
        d_model:   Model dimension.
        d_state:   SSM state dimension.
        d_conv:    Causal conv width.
        expand:    Inner dimension multiplier.
        n_layers:  Number of stacked MambaSSM blocks.
        lr:        Per-layer Adam learning rate.
    """

    def __init__(
        self,
        d_model: int = 256,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        n_layers: int = 4,
        lr: float = 1e-3,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers

        self.layers = nn.ModuleList([
            MambaSSM(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
            for _ in range(n_layers)
        ])

        # Each optimizer owns ONLY that layer's parameters — no cross-contamination.
        self.optimizers: list[torch.optim.Adam] = [
            torch.optim.Adam(list(layer.parameters()), lr=lr)
            for layer in self.layers
        ]

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, list[Optional[torch.Tensor]]]:
        """Run all layers, compute local PC losses between adjacent outputs.

        Args:
            x: (B, S, d_model) float32

        Returns:
            hidden:       (B, S, d_model) bfloat16  — output of final layer
            local_losses: list of length n_layers.
                          local_losses[l] = MSE(out_l, sg(out_{l+1})) for l < n_layers-1.
                          local_losses[-1] = None (top layer loss supplied by caller).
        """
        outputs: list[torch.Tensor] = []
        current = x
        for layer in self.layers:
            out = layer(current)
            outputs.append(out)
            # Feed bfloat16 output as float32 input to next layer
            current = out.float()

        # Compute local losses — stop_gradient on the target (layer above)
        local_losses: list[Optional[torch.Tensor]] = []
        for l in range(self.n_layers - 1):
            target = outputs[l + 1].detach().float()
            pred = outputs[l].float()
            local_losses.append(F.mse_loss(pred, target))
        local_losses.append(None)  # top layer loss provided by caller

        return outputs[-1], local_losses

    def train_step(
        self,
        x: torch.Tensor,
        top_loss_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> list[float]:
        """Perform one PCN training step.

        Each layer's loss is backpropagated independently — no cross-layer
        gradient flow.  The graph is broken by .detach() on the target.

        Args:
            x:           (B, S, d_model) float32
            top_loss_fn: Callable[Tensor → scalar Tensor] — called with the top
                         layer's fresh output to produce a task loss that is
                         independent of previous-layer parameter versions.

        Returns:
            loss_values: list[float] of length n_layers (scalar float per layer).
        """
        # Collect per-layer outputs with independent graphs.
        # We must run each layer fresh so that loss_l.backward() only touches layer l.
        outputs_detached: list[torch.Tensor] = []
        current = x
        for layer in self.layers:
            out = layer(current)
            outputs_detached.append(out)
            # Detach between layers: graph for layer l starts at out_{l-1}.detach()
            current = out.detach().float()

        loss_values: list[float] = []
        for l, (layer, opt) in enumerate(zip(self.layers, self.optimizers)):
            if l < self.n_layers - 1:
                # Re-run this layer from its detached input to get a fresh graph
                inp = (
                    x.detach() if l == 0
                    else outputs_detached[l - 1].detach().float()
                )
                out_l = layer(inp)
                target_l = outputs_detached[l + 1].detach().float()
                loss_l = F.mse_loss(out_l.float(), target_l)
            else:
                # Top layer: fresh graph — top_loss_fn is called with this layer's
                # output so gradients only flow through this layer's parameters.
                inp = outputs_detached[l - 1].detach().float() if l > 0 else x.detach()
                out_l = layer(inp)
                loss_l = top_loss_fn(out_l)

            opt.zero_grad()
            loss_l.backward(retain_graph=False)
            opt.step()
            loss_values.append(loss_l.item())

        return loss_values

    def reset_state(self) -> None:
        """Clear any internal recurrent state (reserved for future stateful impl)."""
        pass
