"""ESN Reservoir — DREX-UNIFIED COMPONENT 5, OBJECTIVE 2a.

Sparsely-connected recurrent network with fixed random weights.
Never updated after initialisation (zero training cost for the reservoir).
Only the linear readout is trained (ridge regression, closed-form).

Interface (per DREX_UNIFIED_SPEC.md v0.2 § COMPONENT 5):
  Input:   x_seq (B, S, d_model) float32
  Output:  states (B, S, N_reservoir) float32

Dtype contract:
  ALL tensors in this module are float32.  The bfloat16 cast belongs
  exclusively at the Mamba input-projection boundary (not here).

Ref: DREX_UNIFIED_SPEC.md § COMPONENT 5: WORKING MEMORY — L1 ESN RESERVOIR
"""
from __future__ import annotations

import torch
import torch.nn as nn


class EchoStateNetwork(nn.Module):
    """Fixed-random-weight echo state network with trainable linear readout.

    W_res, W_in, and W_fb are registered as buffers — they are saved/loaded
    with the model but receive ZERO gradient (never in optimizer param groups).

    Args:
        d_model:          Input/output dimension.
        n_reservoir:      Reservoir size (N).  Production default: d_model * 4.
        spectral_radius:  Max absolute eigenvalue of W_res.  Must be < 1.0.
        sparsity:         Fraction of nonzero connections in W_res.
        d_read:           Readout dimension.  Defaults to d_model.
        feedback:         Whether to use output→reservoir feedback via W_fb.
        ridge_alpha:      Ridge regression regularisation λ.
        seed:             RNG seed for all fixed matrices.
    """

    def __init__(
        self,
        d_model: int,
        n_reservoir: int,
        spectral_radius: float = 0.95,
        sparsity: float = 0.01,
        d_read: int | None = None,
        feedback: bool = True,
        ridge_alpha: float = 1e-4,
        seed: int = 42,
    ) -> None:
        super().__init__()
        if spectral_radius >= 1.0:
            raise ValueError(
                f"spectral_radius must be < 1.0 for the echo state property; "
                f"got {spectral_radius}"
            )

        self.d_model = d_model
        self.n_reservoir = n_reservoir
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.d_read = d_read if d_read is not None else d_model
        self.feedback = feedback
        self.ridge_alpha = ridge_alpha

        gen = torch.Generator()
        gen.manual_seed(seed)

        # -----------------------------------------------------------------
        # W_res: (N, N) — sparse recurrent weights scaled to spectral_radius
        # -----------------------------------------------------------------
        # For small N, 1% sparsity gives too few edges and the matrix can be
        # nilpotent (all eigenvalues = 0, unable to rescale).  Ensure at
        # least 5 expected connections per neuron so ring-less DAGs are rare.
        effective_sparsity = max(sparsity, 5.0 / n_reservoir)

        W_raw = torch.randn(n_reservoir, n_reservoir, generator=gen)
        mask = (torch.rand(n_reservoir, n_reservoir, generator=gen) < effective_sparsity).float()
        W_raw = W_raw * mask

        # Spectral radius enforcement
        eigvals = torch.linalg.eigvals(W_raw)
        max_eig = eigvals.abs().max().item()
        if max_eig > 0:
            W_raw = W_raw * (spectral_radius / max_eig)

        # Final assertion — must hold before any forward pass
        final_max = torch.linalg.eigvals(W_raw).abs().max().item()
        if final_max >= 1.0:
            raise ValueError(
                f"Failed to enforce spectral_radius < 1.0 after rescaling; "
                f"max eigenvalue = {final_max:.6f}"
            )

        self.register_buffer("W_res", W_raw.float())  # (N, N) float32

        # -----------------------------------------------------------------
        # W_in: (N, d_model) — fixed random input projection
        # -----------------------------------------------------------------
        W_in = torch.randn(n_reservoir, d_model, generator=gen) * 0.1
        self.register_buffer("W_in", W_in.float())  # (N, d_model) float32

        # -----------------------------------------------------------------
        # W_fb: (N, d_read) — fixed random feedback projection
        # -----------------------------------------------------------------
        W_fb = torch.randn(n_reservoir, self.d_read, generator=gen) * 0.1
        self.register_buffer("W_fb", W_fb.float())  # (N, d_read) float32

        # W_readout is None until fit_readout() is called
        self.register_buffer("W_readout", None)  # will be (N, d_read) float32

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def reset_state(self, batch_size: int) -> torch.Tensor:
        """Return a zeroed initial reservoir state (B, N_reservoir) float32."""
        return torch.zeros(batch_size, self.n_reservoir, dtype=torch.float32)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        x_seq: torch.Tensor,
        feedback_seq: torch.Tensor | None = None,
        initial_state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run reservoir over a full sequence.

        Args:
            x_seq:        (B, S, d_model) float32
            feedback_seq: (B, S, d_read) float32 or None
            initial_state:(B, N_reservoir) float32 or None (zeros if None)

        Returns:
            states: (B, S, N_reservoir) float32 — one state per timestep
        """
        assert x_seq.dtype == torch.float32, (
            f"EchoStateNetwork.forward expects float32, got {x_seq.dtype}"
        )
        B, S, _ = x_seq.shape
        state = (
            initial_state
            if initial_state is not None
            else self.reset_state(B).to(x_seq.device)
        )

        all_states = []
        for t in range(S):
            x_t = x_seq[:, t, :]  # (B, d_model)
            # pre-activation: W_res @ state + W_in @ x_t  [+ W_fb @ fb_t]
            # shapes: (B, N) = (B, N) @ (N, N)^T  ... use einsum for clarity
            pre = (
                torch.einsum("bi,ni->bn", state, self.W_res)   # (B, N)
                + torch.einsum("bi,ni->bn", x_t, self.W_in)   # (B, N)
            )
            if self.feedback and feedback_seq is not None:
                fb_t = feedback_seq[:, t, :]  # (B, d_read)
                pre = pre + torch.einsum("bi,ni->bn", fb_t, self.W_fb)

            state = torch.tanh(pre)  # (B, N)
            all_states.append(state)

        return torch.stack(all_states, dim=1)  # (B, S, N) float32

    # ------------------------------------------------------------------
    # Readout (ridge regression — closed form, no gradient)
    # ------------------------------------------------------------------

    def fit_readout(self, states: torch.Tensor, targets: torch.Tensor) -> None:
        """Fit W_readout via closed-form ridge regression.

        Args:
            states:  (B, S, N_reservoir) or (M, N_reservoir) float32
            targets: (B, S, d_read)     or (M, d_read)       float32
        """
        if states.dim() == 3:
            M = states.shape[0] * states.shape[1]
            X = states.reshape(M, self.n_reservoir).float()
            Y = targets.reshape(M, self.d_read).float()
        else:
            X = states.float()
            Y = targets.float()

        # W = (X^T X + λI)^{-1} X^T Y  — standard ridge normal equations
        I = torch.eye(self.n_reservoir, device=X.device, dtype=torch.float32)
        W = torch.linalg.solve(X.T @ X + self.ridge_alpha * I, X.T @ Y)
        # Register as buffer so it persists in state_dict without gradients
        self.register_buffer("W_readout", W.float())  # (N, d_read)

    def read(self, state: torch.Tensor) -> torch.Tensor:
        """Apply W_readout to a reservoir state.

        Args:
            state: (B, N_reservoir) float32
        Returns:
            (B, d_read) float32
        """
        if self.W_readout is None:
            raise RuntimeError("Call fit_readout() before read()")
        return state @ self.W_readout  # (B, d_read) float32
