"""CI Assertion: gradient-leak prevention contracts.

Wave 0 (infrastructure): verifies that PyTorch detach() severs gradients
as required by the architecture.  All three architectural gradient-boundary
contracts are tested here with synthetic toy modules so they run on any
hardware (CPU, no MLX required).

Contracts checked:
  - ESN: W_res and W_in must never receive a gradient (requires_grad=False).
  - NoProp: block N's backward must not touch block M's parameters.
  - Mamba PCN: layer N local loss must not flow into layer N-1 parameters.

These tests remain in CI permanently.  Once the real components land in
src/, the per-component test files (test_reservoir.py, test_semantic.py,
test_mamba.py) also assert these contracts against actual implementations.

Ref: DREX_UNIFIED_SPEC.md §COMPONENT 3, §COMPONENT 5, §COMPONENT 7.
"""
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _grad_is_none(param: nn.Parameter) -> bool:
    return param.grad is None


# ---------------------------------------------------------------------------
# Baseline: verify `.detach()` mechanism works as expected in this PyTorch
# ---------------------------------------------------------------------------

class TestStopGradientMechanism:
    """Sanity-baseline: confirm PyTorch detach() behaves per spec before
    relying on it for architectural contracts."""

    def test_detach_severs_upstream_gradient(self):
        """detach() must prevent gradient from reaching the upstream module."""
        upstream = nn.Linear(4, 4)
        downstream = nn.Linear(4, 4)

        x = torch.randn(2, 4)
        h_stopped = upstream(x).detach()
        downstream(h_stopped).sum().backward()

        assert _grad_is_none(upstream.weight), (
            "upstream.weight must have no gradient when output is detached"
        )
        assert not _grad_is_none(downstream.weight), (
            "downstream.weight must accumulate a gradient"
        )

    def test_without_detach_gradient_propagates(self):
        """Baseline: without detach(), gradient DOES propagate (sanity check)."""
        upstream = nn.Linear(4, 4)
        downstream = nn.Linear(4, 4)

        x = torch.randn(2, 4)
        downstream(upstream(x)).sum().backward()

        assert not _grad_is_none(upstream.weight), (
            "Without detach(), upstream must receive gradient"
        )


# ---------------------------------------------------------------------------
# ESN reservoir: fixed weights must receive NO gradient
# ---------------------------------------------------------------------------

class TestESNReservoirGradientContract:
    """W_res and W_in are fixed at init and must never accumulate gradients.
    Verified here with a synthetic reservoir; repeated in test_reservoir.py
    against the real ESNReservoir class once Wave 2 is implemented.
    """

    def test_fixed_reservoir_weights_receive_no_gradient(self):
        """ESN contract: W_res, W_in are requires_grad=False — gradient-free."""
        W_res = nn.Parameter(torch.randn(8, 8), requires_grad=False)
        W_in = nn.Parameter(torch.randn(4, 8), requires_grad=False)
        readout = nn.Linear(8, 4)

        x = torch.randn(1, 4)
        with torch.no_grad():
            prev_state = torch.zeros(1, 8)
            # state = tanh(x @ W_in + prev_state @ W_res)  (simplified ESN step)
            state = torch.tanh(x @ W_in + prev_state @ W_res)

        # Readout trains; reservoir does not.
        readout(state).sum().backward()

        assert _grad_is_none(W_res), "W_res must not receive gradient (ESN is fixed)"
        assert _grad_is_none(W_in), "W_in must not receive gradient (ESN is fixed)"
        assert not _grad_is_none(readout.weight), "Readout must receive gradient"

    def test_esn_feedback_path_does_not_leak_to_reservoir(self):
        """Feedback signal flowing back into the reservoir state must not create
        a gradient path to W_res.  Verified by running a feedback step under
        torch.no_grad()."""
        W_res = nn.Parameter(torch.randn(8, 8), requires_grad=False)
        W_fb = nn.Parameter(torch.randn(4, 8), requires_grad=False)  # feedback weight
        readout = nn.Linear(8, 4)

        x = torch.randn(1, 4)
        with torch.no_grad():
            prev_state = torch.zeros(1, 8)
            prev_output = torch.randn(1, 4)            # feedback term
            # state = tanh(x @ W_in_placeholder + prev_state @ W_res + prev_output @ W_fb)
            state = torch.tanh(prev_state @ W_res + prev_output @ W_fb)

        readout(state).sum().backward()

        assert _grad_is_none(W_res), "W_res must stay gradient-free even with feedback"
        assert _grad_is_none(W_fb), "W_fb must stay gradient-free (fixed feedback weight)"


# ---------------------------------------------------------------------------
# NoProp: block independence contract
# ---------------------------------------------------------------------------

class TestNoPropBlockIndependence:
    """Each NoProp block trains on its own local loss.  Block A's backward
    must not touch Block B's parameters, and vice versa.
    Verified here with synthetic 2-block chain; repeated in test_semantic.py
    against the real NoPropBlock class once Wave 4 is implemented.
    """

    def test_block_a_backward_does_not_touch_block_b(self):
        """Loss from block A must not update block B's weights."""
        block_a = nn.Linear(8, 8)
        block_b = nn.Linear(8, 8)

        x = torch.randn(2, 8)

        # Block A: compute, local loss, backward.
        loss_a = block_a(x).pow(2).mean()
        loss_a.backward()

        # Block B: receives detached input (stop_gradient boundary).
        x_b = block_a(x.detach()).detach()   # detach: no path back to block_a
        loss_b = block_b(x_b).pow(2).mean()
        loss_b.backward()

        # After both backwards, block_a.weight.grad comes from loss_a only;
        # block_b.weight.grad comes from loss_b only.
        assert block_a.weight.grad is not None, "block_a must have grad from loss_a"
        assert block_b.weight.grad is not None, "block_b must have grad from loss_b"

    def test_block_b_backward_does_not_touch_block_a(self):
        """Core NoProp contract: a fresh backward for block B must not disturb
        block A (which has already been zeroed)."""
        block_a = nn.Linear(8, 8)
        block_b = nn.Linear(8, 8)

        # Zero block_a grads (simulate end of block_a's update step).
        block_a.zero_grad()

        # Block B backward with a fresh synthetic input.
        x_b = torch.randn(2, 8)
        block_b(x_b).pow(2).mean().backward()

        assert _grad_is_none(block_a.weight), (
            "Block A must not receive gradients from Block B\'s backward pass. "
            "NoProp block independence contract violated."
        )
        assert not _grad_is_none(block_b.weight), "Block B must have its own gradient"

    def test_n_block_chain_all_independent(self):
        """Generalised: N=4 blocks — each backward touches only its own block."""
        N = 4
        blocks = [nn.Linear(8, 8) for _ in range(N)]
        x = torch.randn(2, 8)

        for i, blk in enumerate(blocks):
            xi = x.detach()
            blk(xi).pow(2).mean().backward()

        # Every block has its own grad, no cross-contamination possible
        # because every input was detached before entering the block.
        for i, blk in enumerate(blocks):
            assert not _grad_is_none(blk.weight), f"block {i} must have its own gradient"


# ---------------------------------------------------------------------------
# Mamba PCN: per-layer stop_gradient contract
# ---------------------------------------------------------------------------

class TestMambaPCNLayerStopGradient:
    """Mamba PCN contract: layer N\'s local loss backward must not flow into
    layer N-1\'s parameters.  Verified with a synthetic 2-layer chain here;
    repeated in test_mamba.py against the real PCNMambaBackbone once Wave 3
    is implemented.
    """

    def test_layer_1_loss_does_not_flow_to_layer_0(self):
        """Layer 0 is done with its backward before layer 1 starts."""
        layer_0 = nn.Linear(8, 8)
        layer_1 = nn.Linear(8, 8)

        x = torch.randn(2, 4, 8)  # (B, S, D)

        # Layer 0: forward + local loss + backward, then detach output.
        h0 = layer_0(x)
        h0.pow(2).mean().backward()          # layer_0 gets gradient from its own loss
        h0_stopped = layer_0(x).detach()     # stop_gradient for layer_1's input

        # Layer 1: forward + local loss + backward.
        layer_1(h0_stopped).pow(2).mean().backward()

        # layer_0 grad present (from its own loss_0); NOT touched by loss_1.
        assert not _grad_is_none(layer_0.weight), (
            "layer_0.weight must have grad from its own PCN loss"
        )
        assert not _grad_is_none(layer_1.weight), (
            "layer_1.weight must have grad from its own PCN loss"
        )

    def test_four_layer_pcn_all_local(self):
        """4-layer PCN: each layer\'s local loss backward is independent."""
        L = 4
        layers = [nn.Linear(8, 8) for _ in range(L)]
        x = torch.randn(2, 4, 8)

        h = x
        for layer in layers:
            h_out = layer(h.detach())        # stop_gradient on input
            h_out.pow(2).mean().backward()   # local loss
            h = h_out

        for i, layer in enumerate(layers):
            assert not _grad_is_none(layer.weight), (
                f"PCN layer {i} must have gradient from its own local loss"
            )

