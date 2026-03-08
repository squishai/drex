"""
Tests for drex.models.attention — SlidingWindowAttention, InfiniAttention, HybridAttention.
"""

import pytest
import torch

from drex.models.attention import HybridAttention, InfiniAttention, SlidingWindowAttention
from drex.models.memory import MemoryState


@pytest.fixture
def batch():
    return 2


@pytest.fixture
def seq():
    return 16


@pytest.fixture
def d_model():
    return 64


@pytest.fixture
def n_heads():
    return 4


class TestSlidingWindowAttention:
    def test_output_shape(self, batch, seq, d_model, n_heads, device):
        dev = torch.device(device)
        attn = SlidingWindowAttention(d_model, n_heads, window_size=32).to(dev)
        x = torch.randn(batch, seq, d_model, device=dev)
        out = attn(x)
        assert out.shape == (batch, seq, d_model)

    def test_output_dtype(self, batch, seq, d_model, n_heads, device):
        dev = torch.device(device)
        attn = SlidingWindowAttention(d_model, n_heads).to(dev)
        x = torch.randn(batch, seq, d_model, device=dev, dtype=torch.float32)
        out = attn(x)
        assert out.dtype == torch.float32

    def test_backward(self, batch, seq, d_model, n_heads, device):
        dev = torch.device(device)
        attn = SlidingWindowAttention(d_model, n_heads).to(dev)
        x = torch.randn(batch, seq, d_model, device=dev, requires_grad=True)
        out = attn(x)
        out.sum().backward()
        assert x.grad is not None

    def test_causal_property(self, batch, d_model, n_heads, device):
        """Token at position i should not depend on position j > i."""
        dev = torch.device(device)
        attn = SlidingWindowAttention(d_model, n_heads).to(dev)
        attn.eval()
        seq = 8
        x1 = torch.randn(1, seq, d_model, device=dev)
        x2 = x1.clone()
        # Modify last position in x2
        x2[0, -1] = torch.randn(d_model, device=dev)
        with torch.no_grad():
            out1 = attn(x1)
            out2 = attn(x2)
        # First S-1 positions should be identical (causal mask)
        assert torch.allclose(out1[0, :-1], out2[0, :-1], atol=1e-5)


class TestInfiniAttention:
    def _make_state(self, batch, n_heads, d_k, device):
        return MemoryState.zeros(batch, n_heads, d_k, d_k, torch.device(device))

    def test_output_shape(self, batch, seq, d_model, n_heads, device):
        dev = torch.device(device)
        attn = InfiniAttention(d_model, n_heads).to(dev)
        d_k = d_model // n_heads
        state = self._make_state(batch, n_heads, d_k, device)
        x = torch.randn(batch, seq, d_model, device=dev)
        out, new_state = attn(x, state)
        assert out.shape == (batch, seq, d_model)

    def test_state_changes(self, batch, seq, d_model, n_heads, device):
        dev = torch.device(device)
        attn = InfiniAttention(d_model, n_heads).to(dev)
        d_k = d_model // n_heads
        state = self._make_state(batch, n_heads, d_k, device)
        x = torch.randn(batch, seq, d_model, device=dev)
        _, new_state = attn(x, state)
        # M should change from zero
        assert not torch.allclose(new_state.M, state.M)
        # z should be positive and non-zero
        assert (new_state.z > 0).all()

    def test_backward(self, batch, seq, d_model, n_heads, device):
        dev = torch.device(device)
        attn = InfiniAttention(d_model, n_heads).to(dev)
        d_k = d_model // n_heads
        state = self._make_state(batch, n_heads, d_k, device)
        x = torch.randn(batch, seq, d_model, device=dev, requires_grad=True)
        out, new_state = attn(x, state)
        out.sum().backward()
        assert x.grad is not None

    def test_memory_accumulates_across_segments(self, batch, d_model, n_heads, device):
        """Running two segments should accumulate more information in M."""
        dev = torch.device(device)
        attn = InfiniAttention(d_model, n_heads).to(dev)
        d_k = d_model // n_heads
        state = self._make_state(batch, n_heads, d_k, device)
        seq = 8
        x = torch.randn(batch, seq, d_model, device=dev)
        _, state1 = attn(x, state)
        _, state2 = attn(x, state1)
        # M should have grown (norm increases with information)
        assert state2.M.norm() > state1.M.norm()


class TestHybridAttention:
    def _make_state(self, batch, n_heads, d_k, device):
        return MemoryState.zeros(batch, n_heads, d_k, d_k, torch.device(device))

    def test_output_shape(self, batch, seq, d_model, n_heads, device):
        dev = torch.device(device)
        attn = HybridAttention(d_model, n_heads).to(dev)
        d_k = d_model // n_heads
        state = self._make_state(batch, n_heads, d_k, device)
        x = torch.randn(batch, seq, d_model, device=dev)
        out, new_state = attn(x, state)
        assert out.shape == (batch, seq, d_model)
        assert new_state.M.shape == (batch, n_heads, d_k, d_k)

    def test_backward(self, batch, seq, d_model, n_heads, device):
        dev = torch.device(device)
        attn = HybridAttention(d_model, n_heads).to(dev)
        d_k = d_model // n_heads
        state = self._make_state(batch, n_heads, d_k, device)
        x = torch.randn(batch, seq, d_model, device=dev, requires_grad=True)
        out, _ = attn(x, state)
        out.sum().backward()
        assert x.grad is not None

    def test_beta_grad(self, batch, seq, d_model, n_heads, device):
        """Gate parameter β should receive gradients."""
        dev = torch.device(device)
        attn = HybridAttention(d_model, n_heads).to(dev)
        d_k = d_model // n_heads
        state = self._make_state(batch, n_heads, d_k, device)
        x = torch.randn(batch, seq, d_model, device=dev)
        out, _ = attn(x, state)
        out.sum().backward()
        assert attn.beta.grad is not None

    def test_state_is_updated(self, batch, seq, d_model, n_heads, device):
        dev = torch.device(device)
        attn = HybridAttention(d_model, n_heads).to(dev)
        d_k = d_model // n_heads
        state = self._make_state(batch, n_heads, d_k, device)
        x = torch.randn(batch, seq, d_model, device=dev)
        _, new_state = attn(x, state)
        assert not torch.allclose(new_state.M, state.M)
