"""
Shared pytest fixtures for the drex test suite.
"""

import pytest
import torch


@pytest.fixture(scope="session")
def device() -> str:
    """
    Return the best available device for this test run.
    Prefers MPS (Apple Silicon), falls back to CPU.
    CUDA is not expected on Mac development hardware.
    """
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@pytest.fixture(scope="session")
def dtype() -> torch.dtype:
    """
    torch.float32 is the only supported dtype on MPS.
    All model tensors must use this dtype.
    """
    return torch.float32


@pytest.fixture
def small_config():
    """A minimal DrexConfig for fast unit tests (tiny model)."""
    from drex.models.transformer import DrexConfig
    return DrexConfig(
        d_model=64,
        n_heads=4,
        n_layers=2,
        ff_mult=4,
        vocab_size=256,
        window_size=128,
        use_l3=False,
    )
