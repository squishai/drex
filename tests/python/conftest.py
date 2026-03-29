"""DREX pytest configuration and shared fixtures.

Canonical dimension constants used across all DREX tests.
Match DREX_UNIFIED_SPEC.md for any changes.
"""
import pytest
import torch


# ---------------------------------------------------------------------------
# Canonical test dimensions (small enough for fast CI, big enough to catch bugs)
# ---------------------------------------------------------------------------
B = 2           # batch size
S = 16          # sequence length (production = 512)
D_MODEL = 256   # model hidden dimension
D_HDC = 1024    # HDC hyperdimension (production = 10000; 1024 passes tests fast)
N_RESERVOIR = 64  # ESN reservoir size (production = 2000)
VOCAB_SIZE = 512  # vocabulary (production = 32000+)
N_TIERS = 3     # memory routing tiers: L1=ESN, L2=episodic, L3=semantic


@pytest.fixture(scope="session")
def dims():
    """Canonical test dimensions dict, available to every test via fixture."""
    return dict(
        B=B, S=S, D_MODEL=D_MODEL, D_HDC=D_HDC,
        N_RESERVOIR=N_RESERVOIR, VOCAB_SIZE=VOCAB_SIZE, N_TIERS=N_TIERS,
    )


@pytest.fixture()
def token_ids(dims):
    """Synthetic (B, S) int32 token IDs in [0, VOCAB_SIZE)."""
    return torch.randint(0, dims["VOCAB_SIZE"], (dims["B"], dims["S"]), dtype=torch.int32)


@pytest.fixture()
def hdc_output(dims):
    """Synthetic HDC encoder output: (B, S, D_hdc) float32."""
    return torch.randn(dims["B"], dims["S"], dims["D_HDC"], dtype=torch.float32)


@pytest.fixture()
def context_bf16(dims):
    """Synthetic Mamba backbone output: (B, S, D_model) bfloat16."""
    return torch.randn(dims["B"], dims["S"], dims["D_MODEL"], dtype=torch.bfloat16)


@pytest.fixture()
def context_f32(dims):
    """Synthetic context in float32 for components that require it."""
    return torch.randn(dims["B"], dims["S"], dims["D_MODEL"], dtype=torch.float32)


def assert_no_nan_inf(tensor: torch.Tensor, name: str = "tensor") -> None:
    """Reusable NaN/Inf guard. Import from conftest or copy the contract."""
    assert not torch.isnan(tensor).any(), f"NaN in {name}"
    assert not torch.isinf(tensor).any(), f"Inf in {name}"

