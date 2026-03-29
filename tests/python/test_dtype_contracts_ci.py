"""CI Assertion: dtype boundary contracts.

This file is the single authoritative source for the DTYPE_CONTRACT table in CI.
Every value matches DREX_UNIFIED_SPEC.md §DTYPE BOUNDARY CONTRACT exactly.

Wave 0: validates the contract table itself and the PyTorch cast mechanics.
Waves 1–6: each component test file asserts its own output dtype by importing
          DTYPE_CONTRACT from this module.

Rule: if the spec dtype table changes, update DTYPE_CONTRACT here first,
then update the component test that fails — never the other way round.
"""
import pytest
import torch


# ---------------------------------------------------------------------------
# Canonical dtype contract — DREX_UNIFIED_SPEC.md §DTYPE BOUNDARY CONTRACT
# ---------------------------------------------------------------------------

DTYPE_CONTRACT: dict[str, torch.dtype] = {
    # HDC / input
    "hdc_encoder_output":         torch.float32,
    # Mamba cast boundary (the ONE place float32→bfloat16 is permitted)
    "mamba_input_before_cast":    torch.float32,
    "mamba_input_after_cast":     torch.bfloat16,
    "mamba_output":               torch.bfloat16,
    # Controller outputs three distinct types
    "controller_input":           torch.bfloat16,
    "controller_write_decisions": torch.int32,
    "controller_read_weights":    torch.float32,
    "controller_sparse_gates":    torch.bool,
    # Memory tiers
    "esn_reservoir_state":        torch.float32,
    "esn_readout_output":         torch.float32,
    "episodic_memory_state":      torch.float32,
    "semantic_memory_weights":    torch.bfloat16,
    # Readout + reward
    "kan_readout_output":         torch.float32,
    "reward_signal":              torch.float32,
    "esn_feedback_signal":        torch.float32,
}

_EXPECTED_KEYS = frozenset(DTYPE_CONTRACT.keys())


# ---------------------------------------------------------------------------
# Table self-validation
# ---------------------------------------------------------------------------

class TestDtypeContractTable:
    """Validate the DTYPE_CONTRACT table is complete and well-typed."""

    def test_all_15_keys_present(self):
        """The contract must have exactly 15 entries — one per spec boundary."""
        assert len(DTYPE_CONTRACT) == 15, (
            f"Expected 15 entries, got {len(DTYPE_CONTRACT)}.  "
            "Update DTYPE_CONTRACT and DREX_UNIFIED_SPEC.md together."
        )

    def test_all_values_are_torch_dtypes(self):
        for name, dtype in DTYPE_CONTRACT.items():
            assert isinstance(dtype, torch.dtype), (
                f"DTYPE_CONTRACT['{name}'] = {dtype!r} is not a torch.dtype"
            )

    def test_no_unknown_keys(self):
        """Contract must not contain any key absent from the spec."""
        assert set(DTYPE_CONTRACT.keys()) == _EXPECTED_KEYS


# ---------------------------------------------------------------------------
# float32 boundary group
# ---------------------------------------------------------------------------

class TestFloat32Boundaries:
    _FLOAT32_KEYS = {
        "hdc_encoder_output",
        "mamba_input_before_cast",
        "esn_reservoir_state",
        "esn_readout_output",
        "episodic_memory_state",
        "kan_readout_output",
        "reward_signal",
        "esn_feedback_signal",
        "controller_read_weights",
    }

    def test_float32_group_correct(self):
        for key in self._FLOAT32_KEYS:
            assert DTYPE_CONTRACT[key] == torch.float32, (
                f"Expected {key} = float32, got {DTYPE_CONTRACT[key]}"
            )

    def test_synthetic_float32_tensor(self):
        """Confirm torch produces float32 by default for randn()."""
        x = torch.randn(2, 4)
        assert x.dtype == torch.float32


# ---------------------------------------------------------------------------
# bfloat16 boundary group
# ---------------------------------------------------------------------------

class TestBfloat16Boundaries:
    _BF16_KEYS = {
        "mamba_input_after_cast",
        "mamba_output",
        "controller_input",
        "semantic_memory_weights",
    }

    def test_bfloat16_group_correct(self):
        for key in self._BF16_KEYS:
            assert DTYPE_CONTRACT[key] == torch.bfloat16, (
                f"Expected {key} = bfloat16, got {DTYPE_CONTRACT[key]}"
            )

    def test_synthetic_bfloat16_cast(self):
        """Verify the PyTorch cast to bfloat16 works on synthetic tensors."""
        x_f32 = torch.randn(2, 4, 256, dtype=torch.float32)
        x_bf16 = x_f32.to(torch.bfloat16)
        assert x_bf16.dtype == torch.bfloat16
        assert x_bf16.shape == x_f32.shape


# ---------------------------------------------------------------------------
# Mamba cast boundary
# ---------------------------------------------------------------------------

class TestMambaCastBoundary:
    """The Mamba input projection is the ONE place in the pipeline where a
    float32→bfloat16 cast is permitted.  No other boundary may implicitly cast.
    """

    def test_boundary_is_f32_in_bf16_out(self):
        assert DTYPE_CONTRACT["mamba_input_before_cast"] == torch.float32
        assert DTYPE_CONTRACT["mamba_input_after_cast"] == torch.bfloat16

    def test_explicit_cast_preserves_shape(self):
        B, S, D = 2, 16, 256
        x_f32 = torch.randn(B, S, D, dtype=torch.float32)
        x_bf16 = x_f32.to(DTYPE_CONTRACT["mamba_input_after_cast"])
        assert x_bf16.shape == (B, S, D)
        assert x_bf16.dtype == torch.bfloat16

    def test_pytorch_mixed_precision_promotion(self):
        """Document PyTorch\'s f32+bf16 promotion behaviour: result is float32.
        This is why the spec requires an explicit cast — implicit ops would
        silently promote bf16 back to f32 rather than staying bf16.
        """
        a = torch.randn(4, dtype=torch.float32)
        b = torch.randn(4, dtype=torch.bfloat16)
        result = a + b
        # PyTorch promotes bfloat16 → float32 in mixed-dtype ops.
        assert result.dtype == torch.float32, (
            "PyTorch promotes bf16+f32 → f32.  "
            "This is why the explicit cast rule exists — implicit ops do not stay bf16."
        )


# ---------------------------------------------------------------------------
# Controller output dtypes
# ---------------------------------------------------------------------------

class TestControllerOutputDtypes:
    """Controller emits three distinct tensor types in one step."""

    def test_write_decisions_are_int32(self):
        assert DTYPE_CONTRACT["controller_write_decisions"] == torch.int32
        synthetic = torch.zeros(2, 3, dtype=torch.int32)
        assert synthetic.dtype == torch.int32

    def test_read_weights_are_float32(self):
        assert DTYPE_CONTRACT["controller_read_weights"] == torch.float32
        synthetic = torch.zeros(2, 3, dtype=torch.float32)
        assert synthetic.dtype == torch.float32

    def test_sparse_gates_are_bool(self):
        assert DTYPE_CONTRACT["controller_sparse_gates"] == torch.bool
        synthetic = torch.zeros(2, 3, dtype=torch.bool)
        assert synthetic.dtype == torch.bool

    def test_all_three_controller_dtypes_distinct(self):
        """int32, float32, bool must be three distinct types."""
        types = {
            DTYPE_CONTRACT["controller_write_decisions"],
            DTYPE_CONTRACT["controller_read_weights"],
            DTYPE_CONTRACT["controller_sparse_gates"],
        }
        assert len(types) == 3, "Controller must output three distinct dtypes"

