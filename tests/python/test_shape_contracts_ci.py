"""CI Assertion: tensor shape contracts.

All tensors in DREX use batch-first convention: (B, S, D).
B = batch, S = sequence length, D = feature dimension.

Wave 0: defines the shape contract table and validates it with synthetic
        zero-tensors.  No src/ implementations needed.
Waves 1–6: each component test imports the relevant shape constants from
            conftest.py and asserts its own output shape.

Ref: DREX_UNIFIED_SPEC.md §COMPONENT interface specs.
"""
import pytest
import torch


# ---------------------------------------------------------------------------
# Canonical test dimensions (must match conftest.py)
# ---------------------------------------------------------------------------
B = 2
S = 16
D_MODEL = 256
D_HDC = 1024
N_RESERVOIR = 64
VOCAB_SIZE = 512
N_TIERS = 3


# ---------------------------------------------------------------------------
# Shape contract table
# (component_name, input_shape, output_shape, note)
# ---------------------------------------------------------------------------

SHAPE_CONTRACTS = [
    (
        "hdc_encoder",
        (B, S),
        (B, S, D_HDC),
        "int32 token IDs → float32 hypervectors",
    ),
    (
        "mamba_backbone",
        (B, S, D_MODEL),
        (B, S, D_MODEL),
        "bfloat16 in, bfloat16 out — sequence shape preserved",
    ),
    (
        "esn_reservoir_single_step",
        (N_RESERVOIR,),
        (N_RESERVOIR,),
        "reservoir state: (N,) float32 per sample",
    ),
    (
        "esn_readout",
        (S, N_RESERVOIR),
        (S, D_MODEL),
        "unbatched readout across a sequence: (S,N)→(S,D)",
    ),
    (
        "controller",
        (B, S, D_MODEL),
        (B, N_TIERS),
        "routing decision per sample in batch",
    ),
    (
        "kan_readout",
        (B, S, D_MODEL),
        (B, S, VOCAB_SIZE),
        "logit distribution over vocabulary",
    ),
]


# ---------------------------------------------------------------------------
# Table self-validation
# ---------------------------------------------------------------------------

class TestShapeContractTable:
    """The shape contract table must be self-consistent before being used
    as a reference in component tests."""

    def test_all_shapes_are_positive_integer_tuples(self):
        for name, in_shape, out_shape, _ in SHAPE_CONTRACTS:
            for shape, label in [(in_shape, "input"), (out_shape, "output")]:
                assert isinstance(shape, tuple), (
                    f"{name} {label}_shape must be a tuple, got {type(shape)}"
                )
                assert all(isinstance(d, int) and d > 0 for d in shape), (
                    f"{name} {label}_shape must be positive ints, got {shape}"
                )

    def test_contract_has_expected_component_names(self):
        names = {c[0] for c in SHAPE_CONTRACTS}
        required = {
            "hdc_encoder", "mamba_backbone", "esn_reservoir_single_step",
            "esn_readout", "controller", "kan_readout",
        }
        assert required.issubset(names), (
            f"Missing components: {required - names}"
        )


# ---------------------------------------------------------------------------
# Batch-first convention
# ---------------------------------------------------------------------------

class TestBatchFirstConvention:
    """All batched tensors must be (B, ...).  Validates the convention itself."""

    def test_batched_components_have_B_as_first_dim(self):
        batched_components = [
            c for c in SHAPE_CONTRACTS if len(c[1]) > 1 and c[1][0] == B
        ]
        assert len(batched_components) >= 3, "At least 3 batched components expected"
        for name, in_shape, _, _ in batched_components:
            assert in_shape[0] == B, (
                f"{name}: input first dim must be B={B}, got {in_shape[0]}"
            )

    def test_batch_first_synthetic_tensor(self):
        x_correct = torch.zeros(B, S, D_MODEL)   # (B, S, D) — correct
        x_wrong = torch.zeros(D_MODEL, S, B)      # (D, S, B) — wrong
        assert x_correct.shape[0] == B
        assert x_wrong.shape[0] != B


# ---------------------------------------------------------------------------
# Per-component synthetic shape assertions
# ---------------------------------------------------------------------------

class TestHDCEncoderShape:
    def test_input_is_int32_token_ids(self):
        token_ids = torch.randint(0, VOCAB_SIZE, (B, S), dtype=torch.int32)
        assert token_ids.shape == (B, S)
        assert token_ids.dtype == torch.int32

    def test_output_is_float32_hypervectors(self):
        hdc_out = torch.zeros(B, S, D_HDC, dtype=torch.float32)
        assert hdc_out.shape == (B, S, D_HDC)
        assert hdc_out.dtype == torch.float32

    def test_sequence_length_preserved(self):
        token_ids = torch.randint(0, VOCAB_SIZE, (B, S), dtype=torch.int32)
        hdc_out = torch.zeros(B, S, D_HDC, dtype=torch.float32)
        assert token_ids.shape[1] == hdc_out.shape[1] == S


class TestMambaBackboneShape:
    def test_shape_preserved(self):
        x = torch.zeros(B, S, D_MODEL, dtype=torch.bfloat16)
        out = torch.zeros_like(x)
        assert out.shape == (B, S, D_MODEL)
        assert out.dtype == torch.bfloat16

    def test_seq_and_model_dim_unchanged(self):
        x = torch.zeros(B, S, D_MODEL, dtype=torch.bfloat16)
        assert x.shape[1] == S
        assert x.shape[2] == D_MODEL


class TestESNReservoirShape:
    def test_state_shape_per_sample(self):
        state = torch.zeros(N_RESERVOIR, dtype=torch.float32)
        assert state.shape == (N_RESERVOIR,)
        assert state.dtype == torch.float32

    def test_state_sequence_batch(self):
        """When batched over sequence: (S, N_reservoir)."""
        state_seq = torch.zeros(S, N_RESERVOIR, dtype=torch.float32)
        assert state_seq.shape == (S, N_RESERVOIR)

    def test_readout_shape(self):
        readout_in = torch.zeros(S, N_RESERVOIR, dtype=torch.float32)
        readout_out = torch.zeros(S, D_MODEL, dtype=torch.float32)
        assert readout_in.shape[0] == readout_out.shape[0] == S


class TestControllerShape:
    def test_write_decisions_shape(self):
        write_decisions = torch.zeros(B, N_TIERS, dtype=torch.int32)
        assert write_decisions.shape == (B, N_TIERS)
        assert write_decisions.dtype == torch.int32

    def test_read_weights_shape(self):
        read_weights = torch.zeros(B, N_TIERS, dtype=torch.float32)
        assert read_weights.shape == (B, N_TIERS)
        assert read_weights.dtype == torch.float32

    def test_sparse_gates_shape(self):
        gates = torch.zeros(B, N_TIERS, dtype=torch.bool)
        assert gates.shape == (B, N_TIERS)
        assert gates.dtype == torch.bool

    def test_n_tiers_value(self):
        """L1=ESN (0), L2=episodic (1), L3=semantic (2)."""
        assert N_TIERS == 3


class TestKANReadoutShape:
    def test_input_output_shapes(self):
        x_in = torch.zeros(B, S, D_MODEL, dtype=torch.float32)
        logits = torch.zeros(B, S, VOCAB_SIZE, dtype=torch.float32)
        assert x_in.shape == (B, S, D_MODEL)
        assert logits.shape == (B, S, VOCAB_SIZE)
        assert logits.dtype == torch.float32

    def test_sequence_dim_preserved(self):
        x_in = torch.zeros(B, S, D_MODEL)
        logits = torch.zeros(B, S, VOCAB_SIZE)
        assert x_in.shape[1] == logits.shape[1] == S


# ---------------------------------------------------------------------------
# End-to-end shape consistency across pipeline stages
# ---------------------------------------------------------------------------

class TestPipelineShapeConsistency:
    """Synthetic tensors at each pipeline stage must have consistent S dimension
    and compatible dtypes for hand-off.
    """

    def test_sequence_dimension_unchanged_through_pipeline(self):
        token_ids = torch.randint(0, VOCAB_SIZE, (B, S), dtype=torch.int32)
        hdc_out   = torch.zeros(B, S, D_HDC,     dtype=torch.float32)
        ctx       = torch.zeros(B, S, D_MODEL,   dtype=torch.bfloat16)
        logits    = torch.zeros(B, S, VOCAB_SIZE, dtype=torch.float32)

        assert (
            token_ids.shape[1]
            == hdc_out.shape[1]
            == ctx.shape[1]
            == logits.shape[1]
            == S
        ), "S dimension must be identical at every pipeline stage"

    def test_batch_dimension_unchanged_through_pipeline(self):
        token_ids = torch.randint(0, VOCAB_SIZE, (B, S), dtype=torch.int32)
        hdc_out   = torch.zeros(B, S, D_HDC,     dtype=torch.float32)
        ctx       = torch.zeros(B, S, D_MODEL,   dtype=torch.bfloat16)
        logits    = torch.zeros(B, S, VOCAB_SIZE, dtype=torch.float32)

        assert (
            token_ids.shape[0]
            == hdc_out.shape[0]
            == ctx.shape[0]
            == logits.shape[0]
            == B
        ), "B dimension must be identical at every pipeline stage"

    def test_no_nan_inf_in_synthetic_ci_inputs(self):
        """Guard: CI synthetic fixtures must not contain NaN/Inf.
        A contaminated fixture would make gradient tests meaningless.
        """
        tensors = [
            torch.zeros(B, S, D_MODEL, dtype=torch.float32),
            torch.zeros(B, S, D_HDC,   dtype=torch.float32),
            torch.zeros(B, S, D_MODEL, dtype=torch.bfloat16),
            torch.zeros(N_RESERVOIR,   dtype=torch.float32),
        ]
        for t in tensors:
            assert not torch.isnan(t).any(), "CI synthetic tensor must not contain NaN"
            assert not torch.isinf(t).any(), "CI synthetic tensor must not contain Inf"

