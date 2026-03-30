"""Wave 1 validation tests — DREX-UNIFIED COMPONENT 2: HDC ENCODER.

Tests required by DREX_UNIFIED_SPEC.md § COMPONENT 2 / PHASE 1 EXIT CRITERIA:
  1. Similarity     : cosine_sim(encode(t), encode(t)) > 0.999  (100 random tokens)
  2. Orthogonality  : mean cosine_sim(encode(A), encode(B)) < 0.05  (1000 pairs A≠B)
  3. Associativity  : bind→unbind retrieval cosine_sim > 0.9
  4. Sequence order : encode_sequence([A,B]) ≠ encode_sequence([B,A])
  5. Shape contract : forward(token_ids).shape == (B, S, D_HDC) using conftest dims
  6. Dtype contract : forward(token_ids).dtype == torch.float32

Additional sweep:
  7. D_hdc sweep    : orthogonality gate at D=1024, 2048, 4096 — documents minimum.
"""
import pytest
import torch
import torch.nn.functional as F

from hdc.encoder import HDCTokenEncoder

# ---------------------------------------------------------------------------
# Canonical dims (copied from conftest.py constants — accessed via pytest
# fixtures below; constants inlined here to keep tests pure-unit)
# ---------------------------------------------------------------------------
B = 2
S = 16
D_HDC = 1024
VOCAB_SIZE = 512
SEED = 42

# ---------------------------------------------------------------------------
# Shared encoder fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def encoder() -> HDCTokenEncoder:
    enc = HDCTokenEncoder(d_hdc=D_HDC, vocab_size=VOCAB_SIZE, normalize=True, seed=SEED)
    enc.eval()
    return enc


@pytest.fixture(scope="module")
def raw_encoder() -> HDCTokenEncoder:
    """normalize=False — needed for clean algebraic unbind/associativity tests."""
    enc = HDCTokenEncoder(d_hdc=D_HDC, vocab_size=VOCAB_SIZE, normalize=False, seed=SEED)
    enc.eval()
    return enc


# ---------------------------------------------------------------------------
# TEST 1 — Similarity
# SPEC: cosine_sim(encode(t), encode(t)) > 0.999 for 100 random tokens
# ---------------------------------------------------------------------------


class TestSimilarity:
    def test_self_similarity_100_tokens(self, raw_encoder: HDCTokenEncoder) -> None:
        """encode(t) always returns the same vector — self-similarity must be > 0.999."""
        rng = torch.Generator()
        rng.manual_seed(SEED)
        token_ids = torch.randint(0, VOCAB_SIZE, (100,), generator=rng, dtype=torch.int32)

        failures = []
        for tid in token_ids:
            hv_a = raw_encoder.encode_token(tid)
            hv_b = raw_encoder.encode_token(tid)
            sim = F.cosine_similarity(hv_a.unsqueeze(0), hv_b.unsqueeze(0)).item()
            if sim <= 0.999:
                failures.append((tid.item(), sim))

        assert not failures, (
            f"Self-similarity failed for {len(failures)} tokens (expected > 0.999): {failures[:5]}"
        )


# ---------------------------------------------------------------------------
# TEST 2 — Orthogonality
# SPEC: mean(cosine_sim(encode(A), encode(B))) < 0.05 for 1000 distinct pairs
# ---------------------------------------------------------------------------


class TestOrthogonality:
    def test_mean_cosine_below_threshold(self, raw_encoder: HDCTokenEncoder) -> None:
        """Mean |cosine_sim| over 1000 distinct-token pairs must be < 0.05."""
        rng = torch.Generator()
        rng.manual_seed(SEED + 1)

        sims = []
        attempts = 0
        while len(sims) < 1000 and attempts < 5000:
            ids = torch.randint(0, VOCAB_SIZE, (2,), generator=rng, dtype=torch.int32)
            a_id, b_id = ids[0].item(), ids[1].item()
            attempts += 1
            if a_id == b_id:
                continue
            hv_a = raw_encoder.encode_token(ids[0])
            hv_b = raw_encoder.encode_token(ids[1])
            sims.append(
                F.cosine_similarity(hv_a.unsqueeze(0), hv_b.unsqueeze(0)).abs().item()
            )

        assert len(sims) >= 1000, f"Only {len(sims)} distinct pairs generated"
        mean_sim = sum(sims) / len(sims)
        assert mean_sim < 0.05, (
            f"Orthogonality contract FAILED: mean |cosine_sim| = {mean_sim:.4f} ≥ 0.05 "
            f"(D_hdc={D_HDC}, vocab_size={VOCAB_SIZE})"
        )

    def test_high_similarity_fraction(self, raw_encoder: HDCTokenEncoder) -> None:
        """Fraction of pairs with |cosine_sim| > 0.3 must be < 1%."""
        rng = torch.Generator()
        rng.manual_seed(SEED + 2)

        high = 0
        checked = 0
        attempts = 0
        while checked < 500 and attempts < 2000:
            ids = torch.randint(0, VOCAB_SIZE, (2,), generator=rng, dtype=torch.int32)
            attempts += 1
            if ids[0].item() == ids[1].item():
                continue
            hv_a = raw_encoder.encode_token(ids[0])
            hv_b = raw_encoder.encode_token(ids[1])
            sim = F.cosine_similarity(hv_a.unsqueeze(0), hv_b.unsqueeze(0)).abs().item()
            if sim > 0.3:
                high += 1
            checked += 1

        fraction = high / max(checked, 1)
        assert fraction < 0.01, (
            f"{high}/{checked} pairs had |cosine_sim| > 0.3 ({fraction:.2%}). "
            "D_hdc may be too small."
        )


# ---------------------------------------------------------------------------
# TEST 3 — Associativity (bind/unbind)
# SPEC: bind(encode(A), encode(B)) — unbind with encode(A) → cosine_sim > 0.9
# For bipolar {-1,+1}: a*(a*b) = (a*a)*b = 1*b = b (exact for bipolar)
# ---------------------------------------------------------------------------


class TestAssociativity:
    def test_bind_unbind_retrieval(self, raw_encoder: HDCTokenEncoder) -> None:
        """Unbinding with one argument exactly recovers the other (bipolar property)."""
        rng = torch.Generator()
        rng.manual_seed(SEED + 3)

        failures = []
        for _ in range(50):
            ids = torch.randint(0, VOCAB_SIZE, (2,), generator=rng, dtype=torch.int32)
            a_id, b_id = ids[0], ids[1]
            if a_id.item() == b_id.item():
                continue

            hv_a = raw_encoder.encode_token(a_id)
            hv_b = raw_encoder.encode_token(b_id)

            bound = HDCTokenEncoder.bind(hv_a, hv_b)   # bind(a, b)
            retrieved = HDCTokenEncoder.bind(hv_a, bound)  # unbind: a*(a*b) = b

            sim = F.cosine_similarity(retrieved.unsqueeze(0), hv_b.unsqueeze(0)).item()
            if sim <= 0.9:
                failures.append((a_id.item(), b_id.item(), sim))

        assert not failures, (
            f"Bind/unbind retrieval cosine_sim ≤ 0.9 for {len(failures)} pairs: {failures[:5]}"
        )


# ---------------------------------------------------------------------------
# TEST 4 — Sequence order
# SPEC: encode_sequence([A,B]) ≠ encode_sequence([B,A])
# ---------------------------------------------------------------------------


class TestSequenceOrder:
    def test_order_sensitive(self, raw_encoder: HDCTokenEncoder) -> None:
        """[A,B] and [B,A] must produce different hypervectors."""
        rng = torch.Generator()
        rng.manual_seed(SEED + 4)

        failures = []
        for _ in range(50):
            ids = torch.randint(0, VOCAB_SIZE, (2,), generator=rng, dtype=torch.int32)
            a_id, b_id = ids[0], ids[1]
            if a_id.item() == b_id.item():
                continue

            seq_ab = raw_encoder.encode_sequence(torch.stack([a_id, b_id]))
            seq_ba = raw_encoder.encode_sequence(torch.stack([b_id, a_id]))

            if torch.allclose(seq_ab, seq_ba, atol=1e-6):
                failures.append((a_id.item(), b_id.item()))

        assert not failures, (
            f"encode_sequence gave identical result for [A,B] and [B,A] "
            f"in {len(failures)} cases: {failures[:5]}"
        )

    def test_order_cosine_not_one(self, raw_encoder: HDCTokenEncoder) -> None:
        """Cosine sim between [A,B] and [B,A] must be < 0.98 (not effectively identical)."""
        rng = torch.Generator()
        rng.manual_seed(SEED + 5)

        near_identical = 0
        checked = 0
        attempts = 0
        while checked < 50 and attempts < 200:
            ids = torch.randint(0, VOCAB_SIZE, (2,), generator=rng, dtype=torch.int32)
            attempts += 1
            if ids[0].item() == ids[1].item():
                continue

            seq_ab = raw_encoder.encode_sequence(torch.stack([ids[0], ids[1]]))
            seq_ba = raw_encoder.encode_sequence(torch.stack([ids[1], ids[0]]))

            sim = F.cosine_similarity(seq_ab.unsqueeze(0), seq_ba.unsqueeze(0)).abs().item()
            if sim >= 0.98:
                near_identical += 1
            checked += 1

        assert near_identical == 0, (
            f"{near_identical}/{checked} reversed-order pair(s) had |cosine_sim| ≥ 0.98 — "
            "positional permutation may not be working."
        )


# ---------------------------------------------------------------------------
# TEST 5 — Shape contract
# SPEC: forward(token_ids).shape == (B, S, D_hdc)
# ---------------------------------------------------------------------------


class TestShapeContract:
    def test_canonical_batch_shape(self, encoder: HDCTokenEncoder) -> None:
        """forward((B, S) int32) → (B, S, D_HDC) with canonical conftest dims."""
        x = torch.randint(0, VOCAB_SIZE, (B, S), dtype=torch.int32)
        out = encoder(x)
        assert out.shape == (B, S, D_HDC), f"Got {out.shape}, expected ({B}, {S}, {D_HDC})"

    def test_batch_size_1(self, encoder: HDCTokenEncoder) -> None:
        """Shape works for batch size 1."""
        x = torch.randint(0, VOCAB_SIZE, (1, S), dtype=torch.int32)
        out = encoder(x)
        assert out.shape == (1, S, D_HDC)

    def test_variable_seq_len(self, encoder: HDCTokenEncoder) -> None:
        """Shape works for sequence length != canonical S."""
        x = torch.randint(0, VOCAB_SIZE, (3, 32), dtype=torch.int32)
        out = encoder(x)
        assert out.shape == (3, 32, D_HDC)

    def test_item_memory_shape(self, encoder: HDCTokenEncoder) -> None:
        """item_memory must be (vocab_size, D_hdc) exactly."""
        assert encoder.item_memory.shape == (VOCAB_SIZE, D_HDC), (
            f"item_memory shape: {encoder.item_memory.shape}"
        )


# ---------------------------------------------------------------------------
# TEST 6 — Dtype contract
# SPEC: forward() output must be float32
#       (bfloat16 cast belongs ONLY at the Mamba input projection boundary)
# ---------------------------------------------------------------------------


class TestDtypeContract:
    def test_output_float32(self, encoder: HDCTokenEncoder) -> None:
        """forward() must return float32."""
        x = torch.randint(0, VOCAB_SIZE, (B, S), dtype=torch.int32)
        out = encoder(x)
        assert out.dtype == torch.float32, f"Got {out.dtype}, expected torch.float32"

    def test_item_memory_float32(self, encoder: HDCTokenEncoder) -> None:
        """item_memory buffer must be float32."""
        assert encoder.item_memory.dtype == torch.float32, (
            f"item_memory dtype: {encoder.item_memory.dtype}"
        )

    def test_float_input_raises(self, encoder: HDCTokenEncoder) -> None:
        """Passing float32 input must raise AssertionError (expects int tokens)."""
        x_bad = torch.zeros(B, S, dtype=torch.float32)
        with pytest.raises(AssertionError):
            encoder(x_bad)

    def test_encode_token_float32(self, encoder: HDCTokenEncoder) -> None:
        """encode_token() must return float32."""
        hv = encoder.encode_token(torch.tensor(0, dtype=torch.int32))
        assert hv.dtype == torch.float32

    def test_no_nan_inf(self, encoder: HDCTokenEncoder) -> None:
        """No NaN or Inf values in forward() output."""
        x = torch.randint(0, VOCAB_SIZE, (B, S), dtype=torch.int32)
        out = encoder(x)
        assert not torch.isnan(out).any(), "NaN in HDCTokenEncoder output"
        assert not torch.isinf(out).any(), "Inf in HDCTokenEncoder output"


# ---------------------------------------------------------------------------
# TEST 7 — D_hdc sweep
# SPEC: document the minimum D_hdc that passes the orthogonality gate.
# D=1024 is the conftest canonical; 2048 and 4096 are higher-fidelity options.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("d_hdc", [1024, 2048, 4096])
def test_d_hdc_sweep_orthogonality(d_hdc: int) -> None:
    """Orthogonality gate: mean |cosine_sim| < 0.05 at each D_hdc value."""
    enc = HDCTokenEncoder(d_hdc=d_hdc, vocab_size=VOCAB_SIZE, normalize=False, seed=SEED)
    enc.eval()

    rng = torch.Generator()
    rng.manual_seed(99)

    sims = []
    attempts = 0
    while len(sims) < 1000 and attempts < 5000:
        ids = torch.randint(0, VOCAB_SIZE, (2,), generator=rng, dtype=torch.int32)
        attempts += 1
        if ids[0].item() == ids[1].item():
            continue
        hv_a = enc.encode_token(ids[0])
        hv_b = enc.encode_token(ids[1])
        sims.append(
            F.cosine_similarity(hv_a.unsqueeze(0), hv_b.unsqueeze(0)).abs().item()
        )

    assert len(sims) >= 1000
    mean_sim = sum(sims) / len(sims)
    assert mean_sim < 0.05, (
        f"[D_hdc={d_hdc}] Orthogonality FAILED: mean |cosine_sim| = {mean_sim:.4f} ≥ 0.05"
    )
