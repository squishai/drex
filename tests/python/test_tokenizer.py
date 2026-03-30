"""Wave 8 validation tests — DREX-UNIFIED COMPONENT 1: INPUT TOKENIZER.

Tests required by DREX_UNIFIED_SPEC.md § COMPONENT 1 / VALIDATION CRITERIA:
  1. ByteVocabRange        : byte mode token IDs all in [0, 255], dtype=int32
  2. BpeVocabRange         : bpe mode token IDs all in [0, vocab_size), dtype=int32
  3. RoundTrip             : byte encode → decode recovers original ASCII string
  4. NoPaddingWithoutNotice: class docstring documents padding; B=1 shape is (1, S)
"""

import inspect

import pytest
import torch

from input.tokenizer import DrexTokenizer

# ---------------------------------------------------------------------------
# Test corpus constants
# ---------------------------------------------------------------------------

SEED = 42

# 10 ASCII pangram / near-pangram sentences for round-trip and range tests.
ASCII_SENTENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "Pack my box with five dozen liquor jugs.",
    "How vexingly quick daft zebras jump!",
    "The five boxing wizards jump quickly.",
    "Sphinx of black quartz, judge my vow.",
    "Cwm fjord bank glyphs vext quiz.",
    "Mr. Jock, TV quiz PhD, bags few lynx.",
    "The jay, pig, fox, zebra and my wolves quack!",
    "Lazy movers quit hard-packing of papier-mache jewels.",
    "A wizard's job is to vex chumps quickly in fog.",
]

# 20 strings used as both the BPE training corpus and the encode input,
# guaranteeing no UNK tokens appear in the BPE test output.
BPE_TEXTS = [
    "hello world",
    "foo bar baz",
    "the quick brown fox",
    "neural network training",
    "drex architecture test",
    "tokenizer validation",
    "byte pair encoding",
    "deep learning model",
    "language model inference",
    "transformer attention mechanism",
    "mamba backbone forward",
    "hdc encoder output tensor",
    "episodic memory update",
    "semantic memory noprop",
    "reward signal loop stable",
    "kan readout layer spline",
    "sparse router top-k gating",
    "integration pipeline end to end",
    "phase one complete validated",
    "wave eight input tokenizer",
]

# ---------------------------------------------------------------------------
# Shared fixtures (module-scoped — tokenizer objects are stateless or
# lazily-trained once; re-creating per test would be wasteful).
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def byte_tok() -> DrexTokenizer:
    return DrexTokenizer(mode="byte")


@pytest.fixture(scope="module")
def bpe_tok() -> DrexTokenizer:
    """BPE tokenizer with vocab_size=512. Trains lazily on first encode() call."""
    return DrexTokenizer(mode="bpe", vocab_size=512)


# ---------------------------------------------------------------------------
# TEST 1 — ByteVocabRange
# SPEC: byte mode output vocab range is 0–255, no out-of-bounds values
# ---------------------------------------------------------------------------


class TestByteVocabRange:
    def test_ascii_ids_in_range_and_dtype(self, byte_tok: DrexTokenizer) -> None:
        """100 ASCII strings: all IDs in [0, 255], dtype must be torch.int32."""
        texts = ASCII_SENTENCES * 10  # 100 strings
        out = byte_tok.encode(texts)
        assert out.dtype == torch.int32, f"Expected torch.int32, got {out.dtype}"
        assert out.min().item() >= 0, f"min token ID {out.min().item()} < 0"
        assert out.max().item() <= 255, f"max token ID {out.max().item()} > 255"

    def test_utf8_multibyte_ids_in_range(self, byte_tok: DrexTokenizer) -> None:
        """Multi-byte UTF-8 strings: high bytes (128–255) still within [0, 255]."""
        texts = ["こんにちは", "안녕하세요", "Héllo wörld", "naïve café résumé"]
        out = byte_tok.encode(texts)
        assert out.dtype == torch.int32
        assert out.min().item() >= 0
        assert out.max().item() <= 255


# ---------------------------------------------------------------------------
# TEST 2 — BpeVocabRange
# SPEC: bpe mode output vocab range is 0 to vocab_size - 1
# ---------------------------------------------------------------------------


class TestBpeVocabRange:
    def test_bpe_ids_in_range_and_dtype(self, bpe_tok: DrexTokenizer) -> None:
        """20 BPE-encoded strings: all IDs in [0, 511], dtype must be torch.int32."""
        out = bpe_tok.encode(BPE_TEXTS)
        assert out.dtype == torch.int32, f"Expected torch.int32, got {out.dtype}"
        assert out.min().item() >= 0, f"min token ID {out.min().item()} < 0"
        assert out.max().item() < 512, (
            f"max token ID {out.max().item()} >= vocab_size=512"
        )


# ---------------------------------------------------------------------------
# TEST 3 — RoundTrip
# SPEC: encode then decode recovers original string (byte mode only)
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_ascii_roundtrip_10_sentences(self, byte_tok: DrexTokenizer) -> None:
        """encode([text]) → decode must exactly recover each ASCII sentence.

        Single-element batches are used to avoid any padding interaction with
        the decode step — the round-trip contract is about encoding fidelity,
        not padding behaviour.
        """
        for text in ASCII_SENTENCES:
            encoded = byte_tok.encode([text])  # shape (1, S) — no padding
            decoded = byte_tok.decode(encoded)
            assert decoded[0] == text, (
                f"Round-trip failed:\n  original: {text!r}\n  decoded:  {decoded[0]!r}"
            )


# ---------------------------------------------------------------------------
# TEST 4 — NoPaddingWithoutNotice
# SPEC: no padding added silently — document padding behaviour explicitly
# ---------------------------------------------------------------------------


class TestNoPaddingWithoutNotice:
    def test_class_docstring_documents_padding(self) -> None:
        """The DrexTokenizer class docstring must contain the word 'pad'.

        Enforces DREX_UNIFIED_SPEC.md validation criterion 4: padding behaviour
        must be documented explicitly, not applied silently.
        """
        doc = inspect.getdoc(DrexTokenizer)
        assert doc is not None, "DrexTokenizer must have a non-empty class docstring"
        assert "pad" in doc.lower(), (
            "DrexTokenizer class docstring must document padding behaviour — "
            "the word 'pad' was not found in the docstring.\n"
            f"Docstring:\n{doc}"
        )

    def test_single_element_batch_shape(self, byte_tok: DrexTokenizer) -> None:
        """B=1 encode must return shape (1, S) with exactly 2 dimensions."""
        out = byte_tok.encode(["hello"])
        assert out.ndim == 2, f"Output must be 2D, got {out.ndim}D with shape {out.shape}"
        assert out.shape[0] == 1, f"B dimension must be 1, got {out.shape[0]}"
        assert out.shape[1] == 5, (
            f"'hello' is 5 bytes, expected S=5, got {out.shape[1]}"
        )
