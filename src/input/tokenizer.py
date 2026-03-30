"""DREX-UNIFIED COMPONENT 1: Input Tokenizer.

Referenced from DREX_UNIFIED_SPEC.md § COMPONENT 1 — Phase 1 Prerequisite.

Two modes
---------
"byte"
    UTF-8 byte-level encoding. Each byte → one token ID in [0, 255].
    Vocab size is always 256. Preferred for Mamba-based LM tasks; eliminates
    the vocabulary bottleneck and lets the model operate directly on raw bytes.

"bpe"
    Byte Pair Encoding via the HuggingFace ``tokenizers`` library. Vocab size
    is configurable (default 32 000). The BPE tokenizer trains lazily on the
    first call to encode(). Token IDs in [0, vocab_size). Retained for
    benchmarking against transformer baselines that require sub-word tokens.

Padding contract — never applied silently
-----------------------------------------
When a batch contains sequences of unequal length, all sequences are
right-padded to the longest sequence length in the batch using pad_id = 0
(null byte in byte mode; the reserved [PAD] special token in BPE mode).
Padding is applied only within a batch call when sequence lengths differ.
Single-element batches are never padded. The pad_id is exposed as the class
attribute ``PAD_ID``.

Truncation
----------
If ``max_length`` is not None, each sequence is truncated to ``max_length``
tokens BEFORE any padding is applied. ``max_length=None`` (default) means no
truncation is applied.
"""

from __future__ import annotations

import torch


class DrexTokenizer:
    """Encodes raw strings to int32 token ID tensors of shape (B, S).

    See module docstring for full contract on padding and truncation.

    Args:
        mode: "byte" for byte-level encoding (vocab=256) or "bpe" for Byte Pair
              Encoding (vocab=vocab_size). Default "byte".
        vocab_size: target vocabulary size for BPE mode. Ignored in byte mode
                    (always 256). Default 32000.
        max_length: if not None, sequences are truncated to this many tokens
                    before padding. None means no truncation. Default None.
    """

    #: Pad token ID used for right-padding shorter sequences in a batch.
    #: Applied in both byte mode (null byte) and BPE mode ([PAD] special token).
    PAD_ID: int = 0

    def __init__(
        self,
        mode: str = "byte",
        vocab_size: int = 32000,
        max_length: int | None = None,
    ) -> None:
        if mode not in ("byte", "bpe"):
            raise ValueError(f"mode must be 'byte' or 'bpe', got {mode!r}")
        self.mode = mode
        self.vocab_size = vocab_size if mode == "bpe" else 256
        self.max_length = max_length
        self._bpe_tokenizer = None  # lazy-initialised on first encode() call in bpe mode

    def encode(self, texts: list[str]) -> torch.Tensor:
        """Encode a batch of strings to integer token IDs.

        Padding:
            Sequences shorter than the longest in the batch are right-padded
            with PAD_ID (0). See module docstring for full padding contract.

        Args:
            texts: list of B strings.

        Returns:
            Tensor of shape (B, S) with dtype torch.int32.
            B = len(texts). S = max sequence length in the batch after truncation.
        """
        if self.mode == "byte":
            sequences: list[list[int]] = [list(t.encode("utf-8")) for t in texts]
        else:
            sequences = self._encode_bpe(texts)

        if self.max_length is not None:
            sequences = [seq[: self.max_length] for seq in sequences]

        max_len = max((len(s) for s in sequences), default=0)
        padded = [seq + [self.PAD_ID] * (max_len - len(seq)) for seq in sequences]

        result = torch.tensor(padded, dtype=torch.int32)
        assert result.dtype == torch.int32, "encode() output dtype invariant violated"
        return result

    def decode(self, token_ids: torch.Tensor) -> list[str]:
        """Decode a (B, S) int32 token ID tensor back to strings (byte mode only).

        Trailing PAD_ID (0) bytes are stripped before decoding, recovering the
        original sequence when used with single-element batches or equal-length
        batches. For padded batches, trailing null bytes from padding are removed.

        Args:
            token_ids: Tensor of shape (B, S) dtype int32.

        Returns:
            List of B decoded strings.

        Raises:
            NotImplementedError: if called in bpe mode.
        """
        if self.mode != "byte":
            raise NotImplementedError("decode() is implemented for byte mode only.")
        out = []
        for row in token_ids:
            ids = row.tolist()
            # Strip trailing pad bytes (ID 0 = null byte) added during batching.
            while ids and ids[-1] == self.PAD_ID:
                ids.pop()
            out.append(bytes(ids).decode("utf-8", errors="replace"))
        return out

    # ------------------------------------------------------------------
    # BPE helpers — tokenizers library is a hard dep but imported lazily
    # so byte mode has zero import cost from the HF library.
    # ------------------------------------------------------------------

    def _encode_bpe(self, texts: list[str]) -> list[list[int]]:
        """Encode texts via BPE. Trains the tokenizer lazily on first call."""
        if self._bpe_tokenizer is None:
            self._train_bpe(texts)
        encodings = self._bpe_tokenizer.encode_batch(texts)
        return [enc.ids for enc in encodings]

    def _train_bpe(self, texts: list[str]) -> None:
        """Train a byte-level BPE tokenizer on the provided corpus.

        Uses ByteLevel pre-tokenizer with full 256-byte initial alphabet so that
        no token in any UTF-8 string is ever mapped to [UNK]. The resulting
        vocabulary covers all bytes [0, 255] as base tokens plus BPE merges up
        to vocab_size. All token IDs are therefore in [0, vocab_size).
        """
        from tokenizers import Tokenizer
        from tokenizers.models import BPE
        from tokenizers.pre_tokenizers import ByteLevel as ByteLevelPre
        from tokenizers.trainers import BpeTrainer

        tok = Tokenizer(BPE(unk_token="[UNK]"))
        tok.pre_tokenizer = ByteLevelPre(add_prefix_space=False)
        trainer = BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=["[PAD]", "[UNK]"],
            initial_alphabet=ByteLevelPre.alphabet(),
            show_progress=False,
        )
        tok.train_from_iterator(iter(texts), trainer=trainer)
        self._bpe_tokenizer = tok
