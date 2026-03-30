#!/usr/bin/env python3
"""
scripts/eval_bow_baseline.py — Compute unigram bag-of-words character-level
perplexity from a text file.

Establishes the lower-bound perplexity that any model must beat. Uses Laplace
smoothing over all 256 byte values so zero-probability characters cannot cause
infinite perplexity.

Usage:
    python scripts/eval_bow_baseline.py data/tinystories_val.txt

Exit codes:
    0  success
    1  file not found / empty
    2  unexpected error
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


def compute_unigram_distribution(text: str) -> dict[int, float]:
    """Return a probability distribution over all 256 byte values.

    Counts are Laplace-smoothed (count + 1e-10) so every byte value has
    non-zero probability.  The returned dict maps byte value (0–255) to
    probability; values sum to 1.0.
    """
    counts: dict[int, float] = {b: 1e-10 for b in range(256)}
    for ch in text:
        byte_val = ord(ch) & 0xFF  # treat as latin-1 byte value
        counts[byte_val] = counts[byte_val] + 1.0

    total = sum(counts.values())
    return {b: c / total for b, c in counts.items()}


def unigram_perplexity(text: str, dist: dict[int, float]) -> float:
    """Return per-character perplexity of text under dist.

    ppl = exp(-mean(log(p(c)) for c in text))

    Always >= 1.0.  Raises ValueError if text is empty.
    """
    if not text:
        raise ValueError("text must not be empty")

    neg_log_sum = 0.0
    for ch in text:
        byte_val = ord(ch) & 0xFF
        p = dist.get(byte_val, 1e-10)
        neg_log_sum += -math.log(p)

    return math.exp(neg_log_sum / len(text))


def bow_perplexity_from_file(
    text_file: "str | Path",
    train_frac: float = 0.9,
) -> dict:
    """Compute unigram perplexity using a train/val split of text_file.

    Returns:
        {
            "bow_ppl":     float,  # perplexity on val split
            "vocab_size":  int,    # number of distinct byte values seen
            "train_chars": int,
            "val_chars":   int,
        }
    """
    text = Path(text_file).read_text(encoding="utf-8", errors="replace")
    if not text:
        raise ValueError(f"File is empty: {text_file}")

    split = int(len(text) * train_frac)
    train_text = text[:split]
    val_text = text[split:]

    dist = compute_unigram_distribution(train_text)
    vocab_size = sum(1 for b in range(256) if train_text.count(chr(b)) > 0)

    if not val_text:
        # Edge case: all text in train split
        val_text = train_text

    ppl = unigram_perplexity(val_text, dist)

    return {
        "bow_ppl": ppl,
        "vocab_size": vocab_size,
        "train_chars": len(train_text),
        "val_chars": len(val_text),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Compute unigram BoW character-level perplexity from a text file.",
        epilog="Example: eval_bow_baseline.py data/tinystories_val.txt",
    )
    p.add_argument("text_file", help="Path to text file to evaluate")
    p.add_argument(
        "--train-frac",
        type=float,
        default=0.9,
        metavar="F",
        help="Fraction of file used as training distribution (default: 0.9)",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress informational output (exit code only)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    path = Path(args.text_file)
    if not path.exists():
        if not args.quiet:
            print(f"ERROR file not found: {path}", file=sys.stderr)
        return 1

    try:
        result = bow_perplexity_from_file(path, train_frac=args.train_frac)
    except ValueError as exc:
        if not args.quiet:
            print(f"ERROR {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001
        if not args.quiet:
            print(f"ERROR unexpected: {exc}", file=sys.stderr)
        return 2

    if not args.quiet:
        print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
