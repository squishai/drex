#!/usr/bin/env python3
"""
scripts/record_sprint_results.py — Parse a train.py stdout log. Extract
val_ppl at a target step. Check against a gate threshold. Emit result.

Usage:
    python scripts/record_sprint_results.py results/poc/sprint2b_seed42.log \
        --target-step 10000 --gate 1.32

Exit codes:
    0  gate pass  (val_ppl <= gate)
    1  gate fail  (val_ppl > gate) or step not found in log
    2  file not found / parse error
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

# Matches lines produced by train.py validation loop:
#   "  [val] step    500  val_loss 0.6359  val_ppl    1.89"
_VAL_RE = re.compile(
    r"\[val\]\s+step\s+(\d+)\s+val_loss\s+\S+\s+val_ppl\s+(\S+)"
)

# Matches write-rate log lines (planned format, not yet emitted by any run):
#   "wr 0.72 [...]"
_WR_RE = re.compile(r"\bwr\s+(\S+)\s+\[")


def parse_log(log_path: "str | Path") -> list[dict]:
    """Return list of {step: int, val_ppl: float} records sorted by step.

    Returns [] on missing file, empty file, or no matching lines.
    """
    path = Path(log_path)
    if not path.exists():
        return []
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []

    records: list[dict] = []
    for m in _VAL_RE.finditer(text):
        step = int(m.group(1))
        try:
            val_ppl = float(m.group(2))
        except ValueError:
            continue
        if not math.isfinite(val_ppl):
            continue
        records.append({"step": step, "val_ppl": val_ppl})

    records.sort(key=lambda r: r["step"])
    return records


def extract_at_step(records: list[dict], target_step: int) -> "float | None":
    """Return val_ppl for the record matching target_step, or None."""
    for r in records:
        if r["step"] == target_step:
            return r["val_ppl"]
    return None


def check_gate(val_ppl: float, gate: float) -> bool:
    """Return True iff val_ppl <= gate (inclusive)."""
    return val_ppl <= gate


def extract_write_rates(log_path: "str | Path") -> list[float]:
    """Return list of write-rate floats from 'wr <F> [...]' lines.

    Returns [] for all current logs (the wr line format is planned but not
    yet emitted by any training run).
    """
    path = Path(log_path)
    if not path.exists():
        return []
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []

    rates: list[float] = []
    for m in _WR_RE.finditer(text):
        try:
            rates.append(float(m.group(1)))
        except ValueError:
            continue
    return rates


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Parse a train.py log and check val_ppl against a gate.",
        epilog="Example: record_sprint_results.py results/poc/sprint2b_seed42.log"
               " --target-step 10000 --gate 1.32",
    )
    p.add_argument("log_file", help="Path to train.py stdout log")
    p.add_argument(
        "--target-step",
        type=int,
        default=10000,
        metavar="N",
        help="Training step to read val_ppl from (default: 10000)",
    )
    p.add_argument(
        "--gate",
        type=float,
        default=1.32,
        metavar="G",
        help="val_ppl gate threshold; pass if val_ppl <= gate (default: 1.32)",
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

    path = Path(args.log_file)
    if not path.exists():
        if not args.quiet:
            print(f"ERROR file not found: {path}", file=sys.stderr)
        return 2

    try:
        records = parse_log(path)
    except Exception as exc:  # noqa: BLE001
        if not args.quiet:
            print(f"ERROR parse failed: {exc}", file=sys.stderr)
        return 2

    val_ppl = extract_at_step(records, args.target_step)

    if val_ppl is None:
        if not args.quiet:
            print(f"MISSING step={args.target_step} not found")
        return 1

    if check_gate(val_ppl, args.gate):
        if not args.quiet:
            print(f"PASS val_ppl={val_ppl:.4f} gate={args.gate}")
        return 0
    else:
        if not args.quiet:
            print(f"FAIL val_ppl={val_ppl:.4f} gate={args.gate}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
