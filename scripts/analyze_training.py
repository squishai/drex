#!/usr/bin/env python3
"""
scripts/analyze_training.py — Parse and summarise a drex training log.

Produces three reports from a training log produced by scripts/train.py:
  1. Validation perplexity table   (step → val_ppl)
  2. Write-rate convergence report  (Exp B memory logs only)
  3. Training-step summary          (loss, ppl, lr at every logged step)

Usage:
    # Summarise a single log
    python scripts/analyze_training.py results/exp_a_train.log

    # Compare two logs side-by-side and check write-rate convergence
    python scripts/analyze_training.py \\
        --baseline results/exp_a_train.log \\
        --memory   results/exp_b_train.log \\
        --wr-converge-step 30000

Exit codes:
    0 — success (write rate converged, or no memory log provided)
    1 — write rate did NOT converge to [0.10, 0.85] by --wr-converge-step
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ─────────────────────────────────────────────────────────────────────────────
# Range constants (must match drex.models.memory)
# ─────────────────────────────────────────────────────────────────────────────
WRITE_RATE_LO: float = 0.10
WRITE_RATE_HI: float = 0.85

# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TrainRow:
    step: int
    loss: float
    ppl: float
    lr: float
    toks_per_sec: float
    wr_mean: Optional[float] = None
    wr_min: Optional[float] = None
    wr_max: Optional[float] = None


@dataclass
class ValRow:
    step: int
    val_loss: float
    val_ppl: float


@dataclass
class ParsedLog:
    path: Path
    train_rows: list[TrainRow] = field(default_factory=list)
    val_rows: list[ValRow] = field(default_factory=list)
    resumed_from: Optional[int] = None
    has_write_rates: bool = False


# ─────────────────────────────────────────────────────────────────────────────
# Parsing
# ─────────────────────────────────────────────────────────────────────────────

# step  15200  loss 0.1909  ppl    1.21  lr 2.53e-04  29,147 tok/s
_TRAIN_RE = re.compile(
    r"^step\s+(\d+)\s+"
    r"loss\s+([\d.]+)\s+"
    r"ppl\s+([\d.]+)\s+"
    r"lr\s+([\de.+-]+)"
    r"(?:\s+wr\s+([\d.]+)\s+\[([\d.]+),([\d.]+)\])?"
    r".*?([\d,]+)\s+tok/s"
)

# [val] step  15000  val_loss 1.2678  val_ppl    3.55
_VAL_RE = re.compile(
    r"\[val\]\s+step\s+(\d+)\s+"
    r"val_loss\s+([\d.]+)\s+"
    r"val_ppl\s+([\d.]+)"
)

# Resumed from step 15000
_RESUME_RE = re.compile(r"Resumed from step\s+(\d+)")


def parse_log(path: Path) -> ParsedLog:
    result = ParsedLog(path=path)
    with open(path) as fh:
        for line in fh:
            line = line.strip()

            m = _TRAIN_RE.match(line)
            if m:
                wr_mean = float(m.group(5)) if m.group(5) else None
                wr_min  = float(m.group(6)) if m.group(6) else None
                wr_max  = float(m.group(7)) if m.group(7) else None
                toks    = float(m.group(8).replace(",", ""))
                result.train_rows.append(TrainRow(
                    step=int(m.group(1)),
                    loss=float(m.group(2)),
                    ppl=float(m.group(3)),
                    lr=float(m.group(4)),
                    toks_per_sec=toks,
                    wr_mean=wr_mean,
                    wr_min=wr_min,
                    wr_max=wr_max,
                ))
                if wr_mean is not None:
                    result.has_write_rates = True
                continue

            m = _VAL_RE.search(line)
            if m:
                result.val_rows.append(ValRow(
                    step=int(m.group(1)),
                    val_loss=float(m.group(2)),
                    val_ppl=float(m.group(3)),
                ))
                continue

            m = _RESUME_RE.search(line)
            if m and result.resumed_from is None:
                result.resumed_from = int(m.group(1))

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Reports
# ─────────────────────────────────────────────────────────────────────────────

def _header(title: str) -> None:
    print(f"\n{'─' * 70}")
    print(f"  {title}")
    print(f"{'─' * 70}")


def report_val_table(log: ParsedLog, label: str) -> None:
    _header(f"Validation perplexity — {label} ({log.path.name})")
    if not log.val_rows:
        print("  (no validation entries found)")
        return
    print(f"  {'Step':>8}  {'val_loss':>10}  {'val_ppl':>10}")
    print(f"  {'──────':>8}  {'────────':>10}  {'───────':>10}")
    for r in log.val_rows:
        print(f"  {r.step:>8}  {r.val_loss:>10.4f}  {r.val_ppl:>10.2f}")
    best = min(log.val_rows, key=lambda r: r.val_ppl)
    print(f"\n  Best val_ppl: {best.val_ppl:.2f} at step {best.step}")
    if log.resumed_from is not None:
        print(f"  (resumed from step {log.resumed_from})")


def report_wr_convergence(log: ParsedLog, converge_step: int) -> bool:
    """
    Print write-rate trajectory and return True if wr is in [0.10, 0.85]
    at or before *converge_step*, False otherwise.
    """
    _header(f"Write-rate convergence — {log.path.name}")
    if not log.has_write_rates:
        print("  (no write-rate entries — this is a baseline log)")
        return True

    rows_with_wr = [r for r in log.train_rows if r.wr_mean is not None]
    if not rows_with_wr:
        print("  (no write-rate data found in training steps)")
        return True

    print(f"  {'Step':>8}  {'wr_mean':>9}  {'wr_min':>7}  {'wr_max':>7}  {'in range':>9}")
    print(f"  {'──────':>8}  {'───────':>9}  {'─────':>7}  {'─────':>7}  {'────────':>9}")

    # Report at every 5000-step boundary that exists in the log
    milestones = {0, converge_step}
    milestones |= {r.step for r in rows_with_wr if r.step % 5000 == 0}
    milestones |= {rows_with_wr[-1].step}

    reported_steps: set[int] = set()
    converged_at: Optional[int] = None

    for r in rows_with_wr:
        in_range = WRITE_RATE_LO <= r.wr_mean <= WRITE_RATE_HI
        if in_range and converged_at is None:
            converged_at = r.step

        if r.step in milestones and r.step not in reported_steps:
            flag = "✓" if in_range else "✗"
            print(
                f"  {r.step:>8}  {r.wr_mean:>9.3f}  "
                f"{r.wr_min:>7.3f}  {r.wr_max:>7.3f}  {flag:>9}"
            )
            reported_steps.add(r.step)

    # Final entry if not already shown
    last = rows_with_wr[-1]
    if last.step not in reported_steps:
        in_range = WRITE_RATE_LO <= last.wr_mean <= WRITE_RATE_HI
        flag = "✓" if in_range else "✗"
        print(
            f"  {last.step:>8}  {last.wr_mean:>9.3f}  "
            f"{last.wr_min:>7.3f}  {last.wr_max:>7.3f}  {flag:>9}"
        )

    print()
    if converged_at is not None:
        print(f"  CONVERGED: wr first entered [0.10, 0.85] at step {converged_at}")
    else:
        print(
            f"  NOT YET CONVERGED: wr has not reached [0.10, 0.85] "
            f"in {last.step} logged steps"
        )

    # Verdict against the requested milestone
    step_at_milestone = next(
        (r for r in rows_with_wr if r.step >= converge_step), None
    )
    if step_at_milestone is None:
        print(
            f"\n  NOTE: log does not yet reach step {converge_step} "
            f"— verdict deferred"
        )
        return True  # not a failure, run is still in progress

    ok = WRITE_RATE_LO <= step_at_milestone.wr_mean <= WRITE_RATE_HI
    verdict = "PASS" if ok else "FAIL"
    print(
        f"\n  Phase 18 verdict at step {converge_step}: {verdict}  "
        f"(wr={step_at_milestone.wr_mean:.3f}, "
        f"target=[{WRITE_RATE_LO}, {WRITE_RATE_HI}])"
    )
    return ok


def report_comparison(baseline: ParsedLog, memory: ParsedLog) -> None:
    """Side-by-side val_ppl comparison at shared checkpoints."""
    _header(
        f"Side-by-side comparison — "
        f"{baseline.path.name} vs {memory.path.name}"
    )
    base_map = {r.step: r for r in baseline.val_rows}
    mem_map  = {r.step: r for r in memory.val_rows}
    shared   = sorted(set(base_map) & set(mem_map))
    if not shared:
        print("  (no shared validation steps yet)")
        return

    print(f"  {'Step':>8}  {'baseline val_ppl':>18}  {'memory val_ppl':>16}  {'Δ ppl':>8}")
    print(f"  {'──────':>8}  {'────────────────':>18}  {'──────────────':>16}  {'─────':>8}")
    for step in shared:
        b_ppl = base_map[step].val_ppl
        m_ppl = mem_map[step].val_ppl
        delta  = m_ppl - b_ppl
        sign   = "+" if delta > 0 else ""
        print(f"  {step:>8}  {b_ppl:>18.2f}  {m_ppl:>16.2f}  {sign}{delta:>7.2f}")


def report_throughput(log: ParsedLog, label: str) -> None:
    _header(f"Throughput — {label}")
    if not log.train_rows:
        print("  (no training steps found)")
        return
    toks = [r.toks_per_sec for r in log.train_rows]
    print(f"  Mean:   {sum(toks)/len(toks):>10,.0f} tok/s")
    print(f"  Median: {sorted(toks)[len(toks)//2]:>10,.0f} tok/s")
    print(f"  Min:    {min(toks):>10,.0f} tok/s")
    print(f"  Max:    {max(toks):>10,.0f} tok/s")
    print(f"  Steps logged: {len(log.train_rows)}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "log",
        nargs="?",
        help="Training log to analyse (standalone mode)",
    )
    p.add_argument(
        "--baseline",
        metavar="PATH",
        help="Baseline log (no episodic memory) for comparison",
    )
    p.add_argument(
        "--memory",
        metavar="PATH",
        help="Memory log (episodic memory) for comparison + write-rate report",
    )
    p.add_argument(
        "--wr-converge-step",
        metavar="N",
        type=int,
        default=30000,
        help="Step at which write-rate convergence is assessed (default: 30000)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    converged_ok = True

    if args.log:
        # Standalone mode: analyse a single log
        log = parse_log(Path(args.log))
        label = log.path.stem
        report_val_table(log, label)
        report_throughput(log, label)
        if log.has_write_rates:
            ok = report_wr_convergence(log, args.wr_converge_step)
            converged_ok = converged_ok and ok

    if args.baseline or args.memory:
        # Comparison mode
        if args.baseline:
            baseline = parse_log(Path(args.baseline))
            report_val_table(baseline, "baseline (Exp A)")
            report_throughput(baseline, "baseline (Exp A)")
        if args.memory:
            memory = parse_log(Path(args.memory))
            report_val_table(memory, "memory (Exp B)")
            report_throughput(memory, "memory (Exp B)")
            ok = report_wr_convergence(memory, args.wr_converge_step)
            converged_ok = converged_ok and ok
        if args.baseline and args.memory:
            report_comparison(baseline, memory)

    if not args.log and not args.baseline and not args.memory:
        _build_parser().print_help()
        return 2

    print()
    return 0 if converged_ok else 1


if __name__ == "__main__":
    sys.exit(main())
