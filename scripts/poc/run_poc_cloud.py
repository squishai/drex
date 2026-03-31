#!/usr/bin/env python3
"""
scripts/poc/run_poc_cloud.py — Cloud GPU runner for all 5 DREX-UNIFIED POC sprints.

Usage:
    # Run Sprint 1 (baseline) with seeds 42, 43, 44:
    python scripts/poc/run_poc_cloud.py --sprint 1

    # Run Sprint 2 (Mamba) with custom seeds:
    python scripts/poc/run_poc_cloud.py --sprint 2 --seeds 42 43 44

    # Run Sprint 5 (scale, seed 42 only) with GPU batch size:
    python scripts/poc/run_poc_cloud.py --sprint 5 --seeds 42 --batch-size 32

    # Run all 5 sprints sequentially:
    python scripts/poc/run_poc_cloud.py --all --batch-size 32

Environment:
    Run from the root of the drex repo after cloning.
    Requires: pip install datasets safetensors torch

Sprint → exp mapping:
    Sprint 1 = exp_poc_a  (baseline transformer)
    Sprint 2 = exp_poc_b  (+ Mamba SSM backbone)
    Sprint 3 = exp_poc_c  (+ ESN episodic memory)
    Sprint 4 = exp_poc_d  (+ HDC encoder)
    Sprint 5 = exp_poc_e  (scale: d=256, 8L, 50k steps, seed 42 only)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Sprint configuration tables.
# Sprints 1–4 share base arch flags (d=128, 4L). Sprint 5 scales up.
# ---------------------------------------------------------------------------

_SPRINT_NAMES = {
    1: "exp_poc_a",
    2: "exp_poc_b",
    3: "exp_poc_c",
    4: "exp_poc_d",
    5: "exp_poc_e",
}

_SPRINT_DESCRIPTIONS = {
    1: "Baseline transformer (L1 SWA + L2 InfiniAttention)",
    2: "Mamba SSM backbone replacing L1 SWA",
    3: "Mamba + ESN episodic memory",
    4: "Mamba + ESN + HDC encoder (full DREX-UNIFIED core)",
    5: "Scale: d=256, 8L, 512-seg, 50k steps (proof run)",
}

_DEFAULT_SEEDS = {
    1: [42, 43, 44],
    2: [42, 43, 44],
    3: [42, 43, 44],
    4: [42, 43, 44],
    5: [42],  # Sprint 5 is single-seed — it's a 50k-step scale run
}

# ALL sprints must use batch_size ≤ 8.
#
# Background: LR=3e-4 is tuned for batch=8.  Running batch=32 with the same LR
# causes gradient instability on CUDA that is NOT caught by the [skip] NaN
# guard (unlike on MPS where NaN is generated and skipped).  Cloud Sprint 1
# with batch=32 converged to val_ppl=3.98 instead of the expected ~1.12,
# reproducing locally as NaN on 12/13 steps.  Fixing to batch=8 makes cloud
# results directly comparable to local experiments.
#
# Sprint 5 additionally has a VRAM constraint: segment-len=512 with Mamba
# creates large (B,S,d_inner,d_state) tensors that overflow T4 at batch>8.
_MAX_BATCH = 8

# Base architecture flags (override per sprint where needed)
_BASE_FLAGS_S1_4 = [
    "--d-model", "128",
    "--n-heads", "4",
    "--n-layers", "4",
    "--ff-mult", "4",
    "--segment-len", "128",
    "--steps", "10000",
    "--val-every", "500",
    "--log-every", "100",
    "--save-every", "5000",
]

_BASE_FLAGS_S5 = [
    "--d-model", "256",
    "--n-heads", "4",
    "--n-layers", "8",
    "--ff-mult", "4",
    "--segment-len", "512",
    "--steps", "50000",
    "--val-every", "1000",
    "--log-every", "200",
    "--save-every", "5000",
]

# Extra component flags toggled per sprint
_SPRINT_EXTRA_FLAGS: dict[int, list[str]] = {
    1: [],
    2: [
        "--use-mamba",
        "--mamba-d-state", "16",
        "--mamba-d-conv", "4",
        "--mamba-expand", "2",
    ],
    3: [
        "--use-mamba",
        "--mamba-d-state", "16",
        "--mamba-d-conv", "4",
        "--mamba-expand", "2",
        "--use-episodic-memory",
        "--use-esn-memory",
        "--esn-reservoir-mult", "4",
        "--esn-spectral-radius", "0.95",
    ],
    4: [
        "--use-mamba",
        "--mamba-d-state", "16",
        "--mamba-d-conv", "4",
        "--mamba-expand", "2",
        "--use-episodic-memory",
        "--use-esn-memory",
        "--esn-reservoir-mult", "4",
        "--esn-spectral-radius", "0.95",
        "--use-hdc-encoder",
        "--hdc-dim", "512",
        "--hdc-seed", "0",
    ],
    5: [
        "--use-mamba",
        "--mamba-d-state", "16",
        "--mamba-d-conv", "4",
        "--mamba-expand", "2",
        "--use-episodic-memory",
        "--use-esn-memory",
        "--esn-reservoir-mult", "4",
        "--esn-spectral-radius", "0.95",
        "--use-hdc-encoder",
        "--hdc-dim", "1024",
        "--hdc-seed", "0",
    ],
}


def _base_flags(sprint: int) -> list[str]:
    return _BASE_FLAGS_S5 if sprint == 5 else _BASE_FLAGS_S1_4


def run_sprint(
    sprint: int,
    seeds: list[int],
    out_dir: Path,
    batch_size: int,
    device: str,
    repo_root: Path,
) -> dict:
    """
    Run one sprint over all requested seeds.

    Returns a summary dict with per-seed results and the median final val_ppl.
    """
    name = _SPRINT_NAMES[sprint]
    description = _SPRINT_DESCRIPTIONS[sprint]
    print(f"\n{'='*70}", flush=True)
    print(f"Sprint {sprint}: {description}", flush=True)
    print(f"  Name:   {name}", flush=True)
    print(f"  Seeds:  {seeds}", flush=True)
    print(f"  OutDir: {out_dir}", flush=True)
    print(f"{'='*70}\n", flush=True)

    sprint_summary: dict = {
        "sprint": sprint,
        "name": name,
        "description": description,
        "seeds": seeds,
        "results": {},
    }

    ckpt_base = out_dir / "checkpoints"
    log_dir = out_dir / "logs" / name
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_base.mkdir(parents=True, exist_ok=True)

    effective_batch = batch_size
    if batch_size > _MAX_BATCH:
        reason = (
            "Mamba segment-len=512 on T4/P100 overflows VRAM at larger sizes"
            if sprint == 5
            else "LR=3e-4 is tuned for batch=8; larger batches cause gradient "
                 "instability on CUDA (validated: batch=32 → val_ppl=3.98 vs "
                 "expected ~1.12)"
        )
        print(
            f"Sprint {sprint}: capping batch_size {batch_size} → {_MAX_BATCH} "
            f"({reason})",
            flush=True,
        )
        effective_batch = _MAX_BATCH

    for seed in seeds:
        seed_name = f"{name}_s{seed}"
        ckpt_dir = ckpt_base / seed_name
        log_file = log_dir / f"{seed_name}.log"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            str(repo_root / "scripts" / "train.py"),
            *_base_flags(sprint),
            *_SPRINT_EXTRA_FLAGS[sprint],
            "--batch-size", str(effective_batch),
            "--device", device,
            "--seed", str(seed),
            "--ckpt-dir", str(ckpt_dir),
        ]

        print(f"[seed {seed}] Starting: {' '.join(cmd)}", flush=True)
        print(f"[seed {seed}] Log: {log_file}\n", flush=True)

        import os
        env = os.environ.copy()
        if sprint == 5 and device == "cuda":
            # Reduce CUDA memory fragmentation — helps with many small Mamba SSM allocs
            env.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

        start_ts = time.time()
        last_val_ppl: float | None = None

        with open(log_file, "w") as log_fh:
            proc = subprocess.Popen(
                cmd,
                cwd=str(repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                # Live stream to terminal and file simultaneously
                print(line, end="", flush=True)
                log_fh.write(line)
                log_fh.flush()
                # Parse val_ppl from log lines.
                # train.py emits: "val_ppl  4.01" (space-padded, not colon or equals)
                # Also accept legacy: "val_ppl=X.XX" or "val_ppl: X.XX"
                for prefix in ("val_ppl ", "val_ppl=", "val_ppl: "):
                    if prefix in line:
                        try:
                            val_str = line.split(prefix, 1)[1].strip().split()[0].rstrip(",")
                            last_val_ppl = float(val_str)
                        except (ValueError, IndexError):
                            pass
                        break

            proc.wait()

        elapsed = time.time() - start_ts
        returncode = proc.returncode

        seed_result = {
            "seed": seed,
            "returncode": returncode,
            "elapsed_s": round(elapsed, 1),
            "final_val_ppl": last_val_ppl,
            "log": str(log_file),
            "ckpt_dir": str(ckpt_dir),
        }
        sprint_summary["results"][str(seed)] = seed_result

        status = "OK" if returncode == 0 else f"FAILED (rc={returncode})"
        ppl_str = f"{last_val_ppl:.4f}" if last_val_ppl is not None else "n/a"
        print(f"\n[seed {seed}] {status}  val_ppl={ppl_str}  elapsed={elapsed:.0f}s\n", flush=True)

    # Compute median final val_ppl across seeds that completed successfully
    ppls = [
        r["final_val_ppl"]
        for r in sprint_summary["results"].values()
        if r["final_val_ppl"] is not None and r["returncode"] == 0
    ]
    if ppls:
        ppls.sort()
        n = len(ppls)
        median_ppl = ppls[n // 2] if n % 2 != 0 else (ppls[n // 2 - 1] + ppls[n // 2]) / 2.0
        sprint_summary["median_val_ppl"] = round(median_ppl, 4)
    else:
        sprint_summary["median_val_ppl"] = None

    # Save per-sprint JSON summary
    summary_path = out_dir / f"{name}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(sprint_summary, f, indent=2)
    print(f"\nSummary saved → {summary_path}", flush=True)
    print(f"Sprint {sprint} median val_ppl: {sprint_summary['median_val_ppl']}\n", flush=True)

    return sprint_summary


def print_results_table(summaries: list[dict]) -> None:
    """Print a compact sprint comparison table to stdout."""
    print("\n" + "=" * 70, flush=True)
    print("POC SPRINT RESULTS SUMMARY", flush=True)
    print("=" * 70, flush=True)
    header = f"{'Sprint':<8} {'Name':<14} {'Median val_ppl':>16} {'Description'}"
    print(header, flush=True)
    print("-" * 70, flush=True)
    baseline_ppl: float | None = None
    for s in summaries:
        ppl = s.get("median_val_ppl")
        ppl_str = f"{ppl:.4f}" if ppl is not None else "n/a"
        if s["sprint"] == 1:
            baseline_ppl = ppl
        delta_str = ""
        if s["sprint"] > 1 and ppl is not None and baseline_ppl is not None:
            delta = ppl - baseline_ppl
            delta_str = f"  ({delta:+.4f} vs S1)"
        print(f"{s['sprint']:<8} {s['name']:<14} {ppl_str:>16}{delta_str}   {s['description']}", flush=True)
    print("=" * 70, flush=True)


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Cloud runner for all 5 DREX-UNIFIED POC sprints.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--sprint",
        type=int,
        choices=[1, 2, 3, 4, 5],
        help="Run a single sprint (1–5).",
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Run all 5 sprints sequentially.",
    )
    p.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Seeds to run. Default: [42, 43, 44] for sprints 1–4; [42] for sprint 5."
            " Ignored when --all is set (each sprint uses its default seeds)."
        ),
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help=(
            "Root output directory for logs, checkpoints, and JSON summaries. "
            "Default: auto-detected from RESULTS_ROOT env var, or /tmp/drex_poc."
        ),
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Training batch size. Use 32 on cloud GPU (T4/P100), 8 on CPU/MPS.",
    )
    p.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="PyTorch device. 'auto' picks CUDA > MPS > CPU in that order.",
    )
    p.add_argument(
        "--repo-root",
        type=str,
        default=None,
        help=(
            "Path to the drex repo root. Defaults to the parent of this file's "
            "grandparent directory (i.e. the standard scripts/poc/ layout)."
        ),
    )
    return p


def _resolve_out_dir(cli_value: str | None) -> Path:
    """Resolve the output directory with fallback chain."""
    import os

    if cli_value is not None:
        return Path(cli_value)
    env_val = os.environ.get("RESULTS_ROOT")
    if env_val:
        return Path(env_val) / "drex_poc"
    # Kaggle: /kaggle/working exists and is writable
    kaggle_path = Path("/kaggle/working")
    if kaggle_path.exists():
        return kaggle_path / "drex_poc"
    # Colab: /content/drive/MyDrive exists when Drive is mounted
    colab_path = Path("/content/drive/MyDrive/drex_poc")
    if Path("/content/drive/MyDrive").exists():
        return colab_path
    return Path("/tmp/drex_poc")


def main() -> None:
    args = _parser().parse_args()

    # Resolve repo root
    if args.repo_root is not None:
        repo_root = Path(args.repo_root).resolve()
    else:
        # scripts/poc/run_poc_cloud.py → scripts/poc → scripts → repo root
        repo_root = Path(__file__).resolve().parent.parent.parent
    if not (repo_root / "scripts" / "train.py").exists():
        print(
            f"ERROR: scripts/train.py not found under repo_root={repo_root}\n"
            "Pass --repo-root /path/to/drex if running from a different directory.",
            file=sys.stderr,
        )
        sys.exit(1)

    out_dir = _resolve_out_dir(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sprints_to_run = list(range(1, 6)) if args.all else [args.sprint]
    all_summaries: list[dict] = []

    with open(out_dir / "overall_summary.json", "w") as _:
        pass  # touch the file early so callers know where to look

    for sprint in sprints_to_run:
        seeds = _DEFAULT_SEEDS[sprint] if args.all or args.seeds is None else args.seeds
        summary = run_sprint(
            sprint=sprint,
            seeds=seeds,
            out_dir=out_dir,
            batch_size=args.batch_size,
            device=args.device,
            repo_root=repo_root,
        )
        all_summaries.append(summary)
        # Write cumulative overall summary after each sprint so crash-safe
        with open(out_dir / "overall_summary.json", "w") as f:
            json.dump(all_summaries, f, indent=2)

    if len(all_summaries) > 1:
        print_results_table(all_summaries)

    print(f"\nAll done. Results in: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
