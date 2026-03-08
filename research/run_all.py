"""
run_all.py — Multi-seed runner for all drex research experiments.

Discovers every experiment in experiments/cat*/, runs each 3 times with
different seeds, and writes timestamped JSON results to results/.

Usage:
    python3 run_all.py                  # run all experiments, 3 seeds
    python3 run_all.py --seeds 42       # single seed run
    python3 run_all.py --filter exp_1   # only experiments whose ID contains "exp_1"
    python3 run_all.py --dry-run        # list experiments without running

Progress is printed to stdout. Errors are caught per-run and saved as ERROR
results so the full suite always completes.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

# Ensure the research/ directory is on the path
RESEARCH_DIR = Path(__file__).parent
sys.path.insert(0, str(RESEARCH_DIR))

from experiments.base import Experiment, OUTCOME_ERROR

SEEDS = [42, 123, 777]


# ── Discovery ─────────────────────────────────────────────────────────────────

def discover_experiment_files() -> list[Path]:
    """Return all experiment Python files, sorted by category then number."""
    files = sorted(
        RESEARCH_DIR.glob("experiments/cat*_*/exp_*.py"),
        key=lambda p: (p.parent.name, p.name),
    )
    return [f for f in files if not f.name.startswith("__")]


def load_experiment_class(filepath: Path) -> type[Experiment] | None:
    """Dynamically import a file and return the first Experiment subclass found."""
    spec = importlib.util.spec_from_file_location(filepath.stem, filepath)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        print(f"  IMPORT ERROR {filepath.name}: {e}")
        return None

    for obj in vars(module).values():
        if (
            isinstance(obj, type)
            and issubclass(obj, Experiment)
            and obj is not Experiment
            and getattr(obj, "experiment_id", "undefined") != "undefined"
        ):
            return obj
    return None


# ── Runner ────────────────────────────────────────────────────────────────────

def run_all(seeds: list[int], filter_str: str = "", dry_run: bool = False) -> None:
    files = discover_experiment_files()
    if filter_str:
        files = [f for f in files if filter_str in f.name]

    print(f"\ndrex experiment runner")
    print(f"Seeds:       {seeds}")
    print(f"Experiments: {len(files)}")
    print(f"Total runs:  {len(files) * len(seeds)}")
    if dry_run:
        print("\n--- DRY RUN (listing only) ---")
        for f in files:
            cls = load_experiment_class(f)
            name = cls.experiment_id if cls else "(no class found)"
            print(f"  {name:20s}  {f.relative_to(RESEARCH_DIR)}")
        return

    total = len(files) * len(seeds)
    n = 0
    t_start = time.perf_counter()

    skipped = []
    errors  = []

    for filepath in files:
        ExpClass = load_experiment_class(filepath)
        if ExpClass is None:
            print(f"\n[SKIP] No experiment class in {filepath.name}")
            skipped.append(filepath.name)
            continue

        for seed in seeds:
            n += 1
            elapsed = time.perf_counter() - t_start
            print(
                f"\n[{n}/{total}]  {ExpClass.experiment_id}  seed={seed}"
                f"  elapsed={elapsed:.0f}s"
            )
            try:
                result = ExpClass().execute(seed=seed)
                if result.outcome == OUTCOME_ERROR:
                    errors.append(f"{ExpClass.experiment_id} seed={seed}")
            except Exception as e:
                print(f"  RUNNER ERROR: {e}")
                errors.append(f"{ExpClass.experiment_id} seed={seed} (runner)")

    elapsed_total = time.perf_counter() - t_start
    print(f"\n{'='*60}")
    print(f"DONE — {total} runs in {elapsed_total:.0f}s")
    print(f"Skipped:  {len(skipped)}")
    print(f"Errors:   {len(errors)}")
    if skipped:
        for s in skipped:
            print(f"  skip: {s}")
    if errors:
        for e in errors:
            print(f"  err:  {e}")
    print(f"\nResults saved to: {RESEARCH_DIR / 'results'}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run all drex experiments")
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=SEEDS,
        help="Random seeds to use (default: 42 123 777)",
    )
    parser.add_argument(
        "--filter", default="",
        help="Only run experiments whose filename contains this string",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="List experiments without running them",
    )
    args = parser.parse_args()
    run_all(seeds=args.seeds, filter_str=args.filter, dry_run=args.dry_run)
