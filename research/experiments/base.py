"""
Base experiment class for drex memory controller research.

Every experiment inherits from Experiment and implements run().
Results are written to research/results/ as JSON automatically.
"""

from __future__ import annotations

import json
import random
import time
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

OUTCOME_SUPPORTED = "SUPPORTED"
OUTCOME_REFUTED = "REFUTED"
OUTCOME_INCONCLUSIVE = "INCONCLUSIVE"
OUTCOME_ERROR = "ERROR"


@dataclass
class ExperimentResult:
    experiment_id: str
    hypothesis: str
    outcome: str                          # SUPPORTED / REFUTED / INCONCLUSIVE / ERROR
    metrics: dict[str, Any] = field(default_factory=dict)
    notes: str = ""
    error: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    duration_seconds: float = 0.0
    config: dict[str, Any] = field(default_factory=dict)
    seed: int = -1

    def save(self) -> Path:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        seed_tag = f"_s{self.seed}" if self.seed >= 0 else ""
        path = RESULTS_DIR / f"{self.experiment_id}{seed_tag}_{ts}.json"
        path.write_text(json.dumps(asdict(self), indent=2))
        return path


class Experiment(ABC):
    """
    Base class for all research experiments.

    Subclasses must define:
        experiment_id: str       — unique identifier e.g. "exp_1_5"
        hypothesis: str          — the falsifiable statement being tested

    Subclasses must implement:
        run() -> ExperimentResult
    """

    experiment_id: str = "undefined"
    hypothesis: str = "undefined"

    def execute(self, seed: int = 42) -> ExperimentResult:
        """Run the experiment, catch errors, print summary, save result."""
        # Set all random seeds for reproducibility
        try:
            import torch
            torch.manual_seed(seed)
        except ImportError:
            pass
        try:
            import numpy as np
            np.random.seed(seed)
        except ImportError:
            pass
        random.seed(seed)

        print(f"\n{'='*60}")
        print(f"Experiment: {self.experiment_id}  seed={seed}")
        print(f"Hypothesis: {self.hypothesis}")
        print(f"{'='*60}")

        t0 = time.perf_counter()
        try:
            result = self.run()
        except Exception as e:
            duration = time.perf_counter() - t0
            tb = traceback.format_exc()
            result = ExperimentResult(
                experiment_id=self.experiment_id,
                hypothesis=self.hypothesis,
                outcome=OUTCOME_ERROR,
                error=tb,
                duration_seconds=duration,
                seed=seed,
            )
            print(f"\nERROR: {e}")
            print(tb)

        result.experiment_id = self.experiment_id
        result.hypothesis = self.hypothesis
        result.duration_seconds = time.perf_counter() - t0
        result.seed = seed

        path = result.save()

        print(f"\nOutcome:  {result.outcome}")
        if result.metrics:
            print("Metrics:")
            for k, v in result.metrics.items():
                print(f"  {k}: {v}")
        if result.notes:
            print(f"Notes:    {result.notes}")
        print(f"Saved to: {path.name}")
        print(f"Duration: {result.duration_seconds:.1f}s")

        return result

    @abstractmethod
    def run(self) -> ExperimentResult:
        """Implement the actual experiment. Return an ExperimentResult."""
        ...

    def result(
        self,
        outcome: str,
        metrics: dict[str, Any] | None = None,
        notes: str = "",
        config: dict[str, Any] | None = None,
    ) -> ExperimentResult:
        """Convenience helper for building a result from within run()."""
        return ExperimentResult(
            experiment_id=self.experiment_id,
            hypothesis=self.hypothesis,
            outcome=outcome,
            metrics=metrics or {},
            notes=notes,
            config=config or {},
        )
