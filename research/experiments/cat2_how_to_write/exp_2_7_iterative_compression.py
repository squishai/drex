"""
Experiment 2.7 — Iterative Compression (Hierarchical Memory)

Hypothesis: A hierarchy of increasingly abstract memory levels can be built by
iterative compression without catastrophic information loss at each stage.

Setup:
  - Three compression stages, each 4x:
      Stage 1: 256-dim → 64-dim  (total: 4x from original)
      Stage 2: 64-dim  → 16-dim  (total: 16x from original)
      Stage 3: 16-dim  → 4-dim   (total: 64x from original)
  - Each stage is an autoencoder trained independently.
  - Stage k is trained on the ENCODER output of stage k-1.
  - Cosine similarity to the original 256-dim input is measured at each stage
    by cascading the decoders: x → enc1 → enc2 → enc3 → dec3 → dec2 → dec1 → x_hat
  - SUPPORTED if cosim_stage2 > 0.3 (not catastrophic after 16x total compression)
  - REFUTED   if cosim_stage2 < 0.1 (catastrophic loss by stage 2)
  - INCONCLUSIVE otherwise
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from experiments.base import (
    Experiment, ExperimentResult,
    OUTCOME_SUPPORTED, OUTCOME_REFUTED, OUTCOME_INCONCLUSIVE,
)

# ── Config ────────────────────────────────────────────────────────────────────

VOCAB_SIZE   = 256
HIDDEN_DIM   = 64
INPUT_DIM    = 256           # raw input dimension for stage 1
STAGE_DIMS   = [256, 64, 16, 4]   # [input, after_stage1, after_stage2, after_stage3]
BATCH_SIZE   = 32
TRAIN_STEPS  = 2000
EVAL_BATCHES = 200
LR           = 1e-3
DEVICE       = "cpu"
SUPPORT_THRESHOLD = 0.3    # cosim_stage2 > this → SUPPORTED
REFUTE_THRESHOLD  = 0.1    # cosim_stage2 < this → REFUTED


# ── Model ─────────────────────────────────────────────────────────────────────

class StageAutoencoder(nn.Module):
    """Single compression stage: in_dim → out_dim → in_dim."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        mid = max(out_dim * 2, in_dim // 2, 16)
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, mid),
            nn.ReLU(),
            nn.Linear(mid, out_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(out_dim, mid),
            nn.ReLU(),
            nn.Linear(mid, in_dim),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        return z, self.decode(z)


# ── Data helpers ──────────────────────────────────────────────────────────────

def make_raw_batch(batch_size: int) -> torch.Tensor:
    """Random INPUT_DIM-dim float vectors."""
    return torch.randn(batch_size, INPUT_DIM)


# ── Training ──────────────────────────────────────────────────────────────────

def train_stage(
    model: StageAutoencoder,
    prev_encoder: nn.Module | None,
    stage_idx: int,
) -> None:
    """
    Train stage (stage_idx+1) on the ENCODER output of the previous stage.
    If prev_encoder is None, trains directly on raw INPUT_DIM data.
    """
    opt = Adam(model.parameters(), lr=LR)
    for step in range(TRAIN_STEPS):
        x = make_raw_batch(BATCH_SIZE)

        # Feed through all previous encoders (frozen)
        with torch.no_grad():
            if prev_encoder is not None:
                x = prev_encoder(x)

        _, x_hat = model(x)
        loss     = F.mse_loss(x_hat, x)

        opt.zero_grad()
        loss.backward()
        opt.step()

        if (step + 1) % 500 == 0:
            print(f"    [stage {stage_idx}] step {step+1}/{TRAIN_STEPS}  "
                  f"loss={loss.item():.4f}")


class CascadedEncoder(nn.Module):
    """Runs multiple encoders in sequence (for building prev_encoder arg)."""

    def __init__(self, stages: list[StageAutoencoder]) -> None:
        super().__init__()
        self.stages = nn.ModuleList(stages)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for s in self.stages:
            x = s.encode(x)
        return x


# ── Evaluation ────────────────────────────────────────────────────────────────

def eval_cosim_at_stage(
    stages: list[StageAutoencoder],
    num_stages: int,
) -> float:
    """
    Compress through `num_stages` encoders, then decompress through all decoders
    in reverse, and compute cosine similarity with the original input.
    """
    sims: list[float] = []
    with torch.no_grad():
        for _ in range(EVAL_BATCHES):
            x = make_raw_batch(BATCH_SIZE)
            z = x
            # encode through num_stages stages
            for i in range(num_stages):
                z = stages[i].encode(z)
            # decode in reverse
            for i in range(num_stages - 1, -1, -1):
                z = stages[i].decode(z)
            x_hat = z
            sims.append(F.cosine_similarity(x, x_hat, dim=-1).mean().item())
    return sum(sims) / len(sims)


# ── Experiment ────────────────────────────────────────────────────────────────

class Exp27IterativeCompression(Experiment):
    experiment_id = "exp_2_7"
    hypothesis = (
        "A hierarchy of increasingly abstract memory levels can be built by iterative "
        "compression without catastrophic information loss at each stage."
    )

    def run(self) -> ExperimentResult:
        torch.manual_seed(42)

        # Build three autoencoders for stages 1, 2, 3
        stage1 = StageAutoencoder(STAGE_DIMS[0], STAGE_DIMS[1]).to(DEVICE)  # 256→64
        stage2 = StageAutoencoder(STAGE_DIMS[1], STAGE_DIMS[2]).to(DEVICE)  # 64→16
        stage3 = StageAutoencoder(STAGE_DIMS[2], STAGE_DIMS[3]).to(DEVICE)  # 16→4

        print(f"\nStage 1: {STAGE_DIMS[0]}-dim → {STAGE_DIMS[1]}-dim (4x)")
        train_stage(stage1, prev_encoder=None, stage_idx=1)

        print(f"\nStage 2: {STAGE_DIMS[1]}-dim → {STAGE_DIMS[2]}-dim (4x, 16x total)")
        enc1 = CascadedEncoder([stage1])
        for p in enc1.parameters():
            p.requires_grad_(False)
        train_stage(stage2, prev_encoder=enc1, stage_idx=2)

        print(f"\nStage 3: {STAGE_DIMS[2]}-dim → {STAGE_DIMS[3]}-dim (4x, 64x total)")
        enc12 = CascadedEncoder([stage1, stage2])
        for p in enc12.parameters():
            p.requires_grad_(False)
        train_stage(stage3, prev_encoder=enc12, stage_idx=3)

        print("\nEvaluating cosine similarity at each stage...")
        for s in [stage1, stage2, stage3]:
            s.eval()

        stages = [stage1, stage2, stage3]
        cosim_stage1 = eval_cosim_at_stage(stages, num_stages=1)
        cosim_stage2 = eval_cosim_at_stage(stages, num_stages=2)
        cosim_stage3 = eval_cosim_at_stage(stages, num_stages=3)

        print(f"  cosim_stage1={cosim_stage1:.4f}  (4x total)")
        print(f"  cosim_stage2={cosim_stage2:.4f}  (16x total)")
        print(f"  cosim_stage3={cosim_stage3:.4f}  (64x total)")

        # Information retention per stage (fraction of previous stage's quality retained)
        ret_1 = cosim_stage1
        ret_2 = cosim_stage2 / cosim_stage1 if cosim_stage1 > 1e-9 else 0.0
        ret_3 = cosim_stage3 / cosim_stage2 if cosim_stage2 > 1e-9 else 0.0

        if cosim_stage2 > SUPPORT_THRESHOLD:
            outcome = OUTCOME_SUPPORTED
        elif cosim_stage2 < REFUTE_THRESHOLD:
            outcome = OUTCOME_REFUTED
        else:
            outcome = OUTCOME_INCONCLUSIVE

        metrics = {
            "cosim_stage1": cosim_stage1,
            "cosim_stage2": cosim_stage2,
            "cosim_stage3": cosim_stage3,
            "info_retention_pct_per_stage": {
                "stage1": round(ret_1, 4),
                "stage2": round(ret_2, 4),
                "stage3": round(ret_3, 4),
            },
            "total_compression_stage1": STAGE_DIMS[0] // STAGE_DIMS[1],
            "total_compression_stage2": STAGE_DIMS[0] // STAGE_DIMS[2],
            "total_compression_stage3": STAGE_DIMS[0] // STAGE_DIMS[3],
        }
        notes = (
            f"Cosine similarities after iterative compression: "
            f"stage1={cosim_stage1:.3f} (4x), "
            f"stage2={cosim_stage2:.3f} (16x), "
            f"stage3={cosim_stage3:.3f} (64x). "
            f"Stage2 above support threshold {SUPPORT_THRESHOLD}: "
            f"{cosim_stage2 > SUPPORT_THRESHOLD}. "
            f"Stage2 below refute threshold {REFUTE_THRESHOLD}: "
            f"{cosim_stage2 < REFUTE_THRESHOLD}."
        )
        config = {
            "stage_dims": STAGE_DIMS, "train_steps": TRAIN_STEPS,
            "support_threshold": SUPPORT_THRESHOLD,
            "refute_threshold": REFUTE_THRESHOLD,
        }
        return self.result(outcome, metrics, notes, config)


if __name__ == "__main__":
    Exp27IterativeCompression().execute()
