"""
Experiment 2.4 — Chunk Size Sensitivity

Hypothesis: There exists an optimal chunk size for compression beyond which quality
degrades independent of compression ratio.

Setup:
  - Fixed bottleneck of 64-dim for all chunk sizes (output of compression)
  - Compress chunks of [4, 8, 16, 32, 64] tokens each into a single 64-dim vector
  - Token embedding dim = HIDDEN_DIM = 64
  - Input sizes: 4×64=256, 8×64=512, 16×64=1024, 32×64=2048, 64×64=4096
  - For each chunk size: train linear autoencoder, measure cosine similarity recall
  - Compare quality across chunk sizes to find peak
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
BOTTLENECK   = 64          # fixed bottleneck dim for all chunk sizes
CHUNK_SIZES  = [4, 8, 16, 32, 64]
BATCH_SIZE   = 32
TRAIN_STEPS  = 2000
EVAL_BATCHES = 200
LR           = 1e-3
DEVICE       = "cpu"
FLAT_THRESH  = 0.05        # quality range below this = "flat" result


# ── Model ─────────────────────────────────────────────────────────────────────

class ChunkAutoencoder(nn.Module):
    """Linear autoencoder: input_dim → BOTTLENECK → input_dim."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        mid = max(BOTTLENECK * 2, input_dim // 2, 128)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, mid),
            nn.ReLU(),
            nn.Linear(mid, BOTTLENECK),
        )
        self.decoder = nn.Sequential(
            nn.Linear(BOTTLENECK, mid),
            nn.ReLU(),
            nn.Linear(mid, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


# ── Data helpers ──────────────────────────────────────────────────────────────

def make_chunk_batch(
    batch_size: int, chunk_size: int, embed: nn.Embedding
) -> torch.Tensor:
    tokens = torch.randint(0, VOCAB_SIZE, (batch_size, chunk_size))
    with torch.no_grad():
        h = embed(tokens)                   # (B, chunk_size, HIDDEN_DIM)
    return h.view(batch_size, -1)           # (B, chunk_size * HIDDEN_DIM)


# ── Train + eval one chunk size ───────────────────────────────────────────────

def train_and_eval_chunk(
    chunk_size: int, embed: nn.Embedding
) -> dict[str, float]:
    input_dim = chunk_size * HIDDEN_DIM
    model     = ChunkAutoencoder(input_dim).to(DEVICE)
    opt       = Adam(model.parameters(), lr=LR)

    for _ in range(TRAIN_STEPS):
        x     = make_chunk_batch(BATCH_SIZE, chunk_size, embed)
        x_hat = model(x)
        loss  = F.mse_loss(x_hat, x)
        opt.zero_grad()
        loss.backward()
        opt.step()

    model.eval()
    sims: list[float] = []
    with torch.no_grad():
        for _ in range(EVAL_BATCHES):
            x     = make_chunk_batch(BATCH_SIZE, chunk_size, embed)
            x_hat = model(x)
            sims.append(F.cosine_similarity(x, x_hat, dim=-1).mean().item())

    mean_sim = sum(sims) / len(sims)
    compression_ratio = input_dim / BOTTLENECK
    print(f"  chunk_size={chunk_size:3d}  input_dim={input_dim:5d}  "
          f"ratio={compression_ratio:.1f}x  cosim={mean_sim:.4f}")
    return {"chunk_size": chunk_size, "input_dim": input_dim,
            "compression_ratio": compression_ratio, "cosim": mean_sim}


# ── Experiment ────────────────────────────────────────────────────────────────

class Exp24ChunkSizeSensitivity(Experiment):
    experiment_id = "exp_2_4"
    hypothesis = (
        "There exists an optimal chunk size for compression beyond which quality "
        "degrades independent of compression ratio."
    )

    def run(self) -> ExperimentResult:
        torch.manual_seed(42)
        embed = nn.Embedding(VOCAB_SIZE, HIDDEN_DIM).to(DEVICE)

        print(f"\nTraining autoencoders for chunk sizes {CHUNK_SIZES} "
              f"(bottleneck={BOTTLENECK}):")
        results = [train_and_eval_chunk(cs, embed) for cs in CHUNK_SIZES]

        sims        = [r["cosim"] for r in results]
        best_idx    = int(max(range(len(sims)), key=lambda i: sims[i]))
        best_cs     = CHUNK_SIZES[best_idx]
        quality_range = max(sims) - min(sims)

        # Check patterns
        is_mono_decrease = all(sims[i] >= sims[i + 1] for i in range(len(sims) - 1))
        is_flat          = quality_range < FLAT_THRESH
        is_peak_at_middle = (best_idx > 0) and (best_idx < len(CHUNK_SIZES) - 1)

        # Confirm "clear peak": adjacent values are both lower
        if is_peak_at_middle:
            left_lower  = sims[best_idx] > sims[best_idx - 1]
            right_lower = sims[best_idx] > sims[best_idx + 1]
            clear_peak  = left_lower and right_lower
        else:
            clear_peak = False

        if clear_peak:
            outcome = OUTCOME_SUPPORTED
        elif is_mono_decrease:
            outcome = OUTCOME_REFUTED
        elif is_flat:
            outcome = OUTCOME_INCONCLUSIVE
        else:
            outcome = OUTCOME_INCONCLUSIVE

        metrics = {
            "quality_at_4":    sims[0],
            "quality_at_8":    sims[1],
            "quality_at_16":   sims[2],
            "quality_at_32":   sims[3],
            "quality_at_64":   sims[4],
            "optimal_chunk_size":   best_cs,
            "quality_range":        quality_range,
            "is_monotone_decrease": is_mono_decrease,
            "is_flat":              is_flat,
            "clear_peak_detected":  clear_peak,
            "cosine_sim": {str(r["chunk_size"]): r["cosim"] for r in results},
        }
        notes = (
            f"Quality scores {[round(s, 3) for s in sims]} for chunk sizes {CHUNK_SIZES}. "
            f"Best quality at chunk_size={best_cs} (index {best_idx}). "
            f"Clear peak at middle: {clear_peak}. "
            f"Monotone decrease: {is_mono_decrease}. Flat: {is_flat}."
        )
        config = {
            "chunk_sizes": CHUNK_SIZES, "bottleneck": BOTTLENECK,
            "hidden_dim": HIDDEN_DIM, "train_steps": TRAIN_STEPS,
        }
        return self.result(outcome, metrics, notes, config)


if __name__ == "__main__":
    Exp24ChunkSizeSensitivity().execute()
