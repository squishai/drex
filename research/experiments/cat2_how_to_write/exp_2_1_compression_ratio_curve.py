"""
Experiment 2.1 — Compression Ratio vs Recall Fidelity Curve

Hypothesis: There exists a compression ratio threshold beyond which recall
fidelity degrades catastrophically rather than gracefully.

Setup:
  - Fixed corpus of text sequences (represented as token embeddings)
  - Autoencoder compresses sequences into bottleneck vectors at 2x to 100x ratios
  - Each compression is trained to minimize reconstruction loss
  - Recall fidelity = cosine similarity between original and reconstructed
  - Key question: where does the curve break?
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

VOCAB_SIZE     = 512
SEQ_LEN        = 64           # tokens per sequence
EMBED_DIM      = 64           # token embedding dimension
FLAT_DIM       = SEQ_LEN * EMBED_DIM   # 4096 — the "uncompressed" size
RATIOS         = [2, 4, 8, 16, 32, 64, 100]
TRAIN_STEPS    = 3000
EVAL_BATCHES   = 200
BATCH_SIZE     = 32
LR             = 1e-3
DEVICE         = "cpu"
CATASTROPHE_THRESHOLD = 0.15  # drop in cosine sim that counts as catastrophic


# ── Autoencoder ───────────────────────────────────────────────────────────────

class Compressor(nn.Module):
    def __init__(self, bottleneck_dim: int):
        super().__init__()
        mid = max(bottleneck_dim * 2, 128)
        self.encoder = nn.Sequential(
            nn.Linear(FLAT_DIM, mid),
            nn.ReLU(),
            nn.Linear(mid, bottleneck_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, mid),
            nn.ReLU(),
            nn.Linear(mid, FLAT_DIM),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (B, FLAT_DIM) → compressed: (B, K) → reconstructed: (B, FLAT_DIM)"""
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return z, x_hat

    def bottleneck_dim(self) -> int:
        return self.encoder[-1].out_features


def make_sequences(batch_size: int, embed: nn.Embedding) -> torch.Tensor:
    tokens = torch.randint(0, VOCAB_SIZE, (batch_size, SEQ_LEN))
    with torch.no_grad():
        h = embed(tokens)   # (B, L, E)
    return h.view(batch_size, -1)   # (B, FLAT_DIM)


def train_compressor(ratio: int, embed: nn.Embedding) -> dict:
    bottleneck = max(1, FLAT_DIM // ratio)
    model = Compressor(bottleneck).to(DEVICE)
    opt = Adam(model.parameters(), lr=LR)

    for step in range(TRAIN_STEPS):
        x = make_sequences(BATCH_SIZE, embed)
        _, x_hat = model(x)
        loss = F.mse_loss(x_hat, x)
        opt.zero_grad()
        loss.backward()
        opt.step()

    # evaluate
    model.eval()
    cos_sims = []
    recon_losses = []
    with torch.no_grad():
        for _ in range(EVAL_BATCHES):
            x = make_sequences(BATCH_SIZE, embed)
            _, x_hat = model(x)
            # cosine similarity per sample
            sim = F.cosine_similarity(x, x_hat, dim=-1)
            cos_sims.append(sim.mean().item())
            recon_losses.append(F.mse_loss(x_hat, x).item())

    mean_cos = sum(cos_sims) / len(cos_sims)
    mean_mse = sum(recon_losses) / len(recon_losses)
    print(f"  ratio={ratio:4d}x  bottleneck={bottleneck:6d}  "
          f"cosine_sim={mean_cos:.4f}  mse={mean_mse:.4f}")
    return {"ratio": ratio, "bottleneck_dim": bottleneck,
            "cosine_sim": mean_cos, "mse": mean_mse}


# ── Experiment ────────────────────────────────────────────────────────────────

class Exp21CompressionRatioCurve(Experiment):
    experiment_id = "exp_2_1"
    hypothesis = (
        "There exists a compression ratio threshold beyond which recall fidelity "
        "degrades catastrophically rather than gracefully."
    )

    def run(self) -> ExperimentResult:
        torch.manual_seed(42)
        embed = nn.Embedding(VOCAB_SIZE, EMBED_DIM).to(DEVICE)

        print(f"\nCompressing {FLAT_DIM}-dim vectors at {RATIOS} ratios:")
        results = []
        for ratio in RATIOS:
            res = train_compressor(ratio, embed)
            results.append(res)

        # detect catastrophic cliff: largest single-step drop in cosine sim
        sims = [r["cosine_sim"] for r in results]
        drops = [sims[i] - sims[i+1] for i in range(len(sims)-1)]
        max_drop = max(drops)
        cliff_ratio_idx = drops.index(max_drop)
        cliff_ratio = RATIOS[cliff_ratio_idx]
        cliff_ratio_after = RATIOS[cliff_ratio_idx + 1]

        catastrophic = max_drop > CATASTROPHE_THRESHOLD

        if catastrophic:
            outcome = OUTCOME_SUPPORTED
        elif max_drop > 0.05:
            outcome = OUTCOME_INCONCLUSIVE
        else:
            outcome = OUTCOME_REFUTED

        metrics = {
            "cosine_sims": {str(r["ratio"]): r["cosine_sim"] for r in results},
            "mse_values":  {str(r["ratio"]): r["mse"] for r in results},
            "max_single_step_drop": max_drop,
            "cliff_at_ratio": cliff_ratio,
            "cliff_after_ratio": cliff_ratio_after,
            "catastrophic_cliff_detected": catastrophic,
        }
        notes = (
            f"Largest quality drop: {max_drop:.3f} between {cliff_ratio}x and "
            f"{cliff_ratio_after}x compression. "
            f"Cosine sim at 2x={sims[0]:.3f}, at 100x={sims[-1]:.3f}."
        )

        return self.result(outcome, metrics, notes, config={
            "flat_dim": FLAT_DIM, "ratios": RATIOS, "train_steps": TRAIN_STEPS,
        })


if __name__ == "__main__":
    Exp21CompressionRatioCurve().execute()
