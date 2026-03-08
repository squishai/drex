"""
Experiment 2.8 — Compression Under Distribution Shift

Hypothesis: A compressor trained on a fixed distribution degrades gracefully
(not catastrophically) under mid-context distribution shift.

Setup:
  - Train autoencoder on domain A (token IDs 0–63), FLAT_DIM=512
  - At eval: feed sequences where first HALF_SEQ tokens are domain A,
    last HALF_SEQ tokens are domain B (IDs 64–127)
  - Measure cosine similarity for the domain A portion vs the domain B portion
    of the same decoded output
  - quality_domain_a:        cosim on the first-half embedding subvector
  - quality_domain_b_shifted: cosim on the second-half embedding subvector
  - quality_drop = quality_domain_a - quality_domain_b_shifted
  - CATASTROPHIC_THRESHOLD = 0.20
  - SUPPORTED  if quality_drop < CATASTROPHIC_THRESHOLD
  - REFUTED    if quality_drop > CATASTROPHIC_THRESHOLD AND quality_domain_b < 0.1
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

VOCAB_SIZE            = 256
HIDDEN_DIM            = 64
SEQ_LEN               = 8
HALF_SEQ              = SEQ_LEN // 2          # 4 tokens each domain in mixed seq
FLAT_DIM              = SEQ_LEN * HIDDEN_DIM  # 512
BOTTLENECK            = 64                    # 8x compression
DOMAIN_A_MIN          = 0
DOMAIN_A_MAX          = 64    # tokens 0–63
DOMAIN_B_MIN          = 64
DOMAIN_B_MAX          = 128   # tokens 64–127
BATCH_SIZE            = 32
TRAIN_STEPS           = 2000
EVAL_BATCHES          = 200
LR                    = 1e-3
DEVICE                = "cpu"
CATASTROPHIC_THRESHOLD = 0.20


# ── Model ─────────────────────────────────────────────────────────────────────

class Compressor(nn.Module):
    """Autoencoder: FLAT_DIM → BOTTLENECK → FLAT_DIM."""

    def __init__(self) -> None:
        super().__init__()
        mid = BOTTLENECK * 2
        self.encoder = nn.Sequential(
            nn.Linear(FLAT_DIM, mid),
            nn.ReLU(),
            nn.Linear(mid, BOTTLENECK),
        )
        self.decoder = nn.Sequential(
            nn.Linear(BOTTLENECK, mid),
            nn.ReLU(),
            nn.Linear(mid, FLAT_DIM),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z     = self.encoder(x)
        x_hat = self.decoder(z)
        return z, x_hat


# ── Data helpers ──────────────────────────────────────────────────────────────

def make_domain_a_batch(batch_size: int, embed: nn.Embedding) -> torch.Tensor:
    """Full sequences with domain A tokens (0–63)."""
    tokens = torch.randint(DOMAIN_A_MIN, DOMAIN_A_MAX, (batch_size, SEQ_LEN))
    with torch.no_grad():
        h = embed(tokens)
    return h.view(batch_size, -1)   # (B, FLAT_DIM)


def make_mixed_batch(
    batch_size: int, embed: nn.Embedding
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    First HALF_SEQ tokens from domain A, last HALF_SEQ from domain B.
    Returns:
        x_flat   (B, FLAT_DIM)   — full flattened sequence
        x_a_half (B, HALF_DIM)   — domain A sub-vector (first half)
        x_b_half (B, HALF_DIM)   — domain B sub-vector (second half)
    where HALF_DIM = HALF_SEQ * HIDDEN_DIM.
    """
    toks_a = torch.randint(DOMAIN_A_MIN, DOMAIN_A_MAX, (batch_size, HALF_SEQ))
    toks_b = torch.randint(DOMAIN_B_MIN, DOMAIN_B_MAX, (batch_size, HALF_SEQ))
    tokens = torch.cat([toks_a, toks_b], dim=1)    # (B, SEQ_LEN)

    with torch.no_grad():
        h = embed(tokens)                           # (B, SEQ_LEN, HIDDEN_DIM)

    half_dim  = HALF_SEQ * HIDDEN_DIM
    x_flat    = h.view(batch_size, -1)              # (B, FLAT_DIM)
    x_a_half  = h[:, :HALF_SEQ, :].reshape(batch_size, half_dim)
    x_b_half  = h[:, HALF_SEQ:, :].reshape(batch_size, half_dim)
    return x_flat, x_a_half, x_b_half


# ── Training ──────────────────────────────────────────────────────────────────

def train(model: Compressor, embed: nn.Embedding) -> None:
    opt = Adam(model.parameters(), lr=LR)
    for step in range(TRAIN_STEPS):
        x        = make_domain_a_batch(BATCH_SIZE, embed)
        _, x_hat = model(x)
        loss     = F.mse_loss(x_hat, x)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (step + 1) % 500 == 0:
            print(f"    step {step+1}/{TRAIN_STEPS}  loss={loss.item():.4f}")


# ── Evaluation ────────────────────────────────────────────────────────────────

def eval_quality(model: Compressor, embed: nn.Embedding) -> dict[str, float]:
    half_dim = HALF_SEQ * HIDDEN_DIM

    # --- in-distribution: full domain A sequences ---
    sims_a: list[float] = []
    with torch.no_grad():
        for _ in range(EVAL_BATCHES):
            x        = make_domain_a_batch(BATCH_SIZE, embed)
            _, x_hat = model(x)
            sims_a.append(F.cosine_similarity(x, x_hat, dim=-1).mean().item())
    quality_domain_a = sum(sims_a) / len(sims_a)

    # --- distribution-shifted: mixed sequences, measure quality per half ---
    sims_a_mixed: list[float] = []
    sims_b_mixed: list[float] = []
    with torch.no_grad():
        for _ in range(EVAL_BATCHES):
            x_flat, x_a_half, x_b_half = make_mixed_batch(BATCH_SIZE, embed)
            _, x_hat_flat = model(x_flat)

            # split reconstruction into domain-A and domain-B halves
            x_hat_a = x_hat_flat[:, :half_dim]
            x_hat_b = x_hat_flat[:, half_dim:]

            sims_a_mixed.append(
                F.cosine_similarity(x_hat_a, x_a_half, dim=-1).mean().item()
            )
            sims_b_mixed.append(
                F.cosine_similarity(x_hat_b, x_b_half, dim=-1).mean().item()
            )

    quality_a_in_mixed = sum(sims_a_mixed) / len(sims_a_mixed)
    quality_b_shifted  = sum(sims_b_mixed) / len(sims_b_mixed)

    return {
        "quality_domain_a":       quality_domain_a,
        "quality_a_in_mixed_seq": quality_a_in_mixed,
        "quality_domain_b_shifted": quality_b_shifted,
        "quality_drop":           quality_domain_a - quality_b_shifted,
    }


# ── Experiment ────────────────────────────────────────────────────────────────

class Exp28CompressionUnderDistributionShift(Experiment):
    experiment_id = "exp_2_8"
    hypothesis = (
        "A compressor trained on a fixed distribution degrades gracefully "
        "(not catastrophically) under mid-context distribution shift."
    )

    def run(self) -> ExperimentResult:
        torch.manual_seed(42)
        embed = nn.Embedding(VOCAB_SIZE, HIDDEN_DIM).to(DEVICE)
        model = Compressor().to(DEVICE)

        print(f"\nTraining autoencoder on domain A (tokens {DOMAIN_A_MIN}–"
              f"{DOMAIN_A_MAX - 1}) for {TRAIN_STEPS} steps...")
        train(model, embed)

        print("\nEvaluating under distribution shift...")
        model.eval()
        m = eval_quality(model, embed)

        quality_domain_a     = m["quality_domain_a"]
        quality_b_shifted    = m["quality_domain_b_shifted"]
        quality_drop         = m["quality_drop"]
        catastrophic         = (
            quality_drop > CATASTROPHIC_THRESHOLD and quality_b_shifted < 0.1
        )

        print(f"  quality_domain_a      = {quality_domain_a:.4f}")
        print(f"  quality_a_in_mixed    = {m['quality_a_in_mixed_seq']:.4f}")
        print(f"  quality_domain_b_shifted = {quality_b_shifted:.4f}")
        print(f"  quality_drop          = {quality_drop:.4f}  "
              f"(threshold={CATASTROPHIC_THRESHOLD})")
        print(f"  catastrophic          = {catastrophic}")

        if quality_drop < CATASTROPHIC_THRESHOLD:
            outcome = OUTCOME_SUPPORTED
        elif catastrophic:
            outcome = OUTCOME_REFUTED
        else:
            outcome = OUTCOME_INCONCLUSIVE

        metrics = {
            "quality_domain_a":         quality_domain_a,
            "quality_a_in_mixed_seq":   m["quality_a_in_mixed_seq"],
            "quality_domain_b_shifted": quality_b_shifted,
            "quality_drop":             quality_drop,
            "catastrophic":             catastrophic,
        }
        notes = (
            f"Domain A (in-distribution): {quality_domain_a:.3f}. "
            f"Domain B (shifted): {quality_b_shifted:.3f}. "
            f"Drop: {quality_drop:.3f} vs catastrophic threshold {CATASTROPHIC_THRESHOLD}. "
            f"Graceful degradation (drop < threshold): "
            f"{quality_drop < CATASTROPHIC_THRESHOLD}. "
            f"Catastrophic failure: {catastrophic}."
        )
        config = {
            "flat_dim": FLAT_DIM, "bottleneck": BOTTLENECK,
            "seq_len": SEQ_LEN, "half_seq": HALF_SEQ,
            "domain_a": f"{DOMAIN_A_MIN}-{DOMAIN_A_MAX}",
            "domain_b": f"{DOMAIN_B_MIN}-{DOMAIN_B_MAX}",
            "catastrophic_threshold": CATASTROPHIC_THRESHOLD,
            "train_steps": TRAIN_STEPS,
        }
        return self.result(outcome, metrics, notes, config)


if __name__ == "__main__":
    Exp28CompressionUnderDistributionShift().execute()
