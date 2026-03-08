"""
Experiment 2.6 — Compression Generalization

Hypothesis: A compressor trained on domain A produces meaningfully worse retrieval
on domain B, indicating compression overfits to domain.

Setup:
  - Domain A: token IDs 0–63 (numeric domain)
  - Domain B: token IDs 64–127 (lexical domain)
  - FLAT_DIM=256, BOTTLENECK=32 (8x compression)
  - Three compressors:
      Compressor A  — trained only on domain A
      Compressor B  — trained only on domain B
      Compressor AB — trained on mixed data (both domains)
  - Evaluate every compressor on both domains independently
  - DEGRADATION_THRESHOLD = 0.10 cosine sim drop
  - SUPPORTED if comp_a_on_b < comp_a_on_a - threshold (cross-domain degradation)
  - REFUTED if cross-domain quality matches in-domain quality
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

VOCAB_SIZE           = 256
HIDDEN_DIM           = 64
SEQ_LEN              = 4             # tokens per sequence
FLAT_DIM             = SEQ_LEN * HIDDEN_DIM   # 256
BOTTLENECK           = 32            # 8x compression
DOMAIN_A_MIN         = 0
DOMAIN_A_MAX         = 64            # tokens 0–63
DOMAIN_B_MIN         = 64
DOMAIN_B_MAX         = 128           # tokens 64–127
BATCH_SIZE           = 32
TRAIN_STEPS          = 2000
EVAL_BATCHES         = 200
LR                   = 1e-3
DEVICE               = "cpu"
DEGRADATION_THRESHOLD = 0.10         # cosine sim drop to call it domain-sensitive


# ── Model ─────────────────────────────────────────────────────────────────────

class Compressor(nn.Module):
    """Two-layer autoencoder: FLAT_DIM → BOTTLENECK → FLAT_DIM."""

    def __init__(self) -> None:
        super().__init__()
        mid = max(BOTTLENECK * 2, 64)
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

def make_domain_batch(
    batch_size: int,
    embed: nn.Embedding,
    tok_min: int,
    tok_max: int,
) -> torch.Tensor:
    tokens = torch.randint(tok_min, tok_max, (batch_size, SEQ_LEN))
    with torch.no_grad():
        h = embed(tokens)               # (B, SEQ_LEN, HIDDEN_DIM)
    return h.view(batch_size, -1)       # (B, FLAT_DIM)


def make_mixed_batch(batch_size: int, embed: nn.Embedding) -> torch.Tensor:
    half   = batch_size // 2
    batch_a = make_domain_batch(half, embed, DOMAIN_A_MIN, DOMAIN_A_MAX)
    batch_b = make_domain_batch(batch_size - half, embed, DOMAIN_B_MIN, DOMAIN_B_MAX)
    return torch.cat([batch_a, batch_b], dim=0)


# ── Training ──────────────────────────────────────────────────────────────────

def train_on_domain(
    model: Compressor,
    embed: nn.Embedding,
    tok_min: int,
    tok_max: int,
    label: str,
) -> None:
    opt = Adam(model.parameters(), lr=LR)
    for step in range(TRAIN_STEPS):
        x = make_domain_batch(BATCH_SIZE, embed, tok_min, tok_max)
        _, x_hat = model(x)
        loss = F.mse_loss(x_hat, x)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (step + 1) % 500 == 0:
            print(f"    [{label}] step {step+1}/{TRAIN_STEPS}  loss={loss.item():.4f}")


def train_on_mixed(model: Compressor, embed: nn.Embedding) -> None:
    opt = Adam(model.parameters(), lr=LR)
    for step in range(TRAIN_STEPS):
        x = make_mixed_batch(BATCH_SIZE, embed)
        _, x_hat = model(x)
        loss = F.mse_loss(x_hat, x)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (step + 1) % 500 == 0:
            print(f"    [mixed] step {step+1}/{TRAIN_STEPS}  loss={loss.item():.4f}")


# ── Evaluation ────────────────────────────────────────────────────────────────

def eval_cosim(
    model: Compressor,
    embed: nn.Embedding,
    tok_min: int,
    tok_max: int,
) -> float:
    sims: list[float] = []
    with torch.no_grad():
        for _ in range(EVAL_BATCHES):
            x        = make_domain_batch(BATCH_SIZE, embed, tok_min, tok_max)
            _, x_hat = model(x)
            sims.append(F.cosine_similarity(x, x_hat, dim=-1).mean().item())
    return sum(sims) / len(sims)


# ── Experiment ────────────────────────────────────────────────────────────────

class Exp26CompressionGeneralization(Experiment):
    experiment_id = "exp_2_6"
    hypothesis = (
        "A compressor trained on domain A produces meaningfully worse retrieval on "
        "domain B, indicating compression overfits to domain."
    )

    def run(self) -> ExperimentResult:
        torch.manual_seed(42)
        embed = nn.Embedding(VOCAB_SIZE, HIDDEN_DIM).to(DEVICE)

        comp_a  = Compressor().to(DEVICE)
        comp_b  = Compressor().to(DEVICE)
        comp_ab = Compressor().to(DEVICE)

        print("\nTraining Compressor A (domain A only)...")
        train_on_domain(comp_a, embed, DOMAIN_A_MIN, DOMAIN_A_MAX, "comp_a")

        print("Training Compressor B (domain B only)...")
        train_on_domain(comp_b, embed, DOMAIN_B_MIN, DOMAIN_B_MAX, "comp_b")

        print("Training Compressor AB (mixed)...")
        train_on_mixed(comp_ab, embed)

        print("\nEvaluating all compressors on both domains...")
        for m in [comp_a, comp_b, comp_ab]:
            m.eval()

        comp_a_on_a  = eval_cosim(comp_a,  embed, DOMAIN_A_MIN, DOMAIN_A_MAX)
        comp_a_on_b  = eval_cosim(comp_a,  embed, DOMAIN_B_MIN, DOMAIN_B_MAX)
        comp_b_on_b  = eval_cosim(comp_b,  embed, DOMAIN_B_MIN, DOMAIN_B_MAX)
        comp_b_on_a  = eval_cosim(comp_b,  embed, DOMAIN_A_MIN, DOMAIN_A_MAX)
        comp_ab_on_a = eval_cosim(comp_ab, embed, DOMAIN_A_MIN, DOMAIN_A_MAX)
        comp_ab_on_b = eval_cosim(comp_ab, embed, DOMAIN_B_MIN, DOMAIN_B_MAX)

        print(f"  comp_a:  on_a={comp_a_on_a:.4f}  on_b={comp_a_on_b:.4f}  "
              f"drop={comp_a_on_a - comp_a_on_b:.4f}")
        print(f"  comp_b:  on_b={comp_b_on_b:.4f}  on_a={comp_b_on_a:.4f}  "
              f"drop={comp_b_on_b - comp_b_on_a:.4f}")
        print(f"  comp_ab: on_a={comp_ab_on_a:.4f}  on_b={comp_ab_on_b:.4f}")

        drop_a = comp_a_on_a - comp_a_on_b     # how much comp_a degrades on domain B
        drop_b = comp_b_on_b - comp_b_on_a     # how much comp_b degrades on domain A

        if drop_a > DEGRADATION_THRESHOLD:
            outcome = OUTCOME_SUPPORTED
        elif drop_a < 0 and drop_b < 0:
            # comp_a is BETTER on domain B — no overfitting detected
            outcome = OUTCOME_REFUTED
        else:
            outcome = OUTCOME_INCONCLUSIVE

        metrics = {
            "comp_a_on_a":   comp_a_on_a,
            "comp_a_on_b":   comp_a_on_b,
            "comp_b_on_b":   comp_b_on_b,
            "comp_b_on_a":   comp_b_on_a,
            "comp_ab_on_a":  comp_ab_on_a,
            "comp_ab_on_b":  comp_ab_on_b,
            "drop_a_cross_domain": drop_a,
            "drop_b_cross_domain": drop_b,
        }
        notes = (
            f"Comp_A: in-domain={comp_a_on_a:.3f}, cross-domain={comp_a_on_b:.3f}, "
            f"drop={drop_a:.3f} (threshold={DEGRADATION_THRESHOLD}). "
            f"Comp_B: in-domain={comp_b_on_b:.3f}, cross-domain={comp_b_on_a:.3f}, "
            f"drop={drop_b:.3f}. "
            f"Mixed compressor: A={comp_ab_on_a:.3f}, B={comp_ab_on_b:.3f}."
        )
        config = {
            "flat_dim": FLAT_DIM, "bottleneck": BOTTLENECK,
            "domain_a": f"{DOMAIN_A_MIN}-{DOMAIN_A_MAX}",
            "domain_b": f"{DOMAIN_B_MIN}-{DOMAIN_B_MAX}",
            "degradation_threshold": DEGRADATION_THRESHOLD,
            "train_steps": TRAIN_STEPS,
        }
        return self.result(outcome, metrics, notes, config)


if __name__ == "__main__":
    Exp26CompressionGeneralization().execute()
