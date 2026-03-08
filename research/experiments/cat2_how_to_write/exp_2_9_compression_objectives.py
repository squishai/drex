"""
Experiment 2.9 — Retrieval-Oriented vs Storage-Oriented Compression

Hypothesis: Minimizing reconstruction loss and maximizing downstream retrieval
accuracy are fundamentally different objectives and produce measurably different
representations.

Setup:
  - Two compressors with identical architectures and training data
  - Compressor A trained on reconstruction loss (MSE)
  - Compressor B trained on retrieval accuracy (contrastive / ranking loss)
  - Both tested on: (1) reconstruction quality, (2) retrieval accuracy
  - If they diverge significantly → hypothesis SUPPORTED
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

VOCAB_SIZE      = 256
SEQ_LEN         = 32
EMBED_DIM       = 64
FLAT_DIM        = SEQ_LEN * EMBED_DIM
BOTTLENECK_DIM  = FLAT_DIM // 8    # 8x compression
TRAIN_STEPS     = 3000
EVAL_BATCHES    = 300
BATCH_SIZE      = 32
LR              = 1e-3
DIVERGENCE_THRESHOLD = 0.10   # absolute difference in retrieval Acc@1
DEVICE          = "cpu"


# ── Shared Architecture ───────────────────────────────────────────────────────

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        mid = BOTTLENECK_DIM * 4
        self.net = nn.Sequential(
            nn.Linear(FLAT_DIM, mid), nn.ReLU(),
            nn.Linear(mid, BOTTLENECK_DIM),
        )

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        mid = BOTTLENECK_DIM * 4
        self.net = nn.Sequential(
            nn.Linear(BOTTLENECK_DIM, mid), nn.ReLU(),
            nn.Linear(mid, FLAT_DIM),
        )

    def forward(self, z):
        return self.net(z)


# ── Data ──────────────────────────────────────────────────────────────────────

def make_batch(batch_size: int, embed: nn.Embedding) -> torch.Tensor:
    tokens = torch.randint(0, VOCAB_SIZE, (batch_size, SEQ_LEN))
    with torch.no_grad():
        h = embed(tokens)
    return h.view(batch_size, -1).detach()


# ── Loss Functions ────────────────────────────────────────────────────────────

def reconstruction_loss(enc: Encoder, dec: Decoder, x: torch.Tensor) -> torch.Tensor:
    z = enc(x)
    x_hat = dec(z)
    return F.mse_loss(x_hat, x)


def retrieval_loss(enc: Encoder, x: torch.Tensor) -> torch.Tensor:
    """
    InfoNCE contrastive loss: each sequence should be closest to itself.
    Positive pair: (x, x) — same sequence.
    Negative pairs: all other sequences in batch.
    """
    z = enc(x)     # (B, K)
    z_norm = F.normalize(z, dim=-1)
    logits = z_norm @ z_norm.T   # (B, B)
    # scale temperature
    logits = logits / 0.07
    labels = torch.arange(z.shape[0], device=z.device)
    # mask diagonal as positive
    return F.cross_entropy(logits, labels)


# ── Evaluation ────────────────────────────────────────────────────────────────

def eval_reconstruction(enc: Encoder, dec: Decoder, embed: nn.Embedding) -> float:
    enc.eval(); dec.eval()
    sims = []
    with torch.no_grad():
        for _ in range(EVAL_BATCHES):
            x = make_batch(BATCH_SIZE, embed)
            z = enc(x)
            x_hat = dec(z)
            sims.append(F.cosine_similarity(x, x_hat, dim=-1).mean().item())
    enc.train(); dec.train()
    return sum(sims) / len(sims)


def eval_retrieval_acc1(enc: Encoder, embed: nn.Embedding) -> float:
    """Acc@1: what fraction of queries retrieve the correct item from a gallery."""
    enc.eval()
    hits = 0
    total = 0
    with torch.no_grad():
        for _ in range(EVAL_BATCHES):
            # gallery = batch of sequences; query = same sequences
            x = make_batch(BATCH_SIZE, embed)
            z = enc(x)
            z_norm = F.normalize(z, dim=-1)
            sims = z_norm @ z_norm.T   # (B, B)
            # exclude self by zeroing diagonal would be trivial — instead,
            # add small noise to query to simulate approximate lookup
            noise = torch.randn_like(x) * 0.05
            z_q = enc(x + noise)
            z_q_norm = F.normalize(z_q, dim=-1)
            sim_q = z_q_norm @ z_norm.T  # (B, B)
            preds = sim_q.argmax(dim=-1)
            labels = torch.arange(BATCH_SIZE)
            hits  += (preds == labels).sum().item()
            total += BATCH_SIZE
    enc.train()
    return hits / total


# ── Experiment ────────────────────────────────────────────────────────────────

class Exp29CompressionObjectives(Experiment):
    experiment_id = "exp_2_9"
    hypothesis = (
        "Minimizing reconstruction loss and maximizing downstream retrieval accuracy "
        "are fundamentally different objectives and produce measurably different "
        "compressed representations."
    )

    def run(self) -> ExperimentResult:
        torch.manual_seed(42)
        embed = nn.Embedding(VOCAB_SIZE, EMBED_DIM)

        # ── Compressor A: reconstruction objective ──
        enc_a, dec_a = Encoder(), Decoder()
        opt_a = Adam(list(enc_a.parameters()) + list(dec_a.parameters()), lr=LR)
        for _ in range(TRAIN_STEPS):
            x = make_batch(BATCH_SIZE, embed)
            loss = reconstruction_loss(enc_a, dec_a, x)
            opt_a.zero_grad(); loss.backward(); opt_a.step()

        # ── Compressor B: retrieval objective ──
        enc_b, dec_b = Encoder(), Decoder()
        opt_b = Adam(list(enc_b.parameters()) + list(dec_b.parameters()), lr=LR)
        for _ in range(TRAIN_STEPS):
            x = make_batch(BATCH_SIZE, embed)
            # train primarily on retrieval; decoder gets reconstruction gradient
            loss_r = retrieval_loss(enc_b, x)
            loss_rec = reconstruction_loss(enc_b, dec_b, x)
            loss = loss_r + 0.1 * loss_rec   # retrieval-dominant
            opt_b.zero_grad(); loss.backward(); opt_b.step()

        # ── Evaluate both on both metrics ──
        recon_a = eval_reconstruction(enc_a, dec_a, embed)
        recon_b = eval_reconstruction(enc_b, dec_b, embed)
        ret_a   = eval_retrieval_acc1(enc_a, embed)
        ret_b   = eval_retrieval_acc1(enc_b, embed)

        print(f"\n  Compressor A (reconstruction): recon_sim={recon_a:.4f}  acc@1={ret_a:.4f}")
        print(f"  Compressor B (retrieval):      recon_sim={recon_b:.4f}  acc@1={ret_b:.4f}")

        recon_gap   = abs(recon_a - recon_b)
        ret_gap     = abs(ret_a - ret_b)
        objectives_diverge = recon_gap > DIVERGENCE_THRESHOLD or ret_gap > DIVERGENCE_THRESHOLD

        if objectives_diverge and ret_b > ret_a and recon_a > recon_b:
            outcome = OUTCOME_SUPPORTED    # clean split as predicted
        elif objectives_diverge:
            outcome = OUTCOME_INCONCLUSIVE  # diverge but not in the expected direction
        else:
            outcome = OUTCOME_REFUTED

        metrics = {
            "recon_a_cosine": recon_a, "recon_b_cosine": recon_b,
            "retrieval_acc1_a": ret_a,  "retrieval_acc1_b": ret_b,
            "recon_gap": recon_gap,     "retrieval_gap": ret_gap,
            "objectives_diverge": objectives_diverge,
        }
        notes = (
            f"Reconstruction A={recon_a:.3f} vs B={recon_b:.3f} (gap {recon_gap:.3f}). "
            f"Retrieval A={ret_a:.3f} vs B={ret_b:.3f} (gap {ret_gap:.3f}). "
            f"Diverge={objectives_diverge}."
        )

        return self.result(outcome, metrics, notes, config={
            "flat_dim": FLAT_DIM, "bottleneck_dim": BOTTLENECK_DIM,
            "train_steps": TRAIN_STEPS,
        })


if __name__ == "__main__":
    Exp29CompressionObjectives().execute()
