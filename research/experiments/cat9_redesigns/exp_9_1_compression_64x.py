"""
Experiment 9.1 — 64x Compression: Retrieval vs Reconstruction Objective

Hypothesis: At 64x compression with 100-way gallery discrimination, retrieval-objective
compressor achieves at least 15% higher Acc@1 than reconstruction-objective compressor,
because 64x bottleneck forces genuine tradeoffs between fidelity and discriminability.
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

VOCAB_SIZE    = 256
EMBED_DIM     = 64
SEQ_LEN       = 32
FLAT_DIM      = SEQ_LEN * EMBED_DIM   # 2048
BOTTLENECK    = 32                     # 64x compression
GALLERY_SIZE  = 100
BATCH_SIZE    = 64
TRAIN_STEPS   = 3000
LR            = 3e-4
TEMPERATURE   = 0.07
NOISE_STD     = 0.1
DEVICE        = "cpu"

# ── Data helpers ──────────────────────────────────────────────────────────────

def make_seq_embedding(seq_tokens: torch.Tensor, embed: nn.Embedding) -> torch.Tensor:
    """Convert token sequence to flat embedding. Returns (B, FLAT_DIM)."""
    h = embed(seq_tokens)           # (B, L, E)
    return h.view(h.size(0), -1)    # (B, L*E)


def make_random_seq(batch_size: int) -> torch.Tensor:
    return torch.randint(0, VOCAB_SIZE, (batch_size, SEQ_LEN))


# ── Model components ──────────────────────────────────────────────────────────

class Compressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(FLAT_DIM, 512), nn.ReLU(),
            nn.Linear(512, BOTTLENECK),
        )

    def forward(self, x):
        return self.enc(x)   # (B, BOTTLENECK)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.dec = nn.Sequential(
            nn.Linear(BOTTLENECK, 512), nn.ReLU(),
            nn.Linear(512, FLAT_DIM),
        )

    def forward(self, z):
        return self.dec(z)   # (B, FLAT_DIM)


# ── InfoNCE loss ──────────────────────────────────────────────────────────────

def info_nce_loss(z_query: torch.Tensor, z_pos: torch.Tensor, temperature: float = TEMPERATURE) -> torch.Tensor:
    """Symmetric InfoNCE. z_query and z_pos are (B, D)."""
    B = z_query.size(0)
    z_q = F.normalize(z_query, dim=-1)
    z_p = F.normalize(z_pos, dim=-1)
    logits = torch.mm(z_q, z_p.t()) / temperature  # (B, B)
    labels = torch.arange(B, device=z_query.device)
    loss_q = F.cross_entropy(logits, labels)
    loss_p = F.cross_entropy(logits.t(), labels)
    return (loss_q + loss_p) / 2.0


# ── Training ──────────────────────────────────────────────────────────────────

def train_autoencoder(embed: nn.Embedding) -> tuple[Compressor, Decoder]:
    comp = Compressor().to(DEVICE)
    dec  = Decoder().to(DEVICE)
    opt  = Adam(list(comp.parameters()) + list(dec.parameters()), lr=LR)

    comp.train(); dec.train()
    for step in range(TRAIN_STEPS):
        seqs = make_random_seq(BATCH_SIZE)
        with torch.no_grad():
            x = make_seq_embedding(seqs, embed)
        z = comp(x)
        x_hat = dec(z)
        loss = F.mse_loss(x_hat, x)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (step + 1) % 500 == 0:
            print(f"    [AE] step {step+1}/{TRAIN_STEPS}  loss={loss.item():.6f}")

    return comp, dec


def train_contrastive(embed: nn.Embedding) -> Compressor:
    comp = Compressor().to(DEVICE)
    opt  = Adam(comp.parameters(), lr=LR)

    comp.train()
    for step in range(TRAIN_STEPS):
        # Query = anchor sequences, positive = same sequences + small noise in embedding space
        seqs = make_random_seq(BATCH_SIZE)
        with torch.no_grad():
            x = make_seq_embedding(seqs, embed)
        noise = torch.randn_like(x) * NOISE_STD
        x_pos = x + noise

        z_q = comp(x)
        z_p = comp(x_pos)
        loss = info_nce_loss(z_q, z_p)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (step + 1) % 500 == 0:
            print(f"    [CL] step {step+1}/{TRAIN_STEPS}  loss={loss.item():.4f}")

    return comp


# ── Evaluation ────────────────────────────────────────────────────────────────

def eval_retrieval(comp: Compressor, embed: nn.Embedding, n_trials: int = 200) -> float:
    """Compute Acc@1 against a gallery of GALLERY_SIZE items."""
    comp.eval()
    correct = 0

    with torch.no_grad():
        gallery_seqs = make_random_seq(GALLERY_SIZE)
        gallery_x    = make_seq_embedding(gallery_seqs, embed)  # (G, FLAT_DIM)
        gallery_z    = comp(gallery_x)                           # (G, BOTTLENECK)
        gallery_z_n  = F.normalize(gallery_z, dim=-1)

        for _ in range(n_trials):
            # Pick random gallery item as the "true" item
            idx = torch.randint(0, GALLERY_SIZE, (1,)).item()
            query_x = gallery_x[idx:idx+1] + torch.randn(1, FLAT_DIM) * NOISE_STD
            query_z = comp(query_x)
            query_z_n = F.normalize(query_z, dim=-1)

            sims = torch.mm(query_z_n, gallery_z_n.t()).squeeze(0)  # (G,)
            pred_idx = sims.argmax().item()
            if pred_idx == idx:
                correct += 1

    return correct / n_trials


def eval_recon_cosim(comp: Compressor, dec: Decoder, embed: nn.Embedding, n_batches: int = 20) -> float:
    """Compute mean cosine similarity between original and reconstructed embeddings."""
    comp.eval(); dec.eval()
    total = 0.0
    with torch.no_grad():
        for _ in range(n_batches):
            seqs = make_random_seq(BATCH_SIZE)
            x = make_seq_embedding(seqs, embed)
            z = comp(x)
            x_hat = dec(z)
            cos = F.cosine_similarity(x, x_hat, dim=-1).mean().item()
            total += cos
    return total / n_batches


# ── Experiment class ──────────────────────────────────────────────────────────

class Exp91Compression64x(Experiment):
    experiment_id = "exp_9_1"
    hypothesis = (
        "At 64x compression with 100-way gallery discrimination, retrieval-objective "
        "compressor achieves at least 15% higher Acc@1 than reconstruction-objective "
        "compressor, because 64x bottleneck forces genuine tradeoffs between fidelity "
        "and discriminability."
    )

    def run(self) -> ExperimentResult:
        # Shared frozen embedding (not trained — represents fixed sequence representation)
        embed = nn.Embedding(VOCAB_SIZE, EMBED_DIM).to(DEVICE)
        embed.requires_grad_(False)

        # ── Condition A: Autoencoder ──────────────────────────────────────────
        print("  Training autoencoder (Condition A) ...")
        comp_ae, dec_ae = train_autoencoder(embed)
        acc_ae = eval_retrieval(comp_ae, embed)
        recon_cosim_ae = eval_recon_cosim(comp_ae, dec_ae, embed)
        print(f"    AE Acc@1={acc_ae:.4f}, recon_cosim={recon_cosim_ae:.4f}")

        # ── Condition B: Contrastive ──────────────────────────────────────────
        print("  Training contrastive compressor (Condition B) ...")
        comp_cl = train_contrastive(embed)
        # Create dummy decoder for cosim evaluation
        dec_dummy = Decoder().to(DEVICE)
        acc_cl = eval_retrieval(comp_cl, embed)
        recon_cosim_cl = eval_recon_cosim(comp_cl, dec_dummy, embed)
        print(f"    CL Acc@1={acc_cl:.4f}, recon_cosim={recon_cosim_cl:.4f}")

        retrieval_gap = acc_cl - acc_ae
        print(f"  Retrieval gap (CL - AE) = {retrieval_gap:.4f}")

        # ── Outcome ───────────────────────────────────────────────────────────
        if retrieval_gap > 0.15 and recon_cosim_ae > recon_cosim_cl + 0.05:
            outcome = OUTCOME_SUPPORTED
        elif (acc_ae > 0.80 and acc_cl > 0.80) or retrieval_gap < 0.05:
            outcome = OUTCOME_REFUTED
        else:
            outcome = OUTCOME_INCONCLUSIVE

        metrics = {
            "acc_at1_autoencoder":   round(acc_ae, 4),
            "acc_at1_contrastive":   round(acc_cl, 4),
            "retrieval_gap":         round(retrieval_gap, 4),
            "recon_cosim_ae":        round(recon_cosim_ae, 4),
            "recon_cosim_cl":        round(recon_cosim_cl, 4),
        }
        notes = (
            f"AE Acc@1={acc_ae:.4f}, CL Acc@1={acc_cl:.4f}, gap={retrieval_gap:.4f}. "
            f"Recon cosim: AE={recon_cosim_ae:.4f}, CL(dummy dec)={recon_cosim_cl:.4f}."
        )
        config = {
            "vocab_size": VOCAB_SIZE, "embed_dim": EMBED_DIM,
            "seq_len": SEQ_LEN, "flat_dim": FLAT_DIM,
            "bottleneck": BOTTLENECK, "gallery_size": GALLERY_SIZE,
            "train_steps": TRAIN_STEPS, "temperature": TEMPERATURE,
            "noise_std": NOISE_STD,
        }
        return self.result(outcome, metrics, notes, config)


if __name__ == "__main__":
    Exp91Compression64x().execute()
