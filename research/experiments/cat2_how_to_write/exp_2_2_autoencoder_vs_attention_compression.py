"""
Experiment 2.2 — Autoencoder vs Attention-Based Compression

Hypothesis: Attention-based compression produces more retrievable representations
than autoencoder compression, especially on inferential recall tasks.

Setup:
  - FLAT_DIM=512, BOTTLENECK=64 (8x compression)
  - Compressor A: linear autoencoder (encoder → bottleneck → decoder)
  - Compressor B: attention pooling over SEQ_LEN=8 hidden states → 64-dim vector
  - Three recall tasks: exact recall, fuzzy recall, inferential recall
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
SEQ_LEN      = 8
FLAT_DIM     = SEQ_LEN * HIDDEN_DIM   # 512
BOTTLENECK   = 64                      # 8x compression
BATCH_SIZE   = 32
TRAIN_STEPS  = 2000
EVAL_BATCHES = 200
LR           = 1e-3
DEVICE       = "cpu"
NOISE_STD    = 0.1
N_HEADS      = 4

# ── Models ────────────────────────────────────────────────────────────────────

class AutoencoderCompressor(nn.Module):
    """Linear encoder → bottleneck → linear decoder."""

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

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        return z, self.decode(z)


class AttentionCompressor(nn.Module):
    """Pools SEQ_LEN hidden states into BOTTLENECK-dim via a learned query + MHA."""

    def __init__(self) -> None:
        super().__init__()
        # learned query used to pool the SEQ_LEN positions
        self.query = nn.Parameter(torch.randn(1, 1, HIDDEN_DIM))
        self.attn  = nn.MultiheadAttention(HIDDEN_DIM, N_HEADS, batch_first=True)
        self.proj  = nn.Linear(HIDDEN_DIM, BOTTLENECK)
        mid = BOTTLENECK * 2
        self.decoder = nn.Sequential(
            nn.Linear(BOTTLENECK, mid),
            nn.ReLU(),
            nn.Linear(mid, FLAT_DIM),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, FLAT_DIM) → z: (B, BOTTLENECK)"""
        B = x.size(0)
        h = x.view(B, SEQ_LEN, HIDDEN_DIM)          # (B, L, E)
        q = self.query.expand(B, -1, -1)             # (B, 1, E)
        out, _ = self.attn(q, h, h)                  # (B, 1, E)
        return self.proj(out.squeeze(1))             # (B, BOTTLENECK)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        return z, self.decode(z)


# ── Data helpers ──────────────────────────────────────────────────────────────

def make_batch(batch_size: int, embed: nn.Embedding) -> torch.Tensor:
    tokens = torch.randint(0, VOCAB_SIZE, (batch_size, SEQ_LEN))
    with torch.no_grad():
        h = embed(tokens)               # (B, L, E)
    return h.view(batch_size, -1)       # (B, FLAT_DIM)


# ── Training / evaluation ─────────────────────────────────────────────────────

def train_compressor(model: nn.Module, embed: nn.Embedding) -> None:
    opt = Adam(model.parameters(), lr=LR)
    for step in range(TRAIN_STEPS):
        x = make_batch(BATCH_SIZE, embed)
        _, x_hat = model(x)
        loss = F.mse_loss(x_hat, x)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (step + 1) % 500 == 0:
            print(f"    step {step+1}/{TRAIN_STEPS}  loss={loss.item():.4f}")


def eval_exact_cosim(model: nn.Module, embed: nn.Embedding) -> float:
    """Cosine similarity between original and reconstruction."""
    sims: list[float] = []
    with torch.no_grad():
        for _ in range(EVAL_BATCHES):
            x = make_batch(BATCH_SIZE, embed)
            _, x_hat = model(x)
            sims.append(F.cosine_similarity(x, x_hat, dim=-1).mean().item())
    return sum(sims) / len(sims)


def eval_fuzzy_acc(
    model: nn.Module,
    embed: nn.Embedding,
    threshold: float = 0.5,
) -> float:
    """Encode noisy input, decode, check cosim to clean original > threshold."""
    accs: list[float] = []
    with torch.no_grad():
        for _ in range(EVAL_BATCHES):
            x = make_batch(BATCH_SIZE, embed)
            x_noisy = x + torch.randn_like(x) * NOISE_STD
            _, x_hat = model(x_noisy)
            sim = F.cosine_similarity(x, x_hat, dim=-1)
            accs.append((sim > threshold).float().mean().item())
    return sum(accs) / len(accs)


def eval_inferential_acc(model: nn.Module, embed: nn.Embedding) -> float:
    """
    Compress X1 and X2 independently. Average the bottleneck vectors.
    Decode the average. Check whether the decoded average is closer to
    avg(X1, X2) than to either X1 or X2.
    """
    accs: list[float] = []
    with torch.no_grad():
        for _ in range(EVAL_BATCHES):
            x1 = make_batch(BATCH_SIZE, embed)
            x2 = make_batch(BATCH_SIZE, embed)
            x_avg = (x1 + x2) / 2.0

            z1 = model.encode(x1)
            z2 = model.encode(x2)
            z_avg = (z1 + z2) / 2.0
            x_hat_avg = model.decode(z_avg)

            sim_avg = F.cosine_similarity(x_hat_avg, x_avg, dim=-1)
            sim_x1  = F.cosine_similarity(x_hat_avg, x1,    dim=-1)
            sim_x2  = F.cosine_similarity(x_hat_avg, x2,    dim=-1)

            correct = (sim_avg > sim_x1) & (sim_avg > sim_x2)
            accs.append(correct.float().mean().item())
    return sum(accs) / len(accs)


# ── Experiment ────────────────────────────────────────────────────────────────

class Exp22AutoencoderVsAttentionCompression(Experiment):
    experiment_id = "exp_2_2"
    hypothesis = (
        "Attention-based compression produces more retrievable representations than "
        "autoencoder compression, especially on inferential recall tasks."
    )

    def run(self) -> ExperimentResult:
        torch.manual_seed(42)
        embed = nn.Embedding(VOCAB_SIZE, HIDDEN_DIM).to(DEVICE)

        autoenc   = AutoencoderCompressor().to(DEVICE)
        attn_comp = AttentionCompressor().to(DEVICE)

        print("\nTraining autoencoder compressor...")
        train_compressor(autoenc, embed)

        print("Training attention compressor...")
        train_compressor(attn_comp, embed)

        print("\nEvaluating...")
        autoenc.eval()
        attn_comp.eval()

        exact_cosim_autoenc    = eval_exact_cosim(autoenc,    embed)
        exact_cosim_attention  = eval_exact_cosim(attn_comp,  embed)
        fuzzy_acc_autoenc      = eval_fuzzy_acc(autoenc,      embed)
        fuzzy_acc_attention    = eval_fuzzy_acc(attn_comp,    embed)
        inferential_acc_autoenc   = eval_inferential_acc(autoenc,   embed)
        inferential_acc_attention = eval_inferential_acc(attn_comp, embed)

        print(f"  exact_cosim:      autoenc={exact_cosim_autoenc:.4f}  "
              f"attention={exact_cosim_attention:.4f}")
        print(f"  fuzzy_acc:        autoenc={fuzzy_acc_autoenc:.4f}  "
              f"attention={fuzzy_acc_attention:.4f}")
        print(f"  inferential_acc:  autoenc={inferential_acc_autoenc:.4f}  "
              f"attention={inferential_acc_attention:.4f}")

        inferential_gain = inferential_acc_attention - inferential_acc_autoenc
        autoenc_wins_all = (
            exact_cosim_autoenc    > exact_cosim_attention  and
            fuzzy_acc_autoenc      > fuzzy_acc_attention    and
            inferential_acc_autoenc > inferential_acc_attention
        )

        if inferential_gain > 0.05:
            outcome = OUTCOME_SUPPORTED
        elif autoenc_wins_all:
            outcome = OUTCOME_REFUTED
        else:
            outcome = OUTCOME_INCONCLUSIVE

        metrics = {
            "exact_cosim_autoenc":         exact_cosim_autoenc,
            "exact_cosim_attention":        exact_cosim_attention,
            "fuzzy_acc_autoenc":            fuzzy_acc_autoenc,
            "fuzzy_acc_attention":          fuzzy_acc_attention,
            "inferential_acc_autoenc":      inferential_acc_autoenc,
            "inferential_acc_attention":    inferential_acc_attention,
            "inferential_gain_attn_over_autoenc": inferential_gain,
        }
        notes = (
            f"Inferential recall: attention={inferential_acc_attention:.3f}, "
            f"autoenc={inferential_acc_autoenc:.3f}, gain={inferential_gain:+.3f}. "
            f"Attention wins on inferential task by >5pp: {inferential_gain > 0.05}. "
            f"Autoencoder wins all three tasks: {autoenc_wins_all}."
        )
        config = {
            "flat_dim": FLAT_DIM, "bottleneck": BOTTLENECK,
            "seq_len": SEQ_LEN, "train_steps": TRAIN_STEPS,
            "noise_std": NOISE_STD, "n_heads": N_HEADS,
        }
        return self.result(outcome, metrics, notes, config)


if __name__ == "__main__":
    Exp22AutoencoderVsAttentionCompression().execute()
