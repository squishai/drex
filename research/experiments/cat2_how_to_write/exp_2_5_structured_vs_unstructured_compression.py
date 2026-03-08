"""
Experiment 2.5 — Structured vs Unstructured Compression

Hypothesis: Compressing into a structured key-value representation improves
retrieval over compressing into a flat dense vector.

Setup:
  - SEQ_LEN=8 input sequences, EMBED_DIM=64 per token, FLAT_DIM=512
  - Compressor A (flat): linear encoder → 64-dim flat vector → linear decoder
  - Compressor B (structured): 4 slots × (key=8-dim, value=8-dim) = 64 floats total
      Keys are encoded from the full input; values hold the content.
      Retrieval: query (one noisy segment → 8-dim) dot-products all 4 keys,
      softmax weights, weighted-sum values → project back to EMBED_DIM.
  - Both store 64 floats per input (same budget).
  - Retrieval test: noisy version of one segment → retrieve that segment.
  - flat_acc / structured_acc = fraction of samples with cosim > ACC_THRESHOLD.
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

VOCAB_SIZE     = 256
HIDDEN_DIM     = 64      # per-token embedding dimension
SEQ_LEN        = 8
FLAT_DIM       = SEQ_LEN * HIDDEN_DIM   # 512
BOTTLENECK     = 64      # flat compressor output dim
N_SLOTS        = 4       # structured compressor slots
KEY_DIM        = 8       # 4 × 8 = 32 floats for keys
VAL_DIM        = 8       # 4 × 8 = 32 floats for values  (total = 64 ✓)
BATCH_SIZE     = 32
TRAIN_STEPS    = 2000
EVAL_BATCHES   = 200
LR             = 1e-3
DEVICE         = "cpu"
NOISE_STD      = 0.15
ACC_THRESHOLD  = 0.5     # cosim threshold for counting a retrieval as correct


# ── Models ────────────────────────────────────────────────────────────────────

class FlatCompressor(nn.Module):
    """Linear autoencoder trained on noisy input to reconstruct the clean original."""

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, FLAT_DIM) → reconstructed: (B, FLAT_DIM)"""
        return self.decoder(self.encoder(x))


class StructuredCompressor(nn.Module):
    """
    Encodes full input into N_SLOTS key-value pairs (KEY_DIM + VAL_DIM each).
    Retrieval: a noisy segment is projected to a query, used to attend over keys,
    returns weighted sum of values projected back to HIDDEN_DIM.
    """

    def __init__(self) -> None:
        super().__init__()
        # Encoder: flat input → keys and values
        self.key_encoder = nn.Sequential(
            nn.Linear(FLAT_DIM, 64),
            nn.ReLU(),
            nn.Linear(64, N_SLOTS * KEY_DIM),
        )
        self.val_encoder = nn.Sequential(
            nn.Linear(FLAT_DIM, 64),
            nn.ReLU(),
            nn.Linear(64, N_SLOTS * VAL_DIM),
        )
        # Query: noisy segment → KEY_DIM
        self.query_proj = nn.Linear(HIDDEN_DIM, KEY_DIM)
        # Value projection: retrieved VAL_DIM → HIDDEN_DIM
        self.val_proj   = nn.Linear(VAL_DIM, HIDDEN_DIM)

    def encode(
        self, x_flat: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """x_flat: (B, FLAT_DIM) → K: (B, N_SLOTS, KEY_DIM), V: (B, N_SLOTS, VAL_DIM)"""
        B  = x_flat.size(0)
        K  = self.key_encoder(x_flat).view(B, N_SLOTS, KEY_DIM)
        V  = self.val_encoder(x_flat).view(B, N_SLOTS, VAL_DIM)
        return K, V

    def retrieve(
        self,
        K: torch.Tensor,
        V: torch.Tensor,
        query_seg: torch.Tensor,
    ) -> torch.Tensor:
        """
        K:         (B, N_SLOTS, KEY_DIM)
        V:         (B, N_SLOTS, VAL_DIM)
        query_seg: (B, HIDDEN_DIM)  — noisy segment used as retrieval key
        → output:  (B, HIDDEN_DIM)
        """
        q       = self.query_proj(query_seg)          # (B, KEY_DIM)
        scores  = torch.bmm(K, q.unsqueeze(-1))       # (B, N_SLOTS, 1)
        weights = F.softmax(scores.squeeze(-1), dim=-1)  # (B, N_SLOTS)
        # weighted sum of value vectors
        retrieved = (weights.unsqueeze(-1) * V).sum(dim=1)  # (B, VAL_DIM)
        return self.val_proj(retrieved)                      # (B, HIDDEN_DIM)


# ── Data helpers ──────────────────────────────────────────────────────────────

def make_sequence_batch(
    batch_size: int, embed: nn.Embedding
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns x_segs (B, SEQ_LEN, HIDDEN_DIM) and x_flat (B, FLAT_DIM)."""
    tokens = torch.randint(0, VOCAB_SIZE, (batch_size, SEQ_LEN))
    with torch.no_grad():
        h = embed(tokens)               # (B, SEQ_LEN, HIDDEN_DIM)
    return h, h.view(batch_size, -1)    # segs, flat


# ── Training ──────────────────────────────────────────────────────────────────

def train_flat(model: FlatCompressor, embed: nn.Embedding) -> None:
    opt = Adam(model.parameters(), lr=LR)
    for step in range(TRAIN_STEPS):
        x_segs, x_flat = make_sequence_batch(BATCH_SIZE, embed)
        x_noisy         = x_flat + NOISE_STD * torch.randn_like(x_flat)
        x_hat           = model(x_noisy)
        loss            = F.mse_loss(x_hat, x_flat)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (step + 1) % 500 == 0:
            print(f"    [flat]       step {step+1}/{TRAIN_STEPS}  loss={loss.item():.4f}")


def train_structured(model: StructuredCompressor, embed: nn.Embedding) -> None:
    opt = Adam(model.parameters(), lr=LR)
    for step in range(TRAIN_STEPS):
        x_segs, x_flat = make_sequence_batch(BATCH_SIZE, embed)
        # pick a random segment index as the target for this step
        seg_idx    = torch.randint(0, SEQ_LEN, (1,)).item()
        target_seg = x_segs[:, seg_idx, :]                          # (B, HIDDEN_DIM)
        noisy_seg  = target_seg + NOISE_STD * torch.randn_like(target_seg)

        K, V       = model.encode(x_flat)
        retrieved  = model.retrieve(K, V, noisy_seg)
        loss       = F.mse_loss(retrieved, target_seg)

        opt.zero_grad()
        loss.backward()
        opt.step()
        if (step + 1) % 500 == 0:
            print(f"    [structured] step {step+1}/{TRAIN_STEPS}  loss={loss.item():.4f}")


# ── Evaluation ────────────────────────────────────────────────────────────────

def eval_flat(
    model: FlatCompressor, embed: nn.Embedding
) -> tuple[float, float]:
    """Returns (mean_cosim, acc@threshold)."""
    cosims: list[float] = []
    accs:   list[float] = []
    with torch.no_grad():
        for _ in range(EVAL_BATCHES):
            x_segs, x_flat = make_sequence_batch(BATCH_SIZE, embed)
            seg_idx         = torch.randint(0, SEQ_LEN, (1,)).item()
            target_seg      = x_segs[:, seg_idx, :]           # (B, HIDDEN_DIM)
            x_noisy         = x_flat + NOISE_STD * torch.randn_like(x_flat)

            x_hat     = model(x_noisy)                        # (B, FLAT_DIM)
            x_hat_seg = x_hat.view(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM)[:, seg_idx, :]

            sim = F.cosine_similarity(x_hat_seg, target_seg, dim=-1)
            cosims.append(sim.mean().item())
            accs.append((sim > ACC_THRESHOLD).float().mean().item())
    return sum(cosims) / len(cosims), sum(accs) / len(accs)


def eval_structured(
    model: StructuredCompressor, embed: nn.Embedding
) -> tuple[float, float]:
    """Returns (mean_cosim, acc@threshold)."""
    cosims: list[float] = []
    accs:   list[float] = []
    with torch.no_grad():
        for _ in range(EVAL_BATCHES):
            x_segs, x_flat = make_sequence_batch(BATCH_SIZE, embed)
            seg_idx         = torch.randint(0, SEQ_LEN, (1,)).item()
            target_seg      = x_segs[:, seg_idx, :]
            noisy_seg       = target_seg + NOISE_STD * torch.randn_like(target_seg)

            K, V      = model.encode(x_flat)
            retrieved = model.retrieve(K, V, noisy_seg)        # (B, HIDDEN_DIM)

            sim = F.cosine_similarity(retrieved, target_seg, dim=-1)
            cosims.append(sim.mean().item())
            accs.append((sim > ACC_THRESHOLD).float().mean().item())
    return sum(cosims) / len(cosims), sum(accs) / len(accs)


# ── Experiment ────────────────────────────────────────────────────────────────

class Exp25StructuredVsUnstructuredCompression(Experiment):
    experiment_id = "exp_2_5"
    hypothesis = (
        "Compressing into a structured key-value representation improves retrieval "
        "over compressing into a flat dense vector."
    )

    def run(self) -> ExperimentResult:
        torch.manual_seed(42)
        embed      = nn.Embedding(VOCAB_SIZE, HIDDEN_DIM).to(DEVICE)
        flat_model = FlatCompressor().to(DEVICE)
        kv_model   = StructuredCompressor().to(DEVICE)

        print("\nTraining flat compressor...")
        train_flat(flat_model, embed)

        print("Training structured (KV) compressor...")
        train_structured(kv_model, embed)

        print("\nEvaluating...")
        flat_model.eval()
        kv_model.eval()

        flat_cosim,       flat_acc       = eval_flat(flat_model, embed)
        structured_cosim, structured_acc = eval_structured(kv_model, embed)

        print(f"  flat:       cosim={flat_cosim:.4f}  acc={flat_acc:.4f}")
        print(f"  structured: cosim={structured_cosim:.4f}  acc={structured_acc:.4f}")

        acc_gain = structured_acc - flat_acc

        if structured_acc > flat_acc + 0.03:
            outcome = OUTCOME_SUPPORTED
        elif flat_acc > structured_acc:
            outcome = OUTCOME_REFUTED
        else:
            outcome = OUTCOME_INCONCLUSIVE

        metrics = {
            "flat_cosim":       flat_cosim,
            "structured_cosim": structured_cosim,
            "flat_acc":         flat_acc,
            "structured_acc":   structured_acc,
            "acc_gain_structured_over_flat": acc_gain,
        }
        notes = (
            f"Structured acc={structured_acc:.3f}, flat acc={flat_acc:.3f}, "
            f"gain={acc_gain:+.3f}. "
            f"Structured beats flat by >3pp: {acc_gain > 0.03}. "
            f"Flat beats structured: {flat_acc > structured_acc}."
        )
        config = {
            "flat_dim": FLAT_DIM, "bottleneck": BOTTLENECK,
            "n_slots": N_SLOTS, "key_dim": KEY_DIM, "val_dim": VAL_DIM,
            "seq_len": SEQ_LEN, "noise_std": NOISE_STD, "train_steps": TRAIN_STEPS,
            "total_stored_flat": BOTTLENECK,
            "total_stored_structured": N_SLOTS * (KEY_DIM + VAL_DIM),
        }
        return self.result(outcome, metrics, notes, config)


if __name__ == "__main__":
    Exp25StructuredVsUnstructuredCompression().execute()
