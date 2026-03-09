"""
exp_20_1_write_sparsity_auxiliary.py

Hypothesis: L1 auxiliary loss on write gate (targeting ~15% activity) improves
accuracy and avoids degenerate modes.
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

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
VOCAB_SIZE = 64
HIDDEN_DIM = 64
SEQ_LEN = 24
MEMORY_SLOTS = 6
NUM_PAIRS = 4
STEPS = 400
BATCH = 32
K_WRITE = 4
LR = 3e-4
DEVICE = "cpu"


# ---------------------------------------------------------------------------
# Data generator
# ---------------------------------------------------------------------------
def make_assoc_batch(batch_size, seq_len, vocab_size, num_pairs):
    seq = torch.zeros(batch_size, seq_len, dtype=torch.long)
    target = torch.zeros(batch_size, dtype=torch.long)
    for b in range(batch_size):
        keys = torch.randint(4, max(8, vocab_size // 3), (num_pairs * 3,)).unique()[:num_pairs]
        while len(keys) < num_pairs:
            keys = torch.cat([keys, torch.randint(4, vocab_size // 3, (1,))])[:num_pairs]
        vals = torch.randint(vocab_size // 2, vocab_size, (num_pairs,))
        pos = 0
        for i in range(num_pairs):
            if pos + 1 < seq_len - 3:
                seq[b, pos] = keys[i]; seq[b, pos + 1] = vals[i]; pos += 2
        for p in range(pos, seq_len - 3):
            seq[b, p] = 3
        qi = torch.randint(0, num_pairs, (1,)).item()
        seq[b, seq_len - 3] = 2
        seq[b, seq_len - 2] = keys[qi]
        seq[b, seq_len - 1] = 0
        target[b] = vals[qi]
    return seq, target


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------
class Encoder(nn.Module):
    def __init__(self, vocab_size, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        h = self.embed(x)
        return self.norm(h + self.ff(h))


# ---------------------------------------------------------------------------
# Memory module (shared architecture, variable lambda)
# ---------------------------------------------------------------------------
class WriteGateMemory(nn.Module):
    """Write gate with top-k selection; softmax attention read."""

    def __init__(self, hidden_dim, memory_slots, k_write):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.memory_slots = memory_slots
        self.k_write = k_write
        self.write_gate = nn.Linear(hidden_dim, 1)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_head = nn.Linear(hidden_dim, VOCAB_SIZE)

    def forward(self, enc_hidden, query_hidden):
        """Returns logits, sigmoid gate scores (B, T), and mean write rate."""
        B, T, H = enc_hidden.shape

        gate_logits = self.write_gate(enc_hidden).squeeze(-1)    # (B, T)
        gate_scores = torch.sigmoid(gate_logits)                 # (B, T)

        k = min(self.k_write, self.memory_slots, T)
        _, top_idx = gate_scores.topk(k, dim=-1)                # (B, k)

        memory = torch.zeros(B, self.memory_slots, H, device=enc_hidden.device)
        for slot in range(k):
            tok_idx = top_idx[:, slot]
            memory[:, slot] = enc_hidden[torch.arange(B), tok_idx]

        q = self.q_proj(query_hidden).unsqueeze(1)               # (B, 1, H)
        keys = self.k_proj(memory)                               # (B, M, H)
        scores = (q @ keys.transpose(1, 2)).squeeze(1) / (H ** 0.5)  # (B, M)
        attn = torch.softmax(scores, dim=-1)
        retrieved = (attn.unsqueeze(1) @ memory).squeeze(1)     # (B, H)
        logits = self.out_head(retrieved + query_hidden)

        write_rate = gate_scores.mean()                          # scalar tensor
        return logits, gate_scores, write_rate


# ---------------------------------------------------------------------------
# Degenerate mode check
# ---------------------------------------------------------------------------
def is_collapsed(rate: float) -> bool:
    return rate < 0.02 or rate > 0.95


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train_condition(lam: float):
    torch.manual_seed(42)
    encoder = Encoder(VOCAB_SIZE, HIDDEN_DIM).to(DEVICE)
    mem = WriteGateMemory(HIDDEN_DIM, MEMORY_SLOTS, K_WRITE).to(DEVICE)
    opt = Adam(list(encoder.parameters()) + list(mem.parameters()), lr=LR)

    correct = total = 0
    write_rates = []

    for step in range(STEPS):
        seq, target = make_assoc_batch(BATCH, SEQ_LEN, VOCAB_SIZE, NUM_PAIRS)
        seq, target = seq.to(DEVICE), target.to(DEVICE)

        enc_h = encoder(seq)
        query_h = enc_h[:, SEQ_LEN - 2]
        logits, gate_scores, write_rate = mem(enc_h, query_h)

        task_loss = F.cross_entropy(logits, target)
        aux_loss = gate_scores.mean()           # L1 on sigmoid scores
        total_loss = task_loss + lam * aux_loss

        opt.zero_grad()
        total_loss.backward()
        opt.step()

        if step >= STEPS * 3 // 4:
            correct += (logits.argmax(-1) == target).sum().item()
            total += BATCH
            write_rates.append(write_rate.item())

    acc = correct / max(total, 1)
    avg_write_rate = sum(write_rates) / max(len(write_rates), 1)
    return acc, avg_write_rate


# ---------------------------------------------------------------------------
# Experiment class
# ---------------------------------------------------------------------------
class Exp201WriteSparsityAuxiliary(Experiment):
    experiment_id = "exp_20_1"
    hypothesis = (
        "L1 auxiliary loss on write gate (targeting ~15% activity) improves "
        "accuracy and avoids degenerate modes."
    )

    def run(self) -> ExperimentResult:
        config = dict(
            VOCAB_SIZE=VOCAB_SIZE, HIDDEN_DIM=HIDDEN_DIM, SEQ_LEN=SEQ_LEN,
            MEMORY_SLOTS=MEMORY_SLOTS, NUM_PAIRS=NUM_PAIRS, STEPS=STEPS,
            BATCH=BATCH, K_WRITE=K_WRITE, LR=LR,
        )

        print("Training condition A: task loss only (lambda=0)...")
        acc_A, write_rate_A = train_condition(0.0)
        print(f"  acc_A={acc_A:.4f}  write_rate_A={write_rate_A:.4f}")

        print("Training condition B: task + L1 (lambda=0.1)...")
        acc_B, write_rate_B = train_condition(0.1)
        print(f"  acc_B={acc_B:.4f}  write_rate_B={write_rate_B:.4f}")

        print("Training condition C: task + stronger L1 (lambda=0.5)...")
        acc_C, write_rate_C = train_condition(0.5)
        print(f"  acc_C={acc_C:.4f}  write_rate_C={write_rate_C:.4f}")

        col_A = is_collapsed(write_rate_A)
        col_B = is_collapsed(write_rate_B)
        col_C = is_collapsed(write_rate_C)

        metrics = dict(
            acc_A=round(acc_A, 4),
            acc_B=round(acc_B, 4),
            acc_C=round(acc_C, 4),
            write_rate_A=round(write_rate_A, 4),
            write_rate_B=round(write_rate_B, 4),
            write_rate_C=round(write_rate_C, 4),
            is_collapsed_A=col_A,
            is_collapsed_B=col_B,
            is_collapsed_C=col_C,
            acc_gap_B_minus_A=round(acc_B - acc_A, 4),
        )

        if acc_B > acc_A + 0.005 and 0.05 < write_rate_B < 0.35 and not col_B:
            outcome = OUTCOME_SUPPORTED
            notes = (
                f"L1 aux (lambda=0.1) achieves write_rate={write_rate_B:.4f} with "
                f"+{acc_B - acc_A:.4f} accuracy gain, no degenerate mode."
            )
        elif abs(acc_A - acc_B) <= 0.005:
            outcome = OUTCOME_REFUTED
            notes = (
                f"Accuracy difference ({acc_B - acc_A:.4f}) within 0.005 — "
                "auxiliary does not help."
            )
        else:
            outcome = OUTCOME_INCONCLUSIVE
            notes = (
                f"Auxiliary {'fixes degenerate mode' if col_A and not col_B else 'adjusts write rate'} "
                f"but accuracy gain ({acc_B - acc_A:.4f}) below threshold or write_rate out of range."
            )

        return self.result(outcome, metrics=metrics, notes=notes, config=config)


if __name__ == "__main__":
    Exp201WriteSparsityAuxiliary().execute()
