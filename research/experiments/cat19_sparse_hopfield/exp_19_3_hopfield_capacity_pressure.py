"""
exp_19_3_hopfield_capacity_pressure.py

Hypothesis: Sparse Hopfield sustains accuracy 2+ patterns longer than dense
before capacity cliff.
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
VOCAB_SIZE = 128
HIDDEN_DIM = 64
MEMORY_SLOTS = 16
SEQ_LEN = 32
STEPS = 400
BATCH = 32
LR = 3e-4
DEVICE = "cpu"
NUM_PATTERNS_RANGE = list(range(4, 25, 4))   # [4, 8, 12, 16, 20, 24]


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
# Sparse softmax
# ---------------------------------------------------------------------------
def k_sparse_softmax(scores, k):
    """scores: (B, N) — only top-k entries non-zero after softmax."""
    B, N = scores.shape
    k = min(k, N)
    topk_vals, topk_idx = scores.topk(k, dim=-1)
    sparse_scores = torch.full_like(scores, float('-inf'))
    sparse_scores.scatter_(-1, topk_idx, topk_vals)
    return torch.softmax(sparse_scores, dim=-1)


# ---------------------------------------------------------------------------
# Memory read/write with round-robin fill + configurable attention
# ---------------------------------------------------------------------------
class CapacityTestMemory(nn.Module):
    """Round-robin write (no learned gate), configurable read attention."""

    def __init__(self, hidden_dim, memory_slots, read_mode="softmax", read_k=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.memory_slots = memory_slots
        self.read_mode = read_mode
        self.read_k = read_k
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_head = nn.Linear(hidden_dim, VOCAB_SIZE)

    def forward(self, enc_hidden, query_hidden, num_pairs):
        """
        Write: round-robin, fill slots with KV tokens in order.
        When num_pairs > memory_slots, use circular buffer.
        """
        B, T, H = enc_hidden.shape
        memory = torch.zeros(B, self.memory_slots, H, device=enc_hidden.device)

        # KV token positions: even positions 0, 2, 4, ...
        kv_positions = list(range(0, num_pairs * 2, 1))[:num_pairs * 2]
        slot = 0
        for pos in range(0, min(num_pairs * 2, T - 3)):
            if pos >= T - 3:
                break
            cur_slot = slot % self.memory_slots   # circular buffer
            token_h = enc_hidden[:, pos]           # (B, H)
            memory[:, cur_slot] = token_h
            slot += 1

        # Read
        q = self.q_proj(query_hidden).unsqueeze(1)          # (B, 1, H)
        k = self.k_proj(memory)                              # (B, M, H)
        raw = (q @ k.transpose(1, 2)).squeeze(1) / (H ** 0.5)  # (B, M)

        if self.read_mode == "softmax":
            attn = torch.softmax(raw, dim=-1)
        else:  # "sparse"
            attn = k_sparse_softmax(raw, self.read_k)

        retrieved = (attn.unsqueeze(1) @ memory).squeeze(1)     # (B, H)
        logits = self.out_head(retrieved + query_hidden)
        return logits


# ---------------------------------------------------------------------------
# Train one model for a given (read_mode, num_pairs) combination
# ---------------------------------------------------------------------------
def train_one(read_mode, num_pairs, read_k=2):
    torch.manual_seed(42)
    encoder = Encoder(VOCAB_SIZE, HIDDEN_DIM).to(DEVICE)
    mem = CapacityTestMemory(HIDDEN_DIM, MEMORY_SLOTS, read_mode=read_mode, read_k=read_k).to(DEVICE)
    opt = Adam(list(encoder.parameters()) + list(mem.parameters()), lr=LR)

    correct = total = 0
    for step in range(STEPS):
        seq, target = make_assoc_batch(BATCH, SEQ_LEN, VOCAB_SIZE, num_pairs)
        seq, target = seq.to(DEVICE), target.to(DEVICE)
        enc_h = encoder(seq)
        query_h = enc_h[:, SEQ_LEN - 2]
        logits = mem(enc_h, query_h, num_pairs)
        loss = F.cross_entropy(logits, target)
        opt.zero_grad(); loss.backward(); opt.step()
        if step >= STEPS * 3 // 4:
            correct += (logits.argmax(-1) == target).sum().item()
            total += BATCH
    return correct / max(total, 1)


# ---------------------------------------------------------------------------
# Capacity cliff finder
# ---------------------------------------------------------------------------
def find_cliff(acc_by_patterns, patterns):
    """Returns first num_patterns where acc drops >= 0.15 from peak."""
    peak = max(acc_by_patterns)
    for i, (p, a) in enumerate(zip(patterns, acc_by_patterns)):
        if peak - a >= 0.15:
            return p
    return patterns[-1] + 4   # never cliffs in tested range


# ---------------------------------------------------------------------------
# Experiment class
# ---------------------------------------------------------------------------
class Exp193HopfieldCapacityPressure(Experiment):
    experiment_id = "exp_19_3"
    hypothesis = (
        "Sparse Hopfield sustains accuracy 2+ patterns longer than dense "
        "before capacity cliff."
    )

    def run(self) -> ExperimentResult:
        config = dict(
            VOCAB_SIZE=VOCAB_SIZE, HIDDEN_DIM=HIDDEN_DIM, MEMORY_SLOTS=MEMORY_SLOTS,
            SEQ_LEN=SEQ_LEN, STEPS=STEPS, BATCH=BATCH, LR=LR,
            NUM_PATTERNS_RANGE=NUM_PATTERNS_RANGE,
        )

        acc_soft = []
        acc_sparse = []

        for num_patterns in NUM_PATTERNS_RANGE:
            print(f"  num_patterns={num_patterns}")
            a_soft = train_one("softmax", num_patterns)
            a_sparse = train_one("sparse", num_patterns, read_k=2)
            acc_soft.append(a_soft)
            acc_sparse.append(a_sparse)
            print(f"    soft={a_soft:.4f}  sparse={a_sparse:.4f}")

        cliff_soft = find_cliff(acc_soft, NUM_PATTERNS_RANGE)
        cliff_sparse = find_cliff(acc_sparse, NUM_PATTERNS_RANGE)
        cliff_diff = cliff_sparse - cliff_soft

        metrics = {
            f"acc_soft_{p}": round(a, 4)
            for p, a in zip(NUM_PATTERNS_RANGE, acc_soft)
        }
        metrics.update({
            f"acc_sparse_{p}": round(a, 4)
            for p, a in zip(NUM_PATTERNS_RANGE, acc_sparse)
        })
        metrics["capacity_cliff_soft"] = cliff_soft
        metrics["capacity_cliff_sparse"] = cliff_sparse
        metrics["cliff_diff_sparse_minus_soft"] = cliff_diff

        if cliff_diff >= 2:
            outcome = OUTCOME_SUPPORTED
            notes = f"Sparse delays capacity cliff by {cliff_diff} pattern levels vs soft."
        elif cliff_diff < 1:
            outcome = OUTCOME_REFUTED
            notes = f"Capacity cliff differs by only {cliff_diff} (< 1)."
        else:
            outcome = OUTCOME_INCONCLUSIVE
            notes = f"Sparse delays cliff by exactly {cliff_diff} level(s)."

        return self.result(outcome, metrics=metrics, notes=notes, config=config)


if __name__ == "__main__":
    Exp193HopfieldCapacityPressure().execute()
