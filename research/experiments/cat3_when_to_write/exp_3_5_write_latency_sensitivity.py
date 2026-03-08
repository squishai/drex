"""
Experiment 3.5 — Write Latency Sensitivity

Hypothesis: Downstream retrieval quality degrades measurably when write latency
exceeds a specific token distance threshold.

Setup:
  - SEQ_LEN=32
  - After a token appears in sequence, introduce artificial delay: write it to
    memory only after L additional tokens have been processed (L=0,2,4,8,16)
  - For each latency L: train the same model, measure retrieval accuracy
  - The "value" tokens in key-value pairs are the things being retrieved
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

# ── Config ─────────────────────────────────────────────────────────────────────

VOCAB_SIZE    = 64
SEQ_LEN       = 32     # override for this experiment
HIDDEN_DIM    = 64
MEMORY_SLOTS  = 6
BATCH_SIZE    = 32
TRAIN_STEPS   = 1500
LR            = 3e-4
LOG_EVERY     = 150
DEVICE        = "cpu"

LATENCIES     = [0, 2, 4, 8, 16]
NUM_PAIRS     = 4
QUERY_MARKER  = 2


# ── Data Generation ────────────────────────────────────────────────────────────

def make_assoc_batch(batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Associative recall sequences. Returns (seq, target, kv_positions).
    kv_positions: (B, SEQ_LEN) marks where key/value tokens appear.
    """
    seq          = torch.zeros(batch_size, SEQ_LEN, dtype=torch.long)
    target       = torch.zeros(batch_size, dtype=torch.long)
    kv_positions = torch.zeros(batch_size, SEQ_LEN, dtype=torch.long)

    for b in range(batch_size):
        keys = torch.randint(4, 4 + NUM_PAIRS * 3, (NUM_PAIRS,)).unique()[:NUM_PAIRS]
        while len(keys) < NUM_PAIRS:
            keys = torch.cat([keys, torch.randint(4, 32, (1,))])[:NUM_PAIRS]
        vals = torch.randint(32, VOCAB_SIZE, (NUM_PAIRS,))

        pos = 0
        for i in range(NUM_PAIRS):
            if pos + 1 < SEQ_LEN - 3:
                seq[b, pos]     = keys[i]
                seq[b, pos + 1] = vals[i]
                kv_positions[b, pos]     = 1
                kv_positions[b, pos + 1] = 1
                pos += 2

        for p in range(pos, SEQ_LEN - 3):
            seq[b, p] = 3

        query_idx      = torch.randint(0, NUM_PAIRS, (1,)).item()
        seq[b, SEQ_LEN - 3] = QUERY_MARKER
        seq[b, SEQ_LEN - 2] = keys[query_idx]
        seq[b, SEQ_LEN - 1] = 0
        target[b] = vals[query_idx]

    return seq, target, kv_positions


# ── Encoder ───────────────────────────────────────────────────────────────────

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, HIDDEN_DIM)
        self.ff    = nn.Sequential(
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM * 2), nn.ReLU(),
            nn.Linear(HIDDEN_DIM * 2, HIDDEN_DIM),
        )
        self.norm  = nn.LayerNorm(HIDDEN_DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embed(x)
        return self.norm(h + self.ff(h))


class ReadHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.out    = nn.Linear(HIDDEN_DIM, VOCAB_SIZE)

    def forward(self, query_h: torch.Tensor, memory: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        q      = self.q_proj(query_h)
        scores = torch.bmm(memory, q.unsqueeze(-1)).squeeze(-1)
        scores = scores.masked_fill(mask == 0, -1e9)
        attn   = torch.softmax(scores, dim=-1)
        ctx    = (attn.unsqueeze(-1) * memory).sum(1)
        return self.out(ctx)


# ── Latency-Delayed Write Policy ───────────────────────────────────────────────

def latency_write(
    hidden: torch.Tensor,
    kv_positions: torch.Tensor,
    latency: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Write kv_position tokens but delayed by `latency` additional tokens.
    For a token at position p, write hidden[p + latency] (i.e., the representation
    available latency steps later).

    If latency=0: write hidden[p] (immediate).
    If the delayed position exceeds SEQ_LEN, clip to SEQ_LEN-1.

    Returns (memory, mask): (B, MEMORY_SLOTS, H), (B, MEMORY_SLOTS).
    """
    B, L, H = hidden.shape
    all_mem  = []
    all_mask = []

    for b in range(B):
        kv_pos = kv_positions[b].nonzero(as_tuple=True)[0].tolist()  # list of positions
        delayed = []
        for p in kv_pos:
            write_at = min(p + latency, L - 1)
            delayed.append(write_at)

        # Select up to MEMORY_SLOTS distinct delayed positions
        seen    = []
        for wp in delayed:
            if wp not in seen:
                seen.append(wp)
        write_positions = seen[:MEMORY_SLOTS]

        if len(write_positions) == 0:
            # Fallback: evenly spaced
            write_positions = list(range(0, L, max(1, L // MEMORY_SLOTS)))[:MEMORY_SLOTS]

        idx_t  = torch.tensor(write_positions, device=hidden.device)
        mem_b  = hidden[b, idx_t, :]                    # (k, H)
        msk_b  = torch.ones(len(write_positions), device=hidden.device)

        if len(write_positions) < MEMORY_SLOTS:
            pad   = MEMORY_SLOTS - len(write_positions)
            mem_b = F.pad(mem_b, (0, 0, 0, pad))
            msk_b = F.pad(msk_b, (0, pad))

        all_mem.append(mem_b)
        all_mask.append(msk_b)

    memory = torch.stack(all_mem,  dim=0)
    mask   = torch.stack(all_mask, dim=0)
    return memory, mask


# ── Training ───────────────────────────────────────────────────────────────────

def train_with_latency(latency: int) -> float:
    """Train a model with the given write latency. Return final accuracy."""
    enc  = Encoder().to(DEVICE)
    head = ReadHead().to(DEVICE)
    opt  = Adam(list(enc.parameters()) + list(head.parameters()), lr=LR)

    for step in range(TRAIN_STEPS):
        seq, target, kv_positions = make_assoc_batch(BATCH_SIZE)
        seq          = seq.to(DEVICE)
        target       = target.to(DEVICE)
        kv_positions = kv_positions.to(DEVICE)

        hidden    = enc(seq)
        memory, mask = latency_write(hidden, kv_positions, latency)

        query_h   = hidden[:, -1, :]
        logits    = head(query_h, memory, mask)
        task_loss = F.cross_entropy(logits, target)

        opt.zero_grad()
        task_loss.backward()
        opt.step()

        if step % LOG_EVERY == 0:
            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                acc   = (preds == target).float().mean().item()
                print(f"  [latency={latency:2d}] step={step:4d}  loss={task_loss.item():.3f}  acc={acc:.3f}")

    # Final evaluation
    total_acc  = 0.0
    eval_steps = 20
    with torch.no_grad():
        for _ in range(eval_steps):
            seq, target, kv_positions = make_assoc_batch(BATCH_SIZE)
            seq          = seq.to(DEVICE)
            target       = target.to(DEVICE)
            kv_positions = kv_positions.to(DEVICE)
            hidden       = enc(seq)
            memory, mask = latency_write(hidden, kv_positions, latency)
            query_h      = hidden[:, -1, :]
            logits       = head(query_h, memory, mask)
            preds        = logits.argmax(dim=-1)
            total_acc   += (preds == target).float().mean().item()

    return total_acc / eval_steps


# ── Experiment ─────────────────────────────────────────────────────────────────

class Exp35WriteLatencySensitivity(Experiment):
    experiment_id = "exp_3_5"
    hypothesis = (
        "Downstream retrieval quality degrades measurably when write latency "
        "exceeds a specific token distance threshold."
    )

    def run(self) -> ExperimentResult:
        torch.manual_seed(42)

        acc_by_latency: dict[int, float] = {}
        for L in LATENCIES:
            print(f"\nTraining with write latency L={L}...")
            acc = train_with_latency(L)
            acc_by_latency[L] = acc
            print(f"  -> Final acc at L={L}: {acc:.3f}")

        baseline_acc = acc_by_latency[0]
        threshold    = None
        for L in LATENCIES[1:]:
            if acc_by_latency[L] < baseline_acc - 0.05:
                threshold = L
                break

        print(f"\nAccuracy by latency: {acc_by_latency}")
        print(f"Latency threshold (first drop >0.05): {threshold}")

        # Check if accuracy is flat
        acc_values = [acc_by_latency[L] for L in LATENCIES]
        acc_range  = max(acc_values) - min(acc_values)
        is_flat    = acc_range < 0.05

        if threshold is not None and threshold <= 8:
            outcome = OUTCOME_SUPPORTED
        elif is_flat:
            outcome = OUTCOME_REFUTED
        else:
            outcome = OUTCOME_INCONCLUSIVE

        metrics = {
            f"acc_at_latency_{L}": round(acc_by_latency[L], 4)
            for L in LATENCIES
        }
        metrics["latency_threshold"] = threshold
        metrics["acc_range"]         = round(acc_range, 4)

        notes = (
            f"Baseline (L=0): {baseline_acc:.3f}. "
            f"Accuracy range across latencies: {acc_range:.3f}. "
            f"First threshold where drop >0.05: {threshold}. "
            f"Flat (no degradation): {is_flat}."
        )
        config = {
            "vocab_size":   VOCAB_SIZE,
            "seq_len":      SEQ_LEN,
            "hidden_dim":   HIDDEN_DIM,
            "memory_slots": MEMORY_SLOTS,
            "batch_size":   BATCH_SIZE,
            "train_steps":  TRAIN_STEPS,
            "latencies":    LATENCIES,
            "num_pairs":    NUM_PAIRS,
        }
        return self.result(outcome, metrics, notes, config)


if __name__ == "__main__":
    Exp35WriteLatencySensitivity().execute()
