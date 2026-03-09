"""
Experiment 8.2 — Gate Activity Equilibrium vs Task Memory Demand

Hypothesis: The natural ~16-20% gate activity equilibrium is not fixed by architecture
but scales with task memory demand — harder tasks requiring more KV pairs will drive
equilibrium write rates upward.
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

VOCAB_SIZE    = 64
HIDDEN_DIM    = 64
SEQ_LEN       = 32
MEMORY_SLOTS  = 8
BATCH_SIZE    = 32
TRAIN_STEPS   = 2000
LR            = 3e-4
DEVICE        = "cpu"
KV_LEVELS     = [1, 2, 4, 6]   # num_pairs difficulty sweep

# ── Data ──────────────────────────────────────────────────────────────────────

def make_assoc_batch(batch_size, seq_len=SEQ_LEN, vocab_size=VOCAB_SIZE, num_pairs=4):
    seq = torch.zeros(batch_size, seq_len, dtype=torch.long)
    target = torch.zeros(batch_size, dtype=torch.long)
    for b in range(batch_size):
        keys = torch.randint(4, 4 + num_pairs * 4, (num_pairs * 2,)).unique()[:num_pairs]
        while len(keys) < num_pairs:
            keys = torch.cat([keys, torch.randint(4, vocab_size // 2, (1,))])[:num_pairs]
        vals = torch.randint(vocab_size // 2, vocab_size, (num_pairs,))
        pos = 0
        for i in range(num_pairs):
            if pos + 1 < seq_len - 3:
                seq[b, pos] = keys[i]
                seq[b, pos + 1] = vals[i]
                pos += 2
        for p in range(pos, seq_len - 3):
            seq[b, p] = 3
        qi = torch.randint(0, num_pairs, (1,)).item()
        seq[b, seq_len - 3] = 2
        seq[b, seq_len - 2] = keys[qi]
        seq[b, seq_len - 1] = 0
        target[b] = vals[qi]
    return seq, target


# ── Model ─────────────────────────────────────────────────────────────────────

class Encoder(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, hidden_dim=HIDDEN_DIM):
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


class WriteGate(nn.Module):
    def __init__(self, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.gate = nn.Linear(hidden_dim, 1)

    def forward(self, h):
        return torch.sigmoid(self.gate(h)).squeeze(-1)  # (B, L)


class ReadHead(nn.Module):
    def __init__(self, hidden_dim=HIDDEN_DIM, vocab_size=VOCAB_SIZE, memory_slots=MEMORY_SLOTS):
        super().__init__()
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, query_h, memory, mask):
        q = self.q_proj(query_h)
        scores = torch.bmm(memory, q.unsqueeze(-1)).squeeze(-1)
        scores = scores.masked_fill(mask == 0, -1e9)
        attn = torch.softmax(scores, dim=-1)
        ctx = (attn.unsqueeze(-1) * memory).sum(1)
        return self.out(ctx)


# ── Training helper ───────────────────────────────────────────────────────────

def train_and_measure(num_pairs: int) -> tuple[float, float]:
    """Train model for a given difficulty level. Returns (accuracy, mean_write_rate)."""
    enc = Encoder().to(DEVICE)
    gate = WriteGate().to(DEVICE)
    rh = ReadHead().to(DEVICE)
    opt = Adam(list(enc.parameters()) + list(gate.parameters()) + list(rh.parameters()), lr=LR)

    enc.train(); gate.train(); rh.train()
    for step in range(TRAIN_STEPS):
        seq, target = make_assoc_batch(BATCH_SIZE, num_pairs=num_pairs)
        hidden = enc(seq)          # (B, L, H)
        scores = gate(hidden)      # (B, L) — sigmoid gate activations

        # Select top-k tokens by gate score as memory
        k = min(MEMORY_SLOTS, SEQ_LEN)
        topk_idx = scores.topk(k, dim=1).indices   # (B, k)
        B, L, H = hidden.shape
        memory = hidden.gather(1, topk_idx.unsqueeze(-1).expand(-1, -1, H))  # (B, k, H)
        mask = torch.ones(B, k, device=DEVICE)

        logits = rh(hidden[:, -1, :], memory, mask)
        loss = F.cross_entropy(logits, target)
        opt.zero_grad()
        loss.backward()
        opt.step()

    # Measure steady-state write rate = mean sigmoid output across tokens
    enc.eval(); gate.eval(); rh.eval()
    total_acc = 0.0
    total_rate = 0.0
    n_eval = 500
    n_batches = n_eval // BATCH_SIZE + 1
    with torch.no_grad():
        for _ in range(n_batches):
            seq, target = make_assoc_batch(BATCH_SIZE, num_pairs=num_pairs)
            hidden = enc(seq)
            scores = gate(hidden)   # (B, L)
            total_rate += scores.mean().item()

            k = min(MEMORY_SLOTS, SEQ_LEN)
            topk_idx = scores.topk(k, dim=1).indices
            B, L, H = hidden.shape
            memory = hidden.gather(1, topk_idx.unsqueeze(-1).expand(-1, -1, H))
            mask = torch.ones(B, k, device=DEVICE)
            logits = rh(hidden[:, -1, :], memory, mask)
            total_acc += (logits.argmax(-1) == target).float().mean().item()

    mean_rate = total_rate / n_batches
    mean_acc = total_acc / n_batches
    return mean_acc, mean_rate


# ── Pearson correlation ───────────────────────────────────────────────────────

def pearson_r(x: list, y: list) -> float:
    xt = torch.tensor(x, dtype=torch.float)
    yt = torch.tensor(y, dtype=torch.float)
    xm = xt - xt.mean()
    ym = yt - yt.mean()
    denom = (xm.norm() * ym.norm()).clamp(min=1e-8)
    return ((xm * ym).sum() / denom).item()


# ── Experiment class ──────────────────────────────────────────────────────────

class Exp82GateEquilibrium(Experiment):
    experiment_id = "exp_8_2"
    hypothesis = (
        "The natural ~16-20% gate activity equilibrium is not fixed by architecture but "
        "scales with task memory demand — harder tasks requiring more KV pairs will drive "
        "equilibrium write rates upward."
    )

    def run(self) -> ExperimentResult:
        write_rates = []
        accuracies = []

        for num_pairs in KV_LEVELS:
            print(f"  Training with num_pairs={num_pairs} ...")
            acc, rate = train_and_measure(num_pairs)
            write_rates.append(rate)
            accuracies.append(acc)
            print(f"    acc={acc:.3f}, write_rate={rate:.4f}")

        r = pearson_r(list(range(len(KV_LEVELS))), write_rates)
        rate_variance = torch.tensor(write_rates).var().item()

        print(f"  Pearson r(difficulty_index, write_rate) = {r:.4f}")
        print(f"  Write rate variance = {rate_variance:.6f}")
        print(f"  Write rates: {[round(w, 4) for w in write_rates]}")

        if r > 0.8:
            outcome = OUTCOME_SUPPORTED
        elif rate_variance < 0.02:
            outcome = OUTCOME_REFUTED
        else:
            outcome = OUTCOME_INCONCLUSIVE

        metrics = {
            "pearson_r_difficulty_vs_rate": round(r, 4),
            "write_rate_variance": round(rate_variance, 6),
            "write_rates": [round(w, 4) for w in write_rates],
            "accuracies":  [round(a, 4) for a in accuracies],
            "kv_levels":   KV_LEVELS,
        }
        notes = (
            f"KV levels {KV_LEVELS}: write rates {[round(w,4) for w in write_rates]}. "
            f"Pearson r={r:.4f}, variance={rate_variance:.6f}."
        )
        config = {
            "vocab_size": VOCAB_SIZE, "hidden_dim": HIDDEN_DIM,
            "seq_len": SEQ_LEN, "memory_slots": MEMORY_SLOTS,
            "train_steps": TRAIN_STEPS, "batch_size": BATCH_SIZE,
            "kv_levels": KV_LEVELS,
        }
        return self.result(outcome, metrics, notes, config)


if __name__ == "__main__":
    Exp82GateEquilibrium().execute()
