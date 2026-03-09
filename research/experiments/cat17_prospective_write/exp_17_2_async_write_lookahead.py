"""
exp_17_2_async_write_lookahead.py

Hypothesis: K-token lookahead async write gate outperforms same-time write by
>3% at some K in {2, 4, 6}.
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

# ── Constants ──────────────────────────────────────────────────────────────────
DEVICE       = "cpu"
LR           = 3e-4
VOCAB_SIZE   = 64
HIDDEN_DIM   = 64
SEQ_LEN      = 32
MEMORY_SLOTS = 8
NUM_PAIRS    = 4
STEPS        = 400
BATCH        = 32
K_VALUES     = [0, 2, 4, 6]


# ── Data generator ─────────────────────────────────────────────────────────────
def make_assoc_batch(batch_size, seq_len, vocab_size, num_pairs):
    seq    = torch.zeros(batch_size, seq_len, dtype=torch.long)
    target = torch.zeros(batch_size, dtype=torch.long)
    for b in range(batch_size):
        keys = torch.randint(4, max(8, vocab_size // 3), (num_pairs * 3,)).unique()[:num_pairs]
        while len(keys) < num_pairs:
            keys = torch.cat([keys, torch.randint(4, vocab_size // 3, (1,))])[:num_pairs]
        vals = torch.randint(vocab_size // 2, vocab_size, (num_pairs,))
        pos  = 0
        for i in range(num_pairs):
            if pos + 1 < seq_len - 3:
                seq[b, pos] = keys[i]; seq[b, pos + 1] = vals[i]; pos += 2
        for p in range(pos, seq_len - 3):
            seq[b, p] = 3
        qi                  = torch.randint(0, num_pairs, (1,)).item()
        seq[b, seq_len - 3] = 2
        seq[b, seq_len - 2] = keys[qi]
        seq[b, seq_len - 1] = 0
        target[b]           = vals[qi]
    return seq, target


# ── Shared modules ─────────────────────────────────────────────────────────────
class Encoder(nn.Module):
    def __init__(self, vocab_size, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.ff    = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm  = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        h = self.embed(x)
        return self.norm(h + self.ff(h))


class ReadHead(nn.Module):
    def __init__(self, hidden_dim, vocab_size):
        super().__init__()
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out    = nn.Linear(hidden_dim, vocab_size)

    def forward(self, query_h, memory, mask):
        q      = self.q_proj(query_h)
        scores = torch.bmm(memory, q.unsqueeze(-1)).squeeze(-1)
        scores = scores.masked_fill(mask == 0, -1e9)
        attn   = torch.softmax(scores, dim=-1)
        ctx    = (attn.unsqueeze(-1) * memory).sum(1)
        return self.out(ctx)


# ── Lookahead model ────────────────────────────────────────────────────────────
class LookaheadModel(nn.Module):
    """
    For K=0: gate(hidden[t]) -> sigmoid score.
    For K>0: gate(cat(hidden[t], mean(hidden[t+1:t+1+K]))) -> sigmoid score.
    Zeros pad when window exceeds sequence length.
    """

    def __init__(self, vocab_size, hidden_dim, memory_slots, k):
        super().__init__()
        self.encoder      = Encoder(vocab_size, hidden_dim)
        gate_input_dim    = hidden_dim if k == 0 else hidden_dim * 2
        self.gate_net     = nn.Linear(gate_input_dim, 1)
        self.read_head    = ReadHead(hidden_dim, vocab_size)
        self.hidden_dim   = hidden_dim
        self.memory_slots = memory_slots
        self.k            = k

    def forward(self, seq):
        B, T = seq.shape
        h    = self.encoder(seq)  # (B, T, H)

        if self.k == 0:
            gate_input = h
        else:
            # Build lookahead context for every position
            # future_ctx[t] = mean(h[t+1 : t+1+K]), padded with zeros
            future_ctx = torch.zeros_like(h)
            for t in range(T):
                start = t + 1
                end   = min(t + 1 + self.k, T)
                if start < T:
                    future_ctx[:, t, :] = h[:, start:end, :].mean(dim=1)
            gate_input = torch.cat([h, future_ctx], dim=-1)  # (B, T, 2H)

        gate_logits = self.gate_net(gate_input).squeeze(-1)  # (B, T)
        gate        = torch.sigmoid(gate_logits)

        topk        = min(self.memory_slots, T)
        _, top_idx  = torch.topk(gate, topk, dim=-1)
        idx_exp     = top_idx.unsqueeze(-1).expand(-1, -1, self.hidden_dim)
        memory      = torch.gather(h, 1, idx_exp)            # (B, topk, H)
        mask        = torch.ones(B, topk)

        query_h = h[:, -1, :]
        return self.read_head(query_h, memory, mask)


# ── Training / evaluation ─────────────────────────────────────────────────────
def train_model(k, steps):
    model = LookaheadModel(VOCAB_SIZE, HIDDEN_DIM, MEMORY_SLOTS, k)
    opt   = Adam(model.parameters(), lr=LR)
    for _ in range(steps):
        seq, target = make_assoc_batch(BATCH, SEQ_LEN, VOCAB_SIZE, NUM_PAIRS)
        loss = F.cross_entropy(model(seq), target)
        opt.zero_grad(); loss.backward(); opt.step()
    return model


def eval_model(model, steps=400):
    correct = 0
    for _ in range(steps):
        seq, target = make_assoc_batch(BATCH, SEQ_LEN, VOCAB_SIZE, NUM_PAIRS)
        with torch.no_grad():
            logits = model(seq)
        correct += (logits.argmax(-1) == target).sum().item()
    return correct / (steps * BATCH)


# ── Experiment class ───────────────────────────────────────────────────────────
class Exp172AsyncWriteLookahead(Experiment):
    experiment_id = "exp_17_2"
    hypothesis    = (
        "K-token lookahead async write gate outperforms same-time write by "
        ">3% at some K in {2, 4, 6}."
    )

    def run(self) -> ExperimentResult:
        config = dict(
            VOCAB_SIZE=VOCAB_SIZE, HIDDEN_DIM=HIDDEN_DIM, SEQ_LEN=SEQ_LEN,
            MEMORY_SLOTS=MEMORY_SLOTS, NUM_PAIRS=NUM_PAIRS,
            STEPS=STEPS, BATCH=BATCH, K_VALUES=K_VALUES,
        )

        accs = {}
        for k in K_VALUES:
            print(f"Training K={k} …")
            model   = train_model(k, STEPS)
            acc     = eval_model(model)
            accs[k] = acc
            print(f"  K={k} acc={acc:.4f}")

        acc_0    = accs[0]
        best_k   = max((k for k in K_VALUES if k > 0), key=lambda k: accs[k])
        best_acc = accs[best_k]
        best_gap = best_acc - acc_0

        metrics = {f"acc_K{k}": round(accs[k], 4) for k in K_VALUES}
        metrics.update(dict(best_k=best_k, best_gap=round(best_gap, 4)))

        if best_gap > 0.03:
            outcome = OUTCOME_SUPPORTED
            notes   = f"K={best_k} outperforms K=0 by {best_gap:.3f}"
        elif all(accs[k] <= acc_0 + 0.02 for k in K_VALUES if k > 0):
            outcome = OUTCOME_REFUTED
            notes   = f"All lookahead K within 0.02 of K=0 (best gap={best_gap:.3f})"
        else:
            outcome = OUTCOME_INCONCLUSIVE
            notes   = f"Best gap {best_gap:.3f} is between 0.01 and 0.03"

        return self.result(outcome, metrics, notes, config)


if __name__ == "__main__":
    Exp172AsyncWriteLookahead().execute()
