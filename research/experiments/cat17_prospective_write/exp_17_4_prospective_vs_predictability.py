"""
exp_17_4_prospective_vs_predictability.py

Hypothesis: Query-conditioned write gain scales linearly with query
predictability (Pearson r > 0.85).
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from experiments.base import (
    Experiment, ExperimentResult,
    OUTCOME_SUPPORTED, OUTCOME_REFUTED, OUTCOME_INCONCLUSIVE,
)

# ── Constants ──────────────────────────────────────────────────────────────────
DEVICE               = "cpu"
LR                   = 3e-4
VOCAB_SIZE           = 64
HIDDEN_DIM           = 64
SEQ_LEN              = 32
NUM_QUERY_TYPES      = 4
STEPS                = 400
BATCH                = 32
MEMORY_SLOTS         = 8
NUM_PAIRS            = 4
PREDICTABILITY_LEVELS = [0.0, 0.33, 0.66, 1.0]


# ── Data generator ─────────────────────────────────────────────────────────────
def make_predictability_batch(batch_size, seq_len, vocab_size, num_pairs,
                              num_types, predictability):
    """
    At predictability P:
      - with prob P: place a 'correct hint' token at pos 3 (matching query type qt)
      - with prob (1-P): place a 'random hint' token at pos 3 (random 0..3)
    """
    seq         = torch.zeros(batch_size, seq_len, dtype=torch.long)
    target      = torch.zeros(batch_size, dtype=torch.long)
    query_types = torch.randint(0, num_types, (batch_size,))

    for b in range(batch_size):
        qt        = query_types[b].item()
        key_start = 4 + qt * 10
        key_end   = key_start + 10
        val_start = vocab_size // 2
        val_end   = vocab_size

        keys = torch.randint(key_start, key_end, (num_pairs,))
        vals = torch.randint(val_start, val_end, (num_pairs,))

        pos = 0
        for i in range(num_pairs):
            if pos + 1 < seq_len - 4:
                seq[b, pos]     = keys[i]
                seq[b, pos + 1] = vals[i]
                pos += 2
        for p in range(pos, seq_len - 4):
            seq[b, p] = 3

        # hint token at position seq_len - 4
        if torch.rand(1).item() < predictability:
            hint = qt           # correct hint
        else:
            hint = torch.randint(0, num_types, (1,)).item()
        seq[b, seq_len - 4] = hint

        qi                  = torch.randint(0, num_pairs, (1,)).item()
        seq[b, seq_len - 3] = 2
        seq[b, seq_len - 2] = keys[qi]
        seq[b, seq_len - 1] = 0
        target[b]           = vals[qi]

    return seq, target, query_types


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


# ── Model A: context-only gate ─────────────────────────────────────────────────
class ContextOnlyModel(nn.Module):
    def __init__(self, vocab_size, hidden_dim, memory_slots):
        super().__init__()
        self.encoder      = Encoder(vocab_size, hidden_dim)
        self.gate_net     = nn.Linear(hidden_dim, 1)
        self.read_head    = ReadHead(hidden_dim, vocab_size)
        self.hidden_dim   = hidden_dim
        self.memory_slots = memory_slots

    def forward(self, seq):
        B, T = seq.shape
        h    = self.encoder(seq)
        gate = torch.sigmoid(self.gate_net(h).squeeze(-1))
        topk = min(self.memory_slots, T)
        _, top_idx = torch.topk(gate, topk, dim=-1)
        memory = torch.gather(h, 1, top_idx.unsqueeze(-1).expand(-1, -1, self.hidden_dim))
        mask   = torch.ones(B, topk)
        return self.read_head(h[:, -1, :], memory, mask)


# ── Model B: query-conditioned gate ───────────────────────────────────────────
class QueryConditionedModel(nn.Module):
    def __init__(self, vocab_size, hidden_dim, memory_slots, num_types):
        super().__init__()
        self.encoder      = Encoder(vocab_size, hidden_dim)
        self.type_pred    = nn.Linear(hidden_dim, num_types)
        self.type_proj    = nn.Linear(num_types, hidden_dim // 2)
        self.gate_net     = nn.Linear(hidden_dim + hidden_dim // 2, 1)
        self.read_head    = ReadHead(hidden_dim, vocab_size)
        self.hidden_dim   = hidden_dim
        self.memory_slots = memory_slots

    def forward(self, seq):
        B, T      = seq.shape
        h         = self.encoder(seq)
        mean_h    = h.mean(dim=1)
        type_dist = torch.softmax(self.type_pred(mean_h), dim=-1)
        type_emb  = self.type_proj(type_dist).unsqueeze(1).expand(-1, T, -1)
        gate_input = torch.cat([h, type_emb], dim=-1)
        gate       = torch.sigmoid(self.gate_net(gate_input).squeeze(-1))
        topk       = min(self.memory_slots, T)
        _, top_idx = torch.topk(gate, topk, dim=-1)
        memory     = torch.gather(h, 1, top_idx.unsqueeze(-1).expand(-1, -1, self.hidden_dim))
        mask       = torch.ones(B, topk)
        return self.read_head(h[:, -1, :], memory, mask), type_dist


# ── Pearson correlation ────────────────────────────────────────────────────────
def pearson_r(xs, ys):
    xs_t = torch.tensor(xs, dtype=torch.float32)
    ys_t = torch.tensor(ys, dtype=torch.float32)
    mx   = xs_t.mean(); my = ys_t.mean()
    cov  = ((xs_t - mx) * (ys_t - my)).mean()
    sx   = xs_t.std(unbiased=False); sy = ys_t.std(unbiased=False)
    if sx < 1e-8 or sy < 1e-8:
        return 0.0
    return (cov / (sx * sy)).item()


# ── Training / evaluation per predictability level ────────────────────────────
def train_context_only(pred_level, steps):
    model = ContextOnlyModel(VOCAB_SIZE, HIDDEN_DIM, MEMORY_SLOTS)
    opt   = Adam(model.parameters(), lr=LR)
    for _ in range(steps):
        seq, target, _ = make_predictability_batch(
            BATCH, SEQ_LEN, VOCAB_SIZE, NUM_PAIRS, NUM_QUERY_TYPES, pred_level)
        loss = F.cross_entropy(model(seq), target)
        opt.zero_grad(); loss.backward(); opt.step()
    return model


def train_query_conditioned(pred_level, steps):
    model = QueryConditionedModel(VOCAB_SIZE, HIDDEN_DIM, MEMORY_SLOTS, NUM_QUERY_TYPES)
    opt   = Adam(model.parameters(), lr=LR)
    for _ in range(steps):
        seq, target, qt = make_predictability_batch(
            BATCH, SEQ_LEN, VOCAB_SIZE, NUM_PAIRS, NUM_QUERY_TYPES, pred_level)
        logits, type_dist = model(seq)
        loss = F.cross_entropy(logits, target) + 0.3 * F.cross_entropy(type_dist, qt)
        opt.zero_grad(); loss.backward(); opt.step()
    return model


def eval_acc(model, pred_level, steps=300, context_only=True):
    correct = 0
    for _ in range(steps):
        seq, target, qt = make_predictability_batch(
            BATCH, SEQ_LEN, VOCAB_SIZE, NUM_PAIRS, NUM_QUERY_TYPES, pred_level)
        with torch.no_grad():
            if context_only:
                logits = model(seq)
            else:
                logits, _ = model(seq)
        correct += (logits.argmax(-1) == target).sum().item()
    return correct / (steps * BATCH)


# ── Experiment class ───────────────────────────────────────────────────────────
class Exp174ProspectiveVsPredictability(Experiment):
    experiment_id = "exp_17_4"
    hypothesis    = (
        "Query-conditioned write gain scales linearly with query predictability "
        "(Pearson r > 0.85)."
    )

    def run(self) -> ExperimentResult:
        config = dict(
            VOCAB_SIZE=VOCAB_SIZE, HIDDEN_DIM=HIDDEN_DIM, SEQ_LEN=SEQ_LEN,
            NUM_QUERY_TYPES=NUM_QUERY_TYPES, STEPS=STEPS, BATCH=BATCH,
            MEMORY_SLOTS=MEMORY_SLOTS, PREDICTABILITY_LEVELS=PREDICTABILITY_LEVELS,
        )

        gaps    = []
        metrics = {}

        for p in PREDICTABILITY_LEVELS:
            print(f"Predictability={p:.2f}: training A (context-only) …")
            model_a = train_context_only(p, STEPS)
            acc_a   = eval_acc(model_a, p, context_only=True)

            print(f"Predictability={p:.2f}: training B (query-conditioned) …")
            model_b = train_query_conditioned(p, STEPS)
            acc_b   = eval_acc(model_b, p, context_only=False)

            gap = acc_b - acc_a
            gaps.append(gap)
            pkey = f"p{int(p*100):03d}"
            metrics[f"acc_A_{pkey}"] = round(acc_a, 4)
            metrics[f"acc_B_{pkey}"] = round(acc_b, 4)
            metrics[f"gap_{pkey}"]   = round(gap, 4)
            print(f"  acc_A={acc_a:.4f}  acc_B={acc_b:.4f}  gap={gap:.4f}")

        r = pearson_r(PREDICTABILITY_LEVELS, gaps)
        gap_variance = torch.tensor(gaps).var(unbiased=False).item()
        metrics["pearson_r"]    = round(r, 4)
        metrics["gap_variance"] = round(gap_variance, 6)
        print(f"Pearson r = {r:.4f}  gap_variance = {gap_variance:.6f}")

        if r > 0.85:
            outcome = OUTCOME_SUPPORTED
            notes   = f"Pearson r={r:.3f} > 0.85 — linear scaling confirmed"
        elif gap_variance < 0.005:
            outcome = OUTCOME_REFUTED
            notes   = f"Gap is effectively constant (variance={gap_variance:.6f})"
        else:
            outcome = OUTCOME_INCONCLUSIVE
            notes   = f"r={r:.3f} between 0.5 and 0.85"

        return self.result(outcome, metrics, notes, config)


if __name__ == "__main__":
    Exp174ProspectiveVsPredictability().execute()
