"""
exp_17_1_query_conditioned_write.py

Hypothesis: A write gate conditioned on predicted future query type outperforms
context-only gate by >5% on tasks with 4 different query types.
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
NUM_QUERY_TYPES = 4
MEMORY_SLOTS = 8
STEPS        = 400
BATCH        = 32
NUM_PAIRS    = 4


# ── Data generator ─────────────────────────────────────────────────────────────
def make_typed_batch(batch_size, seq_len, vocab_size, num_pairs, num_types):
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

        seq[b, seq_len - 4] = qt   # type marker (0-3)
        qi                  = torch.randint(0, num_pairs, (1,)).item()
        seq[b, seq_len - 3] = 2
        seq[b, seq_len - 2] = keys[qi]
        seq[b, seq_len - 1] = 0
        target[b]           = vals[qi]

    return seq, target, query_types


# ── Shared building blocks ─────────────────────────────────────────────────────
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


# ── Condition A: context-only write gate ──────────────────────────────────────
class ContextOnlyModel(nn.Module):
    def __init__(self, vocab_size, hidden_dim, memory_slots):
        super().__init__()
        self.encoder  = Encoder(vocab_size, hidden_dim)
        self.gate_net = nn.Linear(hidden_dim, 1)
        self.read_head = ReadHead(hidden_dim, vocab_size)
        self.hidden_dim   = hidden_dim
        self.memory_slots = memory_slots

    def forward(self, seq):
        B, T      = seq.shape
        h         = self.encoder(seq)                           # (B, T, H)
        gate_logits = self.gate_net(h).squeeze(-1)              # (B, T)
        gate      = torch.sigmoid(gate_logits)

        # select top-MEMORY_SLOTS tokens
        topk      = min(self.memory_slots, T)
        _, top_idx = torch.topk(gate, topk, dim=-1)            # (B, topk)
        idx_expanded = top_idx.unsqueeze(-1).expand(-1, -1, self.hidden_dim)
        memory    = torch.gather(h, 1, idx_expanded)            # (B, topk, H)
        mask      = torch.ones(B, topk)

        query_h   = h[:, -1, :]                                 # last token as query
        return self.read_head(query_h, memory, mask)


# ── Condition B: query-conditioned write gate ─────────────────────────────────
class QueryConditionedModel(nn.Module):
    def __init__(self, vocab_size, hidden_dim, memory_slots, num_types):
        super().__init__()
        self.encoder      = Encoder(vocab_size, hidden_dim)
        self.type_pred    = nn.Linear(hidden_dim, num_types)    # predicts query type
        self.type_proj    = nn.Linear(num_types, hidden_dim // 2)
        self.gate_net     = nn.Linear(hidden_dim + hidden_dim // 2, 1)
        self.read_head    = ReadHead(hidden_dim, vocab_size)
        self.hidden_dim   = hidden_dim
        self.memory_slots = memory_slots

    def forward(self, seq):
        B, T      = seq.shape
        h         = self.encoder(seq)                           # (B, T, H)
        mean_h    = h.mean(dim=1)                               # (B, H)
        type_dist = torch.softmax(self.type_pred(mean_h), dim=-1)  # (B, num_types)
        type_emb  = self.type_proj(type_dist)                   # (B, H/2)
        type_emb_expanded = type_emb.unsqueeze(1).expand(-1, T, -1)

        gate_input  = torch.cat([h, type_emb_expanded], dim=-1) # (B, T, H+H/2)
        gate_logits = self.gate_net(gate_input).squeeze(-1)     # (B, T)
        gate        = torch.sigmoid(gate_logits)

        topk        = min(self.memory_slots, T)
        _, top_idx  = torch.topk(gate, topk, dim=-1)
        idx_expanded = top_idx.unsqueeze(-1).expand(-1, -1, self.hidden_dim)
        memory      = torch.gather(h, 1, idx_expanded)          # (B, topk, H)
        mask        = torch.ones(B, topk)

        query_h     = h[:, -1, :]
        return self.read_head(query_h, memory, mask), type_dist


# ── Training helpers ───────────────────────────────────────────────────────────
def train_context_only(steps, batch, seq_len, vocab_size, num_pairs, num_types):
    model = ContextOnlyModel(vocab_size, HIDDEN_DIM, MEMORY_SLOTS)
    opt   = Adam(model.parameters(), lr=LR)
    for step in range(steps):
        seq, target, _ = make_typed_batch(batch, seq_len, vocab_size, num_pairs, num_types)
        logits = model(seq)
        loss   = F.cross_entropy(logits, target)
        opt.zero_grad(); loss.backward(); opt.step()
    return model


def eval_context_only(model, steps=500):
    correct = 0
    for _ in range(steps):
        seq, target, _ = make_typed_batch(BATCH, SEQ_LEN, VOCAB_SIZE, NUM_PAIRS, NUM_QUERY_TYPES)
        with torch.no_grad():
            logits = model(seq)
        correct += (logits.argmax(-1) == target).sum().item()
    return correct / (steps * BATCH)


def train_query_conditioned(steps, batch, seq_len, vocab_size, num_pairs, num_types):
    model = QueryConditionedModel(vocab_size, HIDDEN_DIM, MEMORY_SLOTS, num_types)
    opt   = Adam(model.parameters(), lr=LR)
    for step in range(steps):
        seq, target, query_types = make_typed_batch(batch, seq_len, vocab_size, num_pairs, num_types)
        logits, type_dist = model(seq)
        task_loss = F.cross_entropy(logits, target)
        type_loss = F.cross_entropy(type_dist, query_types)
        loss      = task_loss + 0.3 * type_loss
        opt.zero_grad(); loss.backward(); opt.step()
    return model


def eval_query_conditioned(model, steps=500):
    correct      = 0
    type_correct = 0
    for _ in range(steps):
        seq, target, query_types = make_typed_batch(BATCH, SEQ_LEN, VOCAB_SIZE, NUM_PAIRS, NUM_QUERY_TYPES)
        with torch.no_grad():
            logits, type_dist = model(seq)
        correct      += (logits.argmax(-1) == target).sum().item()
        type_correct += (type_dist.argmax(-1) == query_types).sum().item()
    n = steps * BATCH
    return correct / n, type_correct / n


# ── Experiment class ───────────────────────────────────────────────────────────
class Exp171QueryConditionedWrite(Experiment):
    experiment_id = "exp_17_1"
    hypothesis    = (
        "A write gate conditioned on predicted future query type outperforms "
        "context-only gate by >5% on tasks with 4 different query types."
    )

    def run(self) -> ExperimentResult:
        config = dict(
            VOCAB_SIZE=VOCAB_SIZE, HIDDEN_DIM=HIDDEN_DIM, SEQ_LEN=SEQ_LEN,
            NUM_QUERY_TYPES=NUM_QUERY_TYPES, MEMORY_SLOTS=MEMORY_SLOTS,
            STEPS=STEPS, BATCH=BATCH,
        )

        print("Training Condition A: context-only gate …")
        model_a = train_context_only(STEPS, BATCH, SEQ_LEN, VOCAB_SIZE, NUM_PAIRS, NUM_QUERY_TYPES)
        acc_a   = eval_context_only(model_a)
        print(f"  acc_A = {acc_a:.4f}")

        print("Training Condition B: query-conditioned gate …")
        model_b = train_query_conditioned(STEPS, BATCH, SEQ_LEN, VOCAB_SIZE, NUM_PAIRS, NUM_QUERY_TYPES)
        acc_b, qp_acc = eval_query_conditioned(model_b)
        print(f"  acc_B = {acc_b:.4f}  query_pred_acc = {qp_acc:.4f}")

        gap = acc_b - acc_a
        metrics = dict(acc_A=round(acc_a, 4), acc_B=round(acc_b, 4),
                       gap=round(gap, 4), query_pred_acc=round(qp_acc, 4))

        if acc_b > acc_a + 0.05 and qp_acc > 0.40:
            outcome = OUTCOME_SUPPORTED
            notes   = f"Query-conditioned gate better by {gap:.3f}; predictor acc {qp_acc:.3f}"
        elif acc_a >= acc_b - 0.02:
            outcome = OUTCOME_REFUTED
            notes   = f"Context-only gate matches or beats query-conditioned (gap={gap:.3f})"
        else:
            outcome = OUTCOME_INCONCLUSIVE
            notes   = f"Gap {gap:.3f} < 0.05 but predictor acc {qp_acc:.3f}"

        return self.result(outcome, metrics, notes, config)


if __name__ == "__main__":
    Exp171QueryConditionedWrite().execute()
