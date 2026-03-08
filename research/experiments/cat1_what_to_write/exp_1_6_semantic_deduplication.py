"""
Experiment 1.6 — Semantic Deduplication

Hypothesis: Cosine-similarity deduplication at write time improves retrieval
precision without dangerous information loss.

Setup:
  - Memory with 8 slots
  - Policy A (standard): gate selects top-8 tokens
  - Policy B (dedup): before adding a token, check max cosine similarity to
    existing entries; only write if max_sim < 0.7; if full, evict LRW entry
  - Diverse planted pairs (each pair has a unique key type) to test coverage
  - Metrics: precision (% queries correctly answered), coverage
    (% of distinct key types with at least one relevant memory entry)
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
SEQ_LEN       = 24
BATCH_SIZE    = 32
TRAIN_STEPS   = 1500
EVAL_STEPS    = 300
MEMORY_SLOTS  = 8
NUM_PAIRS     = 6     # distinct key types per sequence
LR            = 3e-4
DEDUP_THRESH  = 0.7   # cosine similarity threshold
DEVICE        = "cpu"

# ── Task with diverse key types ────────────────────────────────────────────────

# We allocate NUM_PAIRS distinct key "types" each from a unique sub-range.
# KEY_RANGES[i] = (lo, hi) for key type i
KEY_STRIDE  = (VOCAB_SIZE // 2) // NUM_PAIRS
KEY_RANGES  = [(1 + i * KEY_STRIDE, 1 + (i + 1) * KEY_STRIDE) for i in range(NUM_PAIRS)]
VAL_LO      = VOCAB_SIZE // 2
VAL_HI      = VOCAB_SIZE


def make_diverse_batch(batch_size: int):
    """
    Each sequence has NUM_PAIRS planted key->value pairs, one per key type.
    Returns (seqs, queries, targets, key_types).
    key_types: (B,) int — which key range was queried.
    """
    seqs      = torch.randint(VAL_LO, VOCAB_SIZE, (batch_size, SEQ_LEN))
    queries   = torch.zeros(batch_size, dtype=torch.long)
    targets   = torch.zeros(batch_size, dtype=torch.long)
    key_types = torch.zeros(batch_size, dtype=torch.long)

    for b in range(batch_size):
        positions = torch.randperm(SEQ_LEN - 1)[:NUM_PAIRS * 2]
        pairs = []
        for i in range(NUM_PAIRS):
            k_pos   = positions[i * 2].item()
            v_pos   = positions[i * 2 + 1].item()
            lo, hi  = KEY_RANGES[i]
            hi      = min(hi, VOCAB_SIZE // 2)
            if lo >= hi:
                hi = lo + 1
            key     = torch.randint(lo, hi, (1,)).item()
            val     = torch.randint(VAL_LO, VAL_HI, (1,)).item()
            seqs[b, k_pos] = key
            seqs[b, v_pos] = val
            pairs.append((k_pos, v_pos, i))

        idx = torch.randint(NUM_PAIRS, (1,)).item()
        queries[b]   = seqs[b, pairs[idx][0]]
        targets[b]   = seqs[b, pairs[idx][1]]
        key_types[b] = pairs[idx][2]

    return seqs, queries, targets, key_types


# ── Encoder ───────────────────────────────────────────────────────────────────

class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, HIDDEN_DIM)
        self.attn  = nn.MultiheadAttention(HIDDEN_DIM, 2, batch_first=True)
        self.ff    = nn.Sequential(
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM * 2), nn.ReLU(),
            nn.Linear(HIDDEN_DIM * 2, HIDDEN_DIM),
        )
        self.norm1 = nn.LayerNorm(HIDDEN_DIM)
        self.norm2 = nn.LayerNorm(HIDDEN_DIM)

    def forward(self, x):
        h = self.embed(x)
        a, attn_w = self.attn(h, h, h, need_weights=True)
        h = self.norm1(h + a)
        h = self.norm2(h + self.ff(h))
        return h, attn_w


class WriteGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = nn.Linear(HIDDEN_DIM, 1)

    def scores(self, hidden):
        return self.gate(hidden).squeeze(-1).sigmoid()


class MemoryReader(nn.Module):
    def __init__(self):
        super().__init__()
        self.query_proj = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.out_proj   = nn.Linear(HIDDEN_DIM, VOCAB_SIZE)

    def forward(self, q_h, mem_k, mem_v):
        q       = self.query_proj(q_h).unsqueeze(1)
        scores  = (q * mem_k).sum(-1) / (HIDDEN_DIM ** 0.5)
        weights = F.softmax(scores, dim=-1).unsqueeze(-1)
        return self.out_proj((weights * mem_v).sum(1))


# ── Dedup selection (per-sample, sequential) ──────────────────────────────────

def dedup_select(hidden_b: torch.Tensor, scores_b: torch.Tensor,
                 threshold: float = DEDUP_THRESH) -> torch.Tensor:
    """
    hidden_b: (L, H)  hidden states for one batch item
    scores_b: (L,)    gate scores
    Returns indices of selected slots (up to MEMORY_SLOTS), LRW eviction.
    """
    L, H = hidden_b.shape
    order = scores_b.argsort(descending=True)
    slots: list[int] = []      # selected position indices
    slot_vecs: list[torch.Tensor] = []   # their normalised vectors
    write_order: list[int] = []  # for LRW eviction tracking

    for idx in order:
        idx_i = idx.item()
        vec   = F.normalize(hidden_b[idx_i].unsqueeze(0), dim=-1)  # (1, H)

        if slot_vecs:
            stacked = torch.cat(slot_vecs, dim=0)                  # (M, H)
            sims    = (vec * stacked).sum(-1)                      # (M,)
            max_sim = sims.max().item()
        else:
            max_sim = 0.0

        if max_sim < threshold:
            if len(slots) < MEMORY_SLOTS:
                slots.append(idx_i)
                slot_vecs.append(vec)
                write_order.append(idx_i)
            else:
                # evict least-recently-written (first in write_order)
                evict_pos   = write_order.pop(0)
                evict_slot  = slots.index(evict_pos)
                slots[evict_slot]      = idx_i
                slot_vecs[evict_slot]  = vec
                write_order.append(idx_i)

        if len(slots) == MEMORY_SLOTS and max_sim >= threshold:
            # once full and all remaining are duplicates, stop
            pass

    # pad if fewer than MEMORY_SLOTS selected
    while len(slots) < MEMORY_SLOTS:
        slots.append(slots[-1] if slots else 0)

    return torch.tensor(slots[:MEMORY_SLOTS], dtype=torch.long, device=hidden_b.device)


def batch_dedup_select(hidden: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
    """Apply dedup_select per batch item. Returns (B, MEMORY_SLOTS)."""
    B = hidden.shape[0]
    indices = []
    for b in range(B):
        indices.append(dedup_select(hidden[b], scores[b]))
    return torch.stack(indices, dim=0)


# ── Full Models ───────────────────────────────────────────────────────────────

class StandardController(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder     = TinyEncoder()
        self.gate        = WriteGate()
        self.reader      = MemoryReader()
        self.query_embed = nn.Embedding(VOCAB_SIZE, HIDDEN_DIM)

    def _select(self, hidden, scores):
        return scores.topk(MEMORY_SLOTS, dim=1).indices

    def forward(self, seq, query, target):
        hidden, _  = self.encoder(seq)
        scores     = self.gate.scores(hidden)
        topk       = self._select(hidden, scores)
        B, L, H    = hidden.shape
        mem        = hidden.gather(1, topk.unsqueeze(-1).expand(-1, -1, H))
        q_h        = self.query_embed(query)
        logits     = self.reader(q_h, mem, mem)
        return F.cross_entropy(logits, target)


class DedupController(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder     = TinyEncoder()
        self.gate        = WriteGate()
        self.reader      = MemoryReader()
        self.query_embed = nn.Embedding(VOCAB_SIZE, HIDDEN_DIM)

    def _select(self, hidden, scores):
        return batch_dedup_select(hidden.detach(), scores.detach())

    def forward(self, seq, query, target):
        hidden, _  = self.encoder(seq)
        scores     = self.gate.scores(hidden)
        topk       = self._select(hidden, scores)
        B, L, H    = hidden.shape
        mem        = hidden.gather(1, topk.unsqueeze(-1).expand(-1, -1, H))
        q_h        = self.query_embed(query)
        logits     = self.reader(q_h, mem, mem)
        return F.cross_entropy(logits, target)


# ── Train / Eval ──────────────────────────────────────────────────────────────

def train_model(model: nn.Module, steps: int) -> None:
    opt = Adam(model.parameters(), lr=LR)
    model.train()
    for step in range(steps):
        seq, q, t, _ = make_diverse_batch(BATCH_SIZE)
        seq, q, t    = seq.to(DEVICE), q.to(DEVICE), t.to(DEVICE)
        loss         = model(seq, q, t)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (step + 1) % 500 == 0:
            print(f"    step {step+1}/{steps}  loss={loss.item():.4f}")


def eval_model(model: nn.Module, steps: int):
    """Returns (precision, coverage)."""
    model.eval()
    correct  = 0
    total    = 0
    # track coverage: for each key type, did we get at least one correct answer?
    type_correct = [0] * NUM_PAIRS
    type_total   = [0] * NUM_PAIRS

    with torch.no_grad():
        for _ in range(steps):
            seq, q, t, kt = make_diverse_batch(BATCH_SIZE)
            seq, q, t     = seq.to(DEVICE), q.to(DEVICE), t.to(DEVICE)
            hidden, _     = model.encoder(seq)
            scores        = model.gate.scores(hidden)
            topk          = model._select(hidden, scores)
            B, L, H       = hidden.shape
            mem           = hidden.gather(1, topk.unsqueeze(-1).expand(-1, -1, H))
            q_h           = model.query_embed(q)
            logits        = model.reader(q_h, mem, mem)
            preds         = logits.argmax(-1)

            for b in range(BATCH_SIZE):
                is_correct = (preds[b] == t[b]).item()
                correct   += int(is_correct)
                total     += 1
                ktype      = kt[b].item()
                type_correct[ktype] += int(is_correct)
                type_total[ktype]   += 1

    precision = correct / total
    coverage  = sum(
        1 for i in range(NUM_PAIRS) if type_total[i] > 0 and type_correct[i] > 0
    ) / NUM_PAIRS
    return precision, coverage


# ── Experiment ────────────────────────────────────────────────────────────────

class Exp16SemanticDeduplication(Experiment):
    experiment_id = "exp_1_6"
    hypothesis = (
        "Cosine-similarity deduplication at write time improves retrieval "
        "precision without dangerous information loss."
    )

    def run(self) -> ExperimentResult:
        torch.manual_seed(42)

        print("  Training standard controller ...")
        std_model = StandardController().to(DEVICE)
        train_model(std_model, TRAIN_STEPS)
        std_prec, std_cov = eval_model(std_model, EVAL_STEPS)
        print(f"    precision={std_prec:.3f}  coverage={std_cov:.3f}")

        print("  Training dedup controller ...")
        dup_model = DedupController().to(DEVICE)
        train_model(dup_model, TRAIN_STEPS)
        dup_prec, dup_cov = eval_model(dup_model, EVAL_STEPS)
        print(f"    precision={dup_prec:.3f}  coverage={dup_cov:.3f}")

        if dup_prec >= std_prec and dup_cov >= 0.70:
            outcome = OUTCOME_SUPPORTED
        elif dup_cov < 0.50:
            outcome = OUTCOME_REFUTED
        else:
            outcome = OUTCOME_INCONCLUSIVE

        metrics = {
            "standard_precision": round(std_prec, 4),
            "standard_coverage":  round(std_cov, 4),
            "dedup_precision":    round(dup_prec, 4),
            "dedup_coverage":     round(dup_cov, 4),
            "precision_delta":    round(dup_prec - std_prec, 4),
            "coverage_delta":     round(dup_cov - std_cov, 4),
        }
        notes = (
            f"Dedup: precision={dup_prec:.3f} (std={std_prec:.3f}), "
            f"coverage={dup_cov:.3f} (std={std_cov:.3f}). "
            f"Threshold={DEDUP_THRESH}."
        )
        return self.result(outcome, metrics, notes, config={
            "vocab_size": VOCAB_SIZE, "hidden_dim": HIDDEN_DIM,
            "seq_len": SEQ_LEN, "batch_size": BATCH_SIZE,
            "train_steps": TRAIN_STEPS, "memory_slots": MEMORY_SLOTS,
            "num_pairs": NUM_PAIRS, "dedup_threshold": DEDUP_THRESH,
        })


if __name__ == "__main__":
    Exp16SemanticDeduplication().execute()
