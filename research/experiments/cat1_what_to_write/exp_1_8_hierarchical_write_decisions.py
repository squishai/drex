"""
Experiment 1.8 — Hierarchical Write Decisions

Hypothesis: A two-stage write decision (coarse filter then fine ranking)
outperforms a single-stage write gate.

Setup:
  - Controller A (single): Linear(HIDDEN_DIM,1) -> sigmoid -> top-k selection
  - Controller B (two-stage):
      stage1 = Linear(HIDDEN_DIM,1) -> sigmoid -> keep tokens with score > 0.5
      stage2 = Linear(HIDDEN_DIM,1) -> sigmoid -> top-k from kept tokens
      If stage1 keeps fewer than k tokens, take all kept tokens
  - Both use STE-style discrete selection
  - Same MEMORY_SLOTS=6 budget
  - Same associative recall task
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
MEMORY_SLOTS  = 6
NUM_PAIRS     = 5
LR            = 3e-4
STAGE1_THRESH = 0.5
DEVICE        = "cpu"

# ── Task ──────────────────────────────────────────────────────────────────────

def make_recall_batch(batch_size: int):
    seqs    = torch.randint(2, VOCAB_SIZE, (batch_size, SEQ_LEN))
    queries = torch.zeros(batch_size, dtype=torch.long)
    targets = torch.zeros(batch_size, dtype=torch.long)
    for b in range(batch_size):
        positions = torch.randperm(SEQ_LEN - 1)[:NUM_PAIRS * 2]
        pairs = []
        for i in range(NUM_PAIRS):
            k_pos = positions[i * 2].item()
            v_pos = positions[i * 2 + 1].item()
            key   = torch.randint(2, VOCAB_SIZE // 2, (1,)).item()
            val   = torch.randint(VOCAB_SIZE // 2, VOCAB_SIZE, (1,)).item()
            seqs[b, k_pos] = key
            seqs[b, v_pos] = val
            pairs.append((k_pos, v_pos))
        idx = torch.randint(NUM_PAIRS, (1,)).item()
        queries[b] = seqs[b, pairs[idx][0]]
        targets[b] = seqs[b, pairs[idx][1]]
    return seqs, queries, targets


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
        a, _ = self.attn(h, h, h, need_weights=False)
        h = self.norm1(h + a)
        h = self.norm2(h + self.ff(h))
        return h


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


# ── Single-gate selection (STE top-k) ────────────────────────────────────────

def topk_ste(scores: torch.Tensor, k: int) -> torch.Tensor:
    """
    Straight-through hard top-k selection.
    Returns mask (B, L) with exactly k True per row.
    Forward: hard 0/1. Backward: gradient flows through soft scores.
    """
    B, L    = scores.shape
    k       = min(k, L)
    topk_v, topk_i = scores.topk(k, dim=1)
    hard    = torch.zeros_like(scores)
    hard.scatter_(1, topk_i, 1.0)
    # STE: forward hard, backward through scores
    mask_f  = (hard - scores.detach() + scores)
    return mask_f


# ── Single-stage controller ───────────────────────────────────────────────────

class SingleStageController(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder     = TinyEncoder()
        self.gate        = nn.Linear(HIDDEN_DIM, 1)
        self.reader      = MemoryReader()
        self.query_embed = nn.Embedding(VOCAB_SIZE, HIDDEN_DIM)

    def _select(self, hidden):
        B, L, H = hidden.shape
        scores  = self.gate(hidden).squeeze(-1).sigmoid()   # (B, L)
        mask_f  = topk_ste(scores, MEMORY_SLOTS)            # (B, L) STE
        # hard indices for gathering
        topk    = scores.topk(MEMORY_SLOTS, dim=1).indices
        return topk, mask_f

    def forward(self, seq, query, target):
        hidden    = self.encoder(seq)
        topk, _   = self._select(hidden)
        B, L, H   = hidden.shape
        mem       = hidden.gather(1, topk.unsqueeze(-1).expand(-1, -1, H))
        q_h       = self.query_embed(query)
        logits    = self.reader(q_h, mem, mem)
        return F.cross_entropy(logits, target)


# ── Two-stage controller ──────────────────────────────────────────────────────

class TwoStageController(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder     = TinyEncoder()
        self.stage1_gate = nn.Linear(HIDDEN_DIM, 1)
        self.stage2_gate = nn.Linear(HIDDEN_DIM, 1)
        self.reader      = MemoryReader()
        self.query_embed = nn.Embedding(VOCAB_SIZE, HIDDEN_DIM)

    def _select(self, hidden):
        B, L, H  = hidden.shape
        s1       = self.stage1_gate(hidden).squeeze(-1).sigmoid()  # (B, L)
        # coarse filter: keep positions where s1 > threshold
        keep_mask = (s1 > STAGE1_THRESH)                           # (B, L) bool

        # Fine ranking on kept tokens
        s2       = self.stage2_gate(hidden).squeeze(-1).sigmoid()  # (B, L)
        # Zero out non-kept positions for ranking
        s2_filtered = s2 * keep_mask.float()

        # For each batch item, pick top-k from filtered; if kept < k, just use all kept
        topk_indices = []
        for b in range(B):
            n_kept = keep_mask[b].sum().item()
            k_eff  = min(int(n_kept), MEMORY_SLOTS) if n_kept > 0 else MEMORY_SLOTS
            if k_eff == 0:
                # fallback to stage1 scores
                ti = s1[b].topk(MEMORY_SLOTS).indices
            else:
                ti = s2_filtered[b].topk(MEMORY_SLOTS).indices
            topk_indices.append(ti)

        topk = torch.stack(topk_indices, dim=0)   # (B, MEMORY_SLOTS)
        return topk

    def forward(self, seq, query, target):
        hidden    = self.encoder(seq)
        topk      = self._select(hidden)
        B, L, H   = hidden.shape
        mem       = hidden.gather(1, topk.unsqueeze(-1).expand(-1, -1, H))
        q_h       = self.query_embed(query)
        logits    = self.reader(q_h, mem, mem)
        return F.cross_entropy(logits, target)


# ── Train / Eval ──────────────────────────────────────────────────────────────

def train_model(model: nn.Module, steps: int, name: str) -> None:
    opt = Adam(model.parameters(), lr=LR)
    model.train()
    for step in range(steps):
        seq, q, t = make_recall_batch(BATCH_SIZE)
        seq, q, t = seq.to(DEVICE), q.to(DEVICE), t.to(DEVICE)
        loss      = model(seq, q, t)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (step + 1) % 500 == 0:
            print(f"    [{name}] step {step+1}/{steps}  loss={loss.item():.4f}")


def eval_model(model: nn.Module, steps: int) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for _ in range(steps):
            seq, q, t = make_recall_batch(BATCH_SIZE)
            seq, q, t = seq.to(DEVICE), q.to(DEVICE), t.to(DEVICE)
            hidden    = model.encoder(seq)

            if isinstance(model, SingleStageController):
                topk, _ = model._select(hidden)
            else:
                topk = model._select(hidden)

            B, L, H = hidden.shape
            mem     = hidden.gather(1, topk.unsqueeze(-1).expand(-1, -1, H))
            q_h     = model.query_embed(q)
            logits  = model.reader(q_h, mem, mem)
            preds   = logits.argmax(-1)
            correct += (preds == t).sum().item()
            total   += t.shape[0]
    return correct / total


# ── Experiment ────────────────────────────────────────────────────────────────

class Exp18HierarchicalWriteDecisions(Experiment):
    experiment_id = "exp_1_8"
    hypothesis = (
        "A two-stage write decision (coarse filter then fine ranking) "
        "outperforms a single-stage write gate."
    )

    def run(self) -> ExperimentResult:
        torch.manual_seed(42)

        print("  Training single-stage controller ...")
        single_model = SingleStageController().to(DEVICE)
        train_model(single_model, TRAIN_STEPS, "single")
        single_acc = eval_model(single_model, EVAL_STEPS)
        print(f"    single_stage_acc={single_acc:.3f}")

        print("  Training two-stage controller ...")
        two_model = TwoStageController().to(DEVICE)
        train_model(two_model, TRAIN_STEPS, "two_stage")
        two_acc = eval_model(two_model, EVAL_STEPS)
        print(f"    two_stage_acc={two_acc:.3f}")

        gap = two_acc - single_acc

        if two_acc > single_acc:
            outcome = OUTCOME_SUPPORTED
        elif single_acc > two_acc + 0.01:
            outcome = OUTCOME_REFUTED
        else:
            outcome = OUTCOME_INCONCLUSIVE

        metrics = {
            "single_stage_acc": round(single_acc, 4),
            "two_stage_acc":    round(two_acc, 4),
            "gap_two_minus_single": round(gap, 4),
        }
        notes = (
            f"Two-stage acc={two_acc:.3f} vs single-stage acc={single_acc:.3f}. "
            f"Gap={gap:+.4f}. Stage1 threshold={STAGE1_THRESH}."
        )
        return self.result(outcome, metrics, notes, config={
            "vocab_size": VOCAB_SIZE, "hidden_dim": HIDDEN_DIM,
            "seq_len": SEQ_LEN, "batch_size": BATCH_SIZE,
            "train_steps": TRAIN_STEPS, "memory_slots": MEMORY_SLOTS,
            "num_pairs": NUM_PAIRS, "stage1_threshold": STAGE1_THRESH,
        })


if __name__ == "__main__":
    Exp18HierarchicalWriteDecisions().execute()
