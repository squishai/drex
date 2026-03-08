"""
Experiment 1.1 — Relevance Signal Baseline

Hypothesis: Attention weight correlates positively with memory importance and
attention-based memory outperforms random memory on retrieval tasks.

Setup:
  - Tiny 2-layer transformer with MultiheadAttention
  - Tokens 0-15 are "important" (numeric), 16-63 are "noise"
  - Train on next-token prediction
  - Measure Pearson correlation between avg-attention-received and is_important flag
  - Compare attention-selected, random-selected, and oracle-important memory on
    associative recall
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
LR            = 3e-4
DEVICE        = "cpu"
IMPORTANT_MAX = 16   # tokens 0..15 are "important"

# ── Task helpers ──────────────────────────────────────────────────────────────

def make_lm_batch(batch_size: int) -> torch.Tensor:
    """Random token sequences with ~50% important tokens."""
    seqs = torch.zeros(batch_size, SEQ_LEN, dtype=torch.long)
    for b in range(batch_size):
        for t in range(SEQ_LEN):
            if torch.rand(1).item() < 0.5:
                seqs[b, t] = torch.randint(1, IMPORTANT_MAX, (1,)).item()
            else:
                seqs[b, t] = torch.randint(IMPORTANT_MAX, VOCAB_SIZE, (1,)).item()
    return seqs


def make_recall_batch(batch_size: int, num_pairs: int = 4):
    """
    Associative recall: plant key->value pairs.
    Keys from important range (1..15), values from noise range (IMPORTANT_MAX..VOCAB_SIZE-1).
    Returns (seqs, queries, targets).
    """
    seqs    = torch.randint(IMPORTANT_MAX, VOCAB_SIZE, (batch_size, SEQ_LEN))
    queries = torch.zeros(batch_size, dtype=torch.long)
    targets = torch.zeros(batch_size, dtype=torch.long)
    kv_positions = torch.zeros(batch_size, num_pairs, 2, dtype=torch.long)

    for b in range(batch_size):
        positions = torch.randperm(SEQ_LEN)[:num_pairs * 2]
        for i in range(num_pairs):
            k_pos = positions[i * 2].item()
            v_pos = positions[i * 2 + 1].item()
            key   = torch.randint(1, IMPORTANT_MAX, (1,)).item()
            val   = torch.randint(IMPORTANT_MAX, VOCAB_SIZE, (1,)).item()
            seqs[b, k_pos] = key
            seqs[b, v_pos] = val
            kv_positions[b, i, 0] = k_pos
            kv_positions[b, i, 1] = v_pos

        idx = torch.randint(num_pairs, (1,)).item()
        queries[b] = seqs[b, kv_positions[b, idx, 0].item()].item()
        targets[b] = seqs[b, kv_positions[b, idx, 1].item()].item()

    return seqs, queries, targets, kv_positions


# ── Model ─────────────────────────────────────────────────────────────────────

class TinyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed  = nn.Embedding(VOCAB_SIZE, HIDDEN_DIM)
        self.attn1  = nn.MultiheadAttention(HIDDEN_DIM, 2, batch_first=True)
        self.attn2  = nn.MultiheadAttention(HIDDEN_DIM, 2, batch_first=True)
        self.ff1    = nn.Sequential(nn.Linear(HIDDEN_DIM, HIDDEN_DIM * 2), nn.ReLU(),
                                    nn.Linear(HIDDEN_DIM * 2, HIDDEN_DIM))
        self.ff2    = nn.Sequential(nn.Linear(HIDDEN_DIM, HIDDEN_DIM * 2), nn.ReLU(),
                                    nn.Linear(HIDDEN_DIM * 2, HIDDEN_DIM))
        self.norm1a = nn.LayerNorm(HIDDEN_DIM)
        self.norm1b = nn.LayerNorm(HIDDEN_DIM)
        self.norm2a = nn.LayerNorm(HIDDEN_DIM)
        self.norm2b = nn.LayerNorm(HIDDEN_DIM)
        self.lm_head = nn.Linear(HIDDEN_DIM, VOCAB_SIZE)

    def forward(self, x: torch.Tensor):
        """Returns (hidden: B,L,H), (attn_w_layer2: B,L,L)."""
        h = self.embed(x)
        a1, _    = self.attn1(h, h, h, need_weights=False)
        h = self.norm1a(h + a1)
        h = self.norm1b(h + self.ff1(h))
        a2, attn_w = self.attn2(h, h, h, need_weights=True)
        h = self.norm2a(h + a2)
        h = self.norm2b(h + self.ff2(h))
        return h, attn_w


class MemoryReader(nn.Module):
    def __init__(self):
        super().__init__()
        self.query_proj = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.out_proj   = nn.Linear(HIDDEN_DIM, VOCAB_SIZE)

    def forward(self, q_h: torch.Tensor, mem_k: torch.Tensor, mem_v: torch.Tensor):
        q   = self.query_proj(q_h).unsqueeze(1)
        scores  = (q * mem_k).sum(-1) / (HIDDEN_DIM ** 0.5)
        weights = F.softmax(scores, dim=-1).unsqueeze(-1)
        return self.out_proj((weights * mem_v).sum(1))


class RecallModel(nn.Module):
    def __init__(self, write_mode: str = "attention"):
        """write_mode: 'attention', 'random', 'oracle'."""
        super().__init__()
        self.encoder      = TinyTransformer()
        self.reader       = MemoryReader()
        self.query_embed  = nn.Embedding(VOCAB_SIZE, HIDDEN_DIM)
        self.write_mode   = write_mode

    def _select_slots(self, hidden, attn_w, oracle_mask=None):
        B, L, H = hidden.shape
        if self.write_mode == "attention":
            importance = attn_w.mean(dim=1)
            topk = importance.topk(MEMORY_SLOTS, dim=1).indices
        elif self.write_mode == "random":
            topk = torch.stack(
                [torch.randperm(L)[:MEMORY_SLOTS] for _ in range(B)]
            ).to(hidden.device)
        else:  # oracle
            # prefer positions containing important tokens, else random
            if oracle_mask is not None:
                scores = oracle_mask.float()
            else:
                scores = torch.ones(B, L, device=hidden.device)
            topk = scores.topk(MEMORY_SLOTS, dim=1).indices
        return topk

    def forward(self, seq, query, target, oracle_mask=None):
        hidden, attn_w = self.encoder(seq)
        topk = self._select_slots(hidden, attn_w, oracle_mask)
        B, L, H = hidden.shape
        mem = hidden.gather(1, topk.unsqueeze(-1).expand(-1, -1, H))
        q_h = self.query_embed(query)
        logits = self.reader(q_h, mem, mem)
        loss = F.cross_entropy(logits, target)
        return loss, attn_w


# ── Training ──────────────────────────────────────────────────────────────────

def train_recall(model: RecallModel, steps: int) -> None:
    opt = Adam(model.parameters(), lr=LR)
    model.train()
    for _ in range(steps):
        seq, q, t, kv = make_recall_batch(BATCH_SIZE)
        # oracle mask: positions that hold important (key) tokens
        oracle_mask = (seq < IMPORTANT_MAX).float()
        loss, _ = model(seq.to(DEVICE), q.to(DEVICE), t.to(DEVICE),
                        oracle_mask.to(DEVICE))
        opt.zero_grad()
        loss.backward()
        opt.step()


def eval_recall(model: RecallModel, steps: int) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for _ in range(steps):
            seq, q, t, kv = make_recall_batch(BATCH_SIZE)
            oracle_mask = (seq < IMPORTANT_MAX).float()
            hidden, attn_w = model.encoder(seq.to(DEVICE))
            topk = model._select_slots(hidden, attn_w, oracle_mask.to(DEVICE))
            B, L, H = hidden.shape
            mem = hidden.gather(1, topk.unsqueeze(-1).expand(-1, -1, H))
            q_h = model.query_embed(q.to(DEVICE))
            logits = model.reader(q_h, mem, mem)
            preds = logits.argmax(-1)
            correct += (preds == t.to(DEVICE)).sum().item()
            total   += t.shape[0]
    return correct / total


def measure_attention_correlation(model: RecallModel, steps: int = 100) -> float:
    """Pearson r between mean-attention-received and is_important per token position."""
    model.eval()
    all_attn = []   # (B*steps, L)
    all_flag = []   # (B*steps, L)  1 if token < IMPORTANT_MAX
    with torch.no_grad():
        for _ in range(steps):
            seq = make_lm_batch(BATCH_SIZE).to(DEVICE)
            _, attn_w = model.encoder(seq)
            importance = attn_w.mean(dim=1)  # (B, L)
            flag = (seq < IMPORTANT_MAX).float()
            all_attn.append(importance)
            all_flag.append(flag)
    attn_flat = torch.cat(all_attn, dim=0).view(-1).cpu()
    flag_flat = torch.cat(all_flag, dim=0).view(-1).cpu()
    # Pearson correlation
    attn_m = attn_flat - attn_flat.mean()
    flag_m = flag_flat - flag_flat.mean()
    denom  = (attn_m.norm() * flag_m.norm()).clamp(min=1e-8)
    r      = (attn_m * flag_m).sum() / denom
    return r.item()


# ── Experiment ────────────────────────────────────────────────────────────────

class Exp11RelevanceSignalBaseline(Experiment):
    experiment_id = "exp_1_1"
    hypothesis = (
        "Attention weight correlates positively with memory importance and "
        "attention-based memory outperforms random memory on retrieval tasks."
    )

    def run(self) -> ExperimentResult:
        torch.manual_seed(42)

        results = {}
        for mode in ("attention", "random", "oracle"):
            print(f"  Training write_mode={mode} ...")
            m = RecallModel(write_mode=mode).to(DEVICE)
            train_recall(m, TRAIN_STEPS)
            acc = eval_recall(m, EVAL_STEPS)
            results[mode] = acc
            print(f"    acc={acc:.3f}")

        # measure correlation using the attention model
        attn_model = RecallModel(write_mode="attention").to(DEVICE)
        train_recall(attn_model, TRAIN_STEPS)
        corr = measure_attention_correlation(attn_model, steps=100)
        print(f"  Attention-importance Pearson r = {corr:.4f}")

        attention_acc = results["attention"]
        random_acc    = results["random"]
        oracle_acc    = results["oracle"]

        if corr > 0.05 and attention_acc > random_acc:
            outcome = OUTCOME_SUPPORTED
        elif corr <= 0.0:
            outcome = OUTCOME_REFUTED
        else:
            outcome = OUTCOME_INCONCLUSIVE

        metrics = {
            "attention_correlation": round(corr, 4),
            "attention_memory_acc":  round(attention_acc, 4),
            "random_memory_acc":     round(random_acc, 4),
            "oracle_memory_acc":     round(oracle_acc, 4),
        }
        notes = (
            f"Pearson r={corr:.4f}. "
            f"Attention acc={attention_acc:.3f} vs random={random_acc:.3f} "
            f"vs oracle={oracle_acc:.3f}."
        )
        return self.result(outcome, metrics, notes, config={
            "vocab_size": VOCAB_SIZE, "hidden_dim": HIDDEN_DIM,
            "seq_len": SEQ_LEN, "batch_size": BATCH_SIZE,
            "train_steps": TRAIN_STEPS, "memory_slots": MEMORY_SLOTS,
            "important_max": IMPORTANT_MAX,
        })


if __name__ == "__main__":
    Exp11RelevanceSignalBaseline().execute()
