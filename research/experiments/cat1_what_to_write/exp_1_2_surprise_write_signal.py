"""
Experiment 1.2 — Surprise Write Signal

Hypothesis: A memory built from high-surprise (high-perplexity) tokens supports
better retrieval than attention-based memory.

Setup:
  - Tiny LM trained on associative recall sequences
  - Per-token cross-entropy loss (surprise) computed during forward pass
  - Memory A: top-8 highest-loss tokens
  - Memory B: top-8 highest-attention tokens
  - Compare both vs no-memory baseline on associative recall accuracy
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
NUM_PAIRS     = 6
LR            = 3e-4
DEVICE        = "cpu"

# ── Task ──────────────────────────────────────────────────────────────────────

def make_recall_batch(batch_size: int):
    """
    Sequences with NUM_PAIRS planted key->value pairs.
    Keys in [2, VOCAB_SIZE//2), values in [VOCAB_SIZE//2, VOCAB_SIZE).
    Returns (seqs, queries, targets).
    """
    seqs    = torch.randint(2, VOCAB_SIZE, (batch_size, SEQ_LEN))
    queries = torch.zeros(batch_size, dtype=torch.long)
    targets = torch.zeros(batch_size, dtype=torch.long)
    kv_pos  = []

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
        kv_pos.append(pairs)

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
        self.lm_head = nn.Linear(HIDDEN_DIM, VOCAB_SIZE)

    def forward(self, x: torch.Tensor):
        h = self.embed(x)
        a, attn_w = self.attn(h, h, h, need_weights=True)
        h = self.norm1(h + a)
        h = self.norm2(h + self.ff(h))
        return h, attn_w

    def lm_loss_per_token(self, x: torch.Tensor, hidden: torch.Tensor):
        """Per-token cross-entropy loss (surprise), padded to seq length."""
        logits = self.lm_head(hidden)
        shift_logits  = logits[:, :-1, :].contiguous()
        shift_targets = x[:, 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, VOCAB_SIZE),
            shift_targets.view(-1),
            reduction="none",
        ).view(shift_logits.shape[0], -1)
        pad = torch.zeros(loss.shape[0], 1, device=loss.device)
        return torch.cat([loss, pad], dim=1)


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


# ── Full model ────────────────────────────────────────────────────────────────

class RecallModel(nn.Module):
    def __init__(self, write_mode: str = "surprise"):
        """write_mode: 'surprise', 'attention', 'none'."""
        super().__init__()
        self.encoder     = TinyEncoder()
        self.reader      = MemoryReader()
        self.query_embed = nn.Embedding(VOCAB_SIZE, HIDDEN_DIM)
        self.no_mem_head = nn.Linear(HIDDEN_DIM, VOCAB_SIZE)
        self.write_mode  = write_mode

    def _select_memory(self, hidden, attn_w, loss_per_token):
        B, L, H = hidden.shape
        if self.write_mode == "surprise":
            scores = loss_per_token
        elif self.write_mode == "attention":
            scores = attn_w.mean(dim=1)
        else:
            return None
        topk = scores.topk(MEMORY_SLOTS, dim=1).indices
        return hidden.gather(1, topk.unsqueeze(-1).expand(-1, -1, H))

    def forward(self, seq, query, target):
        hidden, attn_w    = self.encoder(seq)
        loss_per_token    = self.encoder.lm_loss_per_token(seq, hidden)

        if self.write_mode == "none":
            # no memory: predict from query embedding alone
            q_h    = self.query_embed(query)
            logits = self.no_mem_head(q_h)
        else:
            mem    = self._select_memory(hidden, attn_w, loss_per_token)
            q_h    = self.query_embed(query)
            logits = self.reader(q_h, mem, mem)

        task_loss = F.cross_entropy(logits, target)
        return task_loss


# ── Train / Eval ──────────────────────────────────────────────────────────────

def train_model(model: RecallModel, steps: int) -> None:
    opt = Adam(model.parameters(), lr=LR)
    model.train()
    for _ in range(steps):
        seq, q, t = make_recall_batch(BATCH_SIZE)
        loss = model(seq.to(DEVICE), q.to(DEVICE), t.to(DEVICE))
        opt.zero_grad()
        loss.backward()
        opt.step()


def eval_model(model: RecallModel, steps: int) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for _ in range(steps):
            seq, q, t = make_recall_batch(BATCH_SIZE)
            hidden, attn_w = model.encoder(seq.to(DEVICE))
            loss_per_token = model.encoder.lm_loss_per_token(seq.to(DEVICE), hidden)

            if model.write_mode == "none":
                q_h    = model.query_embed(q.to(DEVICE))
                logits = model.no_mem_head(q_h)
            else:
                mem    = model._select_memory(hidden, attn_w, loss_per_token)
                q_h    = model.query_embed(q.to(DEVICE))
                logits = model.reader(q_h, mem, mem)

            preds = logits.argmax(-1)
            correct += (preds == t.to(DEVICE)).sum().item()
            total   += t.shape[0]
    return correct / total


# ── Experiment ────────────────────────────────────────────────────────────────

class Exp12SurpriseWriteSignal(Experiment):
    experiment_id = "exp_1_2"
    hypothesis = (
        "A memory built from high-surprise (high-perplexity) tokens supports "
        "better retrieval than attention-based memory."
    )

    def run(self) -> ExperimentResult:
        torch.manual_seed(42)

        accs = {}
        for mode in ("surprise", "attention", "none"):
            print(f"  Training write_mode={mode} ...")
            m = RecallModel(write_mode=mode).to(DEVICE)
            train_model(m, TRAIN_STEPS)
            acc = eval_model(m, EVAL_STEPS)
            accs[mode] = acc
            print(f"    acc={acc:.3f}")

        surprise_acc  = accs["surprise"]
        attention_acc = accs["attention"]
        baseline_acc  = accs["none"]
        gap = surprise_acc - attention_acc

        if gap > 0.02:
            outcome = OUTCOME_SUPPORTED
        elif surprise_acc < attention_acc:
            outcome = OUTCOME_REFUTED
        else:
            outcome = OUTCOME_INCONCLUSIVE

        metrics = {
            "surprise_acc":  round(surprise_acc, 4),
            "attention_acc": round(attention_acc, 4),
            "baseline_acc":  round(baseline_acc, 4),
            "gap_surprise_minus_attention": round(gap, 4),
        }
        notes = (
            f"Surprise acc={surprise_acc:.3f}, attention acc={attention_acc:.3f}, "
            f"no-memory baseline={baseline_acc:.3f}. Gap={gap:+.4f}."
        )
        return self.result(outcome, metrics, notes, config={
            "vocab_size": VOCAB_SIZE, "hidden_dim": HIDDEN_DIM,
            "seq_len": SEQ_LEN, "batch_size": BATCH_SIZE,
            "train_steps": TRAIN_STEPS, "memory_slots": MEMORY_SLOTS,
            "num_pairs": NUM_PAIRS,
        })


if __name__ == "__main__":
    Exp12SurpriseWriteSignal().execute()
