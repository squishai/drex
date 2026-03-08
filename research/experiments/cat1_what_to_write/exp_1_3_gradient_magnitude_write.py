"""
Experiment 1.3 — Gradient Magnitude Write Signal

Hypothesis: Storing tokens where gradient magnitude is highest produces memories
that generalize better than attention-selected memories.

Setup:
  - Train tiny model on associative recall
  - Gradient selection: forward+backward pass; read embedding.weight.grad;
    select top-8 token positions with highest |embed.weight.grad[token_id]| norm
  - Compare: (A) gradient-selected, (B) attention-selected, (C) random memory
  - Evaluate on associative recall accuracy
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


# ── Model components ──────────────────────────────────────────────────────────

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

    def forward(self, x):
        h = self.embed(x)
        a, attn_w = self.attn(h, h, h, need_weights=True)
        h = self.norm1(h + a)
        h = self.norm2(h + self.ff(h))
        return h, attn_w


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


# ── Write policies ────────────────────────────────────────────────────────────

def select_attention(hidden, attn_w):
    importance = attn_w.mean(dim=1)
    return importance.topk(MEMORY_SLOTS, dim=1).indices


def select_random(hidden, attn_w):
    B, L, H = hidden.shape
    return torch.stack([torch.randperm(L)[:MEMORY_SLOTS] for _ in range(B)]).to(hidden.device)


def select_gradient(hidden, seq, encoder):
    """
    Compute per-token gradient magnitude from embedding.weight.grad.
    Requires a temporary backward pass using LM loss.
    Returns topk indices (B, MEMORY_SLOTS).
    """
    B, L, H = hidden.shape
    # We need a fresh forward pass with grad enabled to get embedding grads
    encoder.embed.weight.grad = None
    with torch.enable_grad():
        emb = encoder.embed(seq)           # (B, L, H)  — leaf via embedding weight
        a, _  = encoder.attn(emb, emb, emb, need_weights=False)
        h_    = encoder.norm1(emb + a)
        h_    = encoder.norm2(h_ + encoder.ff(h_))
        lm_logits = encoder.lm_head(h_)
        shift_l = lm_logits[:, :-1, :].contiguous()
        shift_t = seq[:, 1:].contiguous()
        lm_loss = F.cross_entropy(
            shift_l.view(-1, VOCAB_SIZE), shift_t.view(-1), reduction="mean"
        )
        lm_loss.backward()

    # grad shape: (VOCAB_SIZE, HIDDEN_DIM)
    grad = encoder.embed.weight.grad  # (V, H)
    if grad is None:
        return select_random(hidden, None)

    # norm per vocab id -> score per position = norm of grad[token_id]
    grad_norms = grad.norm(dim=1)           # (V,)
    token_ids  = seq                        # (B, L)
    scores     = grad_norms[token_ids]      # (B, L)
    topk       = scores.topk(MEMORY_SLOTS, dim=1).indices
    encoder.embed.weight.grad = None        # clean up
    return topk


# ── Full model ────────────────────────────────────────────────────────────────

class RecallModel(nn.Module):
    def __init__(self, write_mode: str = "gradient"):
        super().__init__()
        self.encoder     = TinyEncoder()
        self.reader      = MemoryReader()
        self.query_embed = nn.Embedding(VOCAB_SIZE, HIDDEN_DIM)
        self.write_mode  = write_mode

    def _select(self, hidden, attn_w, seq):
        if self.write_mode == "gradient":
            return select_gradient(hidden, seq, self.encoder)
        elif self.write_mode == "attention":
            return select_attention(hidden, attn_w)
        else:
            return select_random(hidden, attn_w)

    def forward(self, seq, query, target):
        hidden, attn_w = self.encoder(seq)
        topk = self._select(hidden, attn_w, seq)
        B, L, H = hidden.shape
        mem    = hidden.detach().gather(1, topk.unsqueeze(-1).expand(-1, -1, H))
        q_h    = self.query_embed(query)
        logits = self.reader(q_h, mem, mem)
        return F.cross_entropy(logits, target)


# ── Train / Eval ──────────────────────────────────────────────────────────────

def train_model(model: RecallModel, steps: int) -> None:
    opt = Adam(model.parameters(), lr=LR)
    model.train()
    for step in range(steps):
        seq, q, t = make_recall_batch(BATCH_SIZE)
        seq, q, t = seq.to(DEVICE), q.to(DEVICE), t.to(DEVICE)
        loss = model(seq, q, t)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (step + 1) % 500 == 0:
            print(f"    step {step+1}/{steps}  loss={loss.item():.4f}")


def eval_model(model: RecallModel, steps: int) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for _ in range(steps):
            seq, q, t = make_recall_batch(BATCH_SIZE)
            seq, q, t = seq.to(DEVICE), q.to(DEVICE), t.to(DEVICE)
            hidden, attn_w = model.encoder(seq)
            # For gradient mode at eval, fall back to attention (no backward needed)
            if model.write_mode == "gradient":
                topk = select_attention(hidden, attn_w)
            else:
                topk = model._select(hidden, attn_w, seq)
            B, L, H = hidden.shape
            mem    = hidden.gather(1, topk.unsqueeze(-1).expand(-1, -1, H))
            q_h    = model.query_embed(q)
            logits = model.reader(q_h, mem, mem)
            preds  = logits.argmax(-1)
            correct += (preds == t).sum().item()
            total   += t.shape[0]
    return correct / total


def eval_gradient_mode(model: RecallModel, steps: int) -> float:
    """Eval using actual gradient selection (requires enable_grad)."""
    model.eval()
    correct = total = 0
    for _ in range(steps):
        seq, q, t = make_recall_batch(BATCH_SIZE)
        seq, q, t = seq.to(DEVICE), q.to(DEVICE), t.to(DEVICE)
        with torch.no_grad():
            hidden, attn_w = model.encoder(seq)
        topk = select_gradient(hidden, seq, model.encoder)
        with torch.no_grad():
            B, L, H = hidden.shape
            mem    = hidden.gather(1, topk.unsqueeze(-1).expand(-1, -1, H))
            q_h    = model.query_embed(q)
            logits = model.reader(q_h, mem, mem)
        preds = logits.argmax(-1)
        correct += (preds == t).sum().item()
        total   += t.shape[0]
    return correct / total


# ── Experiment ────────────────────────────────────────────────────────────────

class Exp13GradientMagnitudeWrite(Experiment):
    experiment_id = "exp_1_3"
    hypothesis = (
        "Storing tokens where gradient magnitude is highest produces memories "
        "that generalize better than attention-selected memories."
    )

    def run(self) -> ExperimentResult:
        torch.manual_seed(42)

        accs = {}
        for mode in ("gradient", "attention", "random"):
            print(f"  Training write_mode={mode} ...")
            m = RecallModel(write_mode=mode).to(DEVICE)
            train_model(m, TRAIN_STEPS)
            if mode == "gradient":
                acc = eval_gradient_mode(m, EVAL_STEPS)
            else:
                acc = eval_model(m, EVAL_STEPS)
            accs[mode] = acc
            print(f"    acc={acc:.3f}")

        gradient_acc  = accs["gradient"]
        attention_acc = accs["attention"]
        random_acc    = accs["random"]
        gap = gradient_acc - attention_acc

        if gap > 0.02:
            outcome = OUTCOME_SUPPORTED
        elif gradient_acc == min(accs.values()):
            outcome = OUTCOME_REFUTED
        else:
            outcome = OUTCOME_INCONCLUSIVE

        metrics = {
            "gradient_acc":  round(gradient_acc, 4),
            "attention_acc": round(attention_acc, 4),
            "random_acc":    round(random_acc, 4),
            "gap_gradient_minus_attention": round(gap, 4),
        }
        notes = (
            f"Gradient acc={gradient_acc:.3f}, attention={attention_acc:.3f}, "
            f"random={random_acc:.3f}. Gap={gap:+.4f}."
        )
        return self.result(outcome, metrics, notes, config={
            "vocab_size": VOCAB_SIZE, "hidden_dim": HIDDEN_DIM,
            "seq_len": SEQ_LEN, "batch_size": BATCH_SIZE,
            "train_steps": TRAIN_STEPS, "memory_slots": MEMORY_SLOTS,
            "num_pairs": NUM_PAIRS,
        })


if __name__ == "__main__":
    Exp13GradientMagnitudeWrite().execute()
