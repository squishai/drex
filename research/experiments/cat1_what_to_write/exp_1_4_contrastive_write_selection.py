"""
Experiment 1.4 — Contrastive Write Selection

Hypothesis: Diversity-driven storage (maximally dissimilar entries) outperforms
importance-driven storage on recall tasks.

Setup:
  - Controller A (Importance): top-k by gate score
  - Controller B (Diversity): same gate + contrastive loss term minimizing
    mean pairwise cosine similarity among written token embeddings
  - Train both on associative recall
  - Measure accuracy and mean pairwise similarity of written tokens
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
DIVERSITY_LAM = 0.1   # weight for diversity loss in controller B
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
        return self.gate(hidden).squeeze(-1).sigmoid()  # (B, L)


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


def mean_pairwise_cosine(vectors: torch.Tensor) -> torch.Tensor:
    """
    vectors: (B, K, H)
    Returns mean pairwise cosine similarity (B,) averaged across batch.
    """
    B, K, H = vectors.shape
    norm = F.normalize(vectors, dim=-1)             # (B, K, H)
    sim  = torch.bmm(norm, norm.transpose(1, 2))    # (B, K, K)
    # exclude self-similarity diagonal
    mask = ~torch.eye(K, dtype=torch.bool, device=vectors.device).unsqueeze(0)
    off_diag = sim[mask.expand(B, -1, -1)].view(B, K * (K - 1))
    return off_diag.mean(dim=-1)                    # (B,)


# ── Full models ───────────────────────────────────────────────────────────────

class ImportanceController(nn.Module):
    """Controller A: pure gate-score top-k selection."""

    def __init__(self):
        super().__init__()
        self.encoder     = TinyEncoder()
        self.gate        = WriteGate()
        self.reader      = MemoryReader()
        self.query_embed = nn.Embedding(VOCAB_SIZE, HIDDEN_DIM)

    def forward(self, seq, query, target):
        hidden, _ = self.encoder(seq)
        scores    = self.gate.scores(hidden)                         # (B, L)
        topk      = scores.topk(MEMORY_SLOTS, dim=1).indices        # (B, K)
        B, L, H   = hidden.shape
        mem       = hidden.gather(1, topk.unsqueeze(-1).expand(-1, -1, H))
        q_h       = self.query_embed(query)
        logits    = self.reader(q_h, mem, mem)
        task_loss = F.cross_entropy(logits, target)
        return task_loss, torch.zeros(1, device=seq.device)

    def get_memory_similarity(self, seq):
        with torch.no_grad():
            hidden, _ = self.encoder(seq)
            scores    = self.gate.scores(hidden)
            topk      = scores.topk(MEMORY_SLOTS, dim=1).indices
            B, L, H   = hidden.shape
            mem       = hidden.gather(1, topk.unsqueeze(-1).expand(-1, -1, H))
        return mean_pairwise_cosine(mem).mean().item()


class DiversityController(nn.Module):
    """
    Controller B: gate score + diversity loss that penalises mean pairwise
    cosine similarity among selected tokens.
    """

    def __init__(self):
        super().__init__()
        self.encoder     = TinyEncoder()
        self.gate        = WriteGate()
        self.reader      = MemoryReader()
        self.query_embed = nn.Embedding(VOCAB_SIZE, HIDDEN_DIM)

    def forward(self, seq, query, target):
        hidden, _ = self.encoder(seq)
        scores    = self.gate.scores(hidden)                    # (B, L)

        # Straight-through hard top-k for selection
        topk      = scores.topk(MEMORY_SLOTS, dim=1).indices   # (B, K)
        B, L, H   = hidden.shape
        mem       = hidden.gather(1, topk.unsqueeze(-1).expand(-1, -1, H))

        # Diversity loss: penalise pairwise similarity of written tokens
        div_loss  = mean_pairwise_cosine(mem).mean()

        q_h       = self.query_embed(query)
        logits    = self.reader(q_h, mem, mem)
        task_loss = F.cross_entropy(logits, target)
        total     = task_loss + DIVERSITY_LAM * div_loss
        return total, div_loss.unsqueeze(0)

    def get_memory_similarity(self, seq):
        with torch.no_grad():
            hidden, _ = self.encoder(seq)
            scores    = self.gate.scores(hidden)
            topk      = scores.topk(MEMORY_SLOTS, dim=1).indices
            B, L, H   = hidden.shape
            mem       = hidden.gather(1, topk.unsqueeze(-1).expand(-1, -1, H))
        return mean_pairwise_cosine(mem).mean().item()


# ── Train / Eval ──────────────────────────────────────────────────────────────

def train_controller(model: nn.Module, steps: int) -> None:
    opt = Adam(model.parameters(), lr=LR)
    model.train()
    for step in range(steps):
        seq, q, t = make_recall_batch(BATCH_SIZE)
        seq, q, t = seq.to(DEVICE), q.to(DEVICE), t.to(DEVICE)
        loss, _ = model(seq, q, t)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (step + 1) % 500 == 0:
            print(f"    step {step+1}/{steps}  loss={loss.item():.4f}")


def eval_controller(model: nn.Module, steps: int) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for _ in range(steps):
            seq, q, t = make_recall_batch(BATCH_SIZE)
            seq, q, t = seq.to(DEVICE), q.to(DEVICE), t.to(DEVICE)
            hidden, _ = model.encoder(seq)
            scores    = model.gate.scores(hidden)
            topk      = scores.topk(MEMORY_SLOTS, dim=1).indices
            B, L, H   = hidden.shape
            mem       = hidden.gather(1, topk.unsqueeze(-1).expand(-1, -1, H))
            q_h       = model.query_embed(q)
            logits    = model.reader(q_h, mem, mem)
            preds     = logits.argmax(-1)
            correct  += (preds == t).sum().item()
            total    += t.shape[0]
    return correct / total


def eval_similarity(model: nn.Module, steps: int = 50) -> float:
    sims = []
    for _ in range(steps):
        seq, _, _ = make_recall_batch(BATCH_SIZE)
        sims.append(model.get_memory_similarity(seq.to(DEVICE)))
    return sum(sims) / len(sims)


# ── Experiment ────────────────────────────────────────────────────────────────

class Exp14ContrastiveWriteSelection(Experiment):
    experiment_id = "exp_1_4"
    hypothesis = (
        "Diversity-driven storage (maximally dissimilar entries) outperforms "
        "importance-driven storage on recall tasks."
    )

    def run(self) -> ExperimentResult:
        torch.manual_seed(42)

        print("  Training importance controller ...")
        imp_model = ImportanceController().to(DEVICE)
        train_controller(imp_model, TRAIN_STEPS)
        importance_acc = eval_controller(imp_model, EVAL_STEPS)
        importance_sim = eval_similarity(imp_model)
        print(f"    acc={importance_acc:.3f}  mean_pairwise_sim={importance_sim:.4f}")

        print("  Training diversity controller ...")
        div_model = DiversityController().to(DEVICE)
        train_controller(div_model, TRAIN_STEPS)
        diversity_acc = eval_controller(div_model, EVAL_STEPS)
        diversity_sim = eval_similarity(div_model)
        print(f"    acc={diversity_acc:.3f}  mean_pairwise_sim={diversity_sim:.4f}")

        gap = diversity_acc - importance_acc

        if diversity_acc > importance_acc:
            outcome = OUTCOME_SUPPORTED
        elif importance_acc > diversity_acc + 0.01:
            outcome = OUTCOME_REFUTED
        else:
            outcome = OUTCOME_INCONCLUSIVE

        metrics = {
            "importance_acc":           round(importance_acc, 4),
            "diversity_acc":            round(diversity_acc, 4),
            "importance_mean_pairwise_sim": round(importance_sim, 4),
            "diversity_mean_pairwise_sim":  round(diversity_sim, 4),
            "gap_diversity_minus_importance": round(gap, 4),
        }
        notes = (
            f"Diversity acc={diversity_acc:.3f} vs importance acc={importance_acc:.3f}. "
            f"Gap={gap:+.4f}. Diversity sim={diversity_sim:.4f} vs importance sim={importance_sim:.4f}."
        )
        return self.result(outcome, metrics, notes, config={
            "vocab_size": VOCAB_SIZE, "hidden_dim": HIDDEN_DIM,
            "seq_len": SEQ_LEN, "batch_size": BATCH_SIZE,
            "train_steps": TRAIN_STEPS, "memory_slots": MEMORY_SLOTS,
            "num_pairs": NUM_PAIRS, "diversity_lambda": DIVERSITY_LAM,
        })


if __name__ == "__main__":
    Exp14ContrastiveWriteSelection().execute()
