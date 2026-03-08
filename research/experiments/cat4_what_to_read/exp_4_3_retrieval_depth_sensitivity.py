"""
Experiment 4.3 — Retrieval Depth Sensitivity

Hypothesis: There exists an optimal retrieval depth (top-k) beyond which
additional retrieved entries introduce more noise than signal.

Setup:
  - Memory of 16 slots
  - Query retrieves top-k entries (k = 1, 2, 4, 8, 12, 16)
  - Retrieved entries are averaged (soft retrieval over top-k)
  - Task: associative recall where exactly 1 entry in memory is relevant
  - Noise entries are random vectors
  - SUPPORTED if accuracy peaks at some k < 16 and degrades at k=16
  - REFUTED if accuracy monotonically improves with k
  - INCONCLUSIVE if flat
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

# ── Config ─────────────────────────────────────────────────────────────────────

HIDDEN_DIM   = 64
VOCAB_SIZE   = 64
SEQ_LEN      = 24
BATCH_SIZE   = 32
TRAIN_STEPS  = 1500
MEMORY_SLOTS = 16
K_VALUES     = [1, 2, 4, 8, 12, 16]
LR           = 3e-4
DEVICE       = "cpu"
EVAL_STEPS   = 300


# ── Data ───────────────────────────────────────────────────────────────────────

def make_batch(batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Associative recall with 1 relevant slot.
    Returns (query_h, memory, targets).
      query_h: (B, H)   — embedding of query key
      memory:  (B, M, H) — slot 0 is relevant, rest are random noise
      targets: (B,)
    """
    targets  = torch.randint(0, VOCAB_SIZE, (batch_size,))
    query_h  = torch.randn(batch_size, HIDDEN_DIM)

    # relevant slot: positioned so it has high similarity with query
    # slot 0 = query_h + small noise (the relevant entry)
    relevant = query_h + torch.randn(batch_size, HIDDEN_DIM) * 0.05  # (B, H)
    relevant = relevant / (relevant.norm(dim=-1, keepdim=True) + 1e-8)

    noise    = torch.randn(batch_size, MEMORY_SLOTS - 1, HIDDEN_DIM)
    noise    = noise / (noise.norm(dim=-1, keepdim=True) + 1e-8)

    memory   = torch.cat([relevant.unsqueeze(1), noise], dim=1)  # (B, M, H)

    # normalize query_h
    query_h  = query_h / (query_h.norm(dim=-1, keepdim=True) + 1e-8)

    return query_h.detach(), memory.detach(), targets


# ── Model ──────────────────────────────────────────────────────────────────────

class TopKRetrievalModel(nn.Module):
    """Retrieves top-k entries, averages them, predicts target."""

    def __init__(self, k: int) -> None:
        super().__init__()
        self.k          = k
        self.query_proj = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.out_proj   = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.classifier = nn.Linear(HIDDEN_DIM, VOCAB_SIZE)

    def forward(self, query_h: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        q    = self.query_proj(query_h)           # (B, H)
        sims = torch.bmm(memory, q.unsqueeze(-1)).squeeze(-1) / (HIDDEN_DIM ** 0.5)  # (B, M)

        k = min(self.k, memory.shape[1])

        if k == memory.shape[1]:
            # retrieve all: standard soft attention
            weights  = F.softmax(sims, dim=-1).unsqueeze(-1)  # (B, M, 1)
            retrieved = (weights * memory).sum(1)
        else:
            # top-k: zero out all but top-k before softmax
            topk_vals, topk_idx = sims.topk(k, dim=-1)  # (B, k)
            mask = torch.full_like(sims, float('-inf'))
            mask.scatter_(1, topk_idx, topk_vals)
            weights  = F.softmax(mask, dim=-1).unsqueeze(-1)
            retrieved = (weights * memory).sum(1)

        return self.classifier(self.out_proj(retrieved))


# ── Training helper ────────────────────────────────────────────────────────────

def train_and_eval(k: int) -> float:
    """Train a model with retrieval depth k, return accuracy."""
    torch.manual_seed(42)
    model = TopKRetrievalModel(k).to(DEVICE)
    opt   = Adam(model.parameters(), lr=LR)
    model.train()

    for step in range(TRAIN_STEPS):
        query_h, memory, targets = make_batch(BATCH_SIZE)
        logits = model(query_h, memory)
        loss   = F.cross_entropy(logits, targets)
        opt.zero_grad()
        loss.backward()
        opt.step()

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for _ in range(EVAL_STEPS):
            query_h, memory, targets = make_batch(BATCH_SIZE)
            preds = model(query_h, memory).argmax(dim=-1)
            correct += (preds == targets).sum().item()
            total   += BATCH_SIZE

    acc = correct / total
    print(f"  k={k:2d} -> acc={acc:.4f}")
    return acc


def is_monotonically_increasing(values: list[float]) -> bool:
    return all(values[i] <= values[i + 1] for i in range(len(values) - 1))


def is_flat(values: list[float], tol: float = 0.02) -> bool:
    return (max(values) - min(values)) < tol


def find_peak_k(k_values: list[int], accs: list[float]) -> tuple[int, float]:
    idx = accs.index(max(accs))
    return k_values[idx], accs[idx]


# ── Experiment ─────────────────────────────────────────────────────────────────

class Exp43RetrievalDepthSensitivity(Experiment):
    experiment_id = "exp_4_3"
    hypothesis = (
        "There exists an optimal retrieval depth (top-k) beyond which additional "
        "retrieved entries introduce more noise than signal."
    )

    def run(self) -> ExperimentResult:
        print(f"\n  Memory slots: {MEMORY_SLOTS}, testing k in {K_VALUES}")
        accs: list[float] = []

        for k in K_VALUES:
            acc = train_and_eval(k)
            accs.append(acc)

        optimal_k, peak_acc = find_peak_k(K_VALUES, accs)
        acc_at_max_k = accs[-1]  # k=16

        print(f"\n  Optimal k={optimal_k}  peak_acc={peak_acc:.4f}  acc@k=16={acc_at_max_k:.4f}")

        acc_dict = {f"acc_at_k_{k}": round(a, 4) for k, a in zip(K_VALUES, accs)}

        if is_flat(accs):
            outcome = OUTCOME_INCONCLUSIVE
            notes = f"Accuracy is flat across k values (range={max(accs)-min(accs):.4f})."
        elif is_monotonically_increasing(accs):
            outcome = OUTCOME_REFUTED
            notes = f"Accuracy monotonically improves with k; no optimal depth found."
        elif optimal_k < MEMORY_SLOTS and acc_at_max_k < peak_acc - 0.01:
            outcome = OUTCOME_SUPPORTED
            notes = (
                f"Accuracy peaks at k={optimal_k} ({peak_acc:.4f}) and degrades "
                f"at k={MEMORY_SLOTS} ({acc_at_max_k:.4f})."
            )
        else:
            outcome = OUTCOME_INCONCLUSIVE
            notes = f"Peak at k={optimal_k} but degradation at k_max is marginal."

        metrics = {
            **acc_dict,
            "optimal_k":    optimal_k,
            "peak_acc":     round(peak_acc, 4),
            "acc_at_max_k": round(acc_at_max_k, 4),
        }
        config = {
            "hidden_dim":   HIDDEN_DIM,
            "vocab_size":   VOCAB_SIZE,
            "memory_slots": MEMORY_SLOTS,
            "k_values":     K_VALUES,
            "train_steps":  TRAIN_STEPS,
            "batch_size":   BATCH_SIZE,
        }
        return self.result(outcome, metrics, notes, config)


if __name__ == "__main__":
    Exp43RetrievalDepthSensitivity().execute()
