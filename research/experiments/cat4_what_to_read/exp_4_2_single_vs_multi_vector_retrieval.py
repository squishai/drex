"""
Experiment 4.2 — Single vs Multi-Vector Retrieval

Hypothesis: Multi-vector retrieval captures more relevant content than
single-vector retrieval for queries with multi-faceted information needs.

Setup:
  - Task requires retrieving TWO pieces of information to answer correctly
  - Memory has 8 slots
  - Model A (Single): one query vector, retrieve top-1, use it to answer
  - Model B (Multi): two query vectors, retrieve top-1 for each, combine
  - Both models have same parameter budget
  - SUPPORTED if multi_acc > single_acc + 0.03
  - REFUTED if single_acc >= multi_acc
  - INCONCLUSIVE otherwise
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
MEMORY_SLOTS = 8
LR           = 3e-4
DEVICE       = "cpu"
EVAL_STEPS   = 300


# ── Data ───────────────────────────────────────────────────────────────────────

def make_batch(batch_size: int) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor
]:
    """
    Two-fact retrieval task.
    Memory holds pairs: slot i -> (key_i, fact_A_i, fact_B_i).
    Query encodes a key; answer = fact_A + fact_B combined as a single token
    index: target = (fact_A_token + fact_B_token) % VOCAB_SIZE.
    Returns (context_h, memory, targets).
      context_h: (B, H)   — context that encodes the query key
      memory:    (B, M, H) — M slots, each = embedding of (key XOR factA XOR factB)
      targets:   (B,)     — scalar token index
    """
    # generate token indices for keys, factA, factB
    keys   = torch.randint(0, VOCAB_SIZE, (batch_size,))
    fact_a = torch.randint(0, VOCAB_SIZE // 2, (batch_size,))
    fact_b = torch.randint(0, VOCAB_SIZE // 2, (batch_size,))
    targets = (fact_a + fact_b) % VOCAB_SIZE  # (B,)

    # build memory: one slot per item holds the relevant pair, others are noise
    # slot layout: slot 0 = fact_A evidence, slot 1 = fact_B evidence, slots 2-7 = noise
    memory = torch.randn(batch_size, MEMORY_SLOTS, HIDDEN_DIM)

    # encode fact_A into slot 0: use a deterministic projection from token
    for b in range(batch_size):
        torch.manual_seed(int(keys[b].item()) * 1000 + int(fact_a[b].item()))
        fa_vec = torch.randn(HIDDEN_DIM)
        fa_vec = fa_vec / fa_vec.norm()
        memory[b, 0] = fa_vec

        torch.manual_seed(int(keys[b].item()) * 1000 + int(fact_b[b].item()) + 500)
        fb_vec = torch.randn(HIDDEN_DIM)
        fb_vec = fb_vec / fb_vec.norm()
        memory[b, 1] = fb_vec

    # context_h: a simple embedding of the key that must activate both slots
    # we encode it as a blend of the two fact directions plus noise
    context_h = torch.zeros(batch_size, HIDDEN_DIM)
    for b in range(batch_size):
        context_h[b] = (memory[b, 0] + memory[b, 1]) / 2.0 + torch.randn(HIDDEN_DIM) * 0.1

    return context_h.detach(), memory.detach(), targets


# ── Models ─────────────────────────────────────────────────────────────────────

class SingleVectorModel(nn.Module):
    """One query vector, retrieve top-1 (soft), predict from retrieved."""

    def __init__(self) -> None:
        super().__init__()
        # query projection (same budget as multi's two projections by doubling size)
        self.query_proj = nn.Sequential(
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM * 2),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM * 2, HIDDEN_DIM),
        )
        self.out_proj   = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.classifier = nn.Linear(HIDDEN_DIM, VOCAB_SIZE)

    def forward(self, ctx: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        q = self.query_proj(ctx)  # (B, H)
        sims    = torch.bmm(memory, q.unsqueeze(-1)).squeeze(-1) / (HIDDEN_DIM ** 0.5)
        weights = F.softmax(sims, dim=-1).unsqueeze(-1)
        retr    = (weights * memory).sum(1)   # (B, H)
        return self.classifier(self.out_proj(retr))


class MultiVectorModel(nn.Module):
    """Two query vectors, retrieve top-1 for each, average, predict."""

    def __init__(self) -> None:
        super().__init__()
        # two separate query projections
        self.query_proj_a = nn.Sequential(
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
        )
        self.query_proj_b = nn.Sequential(
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
        )
        self.out_proj   = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.classifier = nn.Linear(HIDDEN_DIM, VOCAB_SIZE)

    def forward(self, ctx: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        qa = self.query_proj_a(ctx)  # (B, H)
        qb = self.query_proj_b(ctx)  # (B, H)

        sims_a  = torch.bmm(memory, qa.unsqueeze(-1)).squeeze(-1) / (HIDDEN_DIM ** 0.5)
        sims_b  = torch.bmm(memory, qb.unsqueeze(-1)).squeeze(-1) / (HIDDEN_DIM ** 0.5)

        w_a = F.softmax(sims_a, dim=-1).unsqueeze(-1)
        w_b = F.softmax(sims_b, dim=-1).unsqueeze(-1)

        retr_a = (w_a * memory).sum(1)  # (B, H)
        retr_b = (w_b * memory).sum(1)  # (B, H)
        retr   = (retr_a + retr_b) / 2.0
        return self.classifier(self.out_proj(retr))


# ── Training helper ────────────────────────────────────────────────────────────

def train_and_eval(model: nn.Module, label: str) -> tuple[float, float]:
    opt = Adam(model.parameters(), lr=LR)
    model.train()
    final_loss = 0.0

    for step in range(TRAIN_STEPS):
        ctx, mem, targets = make_batch(BATCH_SIZE)
        logits = model(ctx, mem)
        loss   = F.cross_entropy(logits, targets)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step >= TRAIN_STEPS - 50:
            final_loss += loss.item()
        if (step + 1) % 300 == 0:
            print(f"  [{label}] step {step+1}/{TRAIN_STEPS}  loss={loss.item():.4f}")

    final_loss /= 50

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for _ in range(EVAL_STEPS):
            ctx, mem, targets = make_batch(BATCH_SIZE)
            preds = model(ctx, mem).argmax(dim=-1)
            correct += (preds == targets).sum().item()
            total   += BATCH_SIZE

    return final_loss, correct / total


# ── Experiment ─────────────────────────────────────────────────────────────────

class Exp42SingleVsMultiVectorRetrieval(Experiment):
    experiment_id = "exp_4_2"
    hypothesis = (
        "Multi-vector retrieval captures more relevant content than single-vector "
        "retrieval for queries with multi-faceted information needs."
    )

    def run(self) -> ExperimentResult:
        torch.manual_seed(42)

        print("\n  Training Single-vector model...")
        single_model = SingleVectorModel().to(DEVICE)
        single_loss, single_acc = train_and_eval(single_model, "Single")

        torch.manual_seed(42)
        print("\n  Training Multi-vector model...")
        multi_model = MultiVectorModel().to(DEVICE)
        multi_loss, multi_acc = train_and_eval(multi_model, "Multi")

        gap = multi_acc - single_acc
        print(f"\n  Single — acc={single_acc:.4f}  loss={single_loss:.4f}")
        print(f"  Multi  — acc={multi_acc:.4f}   loss={multi_loss:.4f}")
        print(f"  Gap (multi - single): {gap:+.4f}")

        if multi_acc > single_acc + 0.03:
            outcome = OUTCOME_SUPPORTED
        elif single_acc >= multi_acc:
            outcome = OUTCOME_REFUTED
        else:
            outcome = OUTCOME_INCONCLUSIVE

        metrics = {
            "single_acc":  round(single_acc, 4),
            "multi_acc":   round(multi_acc, 4),
            "single_loss": round(single_loss, 4),
            "multi_loss":  round(multi_loss, 4),
            "gap_multi_minus_single": round(gap, 4),
        }
        notes = (
            f"Multi acc={multi_acc:.4f} vs Single acc={single_acc:.4f}, "
            f"gap={gap:+.4f} (threshold +0.03 for SUPPORTED)."
        )
        config = {
            "hidden_dim":   HIDDEN_DIM,
            "vocab_size":   VOCAB_SIZE,
            "memory_slots": MEMORY_SLOTS,
            "train_steps":  TRAIN_STEPS,
            "batch_size":   BATCH_SIZE,
        }
        return self.result(outcome, metrics, notes, config)


if __name__ == "__main__":
    Exp42SingleVsMultiVectorRetrieval().execute()
