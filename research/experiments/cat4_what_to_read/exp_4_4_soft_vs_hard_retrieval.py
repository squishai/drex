"""
Experiment 4.4 — Soft vs Hard Retrieval

Hypothesis: Soft retrieval (weighted average) produces more stable training than
hard retrieval (discrete selection), though hard retrieval may achieve higher
peak task accuracy.

Setup:
  - Same associative recall task (memory 8 slots)
  - Model A (Soft): attention softmax over memory entries, weighted average
  - Model B (Hard): argmax selection with straight-through gradient estimator
  - Track (1) loss variance over training windows (stability)
  - Track (2) final accuracy (performance)
  - SUPPORTED if soft_loss_variance < hard_loss_variance AND hard_final_acc >= soft_final_acc
  - REFUTED if hard is more stable (hard_loss_variance < soft_loss_variance)
  - INCONCLUSIVE if stability and accuracy ranking agree on the same model
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
VARIANCE_WINDOW = 50   # compute variance over trailing N loss values


# ── Data ───────────────────────────────────────────────────────────────────────

def make_batch(batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Associative recall: slot 0 is relevant (high cosine similarity with query).
    Returns (query_h, memory, targets).
    """
    targets  = torch.randint(0, VOCAB_SIZE, (batch_size,))
    query_h  = torch.randn(batch_size, HIDDEN_DIM)
    query_h  = F.normalize(query_h, dim=-1)

    relevant = query_h + torch.randn(batch_size, HIDDEN_DIM) * 0.1
    relevant = F.normalize(relevant, dim=-1)

    noise    = torch.randn(batch_size, MEMORY_SLOTS - 1, HIDDEN_DIM)
    noise    = F.normalize(noise, dim=-1)

    memory   = torch.cat([relevant.unsqueeze(1), noise], dim=1)  # (B, M, H)
    return query_h.detach(), memory.detach(), targets


# ── Models ─────────────────────────────────────────────────────────────────────

class SoftRetrievalModel(nn.Module):
    """Softmax weighted average retrieval."""

    def __init__(self) -> None:
        super().__init__()
        self.query_proj = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.out_proj   = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.classifier = nn.Linear(HIDDEN_DIM, VOCAB_SIZE)

    def forward(self, query_h: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        q       = self.query_proj(query_h)
        sims    = torch.bmm(memory, q.unsqueeze(-1)).squeeze(-1) / (HIDDEN_DIM ** 0.5)
        weights = F.softmax(sims, dim=-1).unsqueeze(-1)
        retr    = (weights * memory).sum(1)
        return self.classifier(self.out_proj(retr))


class HardRetrievalModel(nn.Module):
    """
    Hard argmax retrieval with straight-through gradient estimator.
    Forward: argmax one-hot selection.
    Backward: gradient flows through softmax (straight-through trick).
    """

    def __init__(self) -> None:
        super().__init__()
        self.query_proj = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.out_proj   = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.classifier = nn.Linear(HIDDEN_DIM, VOCAB_SIZE)

    def forward(self, query_h: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        q    = self.query_proj(query_h)
        sims = torch.bmm(memory, q.unsqueeze(-1)).squeeze(-1) / (HIDDEN_DIM ** 0.5)

        # soft weights for backward pass
        soft_weights = F.softmax(sims, dim=-1)  # (B, M)

        # hard one-hot for forward pass
        hard_idx     = sims.argmax(dim=-1)      # (B,)
        hard_weights = F.one_hot(hard_idx, num_classes=memory.shape[1]).float()  # (B, M)

        # straight-through: forward uses hard, backward uses soft
        st_weights = (hard_weights - soft_weights).detach() + soft_weights  # (B, M)

        retr = torch.bmm(st_weights.unsqueeze(1), memory).squeeze(1)  # (B, H)
        return self.classifier(self.out_proj(retr))


# ── Training helper ────────────────────────────────────────────────────────────

def train_and_eval(
    model: nn.Module,
    label: str,
) -> tuple[list[float], float, float]:
    """
    Train model. Returns (loss_history, loss_variance, final_acc).
    loss_history is recorded every step.
    """
    opt = Adam(model.parameters(), lr=LR)
    model.train()
    loss_history: list[float] = []

    for step in range(TRAIN_STEPS):
        query_h, memory, targets = make_batch(BATCH_SIZE)
        logits = model(query_h, memory)
        loss   = F.cross_entropy(logits, targets)
        opt.zero_grad()
        loss.backward()
        opt.step()
        loss_history.append(loss.item())

        if (step + 1) % 300 == 0:
            print(f"  [{label}] step {step+1}/{TRAIN_STEPS}  loss={loss.item():.4f}")

    # variance computed over the last VARIANCE_WINDOW losses (stable region)
    window      = loss_history[-VARIANCE_WINDOW:]
    mean_loss   = sum(window) / len(window)
    variance    = sum((x - mean_loss) ** 2 for x in window) / len(window)

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for _ in range(EVAL_STEPS):
            query_h, memory, targets = make_batch(BATCH_SIZE)
            preds = model(query_h, memory).argmax(dim=-1)
            correct += (preds == targets).sum().item()
            total   += BATCH_SIZE

    final_acc = correct / total
    return loss_history, variance, final_acc


# ── Experiment ─────────────────────────────────────────────────────────────────

class Exp44SoftVsHardRetrieval(Experiment):
    experiment_id = "exp_4_4"
    hypothesis = (
        "Soft retrieval (weighted average) produces more stable training than hard "
        "retrieval (discrete selection), though hard retrieval may achieve higher "
        "peak task accuracy."
    )

    def run(self) -> ExperimentResult:
        torch.manual_seed(42)
        print("\n  Training Soft retrieval model...")
        soft_model = SoftRetrievalModel().to(DEVICE)
        _, soft_variance, soft_acc = train_and_eval(soft_model, "Soft")

        torch.manual_seed(42)
        print("\n  Training Hard retrieval model (straight-through)...")
        hard_model = HardRetrievalModel().to(DEVICE)
        _, hard_variance, hard_acc = train_and_eval(hard_model, "Hard")

        soft_is_more_stable  = soft_variance < hard_variance
        hard_is_more_accurate = hard_acc >= soft_acc

        print(f"\n  Soft — acc={soft_acc:.4f}  loss_var={soft_variance:.6f}")
        print(f"  Hard — acc={hard_acc:.4f}  loss_var={hard_variance:.6f}")
        print(f"  Soft more stable: {soft_is_more_stable}, Hard more accurate: {hard_is_more_accurate}")

        # SUPPORTED: soft more stable AND hard more accurate (each wins its domain)
        # REFUTED: hard is more stable
        # INCONCLUSIVE: same model wins both, or the same model wins neither clearly
        if soft_is_more_stable and hard_is_more_accurate:
            outcome = OUTCOME_SUPPORTED
        elif not soft_is_more_stable:
            outcome = OUTCOME_REFUTED
        else:
            outcome = OUTCOME_INCONCLUSIVE

        metrics = {
            "soft_loss_variance":  round(soft_variance, 6),
            "hard_loss_variance":  round(hard_variance, 6),
            "soft_final_acc":      round(soft_acc, 4),
            "hard_final_acc":      round(hard_acc, 4),
            "soft_is_more_stable": soft_is_more_stable,
            "hard_is_more_accurate": hard_is_more_accurate,
        }
        notes = (
            f"Soft var={soft_variance:.6f} vs Hard var={hard_variance:.6f}; "
            f"Soft acc={soft_acc:.4f} vs Hard acc={hard_acc:.4f}."
        )
        config = {
            "hidden_dim":      HIDDEN_DIM,
            "vocab_size":      VOCAB_SIZE,
            "memory_slots":    MEMORY_SLOTS,
            "train_steps":     TRAIN_STEPS,
            "batch_size":      BATCH_SIZE,
            "variance_window": VARIANCE_WINDOW,
        }
        return self.result(outcome, metrics, notes, config)


if __name__ == "__main__":
    Exp44SoftVsHardRetrieval().execute()
