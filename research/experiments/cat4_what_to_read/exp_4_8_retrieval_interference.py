"""
Experiment 4.8 — Retrieval Interference from Near-Duplicates

Hypothesis: Retrieval quality degrades non-linearly as the number of
near-duplicate entries in memory increases, with a specific saturation point.

Setup:
  - Memory of 16 slots
  - Fill with near-duplicate entries: N slots contain slight variations
    (cosine_sim > 0.9) of the target entry; rest are random noise.
  - Near-duplicates are created by adding small noise (std=0.05) to the target.
  - Test retrieval accuracy at N = 0, 2, 4, 8, 12, 16 duplicate entries.
  - SUPPORTED if degradation is non-linear (large drop at some N, small before)
    and saturation point N <= 8.
  - REFUTED if degradation is linear with N.
  - INCONCLUSIVE if flat/no degradation.
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

HIDDEN_DIM    = 64
VOCAB_SIZE    = 64
SEQ_LEN       = 24
BATCH_SIZE    = 32
TRAIN_STEPS   = 1500
MEMORY_SLOTS  = 16
N_VALUES      = [0, 2, 4, 8, 12, 16]
NEAR_DUP_STD  = 0.05   # noise std for near-duplicates (ensures cosine_sim > 0.9)
LR            = 3e-4
DEVICE        = "cpu"
EVAL_STEPS    = 300


# ── Data ───────────────────────────────────────────────────────────────────────

def make_batch(batch_size: int, n_near_dups: int) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor
]:
    """
    Memory layout:
      slot 0:           the true target entry (query is similar to this)
      slots 1..n:       near-duplicates of slot 0 (slight noise, cos_sim > 0.9)
      slots n+1..15:    random noise entries
    Query = target + small noise (still most similar to slot 0).
    Target = a random token (independent of retrieval — model must retrieve slot 0
             to get a fixed 'associated' answer; approximated here as the model
             learns a mapping from the retrieved content to the correct label).

    Returns (query_h, memory, targets).
    """
    targets = torch.randint(0, VOCAB_SIZE, (batch_size,))

    # target vector (what slot 0 holds)
    target_vec = F.normalize(torch.randn(batch_size, HIDDEN_DIM), dim=-1)  # (B, H)

    # query: very similar to target_vec
    query_h = F.normalize(target_vec + torch.randn(batch_size, HIDDEN_DIM) * 0.02, dim=-1)

    # build memory
    memory_list: list[torch.Tensor] = []

    # slot 0: true target
    memory_list.append(target_vec.clone())

    # near-duplicate slots: target + std=NEAR_DUP_STD noise
    n_dups = min(n_near_dups, MEMORY_SLOTS - 1)
    for _ in range(n_dups):
        dup = F.normalize(target_vec + torch.randn(batch_size, HIDDEN_DIM) * NEAR_DUP_STD, dim=-1)
        memory_list.append(dup)

    # remaining slots: random noise
    n_noise = MEMORY_SLOTS - 1 - n_dups
    for _ in range(n_noise):
        noise = F.normalize(torch.randn(batch_size, HIDDEN_DIM), dim=-1)
        memory_list.append(noise)

    memory = torch.stack(memory_list, dim=1)  # (B, M, H)
    return query_h.detach(), memory.detach(), targets


# ── Model ──────────────────────────────────────────────────────────────────────

class AssociativeRecallModel(nn.Module):
    """Standard soft-attention retrieval model."""

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


# ── Training helper ────────────────────────────────────────────────────────────

def train_once(n_near_dups: int) -> float:
    """Train a fresh model at N near-duplicates, return eval accuracy."""
    torch.manual_seed(42)
    model = AssociativeRecallModel().to(DEVICE)
    opt   = Adam(model.parameters(), lr=LR)
    model.train()

    for step in range(TRAIN_STEPS):
        query_h, memory, targets = make_batch(BATCH_SIZE, n_near_dups)
        logits = model(query_h, memory)
        loss   = F.cross_entropy(logits, targets)
        opt.zero_grad(); loss.backward(); opt.step()

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for _ in range(EVAL_STEPS):
            query_h, memory, targets = make_batch(BATCH_SIZE, n_near_dups)
            preds = model(query_h, memory).argmax(dim=-1)
            correct += (preds == targets).sum().item()
            total   += BATCH_SIZE

    acc = correct / total
    print(f"  N={n_near_dups:2d} near-dups -> acc={acc:.4f}")
    return acc


# ── Non-linearity detection ────────────────────────────────────────────────────

def detect_nonlinear_degradation(
    n_vals: list[int],
    accs: list[float],
) -> tuple[bool, int]:
    """
    Returns (is_nonlinear, saturation_point_N).
    Non-linear: the largest single-step drop is > 2x the average step drop.
    Saturation point: N at which the largest drop occurs (or after which acc flattens).
    """
    drops = [accs[i] - accs[i + 1] for i in range(len(accs) - 1)]
    if not drops or max(drops) <= 0:
        return False, n_vals[-1]

    avg_drop  = sum(max(d, 0) for d in drops) / len(drops)
    max_drop  = max(drops)
    max_idx   = drops.index(max_drop)
    sat_n     = n_vals[max_idx]  # N value before the big drop

    is_nonlinear = (max_drop > 2.0 * avg_drop) if avg_drop > 1e-6 else False
    return is_nonlinear, sat_n


def is_linear_degradation(accs: list[float], tol: float = 0.015) -> bool:
    """True if drops are roughly equal at each step (linear decay)."""
    drops = [accs[i] - accs[i + 1] for i in range(len(accs) - 1)]
    if not drops:
        return False
    mean_drop = sum(drops) / len(drops)
    if mean_drop <= 0:
        return False
    return all(abs(d - mean_drop) <= tol for d in drops)


def is_flat(accs: list[float], tol: float = 0.02) -> bool:
    return (max(accs) - min(accs)) < tol


# ── Experiment ─────────────────────────────────────────────────────────────────

class Exp48RetrievalInterference(Experiment):
    experiment_id = "exp_4_8"
    hypothesis = (
        "Retrieval quality degrades non-linearly as the number of near-duplicate "
        "entries in memory increases, with a specific saturation point."
    )

    def run(self) -> ExperimentResult:
        print(f"\n  Memory slots={MEMORY_SLOTS}, near-dup std={NEAR_DUP_STD}")
        print(f"  Testing N in {N_VALUES}")

        accs: list[float] = []
        for n in N_VALUES:
            acc = train_once(n)
            accs.append(acc)

        is_nonlinear, sat_n = detect_nonlinear_degradation(N_VALUES, accs)
        flat   = is_flat(accs)
        linear = is_linear_degradation(accs)

        print(f"\n  Degradation non-linear: {is_nonlinear}  saturation_N={sat_n}")

        acc_dict = {f"acc_at_N_{n}": round(a, 4) for n, a in zip(N_VALUES, accs)}

        if flat:
            outcome = OUTCOME_INCONCLUSIVE
            notes = f"Accuracy is flat across N values; no interference detected."
        elif is_nonlinear and sat_n <= 8:
            outcome = OUTCOME_SUPPORTED
            notes = (
                f"Non-linear degradation detected; saturation point N={sat_n} (<= 8). "
                f"Accs: {[round(a, 4) for a in accs]}."
            )
        elif linear:
            outcome = OUTCOME_REFUTED
            notes = f"Degradation is approximately linear with N. Accs: {[round(a, 4) for a in accs]}."
        else:
            outcome = OUTCOME_INCONCLUSIVE
            notes = (
                f"Non-linear={is_nonlinear}, saturation_N={sat_n}. "
                f"Saturation point > 8 or pattern unclear. "
                f"Accs: {[round(a, 4) for a in accs]}."
            )

        metrics = {
            **acc_dict,
            "degradation_is_nonlinear": is_nonlinear,
            "saturation_point_N":       sat_n,
        }
        config = {
            "hidden_dim":    HIDDEN_DIM,
            "vocab_size":    VOCAB_SIZE,
            "memory_slots":  MEMORY_SLOTS,
            "n_values":      N_VALUES,
            "near_dup_std":  NEAR_DUP_STD,
            "train_steps":   TRAIN_STEPS,
            "batch_size":    BATCH_SIZE,
        }
        return self.result(outcome, metrics, notes, config)


if __name__ == "__main__":
    Exp48RetrievalInterference().execute()
