"""
Experiment 6.5 — Forgetting Curve Mimicry (Ebbinghaus-Style Decay)

Hypothesis: A biologically-inspired memory decay function (Ebbinghaus-style)
improves long-horizon task performance compared to instant eviction.

Setup:
  - Memory entries have an "age" counter incremented each step.
  - Ebbinghaus retention: R(t) = exp(-t / S) where t=age, S=stability (learnable).
  - At each step, entries with R < threshold are soft-evicted (multiplied by R).
  - Baseline: instant LRU eviction when full.
  - Task: long sequence (SEQ_LEN=48) where early entries are critical.
  - SUPPORTED if ebbinghaus_acc > lru_acc.
  - REFUTED if LRU >= ebbinghaus + 0.02.
  - INCONCLUSIVE if |gap| < 0.02.
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from experiments.base import (
    Experiment, ExperimentResult,
    OUTCOME_SUPPORTED, OUTCOME_REFUTED, OUTCOME_INCONCLUSIVE,
)

# ── Config ─────────────────────────────────────────────────────────────────────

HIDDEN_DIM        = 64
VOCAB_SIZE        = 64
SEQ_LEN           = 48    # long sequences per spec
BATCH_SIZE        = 32
TRAIN_STEPS       = 1500
MEMORY_SLOTS      = 8
DECAY_THRESHOLD   = 0.05  # entries with R < this are fully removed
EVAL_BATCHES      = 200
LR                = 3e-4
DEVICE            = "cpu"


# ── Data generator ─────────────────────────────────────────────────────────────

def make_batch(batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Long sequences with an early critical key-value pair.
    Returns seqs (B, SEQ_LEN), query_tok (B,), target (B,).
    """
    seqs      = torch.zeros(batch_size, SEQ_LEN, dtype=torch.long)
    query_tok = torch.zeros(batch_size, dtype=torch.long)
    target    = torch.zeros(batch_size, dtype=torch.long)

    for b in range(batch_size):
        key   = torch.randint(0, 16, (1,)).item()
        value = torch.randint(16, VOCAB_SIZE, (1,)).item()
        pos   = torch.randint(0, SEQ_LEN // 6, (1,)).item()

        seqs[b, pos]     = key
        seqs[b, pos + 1] = value

        for i in range(SEQ_LEN):
            if i in (pos, pos + 1):
                continue
            seqs[b, i] = torch.randint(1, VOCAB_SIZE, (1,)).item()

        query_tok[b] = key
        target[b]    = value

    return seqs, query_tok, target


# ── LRU baseline ───────────────────────────────────────────────────────────────

class LRUModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed     = nn.Embedding(VOCAB_SIZE + 2, HIDDEN_DIM)
        self.read_head = nn.Sequential(
            nn.Linear(HIDDEN_DIM * 2, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, VOCAB_SIZE),
        )

    def forward(self, seqs: torch.Tensor, query_tok: torch.Tensor) -> torch.Tensor:
        B = seqs.size(0)
        out = []
        for b in range(B):
            mem: list[torch.Tensor] = []
            for t in range(SEQ_LEN - 1):
                emb = self.embed(seqs[b, t].unsqueeze(0)).squeeze(0)
                if len(mem) >= MEMORY_SLOTS:
                    mem.pop(0)
                mem.append(emb)
            mem_summary = torch.stack(mem).mean(0) if mem else torch.zeros(HIDDEN_DIM)
            q_emb  = self.embed(query_tok[b].unsqueeze(0)).squeeze(0)
            logits = self.read_head(torch.cat([q_emb, mem_summary]).unsqueeze(0))
            out.append(logits)
        return torch.cat(out, dim=0)


# ── Ebbinghaus model ───────────────────────────────────────────────────────────

class EbbinghausModel(nn.Module):
    """
    Memory with Ebbinghaus-style decay. The stability S is a learnable scalar.
    Each memory entry is weighted by R(t) = exp(-age / S).
    Entries with R < DECAY_THRESHOLD are dropped.
    """
    def __init__(self):
        super().__init__()
        self.embed     = nn.Embedding(VOCAB_SIZE + 2, HIDDEN_DIM)
        self.read_head = nn.Sequential(
            nn.Linear(HIDDEN_DIM * 2, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, VOCAB_SIZE),
        )
        # Log-parameterise S to ensure S > 0; initialise to S ≈ 10 steps
        self.log_stability = nn.Parameter(torch.tensor(math.log(10.0)))

        # Track retention stats
        self._last_mean_retention: float = 1.0

    @property
    def stability(self) -> torch.Tensor:
        return torch.exp(self.log_stability)

    def forward(self, seqs: torch.Tensor, query_tok: torch.Tensor) -> torch.Tensor:
        B = seqs.size(0)
        out = []
        S = self.stability

        all_retentions: list[float] = []

        for b in range(B):
            mem_embs: list[torch.Tensor] = []
            mem_ages: list[int]          = []

            for t in range(SEQ_LEN - 1):
                # Age all existing entries
                mem_ages = [a + 1 for a in mem_ages]

                emb = self.embed(seqs[b, t].unsqueeze(0)).squeeze(0)
                mem_embs.append(emb)
                mem_ages.append(0)

                # Apply decay and prune entries with R < threshold
                retain_embs: list[torch.Tensor] = []
                retain_ages: list[int] = []
                for emb_i, age_i in zip(mem_embs, mem_ages):
                    R = torch.exp(-torch.tensor(age_i, dtype=torch.float) / S)
                    if R.item() >= DECAY_THRESHOLD:
                        retain_embs.append(emb_i * R)
                        retain_ages.append(age_i)
                mem_embs = retain_embs
                mem_ages = retain_ages

                # Hard cap if necessary (safety)
                if len(mem_embs) > MEMORY_SLOTS:
                    mem_embs = mem_embs[-MEMORY_SLOTS:]
                    mem_ages = mem_ages[-MEMORY_SLOTS:]

            # Compute mean retention at query time
            if mem_ages:
                retentions = [
                    torch.exp(-torch.tensor(a, dtype=torch.float) / S).item()
                    for a in mem_ages
                ]
                all_retentions.extend(retentions)

            mem_summary = torch.stack(mem_embs).mean(0) if mem_embs else torch.zeros(HIDDEN_DIM)
            q_emb  = self.embed(query_tok[b].unsqueeze(0)).squeeze(0)
            logits = self.read_head(torch.cat([q_emb, mem_summary]).unsqueeze(0))
            out.append(logits)

        self._last_mean_retention = (
            sum(all_retentions) / len(all_retentions) if all_retentions else 0.0
        )

        return torch.cat(out, dim=0)


# ── Evaluation ─────────────────────────────────────────────────────────────────

def evaluate(model: nn.Module, n_batches: int) -> tuple[float, float]:
    """Returns (accuracy, mean_retention_at_query_time)."""
    model.eval()
    correct = total = 0
    retentions = []
    with torch.no_grad():
        for _ in range(n_batches):
            seqs, query_tok, target = make_batch(BATCH_SIZE)
            logits = model(seqs, query_tok)
            correct += (logits.argmax(-1) == target).sum().item()
            total   += BATCH_SIZE
            if hasattr(model, '_last_mean_retention'):
                retentions.append(model._last_mean_retention)
    model.train()
    avg_ret = sum(retentions) / max(len(retentions), 1)
    return correct / total, avg_ret


# ── Experiment ──────────────────────────────────────────────────────────────────

class Exp65ForgettingCurveMimicry(Experiment):
    experiment_id = "exp_6_5"
    hypothesis = (
        "A biologically-inspired memory decay function (Ebbinghaus-style) improves "
        "long-horizon task performance compared to instant eviction."
    )

    def run(self) -> ExperimentResult:
        torch.manual_seed(42)

        config = {
            "hidden_dim": HIDDEN_DIM, "vocab_size": VOCAB_SIZE,
            "seq_len": SEQ_LEN, "batch_size": BATCH_SIZE,
            "train_steps": TRAIN_STEPS, "memory_slots": MEMORY_SLOTS,
            "decay_threshold": DECAY_THRESHOLD,
        }

        # ── Train LRU baseline ────────────────────────────────────────────────
        print("  Training LRU baseline...")
        lru = LRUModel().to(DEVICE)
        opt_lru = Adam(lru.parameters(), lr=LR)

        for step in range(TRAIN_STEPS):
            seqs, query_tok, target = make_batch(BATCH_SIZE)
            loss = F.cross_entropy(lru(seqs, query_tok), target)
            opt_lru.zero_grad(); loss.backward(); opt_lru.step()
            if (step + 1) % 500 == 0:
                print(f"    LRU step {step+1}/{TRAIN_STEPS}  loss={loss.item():.3f}")

        # ── Train Ebbinghaus model ────────────────────────────────────────────
        print("  Training Ebbinghaus decay model...")
        ebb = EbbinghausModel().to(DEVICE)
        opt_ebb = Adam(ebb.parameters(), lr=LR)

        for step in range(TRAIN_STEPS):
            seqs, query_tok, target = make_batch(BATCH_SIZE)
            loss = F.cross_entropy(ebb(seqs, query_tok), target)
            opt_ebb.zero_grad(); loss.backward(); opt_ebb.step()
            if (step + 1) % 500 == 0:
                learned_S = ebb.stability.item()
                print(f"    Ebb step {step+1}/{TRAIN_STEPS}  loss={loss.item():.3f}  S={learned_S:.2f}")

        # ── Evaluate ──────────────────────────────────────────────────────────
        lru_acc,  _        = evaluate(lru, EVAL_BATCHES)
        ebb_acc,  avg_ret  = evaluate(ebb, EVAL_BATCHES)
        learned_S          = ebb.stability.item()

        print(f"  LRU acc:            {lru_acc:.3f}")
        print(f"  Ebbinghaus acc:     {ebb_acc:.3f}")
        print(f"  Learned stability S:{learned_S:.3f}")
        print(f"  Mean retention at query time: {avg_ret:.3f}")

        gap = ebb_acc - lru_acc
        if abs(gap) < 0.02:
            outcome = OUTCOME_INCONCLUSIVE
        elif ebb_acc > lru_acc:
            outcome = OUTCOME_SUPPORTED
        else:
            outcome = OUTCOME_REFUTED

        return self.result(outcome, {
            "lru_acc":                  round(lru_acc,    4),
            "ebbinghaus_acc":           round(ebb_acc,    4),
            "gap_ebbinghaus_minus_lru": round(gap,        4),
            "learned_stability_S":      round(learned_S,  4),
            "mean_retention_at_query_time": round(avg_ret, 4),
        }, notes=(
            f"Ebbinghaus vs LRU gap: {gap:.3f}. "
            f"Learned S={learned_S:.3f} steps. "
            f"Mean retention at query: {avg_ret:.3f}."
        ), config=config)


if __name__ == "__main__":
    Exp65ForgettingCurveMimicry().execute()
