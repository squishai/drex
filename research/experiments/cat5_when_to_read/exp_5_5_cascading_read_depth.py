"""
Experiment 5.5 — Cascading Read Depth

Hypothesis: Confidence-gated cascading retrieval matches full-depth retrieval
quality at significantly lower average compute cost.

Setup:
  - Three-tier memory, focus on efficiency
  - Cascading policy: query tier 1 only. If confidence > 0.7, stop.
    Else query tier 2. If still < 0.7, query tier 3.
  - Full-depth policy: always query all three tiers.
  - SUPPORTED if cascading_acc >= full_acc - 0.02 AND
                  avg_tiers_cascading < 2.0
  - REFUTED if cascading_acc < full_acc - 0.05
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

# ── Config ────────────────────────────────────────────────────────────────────

VOCAB_SIZE      = 64
HIDDEN_DIM      = 64
MEMORY_SLOTS    = 8    # slots per tier
N_TIERS         = 3
SEQ_LEN         = 24
BATCH_SIZE      = 32
TRAIN_STEPS     = 1500
LR              = 3e-4
DEVICE          = "cpu"
CONFIDENCE_STOP = 0.7  # max attention weight threshold


# ── Tier Memory ───────────────────────────────────────────────────────────────

class TierMemory(nn.Module):
    """Single-tier soft attention memory with a confidence score."""

    def __init__(self, tier_id: int):
        super().__init__()
        self.tier_id = tier_id
        self.q_proj  = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.k_proj  = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.v_proj  = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)

    def forward(
        self, query: torch.Tensor, memory: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            retrieved:   (B, H)
            confidence:  (B,) — max attention weight across slots
        """
        q  = self.q_proj(query).unsqueeze(1)                 # (B, 1, H)
        k  = self.k_proj(memory)                             # (B, M, H)
        v  = self.v_proj(memory)                             # (B, M, H)
        w  = F.softmax((q * k).sum(-1) / HIDDEN_DIM ** 0.5, dim=-1)  # (B, M)
        retrieved   = (w.unsqueeze(-1) * v).sum(1)           # (B, H)
        confidence  = w.max(dim=-1).values                   # (B,)
        return retrieved, confidence


# ── Shared Backbone ───────────────────────────────────────────────────────────

class SequenceEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, HIDDEN_DIM)
        self.gru   = nn.GRU(HIDDEN_DIM, HIDDEN_DIM, batch_first=True)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        """Returns final hidden state (B, H)."""
        emb = self.embed(seq)
        _, h_n = self.gru(emb)
        return h_n.squeeze(0)


# ── Full-Depth Model ──────────────────────────────────────────────────────────

class FullDepthModel(nn.Module):
    """Always queries all three tiers and concatenates/averages."""

    def __init__(self):
        super().__init__()
        self.encoder = SequenceEncoder()
        self.tiers   = nn.ModuleList([TierMemory(i) for i in range(N_TIERS)])
        self.merge   = nn.Linear(HIDDEN_DIM * (N_TIERS + 1), HIDDEN_DIM)
        self.head    = nn.Linear(HIDDEN_DIM, VOCAB_SIZE)

    def forward(
        self,
        seq: torch.Tensor,
        memories: list[torch.Tensor],
    ) -> tuple[torch.Tensor, float]:
        """Returns logits and average tiers queried (always N_TIERS)."""
        h = self.encoder(seq)
        retrieved_list = [h]
        for tier, mem in zip(self.tiers, memories):
            r, _ = tier(h, mem)
            retrieved_list.append(r)
        fused  = self.merge(torch.cat(retrieved_list, dim=-1))
        logits = self.head(F.relu(fused))
        return logits, float(N_TIERS)


# ── Cascading Model ───────────────────────────────────────────────────────────

class CascadingModel(nn.Module):
    """
    Queries tiers one by one. Stops when max attention weight > CONFIDENCE_STOP
    or all tiers exhausted.
    """

    def __init__(self):
        super().__init__()
        self.encoder   = SequenceEncoder()
        self.tiers     = nn.ModuleList([TierMemory(i) for i in range(N_TIERS)])
        self.merge     = nn.Linear(HIDDEN_DIM * 2, HIDDEN_DIM)
        self.head      = nn.Linear(HIDDEN_DIM, VOCAB_SIZE)

    def forward(
        self,
        seq: torch.Tensor,
        memories: list[torch.Tensor],
        training: bool = False,
    ) -> tuple[torch.Tensor, float]:
        """Returns logits and average tiers queried per item."""
        B = seq.size(0)
        h = self.encoder(seq)

        if training:
            # During training: always use all tiers for gradient flow
            accumulated = torch.zeros_like(h)
            for tier, mem in zip(self.tiers, memories):
                r, _ = tier(h, mem)
                accumulated = accumulated + r
            accumulated = accumulated / N_TIERS
            fused  = self.merge(torch.cat([h, accumulated], dim=-1))
            logits = self.head(F.relu(fused))
            return logits, float(N_TIERS)
        else:
            # Evaluation: true cascading with early stop
            accumulated    = torch.zeros_like(h)
            tiers_queried  = torch.zeros(B)
            done           = torch.zeros(B, dtype=torch.bool)

            for tier_idx, (tier, mem) in enumerate(zip(self.tiers, memories)):
                # Only query for items not yet done
                if done.all():
                    break
                r, conf = tier(h, mem)          # (B, H), (B,)

                accumulated = accumulated + r * (~done).float().unsqueeze(-1)
                tiers_queried += (~done).float()

                # Stop if confidence is high enough
                done = done | (conf > CONFIDENCE_STOP)

            # Normalise accumulated by actual tiers queried
            norm = tiers_queried.clamp(min=1).unsqueeze(-1)
            accumulated = accumulated / norm

            fused  = self.merge(torch.cat([h, accumulated], dim=-1))
            logits = self.head(F.relu(fused))
            avg_tiers = tiers_queried.mean().item()
            return logits, avg_tiers


# ── Task ──────────────────────────────────────────────────────────────────────

def make_tiered_task(batch_size: int):
    """
    Answer is stored in a randomly chosen tier (uniform over 3 tiers).
    The tier with the matching answer has a strong signal; others are noise.
    """
    B         = batch_size
    facts     = torch.randint(2, VOCAB_SIZE, (B,))
    answer_tier = torch.randint(0, N_TIERS, (B,))
    seq       = torch.randint(1, VOCAB_SIZE, (B, SEQ_LEN))

    memories = []
    for t in range(N_TIERS):
        mem = torch.randn(B, MEMORY_SLOTS, HIDDEN_DIM) * 0.1
        for b in range(B):
            if answer_tier[b].item() == t:
                mem[b, 0, :] = 0.0
                mem[b, 0, facts[b].item() % HIDDEN_DIM] = 2.0
        memories.append(mem)

    return seq, facts, memories


# ── Train / Eval ──────────────────────────────────────────────────────────────

def train_model(model: nn.Module) -> None:
    opt = Adam(model.parameters(), lr=LR)
    for _ in range(TRAIN_STEPS):
        seq, tgt, mems = make_tiered_task(BATCH_SIZE)
        if isinstance(model, CascadingModel):
            logits, _ = model(seq, mems, training=True)
        else:
            logits, _ = model(seq, mems)
        loss = F.cross_entropy(logits, tgt)
        opt.zero_grad(); loss.backward(); opt.step()


def eval_model(
    model: nn.Module, n_batches: int = 50
) -> tuple[float, float]:
    """Returns (accuracy, avg_tiers_queried)."""
    model.eval()
    correct = total = 0
    total_tiers = 0.0

    with torch.no_grad():
        for _ in range(n_batches):
            seq, tgt, mems = make_tiered_task(BATCH_SIZE)
            if isinstance(model, CascadingModel):
                logits, avg_t = model(seq, mems, training=False)
            else:
                logits, avg_t = model(seq, mems)
            preds = logits.argmax(-1)
            correct += (preds == tgt).sum().item()
            total   += tgt.size(0)
            total_tiers += avg_t

    model.train()
    return correct / total, total_tiers / n_batches


# ── Experiment ────────────────────────────────────────────────────────────────

class Exp55CascadingReadDepth(Experiment):
    experiment_id = "exp_5_5"
    hypothesis = (
        "Confidence-gated cascading retrieval matches full-depth retrieval "
        "quality at significantly lower average compute cost."
    )

    def run(self) -> ExperimentResult:
        torch.manual_seed(42)

        config = {
            "vocab_size":      VOCAB_SIZE,
            "hidden_dim":      HIDDEN_DIM,
            "memory_slots":    MEMORY_SLOTS,
            "n_tiers":         N_TIERS,
            "seq_len":         SEQ_LEN,
            "batch_size":      BATCH_SIZE,
            "train_steps":     TRAIN_STEPS,
            "confidence_stop": CONFIDENCE_STOP,
        }

        print("\n  Training full-depth model...")
        full_model = FullDepthModel().to(DEVICE)
        train_model(full_model)
        full_acc, full_tiers = eval_model(full_model)
        print(f"    full_acc={full_acc:.3f}  avg_tiers={full_tiers:.2f}")

        print("  Training cascading model...")
        casc_model = CascadingModel().to(DEVICE)
        train_model(casc_model)
        casc_acc, casc_tiers = eval_model(casc_model)
        print(f"    cascading_acc={casc_acc:.3f}  avg_tiers={casc_tiers:.2f}")

        compute_savings_pct = (full_tiers - casc_tiers) / full_tiers * 100.0

        metrics = {
            "full_acc":              round(full_acc, 4),
            "cascading_acc":         round(casc_acc, 4),
            "full_tiers_avg":        round(full_tiers, 3),
            "cascading_tiers_avg":   round(casc_tiers, 3),
            "compute_savings_pct":   round(compute_savings_pct, 2),
        }

        acc_drop = full_acc - casc_acc

        if casc_acc >= full_acc - 0.02 and casc_tiers < 2.0:
            outcome = OUTCOME_SUPPORTED
        elif acc_drop > 0.05:
            outcome = OUTCOME_REFUTED
        else:
            outcome = OUTCOME_INCONCLUSIVE

        notes = (
            f"Full-depth: acc={full_acc:.3f}, tiers={full_tiers:.2f}. "
            f"Cascading: acc={casc_acc:.3f}, tiers={casc_tiers:.2f}. "
            f"Acc drop={acc_drop:.4f}. "
            f"Compute savings={compute_savings_pct:.1f}%. "
            f"Threshold: acc_drop<=0.02 AND casc_tiers<2.0 for SUPPORTED."
        )

        return self.result(outcome, metrics, notes, config)


if __name__ == "__main__":
    Exp55CascadingReadDepth().execute()
