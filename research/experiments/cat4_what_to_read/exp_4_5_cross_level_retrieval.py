"""
Experiment 4.5 — Cross-Level Retrieval

Hypothesis: Simultaneous cross-tier retrieval achieves better recall than
sequential cascading retrieval.

Setup:
  - Three-tier memory:
      Tier 1 (fast):   4 slots, full 64-dim
      Tier 2 (medium): 8 slots, 32-dim compressed
      Tier 3 (slow):  16 slots, 16-dim compressed
  - Policy A (Sequential): query tier 1; if max attention weight < 0.6 query
    tier 2; if still < 0.6 query tier 3. Confidence = max attention weight.
  - Policy B (Simultaneous): query all three tiers at once, combine results
    weighted by per-tier confidence.
  - Task: associative recall with items distributed uniformly across tiers.
  - SUPPORTED if simultaneous_acc > sequential_acc + 0.02
  - REFUTED if sequential wins
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

HIDDEN_DIM      = 64
VOCAB_SIZE      = 64
SEQ_LEN         = 24
BATCH_SIZE      = 32
TRAIN_STEPS     = 1500
CONFIDENCE_THR  = 0.6
LR              = 3e-4
DEVICE          = "cpu"
EVAL_STEPS      = 300

# Tier specs: (n_slots, entry_dim)
TIER_SPECS = [
    (4,  64),   # Tier 1
    (8,  32),   # Tier 2
    (16, 16),   # Tier 3
]
TOTAL_SLOTS = sum(s for s, _ in TIER_SPECS)  # 28 total


# ── Data ───────────────────────────────────────────────────────────────────────

def make_batch(batch_size: int) -> tuple[
    torch.Tensor,
    list[torch.Tensor],
    torch.Tensor,
    torch.Tensor,
]:
    """
    Task: one relevant entry is placed in one of the three tiers uniformly.
    Returns (query_h, [tier1_mem, tier2_mem, tier3_mem], targets, tier_assignment).
      query_h:         (B, HIDDEN_DIM)
      tierX_mem:       (B, slots_X, dim_X)
      targets:         (B,)
      tier_assignment: (B,) in {0, 1, 2}
    """
    targets         = torch.randint(0, VOCAB_SIZE, (batch_size,))
    query_h         = F.normalize(torch.randn(batch_size, HIDDEN_DIM), dim=-1)
    tier_assignment = torch.randint(0, 3, (batch_size,))  # which tier holds relevant

    tier_mems = []
    for tier_idx, (n_slots, dim) in enumerate(TIER_SPECS):
        # project query down to this tier's dimension
        proj_q = query_h[:, :dim]  # take first `dim` dims as simple projection
        proj_q = F.normalize(proj_q, dim=-1)

        relevant = proj_q + torch.randn(batch_size, dim) * 0.05
        relevant = F.normalize(relevant, dim=-1)

        noise    = torch.randn(batch_size, n_slots - 1, dim)
        noise    = F.normalize(noise, dim=-1)

        full_mem = torch.cat([relevant.unsqueeze(1), noise], dim=1)  # (B, slots, dim)

        # only plant the relevant entry in items assigned to this tier;
        # otherwise slot 0 is also noise
        for b in range(batch_size):
            if tier_assignment[b].item() != tier_idx:
                full_mem[b, 0] = F.normalize(torch.randn(dim), dim=-1)

        tier_mems.append(full_mem.detach())

    return query_h.detach(), tier_mems, targets, tier_assignment


# ── Models ─────────────────────────────────────────────────────────────────────

class SequentialCascadeModel(nn.Module):
    """
    Query tier 1; if confidence < threshold, query tier 2; if still low, tier 3.
    Each tier has its own projection + attention.
    """

    def __init__(self) -> None:
        super().__init__()
        self.projs = nn.ModuleList([
            nn.Linear(HIDDEN_DIM, dim) for _, dim in TIER_SPECS
        ])
        # upproject retrieved back to HIDDEN_DIM
        self.up_projs = nn.ModuleList([
            nn.Linear(dim, HIDDEN_DIM) for _, dim in TIER_SPECS
        ])
        self.classifier = nn.Linear(HIDDEN_DIM, VOCAB_SIZE)

    def forward(
        self, query_h: torch.Tensor, tier_mems: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (logits, avg_tiers_queried)."""
        B = query_h.shape[0]
        result    = torch.zeros(B, HIDDEN_DIM, device=query_h.device)
        tiers_used = torch.zeros(B, device=query_h.device)

        # We must process without branching (differentiable cascade via gating)
        # confidence_so_far tracks whether we already have a confident result
        confident = torch.zeros(B, dtype=torch.bool, device=query_h.device)

        for tier_idx, (n_slots, dim) in enumerate(TIER_SPECS):
            q_proj   = self.projs[tier_idx](query_h)              # (B, dim)
            mem      = tier_mems[tier_idx]                         # (B, slots, dim)
            sims     = torch.bmm(mem, q_proj.unsqueeze(-1)).squeeze(-1) / (dim ** 0.5)
            weights  = F.softmax(sims, dim=-1)                     # (B, slots)
            max_conf = weights.max(dim=-1).values                  # (B,)
            retrieved = (weights.unsqueeze(-1) * mem).sum(1)       # (B, dim)
            retrieved_h = self.up_projs[tier_idx](retrieved)       # (B, H)

            # gate: only update result for items not yet confident
            gate = (~confident).float()                            # (B,)
            result = result + gate.unsqueeze(-1) * retrieved_h
            tiers_used = tiers_used + gate

            # update confident mask: now confident if max_conf >= threshold
            newly_confident = max_conf >= CONFIDENCE_THR
            confident = confident | newly_confident

        avg_tiers = tiers_used.mean()
        return self.classifier(result), avg_tiers


class SimultaneousRetrievalModel(nn.Module):
    """
    Query all three tiers simultaneously, weight results by per-tier confidence.
    """

    def __init__(self) -> None:
        super().__init__()
        self.projs = nn.ModuleList([
            nn.Linear(HIDDEN_DIM, dim) for _, dim in TIER_SPECS
        ])
        self.up_projs = nn.ModuleList([
            nn.Linear(dim, HIDDEN_DIM) for _, dim in TIER_SPECS
        ])
        self.classifier = nn.Linear(HIDDEN_DIM, VOCAB_SIZE)

    def forward(
        self, query_h: torch.Tensor, tier_mems: list[torch.Tensor]
    ) -> tuple[torch.Tensor, None]:
        B = query_h.shape[0]
        tier_results  = []
        tier_confs    = []

        for tier_idx, (n_slots, dim) in enumerate(TIER_SPECS):
            q_proj    = self.projs[tier_idx](query_h)
            mem       = tier_mems[tier_idx]
            sims      = torch.bmm(mem, q_proj.unsqueeze(-1)).squeeze(-1) / (dim ** 0.5)
            weights   = F.softmax(sims, dim=-1)
            max_conf  = weights.max(dim=-1).values                 # (B,)
            retrieved = (weights.unsqueeze(-1) * mem).sum(1)
            retrieved_h = self.up_projs[tier_idx](retrieved)       # (B, H)
            tier_results.append(retrieved_h)
            tier_confs.append(max_conf)

        confs  = torch.stack(tier_confs, dim=1)               # (B, 3)
        confs  = F.softmax(confs, dim=-1)                     # normalise across tiers
        result = sum(
            confs[:, i].unsqueeze(-1) * tier_results[i]
            for i in range(len(TIER_SPECS))
        )
        return self.classifier(result), None


# ── Training helpers ───────────────────────────────────────────────────────────

def train_and_eval_sequential(
    model: SequentialCascadeModel,
) -> tuple[float, float]:
    opt = Adam(model.parameters(), lr=LR)
    model.train()

    for step in range(TRAIN_STEPS):
        query_h, tier_mems, targets, _ = make_batch(BATCH_SIZE)
        logits, _ = model(query_h, tier_mems)
        loss = F.cross_entropy(logits, targets)
        opt.zero_grad(); loss.backward(); opt.step()
        if (step + 1) % 300 == 0:
            print(f"  [Sequential] step {step+1}/{TRAIN_STEPS}  loss={loss.item():.4f}")

    model.eval()
    correct = total = 0
    avg_tiers_total = 0.0
    with torch.no_grad():
        for _ in range(EVAL_STEPS):
            query_h, tier_mems, targets, _ = make_batch(BATCH_SIZE)
            logits, avg_t = model(query_h, tier_mems)
            preds = logits.argmax(dim=-1)
            correct += (preds == targets).sum().item()
            total   += BATCH_SIZE
            avg_tiers_total += avg_t.item()

    return correct / total, avg_tiers_total / EVAL_STEPS


def train_and_eval_simultaneous(
    model: SimultaneousRetrievalModel,
) -> float:
    opt = Adam(model.parameters(), lr=LR)
    model.train()

    for step in range(TRAIN_STEPS):
        query_h, tier_mems, targets, _ = make_batch(BATCH_SIZE)
        logits, _ = model(query_h, tier_mems)
        loss = F.cross_entropy(logits, targets)
        opt.zero_grad(); loss.backward(); opt.step()
        if (step + 1) % 300 == 0:
            print(f"  [Simultaneous] step {step+1}/{TRAIN_STEPS}  loss={loss.item():.4f}")

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for _ in range(EVAL_STEPS):
            query_h, tier_mems, targets, _ = make_batch(BATCH_SIZE)
            logits, _ = model(query_h, tier_mems)
            preds = logits.argmax(dim=-1)
            correct += (preds == targets).sum().item()
            total   += BATCH_SIZE

    return correct / total


# ── Experiment ─────────────────────────────────────────────────────────────────

class Exp45CrossLevelRetrieval(Experiment):
    experiment_id = "exp_4_5"
    hypothesis = (
        "Simultaneous cross-tier retrieval achieves better recall than sequential "
        "cascading retrieval."
    )

    def run(self) -> ExperimentResult:
        torch.manual_seed(42)
        print("\n  Training Sequential cascade model...")
        seq_model = SequentialCascadeModel().to(DEVICE)
        seq_acc, avg_tiers = train_and_eval_sequential(seq_model)

        torch.manual_seed(42)
        print("\n  Training Simultaneous model...")
        sim_model = SimultaneousRetrievalModel().to(DEVICE)
        sim_acc = train_and_eval_simultaneous(sim_model)

        gap = sim_acc - seq_acc
        print(f"\n  Sequential   — acc={seq_acc:.4f}  avg_tiers_queried={avg_tiers:.2f}")
        print(f"  Simultaneous — acc={sim_acc:.4f}")
        print(f"  Gap (sim - seq): {gap:+.4f}")

        if sim_acc > seq_acc + 0.02:
            outcome = OUTCOME_SUPPORTED
        elif seq_acc >= sim_acc:
            outcome = OUTCOME_REFUTED
        else:
            outcome = OUTCOME_INCONCLUSIVE

        metrics = {
            "sequential_acc":            round(seq_acc, 4),
            "simultaneous_acc":          round(sim_acc, 4),
            "gap_sim_minus_seq":         round(gap, 4),
            "avg_tiers_queried_sequential": round(avg_tiers, 3),
        }
        notes = (
            f"Simultaneous acc={sim_acc:.4f} vs Sequential acc={seq_acc:.4f}, "
            f"gap={gap:+.4f} (threshold +0.02 for SUPPORTED). "
            f"Sequential avg tiers queried={avg_tiers:.2f}."
        )
        config = {
            "hidden_dim":      HIDDEN_DIM,
            "vocab_size":      VOCAB_SIZE,
            "tier_specs":      TIER_SPECS,
            "confidence_thr":  CONFIDENCE_THR,
            "train_steps":     TRAIN_STEPS,
            "batch_size":      BATCH_SIZE,
        }
        return self.result(outcome, metrics, notes, config)


if __name__ == "__main__":
    Exp45CrossLevelRetrieval().execute()
