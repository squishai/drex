"""
Experiment 6.3 — Selective Forgetting Under Distribution Shift

Hypothesis: A controller can learn to evict domain-mismatched memories when input
distribution shifts, without explicit domain labels.

Setup:
  - Sequences have two phases.
  - Phase 1: domain A tokens (0-31). Memory fills.
  - Phase 2: domain B tokens (32-63). Selective gate can evict phase-1 memories.
  - Baseline: LRU eviction.
  - Selective: MLP scores (current_context_embedding, memory_entry_embedding) → evict?
  - Test: queries in phase 2 should prefer phase-2 memories.
  - SUPPORTED if selective_acc > lru_acc on phase-2 queries.
  - REFUTED if LRU matches selective.
  - INCONCLUSIVE otherwise.
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
DOMAIN_A_MAX  = 32    # tokens 0–31
DOMAIN_B_MIN  = 32    # tokens 32–63
SEQ_LEN       = 24
PHASE1_LEN    = SEQ_LEN // 2   # first half = domain A
PHASE2_LEN    = SEQ_LEN // 2   # second half = domain B
BATCH_SIZE    = 32
TRAIN_STEPS   = 1500
MEMORY_SLOTS  = 8
EVAL_BATCHES  = 200
LR            = 3e-4
DEVICE        = "cpu"


# ── Data generator ─────────────────────────────────────────────────────────────

def make_batch(batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
      seqs:       (B, SEQ_LEN) — phase1 domain A, phase2 domain B
      phase_mask: (B, SEQ_LEN) — 0=domain A, 1=domain B
      query_tok:  (B,) — a domain B token used as query key
      target:     (B,) — the domain B value associated with the query key
    """
    seqs       = torch.zeros(batch_size, SEQ_LEN, dtype=torch.long)
    phase_mask = torch.zeros(batch_size, SEQ_LEN, dtype=torch.long)
    query_tok  = torch.zeros(batch_size, dtype=torch.long)
    target     = torch.zeros(batch_size, dtype=torch.long)

    for b in range(batch_size):
        # Phase 1: domain A tokens
        seqs[b, :PHASE1_LEN] = torch.randint(0, DOMAIN_A_MAX, (PHASE1_LEN,))
        phase_mask[b, :PHASE1_LEN] = 0

        # Phase 2: domain B — plant key-value pair
        key   = torch.randint(DOMAIN_B_MIN, VOCAB_SIZE - 1, (1,)).item()
        value = torch.randint(DOMAIN_B_MIN, VOCAB_SIZE, (1,)).item()
        kv_pos = PHASE1_LEN

        seqs[b, kv_pos]     = key
        seqs[b, kv_pos + 1] = value
        phase_mask[b, PHASE1_LEN:] = 1

        # Fill rest of phase 2 with domain B noise
        for i in range(PHASE1_LEN, SEQ_LEN):
            if i in (kv_pos, kv_pos + 1):
                continue
            seqs[b, i] = torch.randint(DOMAIN_B_MIN, VOCAB_SIZE, (1,)).item()

        query_tok[b] = key
        target[b]    = value

    return seqs, phase_mask, query_tok, target


# ── LRU model ──────────────────────────────────────────────────────────────────

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


# ── Selective forgetting model ─────────────────────────────────────────────────

class SelectiveForgettingGate(nn.Module):
    """
    Given (current context embedding, memory entry embedding), output eviction score.
    Higher score = more likely to evict.
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(HIDDEN_DIM * 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, ctx_emb: torch.Tensor, mem_emb: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([ctx_emb, mem_emb], dim=-1))


class SelectiveModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed     = nn.Embedding(VOCAB_SIZE + 2, HIDDEN_DIM)
        self.gate      = SelectiveForgettingGate()
        self.read_head = nn.Sequential(
            nn.Linear(HIDDEN_DIM * 2, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, VOCAB_SIZE),
        )
        self._last_eviction_rate_phase1: float = 0.0

    def forward(self, seqs: torch.Tensor, query_tok: torch.Tensor,
                track_evictions: bool = False) -> torch.Tensor:
        B = seqs.size(0)
        out = []
        phase1_evictions: list[float] = []
        phase1_writes:    list[int]   = []

        for b in range(B):
            mem_embs: list[torch.Tensor] = []
            mem_phases: list[int] = []   # 0=domain A, 1=domain B
            p1_ev = 0; p1_wr = 0

            for t in range(SEQ_LEN - 1):
                tok = seqs[b, t].item()
                phase = 0 if tok < DOMAIN_A_MAX else 1
                emb   = self.embed(seqs[b, t].unsqueeze(0)).squeeze(0)

                # Context = current token embedding
                ctx_emb = emb

                if len(mem_embs) >= MEMORY_SLOTS:
                    # Score each memory entry for eviction
                    scores = []
                    for m_emb in mem_embs:
                        score = self.gate(ctx_emb.unsqueeze(0), m_emb.unsqueeze(0)).item()
                        scores.append(score)
                    evict_idx = max(range(len(scores)), key=lambda i: scores[i])

                    if track_evictions and mem_phases[evict_idx] == 0:
                        p1_ev += 1
                    mem_embs.pop(evict_idx)
                    mem_phases.pop(evict_idx)

                if track_evictions and phase == 0:
                    p1_wr += 1

                mem_embs.append(emb)
                mem_phases.append(phase)

            if track_evictions:
                phase1_evictions.append(p1_ev)
                phase1_writes.append(p1_wr)

            mem_summary = torch.stack(mem_embs).mean(0) if mem_embs else torch.zeros(HIDDEN_DIM)
            q_emb  = self.embed(query_tok[b].unsqueeze(0)).squeeze(0)
            logits = self.read_head(torch.cat([q_emb, mem_summary]).unsqueeze(0))
            out.append(logits)

        if track_evictions and phase1_writes:
            total_writes = sum(phase1_writes)
            total_evicts = sum(phase1_evictions)
            self._last_eviction_rate_phase1 = (
                total_evicts / total_writes if total_writes > 0 else 0.0
            )

        return torch.cat(out, dim=0)


# ── Evaluation ─────────────────────────────────────────────────────────────────

def evaluate_lru(model: LRUModel, n_batches: int) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for _ in range(n_batches):
            seqs, _, query_tok, target = make_batch(BATCH_SIZE)
            logits = model(seqs, query_tok)
            correct += (logits.argmax(-1) == target).sum().item()
            total   += BATCH_SIZE
    model.train()
    return correct / total


def evaluate_selective(model: SelectiveModel, n_batches: int) -> tuple[float, float]:
    model.eval()
    correct = total = 0
    eviction_rates = []
    with torch.no_grad():
        for _ in range(n_batches):
            seqs, _, query_tok, target = make_batch(BATCH_SIZE)
            logits = model(seqs, query_tok, track_evictions=True)
            correct += (logits.argmax(-1) == target).sum().item()
            total   += BATCH_SIZE
            eviction_rates.append(model._last_eviction_rate_phase1)
    model.train()
    avg_eviction_rate = sum(eviction_rates) / max(len(eviction_rates), 1)
    return correct / total, avg_eviction_rate


# ── Experiment ──────────────────────────────────────────────────────────────────

class Exp63SelectiveForgettingDistributionShift(Experiment):
    experiment_id = "exp_6_3"
    hypothesis = (
        "A controller can learn to evict domain-mismatched memories when input "
        "distribution shifts, without explicit domain labels."
    )

    def run(self) -> ExperimentResult:
        torch.manual_seed(42)

        config = {
            "hidden_dim": HIDDEN_DIM, "vocab_size": VOCAB_SIZE,
            "seq_len": SEQ_LEN, "batch_size": BATCH_SIZE,
            "train_steps": TRAIN_STEPS, "memory_slots": MEMORY_SLOTS,
            "phase1_len": PHASE1_LEN, "phase2_len": PHASE2_LEN,
        }

        # ── Train LRU baseline ────────────────────────────────────────────────
        print("  Training LRU baseline...")
        lru_model = LRUModel().to(DEVICE)
        opt_lru   = Adam(lru_model.parameters(), lr=LR)

        for step in range(TRAIN_STEPS):
            seqs, _, query_tok, target = make_batch(BATCH_SIZE)
            logits = lru_model(seqs, query_tok)
            loss   = F.cross_entropy(logits, target)
            opt_lru.zero_grad(); loss.backward(); opt_lru.step()
            if (step + 1) % 500 == 0:
                print(f"    LRU step {step+1}/{TRAIN_STEPS}  loss={loss.item():.3f}")

        # ── Train selective forgetting model ──────────────────────────────────
        print("  Training selective forgetting model...")
        sel_model = SelectiveModel().to(DEVICE)
        opt_sel   = Adam(sel_model.parameters(), lr=LR)

        for step in range(TRAIN_STEPS):
            seqs, _, query_tok, target = make_batch(BATCH_SIZE)
            logits = sel_model(seqs, query_tok)
            loss   = F.cross_entropy(logits, target)
            opt_sel.zero_grad(); loss.backward(); opt_sel.step()
            if (step + 1) % 500 == 0:
                print(f"    Selective step {step+1}/{TRAIN_STEPS}  loss={loss.item():.3f}")

        # ── Evaluate ──────────────────────────────────────────────────────────
        lru_phase2_acc = evaluate_lru(lru_model, EVAL_BATCHES)
        selective_phase2_acc, eviction_rate = evaluate_selective(sel_model, EVAL_BATCHES)

        print(f"  LRU phase-2 acc:           {lru_phase2_acc:.3f}")
        print(f"  Selective phase-2 acc:     {selective_phase2_acc:.3f}")
        print(f"  Selective eviction rate (phase-1 entries): {eviction_rate:.3f}")

        gap = selective_phase2_acc - lru_phase2_acc
        if selective_phase2_acc > lru_phase2_acc:
            outcome = OUTCOME_SUPPORTED
        elif lru_phase2_acc >= selective_phase2_acc:
            outcome = OUTCOME_REFUTED
        else:
            outcome = OUTCOME_INCONCLUSIVE

        return self.result(outcome, {
            "lru_phase2_acc":         round(lru_phase2_acc, 4),
            "selective_phase2_acc":   round(selective_phase2_acc, 4),
            "gap_selective_minus_lru": round(gap, 4),
            "selective_eviction_rate_for_phase1_entries": round(eviction_rate, 4),
        }, notes=(
            f"Selective vs LRU gap on phase-2 queries: {gap:.3f}. "
            f"Eviction rate for phase-1 entries: {eviction_rate:.3f}."
        ), config=config)


if __name__ == "__main__":
    Exp63SelectiveForgettingDistributionShift().execute()
