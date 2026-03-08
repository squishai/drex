"""
Experiment 6.6 — Controller Catastrophic Forgetting

Hypothesis: The memory controller (as a neural network) suffers measurable
catastrophic forgetting of its learned policies when exposed to a new domain.

This is a meta-level problem: the controller is itself a model that can forget.
If the controller's write/read policies collapse when the data distribution shifts,
then the memory system becomes unreliable in streaming, multi-domain settings.

Setup:
  - Domain A: associative recall on numbers (entity → numeric attribute)
  - Domain B: associative recall on words (entity → lexical attribute)
  - Phase 1: train controller on domain A, measure domain A performance
  - Phase 2: continue training on domain B (no domain A data)
  - Phase 3: re-test on domain A (measure forgetting)
  - Measure: domain A accuracy before and after domain B training
  - Compare: standard fine-tuning vs. EWC-style regularization (elastic weight consolidation)
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from copy import deepcopy

from experiments.base import (
    Experiment, ExperimentResult,
    OUTCOME_SUPPORTED, OUTCOME_REFUTED, OUTCOME_INCONCLUSIVE,
)

# ── Config ────────────────────────────────────────────────────────────────────

VOCAB_A       = 32    # domain A: numeric-like attributes (0..31)
VOCAB_B       = 32    # domain B: lexical-like attributes (32..63)
NUM_ENTITIES  = 32
HIDDEN_DIM    = 64
MEMORY_SLOTS  = 8
SEQ_LEN       = 16
PHASE1_STEPS  = 1500
PHASE2_STEPS  = 1500
EVAL_BATCHES  = 200
BATCH_SIZE    = 32
LR            = 3e-4
FORGET_THRESHOLD = 0.15   # acc drop > this = significant forgetting
EWC_LAMBDA    = 1.0
DEVICE        = "cpu"


# ── Controller model ──────────────────────────────────────────────────────────

class Controller(nn.Module):
    def __init__(self):
        super().__init__()
        full_vocab = VOCAB_A + VOCAB_B + NUM_ENTITIES + 2
        self.embed     = nn.Embedding(full_vocab, HIDDEN_DIM)
        self.write_gate = nn.Sequential(
            nn.Linear(HIDDEN_DIM, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid())
        self.read_head = nn.Sequential(
            nn.Linear(HIDDEN_DIM * 2, HIDDEN_DIM), nn.ReLU(),
            nn.Linear(HIDDEN_DIM, full_vocab))

    def forward(self, seq: torch.Tensor,
                query: torch.Tensor) -> torch.Tensor:
        h    = self.embed(seq)                         # (B, L, H)
        gate = self.write_gate(h).squeeze(-1)          # (B, L)
        soft_gate = gate

        # write: attend over gated hidden states
        written = h * gate.unsqueeze(-1)
        mem_pooled = written.sum(1) / (soft_gate.sum(1, keepdim=True) + 1e-8)  # (B, H)

        q_h  = self.embed(query)                       # (B, H)
        merged = torch.cat([q_h, mem_pooled], dim=-1)
        return self.read_head(merged)


# ── Data generators ───────────────────────────────────────────────────────────

def make_batch_domain(batch_size: int, domain: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate associative recall sequences for domain A or B."""
    offset = 0 if domain == "A" else VOCAB_A
    seq    = torch.zeros(batch_size, SEQ_LEN, dtype=torch.long)
    query  = torch.zeros(batch_size, dtype=torch.long)
    target = torch.zeros(batch_size, dtype=torch.long)

    for b in range(batch_size):
        entity_id  = torch.randint(0, NUM_ENTITIES, (1,)).item()
        attr_id    = torch.randint(0, VOCAB_A, (1,)).item() + offset
        entity_tok = VOCAB_A + VOCAB_B + entity_id

        seq[b, 0] = entity_tok
        seq[b, 1] = attr_id
        # fill rest with noise
        seq[b, 2:] = torch.randint(2, VOCAB_A + VOCAB_B, (SEQ_LEN - 2,))

        query[b]  = entity_tok
        target[b] = attr_id

    return seq, query, target


def evaluate(model: Controller, domain: str) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for _ in range(EVAL_BATCHES):
            seq, q, t = make_batch_domain(BATCH_SIZE, domain)
            logits = model(seq, q)
            correct += (logits.argmax(-1) == t).sum().item()
            total   += BATCH_SIZE
    model.train()
    return correct / total


# ── EWC ───────────────────────────────────────────────────────────────────────

def compute_fisher(model: Controller, domain: str, n_batches: int = 50) -> dict:
    """Approximate diagonal Fisher information."""
    fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
    model.eval()
    for _ in range(n_batches):
        seq, q, t = make_batch_domain(BATCH_SIZE, domain)
        logits = model(seq, q)
        loss   = F.cross_entropy(logits, t)
        model.zero_grad()
        loss.backward()
        for n, p in model.named_parameters():
            if p.grad is not None:
                fisher[n] += p.grad.data ** 2
    for n in fisher:
        fisher[n] /= n_batches
    model.train()
    return fisher


def ewc_penalty(model: Controller, fisher: dict, star_params: dict) -> torch.Tensor:
    penalty = torch.tensor(0.0)
    for n, p in model.named_parameters():
        if n in fisher:
            penalty = penalty + (fisher[n] * (p - star_params[n]) ** 2).sum()
    return penalty


# ── Experiment ────────────────────────────────────────────────────────────────

class Exp66ControllerCatastrophicForgetting(Experiment):
    experiment_id = "exp_6_6"
    hypothesis = (
        "The memory controller suffers measurable catastrophic forgetting of its "
        "learned policies when fine-tuned on a new domain, absent explicit "
        "anti-forgetting mechanisms."
    )

    def run(self) -> ExperimentResult:
        torch.manual_seed(42)

        # ── Standard fine-tuning ──────────────────────────────────────────────
        model_std = Controller().to(DEVICE)
        opt_std   = Adam(model_std.parameters(), lr=LR)

        # phase 1
        for _ in range(PHASE1_STEPS):
            seq, q, t = make_batch_domain(BATCH_SIZE, "A")
            loss = F.cross_entropy(model_std(seq, q), t)
            opt_std.zero_grad(); loss.backward(); opt_std.step()
        acc_a_before = evaluate(model_std, "A")
        print(f"  Standard — domain A acc before B training: {acc_a_before:.3f}")

        # phase 2
        for _ in range(PHASE2_STEPS):
            seq, q, t = make_batch_domain(BATCH_SIZE, "B")
            loss = F.cross_entropy(model_std(seq, q), t)
            opt_std.zero_grad(); loss.backward(); opt_std.step()
        acc_a_after_std = evaluate(model_std, "A")
        acc_b_std       = evaluate(model_std, "B")
        print(f"  Standard — domain A acc after B training:  {acc_a_after_std:.3f}")
        print(f"  Standard — domain B acc:                   {acc_b_std:.3f}")

        # ── EWC fine-tuning ───────────────────────────────────────────────────
        model_ewc = Controller().to(DEVICE)
        opt_ewc   = Adam(model_ewc.parameters(), lr=LR)

        for _ in range(PHASE1_STEPS):
            seq, q, t = make_batch_domain(BATCH_SIZE, "A")
            loss = F.cross_entropy(model_ewc(seq, q), t)
            opt_ewc.zero_grad(); loss.backward(); opt_ewc.step()

        fisher     = compute_fisher(model_ewc, "A")
        star_params = {n: p.data.clone() for n, p in model_ewc.named_parameters()}

        for _ in range(PHASE2_STEPS):
            seq, q, t = make_batch_domain(BATCH_SIZE, "B")
            task_loss = F.cross_entropy(model_ewc(seq, q), t)
            ewc_loss  = EWC_LAMBDA * ewc_penalty(model_ewc, fisher, star_params)
            loss = task_loss + ewc_loss
            opt_ewc.zero_grad(); loss.backward(); opt_ewc.step()

        acc_a_after_ewc = evaluate(model_ewc, "A")
        acc_b_ewc       = evaluate(model_ewc, "B")
        print(f"  EWC     — domain A acc after B training:   {acc_a_after_ewc:.3f}")
        print(f"  EWC     — domain B acc:                    {acc_b_ewc:.3f}")

        forgetting_std = acc_a_before - acc_a_after_std
        forgetting_ewc = acc_a_before - acc_a_after_ewc
        forgetting_significant = forgetting_std > FORGET_THRESHOLD

        if forgetting_significant and forgetting_ewc < forgetting_std * 0.5:
            outcome = OUTCOME_SUPPORTED  # catastrophic forgetting + EWC mitigates
        elif forgetting_significant:
            outcome = OUTCOME_INCONCLUSIVE
        else:
            outcome = OUTCOME_REFUTED

        return self.result(outcome, {
            "acc_a_before":      acc_a_before,
            "acc_a_after_std":   acc_a_after_std,
            "acc_a_after_ewc":   acc_a_after_ewc,
            "acc_b_std":         acc_b_std,
            "acc_b_ewc":         acc_b_ewc,
            "forgetting_std":    forgetting_std,
            "forgetting_ewc":    forgetting_ewc,
            "forgetting_reduction_pct": (
                (forgetting_std - forgetting_ewc) / max(forgetting_std, 1e-8) * 100
            ),
        }, notes=(
            f"Standard forgetting: {forgetting_std:.3f}. "
            f"EWC forgetting: {forgetting_ewc:.3f}. "
            f"Significant: {forgetting_significant}."
        ), config={"phase1": PHASE1_STEPS, "phase2": PHASE2_STEPS,
                   "ewc_lambda": EWC_LAMBDA})


if __name__ == "__main__":
    Exp66ControllerCatastrophicForgetting().execute()
