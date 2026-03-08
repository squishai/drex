"""
Experiment 7.3 — Controller Generalization Across Task Types

Hypothesis: A controller trained on factual QA generalizes its memory management
policies to reasoning tasks but not to generation tasks.

Setup:
  - Train a single memory controller on Task A (factual QA).
  - Freeze controller weights.
  - Evaluate zero-shot on Task B (reasoning: arithmetic combination).
  - Evaluate zero-shot on Task C (generation: next-token prediction).
  - SUPPORTED if factual->reasoning gap < 0.15 AND factual->generation gap > 0.20
  - REFUTED if both gaps are similar (within 0.05 of each other)
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

VOCAB_SIZE    = 64
HIDDEN_DIM    = 64
SEQ_LEN       = 24
MEMORY_SLOTS  = 8
BATCH_SIZE    = 32
TRAIN_STEPS   = 1500
EVAL_BATCHES  = 200
LR            = 3e-4
DEVICE        = "cpu"

# Numeric tokens live in [32, 48) — 16 values representing 0-15
NUM_OFFSET    = 32
NUM_RANGE     = 16


# ── Shared building blocks ────────────────────────────────────────────────────

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, HIDDEN_DIM)
        self.ff    = nn.Sequential(
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM * 2),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM * 2, HIDDEN_DIM),
        )
        self.norm = nn.LayerNorm(HIDDEN_DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embed(x)
        return self.norm(h + self.ff(h))


class WriteController(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, 1),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.gate(hidden).squeeze(-1)  # (B, L)


class ReadHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj  = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.out     = nn.Linear(HIDDEN_DIM, VOCAB_SIZE)
        self.query_e = nn.Embedding(VOCAB_SIZE, HIDDEN_DIM)

    def forward(self, query: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        q    = self.q_proj(self.query_e(query)).unsqueeze(1)
        sims = (q * memory).sum(-1) / (HIDDEN_DIM ** 0.5)
        w    = F.softmax(sims, dim=-1).unsqueeze(-1)
        return self.out((w * memory).sum(1))


def build_memory(hidden: torch.Tensor, ctrl: WriteController) -> torch.Tensor:
    logits  = ctrl(hidden)
    top_idx = logits.topk(MEMORY_SLOTS, dim=-1).indices
    return hidden.gather(1, top_idx.unsqueeze(-1).expand(-1, -1, HIDDEN_DIM))


# ── Task A: Factual QA ────────────────────────────────────────────────────────
# Plant a key-value pair; query is the key, answer is the value.

def make_factual_batch(batch_size: int):
    seq   = torch.randint(0, VOCAB_SIZE, (batch_size, SEQ_LEN))
    k_pos = torch.randint(0, SEQ_LEN // 2, (batch_size,))
    v_pos = torch.randint(SEQ_LEN // 2, SEQ_LEN, (batch_size,))
    keys  = torch.randint(0, VOCAB_SIZE // 2, (batch_size,))
    vals  = torch.randint(VOCAB_SIZE // 2, VOCAB_SIZE, (batch_size,))
    for b in range(batch_size):
        seq[b, k_pos[b]] = keys[b]
        seq[b, v_pos[b]] = vals[b]
    return seq, keys, vals


# ── Task B: Reasoning (arithmetic) ───────────────────────────────────────────
# Store two numeric tokens A and B in memory; query asks for their sum mod NUM_RANGE.

def make_reasoning_batch(batch_size: int):
    seq  = torch.randint(0, VOCAB_SIZE, (batch_size, SEQ_LEN))
    # Plant two numeric values
    pos_a = torch.randint(0, SEQ_LEN // 3, (batch_size,))
    pos_b = torch.randint(SEQ_LEN // 3, 2 * SEQ_LEN // 3, (batch_size,))
    val_a = torch.randint(0, NUM_RANGE, (batch_size,))
    val_b = torch.randint(0, NUM_RANGE, (batch_size,))
    for b in range(batch_size):
        seq[b, pos_a[b]] = NUM_OFFSET + val_a[b]
        seq[b, pos_b[b]] = NUM_OFFSET + val_b[b]
    # Query token = special "sum query" marker = token 1
    query = torch.ones(batch_size, dtype=torch.long)
    # Target = (A + B) mod NUM_RANGE, mapped back to vocab token
    target = ((val_a + val_b) % NUM_RANGE) + NUM_OFFSET
    return seq, query, target


# ── Task C: Generation (next-token) ──────────────────────────────────────────
# The next token in the sequence; memory provides long-range context.
# Evaluate as average log-prob of the last token given memory.

def make_generation_batch(batch_size: int):
    seq    = torch.randint(0, VOCAB_SIZE, (batch_size, SEQ_LEN))
    query  = seq[:, -2]          # second-to-last token as query
    target = seq[:, -1]          # last token is target
    return seq[:, :-1], query, target


# ── Train on Task A ───────────────────────────────────────────────────────────

def train_factual(enc: Encoder, ctrl: WriteController, read: ReadHead) -> None:
    opt = Adam(
        list(enc.parameters()) + list(ctrl.parameters()) + list(read.parameters()),
        lr=LR,
    )
    for _ in range(TRAIN_STEPS):
        seq, keys, vals = make_factual_batch(BATCH_SIZE)
        h    = enc(seq)
        mem  = build_memory(h, ctrl)
        out  = read(keys, mem)
        loss = F.cross_entropy(out, vals)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(enc.parameters()) + list(ctrl.parameters()) + list(read.parameters()), 1.0
        )
        opt.step()


# ── Evaluate helpers ──────────────────────────────────────────────────────────

def eval_factual(enc: Encoder, ctrl: WriteController, read: ReadHead) -> float:
    enc.eval(); ctrl.eval(); read.eval()
    correct = total = 0
    with torch.no_grad():
        for _ in range(EVAL_BATCHES):
            seq, keys, vals = make_factual_batch(BATCH_SIZE)
            h   = enc(seq)
            mem = build_memory(h, ctrl)
            out = read(keys, mem)
            correct += (out.argmax(-1) == vals).sum().item()
            total   += BATCH_SIZE
    enc.train(); ctrl.train(); read.train()
    return correct / total


def eval_reasoning(enc: Encoder, ctrl: WriteController, read: ReadHead) -> float:
    """Accuracy on arithmetic combination task using frozen controller."""
    enc.eval(); ctrl.eval(); read.eval()
    correct = total = 0
    with torch.no_grad():
        for _ in range(EVAL_BATCHES):
            seq, query, target = make_reasoning_batch(BATCH_SIZE)
            h   = enc(seq)
            mem = build_memory(h, ctrl)
            out = read(query, mem)
            correct += (out.argmax(-1) == target).sum().item()
            total   += BATCH_SIZE
    enc.train(); ctrl.train(); read.train()
    return correct / total


def eval_generation(enc: Encoder, ctrl: WriteController, read: ReadHead) -> float:
    """Average log-prob (fluency proxy) on generation task."""
    enc.eval(); ctrl.eval(); read.eval()
    total_logprob = 0.0
    count         = 0
    with torch.no_grad():
        for _ in range(EVAL_BATCHES):
            seq, query, target = make_generation_batch(BATCH_SIZE)
            h   = enc(seq)
            mem = build_memory(h, ctrl)
            out = read(query, mem)                    # (B, VOCAB_SIZE)
            lp  = F.log_softmax(out, dim=-1)
            total_logprob += lp.gather(1, target.unsqueeze(1)).squeeze(1).mean().item()
            count         += 1
    enc.train(); ctrl.train(); read.train()
    # Normalise to [0, 1]-ish by shifting from (-log(V), 0) to probability range
    avg_logprob = total_logprob / count
    # Convert to effective accuracy: exp(avg_logprob) as probability-like scalar
    import math
    return math.exp(avg_logprob)


# ── Experiment ────────────────────────────────────────────────────────────────

class Exp73ControllerGeneralization(Experiment):
    experiment_id = "exp_7_3"
    hypothesis = (
        "A controller trained on factual QA generalizes its memory management "
        "policies to reasoning tasks but not to generation tasks."
    )

    def run(self) -> ExperimentResult:
        torch.manual_seed(42)

        enc  = Encoder()
        ctrl = WriteController()
        read = ReadHead()

        print("  Training on factual QA...")
        train_factual(enc, ctrl, read)

        # Freeze controller
        for p in ctrl.parameters():
            p.requires_grad_(False)

        factual_acc    = eval_factual(enc, ctrl, read)
        reasoning_acc  = eval_reasoning(enc, ctrl, read)
        generation_acc = eval_generation(enc, ctrl, read)

        factual_to_reasoning_gap  = factual_acc - reasoning_acc
        factual_to_generation_gap = factual_acc - generation_acc

        print(f"  factual_acc={factual_acc:.3f}  reasoning_acc={reasoning_acc:.3f}  "
              f"generation_acc={generation_acc:.4f}")
        print(f"  reasoning_gap={factual_to_reasoning_gap:.3f}  "
              f"generation_gap={factual_to_generation_gap:.3f}")

        reasoning_transfers = factual_to_reasoning_gap < 0.15
        generation_fails    = factual_to_generation_gap > 0.20
        gaps_similar        = abs(factual_to_reasoning_gap - factual_to_generation_gap) < 0.05

        if reasoning_transfers and generation_fails:
            outcome = OUTCOME_SUPPORTED
        elif gaps_similar:
            outcome = OUTCOME_REFUTED
        else:
            outcome = OUTCOME_INCONCLUSIVE

        metrics = {
            "factual_acc":                factual_acc,
            "reasoning_acc":              reasoning_acc,
            "generation_acc":             generation_acc,
            "factual_to_reasoning_gap":   factual_to_reasoning_gap,
            "factual_to_generation_gap":  factual_to_generation_gap,
            "reasoning_transfers":        reasoning_transfers,
            "generation_fails":           generation_fails,
        }
        notes = (
            f"Reasoning gap={factual_to_reasoning_gap:.3f} "
            f"(threshold <0.15: {reasoning_transfers}). "
            f"Generation gap={factual_to_generation_gap:.3f} "
            f"(threshold >0.20: {generation_fails})."
        )
        return self.result(outcome, metrics, notes, config={
            "train_steps": TRAIN_STEPS,
            "memory_slots": MEMORY_SLOTS,
            "reasoning_gap_threshold": 0.15,
            "generation_gap_threshold": 0.20,
        })


if __name__ == "__main__":
    Exp73ControllerGeneralization().execute()
