"""
Experiment 5.4 — Read vs Recompute

Hypothesis: A controller can learn to prefer recomputation for cheap
information and retrieval for expensive information.

Setup:
  - Type A (cheap): deterministic function of current token (embedding directly)
  - Type B (expensive): requires remembering a token from 20+ positions ago
  - Controller chooses per-query: (a) recompute via linear layer,
    (b) read from memory
  - Trained on mixed tasks with both types
  - SUPPORTED if retrieval_rate_typeA < 0.3 AND retrieval_rate_typeB > 0.6
  - REFUTED if rates are similar (< 0.1 difference)
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

VOCAB_SIZE     = 64
HIDDEN_DIM     = 64
MEMORY_SLOTS   = 8
SEQ_LEN        = 24
BATCH_SIZE     = 32
TRAIN_STEPS    = 1500
LR             = 3e-4
DEVICE         = "cpu"
DISTANT_OFFSET = 20   # Type B: token from this many positions ago


# ── Memory ────────────────────────────────────────────────────────────────────

class SoftAttentionMemory(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.k_proj = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.v_proj = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)

    def forward(self, query: torch.Tensor,
                memory: torch.Tensor) -> torch.Tensor:
        q = self.q_proj(query).unsqueeze(1)
        k = self.k_proj(memory)
        v = self.v_proj(memory)
        w = F.softmax((q * k).sum(-1) / HIDDEN_DIM ** 0.5, dim=-1).unsqueeze(-1)
        return (w * v).sum(1)


# ── Read-vs-Recompute Controller ──────────────────────────────────────────────

class ReadRecomputeController(nn.Module):
    """
    For each query, decides: recompute (linear from current hidden) or
    retrieve (soft attention over memory). Decision is a soft gate in [0,1]
    where 1 = retrieve, 0 = recompute.
    """

    def __init__(self):
        super().__init__()
        self.embed       = nn.Embedding(VOCAB_SIZE, HIDDEN_DIM)
        self.gru         = nn.GRUCell(HIDDEN_DIM, HIDDEN_DIM)

        # Recompute path: single linear layer from current hidden state
        self.recompute   = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)

        # Retrieve path: soft attention over memory
        self.memory_mod  = SoftAttentionMemory()

        # Controller: decides retrieve vs recompute
        self.controller  = nn.Sequential(
            nn.Linear(HIDDEN_DIM, 32), nn.ReLU(),
            nn.Linear(32, 1), nn.Sigmoid(),
        )

        # Output head
        self.head = nn.Linear(HIDDEN_DIM, VOCAB_SIZE)

    def forward(
        self,
        seq: torch.Tensor,       # (B, L)
        memory: torch.Tensor,    # (B, M, H)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            logits:   (B, V) final prediction logits
            gate_sig: (B,)   soft gate value (1 = retrieve, 0 = recompute)
        """
        B, L = seq.shape
        h = torch.zeros(B, HIDDEN_DIM, device=seq.device)

        for t in range(L):
            x_t = self.embed(seq[:, t])
            h   = self.gru(x_t, h)

        # At end of sequence, make routing decision
        gate = self.controller(h).squeeze(-1)   # (B,) soft 0..1

        recomp_out   = F.relu(self.recompute(h))           # (B, H)
        retrieved    = self.memory_mod(h, memory)           # (B, H)

        # Soft blend: gate * retrieve + (1 - gate) * recompute
        fused = gate.unsqueeze(-1) * retrieved + (1 - gate).unsqueeze(-1) * recomp_out

        logits = self.head(fused)
        return logits, gate


# ── Task Generators ───────────────────────────────────────────────────────────

def make_type_a_batch(batch_size: int):
    """
    Type A — cheap: answer is a deterministic function of the LAST token.
    target = (last_token * 3 + 7) % VOCAB_SIZE
    No memory needed.
    """
    B   = batch_size
    seq = torch.randint(1, VOCAB_SIZE, (B, SEQ_LEN))
    tgt = (seq[:, -1] * 3 + 7) % VOCAB_SIZE
    # Memory contains irrelevant noise
    memory = torch.randn(B, MEMORY_SLOTS, HIDDEN_DIM) * 0.1
    return seq, tgt, memory


def make_type_b_batch(batch_size: int):
    """
    Type B — expensive: answer is a function of the token at position
    SEQ_LEN - DISTANT_OFFSET (20+ steps ago). Memory slot 0 encodes it.
    """
    B      = batch_size
    seq    = torch.randint(1, VOCAB_SIZE, (B, SEQ_LEN))
    anchor = seq[:, SEQ_LEN - DISTANT_OFFSET]             # token from far back
    tgt    = (anchor * 5 + 3) % VOCAB_SIZE
    # Store the answer in memory slot 0
    memory = torch.randn(B, MEMORY_SLOTS, HIDDEN_DIM) * 0.1
    for b in range(B):
        memory[b, 0, :] = 0.0
        memory[b, 0, tgt[b].item() % HIDDEN_DIM] = 2.0
    return seq, tgt, memory


# ── Train / Eval ──────────────────────────────────────────────────────────────

def train_controller() -> ReadRecomputeController:
    model = ReadRecomputeController().to(DEVICE)
    opt   = Adam(model.parameters(), lr=LR)

    for step in range(TRAIN_STEPS):
        # Mixed batch: half type A, half type B
        half = BATCH_SIZE // 2

        seq_a, tgt_a, mem_a = make_type_a_batch(half)
        seq_b, tgt_b, mem_b = make_type_b_batch(half)

        seq_a, tgt_a, mem_a = seq_a.to(DEVICE), tgt_a.to(DEVICE), mem_a.to(DEVICE)
        seq_b, tgt_b, mem_b = seq_b.to(DEVICE), tgt_b.to(DEVICE), mem_b.to(DEVICE)

        logits_a, gate_a = model(seq_a, mem_a)
        logits_b, gate_b = model(seq_b, mem_b)

        loss_a = F.cross_entropy(logits_a, tgt_a)
        loss_b = F.cross_entropy(logits_b, tgt_b)
        loss   = (loss_a + loss_b) / 2

        opt.zero_grad(); loss.backward(); opt.step()

    return model


def eval_controller(
    model: ReadRecomputeController, n_batches: int = 50
) -> tuple[float, float, float, float]:
    """Returns (retrieval_rate_A, retrieval_rate_B, acc_A, acc_B)."""
    model.eval()
    half = BATCH_SIZE // 2

    gate_a_total = gate_b_total = 0.0
    correct_a = correct_b = total = 0

    with torch.no_grad():
        for _ in range(n_batches):
            seq_a, tgt_a, mem_a = make_type_a_batch(half)
            seq_b, tgt_b, mem_b = make_type_b_batch(half)

            logits_a, gate_a = model(seq_a, mem_a)
            logits_b, gate_b = model(seq_b, mem_b)

            gate_a_total += gate_a.mean().item()
            gate_b_total += gate_b.mean().item()

            correct_a += (logits_a.argmax(-1) == tgt_a).sum().item()
            correct_b += (logits_b.argmax(-1) == tgt_b).sum().item()
            total     += half

    model.train()
    return (
        gate_a_total / n_batches,
        gate_b_total / n_batches,
        correct_a / total,
        correct_b / total,
    )


# ── Experiment ────────────────────────────────────────────────────────────────

class Exp54ReadVsRecompute(Experiment):
    experiment_id = "exp_5_4"
    hypothesis = (
        "A controller can learn to prefer recomputation for cheap information "
        "and retrieval for expensive information."
    )

    def run(self) -> ExperimentResult:
        torch.manual_seed(42)

        config = {
            "vocab_size":     VOCAB_SIZE,
            "hidden_dim":     HIDDEN_DIM,
            "memory_slots":   MEMORY_SLOTS,
            "seq_len":        SEQ_LEN,
            "batch_size":     BATCH_SIZE,
            "train_steps":    TRAIN_STEPS,
            "distant_offset": DISTANT_OFFSET,
        }

        print("\n  Training read-vs-recompute controller...")
        model = train_controller()

        ret_a, ret_b, acc_a, acc_b = eval_controller(model)
        rate_diff = ret_b - ret_a

        print(f"    retrieval_rate_typeA={ret_a:.3f}  retrieval_rate_typeB={ret_b:.3f}")
        print(f"    acc_typeA={acc_a:.3f}  acc_typeB={acc_b:.3f}")
        print(f"    rate_diff (B-A): {rate_diff:.3f}")

        metrics = {
            "retrieval_rate_typeA": round(ret_a, 4),
            "retrieval_rate_typeB": round(ret_b, 4),
            "typeA_acc":            round(acc_a, 4),
            "typeB_acc":            round(acc_b, 4),
            "rate_diff":            round(rate_diff, 4),
        }

        if ret_a < 0.3 and ret_b > 0.6:
            outcome = OUTCOME_SUPPORTED
        elif rate_diff < 0.1:
            outcome = OUTCOME_REFUTED
        else:
            outcome = OUTCOME_INCONCLUSIVE

        notes = (
            f"Type A (cheap/recompute): retrieval_rate={ret_a:.3f}, acc={acc_a:.3f}. "
            f"Type B (expensive/retrieve): retrieval_rate={ret_b:.3f}, acc={acc_b:.3f}. "
            f"Rate difference (B-A)={rate_diff:.3f}. "
            f"Threshold: A<0.3 AND B>0.6 for SUPPORTED."
        )

        return self.result(outcome, metrics, notes, config)


if __name__ == "__main__":
    Exp54ReadVsRecompute().execute()
