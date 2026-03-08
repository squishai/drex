"""
Experiment 7.5 — Controller Stability Under Scale

Hypothesis: A controller's learned policy trained at small scale does not
transfer directly to a larger model without additional fine-tuning.

Setup:
  - Train "small" model (HIDDEN_DIM=32, 1 encoder layer) + controller.
  - Extract controller weights.
  - Condition A: plug controller into "large" model (HIDDEN_DIM=64, 2 layers),
    freeze controller, train only model body.
  - Condition B: train controller fresh on large model.
  - Condition C: controller transferred to large model + fine-tune 500 steps.
  - SUPPORTED if zero_shot_transfer_acc < fresh_large_acc - 0.10
  - REFUTED if zero_shot_transfer within 0.05 of fresh_large_acc
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
HIDDEN_DIM     = 64      # large model hidden dim
HIDDEN_DIM_SM  = 32      # small model hidden dim
SEQ_LEN        = 24
MEMORY_SLOTS   = 8
BATCH_SIZE     = 32
TRAIN_STEPS    = 1500    # full training run
FINETUNE_STEPS = 500     # fine-tuning steps for condition C
EVAL_BATCHES   = 200
LR             = 3e-4
DEVICE         = "cpu"


# ── Models ────────────────────────────────────────────────────────────────────

class SmallEncoder(nn.Module):
    """Single-layer encoder with HIDDEN_DIM_SM."""
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, HIDDEN_DIM_SM)
        self.ff    = nn.Sequential(
            nn.Linear(HIDDEN_DIM_SM, HIDDEN_DIM_SM * 2),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM_SM * 2, HIDDEN_DIM_SM),
        )
        self.norm = nn.LayerNorm(HIDDEN_DIM_SM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embed(x)
        return self.norm(h + self.ff(h))


class LargeEncoder(nn.Module):
    """Two-layer encoder with HIDDEN_DIM."""
    def __init__(self):
        super().__init__()
        self.embed  = nn.Embedding(VOCAB_SIZE, HIDDEN_DIM)
        self.ff1    = nn.Sequential(
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM * 2),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM * 2, HIDDEN_DIM),
        )
        self.norm1  = nn.LayerNorm(HIDDEN_DIM)
        self.ff2    = nn.Sequential(
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM * 2),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM * 2, HIDDEN_DIM),
        )
        self.norm2  = nn.LayerNorm(HIDDEN_DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embed(x)
        h = self.norm1(h + self.ff1(h))
        h = self.norm2(h + self.ff2(h))
        return h


class WriteController(nn.Module):
    """Controller that maps hidden states to write gate logits."""
    def __init__(self, in_dim: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, 1),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.gate(hidden).squeeze(-1)


class ReadHead(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.q_proj  = nn.Linear(hidden_dim, hidden_dim)
        self.out     = nn.Linear(hidden_dim, VOCAB_SIZE)
        self.query_e = nn.Embedding(VOCAB_SIZE, hidden_dim)

    def forward(self, query: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        q    = self.q_proj(self.query_e(query)).unsqueeze(1)
        sims = (q * memory).sum(-1) / (memory.shape[-1] ** 0.5)
        w    = F.softmax(sims, dim=-1).unsqueeze(-1)
        return self.out((w * memory).sum(1))


# ── Data ──────────────────────────────────────────────────────────────────────

def make_batch(batch_size: int):
    seq   = torch.randint(0, VOCAB_SIZE, (batch_size, SEQ_LEN))
    k_pos = torch.randint(0, SEQ_LEN // 2, (batch_size,))
    v_pos = torch.randint(SEQ_LEN // 2, SEQ_LEN, (batch_size,))
    keys  = torch.randint(0, VOCAB_SIZE // 2, (batch_size,))
    vals  = torch.randint(VOCAB_SIZE // 2, VOCAB_SIZE, (batch_size,))
    for b in range(batch_size):
        seq[b, k_pos[b]] = keys[b]
        seq[b, v_pos[b]] = vals[b]
    return seq, keys, vals


def forward_and_loss(enc: nn.Module, ctrl: nn.Module, read: nn.Module,
                     seq: torch.Tensor, keys: torch.Tensor,
                     vals: torch.Tensor, hidden_dim: int) -> torch.Tensor:
    h        = enc(seq)
    logits_w = ctrl(h)
    top_idx  = logits_w.topk(MEMORY_SLOTS, dim=-1).indices
    mem      = h.gather(1, top_idx.unsqueeze(-1).expand(-1, -1, hidden_dim))
    out      = read(keys, mem)
    return F.cross_entropy(out, vals)


def evaluate(enc: nn.Module, ctrl: nn.Module, read: nn.Module,
             hidden_dim: int) -> float:
    enc.eval(); ctrl.eval(); read.eval()
    correct = total = 0
    with torch.no_grad():
        for _ in range(EVAL_BATCHES):
            seq, keys, vals = make_batch(BATCH_SIZE)
            h       = enc(seq)
            logits  = ctrl(h)
            top_idx = logits.topk(MEMORY_SLOTS, dim=-1).indices
            mem     = h.gather(1, top_idx.unsqueeze(-1).expand(-1, -1, hidden_dim))
            out     = read(keys, mem)
            correct += (out.argmax(-1) == vals).sum().item()
            total   += BATCH_SIZE
    enc.train(); ctrl.train(); read.train()
    return correct / total


def train(enc: nn.Module, ctrl: nn.Module, read: nn.Module,
          hidden_dim: int, steps: int,
          freeze_ctrl: bool = False) -> None:
    params = list(enc.parameters()) + list(read.parameters())
    if not freeze_ctrl:
        params += list(ctrl.parameters())
    opt = Adam(params, lr=LR)
    for _ in range(steps):
        seq, keys, vals = make_batch(BATCH_SIZE)
        loss = forward_and_loss(enc, ctrl, read, seq, keys, vals, hidden_dim)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()


# ── Experiment ────────────────────────────────────────────────────────────────

class Exp75ControllerStabilityUnderScale(Experiment):
    experiment_id = "exp_7_5"
    hypothesis = (
        "A controller's learned policy trained at small scale does not transfer "
        "directly to a larger model without additional fine-tuning."
    )

    def run(self) -> ExperimentResult:
        torch.manual_seed(42)

        # ── Step 1: Train small model ─────────────────────────────────────────
        print("  Training small model...")
        enc_sm   = SmallEncoder()
        ctrl_sm  = WriteController(HIDDEN_DIM_SM)
        read_sm  = ReadHead(HIDDEN_DIM_SM)
        train(enc_sm, ctrl_sm, read_sm, HIDDEN_DIM_SM, TRAIN_STEPS)
        small_acc = evaluate(enc_sm, ctrl_sm, read_sm, HIDDEN_DIM_SM)
        print(f"  small_acc={small_acc:.3f}")

        # ── Step 2A: Zero-shot transfer — project controller to large dim ─────
        # The small controller has in_dim=32. The large model outputs dim=64.
        # We create an adapter: project HIDDEN_DIM -> HIDDEN_DIM_SM, then run ctrl_sm.
        print("  Condition A: zero-shot transfer (frozen ctrl_sm + adapter)...")

        class TransferController(nn.Module):
            """Wraps a frozen small controller with a projection layer."""
            def __init__(self, frozen_ctrl: WriteController):
                super().__init__()
                self.proj = nn.Linear(HIDDEN_DIM, HIDDEN_DIM_SM, bias=False)
                self.ctrl = frozen_ctrl
                # Freeze original controller
                for p in self.ctrl.parameters():
                    p.requires_grad_(False)

            def forward(self, hidden: torch.Tensor) -> torch.Tensor:
                projected = self.proj(hidden)      # (B, L, HIDDEN_DIM_SM)
                return self.ctrl(projected)

        enc_lg_a   = LargeEncoder()
        ctrl_a     = TransferController(ctrl_sm)
        read_lg_a  = ReadHead(HIDDEN_DIM)
        # Train only body (enc + read + projection); ctrl_sm stays frozen
        train(enc_lg_a, ctrl_a, read_lg_a, HIDDEN_DIM, TRAIN_STEPS, freeze_ctrl=True)
        zero_shot_transfer_acc = evaluate(enc_lg_a, ctrl_a, read_lg_a, HIDDEN_DIM)
        print(f"  zero_shot_transfer_acc={zero_shot_transfer_acc:.3f}")

        # ── Step 2B: Train controller fresh on large model ────────────────────
        print("  Condition B: fresh controller on large model...")
        enc_lg_b  = LargeEncoder()
        ctrl_b    = WriteController(HIDDEN_DIM)
        read_lg_b = ReadHead(HIDDEN_DIM)
        train(enc_lg_b, ctrl_b, read_lg_b, HIDDEN_DIM, TRAIN_STEPS)
        fresh_large_acc = evaluate(enc_lg_b, ctrl_b, read_lg_b, HIDDEN_DIM)
        print(f"  fresh_large_acc={fresh_large_acc:.3f}")

        # ── Step 2C: Transfer + fine-tune ────────────────────────────────────
        print("  Condition C: transfer + fine-tune 500 steps...")
        enc_lg_c  = LargeEncoder()
        # Copy enc_lg_a weights as starting point
        enc_lg_c.load_state_dict(enc_lg_a.state_dict())
        ctrl_c = TransferController(WriteController(HIDDEN_DIM_SM))
        # Copy frozen ctrl from condition A
        ctrl_c.ctrl.load_state_dict(ctrl_sm.state_dict())
        ctrl_c.proj.load_state_dict(ctrl_a.proj.state_dict())
        read_lg_c = ReadHead(HIDDEN_DIM)
        read_lg_c.load_state_dict(read_lg_a.state_dict())
        # Unfreeze ctrl for fine-tuning
        for p in ctrl_c.ctrl.parameters():
            p.requires_grad_(True)
        train(enc_lg_c, ctrl_c, read_lg_c, HIDDEN_DIM, FINETUNE_STEPS, freeze_ctrl=False)
        finetuned_transfer_acc = evaluate(enc_lg_c, ctrl_c, read_lg_c, HIDDEN_DIM)
        print(f"  finetuned_transfer_acc={finetuned_transfer_acc:.3f}")

        transfer_gap = fresh_large_acc - zero_shot_transfer_acc

        if zero_shot_transfer_acc < fresh_large_acc - 0.10:
            outcome = OUTCOME_SUPPORTED
        elif abs(zero_shot_transfer_acc - fresh_large_acc) <= 0.05:
            outcome = OUTCOME_REFUTED
        else:
            outcome = OUTCOME_INCONCLUSIVE

        metrics = {
            "small_acc":               small_acc,
            "zero_shot_transfer_acc":  zero_shot_transfer_acc,
            "fresh_large_acc":         fresh_large_acc,
            "finetuned_transfer_acc":  finetuned_transfer_acc,
            "transfer_gap":            transfer_gap,
        }
        notes = (
            f"Transfer gap (fresh - zero_shot): {transfer_gap:.3f}. "
            f"Supported threshold: gap > 0.10. "
            f"Refuted threshold: gap <= 0.05."
        )
        return self.result(outcome, metrics, notes, config={
            "train_steps":     TRAIN_STEPS,
            "finetune_steps":  FINETUNE_STEPS,
            "hidden_dim_sm":   HIDDEN_DIM_SM,
            "hidden_dim_lg":   HIDDEN_DIM,
            "memory_slots":    MEMORY_SLOTS,
        })


if __name__ == "__main__":
    Exp75ControllerStabilityUnderScale().execute()
