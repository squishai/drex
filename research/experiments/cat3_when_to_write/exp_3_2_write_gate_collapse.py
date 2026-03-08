"""
Experiment 3.2 — Write Gate Collapse Detection

Hypothesis: A learned write gate trained without explicit anti-collapse objectives
will learn to never write (or always write) within N training steps on standard tasks.

This is the most dangerous early failure mode. An always-zero gate has zero gate loss
but produces a useless memory. An always-one gate is equivalent to no gate at all.

Setup:
  - Train a memory controller with a learned binary write gate
  - Track write activity (fraction of tokens written) at each training step
  - Run under four training regimes:
    (A) No anti-collapse signal — expect collapse
    (B) Entropy regularization on gate
    (C) Write quality reconstruction loss
    (D) Task loss with memory dropout (forces memory to be used)
  - Measure: write rate trajectory, final write rate, task performance
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
SEQ_LEN       = 24
HIDDEN_DIM    = 48
MEMORY_SLOTS  = 6
TRAIN_STEPS   = 1000
BATCH_SIZE    = 32
LR            = 3e-4
LOG_EVERY     = 50
COLLAPSE_THRESHOLD = 0.05   # write rate below this = collapsed to zero
ALWAYS_WRITE_THRESHOLD = 0.95  # write rate above this = collapsed to always
DEVICE        = "cpu"


# ── Model ─────────────────────────────────────────────────────────────────────

class EncoderBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, HIDDEN_DIM)
        self.ff    = nn.Sequential(nn.Linear(HIDDEN_DIM, HIDDEN_DIM * 2),
                                    nn.ReLU(), nn.Linear(HIDDEN_DIM * 2, HIDDEN_DIM))
        self.norm  = nn.LayerNorm(HIDDEN_DIM)

    def forward(self, x):
        h = self.embed(x)
        return self.norm(h + self.ff(h))


class WriteGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM // 2), nn.ReLU(),
            nn.Linear(HIDDEN_DIM // 2, 1),
        )

    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns soft_gate (B,L,1) and hard_mask (B,L) via straight-through."""
        logits = self.gate(hidden)           # (B, L, 1)
        soft   = torch.sigmoid(logits)       # (B, L, 1)
        hard   = (soft > 0.5).float()
        # straight-through: backprop through soft, use hard for forward
        hard_st = soft + (hard - soft).detach()
        return hard_st.squeeze(-1), soft.squeeze(-1)


class MemoryHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(HIDDEN_DIM, VOCAB_SIZE)

    def forward(self, hidden: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        """Pool written tokens and classify next token."""
        weighted = hidden * gate.unsqueeze(-1)    # (B, L, H)
        pooled   = weighted.sum(1) / (gate.sum(1, keepdim=True) + 1e-8)
        return self.proj(pooled)


def make_batch(batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    seq = torch.randint(0, VOCAB_SIZE, (batch_size, SEQ_LEN))
    target = seq[:, -1].clone()
    seq[:, -1] = 0   # mask out the target token
    return seq, target


# ── Training Objectives ───────────────────────────────────────────────────────

def regime_a_no_signal(task_loss, soft_gate, hidden, gate_hard, recon_head=None):
    """No anti-collapse signal — gate has no incentive to write."""
    return task_loss


def regime_b_entropy(task_loss, soft_gate, hidden, gate_hard, recon_head=None):
    """Entropy regularization: encourage gate to be uncertain (not collapsed)."""
    p = soft_gate.mean()
    entropy = -(p * (p + 1e-8).log() + (1-p) * (1-p+1e-8).log())
    return task_loss - 0.1 * entropy   # maximize entropy = penalize collapse


def regime_c_reconstruction(task_loss, soft_gate, hidden, gate_hard, recon_head):
    """Write quality loss: written tokens should be reconstructable."""
    written = hidden * gate_hard.unsqueeze(-1)
    recon = recon_head(written)
    recon_loss = F.mse_loss(recon, hidden.detach())
    return task_loss + 0.1 * recon_loss


def regime_d_dropout(task_loss, soft_gate, hidden, gate_hard, recon_head=None):
    """Memory dropout: randomly zero 50% of gate to force memory reliance."""
    # signal that zero gate → bad task loss
    # implemented via add constant push toward write
    gate_mean = soft_gate.mean()
    penalty = F.relu(0.3 - gate_mean)   # penalize if write rate < 0.3
    return task_loss + penalty


REGIMES = {
    "A_no_signal":     (regime_a_no_signal, False),
    "B_entropy":       (regime_b_entropy, False),
    "C_reconstruction":(regime_c_reconstruction, True),
    "D_penalty":       (regime_d_dropout, False),
}


def train_and_monitor(regime_name: str, loss_fn, needs_recon: bool) -> dict:
    enc      = EncoderBlock().to(DEVICE)
    gate_net = WriteGate().to(DEVICE)
    head     = MemoryHead().to(DEVICE)
    recon_head = nn.Linear(HIDDEN_DIM, HIDDEN_DIM).to(DEVICE) if needs_recon else None

    params = list(enc.parameters()) + list(gate_net.parameters()) + list(head.parameters())
    if recon_head:
        params += list(recon_head.parameters())
    opt = Adam(params, lr=LR)

    write_rates = []

    for step in range(TRAIN_STEPS):
        seq, target = make_batch(BATCH_SIZE)
        seq, target = seq.to(DEVICE), target.to(DEVICE)

        hidden    = enc(seq)
        gate_hard, soft_gate = gate_net(hidden)
        logits    = head(hidden, gate_hard)
        task_loss = F.cross_entropy(logits, target)

        loss = loss_fn(task_loss, soft_gate, hidden, gate_hard, recon_head)
        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % LOG_EVERY == 0:
            write_rate = (gate_hard > 0.5).float().mean().item()
            write_rates.append(write_rate)

    final_rate = write_rates[-1]
    collapsed_zero   = final_rate < COLLAPSE_THRESHOLD
    collapsed_always = final_rate > ALWAYS_WRITE_THRESHOLD

    print(f"  {regime_name:20s}  final_write_rate={final_rate:.3f}  "
          f"collapsed={'ZERO' if collapsed_zero else 'ALWAYS' if collapsed_always else 'NO'}")

    return {
        "write_rate_trajectory": write_rates,
        "final_write_rate": final_rate,
        "collapsed_to_zero": collapsed_zero,
        "collapsed_to_always": collapsed_always,
    }


# ── Experiment ────────────────────────────────────────────────────────────────

class Exp32WriteGateCollapse(Experiment):
    experiment_id = "exp_3_2"
    hypothesis = (
        "A learned write gate trained without explicit anti-collapse objectives will "
        "learn to never write (or always write) within N training steps on standard tasks."
    )

    def run(self) -> ExperimentResult:
        torch.manual_seed(42)
        results = {}
        for name, (fn, needs_recon) in REGIMES.items():
            results[name] = train_and_monitor(name, fn, needs_recon)

        # A should collapse; others should resist
        a_collapsed = (results["A_no_signal"]["collapsed_to_zero"] or
                       results["A_no_signal"]["collapsed_to_always"])
        any_other_collapsed = any(
            results[k]["collapsed_to_zero"] or results[k]["collapsed_to_always"]
            for k in results if k != "A_no_signal"
        )

        if a_collapsed and not any_other_collapsed:
            outcome = OUTCOME_SUPPORTED
        elif a_collapsed:
            outcome = OUTCOME_INCONCLUSIVE
        else:
            outcome = OUTCOME_REFUTED

        metrics = {
            name: {
                "final_write_rate": v["final_write_rate"],
                "collapsed_zero":   v["collapsed_to_zero"],
                "collapsed_always": v["collapsed_to_always"],
            }
            for name, v in results.items()
        }
        notes = (
            f"Regime A collapsed: {a_collapsed}. "
            f"Other regimes collapsed: {any_other_collapsed}. "
            f"Write rates: " +
            ", ".join(f"{k}={v['final_write_rate']:.2f}" for k, v in results.items()) +
            "."
        )

        return self.result(outcome, metrics, notes, config={
            "train_steps": TRAIN_STEPS, "vocab_size": VOCAB_SIZE,
            "collapse_threshold": COLLAPSE_THRESHOLD,
        })


if __name__ == "__main__":
    Exp32WriteGateCollapse().execute()
