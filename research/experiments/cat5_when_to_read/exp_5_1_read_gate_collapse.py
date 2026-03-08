"""
Experiment 5.1 — Read Gate Collapse Detection

Hypothesis: A learned read gate trained without explicit anti-collapse objectives
will learn a degenerate policy (always read or never read) within N training steps.

This is the mirror of experiment 3.2. Always-read = useless gate (pay compute for
nothing). Never-read = model ignores memory entirely.

Setup:
  - Controller has a learned binary read gate
  - Gate receives (query, memory_summary) and decides whether to retrieve
  - Regime A: task loss only (no gate signal)
  - Regime B: sparsity regularization (penalize excessive reading)
  - Regime C: coverage reward (penalize zero reading)
  - Regime D: confidence gating (read when prediction uncertainty is high)
  - On each task, only 40% of steps actually need memory — rest has answer in context
  - Measure: read rate trajectory, collapse detection, efficiency on non-memory steps
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
HIDDEN_DIM     = 48
MEMORY_SIZE    = 8
P_NEEDS_MEMORY = 0.40   # fraction of tasks where memory is required
TRAIN_STEPS    = 1000
EVAL_BATCHES   = 500
BATCH_SIZE     = 32
LR             = 3e-4
LOG_EVERY      = 50
COLLAPSE_ALWAYS_THRESHOLD = 0.90
COLLAPSE_NEVER_THRESHOLD  = 0.10
DEVICE         = "cpu"


# ── Model ─────────────────────────────────────────────────────────────────────

class ContextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, HIDDEN_DIM)
        self.pool  = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embed(x).mean(1)
        return F.relu(self.pool(h))


class ReadGateModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(HIDDEN_DIM + MEMORY_SIZE, 32), nn.ReLU(),
            nn.Linear(32, 1), nn.Sigmoid(),
        )

    def forward(self, ctx_h: torch.Tensor,
                mem_summary: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x    = torch.cat([ctx_h, mem_summary], dim=-1)
        soft = self.gate(x).squeeze(-1)
        hard = (soft > 0.5).float()
        hard_st = soft + (hard - soft).detach()
        return hard_st, soft


class MemoryAugModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.ctx_enc  = ContextEncoder()
        self.read_gate = ReadGateModule()
        self.mem_proj  = nn.Linear(HIDDEN_DIM, MEMORY_SIZE)
        self.merge     = nn.Linear(HIDDEN_DIM + HIDDEN_DIM, HIDDEN_DIM)
        self.head      = nn.Linear(HIDDEN_DIM, VOCAB_SIZE)

    def forward(
        self, seq: torch.Tensor, memory: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        seq:    (B, L)
        memory: (B, M, H) stored key-value pairs
        Returns: logits (B, V), gate (B,), uncertainty (B,)
        """
        ctx_h = self.ctx_enc(seq)                          # (B, H)
        mem_pooled = memory.mean(1)                         # (B, H)
        mem_summary = self.mem_proj(mem_pooled)             # (B, M)

        gate, soft_gate = self.read_gate(ctx_h, mem_summary)

        # retrieve: attention over memory
        q = ctx_h.unsqueeze(1)                              # (B, 1, H)
        sims = (q * memory).sum(-1) / (HIDDEN_DIM ** 0.5)  # (B, M)
        w    = F.softmax(sims, dim=-1).unsqueeze(-1)
        retrieved = (w * memory).sum(1)                     # (B, H)

        merged = self.merge(torch.cat([ctx_h, gate.unsqueeze(-1) * retrieved], dim=-1))
        logits = self.head(merged)

        # uncertainty = entropy of logits
        probs = F.softmax(logits.detach(), dim=-1)
        entropy = -(probs * (probs + 1e-8).log()).sum(-1)

        return logits, soft_gate, entropy


def make_batch(batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    B = batch_size
    seq  = torch.randint(1, VOCAB_SIZE, (B, 8))
    tgt  = torch.randint(0, VOCAB_SIZE, (B,))
    mem  = torch.randn(B, MEMORY_SIZE, HIDDEN_DIM)
    needs_memory = torch.rand(B) < P_NEEDS_MEMORY
    # for memory-needing items, answer is encoded in memory slot 0
    for b in range(B):
        if needs_memory[b]:
            # memory[0] "points" to target via a learnable relationship
            mem[b, 0, :4] = tgt[b].float().expand(4)
    return seq, tgt, mem, needs_memory


# ── Regime Loss Functions ─────────────────────────────────────────────────────

def regime_task_only(task_loss, soft_gate, entropy):
    return task_loss

def regime_sparsity(task_loss, soft_gate, entropy):
    return task_loss + 0.1 * soft_gate.mean()

def regime_coverage(task_loss, soft_gate, entropy):
    read_rate = soft_gate.mean()
    penalty   = F.relu(P_NEEDS_MEMORY - read_rate)
    return task_loss + 0.2 * penalty

def regime_confidence(task_loss, soft_gate, entropy):
    # read gate should correlate with entropy
    target_gate = (entropy / entropy.max().clamp(min=1e-8)).detach()
    gate_loss   = F.mse_loss(soft_gate, target_gate)
    return task_loss + 0.1 * gate_loss

REGIMES = {
    "A_task_only":  regime_task_only,
    "B_sparsity":   regime_sparsity,
    "C_coverage":   regime_coverage,
    "D_confidence": regime_confidence,
}


def train_regime(name, loss_fn) -> dict:
    model  = MemoryAugModel().to(DEVICE)
    opt    = Adam(model.parameters(), lr=LR)
    rates  = []

    for step in range(TRAIN_STEPS):
        seq, tgt, mem, _ = make_batch(BATCH_SIZE)
        logits, soft_gate, entropy = model(seq, mem)
        task_loss = F.cross_entropy(logits, tgt)
        loss      = loss_fn(task_loss, soft_gate, entropy)
        opt.zero_grad(); loss.backward(); opt.step()

        if step % LOG_EVERY == 0:
            rates.append((soft_gate > 0.5).float().mean().item())

    final = rates[-1]
    collapsed = (final > COLLAPSE_ALWAYS_THRESHOLD or final < COLLAPSE_NEVER_THRESHOLD)
    mode = "ALWAYS" if final > COLLAPSE_ALWAYS_THRESHOLD else "NEVER" if final < COLLAPSE_NEVER_THRESHOLD else "stable"
    print(f"  {name:20s}  read_rate={final:.3f}  mode={mode}")
    return {"final_read_rate": final, "collapsed": collapsed, "mode": mode, "trajectory": rates}


# ── Experiment ────────────────────────────────────────────────────────────────

class Exp51ReadGateCollapse(Experiment):
    experiment_id = "exp_5_1"
    hypothesis = (
        "A learned read gate trained without explicit anti-collapse objectives will "
        "learn a degenerate policy (always read or never read) within N training steps."
    )

    def run(self) -> ExperimentResult:
        torch.manual_seed(42)
        results = {}
        for name, fn in REGIMES.items():
            results[name] = train_regime(name, fn)

        a_collapsed = results["A_task_only"]["collapsed"]
        others_stable = all(
            not results[k]["collapsed"]
            for k in results if k != "A_task_only"
        )

        if a_collapsed and others_stable:
            outcome = OUTCOME_SUPPORTED
        elif a_collapsed:
            outcome = OUTCOME_INCONCLUSIVE
        else:
            outcome = OUTCOME_REFUTED

        metrics = {
            k: {
                "final_read_rate": v["final_read_rate"],
                "collapsed": v["collapsed"],
                "mode": v["mode"],
            }
            for k, v in results.items()
        }
        notes = (
            f"Regime A collapsed: {a_collapsed} ({results['A_task_only']['mode']}). "
            "Read rates: " +
            ", ".join(f"{k}={v['final_read_rate']:.2f}" for k, v in results.items()) + "."
        )
        return self.result(outcome, metrics, notes,
                           config={"train_steps": TRAIN_STEPS, "p_needs_memory": P_NEEDS_MEMORY})


if __name__ == "__main__":
    Exp51ReadGateCollapse().execute()
