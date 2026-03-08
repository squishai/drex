"""
Experiment 7.6 — Adversarial Memory Probing

Hypothesis: The memory controller is measurably vulnerable to inputs designed
to maximize write activity, and this vulnerability does not self-correct during
training.

Setup:
  - Train memory controller on associative recall.
  - Construct adversarial inputs: gradient ascent on write_gate_score (20 steps).
  - Compare normal_write_rate vs adversarial_write_rate.
  - Track adversarial write rate at checkpoints 500, 1000, 1500.
  - SUPPORTED if adversarial_write_rate > normal_write_rate * 1.5 AND rate
    does NOT decrease with more training.
  - REFUTED if adversarial_write_rate <= normal_write_rate * 1.2.
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

# ── Config ────────────────────────────────────────────────────────────────────

VOCAB_SIZE      = 64
HIDDEN_DIM      = 64
SEQ_LEN         = 24
MEMORY_SLOTS    = 8
BATCH_SIZE      = 32
TRAIN_STEPS     = 1500
EVAL_BATCHES    = 100
ADV_STEPS       = 20       # gradient ascent steps for adversarial inputs
ADV_LR          = 0.1      # step size for adversarial perturbation
ADV_BATCHES     = 50       # batches to average adversarial write rate
LR              = 3e-4
DEVICE          = "cpu"
CHECKPOINTS     = [500, 1000, 1500]


# ── Models ────────────────────────────────────────────────────────────────────

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
        return self.gate(hidden).squeeze(-1)          # (B, L)

    def gate_score(self, hidden: torch.Tensor) -> torch.Tensor:
        """Return sigmoid gate probabilities."""
        return torch.sigmoid(self.forward(hidden))    # (B, L)


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


# ── Normal write rate ─────────────────────────────────────────────────────────

def measure_normal_write_rate(enc: Encoder, ctrl: WriteController) -> float:
    enc.eval(); ctrl.eval()
    total_rate = 0.0
    with torch.no_grad():
        for _ in range(EVAL_BATCHES):
            seq, _, _ = make_batch(BATCH_SIZE)
            h     = enc(seq)
            gates = ctrl.gate_score(h)       # (B, L) in [0,1]
            total_rate += gates.mean().item()
    enc.train(); ctrl.train()
    return total_rate / EVAL_BATCHES


# ── Adversarial write rate ────────────────────────────────────────────────────

def measure_adversarial_write_rate(enc: Encoder, ctrl: WriteController) -> float:
    """
    Construct adversarial inputs via gradient ascent on total gate score.
    We perturb the embedding space directly (continuous relaxation).
    """
    enc.eval(); ctrl.eval()
    total_adv_rate = 0.0

    for _ in range(ADV_BATCHES):
        seq, _, _ = make_batch(BATCH_SIZE)
        # Start from real embeddings, add learnable perturbation
        with torch.no_grad():
            h_base = enc(seq)                # (B, L, H)

        # Perturbation in embedding space
        delta = torch.zeros_like(h_base, requires_grad=True)
        adv_opt = torch.optim.SGD([delta], lr=ADV_LR)

        for _ in range(ADV_STEPS):
            adv_opt.zero_grad()
            h_adv   = h_base + delta
            gates   = ctrl.gate_score(h_adv)          # (B, L)
            # Maximise total gate activation
            loss = -gates.sum()
            loss.backward()
            adv_opt.step()
            # Clip perturbation magnitude (L-inf ball of radius 1.0)
            with torch.no_grad():
                delta.clamp_(-1.0, 1.0)

        with torch.no_grad():
            h_adv = h_base + delta.detach()
            gates = ctrl.gate_score(h_adv)
            total_adv_rate += gates.mean().item()

    enc.train(); ctrl.train()
    return total_adv_rate / ADV_BATCHES


# ── Training with checkpoint evaluation ──────────────────────────────────────

def train_with_checkpoints(
    enc: Encoder, ctrl: WriteController, read: ReadHead,
) -> dict[int, float]:
    """Train and measure adversarial write rate at each checkpoint."""
    opt = Adam(
        list(enc.parameters()) + list(ctrl.parameters()) + list(read.parameters()),
        lr=LR,
    )
    adv_rate_at_ckpt: dict[int, float] = {}

    for step in range(1, TRAIN_STEPS + 1):
        seq, keys, vals = make_batch(BATCH_SIZE)
        h        = enc(seq)
        logits_w = ctrl(h)
        top_idx  = logits_w.topk(MEMORY_SLOTS, dim=-1).indices
        mem      = h.gather(1, top_idx.unsqueeze(-1).expand(-1, -1, HIDDEN_DIM))
        out      = read(keys, mem)
        loss     = F.cross_entropy(out, vals)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(enc.parameters()) + list(ctrl.parameters()) + list(read.parameters()), 1.0
        )
        opt.step()

        if step in CHECKPOINTS:
            adv_rate = measure_adversarial_write_rate(enc, ctrl)
            adv_rate_at_ckpt[step] = adv_rate
            print(f"  step={step}  adv_write_rate={adv_rate:.4f}")

    return adv_rate_at_ckpt


# ── Experiment ────────────────────────────────────────────────────────────────

class Exp76AdversarialMemoryProbing(Experiment):
    experiment_id = "exp_7_6"
    hypothesis = (
        "The memory controller is measurably vulnerable to inputs designed to "
        "maximize write activity, and this vulnerability does not self-correct "
        "during training."
    )

    def run(self) -> ExperimentResult:
        torch.manual_seed(42)

        enc  = Encoder()
        ctrl = WriteController()
        read = ReadHead()

        print("  Training with checkpoint adversarial evaluation...")
        adv_at_ckpt = train_with_checkpoints(enc, ctrl, read)

        normal_write_rate     = measure_normal_write_rate(enc, ctrl)
        adversarial_write_rate = adv_at_ckpt[TRAIN_STEPS]

        print(f"  normal_write_rate={normal_write_rate:.4f}  "
              f"adversarial_write_rate={adversarial_write_rate:.4f}")

        adversarial_ratio = adversarial_write_rate / max(normal_write_rate, 1e-9)

        # Self-correction: does adversarial rate decrease over checkpoints?
        ckpt_rates = [adv_at_ckpt[c] for c in CHECKPOINTS]
        # Monotone decrease = self-correction
        self_corrects = all(
            ckpt_rates[i] >= ckpt_rates[i + 1]
            for i in range(len(ckpt_rates) - 1)
        )

        vulnerable    = adversarial_write_rate > normal_write_rate * 1.5
        not_corrected = not self_corrects

        if vulnerable and not_corrected:
            outcome = OUTCOME_SUPPORTED
        elif adversarial_write_rate <= normal_write_rate * 1.2:
            outcome = OUTCOME_REFUTED
        else:
            outcome = OUTCOME_INCONCLUSIVE

        metrics = {
            "normal_write_rate":        normal_write_rate,
            "adversarial_write_rate":   adversarial_write_rate,
            "adversarial_ratio":        adversarial_ratio,
            "self_corrects":            self_corrects,
            "write_rate_at_500":        adv_at_ckpt.get(500),
            "write_rate_at_1000":       adv_at_ckpt.get(1000),
            "write_rate_at_1500":       adv_at_ckpt.get(1500),
        }
        notes = (
            f"Adversarial ratio: {adversarial_ratio:.3f} "
            f"(threshold >1.5: {vulnerable}). "
            f"Self-corrects: {self_corrects}. "
            f"Checkpoint rates: {ckpt_rates}."
        )
        return self.result(outcome, metrics, notes, config={
            "train_steps":   TRAIN_STEPS,
            "adv_steps":     ADV_STEPS,
            "adv_lr":        ADV_LR,
            "memory_slots":  MEMORY_SLOTS,
            "checkpoints":   CHECKPOINTS,
        })


if __name__ == "__main__":
    Exp76AdversarialMemoryProbing().execute()
