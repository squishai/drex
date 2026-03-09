"""
Experiment 9.3 — EWC Strong Baseline vs Catastrophic Forgetting

Hypothesis: When the memory controller achieves >70% domain A accuracy before domain B
training, EWC with lambda=5.0 reduces catastrophic forgetting to <50% of standard
fine-tuning's forgetting.
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import copy

from experiments.base import (
    Experiment, ExperimentResult,
    OUTCOME_SUPPORTED, OUTCOME_REFUTED, OUTCOME_INCONCLUSIVE,
)

# ── Config ────────────────────────────────────────────────────────────────────

VOCAB_SIZE     = 128
HIDDEN_DIM     = 64
SEQ_LEN        = 24
NUM_PAIRS      = 3
MEMORY_SLOTS   = 6
BATCH_SIZE     = 32
PHASE1_STEPS   = 1000
PHASE2_STEPS   = 500
EWC_LAMBDA     = 5.0
LR             = 3e-4
DEVICE         = "cpu"

# ── Data ──────────────────────────────────────────────────────────────────────

def make_assoc_batch_domain(batch_size, domain: str,
                             seq_len=SEQ_LEN, vocab_size=VOCAB_SIZE, num_pairs=NUM_PAIRS):
    """
    Domain A: keys from [4, 64), values from [64, 128)
    Domain B: keys from [64, 96), values from [96, 128)
    """
    if domain == "A":
        key_lo, key_hi = 4, 64
        val_lo, val_hi = 64, vocab_size
    else:
        key_lo, key_hi = 64, 96
        val_lo, val_hi = 96, vocab_size

    seq = torch.zeros(batch_size, seq_len, dtype=torch.long)
    target = torch.zeros(batch_size, dtype=torch.long)

    for b in range(batch_size):
        keys = torch.randint(key_lo, key_hi, (num_pairs * 2,)).unique()[:num_pairs]
        while len(keys) < num_pairs:
            keys = torch.cat([keys, torch.randint(key_lo, key_hi, (1,))])[:num_pairs]
        vals = torch.randint(val_lo, val_hi, (num_pairs,))
        pos = 0
        for i in range(num_pairs):
            if pos + 1 < seq_len - 3:
                seq[b, pos]     = keys[i]
                seq[b, pos + 1] = vals[i]
                pos += 2
        for p in range(pos, seq_len - 3):
            seq[b, p] = 3
        qi = torch.randint(0, num_pairs, (1,)).item()
        seq[b, seq_len - 3] = 2
        seq[b, seq_len - 2] = keys[qi]
        seq[b, seq_len - 1] = 0
        target[b] = vals[qi]
    return seq, target


# ── Model ─────────────────────────────────────────────────────────────────────

class Encoder(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2), nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        h = self.embed(x)
        return self.norm(h + self.ff(h))


class WriteGate(nn.Module):
    def __init__(self, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, h):
        return torch.sigmoid(self.fc(h)).squeeze(-1)


class ReadHead(nn.Module):
    def __init__(self, hidden_dim=HIDDEN_DIM, vocab_size=VOCAB_SIZE):
        super().__init__()
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out    = nn.Linear(hidden_dim, vocab_size)

    def forward(self, query_h, memory, mask):
        q = self.q_proj(query_h)
        scores = torch.bmm(memory, q.unsqueeze(-1)).squeeze(-1)
        scores = scores.masked_fill(mask == 0, -1e9)
        attn = torch.softmax(scores, dim=-1)
        ctx  = (attn.unsqueeze(-1) * memory).sum(1)
        return self.out(ctx)


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_all_params(enc, gate, rh):
    return list(enc.parameters()) + list(gate.parameters()) + list(rh.parameters())


def forward_pass(enc, gate, rh, seq, target):
    hidden = enc(seq)
    ws = gate(hidden)
    k = min(MEMORY_SLOTS, SEQ_LEN)
    topk_idx = ws.topk(k, dim=1).indices
    B, L, H = hidden.shape
    memory = hidden.gather(1, topk_idx.unsqueeze(-1).expand(-1, -1, H))
    mask = torch.ones(B, k, device=DEVICE)
    logits = rh(hidden[:, -1, :], memory, mask)
    return F.cross_entropy(logits, target)


def eval_accuracy(enc, gate, rh, domain: str, n_batches: int = 50) -> float:
    enc.eval(); gate.eval(); rh.eval()
    total = 0.0
    with torch.no_grad():
        for _ in range(n_batches):
            seq, target = make_assoc_batch_domain(BATCH_SIZE, domain)
            hidden = enc(seq)
            ws = gate(hidden)
            k = min(MEMORY_SLOTS, SEQ_LEN)
            topk_idx = ws.topk(k, dim=1).indices
            B, L, H = hidden.shape
            memory = hidden.gather(1, topk_idx.unsqueeze(-1).expand(-1, -1, H))
            mask = torch.ones(B, k, device=DEVICE)
            logits = rh(hidden[:, -1, :], memory, mask)
            total += (logits.argmax(-1) == target).float().mean().item()
    enc.train(); gate.train(); rh.train()
    return total / n_batches


def compute_fisher(enc, gate, rh, domain: str, n_batches: int = 50):
    """Compute diagonal Fisher information for EWC."""
    params = get_all_params(enc, gate, rh)
    fisher = [torch.zeros_like(p) for p in params]

    enc.train(); gate.train(); rh.train()
    for _ in range(n_batches):
        seq, target = make_assoc_batch_domain(BATCH_SIZE, domain)
        loss = forward_pass(enc, gate, rh, seq, target)
        grads = torch.autograd.grad(loss, params, retain_graph=False, create_graph=False, allow_unused=True)
        for i, g in enumerate(grads):
            if g is not None:
                fisher[i] += g.data ** 2
    for f in fisher:
        f /= n_batches
    return fisher


# ── Training ──────────────────────────────────────────────────────────────────

def train_phase1(enc, gate, rh, max_steps=PHASE1_STEPS) -> tuple[float, int]:
    """Train on domain A until acc > 0.70 or max_steps. Returns (acc, steps)."""
    opt = Adam(get_all_params(enc, gate, rh), lr=LR)
    enc.train(); gate.train(); rh.train()
    acc = 0.0
    for step in range(max_steps):
        seq, target = make_assoc_batch_domain(BATCH_SIZE, "A")
        loss = forward_pass(enc, gate, rh, seq, target)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (step + 1) % 500 == 0:
            acc = eval_accuracy(enc, gate, rh, "A", n_batches=20)
            print(f"    Phase 1 step {step+1}: acc_A={acc:.3f}")
            if acc > 0.70:
                return acc, step + 1
    return eval_accuracy(enc, gate, rh, "A"), max_steps


def train_phase2_standard(enc, gate, rh):
    """Fine-tune on domain B (no EWC)."""
    opt = Adam(get_all_params(enc, gate, rh), lr=LR)
    enc.train(); gate.train(); rh.train()
    for step in range(PHASE2_STEPS):
        seq, target = make_assoc_batch_domain(BATCH_SIZE, "B")
        loss = forward_pass(enc, gate, rh, seq, target)
        opt.zero_grad()
        loss.backward()
        opt.step()


def train_phase2_ewc(enc, gate, rh, fisher, theta_star):
    """Fine-tune on domain B with EWC regularization."""
    params = get_all_params(enc, gate, rh)
    opt = Adam(params, lr=LR)
    enc.train(); gate.train(); rh.train()

    for step in range(PHASE2_STEPS):
        seq, target = make_assoc_batch_domain(BATCH_SIZE, "B")
        task_loss = forward_pass(enc, gate, rh, seq, target)

        # EWC penalty
        ewc_loss = torch.tensor(0.0)
        current_params = get_all_params(enc, gate, rh)
        for i, (p, f, p_star) in enumerate(zip(current_params, fisher, theta_star)):
            ewc_loss = ewc_loss + (f * (p - p_star.detach()) ** 2).sum()
        ewc_loss = EWC_LAMBDA * ewc_loss

        loss = task_loss + ewc_loss
        opt.zero_grad()
        loss.backward()
        opt.step()


# ── Experiment class ──────────────────────────────────────────────────────────

class Exp93EWCStrongBaseline(Experiment):
    experiment_id = "exp_9_3"
    hypothesis = (
        "When the memory controller achieves >70% domain A accuracy before domain B "
        "training, EWC with lambda=5.0 reduces catastrophic forgetting to <50% of "
        "standard fine-tuning's forgetting."
    )

    def run(self) -> ExperimentResult:
        # ── Phase 1: Train both models on Domain A ────────────────────────────
        print("  Phase 1: Training standard model on Domain A ...")
        enc_std  = Encoder().to(DEVICE)
        gate_std = WriteGate().to(DEVICE)
        rh_std   = ReadHead().to(DEVICE)
        acc_before_std, steps_std = train_phase1(enc_std, gate_std, rh_std)
        print(f"  acc_A_before (std) = {acc_before_std:.3f} after {steps_std} steps")

        print("  Phase 1: Training EWC model on Domain A ...")
        enc_ewc  = Encoder().to(DEVICE)
        gate_ewc = WriteGate().to(DEVICE)
        rh_ewc   = ReadHead().to(DEVICE)
        # Copy weights from standard model for fair comparison
        enc_ewc.load_state_dict(copy.deepcopy(enc_std.state_dict()))
        gate_ewc.load_state_dict(copy.deepcopy(gate_std.state_dict()))
        rh_ewc.load_state_dict(copy.deepcopy(rh_std.state_dict()))
        acc_before_ewc = acc_before_std  # same init since we copy

        # Compute Fisher on domain A for EWC model
        print("  Computing Fisher information ...")
        fisher = compute_fisher(enc_ewc, gate_ewc, rh_ewc, "A", n_batches=20)
        theta_star = [p.data.clone() for p in get_all_params(enc_ewc, gate_ewc, rh_ewc)]

        # ── Phase 2: Fine-tune on Domain B ───────────────────────────────────
        print("  Phase 2: Standard fine-tuning on Domain B ...")
        train_phase2_standard(enc_std, gate_std, rh_std)
        acc_a_after_std = eval_accuracy(enc_std, gate_std, rh_std, "A")
        print(f"  acc_A_after (std)  = {acc_a_after_std:.3f}")

        print("  Phase 2: EWC fine-tuning on Domain B ...")
        train_phase2_ewc(enc_ewc, gate_ewc, rh_ewc, fisher, theta_star)
        acc_a_after_ewc = eval_accuracy(enc_ewc, gate_ewc, rh_ewc, "A")
        print(f"  acc_A_after (ewc)  = {acc_a_after_ewc:.3f}")

        forgetting_std = acc_before_std - acc_a_after_std
        forgetting_ewc = acc_before_ewc - acc_a_after_ewc
        forgetting_ratio = forgetting_ewc / max(forgetting_std, 1e-8)

        print(f"  forgetting_std={forgetting_std:.3f}, forgetting_ewc={forgetting_ewc:.3f}, "
              f"ratio={forgetting_ratio:.3f}")

        # ── Outcome ───────────────────────────────────────────────────────────
        if acc_before_std < 0.70:
            outcome = OUTCOME_INCONCLUSIVE
            notes = f"Precondition failed: acc_A_before={acc_before_std:.3f} < 0.70."
        elif forgetting_std > 0.15 and forgetting_ewc < forgetting_std * 0.50:
            outcome = OUTCOME_SUPPORTED
            notes = (f"acc_A_before={acc_before_std:.3f}. forgetting_std={forgetting_std:.3f}, "
                     f"forgetting_ewc={forgetting_ewc:.3f} (ratio={forgetting_ratio:.3f}).")
        elif acc_before_std >= 0.70 and forgetting_ewc >= forgetting_std * 0.50:
            outcome = OUTCOME_REFUTED
            notes = (f"EWC gave no benefit. forgetting_std={forgetting_std:.3f}, "
                     f"forgetting_ewc={forgetting_ewc:.3f}.")
        else:
            outcome = OUTCOME_INCONCLUSIVE
            notes = (f"forgetting_std={forgetting_std:.3f} < 0.15 threshold, "
                     f"forgetting_ewc={forgetting_ewc:.3f}.")

        metrics = {
            "acc_a_before":      round(acc_before_std, 4),
            "acc_a_after_std":   round(acc_a_after_std, 4),
            "acc_a_after_ewc":   round(acc_a_after_ewc, 4),
            "forgetting_std":    round(forgetting_std, 4),
            "forgetting_ewc":    round(forgetting_ewc, 4),
            "forgetting_ratio":  round(forgetting_ratio, 4),
            "phase1_steps":      steps_std,
        }
        config = {
            "vocab_size": VOCAB_SIZE, "hidden_dim": HIDDEN_DIM,
            "seq_len": SEQ_LEN, "num_pairs": NUM_PAIRS,
            "phase1_steps": PHASE1_STEPS, "phase2_steps": PHASE2_STEPS,
            "ewc_lambda": EWC_LAMBDA,
        }
        return self.result(outcome, metrics, notes, config)


if __name__ == "__main__":
    Exp93EWCStrongBaseline().execute()
