"""
Experiment 7.1 — End-to-End Controller Differentiability

Hypothesis: Gumbel-softmax relaxation produces more stable training than
straight-through estimators, and both outperform RL-based approaches for
discrete memory selection.

The fundamental training problem: write and read operations that select discrete
memory entries break standard backpropagation. Three approaches compete here.

Setup:
  - Identical memory controller architecture
  - Three training approaches:
    (A) Straight-through estimator (STE): forward=hard, backward=soft
    (B) Gumbel-softmax: continuous relaxation with temperature annealing
    (C) REINFORCE: policy gradient on categorical discrete selection
  - Identical associative recall task, identical number of gradient steps
  - Measure: loss variance (stability), final accuracy, gradient norm stability
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
SEQ_LEN        = 24
MEMORY_SLOTS   = 8
TRAIN_STEPS    = 2000
EVAL_BATCHES   = 300
BATCH_SIZE     = 32
LR             = 3e-4
GUMBEL_TEMP_START = 2.0
GUMBEL_TEMP_END   = 0.2
LOG_EVERY      = 20
DEVICE         = "cpu"


# ── Shared encoder ────────────────────────────────────────────────────────────

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, HIDDEN_DIM)
        self.ff    = nn.Sequential(nn.Linear(HIDDEN_DIM, HIDDEN_DIM * 2),
                                    nn.ReLU(), nn.Linear(HIDDEN_DIM * 2, HIDDEN_DIM))
        self.norm  = nn.LayerNorm(HIDDEN_DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embed(x)
        return self.norm(h + self.ff(h))   # (B, L, H)


class WriteHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = nn.Linear(HIDDEN_DIM, 1)

    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.gate(hidden).squeeze(-1)   # (B, L)


class ReadHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj  = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.out     = nn.Linear(HIDDEN_DIM, VOCAB_SIZE)
        self.query_e = nn.Embedding(VOCAB_SIZE, HIDDEN_DIM)

    def forward(self, query: torch.Tensor,
                memory: torch.Tensor) -> torch.Tensor:
        q = self.q_proj(self.query_e(query)).unsqueeze(1)   # (B, 1, H)
        sims = (q * memory).sum(-1) / (HIDDEN_DIM ** 0.5)   # (B, M)
        w    = F.softmax(sims, dim=-1).unsqueeze(-1)
        return self.out((w * memory).sum(1))


# ── Data ──────────────────────────────────────────────────────────────────────

def make_batch(batch_size: int):
    seq    = torch.randint(0, VOCAB_SIZE, (batch_size, SEQ_LEN))
    # plant one key-value pair
    k_pos  = torch.randint(0, SEQ_LEN // 2, (batch_size,))
    v_pos  = torch.randint(SEQ_LEN // 2, SEQ_LEN, (batch_size,))
    keys   = torch.randint(0, VOCAB_SIZE // 2, (batch_size,))
    vals   = torch.randint(VOCAB_SIZE // 2, VOCAB_SIZE, (batch_size,))
    for b in range(batch_size):
        seq[b, k_pos[b]] = keys[b]
        seq[b, v_pos[b]] = vals[b]
    return seq, keys, vals


# ── Method A: Straight-Through Estimator ─────────────────────────────────────

def select_ste(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Hard top-k with straight-through gradient."""
    topk = logits.topk(k, dim=-1).indices          # (B, k)
    mask = torch.zeros_like(logits).scatter_(1, topk, 1.0)
    soft = torch.sigmoid(logits)                    # differentiable proxy
    return soft + (mask - soft).detach()            # STE


# ── Method B: Gumbel-Softmax ─────────────────────────────────────────────────

def select_gumbel(logits: torch.Tensor, k: int, temp: float) -> torch.Tensor:
    """Gumbel-softmax top-k: differentiable categorical sample."""
    # Add gumbel noise
    U = torch.rand_like(logits).clamp(1e-20)
    g = -(-U.log()).log()
    perturbed = (logits + g) / temp
    # soft top-k via softmax-then-threshold
    soft = F.softmax(perturbed, dim=-1)
    topk_idx = soft.topk(k, dim=-1).indices
    mask = torch.zeros_like(logits).scatter_(1, topk_idx, 1.0)
    return soft * mask + (mask - soft * mask).detach()


# ── Method C: REINFORCE ───────────────────────────────────────────────────────

def select_reinforce(logits: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample k positions; return hard mask and log-prob for REINFORCE."""
    probs = torch.sigmoid(logits)
    probs_norm = probs.clamp(1e-6, 1 - 1e-6)
    probs_norm = probs_norm / probs_norm.sum(-1, keepdim=True)
    sampled  = torch.multinomial(probs_norm, k, replacement=False)   # (B, K)
    mask     = torch.zeros_like(logits).scatter_(1, sampled, 1.0)
    log_prob = probs_norm.gather(1, sampled).log().sum(-1)            # (B,)
    return mask, log_prob


def build_memory(hidden: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Soft-select k slots: (B, L, H) → (B, k, H)."""
    # weights: (B, L), use as soft mask
    top_idx = weights.topk(MEMORY_SLOTS, dim=-1).indices
    return hidden.gather(1, top_idx.unsqueeze(-1).expand(-1, -1, HIDDEN_DIM))


def evaluate_model(enc: Encoder, write: WriteHead, read: ReadHead,
                   method: str) -> float:
    enc.eval(); write.eval(); read.eval()
    correct = total = 0
    with torch.no_grad():
        for _ in range(EVAL_BATCHES):
            seq, keys, vals = make_batch(BATCH_SIZE)
            h = enc(seq)
            logits_w = write.logits(h)
            if method in ("STE", "Gumbel"):
                weights = torch.sigmoid(logits_w)
                top_idx = weights.topk(MEMORY_SLOTS, dim=-1).indices
            else:
                top_idx = logits_w.topk(MEMORY_SLOTS, dim=-1).indices
            mem = h.gather(1, top_idx.unsqueeze(-1).expand(-1, -1, HIDDEN_DIM))
            out = read(keys, mem)
            correct += (out.argmax(-1) == vals).sum().item()
            total   += BATCH_SIZE
    enc.train(); write.train(); read.train()
    return correct / total


# ── Training ──────────────────────────────────────────────────────────────────

def train_method(method: str) -> dict:
    enc   = Encoder().to(DEVICE)
    write = WriteHead().to(DEVICE)
    read  = ReadHead().to(DEVICE)
    opt   = Adam(list(enc.parameters()) + list(write.parameters()) +
                 list(read.parameters()), lr=LR)

    losses      = []
    grad_norms  = []
    baseline    = 0.0   # REINFORCE baseline

    for step in range(TRAIN_STEPS):
        t = step / TRAIN_STEPS
        temp = GUMBEL_TEMP_START * (GUMBEL_TEMP_END / GUMBEL_TEMP_START) ** t

        seq, keys, vals = make_batch(BATCH_SIZE)
        h = enc(seq)
        logits_w = write.logits(h)   # (B, L)

        if method == "STE":
            weights = select_ste(logits_w, MEMORY_SLOTS)
            top_idx = weights.topk(MEMORY_SLOTS, dim=-1).indices
            mem = h.gather(1, top_idx.unsqueeze(-1).expand(-1, -1, HIDDEN_DIM))
            logits_r = read(keys, mem)
            loss = F.cross_entropy(logits_r, vals)

        elif method == "Gumbel":
            weights = select_gumbel(logits_w, MEMORY_SLOTS, temp)
            top_idx = weights.topk(MEMORY_SLOTS, dim=-1).indices
            mem = h.gather(1, top_idx.unsqueeze(-1).expand(-1, -1, HIDDEN_DIM))
            logits_r = read(keys, mem)
            loss = F.cross_entropy(logits_r, vals)

        else:  # REINFORCE
            with torch.no_grad():
                mask_hard, log_prob = select_reinforce(logits_w.detach(), MEMORY_SLOTS)
                top_idx = mask_hard.topk(MEMORY_SLOTS, dim=-1).indices
                mem = h.detach().gather(1, top_idx.unsqueeze(-1).expand(-1, -1, HIDDEN_DIM))
                logits_r = read(keys, mem)
                reward = -(F.cross_entropy(logits_r, vals, reduction='none'))
                baseline = 0.9 * baseline + 0.1 * reward.mean().item()
                advantage = (reward - baseline).detach()
            # policy gradient loss
            pg_loss = -(log_prob * advantage).mean()
            # also train encoder and reader with detached gradients
            logits_r_diff = read(keys, mem)
            loss = F.cross_entropy(logits_r_diff, vals) + pg_loss

        opt.zero_grad()
        loss.backward()
        total_grad = sum(p.grad.norm().item() for p in list(enc.parameters())
                         if p.grad is not None)
        grad_norms.append(total_grad)
        torch.nn.utils.clip_grad_norm_(list(enc.parameters()) +
                                        list(write.parameters()) +
                                        list(read.parameters()), 1.0)
        opt.step()
        losses.append(loss.item())

    final_acc = evaluate_model(enc, write, read, method)
    N = 100
    loss_var = torch.tensor(losses[-N:]).var().item() if len(losses) >= N else 0.0
    mean_grad = sum(grad_norms[-N:]) / max(len(grad_norms[-N:]), 1)

    print(f"  {method:10s}  acc={final_acc:.3f}  loss_var={loss_var:.4f}  "
          f"mean_grad_norm={mean_grad:.3f}")
    return {"accuracy": final_acc, "loss_variance": loss_var, "mean_grad_norm": mean_grad}


# ── Experiment ────────────────────────────────────────────────────────────────

class Exp71Differentiability(Experiment):
    experiment_id = "exp_7_1"
    hypothesis = (
        "Gumbel-softmax relaxation produces more stable training than "
        "straight-through estimators, and both outperform REINFORCE for "
        "discrete memory selection."
    )

    def run(self) -> ExperimentResult:
        torch.manual_seed(42)
        results = {}
        for method in ["STE", "Gumbel", "REINFORCE"]:
            results[method] = train_method(method)

        gumbel_most_stable = (
            results["Gumbel"]["loss_variance"] <=
            min(results["STE"]["loss_variance"], results["REINFORCE"]["loss_variance"])
        )
        both_beat_reinforce = (
            results["STE"]["accuracy"] >= results["REINFORCE"]["accuracy"] and
            results["Gumbel"]["accuracy"] >= results["REINFORCE"]["accuracy"]
        )

        if gumbel_most_stable and both_beat_reinforce:
            outcome = OUTCOME_SUPPORTED
        elif gumbel_most_stable or both_beat_reinforce:
            outcome = OUTCOME_INCONCLUSIVE
        else:
            outcome = OUTCOME_REFUTED

        metrics = {k: v for k, v in results.items()}
        metrics["ranking_by_accuracy"] = sorted(
            results.keys(), key=lambda k: results[k]["accuracy"], reverse=True)
        metrics["ranking_by_stability"] = sorted(
            results.keys(), key=lambda k: results[k]["loss_variance"])

        return self.result(outcome, metrics,
            notes=(
                f"Gumbel most stable: {gumbel_most_stable}. "
                f"Both beat REINFORCE: {both_beat_reinforce}. "
                f"Acc: STE={results['STE']['accuracy']:.3f}, "
                f"Gumbel={results['Gumbel']['accuracy']:.3f}, "
                f"REINFORCE={results['REINFORCE']['accuracy']:.3f}"
            ),
            config={"train_steps": TRAIN_STEPS, "memory_slots": MEMORY_SLOTS})


if __name__ == "__main__":
    Exp71Differentiability().execute()
