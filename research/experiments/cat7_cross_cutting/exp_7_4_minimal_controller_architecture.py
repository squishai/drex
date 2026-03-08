"""
Experiment 7.4 — Minimal Controller Architecture

Hypothesis: Meaningful memory management behavior requires at minimum two layers
of non-linearity in the controller network.

Setup:
  - Controller depth 0 (linear only), 1 (linear+ReLU), 2 (2 layers+ReLU), 3 (3 layers+ReLU)
  - All same hidden width (32)
  - "Meaningful behavior" requires ALL three:
      (a) write rate between 5% and 80% of sequence positions
      (b) accuracy > random baseline (12.5% for 8-class)
      (c) non-uniform write pattern: write-rate std > 0.05 across positions
  - SUPPORTED if depth=1 fails meaningful threshold AND depth=2 passes
  - REFUTED if depth=0 (linear) passes meaningful threshold
  - INCONCLUSIVE if depth=3 is first to pass
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
CTRL_WIDTH      = 32
SEQ_LEN         = 24
MEMORY_SLOTS    = 8
BATCH_SIZE      = 32
TRAIN_STEPS     = 1500
EVAL_BATCHES    = 200
LR              = 3e-4
DEVICE          = "cpu"

RANDOM_BASELINE = 1.0 / MEMORY_SLOTS   # 12.5% for 8 slots
WRITE_RATE_MIN  = 0.05
WRITE_RATE_MAX  = 0.80
WRITE_STD_MIN   = 0.05


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


def make_controller(depth: int) -> nn.Module:
    """
    depth=0: single linear HIDDEN_DIM -> 1
    depth=1: Linear(HIDDEN_DIM, CTRL_WIDTH) -> ReLU -> Linear(CTRL_WIDTH, 1)
    depth=2: two hidden layers of CTRL_WIDTH with ReLU
    depth=3: three hidden layers of CTRL_WIDTH with ReLU
    """
    layers: list[nn.Module] = []
    in_dim = HIDDEN_DIM
    for _ in range(depth):
        layers.append(nn.Linear(in_dim, CTRL_WIDTH))
        layers.append(nn.ReLU())
        in_dim = CTRL_WIDTH
    layers.append(nn.Linear(in_dim, 1))
    return nn.Sequential(*layers)


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


# ── Train + Evaluate ──────────────────────────────────────────────────────────

def train_and_eval(depth: int) -> dict:
    torch.manual_seed(42)
    enc  = Encoder()
    ctrl = make_controller(depth)
    read = ReadHead()
    opt  = Adam(
        list(enc.parameters()) + list(ctrl.parameters()) + list(read.parameters()),
        lr=LR,
    )

    for _ in range(TRAIN_STEPS):
        seq, keys, vals = make_batch(BATCH_SIZE)
        h        = enc(seq)
        logits_w = ctrl(h).squeeze(-1)                  # (B, L)
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

    # Evaluate accuracy and write statistics
    enc.eval(); ctrl.eval(); read.eval()
    correct = total = 0
    # Accumulate write gates: sum of sigmoid(logits) per position
    gate_accum = torch.zeros(SEQ_LEN)
    gate_count = 0

    with torch.no_grad():
        for _ in range(EVAL_BATCHES):
            seq, keys, vals = make_batch(BATCH_SIZE)
            h        = enc(seq)
            logits_w = ctrl(h).squeeze(-1)
            gate_probs = torch.sigmoid(logits_w)         # (B, L)
            gate_accum += gate_probs.mean(0)             # average over batch
            gate_count += 1

            top_idx = logits_w.topk(MEMORY_SLOTS, dim=-1).indices
            mem     = h.gather(1, top_idx.unsqueeze(-1).expand(-1, -1, HIDDEN_DIM))
            out     = read(keys, mem)
            correct += (out.argmax(-1) == vals).sum().item()
            total   += BATCH_SIZE

    avg_gate = gate_accum / gate_count         # mean gate prob per position (L,)
    write_rate = avg_gate.mean().item()        # overall mean write rate
    write_std  = avg_gate.std().item()         # std across positions

    acc = correct / total

    # "Hard" write rate: fraction of positions chosen as top-k
    hard_write_rate = MEMORY_SLOTS / SEQ_LEN   # always = 8/24 = 0.333

    meaningful = (
        WRITE_RATE_MIN <= write_rate <= WRITE_RATE_MAX
        and acc > RANDOM_BASELINE
        and write_std > WRITE_STD_MIN
    )

    print(f"  depth={depth}  acc={acc:.3f}  write_rate={write_rate:.3f}  "
          f"write_std={write_std:.4f}  meaningful={meaningful}")

    return {
        "accuracy":       acc,
        "write_rate":     write_rate,
        "write_std":      write_std,
        "hard_write_rate": hard_write_rate,
        "meaningful":     meaningful,
    }


# ── Experiment ────────────────────────────────────────────────────────────────

class Exp74MinimalControllerArchitecture(Experiment):
    experiment_id = "exp_7_4"
    hypothesis = (
        "Meaningful memory management behavior requires at minimum two layers "
        "of non-linearity in the controller network."
    )

    def run(self) -> ExperimentResult:
        torch.manual_seed(42)

        depths  = [0, 1, 2, 3]
        results = {}
        for d in depths:
            results[d] = train_and_eval(d)

        # Find minimum depth that passes the meaningful threshold
        min_depth_meaningful = None
        for d in depths:
            if results[d]["meaningful"]:
                min_depth_meaningful = d
                break

        depth0_meaningful = results[0]["meaningful"]
        depth1_meaningful = results[1]["meaningful"]
        depth2_meaningful = results[2]["meaningful"]

        if depth0_meaningful:
            outcome = OUTCOME_REFUTED
        elif not depth1_meaningful and depth2_meaningful:
            outcome = OUTCOME_SUPPORTED
        elif min_depth_meaningful == 3:
            outcome = OUTCOME_INCONCLUSIVE
        else:
            outcome = OUTCOME_INCONCLUSIVE

        metrics = {
            "acc_per_depth":               {d: results[d]["accuracy"]   for d in depths},
            "write_rate_per_depth":        {d: results[d]["write_rate"] for d in depths},
            "write_std_per_depth":         {d: results[d]["write_std"]  for d in depths},
            "meaningful_threshold_per_depth": {d: results[d]["meaningful"] for d in depths},
            "min_depth_for_meaningful":    min_depth_meaningful,
        }
        notes = (
            f"Min depth for meaningful behavior: {min_depth_meaningful}. "
            f"depth=0 meaningful: {depth0_meaningful}. "
            f"depth=1 meaningful: {depth1_meaningful}. "
            f"depth=2 meaningful: {depth2_meaningful}."
        )
        return self.result(outcome, metrics, notes, config={
            "train_steps": TRAIN_STEPS,
            "ctrl_width":  CTRL_WIDTH,
            "memory_slots": MEMORY_SLOTS,
            "random_baseline": RANDOM_BASELINE,
            "write_rate_min": WRITE_RATE_MIN,
            "write_rate_max": WRITE_RATE_MAX,
            "write_std_min":  WRITE_STD_MIN,
        })


if __name__ == "__main__":
    Exp74MinimalControllerArchitecture().execute()
