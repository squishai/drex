"""
Experiment 7.2 — Controller Overhead Budget

Hypothesis: There exists a maximum controller complexity (measured in parameter
count) beyond which the controller's overhead exceeds its efficiency contribution.

Setup:
  - 5 controller complexity levels: Tiny, Small, Medium, Large, XL
  - All control the same 8-slot associative recall task
  - Measure task accuracy, parameter count, FLOPs per forward pass
  - Compute efficiency ratio = accuracy_gain_over_tiny / param_count_ratio
  - SUPPORTED if efficiency peaks at Small or Medium and declines at Large/XL
  - REFUTED if largest controller always best
  - INCONCLUSIVE if all roughly equal
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

VOCAB_SIZE   = 64
HIDDEN_DIM   = 64
SEQ_LEN      = 24
MEMORY_SLOTS = 8
BATCH_SIZE   = 32
TRAIN_STEPS  = 1500
EVAL_BATCHES = 200
LR           = 3e-4
DEVICE       = "cpu"

# ── Complexity level definitions ──────────────────────────────────────────────
# Each entry: (name, list_of_hidden_sizes_for_MLP_layers)
# The MLP takes HIDDEN_DIM input and outputs 1 (write gate logit per token)
COMPLEXITY_LEVELS = [
    ("Tiny",   []),            # 1 linear: HIDDEN_DIM -> 1  (~64 params)
    ("Small",  [64]),          # 2 linear: HIDDEN_DIM -> 64 -> 1  (~4K params)
    ("Medium", [64, 64]),      # 3 linear with wider hidden  (~16K params)
    ("Large",  [64, 64, 64]),  # 4 linear  (~64K but scaled below)
    ("XL",     [64, 64, 64, 64]),  # 5 linear  (~256K but scaled below)
]

# Widen hidden dims to hit approximate param targets
COMPLEXITY_WIDTHS = {
    "Tiny":   [],
    "Small":  [64],
    "Medium": [256],
    "Large":  [512, 256],
    "XL":     [512, 512, 256],
}


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
        return self.norm(h + self.ff(h))  # (B, L, H)


class VariableController(nn.Module):
    """Write gate controller with variable MLP depth/width."""

    def __init__(self, hidden_sizes: list[int]):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = HIDDEN_DIM
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        # hidden: (B, L, H)  ->  logits: (B, L)
        return self.mlp(hidden).squeeze(-1)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def flops_per_forward(self) -> int:
        """Approximate FLOPs as sum of weight matrix sizes (multiply-adds)."""
        total = 0
        for m in self.modules():
            if isinstance(m, nn.Linear):
                total += m.in_features * m.out_features
        return total


class ReadHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj  = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.out     = nn.Linear(HIDDEN_DIM, VOCAB_SIZE)
        self.query_e = nn.Embedding(VOCAB_SIZE, HIDDEN_DIM)

    def forward(self, query: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        q    = self.q_proj(self.query_e(query)).unsqueeze(1)  # (B, 1, H)
        sims = (q * memory).sum(-1) / (HIDDEN_DIM ** 0.5)    # (B, M)
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


# ── Training + Evaluation ─────────────────────────────────────────────────────

def train_and_eval(name: str, hidden_sizes: list[int]) -> dict:
    torch.manual_seed(42)
    enc   = Encoder()
    ctrl  = VariableController(hidden_sizes)
    read  = ReadHead()
    opt   = Adam(
        list(enc.parameters()) + list(ctrl.parameters()) + list(read.parameters()),
        lr=LR,
    )

    for step in range(TRAIN_STEPS):
        seq, keys, vals = make_batch(BATCH_SIZE)
        h        = enc(seq)                              # (B, L, H)
        logits_w = ctrl(h)                               # (B, L)
        top_idx  = logits_w.topk(MEMORY_SLOTS, dim=-1).indices
        mem      = h.gather(1, top_idx.unsqueeze(-1).expand(-1, -1, HIDDEN_DIM))
        logits_r = read(keys, mem)
        loss     = F.cross_entropy(logits_r, vals)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(enc.parameters()) + list(ctrl.parameters()) + list(read.parameters()), 1.0
        )
        opt.step()

    # Evaluate
    enc.eval(); ctrl.eval(); read.eval()
    correct = total = 0
    with torch.no_grad():
        for _ in range(EVAL_BATCHES):
            seq, keys, vals = make_batch(BATCH_SIZE)
            h       = enc(seq)
            logits  = ctrl(h)
            top_idx = logits.topk(MEMORY_SLOTS, dim=-1).indices
            mem     = h.gather(1, top_idx.unsqueeze(-1).expand(-1, -1, HIDDEN_DIM))
            out     = read(keys, mem)
            correct += (out.argmax(-1) == vals).sum().item()
            total   += BATCH_SIZE

    acc         = correct / total
    param_count = ctrl.param_count()
    flops       = ctrl.flops_per_forward()

    print(f"  {name:8s}  params={param_count:>7d}  flops={flops:>7d}  acc={acc:.3f}")
    return {"accuracy": acc, "params": param_count, "flops": flops}


# ── Experiment ────────────────────────────────────────────────────────────────

class Exp72ControllerOverheadBudget(Experiment):
    experiment_id = "exp_7_2"
    hypothesis = (
        "There exists a maximum controller complexity (measured in parameter count) "
        "beyond which the controller's overhead exceeds its efficiency contribution."
    )

    def run(self) -> ExperimentResult:
        torch.manual_seed(42)

        level_names   = ["Tiny", "Small", "Medium", "Large", "XL"]
        results_map: dict[str, dict] = {}

        for name in level_names:
            hidden_sizes = COMPLEXITY_WIDTHS[name]
            results_map[name] = train_and_eval(name, hidden_sizes)

        tiny_acc        = results_map["Tiny"]["accuracy"]
        tiny_params     = max(results_map["Tiny"]["params"], 1)

        acc_per_complexity      = {n: results_map[n]["accuracy"] for n in level_names}
        params_per_complexity   = {n: results_map[n]["params"]   for n in level_names}
        flops_per_complexity    = {n: results_map[n]["flops"]    for n in level_names}

        # Efficiency ratio: accuracy gain over Tiny / log param ratio
        efficiency_ratio: dict[str, float] = {}
        for name in level_names:
            acc_gain    = results_map[name]["accuracy"] - tiny_acc
            param_ratio = results_map[name]["params"] / tiny_params
            # avoid division by zero; Tiny itself gets ratio 0
            if param_ratio <= 1:
                efficiency_ratio[name] = 0.0
            else:
                import math
                efficiency_ratio[name] = acc_gain / math.log(param_ratio + 1e-9)

        # Find where efficiency peaks
        peak_level = max(
            (n for n in level_names if n != "Tiny"),
            key=lambda n: efficiency_ratio[n],
        )
        # Check if peak is at Small or Medium (not Large/XL)
        peaks_early   = peak_level in ("Small", "Medium")
        # Check if efficiency declines at Large and XL relative to peak
        peak_val      = efficiency_ratio[peak_level]
        large_ratio   = efficiency_ratio["Large"]
        xl_ratio      = efficiency_ratio["XL"]
        declines_late = (large_ratio < peak_val) and (xl_ratio < peak_val)

        largest_is_best = (
            acc_per_complexity["XL"] >= max(
                acc_per_complexity[n] for n in level_names if n != "XL"
            )
        )

        all_equal = (
            max(acc_per_complexity.values()) - min(acc_per_complexity.values()) < 0.05
        )

        if peaks_early and declines_late:
            outcome = OUTCOME_SUPPORTED
        elif largest_is_best:
            outcome = OUTCOME_REFUTED
        elif all_equal:
            outcome = OUTCOME_INCONCLUSIVE
        else:
            outcome = OUTCOME_INCONCLUSIVE

        metrics = {
            "acc_per_complexity":            acc_per_complexity,
            "params_per_complexity":         params_per_complexity,
            "flops_per_complexity":          flops_per_complexity,
            "efficiency_ratio_per_complexity": efficiency_ratio,
            "optimal_complexity_level":      peak_level,
            "peaks_early":                   peaks_early,
            "declines_at_large_xl":          declines_late,
            "largest_is_best":               largest_is_best,
        }

        notes = (
            f"Peak efficiency at: {peak_level}. "
            f"Efficiency ratios: {efficiency_ratio}. "
            f"Acc: {acc_per_complexity}."
        )

        return self.result(outcome, metrics, notes, config={
            "train_steps": TRAIN_STEPS,
            "memory_slots": MEMORY_SLOTS,
            "complexity_widths": COMPLEXITY_WIDTHS,
        })


if __name__ == "__main__":
    Exp72ControllerOverheadBudget().execute()
