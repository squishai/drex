"""
Experiment 7.9 — Controller Interpretability Baseline

Hypothesis: The controller's write and read decisions are interpretable
(non-random, correlating with human-meaningful features) in their simplest
form before any task-specific training.

This experiment should be run alongside everything else from day one. Its job is
to build the interpretability tooling and establish what "interpretable" means
as a baseline before things get complex.

Setup:
  - Simplest possible controller (1-layer linear gate)
  - Run against sequences with known structure: position, token frequency,
    punctuation markers, numeric tokens, repeating patterns
  - Measure correlation between gate firing and each structural feature
  - Visualize write/read decisions as heatmaps over token positions
  - Key question: does the gate fire non-randomly even before task training?

Output:
  - Correlation table: gate_activity vs. each structural feature
  - Saved heatmap PNGs in research/results/interpretability/
  - Baseline interpretability score (0–1)
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from experiments.base import (
    Experiment, ExperimentResult,
    OUTCOME_SUPPORTED, OUTCOME_REFUTED, OUTCOME_INCONCLUSIVE,
)

INTERP_DIR = Path(__file__).parent.parent.parent / "results" / "interpretability"
INTERP_DIR.mkdir(parents=True, exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────

VOCAB_SIZE     = 128
SEQ_LEN        = 32
HIDDEN_DIM     = 64
MEMORY_SLOTS   = 8
TRAIN_STEPS    = 1000
N_PROBE_SEQS   = 500
BATCH_SIZE     = 32
LR             = 3e-4
DEVICE         = "cpu"


# ── Token feature definitions ─────────────────────────────────────────────────
# Token IDs carry structural meaning in our synthetic corpus:
#   0:        PAD
#   1..9:     punctuation/boundary markers
#   10..19:   numeric tokens (high information)
#   20..63:   common tokens (high frequency)
#   64..127:  rare tokens (low frequency, high surprise)

PUNCT_IDS  = set(range(1, 10))
NUM_IDS    = set(range(10, 20))
COMMON_IDS = set(range(20, 64))
RARE_IDS   = set(range(64, 128))


def is_punct(t: int) -> float:
    return 1.0 if t in PUNCT_IDS else 0.0

def is_numeric(t: int) -> float:
    return 1.0 if t in NUM_IDS else 0.0

def is_rare(t: int) -> float:
    return 1.0 if t in RARE_IDS else 0.0


def make_structured_seq(batch_size: int) -> torch.Tensor:
    """Generate sequences with mixed token categories."""
    seqs = torch.zeros(batch_size, SEQ_LEN, dtype=torch.long)
    for b in range(batch_size):
        for i in range(SEQ_LEN):
            r = torch.rand(1).item()
            if r < 0.10:   seqs[b, i] = torch.randint(1,  10, (1,))
            elif r < 0.20: seqs[b, i] = torch.randint(10, 20, (1,))
            elif r < 0.70: seqs[b, i] = torch.randint(20, 64, (1,))
            else:          seqs[b, i] = torch.randint(64, VOCAB_SIZE, (1,))
    return seqs


# ── Minimal controller ────────────────────────────────────────────────────────

class MinimalWriteGate(nn.Module):
    """Absolute minimum: one linear layer over token embeddings."""
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, HIDDEN_DIM)
        self.gate  = nn.Linear(HIDDEN_DIM, 1)

    def forward(self, seq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h    = self.embed(seq)
        soft = torch.sigmoid(self.gate(h)).squeeze(-1)
        return soft, h


class MinimalReadGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.query_proj = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.gate       = nn.Linear(HIDDEN_DIM * 2, 1)

    def forward(self, ctx: torch.Tensor, mem: torch.Tensor) -> torch.Tensor:
        """ctx: (B,H), mem: (B,M,H) → read gate probabilities (B,M)."""
        ctx_exp = ctx.unsqueeze(1).expand_as(mem)
        pairs   = torch.cat([ctx_exp, mem], dim=-1)
        return torch.sigmoid(self.gate(pairs)).squeeze(-1)


class MinimalController(nn.Module):
    def __init__(self):
        super().__init__()
        self.write_gate = MinimalWriteGate()
        self.read_gate  = MinimalReadGate()
        self.head       = nn.Linear(HIDDEN_DIM, VOCAB_SIZE)

    def forward(self, seq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        write_prob, h = self.write_gate(seq)      # (B, L), (B, L, H)

        topk_idx = write_prob.topk(MEMORY_SLOTS, dim=-1).indices
        memory   = h.gather(1, topk_idx.unsqueeze(-1).expand(-1, -1, HIDDEN_DIM))

        ctx = h.mean(1)
        read_prob = self.read_gate(ctx, memory)   # (B, M)
        retrieved = (read_prob.unsqueeze(-1) * memory).sum(1) / (read_prob.sum(1, keepdim=True) + 1e-8)

        logits = self.head(retrieved)
        return logits, write_prob, read_prob


# ── Correlation analysis ───────────────────────────────────────────────────────

def pearson_r(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation between two 1D arrays."""
    if a.std() < 1e-8 or b.std() < 1e-8:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def probe_gate_correlations(model: MinimalController, n_seqs: int) -> dict:
    """
    For each token position across n_seqs sequences, record:
      - write gate probability
      - structural features: is_punct, is_numeric, is_rare, position_normalized
    Returns dict of feature → pearson_r with gate probability.
    """
    model.eval()
    all_gate  = []
    all_punct = []
    all_num   = []
    all_rare  = []
    all_pos   = []

    with torch.no_grad():
        for _ in range(n_seqs // BATCH_SIZE):
            seq = make_structured_seq(BATCH_SIZE)
            _, write_prob, _ = model(seq)

            all_gate.extend(write_prob.view(-1).tolist())
            for b in range(BATCH_SIZE):
                for i in range(SEQ_LEN):
                    t = seq[b, i].item()
                    all_punct.append(is_punct(t))
                    all_num.append(is_numeric(t))
                    all_rare.append(is_rare(t))
                    all_pos.append(i / (SEQ_LEN - 1))

    g = np.array(all_gate)
    return {
        "corr_punct_vs_gate":    pearson_r(np.array(all_punct), g),
        "corr_numeric_vs_gate":  pearson_r(np.array(all_num), g),
        "corr_rare_vs_gate":     pearson_r(np.array(all_rare), g),
        "corr_position_vs_gate": pearson_r(np.array(all_pos), g),
        "gate_mean":             float(g.mean()),
        "gate_std":              float(g.std()),
        "gate_nonrandom":        float(g.std()) > 0.05,
    }


def save_heatmap_data(model: MinimalController, label: str) -> None:
    """Save gate activity as JSON heatmap (avg over batch)."""
    model.eval()
    avg_write = torch.zeros(SEQ_LEN)
    n = 20
    with torch.no_grad():
        for _ in range(n):
            seq = make_structured_seq(BATCH_SIZE)
            _, write_prob, _ = model(seq)
            avg_write += write_prob.mean(0).cpu()
    avg_write /= n
    out = {"label": label, "write_gate_avg": avg_write.tolist()}
    path = INTERP_DIR / f"heatmap_{label}.json"
    path.write_text(json.dumps(out, indent=2))


# ── Experiment ────────────────────────────────────────────────────────────────

class Exp79InterpretabilityBaseline(Experiment):
    experiment_id = "exp_7_9"
    hypothesis = (
        "The controller's write and read decisions are interpretable "
        "(non-random, correlating with human-meaningful features) in their "
        "simplest form before any task-specific training."
    )

    def run(self) -> ExperimentResult:
        torch.manual_seed(42)
        model = MinimalController().to(DEVICE)

        # ── Probe BEFORE any training ──────────────────────────────────────
        corrs_untrained = probe_gate_correlations(model, N_PROBE_SEQS)
        save_heatmap_data(model, "untrained")
        print("\n  Untrained controller correlations:")
        for k, v in corrs_untrained.items():
            print(f"    {k}: {v:.4f}")

        # ── Train on next-token prediction ─────────────────────────────────
        opt = Adam(model.parameters(), lr=LR)
        for _ in range(TRAIN_STEPS):
            seq = make_structured_seq(BATCH_SIZE)
            logits, _, _ = model(seq)
            target = seq[:, -1]
            loss = F.cross_entropy(logits, target)
            opt.zero_grad(); loss.backward(); opt.step()

        # ── Probe AFTER training ───────────────────────────────────────────
        corrs_trained = probe_gate_correlations(model, N_PROBE_SEQS)
        save_heatmap_data(model, "trained")
        print("\n  Trained controller correlations:")
        for k, v in corrs_trained.items():
            print(f"    {k}: {v:.4f}")

        # Interpretability score: mean absolute correlation with structural features
        structural_keys = ["corr_punct_vs_gate", "corr_numeric_vs_gate",
                           "corr_rare_vs_gate"]
        score_trained = np.mean([abs(corrs_trained[k]) for k in structural_keys])
        score_untrained = np.mean([abs(corrs_untrained[k]) for k in structural_keys])

        # Non-random if gate has non-trivial variance AND correlates with features
        interpretable = corrs_trained["gate_nonrandom"] and score_trained > 0.05

        print(f"\n  Interpretability score (trained): {score_trained:.4f}")
        print(f"  Interpretability score (untrained): {score_untrained:.4f}")

        if interpretable:
            outcome = OUTCOME_SUPPORTED
        elif corrs_trained["gate_nonrandom"]:
            outcome = OUTCOME_INCONCLUSIVE
        else:
            outcome = OUTCOME_REFUTED

        return self.result(outcome, {
            "untrained": corrs_untrained,
            "trained":   corrs_trained,
            "interpretability_score_trained":   score_trained,
            "interpretability_score_untrained": score_untrained,
            "heatmaps_saved_to": str(INTERP_DIR),
        }, notes=(
            f"Interp score: trained={score_trained:.3f} untrained={score_untrained:.3f}. "
            f"Gate is {'non-random' if corrs_trained['gate_nonrandom'] else 'random'}. "
            f"Heatmaps saved to results/interpretability/."
        ), config={"train_steps": TRAIN_STEPS, "n_probe_seqs": N_PROBE_SEQS})


if __name__ == "__main__":
    Exp79InterpretabilityBaseline().execute()
