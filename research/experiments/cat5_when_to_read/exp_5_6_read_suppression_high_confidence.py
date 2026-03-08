"""
Experiment 5.6 — Read Suppression Under High Confidence

Hypothesis: Suppressing memory reads when next-token prediction confidence
exceeds a threshold costs less than 1% task quality.

Setup:
  - Threshold-based suppression.
  - When model confidence (max softmax probability) > T, skip memory retrieval.
  - Test thresholds T = 0.5, 0.6, 0.7, 0.8, 0.9
  - For each T: (1) accuracy on retrieval-needed task,
                (2) suppression rate (fraction of steps memory is skipped)
  - SUPPORTED if at some T, quality_cost < 0.01 AND suppression_rate > 0.3
  - REFUTED if suppression always hurts by > 0.01
  - INCONCLUSIVE if suppression rate is always low
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
HIDDEN_DIM    = 64
MEMORY_SLOTS  = 8
SEQ_LEN       = 24
BATCH_SIZE    = 32
TRAIN_STEPS   = 1500
LR            = 3e-4
DEVICE        = "cpu"
THRESHOLDS    = [0.5, 0.6, 0.7, 0.8, 0.9]


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


# ── Model ─────────────────────────────────────────────────────────────────────

class MemoryModel(nn.Module):
    """
    Token-by-token GRU with optional memory retrieval at each step.
    During inference, receives a threshold T: skip retrieval if max softmax > T.
    """

    def __init__(self):
        super().__init__()
        self.embed      = nn.Embedding(VOCAB_SIZE, HIDDEN_DIM)
        self.gru        = nn.GRUCell(HIDDEN_DIM + HIDDEN_DIM, HIDDEN_DIM)
        self.memory_mod = SoftAttentionMemory()
        self.head       = nn.Linear(HIDDEN_DIM, VOCAB_SIZE)

    def forward(
        self,
        seq: torch.Tensor,         # (B, L)
        memory: torch.Tensor,      # (B, M, H)
        suppress_threshold: float | None = None,
        # None = always read; float = suppress when confidence > T
    ) -> tuple[torch.Tensor, float]:
        """Returns final logits and suppression rate (fraction of steps skipped)."""
        B, L    = seq.shape
        h       = torch.zeros(B, HIDDEN_DIM, device=seq.device)
        retrieved = torch.zeros(B, HIDDEN_DIM, device=seq.device)

        suppressed_steps = 0
        total_steps      = L

        for t in range(L):
            x_t = self.embed(seq[:, t])
            inp = torch.cat([x_t, retrieved], dim=-1)
            h   = self.gru(inp, h)

            if suppress_threshold is not None:
                # Compute current prediction confidence
                with torch.no_grad():
                    logits_t  = self.head(h)
                    probs_t   = F.softmax(logits_t, dim=-1)
                    max_conf  = probs_t.max(dim=-1).values.mean().item()

                if max_conf > suppress_threshold:
                    # Suppress: keep previous retrieved (zero out to avoid stale info)
                    retrieved = torch.zeros_like(retrieved)
                    suppressed_steps += 1
                else:
                    retrieved = self.memory_mod(h, memory)
            else:
                retrieved = self.memory_mod(h, memory)

        suppression_rate = suppressed_steps / total_steps
        return self.head(h), suppression_rate


# ── Task ──────────────────────────────────────────────────────────────────────

def make_retrieval_task(batch_size: int):
    """
    Retrieval-needed task: answer stored in memory slot 0.
    Query trigger at position SEQ_LEN // 3.
    """
    B      = batch_size
    facts  = torch.randint(2, VOCAB_SIZE, (B,))
    seq    = torch.randint(2, VOCAB_SIZE, (B, SEQ_LEN))
    seq[:, SEQ_LEN // 3] = 1
    memory = torch.randn(B, MEMORY_SLOTS, HIDDEN_DIM) * 0.1
    for b in range(B):
        memory[b, 0, :] = 0.0
        memory[b, 0, facts[b].item() % HIDDEN_DIM] = 2.0
    return seq, facts, memory


# ── Train / Eval ──────────────────────────────────────────────────────────────

def train_model() -> MemoryModel:
    model = MemoryModel().to(DEVICE)
    opt   = Adam(model.parameters(), lr=LR)

    for _ in range(TRAIN_STEPS):
        seq, tgt, mem = make_retrieval_task(BATCH_SIZE)
        # Train with no suppression (always read)
        logits, _ = model(seq, mem, suppress_threshold=None)
        loss = F.cross_entropy(logits, tgt)
        opt.zero_grad(); loss.backward(); opt.step()

    return model


def eval_at_threshold(
    model: MemoryModel, threshold: float | None, n_batches: int = 50
) -> tuple[float, float]:
    """Returns (accuracy, suppression_rate)."""
    model.eval()
    correct = total = 0
    total_suppression = 0.0

    with torch.no_grad():
        for _ in range(n_batches):
            seq, tgt, mem = make_retrieval_task(BATCH_SIZE)
            logits, sup_rate = model(seq, mem, suppress_threshold=threshold)
            preds = logits.argmax(-1)
            correct += (preds == tgt).sum().item()
            total   += tgt.size(0)
            total_suppression += sup_rate

    model.train()
    return correct / total, total_suppression / n_batches


# ── Experiment ────────────────────────────────────────────────────────────────

class Exp56ReadSuppressionHighConfidence(Experiment):
    experiment_id = "exp_5_6"
    hypothesis = (
        "Suppressing memory reads when next-token prediction confidence "
        "exceeds a threshold costs less than 1% task quality."
    )

    def run(self) -> ExperimentResult:
        torch.manual_seed(42)

        config = {
            "vocab_size":   VOCAB_SIZE,
            "hidden_dim":   HIDDEN_DIM,
            "memory_slots": MEMORY_SLOTS,
            "seq_len":      SEQ_LEN,
            "batch_size":   BATCH_SIZE,
            "train_steps":  TRAIN_STEPS,
            "thresholds":   THRESHOLDS,
        }

        print("\n  Training memory model (no suppression during training)...")
        model = train_model()

        # Baseline: no suppression
        baseline_acc, _ = eval_at_threshold(model, threshold=None)
        print(f"    baseline_acc (no suppression) = {baseline_acc:.3f}")

        metrics: dict = {"baseline_acc": round(baseline_acc, 4)}

        threshold_results: list[dict] = []

        for T in THRESHOLDS:
            acc, sup_rate = eval_at_threshold(model, threshold=T)
            quality_cost  = baseline_acc - acc
            t_key         = str(int(T * 100))

            metrics[f"acc_at_T_{t_key}"]             = round(acc, 4)
            metrics[f"suppression_rate_at_T_{t_key}"] = round(sup_rate, 4)

            print(
                f"    T={T:.1f}  acc={acc:.3f}  "
                f"sup_rate={sup_rate:.3f}  quality_cost={quality_cost:.4f}"
            )
            threshold_results.append({
                "T": T, "acc": acc, "sup_rate": sup_rate,
                "quality_cost": quality_cost,
            })

        # Find best threshold: quality_cost < 0.01 AND sup_rate > 0.3
        optimal_T         = None
        optimal_cost      = float("inf")
        supported_at_any_T = False

        for r in threshold_results:
            if r["quality_cost"] < 0.01 and r["sup_rate"] > 0.3:
                supported_at_any_T = True
                if r["quality_cost"] < optimal_cost:
                    optimal_cost = r["quality_cost"]
                    optimal_T    = r["T"]

        # If no single T satisfies both criteria, pick the T with lowest cost
        if optimal_T is None:
            best = min(threshold_results, key=lambda r: r["quality_cost"])
            optimal_T    = best["T"]
            optimal_cost = best["quality_cost"]

        metrics["optimal_T"]               = optimal_T
        metrics["quality_cost_at_optimal_T"] = round(optimal_cost, 4)

        # Suppression rate at the chosen optimal T
        optimal_sup = next(
            r["sup_rate"] for r in threshold_results if r["T"] == optimal_T
        )

        # Determine outcome
        always_hurts = all(r["quality_cost"] > 0.01 for r in threshold_results)
        always_low_sup = all(r["sup_rate"] <= 0.3 for r in threshold_results)

        if supported_at_any_T:
            outcome = OUTCOME_SUPPORTED
        elif always_hurts:
            outcome = OUTCOME_REFUTED
        elif always_low_sup:
            outcome = OUTCOME_INCONCLUSIVE
        else:
            outcome = OUTCOME_INCONCLUSIVE

        notes = (
            f"Baseline acc (no suppression): {baseline_acc:.3f}. "
            f"Optimal T={optimal_T}, quality_cost={optimal_cost:.4f}, "
            f"suppression_rate={optimal_sup:.3f}. "
            f"SUPPORTED criterion: quality_cost<0.01 AND sup_rate>0.3 "
            f"met at any T: {supported_at_any_T}."
        )

        return self.result(outcome, metrics, notes, config)


if __name__ == "__main__":
    Exp56ReadSuppressionHighConfidence().execute()
