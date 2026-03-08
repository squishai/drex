"""
Experiment 5.3 — Predictive Read Triggering

Hypothesis: Anticipatory retrieval (predicting need before it arises) improves
latency without measurably hurting task quality.

Setup:
  - Policy A (Reactive): read gate fires when current token prediction
    confidence is LOW (entropy > threshold). Reads after need is manifest.
  - Policy B (Predictive): small LSTM observes last 4 hidden states and
    predicts 2 steps ahead whether retrieval will be needed. Fires retrieval
    before the need arrives.
  - Both have MEMORY_SLOTS=8
  - Measure: (1) accuracy on retrieval-dependent tasks, (2) read rate
  - SUPPORTED if predictive_acc >= reactive_acc - 0.01 AND
                  predictive_read_rate < reactive_read_rate
  - REFUTED if predictive_acc < reactive_acc - 0.02
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

VOCAB_SIZE    = 64
HIDDEN_DIM    = 64
MEMORY_SLOTS  = 8
SEQ_LEN       = 24
BATCH_SIZE    = 32
TRAIN_STEPS   = 1500
LR            = 3e-4
DEVICE        = "cpu"
ENTROPY_THRESH = 2.5   # reactive trigger: entropy above this => read
PRED_HORIZON   = 2     # predictive policy looks 2 steps ahead
HIST_LEN       = 4     # LSTM observes last 4 hidden states


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


# ── Reactive Model ────────────────────────────────────────────────────────────

class ReactiveModel(nn.Module):
    """
    Processes tokens with a GRU. After each step, computes prediction entropy.
    If entropy > threshold, reads from memory for the NEXT step.
    """

    def __init__(self):
        super().__init__()
        self.embed      = nn.Embedding(VOCAB_SIZE, HIDDEN_DIM)
        self.gru        = nn.GRUCell(HIDDEN_DIM + HIDDEN_DIM, HIDDEN_DIM)
        self.memory_mod = SoftAttentionMemory()
        self.head       = nn.Linear(HIDDEN_DIM, VOCAB_SIZE)

    def forward(self, seq: torch.Tensor,
                memory: torch.Tensor) -> tuple[torch.Tensor, float]:
        """Returns final logits and read_rate."""
        B, L = seq.shape
        h          = torch.zeros(B, HIDDEN_DIM, device=seq.device)
        retrieved  = torch.zeros(B, HIDDEN_DIM, device=seq.device)
        read_count = 0

        for t in range(L):
            x_t = self.embed(seq[:, t])
            inp = torch.cat([x_t, retrieved], dim=-1)
            h   = self.gru(inp, h)

            # Compute entropy of current prediction
            with torch.no_grad():
                logits_t = self.head(h)
                probs    = F.softmax(logits_t, dim=-1)
                entropy  = -(probs * (probs + 1e-8).log()).sum(-1)  # (B,)

            # Read if mean entropy > threshold
            if entropy.mean().item() > ENTROPY_THRESH:
                retrieved   = self.memory_mod(h, memory)
                read_count += 1
            else:
                retrieved = torch.zeros_like(retrieved)

        read_rate = read_count / L
        return self.head(h), read_rate


# ── Predictive Model ──────────────────────────────────────────────────────────

class PredictiveTrigger(nn.Module):
    """
    LSTM that takes the last HIST_LEN hidden states and predicts whether
    memory will be needed 2 steps ahead (binary signal).
    """

    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(HIDDEN_DIM, 32, batch_first=True)
        self.head  = nn.Linear(32, 1)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        """history: (B, HIST_LEN, H) -> (B,) probability of needing memory."""
        out, _ = self.lstm(history)
        return torch.sigmoid(self.head(out[:, -1, :]).squeeze(-1))


class PredictiveModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed     = nn.Embedding(VOCAB_SIZE, HIDDEN_DIM)
        self.gru       = nn.GRUCell(HIDDEN_DIM + HIDDEN_DIM, HIDDEN_DIM)
        self.mem_mod   = SoftAttentionMemory()
        self.predictor = PredictiveTrigger()
        self.head      = nn.Linear(HIDDEN_DIM, VOCAB_SIZE)

    def forward(self, seq: torch.Tensor,
                memory: torch.Tensor) -> tuple[torch.Tensor, float]:
        B, L   = seq.shape
        h          = torch.zeros(B, HIDDEN_DIM, device=seq.device)
        retrieved  = torch.zeros(B, HIDDEN_DIM, device=seq.device)
        h_history: list[torch.Tensor] = []
        read_count = 0

        for t in range(L):
            x_t = self.embed(seq[:, t])
            inp = torch.cat([x_t, retrieved], dim=-1)
            h   = self.gru(inp, h)
            h_history.append(h.detach())
            if len(h_history) > HIST_LEN:
                h_history.pop(0)

            # Predictive trigger: if we have enough history, predict ahead
            if len(h_history) == HIST_LEN:
                hist_tensor = torch.stack(h_history, dim=1)  # (B, HIST_LEN, H)
                need_prob   = self.predictor(hist_tensor)     # (B,)
                if need_prob.mean().item() > 0.5:
                    retrieved   = self.mem_mod(h, memory)
                    read_count += 1
                else:
                    retrieved = torch.zeros_like(retrieved)

        read_rate = read_count / L
        return self.head(h), read_rate


# ── Task ──────────────────────────────────────────────────────────────────────

def make_retrieval_task(batch_size: int):
    """
    Retrieval-dependent task: fact stored in memory slot 0.
    Query trigger appears at SEQ_LEN // 2, answer required at end.
    """
    B      = batch_size
    facts  = torch.randint(2, VOCAB_SIZE, (B,))
    seq    = torch.randint(2, VOCAB_SIZE, (B, SEQ_LEN))
    seq[:, SEQ_LEN // 2] = 1   # query trigger
    memory = torch.randn(B, MEMORY_SLOTS, HIDDEN_DIM) * 0.1
    for b in range(B):
        memory[b, 0, :] = 0.0
        memory[b, 0, facts[b].item() % HIDDEN_DIM] = 2.0
    return seq, facts, memory


# ── Train / Eval ──────────────────────────────────────────────────────────────

def train_reactive() -> ReactiveModel:
    model = ReactiveModel().to(DEVICE)
    opt   = Adam(model.parameters(), lr=LR)

    for _ in range(TRAIN_STEPS):
        seq, tgt, mem = make_retrieval_task(BATCH_SIZE)
        logits, _ = model(seq, mem)
        loss = F.cross_entropy(logits, tgt)
        opt.zero_grad(); loss.backward(); opt.step()

    return model


def train_predictive() -> PredictiveModel:
    model = PredictiveModel().to(DEVICE)
    opt   = Adam(model.parameters(), lr=LR)

    for step in range(TRAIN_STEPS):
        seq, tgt, mem = make_retrieval_task(BATCH_SIZE)
        logits, _ = model(seq, mem)
        task_loss = F.cross_entropy(logits, tgt)
        loss = task_loss
        opt.zero_grad(); loss.backward(); opt.step()

    return model


def eval_model(model: nn.Module, n_batches: int = 50) -> tuple[float, float]:
    """Returns (accuracy, mean_read_rate)."""
    model.eval()
    correct = total = 0
    total_read_rate = 0.0

    with torch.no_grad():
        for _ in range(n_batches):
            seq, tgt, mem = make_retrieval_task(BATCH_SIZE)
            logits, read_rate = model(seq, mem)
            preds = logits.argmax(-1)
            correct += (preds == tgt).sum().item()
            total   += tgt.size(0)
            total_read_rate += read_rate

    model.train()
    return correct / total, total_read_rate / n_batches


# ── Experiment ────────────────────────────────────────────────────────────────

class Exp53PredictiveReadTriggering(Experiment):
    experiment_id = "exp_5_3"
    hypothesis = (
        "Anticipatory retrieval (predicting need before it arises) improves "
        "latency without measurably hurting task quality."
    )

    def run(self) -> ExperimentResult:
        torch.manual_seed(42)

        config = {
            "vocab_size":    VOCAB_SIZE,
            "hidden_dim":    HIDDEN_DIM,
            "memory_slots":  MEMORY_SLOTS,
            "seq_len":       SEQ_LEN,
            "batch_size":    BATCH_SIZE,
            "train_steps":   TRAIN_STEPS,
            "entropy_thresh": ENTROPY_THRESH,
            "pred_horizon":  PRED_HORIZON,
            "hist_len":      HIST_LEN,
        }

        print("\n  Training reactive model...")
        reactive_model = train_reactive()
        reactive_acc, reactive_read_rate = eval_model(reactive_model)
        print(f"    reactive_acc={reactive_acc:.3f}  read_rate={reactive_read_rate:.3f}")

        print("  Training predictive model...")
        predictive_model = train_predictive()
        predictive_acc, predictive_read_rate = eval_model(predictive_model)
        print(f"    predictive_acc={predictive_acc:.3f}  read_rate={predictive_read_rate:.3f}")

        acc_gap = reactive_acc - predictive_acc

        metrics = {
            "reactive_acc":          round(reactive_acc, 4),
            "predictive_acc":        round(predictive_acc, 4),
            "reactive_read_rate":    round(reactive_read_rate, 4),
            "predictive_read_rate":  round(predictive_read_rate, 4),
            "acc_gap":               round(acc_gap, 4),
        }

        if predictive_acc >= reactive_acc - 0.01 and predictive_read_rate < reactive_read_rate:
            outcome = OUTCOME_SUPPORTED
        elif predictive_acc < reactive_acc - 0.02:
            outcome = OUTCOME_REFUTED
        else:
            outcome = OUTCOME_INCONCLUSIVE

        notes = (
            f"Reactive: acc={reactive_acc:.3f}, read_rate={reactive_read_rate:.3f}. "
            f"Predictive: acc={predictive_acc:.3f}, read_rate={predictive_read_rate:.3f}. "
            f"Acc gap: {acc_gap:.4f} (positive = reactive better). "
            f"Read rate reduction: {reactive_read_rate - predictive_read_rate:.4f}."
        )

        return self.result(outcome, metrics, notes, config)


if __name__ == "__main__":
    Exp53PredictiveReadTriggering().execute()
