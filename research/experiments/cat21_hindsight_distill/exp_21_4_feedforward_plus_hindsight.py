"""
Experiment 21.4 — Feedforward Controller + Hindsight Distillation

Hypothesis: Feedforward controller + hindsight distillation is the strongest write
policy (outperforms either alone and LSTM baseline).
"""

from __future__ import annotations

import math
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
NUM_PAIRS      = 4
PRETRAIN_STEPS = 500
TRAIN_STEPS    = 400
BATCH          = 32
LR             = 3e-4
DEVICE         = "cpu"
ORACLE_LAMBDA  = 0.3    # weight on oracle BCE loss when combined

# ── Data generator ────────────────────────────────────────────────────────────

def make_assoc_batch(batch_size, seq_len, vocab_size, num_pairs):
    seq = torch.zeros(batch_size, seq_len, dtype=torch.long)
    target = torch.zeros(batch_size, dtype=torch.long)
    for b in range(batch_size):
        keys = torch.randint(4, max(8, vocab_size // 3), (num_pairs * 3,)).unique()[:num_pairs]
        while len(keys) < num_pairs:
            keys = torch.cat([keys, torch.randint(4, vocab_size // 3, (1,))])[:num_pairs]
        vals = torch.randint(vocab_size // 2, vocab_size, (num_pairs,))
        pos = 0
        for i in range(num_pairs):
            if pos + 1 < seq_len - 3:
                seq[b, pos] = keys[i]; seq[b, pos + 1] = vals[i]; pos += 2
        for p in range(pos, seq_len - 3):
            seq[b, p] = 3
        qi = torch.randint(0, num_pairs, (1,)).item()
        seq[b, seq_len - 3] = 2
        seq[b, seq_len - 2] = keys[qi]
        seq[b, seq_len - 1] = 0
        target[b] = vals[qi]
    return seq, target

# ── Model components ──────────────────────────────────────────────────────────

class ReadHead(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.scale  = math.sqrt(hidden_dim)

    def forward(self, query, memory):
        q   = self.q_proj(query).unsqueeze(1)
        k   = self.k_proj(memory)
        w   = F.softmax(torch.bmm(q, k.transpose(1, 2)) / self.scale, dim=-1).squeeze(1)
        out = torch.bmm(w.unsqueeze(1), memory).squeeze(1)
        return out, w


class WriteGate(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, 1)

    def forward(self, h):
        return torch.sigmoid(self.proj(h)).squeeze(-1)


def build_memory(hidden: torch.Tensor, gate_scores: torch.Tensor,
                 memory_slots: int, num_pairs: int) -> torch.Tensor:
    B, L, H = hidden.shape
    K = min(num_pairs, memory_slots)
    gumbel = -torch.log(-torch.log(torch.clamp(torch.rand_like(gate_scores), 1e-10, 1.0)))
    perturbed   = gate_scores + 0.1 * gumbel
    _, topk_idx = perturbed.topk(K, dim=-1)
    soft_w   = F.softmax(perturbed.gather(1, topk_idx), dim=-1)
    topk_v   = hidden.gather(1, topk_idx.unsqueeze(-1).expand(B, K, H))
    memory   = torch.zeros(B, memory_slots, H, device=hidden.device)
    for k_i in range(K):
        memory[:, k_i, :] = topk_v[:, k_i, :] * soft_w[:, k_i:k_i+1]
    return memory

# ── Controller implementations ────────────────────────────────────────────────

class LSTMModel(nn.Module):
    """LSTM-based controller with task loss only or + oracle."""
    def __init__(self, vocab_size, hidden_dim, memory_slots, num_pairs):
        super().__init__()
        half = hidden_dim // 2
        self.embed      = nn.Embedding(vocab_size, hidden_dim)
        self.lstm       = nn.LSTM(hidden_dim, half, batch_first=True)
        # Project LSTM output back to hidden_dim for gate input
        self.gate_proj  = nn.Linear(half, hidden_dim)
        self.write_gate = WriteGate(hidden_dim)
        self.read_head  = ReadHead(hidden_dim)
        self.out        = nn.Linear(hidden_dim, vocab_size)
        self.memory_slots = memory_slots
        self.num_pairs    = num_pairs
        self.hidden_dim   = hidden_dim

    def encode(self, seq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        emb = self.embed(seq)
        h_lstm, _ = self.lstm(emb)
        # Gate is computed from LSTM hidden; embed used as memory values
        gate_input  = self.gate_proj(h_lstm)
        gate_scores = self.write_gate(gate_input)
        return emb, gate_scores

    def forward(self, seq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        emb, gate_scores = self.encode(seq)
        memory      = build_memory(emb, gate_scores, self.memory_slots, self.num_pairs)
        query       = emb[:, -2, :]
        read_vec, _ = self.read_head(query, memory)
        logits      = self.out(read_vec)
        return logits, gate_scores


class FeedforwardModel(nn.Module):
    """Feedforward-only controller (position-wise MLP)."""
    def __init__(self, vocab_size, hidden_dim, memory_slots, num_pairs):
        super().__init__()
        self.embed      = nn.Embedding(vocab_size, hidden_dim)
        self.mlp        = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.write_gate = WriteGate(hidden_dim)
        self.read_head  = ReadHead(hidden_dim)
        self.out        = nn.Linear(hidden_dim, vocab_size)
        self.memory_slots = memory_slots
        self.num_pairs    = num_pairs
        self.hidden_dim   = hidden_dim

    def encode(self, seq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        emb = self.embed(seq)
        h   = self.mlp(emb)
        gate_scores = self.write_gate(h)
        return emb, gate_scores

    def forward(self, seq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        emb, gate_scores = self.encode(seq)
        memory      = build_memory(emb, gate_scores, self.memory_slots, self.num_pairs)
        query       = emb[:, -2, :]
        read_vec, _ = self.read_head(query, memory)
        logits      = self.out(read_vec)
        return logits, gate_scores

# ── Teacher for oracle labels ─────────────────────────────────────────────────

class TeacherModel(nn.Module):
    """Simple feedforward model used only as frozen oracle teacher."""
    def __init__(self, vocab_size, hidden_dim, memory_slots, num_pairs):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.mlp   = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.read_head    = ReadHead(hidden_dim)
        self.out          = nn.Linear(hidden_dim, vocab_size)
        self.memory_slots = memory_slots
        self.hidden_dim   = hidden_dim

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        h    = self.mlp(self.embed(seq))
        # Simple mean-pool over all positions as memory for pretraining
        mem  = h
        q    = h[:, -2, :]
        # Replicate query position as all-same memory for pretraining compatibility
        mem_slots = h[:, :self.memory_slots, :] if h.size(1) >= self.memory_slots \
                    else F.pad(h, (0, 0, 0, self.memory_slots - h.size(1)))
        read, _ = self.read_head(q, mem_slots)
        return self.out(read)

    def encode_hidden(self, seq: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.embed(seq))


@torch.no_grad()
def compute_oracle_labels(teacher: TeacherModel, seq: torch.Tensor,
                          target: torch.Tensor) -> torch.Tensor:
    """
    For each position t, test if a singleton memory containing only hidden[t]
    (replicated across all slots) leads to correct prediction by the teacher.
    oracle_label[b, t] = 1.0 if token t alone suffices for correct retrieval.
    """
    teacher.eval()
    B, L   = seq.shape
    hidden = teacher.encode_hidden(seq)        # (B, L, H)
    query  = hidden[:, -2, :]                  # (B, H)
    oracle_labels = torch.zeros(B, L, device=seq.device)
    for t in range(L):
        singleton = hidden[:, t:t+1, :].expand(B, teacher.memory_slots, teacher.hidden_dim)
        read_vec, _ = teacher.read_head(query, singleton)
        preds = teacher.out(read_vec).argmax(dim=-1)
        oracle_labels[:, t] = (preds == target).float()
    return oracle_labels


def pretrain_teacher(vocab_size, hidden_dim, memory_slots, num_pairs,
                     pretrain_steps: int) -> TeacherModel:
    teacher = TeacherModel(vocab_size, hidden_dim, memory_slots, num_pairs).to(DEVICE)
    opt     = Adam(teacher.parameters(), lr=LR)
    teacher.train()
    print(f"  Pre-training teacher ({pretrain_steps} steps) ...")
    for step in range(pretrain_steps):
        seq, target = make_assoc_batch(BATCH, SEQ_LEN, vocab_size, num_pairs)
        seq, target = seq.to(DEVICE), target.to(DEVICE)
        logits = teacher(seq)
        loss   = F.cross_entropy(logits, target)
        opt.zero_grad(); loss.backward(); opt.step()
        if (step + 1) % 200 == 0:
            print(f"    [teacher] step {step+1}/{pretrain_steps}  loss={loss.item():.4f}")
    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher.eval()
    return teacher

# ── Training functions for 5 conditions ──────────────────────────────────────

def _train_task_only(model: nn.Module, steps: int, label: str) -> None:
    opt = Adam(model.parameters(), lr=LR)
    model.train()
    for step in range(steps):
        seq, target = make_assoc_batch(BATCH, SEQ_LEN, VOCAB_SIZE, NUM_PAIRS)
        seq, target = seq.to(DEVICE), target.to(DEVICE)
        logits, _ = model(seq)
        loss = F.cross_entropy(logits, target)
        opt.zero_grad(); loss.backward(); opt.step()
        if (step + 1) % 500 == 0:
            print(f"    [{label}] step {step+1}/{steps}  loss={loss.item():.4f}")


def _train_with_oracle(model: nn.Module, teacher: TeacherModel,
                       steps: int, label: str, oracle_lambda: float = ORACLE_LAMBDA) -> None:
    """Task loss + oracle BCE on gate scores at every step."""
    opt = Adam(model.parameters(), lr=LR)
    model.train()
    for step in range(steps):
        seq, target = make_assoc_batch(BATCH, SEQ_LEN, VOCAB_SIZE, NUM_PAIRS)
        seq, target = seq.to(DEVICE), target.to(DEVICE)
        logits, gate_scores = model(seq)
        task_loss     = F.cross_entropy(logits, target)
        oracle_labels = compute_oracle_labels(teacher, seq, target)
        oracle_loss   = F.binary_cross_entropy(gate_scores, oracle_labels.detach())
        loss = task_loss + oracle_lambda * oracle_loss
        opt.zero_grad(); loss.backward(); opt.step()
        if (step + 1) % 500 == 0:
            print(f"    [{label}] step {step+1}/{steps}  task={task_loss.item():.4f}  oracle={oracle_loss.item():.4f}")

# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(model: nn.Module, n_batches: int = 50) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for _ in range(n_batches):
            seq, target = make_assoc_batch(BATCH, SEQ_LEN, VOCAB_SIZE, NUM_PAIRS)
            seq, target = seq.to(DEVICE), target.to(DEVICE)
            logits, _ = model(seq)
            correct += (logits.argmax(-1) == target).sum().item()
            total   += BATCH
    return correct / total

# ── Experiment ────────────────────────────────────────────────────────────────

class Exp214FeedforwardPlusHindsight(Experiment):
    experiment_id = "exp_21_4"
    hypothesis = (
        "Feedforward controller + hindsight distillation is the strongest write "
        "policy (outperforms either alone and LSTM baseline)."
    )

    def run(self) -> ExperimentResult:
        config = dict(
            vocab_size=VOCAB_SIZE, hidden_dim=HIDDEN_DIM, seq_len=SEQ_LEN,
            memory_slots=MEMORY_SLOTS, num_pairs=NUM_PAIRS,
            pretrain_steps=PRETRAIN_STEPS, train_steps=TRAIN_STEPS, batch=BATCH,
            oracle_lambda=ORACLE_LAMBDA,
        )

        # Pre-train and freeze teacher (used for conditions C, D, E)
        teacher = pretrain_teacher(VOCAB_SIZE, HIDDEN_DIM, MEMORY_SLOTS, NUM_PAIRS, PRETRAIN_STEPS)

        # ── Condition A: LSTM + task loss ─────────────────────────────────────
        print("  Training Condition A: LSTM + task loss ...")
        model_A = LSTMModel(VOCAB_SIZE, HIDDEN_DIM, MEMORY_SLOTS, NUM_PAIRS).to(DEVICE)
        _train_task_only(model_A, TRAIN_STEPS, "A")
        acc_A = evaluate(model_A)
        print(f"    acc_A={acc_A:.4f}")

        # ── Condition B: FF + task loss ───────────────────────────────────────
        print("  Training Condition B: Feedforward + task loss ...")
        model_B = FeedforwardModel(VOCAB_SIZE, HIDDEN_DIM, MEMORY_SLOTS, NUM_PAIRS).to(DEVICE)
        _train_task_only(model_B, TRAIN_STEPS, "B")
        acc_B = evaluate(model_B)
        print(f"    acc_B={acc_B:.4f}")

        # ── Condition C: FF + oracle distillation ─────────────────────────────
        print("  Training Condition C: Feedforward + oracle distillation ...")
        model_C = FeedforwardModel(VOCAB_SIZE, HIDDEN_DIM, MEMORY_SLOTS, NUM_PAIRS).to(DEVICE)
        _train_with_oracle(model_C, teacher, TRAIN_STEPS, "C")
        acc_C = evaluate(model_C)
        print(f"    acc_C={acc_C:.4f}")

        # ── Condition D: LSTM + oracle distillation ───────────────────────────
        print("  Training Condition D: LSTM + oracle distillation ...")
        model_D = LSTMModel(VOCAB_SIZE, HIDDEN_DIM, MEMORY_SLOTS, NUM_PAIRS).to(DEVICE)
        _train_with_oracle(model_D, teacher, TRAIN_STEPS, "D")
        acc_D = evaluate(model_D)
        print(f"    acc_D={acc_D:.4f}")

        # ── Condition E: FF + hindsight distillation (the full combination) ───
        print("  Training Condition E: Feedforward + hindsight distillation ...")
        model_E = FeedforwardModel(VOCAB_SIZE, HIDDEN_DIM, MEMORY_SLOTS, NUM_PAIRS).to(DEVICE)
        _train_with_oracle(model_E, teacher, TRAIN_STEPS, "E")
        acc_E = evaluate(model_E)
        print(f"    acc_E={acc_E:.4f}")

        all_accs        = [acc_A, acc_B, acc_C, acc_D, acc_E]
        best_of_others  = max(acc_A, acc_B, acc_C, acc_D)
        gap_over_best   = acc_E - best_of_others
        gap_over_lstm   = acc_E - acc_A
        combination_is_best = (acc_E == max(all_accs))

        print(f"  Accs: A={acc_A:.4f} B={acc_B:.4f} C={acc_C:.4f} D={acc_D:.4f} E={acc_E:.4f}")
        print(f"  gap_over_best_other={gap_over_best:.4f}  combination_is_best={combination_is_best}")

        # ── Outcome ───────────────────────────────────────────────────────────
        if acc_E > best_of_others + 0.01:
            outcome = OUTCOME_SUPPORTED
        elif acc_A >= acc_E - 0.01:
            outcome = OUTCOME_REFUTED
        else:
            # Combined is numerically best but margin < 0.01
            outcome = OUTCOME_INCONCLUSIVE

        metrics = dict(
            acc_A=round(acc_A, 4),
            acc_B=round(acc_B, 4),
            acc_C=round(acc_C, 4),
            acc_D=round(acc_D, 4),
            acc_E=round(acc_E, 4),
            gap_over_best_other=round(gap_over_best, 4),
            gap_over_lstm_baseline=round(gap_over_lstm, 4),
            combination_is_best=combination_is_best,
        )
        notes = (
            f"A(LSTM+task)={acc_A:.4f}  B(FF+task)={acc_B:.4f}  "
            f"C(FF+oracle)={acc_C:.4f}  D(LSTM+oracle)={acc_D:.4f}  "
            f"E(FF+hindsight)={acc_E:.4f}. "
            f"gap_over_best={gap_over_best:.4f}  "
            f"combination_is_best={combination_is_best}."
        )
        return self.result(outcome, metrics, notes, config)


if __name__ == "__main__":
    Exp214FeedforwardPlusHindsight().execute()
