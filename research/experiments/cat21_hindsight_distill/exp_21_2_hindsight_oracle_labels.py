"""
Experiment 21.2 — Hindsight Oracle Labels

Hypothesis: Hindsight oracle labels (which writes were causally relevant) provide
a stronger training signal than task loss alone (+3% accuracy).
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

VOCAB_SIZE      = 64
HIDDEN_DIM      = 64
SEQ_LEN         = 24
MEMORY_SLOTS    = 8
NUM_PAIRS       = 4
PRETRAIN_STEPS  = 500
TRAIN_STEPS     = 400
BATCH           = 32
LAMBDA          = 0.3
LR              = 3e-4
DEVICE          = "cpu"
ORACLE_FREQ     = 0.20    # compute oracle labels for 20% of batches

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

# ── Shared components ─────────────────────────────────────────────────────────

class Encoder(nn.Module):
    """Position-wise 2-layer MLP encoder."""
    def __init__(self, vocab_size, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.mlp   = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.embed(seq))   # (B, L, H)


class ReadHead(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.scale  = math.sqrt(hidden_dim)

    def forward(self, query: torch.Tensor, memory: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        q   = self.q_proj(query).unsqueeze(1)                       # (B, 1, H)
        k   = self.k_proj(memory)                                    # (B, S, H)
        w   = F.softmax(torch.bmm(q, k.transpose(1, 2)) / self.scale, dim=-1).squeeze(1)
        out = torch.bmm(w.unsqueeze(1), memory).squeeze(1)
        return out, w


class WriteGate(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.proj(h)).squeeze(-1)   # (B, L)


def build_memory(hidden: torch.Tensor, gate_scores: torch.Tensor,
                 memory_slots: int, num_pairs: int) -> torch.Tensor:
    """Soft top-k write into memory_slots using straight-through Gumbel."""
    B, L, H = hidden.shape
    K = min(num_pairs, memory_slots)
    gumbel = -torch.log(-torch.log(torch.clamp(torch.rand_like(gate_scores), 1e-10, 1.0)))
    perturbed  = gate_scores + 0.1 * gumbel
    _, topk_idx = perturbed.topk(K, dim=-1)
    soft_w  = F.softmax(perturbed.gather(1, topk_idx), dim=-1)
    topk_v  = hidden.gather(1, topk_idx.unsqueeze(-1).expand(B, K, H))
    memory  = torch.zeros(B, memory_slots, H, device=hidden.device)
    for k_i in range(K):
        memory[:, k_i, :] = topk_v[:, k_i, :] * soft_w[:, k_i:k_i+1]
    return memory


class MemoryModel(nn.Module):
    """Full model: encoder + write gate + memory + read head + output."""
    def __init__(self, vocab_size, hidden_dim, memory_slots, num_pairs):
        super().__init__()
        self.encoder      = Encoder(vocab_size, hidden_dim)
        self.write_gate   = WriteGate(hidden_dim)
        self.read_head    = ReadHead(hidden_dim)
        self.out          = nn.Linear(hidden_dim, vocab_size)
        self.memory_slots = memory_slots
        self.num_pairs    = num_pairs

    def forward(self, seq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden      = self.encoder(seq)                                  # (B, L, H)
        gate_scores = self.write_gate(hidden)                            # (B, L)
        memory      = build_memory(hidden, gate_scores, self.memory_slots, self.num_pairs)
        query       = hidden[:, -2, :]
        read_vec, _ = self.read_head(query, memory)
        logits      = self.out(read_vec)
        return logits, gate_scores, hidden

# ── Oracle label computation ──────────────────────────────────────────────────

@torch.no_grad()
def compute_oracle_labels(teacher: MemoryModel, seq: torch.Tensor,
                          target: torch.Tensor) -> torch.Tensor:
    """
    For each position t in the sequence, build a singleton memory containing
    only hidden[t] replicated across all slots, then check if the teacher's
    top-1 prediction matches the target.
    oracle_label[b, t] = 1.0 if token t alone is sufficient for correct retrieval.
    """
    teacher.eval()
    B, L = seq.shape
    hidden = teacher.encoder(seq)           # (B, L, H)
    H = hidden.shape[-1]
    query  = hidden[:, -2, :]               # (B, H)

    oracle_labels = torch.zeros(B, L, device=seq.device)

    for t in range(L):
        # Singleton memory: all slots contain hidden[:, t, :]
        singleton_mem = hidden[:, t:t+1, :].expand(B, teacher.memory_slots, H)
        read_vec, _   = teacher.read_head(query, singleton_mem)
        logits        = teacher.out(read_vec)                       # (B, V)
        preds         = logits.argmax(dim=-1)                       # (B,)
        oracle_labels[:, t] = (preds == target).float()

    return oracle_labels  # (B, L)

# ── Training helpers ──────────────────────────────────────────────────────────

def pretrain_teacher(vocab_size, hidden_dim, memory_slots, num_pairs,
                     pretrain_steps: int) -> MemoryModel:
    teacher = MemoryModel(vocab_size, hidden_dim, memory_slots, num_pairs).to(DEVICE)
    opt     = Adam(teacher.parameters(), lr=LR)
    teacher.train()
    print(f"  Pre-training teacher for {pretrain_steps} steps ...")
    for step in range(pretrain_steps):
        seq, target = make_assoc_batch(BATCH, SEQ_LEN, vocab_size, num_pairs)
        seq, target = seq.to(DEVICE), target.to(DEVICE)
        logits, _, _ = teacher(seq)
        loss = F.cross_entropy(logits, target)
        opt.zero_grad(); loss.backward(); opt.step()
        if (step + 1) % 200 == 0:
            print(f"    [teacher] step {step+1}/{pretrain_steps}  loss={loss.item():.4f}")
    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher.eval()
    return teacher


def train_condition_A(steps: int) -> MemoryModel:
    """Task loss only — no oracle signal."""
    model = MemoryModel(VOCAB_SIZE, HIDDEN_DIM, MEMORY_SLOTS, NUM_PAIRS).to(DEVICE)
    opt   = Adam(model.parameters(), lr=LR)
    model.train()
    for step in range(steps):
        seq, target = make_assoc_batch(BATCH, SEQ_LEN, VOCAB_SIZE, NUM_PAIRS)
        seq, target = seq.to(DEVICE), target.to(DEVICE)
        logits, _, _ = model(seq)
        loss = F.cross_entropy(logits, target)
        opt.zero_grad(); loss.backward(); opt.step()
        if (step + 1) % 500 == 0:
            print(f"    [A] step {step+1}/{steps}  loss={loss.item():.4f}")
    return model


def train_condition_B(teacher: MemoryModel, steps: int) -> tuple[MemoryModel, list[float], list[float]]:
    """Task loss + oracle BCE labels (lambda=0.3)."""
    model = MemoryModel(VOCAB_SIZE, HIDDEN_DIM, MEMORY_SLOTS, NUM_PAIRS).to(DEVICE)
    opt   = Adam(model.parameters(), lr=LR)
    model.train()
    gate_score_log: list[float] = []
    oracle_label_log: list[float] = []

    for step in range(steps):
        seq, target = make_assoc_batch(BATCH, SEQ_LEN, VOCAB_SIZE, NUM_PAIRS)
        seq, target = seq.to(DEVICE), target.to(DEVICE)
        logits, gate_scores, _ = model(seq)
        task_loss = F.cross_entropy(logits, target)

        oracle_loss = torch.tensor(0.0)
        use_oracle  = (torch.rand(1).item() < ORACLE_FREQ)
        if use_oracle:
            oracle_labels = compute_oracle_labels(teacher, seq, target)  # (B, L)
            oracle_loss   = F.binary_cross_entropy(
                gate_scores, oracle_labels.detach()
            )
            gate_score_log.append(gate_scores.mean().item())
            oracle_label_log.append(oracle_labels.mean().item())

        loss = task_loss + LAMBDA * oracle_loss
        opt.zero_grad(); loss.backward(); opt.step()
        if (step + 1) % 500 == 0:
            print(f"    [B] step {step+1}/{steps}  task={task_loss.item():.4f}  oracle={oracle_loss.item():.4f}")

    return model, gate_score_log, oracle_label_log

# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(model: MemoryModel, n_batches: int = 50) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for _ in range(n_batches):
            seq, target = make_assoc_batch(BATCH, SEQ_LEN, VOCAB_SIZE, NUM_PAIRS)
            seq, target = seq.to(DEVICE), target.to(DEVICE)
            logits, _, _ = model(seq)
            correct += (logits.argmax(-1) == target).sum().item()
            total   += BATCH
    return correct / total


def pearson_r(x: list[float], y: list[float]) -> float:
    """Pearson correlation between two lists."""
    if len(x) < 2:
        return 0.0
    tx = torch.tensor(x)
    ty = torch.tensor(y)
    vx = tx - tx.mean()
    vy = ty - ty.mean()
    denom = (vx.norm() * vy.norm()).clamp(min=1e-8)
    return (vx * vy).sum().item() / denom.item()

# ── Experiment ────────────────────────────────────────────────────────────────

class Exp212HindsightOracleLabels(Experiment):
    experiment_id = "exp_21_2"
    hypothesis = (
        "Hindsight oracle labels (which writes were causally relevant) provide "
        "a stronger training signal than task loss alone (+3% accuracy)."
    )

    def run(self) -> ExperimentResult:
        config = dict(
            vocab_size=VOCAB_SIZE, hidden_dim=HIDDEN_DIM, seq_len=SEQ_LEN,
            memory_slots=MEMORY_SLOTS, num_pairs=NUM_PAIRS,
            pretrain_steps=PRETRAIN_STEPS, train_steps=TRAIN_STEPS,
            batch=BATCH, lambda_=LAMBDA, oracle_freq=ORACLE_FREQ,
        )

        # Pre-train and freeze teacher
        teacher = pretrain_teacher(VOCAB_SIZE, HIDDEN_DIM, MEMORY_SLOTS, NUM_PAIRS, PRETRAIN_STEPS)

        # ── Condition A: task loss only ───────────────────────────────────────
        print("  Training Condition A (task loss only) ...")
        model_A = train_condition_A(TRAIN_STEPS)
        acc_A   = evaluate(model_A)
        print(f"    acc_A={acc_A:.4f}")

        # ── Condition B: task loss + oracle BCE ───────────────────────────────
        print("  Training Condition B (task + oracle BCE) ...")
        model_B, gate_log, oracle_log = train_condition_B(teacher, TRAIN_STEPS)
        acc_B   = evaluate(model_B)
        gate_quality = pearson_r(gate_log, oracle_log)
        print(f"    acc_B={acc_B:.4f}  gate_quality_r={gate_quality:.4f}")

        acc_gap = acc_B - acc_A
        print(f"  acc_gap (B - A) = {acc_gap:.4f}")

        # ── Outcome ───────────────────────────────────────────────────────────
        if acc_B > acc_A + 0.03:
            outcome = OUTCOME_SUPPORTED
        elif acc_A >= acc_B - 0.01:
            outcome = OUTCOME_REFUTED
        else:
            outcome = OUTCOME_INCONCLUSIVE

        metrics = dict(
            acc_A=round(acc_A, 4),
            acc_B=round(acc_B, 4),
            acc_gap=round(acc_gap, 4),
            gate_quality_r=round(gate_quality, 4),
        )
        notes = (
            f"Task-only: acc={acc_A:.4f}. "
            f"Oracle-augmented: acc={acc_B:.4f}. "
            f"gap={acc_gap:.4f}, gate_quality_r={gate_quality:.4f}."
        )
        return self.result(outcome, metrics, notes, config)


if __name__ == "__main__":
    Exp212HindsightOracleLabels().execute()
