"""
Experiment 21.3 — Distilled vs Learned Write Gate

Hypothesis: Write gate distilled from oracle labels (trained primarily on oracle
supervision) achieves higher accuracy than end-to-end learned gate.
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
    def __init__(self, vocab_size, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.mlp   = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.embed(seq))


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
    perturbed  = gate_scores + 0.1 * gumbel
    _, topk_idx = perturbed.topk(K, dim=-1)
    soft_w  = F.softmax(perturbed.gather(1, topk_idx), dim=-1)
    topk_v  = hidden.gather(1, topk_idx.unsqueeze(-1).expand(B, K, H))
    memory  = torch.zeros(B, memory_slots, H, device=hidden.device)
    for k_i in range(K):
        memory[:, k_i, :] = topk_v[:, k_i, :] * soft_w[:, k_i:k_i+1]
    return memory


class MemoryModel(nn.Module):
    def __init__(self, vocab_size, hidden_dim, memory_slots, num_pairs):
        super().__init__()
        self.encoder      = Encoder(vocab_size, hidden_dim)
        self.write_gate   = WriteGate(hidden_dim)
        self.read_head    = ReadHead(hidden_dim)
        self.out          = nn.Linear(hidden_dim, vocab_size)
        self.memory_slots = memory_slots
        self.num_pairs    = num_pairs

    def forward(self, seq):
        hidden      = self.encoder(seq)
        gate_scores = self.write_gate(hidden)
        memory      = build_memory(hidden, gate_scores, self.memory_slots, self.num_pairs)
        query       = hidden[:, -2, :]
        read_vec, _ = self.read_head(query, memory)
        logits      = self.out(read_vec)
        return logits, gate_scores, hidden

# ── Oracle label computation ──────────────────────────────────────────────────

@torch.no_grad()
def compute_oracle_labels(teacher: MemoryModel, seq: torch.Tensor,
                          target: torch.Tensor) -> torch.Tensor:
    teacher.eval()
    B, L   = seq.shape
    hidden = teacher.encoder(seq)
    H      = hidden.shape[-1]
    query  = hidden[:, -2, :]
    oracle_labels = torch.zeros(B, L, device=seq.device)
    for t in range(L):
        singleton = hidden[:, t:t+1, :].expand(B, teacher.memory_slots, H)
        read_vec, _ = teacher.read_head(query, singleton)
        preds = teacher.out(read_vec).argmax(dim=-1)
        oracle_labels[:, t] = (preds == target).float()
    return oracle_labels


def pretrain_teacher(vocab_size, hidden_dim, memory_slots, num_pairs,
                     pretrain_steps: int) -> MemoryModel:
    teacher = MemoryModel(vocab_size, hidden_dim, memory_slots, num_pairs).to(DEVICE)
    opt     = Adam(teacher.parameters(), lr=LR)
    teacher.train()
    print(f"  Pre-training teacher ({pretrain_steps} steps) ...")
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

# ── Three training conditions ─────────────────────────────────────────────────

def train_A_end_to_end(steps: int) -> MemoryModel:
    """Condition A: task loss only, standard end-to-end."""
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


def train_B_oracle_distilled(teacher: MemoryModel, steps: int) -> MemoryModel:
    """
    Condition B: Oracle-distilled gate.
    - Gate trained only on BCE(gate_scores, oracle_labels) — no task gradient to gate.
    - Encoder + read_head trained on task loss with gate frozen during task updates.
    - Schedule: every 5 steps, 1 oracle update for gate, 4 task updates for encoder+read_head.
    """
    model = MemoryModel(VOCAB_SIZE, HIDDEN_DIM, MEMORY_SLOTS, NUM_PAIRS).to(DEVICE)
    # Separate optimisers for gate vs rest
    opt_gate = Adam(model.write_gate.parameters(), lr=LR)
    task_params = (
        list(model.encoder.parameters()) +
        list(model.read_head.parameters()) +
        list(model.out.parameters())
    )
    opt_task = Adam(task_params, lr=LR)
    model.train()

    for step in range(steps):
        seq, target = make_assoc_batch(BATCH, SEQ_LEN, VOCAB_SIZE, NUM_PAIRS)
        seq, target = seq.to(DEVICE), target.to(DEVICE)

        cycle = step % 5
        if cycle == 0:
            # ── Oracle gate update ────────────────────────────────────────────
            # Freeze encoder for this update so gate trains from oracle signal only
            hidden = model.encoder(seq).detach()          # detach: no grad to encoder
            gate_scores = model.write_gate(hidden)
            oracle_labels = compute_oracle_labels(teacher, seq, target)
            gate_loss = F.binary_cross_entropy(gate_scores, oracle_labels.detach())
            opt_gate.zero_grad(); gate_loss.backward(); opt_gate.step()
            if (step + 1) % 500 == 0:
                print(f"    [B] step {step+1}/{steps}  gate_loss={gate_loss.item():.4f}")
        else:
            # ── Task update for encoder + read_head (gate frozen) ─────────────
            logits, _, _ = model(seq)
            task_loss = F.cross_entropy(logits, target)
            opt_task.zero_grad(); task_loss.backward(); opt_task.step()
            if (step + 1) % 500 == 0 and cycle == 4:
                print(f"    [B] step {step+1}/{steps}  task_loss={task_loss.item():.4f}")

    return model


def train_C_mixed(teacher: MemoryModel, steps: int) -> MemoryModel:
    """
    Condition C: Mixed — gate trained with 50% oracle (BCE) + 50% task gradient.
    Both signals flow simultaneously.
    """
    model = MemoryModel(VOCAB_SIZE, HIDDEN_DIM, MEMORY_SLOTS, NUM_PAIRS).to(DEVICE)
    opt   = Adam(model.parameters(), lr=LR)
    model.train()

    for step in range(steps):
        seq, target = make_assoc_batch(BATCH, SEQ_LEN, VOCAB_SIZE, NUM_PAIRS)
        seq, target = seq.to(DEVICE), target.to(DEVICE)
        logits, gate_scores, _ = model(seq)
        task_loss = F.cross_entropy(logits, target)

        oracle_labels = compute_oracle_labels(teacher, seq, target)
        oracle_loss   = F.binary_cross_entropy(gate_scores, oracle_labels.detach())

        # Equal weighting of both losses
        loss = 0.5 * task_loss + 0.5 * oracle_loss
        opt.zero_grad(); loss.backward(); opt.step()
        if (step + 1) % 500 == 0:
            print(f"    [C] step {step+1}/{steps}  task={task_loss.item():.4f}  oracle={oracle_loss.item():.4f}")

    return model

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


def eval_gate_quality(model: MemoryModel, teacher: MemoryModel,
                      n_batches: int = 30) -> float:
    """Pearson r between gate_scores and oracle_labels across eval batches."""
    model.eval()
    all_gates:  list[float] = []
    all_labels: list[float] = []
    with torch.no_grad():
        for _ in range(n_batches):
            seq, target = make_assoc_batch(BATCH, SEQ_LEN, VOCAB_SIZE, NUM_PAIRS)
            seq, target = seq.to(DEVICE), target.to(DEVICE)
            _, gate_scores, _ = model(seq)
            oracle_labels = compute_oracle_labels(teacher, seq, target)
            all_gates.extend(gate_scores.flatten().tolist())
            all_labels.extend(oracle_labels.flatten().tolist())

    tg = torch.tensor(all_gates)
    tl = torch.tensor(all_labels)
    vg = tg - tg.mean()
    vl = tl - tl.mean()
    denom = (vg.norm() * vl.norm()).clamp(min=1e-8)
    return (vg * vl).sum().item() / denom.item()

# ── Experiment ────────────────────────────────────────────────────────────────

class Exp213DistilledVsLearnedGate(Experiment):
    experiment_id = "exp_21_3"
    hypothesis = (
        "Write gate distilled from oracle labels (trained primarily on oracle "
        "supervision) achieves higher accuracy than end-to-end learned gate."
    )

    def run(self) -> ExperimentResult:
        config = dict(
            vocab_size=VOCAB_SIZE, hidden_dim=HIDDEN_DIM, seq_len=SEQ_LEN,
            memory_slots=MEMORY_SLOTS, num_pairs=NUM_PAIRS,
            pretrain_steps=PRETRAIN_STEPS, train_steps=TRAIN_STEPS, batch=BATCH,
        )

        teacher = pretrain_teacher(VOCAB_SIZE, HIDDEN_DIM, MEMORY_SLOTS, NUM_PAIRS, PRETRAIN_STEPS)

        # ── Condition A ───────────────────────────────────────────────────────
        print("  Training Condition A (end-to-end, task loss only) ...")
        model_A = train_A_end_to_end(TRAIN_STEPS)
        acc_A          = evaluate(model_A)
        gate_quality_A = eval_gate_quality(model_A, teacher)
        print(f"    acc_A={acc_A:.4f}  gate_quality_A={gate_quality_A:.4f}")

        # ── Condition B ───────────────────────────────────────────────────────
        print("  Training Condition B (oracle-distilled gate) ...")
        model_B = train_B_oracle_distilled(teacher, TRAIN_STEPS)
        acc_B          = evaluate(model_B)
        gate_quality_B = eval_gate_quality(model_B, teacher)
        print(f"    acc_B={acc_B:.4f}  gate_quality_B={gate_quality_B:.4f}")

        # ── Condition C ───────────────────────────────────────────────────────
        print("  Training Condition C (mixed 50/50) ...")
        model_C = train_C_mixed(teacher, TRAIN_STEPS)
        acc_C          = evaluate(model_C)
        gate_quality_C = eval_gate_quality(model_C, teacher)
        print(f"    acc_C={acc_C:.4f}  gate_quality_C={gate_quality_C:.4f}")

        quality_gap = gate_quality_B - gate_quality_A
        acc_gap_BA  = acc_B - acc_A
        print(f"  acc_gap(B-A)={acc_gap_BA:.4f}  quality_gap(B-A)={quality_gap:.4f}")

        # ── Outcome ───────────────────────────────────────────────────────────
        if acc_B > acc_A + 0.02 and gate_quality_B > gate_quality_A + 0.10:
            outcome = OUTCOME_SUPPORTED
        elif acc_A >= acc_B - 0.01:
            outcome = OUTCOME_REFUTED
        else:
            outcome = OUTCOME_INCONCLUSIVE

        metrics = dict(
            acc_A=round(acc_A, 4),
            acc_B=round(acc_B, 4),
            acc_C=round(acc_C, 4),
            gate_quality_A=round(gate_quality_A, 4),
            gate_quality_B=round(gate_quality_B, 4),
            gate_quality_C=round(gate_quality_C, 4),
            acc_gap_BA=round(acc_gap_BA, 4),
            quality_gap_BA=round(quality_gap, 4),
        )
        notes = (
            f"A(e2e): acc={acc_A:.4f} gq={gate_quality_A:.4f}. "
            f"B(distilled): acc={acc_B:.4f} gq={gate_quality_B:.4f}. "
            f"C(mixed): acc={acc_C:.4f} gq={gate_quality_C:.4f}. "
            f"acc_gap(B-A)={acc_gap_BA:.4f} quality_gap={quality_gap:.4f}."
        )
        return self.result(outcome, metrics, notes, config)


if __name__ == "__main__":
    Exp213DistilledVsLearnedGate().execute()
