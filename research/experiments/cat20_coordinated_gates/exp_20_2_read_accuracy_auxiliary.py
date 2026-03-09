"""
exp_20_2_read_accuracy_auxiliary.py

Hypothesis: Explicit read accuracy auxiliary loss reduces the read bottleneck
more effectively than implicit task loss.
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

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
VOCAB_SIZE = 64
HIDDEN_DIM = 64
SEQ_LEN = 24
MEMORY_SLOTS = 6
NUM_PAIRS = 4
STEPS = 400
BATCH = 32
LR = 3e-4
DEVICE = "cpu"


# ---------------------------------------------------------------------------
# Data generator
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------
class Encoder(nn.Module):
    def __init__(self, vocab_size, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        h = self.embed(x)
        return self.norm(h + self.ff(h))


# ---------------------------------------------------------------------------
# Read head (shared, separable from write gate)
# ---------------------------------------------------------------------------
class ReadHead(nn.Module):
    """Attends over a memory matrix given a query vector."""

    def __init__(self, hidden_dim, vocab_size):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, query_hidden, memory):
        """
        query_hidden: (B, H)
        memory:       (B, M, H)
        Returns: logits (B, vocab_size)
        """
        B, M, H = memory.shape
        q = self.q_proj(query_hidden).unsqueeze(1)               # (B, 1, H)
        k = self.k_proj(memory)                                  # (B, M, H)
        scores = (q @ k.transpose(1, 2)).squeeze(1) / (H ** 0.5)  # (B, M)
        attn = torch.softmax(scores, dim=-1)
        retrieved = (attn.unsqueeze(1) @ memory).squeeze(1)     # (B, H)
        return self.out_head(retrieved + query_hidden)


# ---------------------------------------------------------------------------
# Write gate memory
# ---------------------------------------------------------------------------
class WriteGateMemory(nn.Module):
    """Learned write gate producing a memory matrix."""

    def __init__(self, hidden_dim, memory_slots, k_write=4):
        super().__init__()
        self.memory_slots = memory_slots
        self.k_write = k_write
        self.write_gate = nn.Linear(hidden_dim, 1)

    def forward(self, enc_hidden):
        """Returns memory (B, M, H) and gate_scores (B, T)."""
        B, T, H = enc_hidden.shape
        gate_logits = self.write_gate(enc_hidden).squeeze(-1)    # (B, T)
        gate_scores = torch.sigmoid(gate_logits)

        k = min(self.k_write, self.memory_slots, T)
        _, top_idx = gate_scores.topk(k, dim=-1)                # (B, k)

        memory = torch.zeros(B, self.memory_slots, H, device=enc_hidden.device)
        for slot in range(k):
            tok_idx = top_idx[:, slot]
            memory[:, slot] = enc_hidden[torch.arange(B), tok_idx]

        return memory, gate_scores


# ---------------------------------------------------------------------------
# Oracle memory builder
# ---------------------------------------------------------------------------
def build_oracle_memory(enc_hidden, num_pairs, memory_slots):
    """
    Fill memory directly with KV pair hidden states — no gate, forced write.
    This gives the read head the correct content to supervise it in isolation.
    """
    B, T, H = enc_hidden.shape
    memory = torch.zeros(B, memory_slots, H, device=enc_hidden.device)
    # KV positions: 0,1 for pair 0; 2,3 for pair 1; etc.
    n_write = min(num_pairs * 2, memory_slots, T - 3)
    for i in range(n_write):
        slot = i % memory_slots
        memory[:, slot] = enc_hidden[:, i]
    return memory


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train_condition(lam_aux: float):
    torch.manual_seed(42)
    encoder = Encoder(VOCAB_SIZE, HIDDEN_DIM).to(DEVICE)
    write_mem = WriteGateMemory(HIDDEN_DIM, MEMORY_SLOTS, k_write=4).to(DEVICE)
    read_head = ReadHead(HIDDEN_DIM, VOCAB_SIZE).to(DEVICE)
    opt = Adam(
        list(encoder.parameters()) +
        list(write_mem.parameters()) +
        list(read_head.parameters()),
        lr=LR,
    )

    correct = total = 0
    oracle_correct = oracle_total = 0

    for step in range(STEPS):
        seq, target = make_assoc_batch(BATCH, SEQ_LEN, VOCAB_SIZE, NUM_PAIRS)
        seq, target = seq.to(DEVICE), target.to(DEVICE)

        enc_h = encoder(seq)                                     # (B, T, H)
        query_h = enc_h[:, SEQ_LEN - 2]                         # (B, H)

        # Normal forward path
        memory, gate_scores = write_mem(enc_h)
        logits = read_head(query_h, memory)
        task_loss = F.cross_entropy(logits, target)

        if lam_aux > 0.0:
            # Oracle memory: detached from encoder + write gate
            oracle_memory = build_oracle_memory(enc_h.detach(), NUM_PAIRS, MEMORY_SLOTS)
            # Auxiliary supervises the read head with correct memory content.
            # .detach() on oracle_memory prevents gradient flowing to write gate/encoder.
            aux_logits = read_head(query_h.detach(), oracle_memory.detach())
            aux_loss = F.cross_entropy(aux_logits, target)
            total_loss = task_loss + lam_aux * aux_loss
        else:
            total_loss = task_loss
            oracle_memory = None
            aux_loss = torch.tensor(0.0)

        opt.zero_grad()
        total_loss.backward()
        opt.step()

        if step >= STEPS * 3 // 4:
            correct += (logits.argmax(-1) == target).sum().item()
            total += BATCH
            # Oracle read accuracy (eval only, no grad)
            with torch.no_grad():
                om = build_oracle_memory(enc_h, NUM_PAIRS, MEMORY_SLOTS)
                oracle_logits = read_head(query_h, om)
                oracle_correct += (oracle_logits.argmax(-1) == target).sum().item()
                oracle_total += BATCH

    acc = correct / max(total, 1)
    oracle_read_acc = oracle_correct / max(oracle_total, 1)
    return acc, oracle_read_acc


# ---------------------------------------------------------------------------
# Experiment class
# ---------------------------------------------------------------------------
class Exp202ReadAccuracyAuxiliary(Experiment):
    experiment_id = "exp_20_2"
    hypothesis = (
        "Explicit read accuracy auxiliary loss reduces the read bottleneck "
        "more effectively than implicit task loss."
    )

    def run(self) -> ExperimentResult:
        config = dict(
            VOCAB_SIZE=VOCAB_SIZE, HIDDEN_DIM=HIDDEN_DIM, SEQ_LEN=SEQ_LEN,
            MEMORY_SLOTS=MEMORY_SLOTS, NUM_PAIRS=NUM_PAIRS, STEPS=STEPS,
            BATCH=BATCH, LR=LR,
        )

        print("Training condition A: task loss only (lambda=0)...")
        acc_A, oracle_A = train_condition(0.0)
        print(f"  acc_A={acc_A:.4f}  oracle_read_acc_A={oracle_A:.4f}")

        print("Training condition B: task + read auxiliary (lambda=0.1)...")
        acc_B, oracle_B = train_condition(0.1)
        print(f"  acc_B={acc_B:.4f}  oracle_read_acc_B={oracle_B:.4f}")

        print("Training condition C: task + stronger auxiliary (lambda=0.5)...")
        acc_C, oracle_C = train_condition(0.5)
        print(f"  acc_C={acc_C:.4f}  oracle_read_acc_C={oracle_C:.4f}")

        metrics = dict(
            acc_A=round(acc_A, 4),
            acc_B=round(acc_B, 4),
            acc_C=round(acc_C, 4),
            oracle_read_acc_A=round(oracle_A, 4),
            oracle_read_acc_B=round(oracle_B, 4),
            oracle_read_acc_C=round(oracle_C, 4),
            acc_gap_B_minus_A=round(acc_B - acc_A, 4),
            oracle_gap_B_minus_A=round(oracle_B - oracle_A, 4),
        )

        if acc_B > acc_A + 0.02 and oracle_B > oracle_A + 0.05:
            outcome = OUTCOME_SUPPORTED
            notes = (
                f"Read auxiliary (lambda=0.1) improves task acc by {acc_B - acc_A:.4f} "
                f"and oracle read acc by {oracle_B - oracle_A:.4f}."
            )
        elif acc_B <= acc_A + 0.005:
            outcome = OUTCOME_REFUTED
            notes = f"Task acc unchanged (gap={acc_B - acc_A:.4f} <= 0.005)."
        else:
            outcome = OUTCOME_INCONCLUSIVE
            notes = (
                f"acc gap={acc_B - acc_A:.4f}, oracle gap={oracle_B - oracle_A:.4f} — "
                "one threshold met but not both."
            )

        return self.result(outcome, metrics=metrics, notes=notes, config=config)


if __name__ == "__main__":
    Exp202ReadAccuracyAuxiliary().execute()
