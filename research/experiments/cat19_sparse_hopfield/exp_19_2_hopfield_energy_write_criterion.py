"""
exp_19_2_hopfield_energy_write_criterion.py

Hypothesis: Hopfield energy write criterion (write only if ΔE < 0) produces
<35% write rate with >3% accuracy improvement.
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
MEMORY_SLOTS = 8
SEQ_LEN = 24
NUM_PAIRS = 5
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
# Hopfield energy helper
# ---------------------------------------------------------------------------
def hopfield_energy(memory):
    """E(M) = -0.5 * ||M||_F^2.  memory: (B, M, H) -> (B,)"""
    return -0.5 * (memory ** 2).sum(dim=(1, 2))


# ---------------------------------------------------------------------------
# Memory modules
# ---------------------------------------------------------------------------
class LearnedGateMemory(nn.Module):
    """Condition A: learned write gate, top-k=5 selection."""

    def __init__(self, hidden_dim, memory_slots, k_write=5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.memory_slots = memory_slots
        self.k_write = k_write
        self.write_gate = nn.Linear(hidden_dim, 1)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_head = nn.Linear(hidden_dim, VOCAB_SIZE)

    def forward(self, enc_hidden, query_hidden):
        B, T, H = enc_hidden.shape
        gate_scores = self.write_gate(enc_hidden).squeeze(-1)   # (B, T)
        gate_probs = torch.sigmoid(gate_scores)                 # (B, T)

        k = min(self.k_write, self.memory_slots, T)
        _, top_idx = gate_probs.topk(k, dim=-1)                # (B, k)

        memory = torch.zeros(B, self.memory_slots, H, device=enc_hidden.device)
        for slot in range(k):
            tok_idx = top_idx[:, slot]
            memory[:, slot] = enc_hidden[torch.arange(B), tok_idx]

        return self._read(memory, query_hidden), gate_probs, memory

    def _read(self, memory, query_hidden):
        B, M, H = memory.shape
        q = self.q_proj(query_hidden).unsqueeze(1)              # (B, 1, H)
        k = self.k_proj(memory)                                 # (B, M, H)
        scores = (q @ k.transpose(1, 2)).squeeze(1) / (H ** 0.5)  # (B, M)
        attn = torch.softmax(scores, dim=-1)                    # (B, M)
        retrieved = (attn.unsqueeze(1) @ memory).squeeze(1)    # (B, H)
        return self.out_head(retrieved + query_hidden)


class EnergyGateMemory(nn.Module):
    """Condition B / C: Hopfield energy write criterion."""

    def __init__(self, hidden_dim, memory_slots, use_learned_gate=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.memory_slots = memory_slots
        self.use_learned_gate = use_learned_gate
        if use_learned_gate:
            self.write_gate = nn.Linear(hidden_dim, 1)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_head = nn.Linear(hidden_dim, VOCAB_SIZE)

    def forward(self, enc_hidden, query_hidden):
        B, T, H = enc_hidden.shape

        # Build memory using energy criterion (per-item greedy construction)
        memory = torch.zeros(B, self.memory_slots, H, device=enc_hidden.device)
        slot_ptr = torch.zeros(B, dtype=torch.long)   # circular fill pointer
        total_writes = 0
        total_tokens = 0

        if self.use_learned_gate:
            gate_scores = self.write_gate(enc_hidden).squeeze(-1)   # (B, T)
            gate_probs = torch.sigmoid(gate_scores)                  # (B, T)
        else:
            gate_probs = torch.ones(B, T, device=enc_hidden.device) * 0.5  # neutral

        for t in range(T - 3):  # skip last 3 (sep / query / placeholder)
            token_h = enc_hidden[:, t]    # (B, H)
            total_tokens += B

            for b in range(B):
                tok = token_h[b]                                     # (H,)
                cur_memory = memory[b]                               # (M, H)

                # Find best slot: lowest cosine similarity to current token
                sims = F.cosine_similarity(
                    cur_memory, tok.unsqueeze(0).expand(self.memory_slots, -1), dim=-1
                )
                best_slot = sims.argmin().item()

                # Compute ΔE
                E_old = -0.5 * (cur_memory ** 2).sum().item()
                new_memory = cur_memory.clone()
                new_memory[best_slot] = tok
                E_new = -0.5 * (new_memory ** 2).sum().item()
                delta_E = E_new - E_old

                # Write decision
                energy_ok = delta_E < 0
                if self.use_learned_gate:
                    learned_ok = gate_probs[b, t].item() > 0.5
                    do_write = energy_ok and learned_ok
                else:
                    do_write = energy_ok

                if do_write:
                    memory[b, best_slot] = tok
                    total_writes += 1

        write_rate = total_writes / max(total_tokens, 1)
        logits = self._read(memory, query_hidden)
        return logits, gate_probs, write_rate

    def _read(self, memory, query_hidden):
        B, M, H = memory.shape
        q = self.q_proj(query_hidden).unsqueeze(1)
        k = self.k_proj(memory)
        scores = (q @ k.transpose(1, 2)).squeeze(1) / (H ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        retrieved = (attn.unsqueeze(1) @ memory).squeeze(1)
        return self.out_head(retrieved + query_hidden)


# ---------------------------------------------------------------------------
# Training loops
# ---------------------------------------------------------------------------
def train_condition_A():
    """Condition A: learned write gate only."""
    torch.manual_seed(42)
    encoder = Encoder(VOCAB_SIZE, HIDDEN_DIM).to(DEVICE)
    mem = LearnedGateMemory(HIDDEN_DIM, MEMORY_SLOTS, k_write=5).to(DEVICE)
    opt = Adam(list(encoder.parameters()) + list(mem.parameters()), lr=LR)

    correct = total = 0
    for step in range(STEPS):
        seq, target = make_assoc_batch(BATCH, SEQ_LEN, VOCAB_SIZE, NUM_PAIRS)
        seq, target = seq.to(DEVICE), target.to(DEVICE)
        enc_h = encoder(seq)
        query_h = enc_h[:, SEQ_LEN - 2]
        logits, _, _ = mem(enc_h, query_h)
        loss = F.cross_entropy(logits, target)
        opt.zero_grad(); loss.backward(); opt.step()
        if step >= STEPS * 3 // 4:
            correct += (logits.argmax(-1) == target).sum().item()
            total += BATCH
    return correct / max(total, 1)


def train_condition_B(use_learned_gate=False):
    """Condition B or C: energy criterion (B=no gate, C=gate AND energy)."""
    torch.manual_seed(42)
    encoder = Encoder(VOCAB_SIZE, HIDDEN_DIM).to(DEVICE)
    mem = EnergyGateMemory(HIDDEN_DIM, MEMORY_SLOTS, use_learned_gate=use_learned_gate).to(DEVICE)
    opt = Adam(list(encoder.parameters()) + list(mem.parameters()), lr=LR)

    correct = total = 0
    write_rates = []
    for step in range(STEPS):
        seq, target = make_assoc_batch(BATCH, SEQ_LEN, VOCAB_SIZE, NUM_PAIRS)
        seq, target = seq.to(DEVICE), target.to(DEVICE)
        enc_h = encoder(seq)
        query_h = enc_h[:, SEQ_LEN - 2]
        logits, gate_probs, write_rate = mem(enc_h, query_h)
        loss = F.cross_entropy(logits, target)
        opt.zero_grad(); loss.backward(); opt.step()
        if step >= STEPS * 3 // 4:
            correct += (logits.argmax(-1) == target).sum().item()
            total += BATCH
            write_rates.append(write_rate)
    acc = correct / max(total, 1)
    avg_write_rate = sum(write_rates) / max(len(write_rates), 1)
    return acc, avg_write_rate


# ---------------------------------------------------------------------------
# Experiment class
# ---------------------------------------------------------------------------
class Exp192HopfieldEnergyWriteCriterion(Experiment):
    experiment_id = "exp_19_2"
    hypothesis = (
        "Hopfield energy write criterion (write only if ΔE < 0) produces "
        "<35% write rate with >3% accuracy improvement."
    )

    def run(self) -> ExperimentResult:
        config = dict(
            VOCAB_SIZE=VOCAB_SIZE, HIDDEN_DIM=HIDDEN_DIM, MEMORY_SLOTS=MEMORY_SLOTS,
            SEQ_LEN=SEQ_LEN, NUM_PAIRS=NUM_PAIRS, STEPS=STEPS, BATCH=BATCH, LR=LR,
        )

        print("Training condition A: learned write gate...")
        acc_A = train_condition_A()
        print(f"  acc_A={acc_A:.4f}")

        print("Training condition B: Hopfield energy write criterion...")
        acc_B, write_rate_B = train_condition_B(use_learned_gate=False)
        print(f"  acc_B={acc_B:.4f}  write_rate_B={write_rate_B:.4f}")

        print("Training condition C: energy AND learned gate...")
        acc_C, write_rate_C = train_condition_B(use_learned_gate=True)
        print(f"  acc_C={acc_C:.4f}  write_rate_C={write_rate_C:.4f}")

        metrics = dict(
            acc_A=round(acc_A, 4),
            acc_B=round(acc_B, 4),
            acc_C=round(acc_C, 4),
            write_rate_B=round(write_rate_B, 4),
            write_rate_C=round(write_rate_C, 4),
            acc_gap_B_minus_A=round(acc_B - acc_A, 4),
        )

        if acc_B > acc_A + 0.03 and write_rate_B < 0.35:
            outcome = OUTCOME_SUPPORTED
            notes = "Energy criterion achieves sparse writes with accuracy gain."
        elif write_rate_B > 0.50 or acc_B < acc_A:
            outcome = OUTCOME_REFUTED
            notes = (
                f"Energy writes too frequent ({write_rate_B:.4f}) "
                f"or acc dropped vs learned gate."
            )
        else:
            outcome = OUTCOME_INCONCLUSIVE
            notes = (
                f"Write rate {write_rate_B:.4f} is sparse but accuracy gain "
                f"({acc_B - acc_A:.4f}) < 0.03 threshold."
            )

        return self.result(outcome, metrics=metrics, notes=notes, config=config)


if __name__ == "__main__":
    Exp192HopfieldEnergyWriteCriterion().execute()
