from __future__ import annotations
import sys
import math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from experiments.base import (Experiment, ExperimentResult, OUTCOME_SUPPORTED,
                               OUTCOME_REFUTED, OUTCOME_INCONCLUSIVE)

experiment_id = "exp_14_3"
hypothesis = (
    "Cosine-annealed Gumbel temperature (1.0->0.1) produces higher final accuracy "
    "than constant temperature 0.5 by >2%."
)

DEVICE = "cpu"
LR = 3e-4
VOCAB_SIZE = 64
SEQ_LEN = 24
HIDDEN_DIM = 64
MEMORY_SLOTS = 6
NUM_PAIRS = 4
K_WRITE = 3
STEPS = 500
BATCH = 32


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
        seq[b, seq_len - 3] = 2; seq[b, seq_len - 2] = keys[qi]; seq[b, seq_len - 1] = 0
        target[b] = vals[qi]
    return seq, target


class Encoder(nn.Module):
    def __init__(self, vocab_size, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.ff = nn.Sequential(nn.Linear(hidden_dim, hidden_dim * 2), nn.ReLU(),
                                nn.Linear(hidden_dim * 2, hidden_dim))
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        h = self.embed(x)
        return self.norm(h + self.ff(h))


def gumbel_topk(logits, k, temperature):
    """Differentiable top-k selection via Gumbel-softmax with straight-through estimator."""
    g = -torch.log(-torch.log(torch.rand_like(logits) + 1e-8) + 1e-8)
    perturbed = (logits + g) / temperature
    _, indices = perturbed.topk(k, dim=-1)
    mask = torch.zeros_like(logits).scatter_(-1, indices, 1.0)
    soft = torch.softmax(perturbed, dim=-1)
    return mask + (soft - soft.detach())  # straight-through


class GumbelMemoryModel(nn.Module):
    def __init__(self, vocab_size, hidden_dim, num_slots):
        super().__init__()
        self.encoder = Encoder(vocab_size, hidden_dim)
        self.write_gate = nn.Linear(hidden_dim, 1)  # scalar score per token
        self.read_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, vocab_size)
        self.hidden_dim = hidden_dim
        self.num_slots = num_slots

    def forward(self, seq, temperature):
        hiddens = self.encoder(seq)  # (B, T, D)
        context = hiddens[:, :-1, :]  # (B, T-1, D)
        B, T, D = context.shape

        # Gate scores: (B, T)
        gate_scores = self.write_gate(context).squeeze(-1)

        # Gumbel top-k selection: (B, T) selection weights
        sel_weights = gumbel_topk(gate_scores, min(K_WRITE, T), temperature)  # (B, T)

        # Write to memory: weighted sum of hiddens -> slots
        # Distribute top-k tokens across memory slots
        mem = torch.zeros(B, self.num_slots, D, device=seq.device)
        # Use the top-k hard indices for slot assignment
        _, top_idx = gate_scores.topk(min(K_WRITE, T), dim=-1)  # (B, K)
        # Vectorized write: gather selected hidden states weighted by sel_weights
        k_w = top_idx.shape[1]
        gathered_w = sel_weights.gather(1, top_idx)                           # (B, K)
        top_idx_exp = top_idx.unsqueeze(-1).expand(-1, -1, D)                 # (B, K, D)
        gathered_h = context.gather(1, top_idx_exp)                           # (B, K, D)
        weighted = gathered_h * gathered_w.unsqueeze(-1)                      # (B, K, D)
        slot_indices = (torch.arange(k_w, device=seq.device)
                        .unsqueeze(0).expand(B, -1) % self.num_slots)         # (B, K)
        slot_indices_exp = slot_indices.unsqueeze(-1).expand(-1, -1, D)       # (B, K, D)
        mem.scatter_add_(1, slot_indices_exp, weighted)

        # Read from memory
        query_h = hiddens[:, -1, :]  # (B, D)
        q = self.read_proj(query_h)
        scores = torch.bmm(mem, q.unsqueeze(-1)).squeeze(-1)  # (B, S)
        attn = torch.softmax(scores, dim=-1)
        ctx = (attn.unsqueeze(-1) * mem).sum(1)  # (B, D)
        return self.out(ctx)


def get_temperature(step, schedule, total_steps):
    if schedule == "A":
        return 0.5
    elif schedule == "B":
        if step < 500:
            return 1.0 - (1.0 - 0.5) * (step / 500)
        return 0.5
    elif schedule == "C":
        return 0.1 + 0.45 * (1 + math.cos(math.pi * step / total_steps))
    return 0.5


def train_condition(schedule: str, seed_offset: int) -> float:
    torch.manual_seed(42 + seed_offset)
    model = GumbelMemoryModel(VOCAB_SIZE, HIDDEN_DIM, MEMORY_SLOTS).to(DEVICE)
    opt = Adam(model.parameters(), lr=LR)

    for step in range(STEPS):
        temp = get_temperature(step, schedule, STEPS)
        seq, target = make_assoc_batch(BATCH, SEQ_LEN, VOCAB_SIZE, NUM_PAIRS)
        seq, target = seq.to(DEVICE), target.to(DEVICE)
        logits = model(seq, temperature=temp)
        loss = F.cross_entropy(logits, target)
        opt.zero_grad(); loss.backward(); opt.step()

    # Eval at temperature near-zero (deterministic)
    model.eval()
    correct = 0; total = 0
    with torch.no_grad():
        for _ in range(20):
            seq, target = make_assoc_batch(BATCH, SEQ_LEN, VOCAB_SIZE, NUM_PAIRS)
            seq, target = seq.to(DEVICE), target.to(DEVICE)
            preds = model(seq, temperature=0.01).argmax(-1)
            correct += (preds == target).sum().item()
            total += BATCH
    return correct / total


class Exp143GumbelTemperatureSchedule(Experiment):
    experiment_id = "exp_14_3"
    hypothesis = hypothesis

    def run(self) -> ExperimentResult:
        print("  Training condition A (constant temp=0.5)...")
        acc_A = train_condition("A", 0)
        print("  Training condition B (linear warmup 1.0->0.5)...")
        acc_B = train_condition("B", 1)
        print("  Training condition C (cosine anneal 1.0->0.1)...")
        acc_C = train_condition("C", 2)

        diff_C_vs_A = acc_C - acc_A

        metrics = {
            "acc_A": round(acc_A, 4),
            "acc_B": round(acc_B, 4),
            "acc_C": round(acc_C, 4),
            "diff_C_vs_A": round(diff_C_vs_A, 4),
        }

        config = dict(VOCAB_SIZE=VOCAB_SIZE, SEQ_LEN=SEQ_LEN, HIDDEN_DIM=HIDDEN_DIM,
                      MEMORY_SLOTS=MEMORY_SLOTS, NUM_PAIRS=NUM_PAIRS,
                      K_WRITE=K_WRITE, STEPS=STEPS, BATCH=BATCH)

        if acc_C > acc_A + 0.02:
            outcome = OUTCOME_SUPPORTED
            notes = f"Cosine annealing C exceeds constant A by {diff_C_vs_A:.3f} (>0.02)."
        elif acc_A >= acc_C - 0.01:
            outcome = OUTCOME_REFUTED
            notes = f"Constant A >= cosine C - 0.01: acc_A={acc_A:.3f}, acc_C={acc_C:.3f}."
        else:
            outcome = OUTCOME_INCONCLUSIVE
            notes = f"Difference {diff_C_vs_A:.3f} < 0.02; inconclusive."

        return self.result(outcome, metrics, notes, config)


if __name__ == "__main__":
    Exp143GumbelTemperatureSchedule().execute()
