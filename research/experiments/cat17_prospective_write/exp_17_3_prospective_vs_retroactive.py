"""
exp_17_3_prospective_vs_retroactive.py

Hypothesis: Prospective and retroactive writing are redundant — their
combination yields <1.5x the gain of either alone.
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

# ── Constants ──────────────────────────────────────────────────────────────────
DEVICE       = "cpu"
LR           = 3e-4
VOCAB_SIZE   = 64
SEQ_LEN      = 32
HIDDEN_DIM   = 64
MEMORY_SLOTS = 8
NUM_PAIRS    = 4
STEPS        = 400
BATCH        = 32


# ── Data generator ─────────────────────────────────────────────────────────────
def make_assoc_batch(batch_size, seq_len, vocab_size, num_pairs):
    seq    = torch.zeros(batch_size, seq_len, dtype=torch.long)
    target = torch.zeros(batch_size, dtype=torch.long)
    for b in range(batch_size):
        keys = torch.randint(4, max(8, vocab_size // 3), (num_pairs * 3,)).unique()[:num_pairs]
        while len(keys) < num_pairs:
            keys = torch.cat([keys, torch.randint(4, vocab_size // 3, (1,))])[:num_pairs]
        vals = torch.randint(vocab_size // 2, vocab_size, (num_pairs,))
        pos  = 0
        for i in range(num_pairs):
            if pos + 1 < seq_len - 3:
                seq[b, pos] = keys[i]; seq[b, pos + 1] = vals[i]; pos += 2
        for p in range(pos, seq_len - 3):
            seq[b, p] = 3
        qi                  = torch.randint(0, num_pairs, (1,)).item()
        seq[b, seq_len - 3] = 2
        seq[b, seq_len - 2] = keys[qi]
        seq[b, seq_len - 1] = 0
        target[b]           = vals[qi]
    return seq, target


# ── Shared modules ─────────────────────────────────────────────────────────────
class Encoder(nn.Module):
    def __init__(self, vocab_size, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.ff    = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm  = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        h = self.embed(x)
        return self.norm(h + self.ff(h))


class ReadHead(nn.Module):
    def __init__(self, hidden_dim, vocab_size):
        super().__init__()
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out    = nn.Linear(hidden_dim, vocab_size)

    def forward(self, query_h, memory, mask):
        q      = self.q_proj(query_h)
        scores = torch.bmm(memory, q.unsqueeze(-1)).squeeze(-1)
        scores = scores.masked_fill(mask == 0, -1e9)
        attn   = torch.softmax(scores, dim=-1)
        ctx    = (attn.unsqueeze(-1) * memory).sum(1)
        return self.out(ctx)


# ── Four model variants ────────────────────────────────────────────────────────
# Condition A: forward-only baseline (simple top-K by score)
class ForwardOnlyModel(nn.Module):
    def __init__(self, vocab_size, hidden_dim, memory_slots):
        super().__init__()
        self.encoder      = Encoder(vocab_size, hidden_dim)
        self.gate_net     = nn.Linear(hidden_dim, 1)
        self.read_head    = ReadHead(hidden_dim, vocab_size)
        self.hidden_dim   = hidden_dim
        self.memory_slots = memory_slots

    def forward(self, seq):
        B, T = seq.shape
        h    = self.encoder(seq)
        gate = torch.sigmoid(self.gate_net(h).squeeze(-1))
        topk = min(self.memory_slots, T)
        _, top_idx = torch.topk(gate, topk, dim=-1)
        memory = torch.gather(h, 1, top_idx.unsqueeze(-1).expand(-1, -1, self.hidden_dim))
        mask   = torch.ones(B, topk)
        return self.read_head(h[:, -1, :], memory, mask)


# Condition B: prospective write gate
class ProspectiveModel(nn.Module):
    def __init__(self, vocab_size, hidden_dim, memory_slots):
        super().__init__()
        self.encoder      = Encoder(vocab_size, hidden_dim)
        self.gate_net     = nn.Linear(hidden_dim * 2, 1)
        self.read_head    = ReadHead(hidden_dim, vocab_size)
        self.hidden_dim   = hidden_dim
        self.memory_slots = memory_slots

    def forward(self, seq):
        B, T   = seq.shape
        h      = self.encoder(seq)
        mean_h = h.mean(dim=1, keepdim=True).expand(-1, T, -1)
        gate   = torch.sigmoid(self.gate_net(torch.cat([h, mean_h], dim=-1)).squeeze(-1))
        topk   = min(self.memory_slots, T)
        _, top_idx = torch.topk(gate, topk, dim=-1)
        memory = torch.gather(h, 1, top_idx.unsqueeze(-1).expand(-1, -1, self.hidden_dim))
        mask   = torch.ones(B, topk)
        return self.read_head(h[:, -1, :], memory, mask)


# Condition C: retroactive write (two-pass)
class RetroactiveModel(nn.Module):
    """
    Pass 1: gate selects top-6 slots (primary).
    Pass 2: revision gate over the remaining tokens, using mean-pool context
            from already-selected slots — adds top-2 from skipped.
    Total: min(8, T) slots.
    """

    def __init__(self, vocab_size, hidden_dim, memory_slots):
        super().__init__()
        self.encoder       = Encoder(vocab_size, hidden_dim)
        self.gate1         = nn.Linear(hidden_dim, 1)
        self.gate2         = nn.Linear(hidden_dim * 2, 1)
        self.read_head     = ReadHead(hidden_dim, vocab_size)
        self.hidden_dim    = hidden_dim
        self.memory_slots  = memory_slots
        self.primary_slots = max(1, memory_slots - 2)

    def forward(self, seq):
        B, T = seq.shape
        h    = self.encoder(seq)

        # Pass 1: select primary_slots
        gate1   = torch.sigmoid(self.gate1(h).squeeze(-1))  # (B, T)
        k1      = min(self.primary_slots, T)
        _, idx1 = torch.topk(gate1, k1, dim=-1)             # (B, k1)
        mem1    = torch.gather(h, 1, idx1.unsqueeze(-1).expand(-1, -1, self.hidden_dim))

        # Pass 2: build revision gate using mean of selected context
        ctx1       = mem1.mean(dim=1, keepdim=True).expand(-1, T, -1)
        gate2      = torch.sigmoid(self.gate2(torch.cat([h, ctx1], dim=-1)).squeeze(-1))

        # zero out already-selected positions so we pick different ones
        mask_selected = torch.zeros(B, T)
        mask_selected.scatter_(1, idx1, 1.0)
        gate2 = gate2 * (1.0 - mask_selected)

        k2 = min(self.memory_slots - k1, T - k1)
        if k2 > 0:
            _, idx2  = torch.topk(gate2, k2, dim=-1)
            mem2     = torch.gather(h, 1, idx2.unsqueeze(-1).expand(-1, -1, self.hidden_dim))
            memory   = torch.cat([mem1, mem2], dim=1)
        else:
            memory = mem1

        mask = torch.ones(B, memory.shape[1])
        return self.read_head(h[:, -1, :], memory, mask)


# Condition D: combined (prospective gate + retroactive revision)
class CombinedModel(nn.Module):
    def __init__(self, vocab_size, hidden_dim, memory_slots):
        super().__init__()
        self.encoder       = Encoder(vocab_size, hidden_dim)
        self.gate1         = nn.Linear(hidden_dim * 2, 1)   # prospective
        self.gate2         = nn.Linear(hidden_dim * 2, 1)   # retroactive revision
        self.read_head     = ReadHead(hidden_dim, vocab_size)
        self.hidden_dim    = hidden_dim
        self.memory_slots  = memory_slots
        self.primary_slots = max(1, memory_slots - 2)

    def forward(self, seq):
        B, T   = seq.shape
        h      = self.encoder(seq)
        mean_h = h.mean(dim=1, keepdim=True).expand(-1, T, -1)

        # Prospective first pass
        gate1   = torch.sigmoid(self.gate1(torch.cat([h, mean_h], dim=-1)).squeeze(-1))
        k1      = min(self.primary_slots, T)
        _, idx1 = torch.topk(gate1, k1, dim=-1)
        mem1    = torch.gather(h, 1, idx1.unsqueeze(-1).expand(-1, -1, self.hidden_dim))

        # Retroactive revision
        ctx1   = mem1.mean(dim=1, keepdim=True).expand(-1, T, -1)
        gate2  = torch.sigmoid(self.gate2(torch.cat([h, ctx1], dim=-1)).squeeze(-1))
        mask_s = torch.zeros(B, T)
        mask_s.scatter_(1, idx1, 1.0)
        gate2  = gate2 * (1.0 - mask_s)
        k2     = min(self.memory_slots - k1, T - k1)
        if k2 > 0:
            _, idx2 = torch.topk(gate2, k2, dim=-1)
            mem2    = torch.gather(h, 1, idx2.unsqueeze(-1).expand(-1, -1, self.hidden_dim))
            memory  = torch.cat([mem1, mem2], dim=1)
        else:
            memory = mem1

        mask = torch.ones(B, memory.shape[1])
        return self.read_head(h[:, -1, :], memory, mask)


# ── Training / evaluation helpers ─────────────────────────────────────────────
def train_model(model, steps):
    opt = Adam(model.parameters(), lr=LR)
    for _ in range(steps):
        seq, target = make_assoc_batch(BATCH, SEQ_LEN, VOCAB_SIZE, NUM_PAIRS)
        loss = F.cross_entropy(model(seq), target)
        opt.zero_grad(); loss.backward(); opt.step()
    return model


def eval_model(model, steps=400):
    correct = 0
    for _ in range(steps):
        seq, target = make_assoc_batch(BATCH, SEQ_LEN, VOCAB_SIZE, NUM_PAIRS)
        with torch.no_grad():
            logits = model(seq)
        correct += (logits.argmax(-1) == target).sum().item()
    return correct / (steps * BATCH)


# ── Experiment class ───────────────────────────────────────────────────────────
class Exp173ProspectiveVsRetroactive(Experiment):
    experiment_id = "exp_17_3"
    hypothesis    = (
        "Prospective and retroactive writing are redundant — their combination "
        "yields <1.5x the gain of either alone."
    )

    def run(self) -> ExperimentResult:
        config = dict(
            VOCAB_SIZE=VOCAB_SIZE, SEQ_LEN=SEQ_LEN, HIDDEN_DIM=HIDDEN_DIM,
            MEMORY_SLOTS=MEMORY_SLOTS, NUM_PAIRS=NUM_PAIRS, STEPS=STEPS, BATCH=BATCH,
        )

        models = {
            "A": ForwardOnlyModel(VOCAB_SIZE, HIDDEN_DIM, MEMORY_SLOTS),
            "B": ProspectiveModel(VOCAB_SIZE, HIDDEN_DIM, MEMORY_SLOTS),
            "C": RetroactiveModel(VOCAB_SIZE, HIDDEN_DIM, MEMORY_SLOTS),
            "D": CombinedModel(VOCAB_SIZE, HIDDEN_DIM, MEMORY_SLOTS),
        }
        accs = {}
        for name, model in models.items():
            print(f"Training condition {name} …")
            train_model(model, STEPS)
            accs[name] = eval_model(model)
            print(f"  acc_{name} = {accs[name]:.4f}")

        gap_b  = accs["B"] - accs["A"]
        gap_c  = accs["C"] - accs["A"]
        gap_d  = accs["D"] - accs["A"]
        best_single = max(gap_b, gap_c, 0.001)
        multiplier  = gap_d / best_single

        metrics = dict(
            acc_A=round(accs["A"], 4), acc_B=round(accs["B"], 4),
            acc_C=round(accs["C"], 4), acc_D=round(accs["D"], 4),
            gap_B=round(gap_b, 4), gap_C=round(gap_c, 4),
            gap_D=round(gap_d, 4), multiplier=round(multiplier, 4),
        )

        if gap_b < 0.02 or gap_c < 0.02:
            outcome = OUTCOME_INCONCLUSIVE
            notes   = f"gap_B={gap_b:.3f} or gap_C={gap_c:.3f} too small for valid comparison"
        elif multiplier < 1.5:
            outcome = OUTCOME_SUPPORTED
            notes   = f"Combined multiplier={multiplier:.3f} < 1.5 (redundancy confirmed)"
        elif multiplier > 2.0:
            outcome = OUTCOME_REFUTED
            notes   = f"Combined multiplier={multiplier:.3f} > 2.0 (synergy, not redundancy)"
        else:
            outcome = OUTCOME_INCONCLUSIVE
            notes   = f"Multiplier={multiplier:.3f} in ambiguous range 1.5-2.0"

        return self.result(outcome, metrics, notes, config)


if __name__ == "__main__":
    Exp173ProspectiveVsRetroactive().execute()
