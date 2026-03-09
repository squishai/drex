"""
exp_18_3_tiered_capacity_efficiency.py

Hypothesis: Tiered memory has a capacity crossover point — flat is better
below it, tiered above it.
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
DEVICE             = "cpu"
LR                 = 3e-4
VOCAB_SIZE         = 64
HIDDEN_DIM         = 64
SEQ_LEN            = 64
NUM_PAIRS          = 8
STEPS              = 400
BATCH              = 32
TOTAL_SLOTS_VALUES = [8, 16, 32, 64, 128]


# ── Data generator ─────────────────────────────────────────────────────────────
def make_long_assoc_batch(batch_size, seq_len, vocab_size, num_pairs):
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
            seq[b, p] = torch.randint(4, vocab_size // 3, (1,)).item()
        num_critical        = max(1, num_pairs // 2)
        qi                  = torch.randint(0, num_critical, (1,)).item()
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


# ── Condition A: flat memory ───────────────────────────────────────────────────
class FlatModel(nn.Module):
    def __init__(self, vocab_size, hidden_dim, flat_slots):
        super().__init__()
        self.encoder     = Encoder(vocab_size, hidden_dim)
        self.gate_net    = nn.Linear(hidden_dim, 1)
        self.read_head   = ReadHead(hidden_dim, vocab_size)
        self.hidden_dim  = hidden_dim
        self.flat_slots  = flat_slots

    def forward(self, seq):
        B, T = seq.shape
        h    = self.encoder(seq)
        gate = torch.sigmoid(self.gate_net(h).squeeze(-1))
        k    = min(self.flat_slots, T)
        _, top_idx = torch.topk(gate, k, dim=-1)
        memory = torch.gather(h, 1, top_idx.unsqueeze(-1).expand(-1, -1, self.hidden_dim))
        mask   = torch.ones(B, k)
        return self.read_head(h[:, -1, :], memory, mask)


# ── Condition B: tiered memory ─────────────────────────────────────────────────
class TieredModel(nn.Module):
    """
    fast_slots = total_slots // 4
    slow_slots = 3 * total_slots // 4
    """

    def __init__(self, vocab_size, hidden_dim, total_slots):
        super().__init__()
        self.fast_slots   = max(1, total_slots // 4)
        self.slow_slots   = max(1, 3 * total_slots // 4)
        self.hidden_dim   = hidden_dim
        self.encoder      = Encoder(vocab_size, hidden_dim)
        self.write_gate   = nn.Linear(hidden_dim, 1)
        self.demotion_net = nn.Linear(hidden_dim, 1)
        self.read_head    = ReadHead(hidden_dim, vocab_size)

    def _process(self, h):
        B = h.shape[0]; T = h.shape[1]
        fast_mem  = torch.zeros(B, self.fast_slots, self.hidden_dim)
        slow_mem  = torch.zeros(B, self.slow_slots, self.hidden_dim)
        fast_age  = torch.zeros(B, self.fast_slots, dtype=torch.long)
        slow_age  = torch.zeros(B, self.slow_slots, dtype=torch.long)
        fast_used = torch.zeros(B, self.fast_slots, dtype=torch.bool)
        slow_used = torch.zeros(B, self.slow_slots, dtype=torch.bool)

        for t in range(T - 3):
            tok_h = h[:, t, :]
            ws    = torch.sigmoid(self.write_gate(tok_h)).squeeze(-1)
            fast_age = fast_age + fast_used.long()
            slow_age = slow_age + slow_used.long()

            for b in range(B):
                if ws[b].item() < 0.4:
                    continue
                ff = (~fast_used[b]).nonzero(as_tuple=False)
                if ff.numel() > 0:
                    sl = ff[0, 0].item()
                    fast_mem[b, sl]  = tok_h[b].detach()
                    fast_age[b, sl]  = 0
                    fast_used[b, sl] = True
                else:
                    ds  = self.demotion_net(fast_mem[b]).squeeze(-1)
                    dem = ds.argmin().item()
                    dh  = fast_mem[b, dem].clone()
                    sf  = (~slow_used[b]).nonzero(as_tuple=False)
                    ss  = sf[0, 0].item() if sf.numel() > 0 else slow_age[b].argmax().item()
                    slow_mem[b, ss]  = dh
                    slow_age[b, ss]  = 0
                    slow_used[b, ss] = True
                    fast_mem[b, dem]  = tok_h[b].detach()
                    fast_age[b, dem]  = 0
                    fast_used[b, dem] = True

        all_mem  = torch.cat([fast_mem, slow_mem], dim=1)
        all_mask = torch.cat([fast_used, slow_used], dim=1).float()
        return all_mem, all_mask

    def forward(self, seq):
        h       = self.encoder(seq)
        mem, mk = self._process(h)
        return self.read_head(h[:, -1, :], mem, mk)


# ── Train/eval helpers ─────────────────────────────────────────────────────────
def train_and_eval(model_cls, model_kwargs, steps, eval_steps=400):
    model = model_cls(**model_kwargs)
    opt   = Adam(model.parameters(), lr=LR)
    for _ in range(steps):
        seq, target = make_long_assoc_batch(BATCH, SEQ_LEN, VOCAB_SIZE, NUM_PAIRS)
        loss = F.cross_entropy(model(seq), target)
        opt.zero_grad(); loss.backward(); opt.step()

    correct = 0
    for _ in range(eval_steps):
        seq, target = make_long_assoc_batch(BATCH, SEQ_LEN, VOCAB_SIZE, NUM_PAIRS)
        with torch.no_grad():
            logits = model(seq)
        correct += (logits.argmax(-1) == target).sum().item()
    return correct / (eval_steps * BATCH)


# ── Experiment class ───────────────────────────────────────────────────────────
class Exp183TieredCapacityEfficiency(Experiment):
    experiment_id = "exp_18_3"
    hypothesis    = (
        "Tiered memory has a capacity crossover point — flat is better "
        "below it, tiered above it."
    )

    def run(self) -> ExperimentResult:
        config = dict(
            VOCAB_SIZE=VOCAB_SIZE, HIDDEN_DIM=HIDDEN_DIM, SEQ_LEN=SEQ_LEN,
            NUM_PAIRS=NUM_PAIRS, STEPS=STEPS, BATCH=BATCH,
            TOTAL_SLOTS_VALUES=TOTAL_SLOTS_VALUES,
        )

        results_flat   = {}
        results_tiered = {}

        for total_slots in TOTAL_SLOTS_VALUES:
            print(f"total_slots={total_slots}: training flat …")
            acc_flat = train_and_eval(
                FlatModel,
                dict(vocab_size=VOCAB_SIZE, hidden_dim=HIDDEN_DIM, flat_slots=total_slots),
                STEPS,
            )
            print(f"  acc_flat={acc_flat:.4f}")

            print(f"total_slots={total_slots}: training tiered …")
            acc_tiered = train_and_eval(
                TieredModel,
                dict(vocab_size=VOCAB_SIZE, hidden_dim=HIDDEN_DIM, total_slots=total_slots),
                STEPS,
            )
            print(f"  acc_tiered={acc_tiered:.4f}")

            results_flat[total_slots]   = acc_flat
            results_tiered[total_slots] = acc_tiered

        # Find crossover
        crossover = None
        for s in TOTAL_SLOTS_VALUES:
            if results_tiered[s] >= results_flat[s]:
                crossover = s
                break

        metrics = {}
        for s in TOTAL_SLOTS_VALUES:
            metrics[f"acc_flat_{s}"]   = round(results_flat[s], 4)
            metrics[f"acc_tiered_{s}"] = round(results_tiered[s], 4)
        metrics["crossover_capacity"] = crossover

        tiered_always_better = all(results_tiered[s] >= results_flat[s]
                                   for s in TOTAL_SLOTS_VALUES)
        tiered_always_worse  = all(results_flat[s] >= results_tiered[s]
                                   for s in TOTAL_SLOTS_VALUES)

        if crossover is not None and 16 <= crossover <= 64:
            outcome = OUTCOME_SUPPORTED
            notes   = f"Crossover at total_slots={crossover} (within expected range 16-64)"
        elif tiered_always_worse or crossover is None:
            outcome = OUTCOME_REFUTED
            notes   = "Flat always equal or better — no crossover point found"
        else:
            outcome = OUTCOME_INCONCLUSIVE
            notes   = (
                "Tiered uniformly better (no crossover)" if tiered_always_better
                else f"Crossover at {crossover} outside expected range 16-64"
            )

        return self.result(outcome, metrics, notes, config)


if __name__ == "__main__":
    Exp183TieredCapacityEfficiency().execute()
