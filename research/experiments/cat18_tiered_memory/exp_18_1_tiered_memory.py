"""
exp_18_1_tiered_memory.py

Hypothesis: Two-tier memory (16-slot fast + 64-slot slow with learned demotion)
outperforms flat 64-slot memory by >5% on long-context tasks.
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
FAST_SLOTS         = 16
SLOW_SLOTS         = 64
NUM_PAIRS          = 6
CRITICAL_POSITIONS = list(range(6))    # first 6 KV-pair slots are "critical"
STEPS              = 400
BATCH              = 32


# ── Data generator ─────────────────────────────────────────────────────────────
def make_long_assoc_batch(batch_size, seq_len, vocab_size, num_pairs):
    """
    Critical KV pairs at positions 0-5 (tokens 0-11).
    Query always targets a critical pair (first half).
    """
    seq    = torch.zeros(batch_size, seq_len, dtype=torch.long)
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
            seq[b, p] = torch.randint(4, vocab_size // 3, (1,)).item()

        # Query targets a critical pair from the first num_pairs//2 pairs
        num_critical = max(1, num_pairs // 2)
        qi                  = torch.randint(0, num_critical, (1,)).item()
        seq[b, seq_len - 3] = 2
        seq[b, seq_len - 2] = keys[qi]
        seq[b, seq_len - 1] = 0
        target[b]           = vals[qi]

    return seq, target


# ── Shared encoder / read-head ─────────────────────────────────────────────────
class Encoder(nn.Module):
    def __init__(self, vocab_size, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.ff    = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

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


# ── Condition A: flat memory (64 slots, LRU eviction) ─────────────────────────
class FlatMemoryModel(nn.Module):
    """
    Encodes the full sequence, then fills slots in order (LRU = earliest written).
    If the sequence is longer than slots, later tokens overwrite oldest.
    We process all tokens and keep the last `flat_slots` unique ones (LRU = drop oldest).
    """

    def __init__(self, vocab_size, hidden_dim, flat_slots):
        super().__init__()
        self.encoder   = Encoder(vocab_size, hidden_dim)
        self.gate_net  = nn.Linear(hidden_dim, 1)
        self.read_head = ReadHead(hidden_dim, vocab_size)
        self.hidden_dim  = hidden_dim
        self.flat_slots  = flat_slots

    def forward(self, seq):
        B, T = seq.shape
        h    = self.encoder(seq)  # (B, T, H)

        # gate to score each token for keeping
        gate = torch.sigmoid(self.gate_net(h).squeeze(-1))  # (B, T)
        k    = min(self.flat_slots, T)
        _, top_idx = torch.topk(gate, k, dim=-1)
        memory     = torch.gather(h, 1, top_idx.unsqueeze(-1).expand(-1, -1, self.hidden_dim))
        mask       = torch.ones(B, k)

        query_h = h[:, -1, :]
        return self.read_head(query_h, memory, mask)


# ── Tiered memory module ───────────────────────────────────────────────────────
class TieredMemoryModule(nn.Module):
    """
    Processes a full sequence through fast+slow memory tiers.

    fast_mem: (B, fast_slots, H)
    slow_mem: (B, slow_slots, H)
    fast_valid: (B, fast_slots)  - boolean mask of occupied slots
    slow_valid: (B, slow_slots)

    LRU tracked via age tensors (integer counts; higher = older).

    Demotion: when fast is full and a new token arrives, the fast slot with the
    lowest demotion_net score is evicted to slow memory (oldest slow slot
    dropped if slow is also full).

    Promotion (Condition C): after writing to slow, a promotion gate can pull
    a slow slot back into fast.
    """

    def __init__(self, hidden_dim, fast_slots, slow_slots,
                 use_promotion=False):
        super().__init__()
        self.hidden_dim    = hidden_dim
        self.fast_slots    = fast_slots
        self.slow_slots    = slow_slots
        self.use_promotion = use_promotion

        self.write_gate     = nn.Linear(hidden_dim, 1)      # whether to write at all
        self.demotion_net   = nn.Linear(hidden_dim, 1)      # scores fast slots for demotion
        if use_promotion:
            self.promotion_gate = nn.Linear(hidden_dim * 2, 1)  # (fast_ctx, slow_slot) -> score

        self.read_head = ReadHead(hidden_dim, VOCAB_SIZE)

    def _process_sequence(self, h):
        """
        h: (B, T, H) — encoded hidden states.
        Returns memory (B, F+S, H), mask (B, F+S).
        """
        B    = h.shape[0]
        T    = h.shape[1]

        # Initialize fast & slow memory with zeros
        fast_mem  = torch.zeros(B, self.fast_slots,  self.hidden_dim)
        slow_mem  = torch.zeros(B, self.slow_slots,  self.hidden_dim)
        fast_age  = torch.zeros(B, self.fast_slots,  dtype=torch.long)   # lower = newer
        slow_age  = torch.zeros(B, self.slow_slots,  dtype=torch.long)
        fast_used = torch.zeros(B, self.fast_slots,  dtype=torch.bool)
        slow_used = torch.zeros(B, self.slow_slots,  dtype=torch.bool)

        for t in range(T - 3):   # leave query tokens unwritten
            tok_h = h[:, t, :]  # (B, H)

            # Decide whether to write this token
            write_score = torch.sigmoid(self.write_gate(tok_h)).squeeze(-1)  # (B,)

            # Increment ages
            fast_age = fast_age + fast_used.long()
            slow_age = slow_age + slow_used.long()

            for b in range(B):
                if write_score[b].item() < 0.4:
                    continue

                fast_free = (~fast_used[b]).nonzero(as_tuple=False)
                if fast_free.numel() > 0:
                    slot = fast_free[0, 0].item()
                    fast_mem[b, slot]  = tok_h[b].detach()
                    fast_age[b, slot]  = 0
                    fast_used[b, slot] = True
                else:
                    # Fast is full — demote lowest-scoring fast slot
                    dem_scores = self.demotion_net(fast_mem[b])  # (fast_slots, 1)
                    dem_slot   = dem_scores.squeeze(-1).argmin().item()
                    demoted_h  = fast_mem[b, dem_slot].clone()

                    # Find a free slow slot or evict oldest
                    slow_free = (~slow_used[b]).nonzero(as_tuple=False)
                    if slow_free.numel() > 0:
                        ss = slow_free[0, 0].item()
                    else:
                        ss = slow_age[b].argmax().item()

                    slow_mem[b, ss]  = demoted_h
                    slow_age[b, ss]  = 0
                    slow_used[b, ss] = True

                    # Write new token to freed fast slot
                    fast_mem[b, dem_slot]  = tok_h[b].detach()
                    fast_age[b, dem_slot]  = 0
                    fast_used[b, dem_slot] = True

                    # Optional promotion: pull most relevant slow slot to fast
                    if self.use_promotion and slow_used[b].any():
                        fast_ctx_mean = fast_mem[b][fast_used[b]].mean(0, keepdim=True)  # (1, H)
                        fast_ctx_exp  = fast_ctx_mean.expand(self.slow_slots, -1)
                        prom_input    = torch.cat([fast_ctx_exp, slow_mem[b]], dim=-1)
                        prom_scores   = torch.sigmoid(self.promotion_gate(prom_input)).squeeze(-1)
                        prom_scores   = prom_scores * slow_used[b].float()
                        best_slow     = prom_scores.argmax().item()
                        if prom_scores[best_slow].item() > 0.6:
                            # swap best slow into freed fast slot (use lru fast slot)
                            lru_fast = fast_age[b].argmax().item()
                            # demote lru_fast to slow
                            slow_mem[b, ss]   = fast_mem[b, lru_fast].clone()
                            slow_used[b, ss]  = True
                            # promote
                            fast_mem[b, lru_fast]  = slow_mem[b, best_slow].clone()
                            fast_used[b, lru_fast] = True
                            slow_used[b, best_slow] = False

        all_mem  = torch.cat([fast_mem, slow_mem], dim=1)   # (B, F+S, H)
        all_used = torch.cat([fast_used, slow_used], dim=1) # (B, F+S)
        return all_mem, all_used.float(), slow_used.float()

    def forward(self, seq, encoder):
        h      = encoder(seq)
        memory, mask, slow_mask = self._process_sequence(h)
        query_h = h[:, -1, :]
        logits  = self.read_head(query_h, memory, mask)
        return logits, slow_mask


# ── Full condition models ──────────────────────────────────────────────────────
class TieredModelBase(nn.Module):
    def __init__(self, vocab_size, hidden_dim, fast_slots, slow_slots,
                 use_promotion=False):
        super().__init__()
        self.encoder = Encoder(vocab_size, hidden_dim)
        self.memory  = TieredMemoryModule(hidden_dim, fast_slots, slow_slots,
                                          use_promotion=use_promotion)

    def forward(self, seq):
        return self.memory(seq, self.encoder)


# ── Training helpers ───────────────────────────────────────────────────────────
def train_flat(steps):
    model = FlatMemoryModel(VOCAB_SIZE, HIDDEN_DIM, SLOW_SLOTS)
    opt   = Adam(model.parameters(), lr=LR)
    for _ in range(steps):
        seq, target = make_long_assoc_batch(BATCH, SEQ_LEN, VOCAB_SIZE, NUM_PAIRS)
        loss = F.cross_entropy(model(seq), target)
        opt.zero_grad(); loss.backward(); opt.step()
    return model


def train_tiered(steps, use_promotion=False):
    model = TieredModelBase(VOCAB_SIZE, HIDDEN_DIM, FAST_SLOTS, SLOW_SLOTS,
                            use_promotion=use_promotion)
    opt   = Adam(model.parameters(), lr=LR)
    for _ in range(steps):
        seq, target = make_long_assoc_batch(BATCH, SEQ_LEN, VOCAB_SIZE, NUM_PAIRS)
        logits, _   = model(seq)
        loss        = F.cross_entropy(logits, target)
        opt.zero_grad(); loss.backward(); opt.step()
    return model


def eval_flat(model, steps=400):
    correct = 0
    for _ in range(steps):
        seq, target = make_long_assoc_batch(BATCH, SEQ_LEN, VOCAB_SIZE, NUM_PAIRS)
        with torch.no_grad():
            logits = model(seq)
        correct += (logits.argmax(-1) == target).sum().item()
    return correct / (steps * BATCH)


def eval_tiered(model, steps=400):
    """Returns (accuracy, critical_coverage)."""
    correct  = 0
    coverage = 0.0
    count    = 0
    for _ in range(steps):
        seq, target = make_long_assoc_batch(BATCH, SEQ_LEN, VOCAB_SIZE, NUM_PAIRS)
        with torch.no_grad():
            logits, slow_mask = model(seq)
        correct  += (logits.argmax(-1) == target).sum().item()
        # coverage: fraction of slow slots used (proxy for preservation)
        coverage += slow_mask.mean().item()
        count    += 1
    n = steps * BATCH
    return correct / n, coverage / count


# ── Experiment class ───────────────────────────────────────────────────────────
class Exp181TieredMemory(Experiment):
    experiment_id = "exp_18_1"
    hypothesis    = (
        "Two-tier memory (16-slot fast + 64-slot slow with learned demotion) "
        "outperforms flat 64-slot memory by >5% on long-context tasks."
    )

    def run(self) -> ExperimentResult:
        config = dict(
            VOCAB_SIZE=VOCAB_SIZE, HIDDEN_DIM=HIDDEN_DIM, SEQ_LEN=SEQ_LEN,
            FAST_SLOTS=FAST_SLOTS, SLOW_SLOTS=SLOW_SLOTS,
            NUM_PAIRS=NUM_PAIRS, STEPS=STEPS, BATCH=BATCH,
        )

        print("Training Condition A: flat memory (64 slots) …")
        model_a = train_flat(STEPS)
        acc_a   = eval_flat(model_a)
        print(f"  acc_A = {acc_a:.4f}")

        print("Training Condition B: tiered memory (16 fast + 64 slow) …")
        model_b = train_tiered(STEPS, use_promotion=False)
        acc_b, cov_b = eval_tiered(model_b)
        print(f"  acc_B = {acc_b:.4f}  slow_coverage = {cov_b:.4f}")

        print("Training Condition C: tiered + promotion …")
        model_c = train_tiered(STEPS, use_promotion=True)
        acc_c, cov_c = eval_tiered(model_c)
        print(f"  acc_C = {acc_c:.4f}  slow_coverage = {cov_c:.4f}")

        gap_b = acc_b - acc_a
        gap_c = acc_c - acc_a
        metrics = dict(
            acc_A=round(acc_a, 4), acc_B=round(acc_b, 4), acc_C=round(acc_c, 4),
            gap_B=round(gap_b, 4), gap_C=round(gap_c, 4),
            slow_coverage_B=round(cov_b, 4), slow_coverage_C=round(cov_c, 4),
        )

        if acc_b > acc_a + 0.05 and cov_b > 0.70:
            outcome = OUTCOME_SUPPORTED
            notes   = f"Tiered better by {gap_b:.3f}; coverage={cov_b:.3f}"
        elif acc_a >= acc_b - 0.02:
            outcome = OUTCOME_REFUTED
            notes   = f"Flat memory matches tiered (gap={gap_b:.3f})"
        else:
            outcome = OUTCOME_INCONCLUSIVE
            notes   = f"Coverage={cov_b:.3f} improved but acc gap={gap_b:.3f} < 0.05"

        return self.result(outcome, metrics, notes, config)


if __name__ == "__main__":
    Exp181TieredMemory().execute()
