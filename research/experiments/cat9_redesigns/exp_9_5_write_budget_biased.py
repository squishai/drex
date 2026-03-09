"""
Experiment 9.5 — Write Budget Biased: Adaptive Block Allocation

Hypothesis: With all KV pairs concentrated in block 1 (positions 0-7 of 32), an adaptive
write budget allocator learns non-uniform allocation (Gini > 0.5, block-1 fraction > 0.60),
outperforming uniform allocation by > 5%.
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
SEQ_LEN       = 32
HIDDEN_DIM    = 64
MEMORY_SLOTS  = 8
N_BLOCKS      = 4
BLOCK_SIZE    = SEQ_LEN // N_BLOCKS   # 8
NUM_PAIRS     = 4
BATCH_SIZE    = 32
TRAIN_STEPS   = 2000
LR            = 3e-4
DEVICE        = "cpu"
UNIFORM_SLOTS_PER_BLOCK = MEMORY_SLOTS // N_BLOCKS   # 2 per block

# ── Data ──────────────────────────────────────────────────────────────────────

def make_assoc_batch_block1(batch_size, seq_len=SEQ_LEN, vocab_size=VOCAB_SIZE, num_pairs=NUM_PAIRS):
    """All KV pairs are placed in block 0 (positions 0 to BLOCK_SIZE-1 = 0..7)."""
    seq = torch.zeros(batch_size, seq_len, dtype=torch.long)
    target = torch.zeros(batch_size, dtype=torch.long)

    for b in range(batch_size):
        keys = torch.randint(4, vocab_size // 2, (num_pairs * 2,)).unique()[:num_pairs]
        while len(keys) < num_pairs:
            keys = torch.cat([keys, torch.randint(4, vocab_size // 2, (1,))])[:num_pairs]
        vals = torch.randint(vocab_size // 2, vocab_size, (num_pairs,))

        # Place all KV pairs in block 0 (positions 0..BLOCK_SIZE-1)
        pos = 0
        for i in range(num_pairs):
            if pos + 1 < BLOCK_SIZE:
                seq[b, pos]     = keys[i]
                seq[b, pos + 1] = vals[i]
                pos += 2

        # Fill remaining positions in block 0 and all other blocks with padding
        for p in range(pos, seq_len - 3):
            seq[b, p] = 3

        qi = torch.randint(0, num_pairs, (1,)).item()
        seq[b, seq_len - 3] = 2
        seq[b, seq_len - 2] = keys[qi]
        seq[b, seq_len - 1] = 0
        target[b] = vals[qi]
    return seq, target


# ── Model components ──────────────────────────────────────────────────────────

class Encoder(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2), nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        h = self.embed(x)
        return self.norm(h + self.ff(h))


class ReadHead(nn.Module):
    def __init__(self, hidden_dim=HIDDEN_DIM, vocab_size=VOCAB_SIZE):
        super().__init__()
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out    = nn.Linear(hidden_dim, vocab_size)

    def forward(self, query_h, memory, mask):
        q = self.q_proj(query_h)
        scores = torch.bmm(memory, q.unsqueeze(-1)).squeeze(-1)
        scores = scores.masked_fill(mask == 0, -1e9)
        attn = torch.softmax(scores, dim=-1)
        ctx  = (attn.unsqueeze(-1) * memory).sum(1)
        return self.out(ctx)


class BlockAllocator(nn.Module):
    """
    Learns per-block allocation weights using block summaries.
    Outputs a probability distribution over N_BLOCKS which scales to MEMORY_SLOTS counts.
    """
    def __init__(self, hidden_dim=HIDDEN_DIM, n_blocks=N_BLOCKS):
        super().__init__()
        self.block_proj = nn.Linear(hidden_dim, hidden_dim)
        self.alloc_head = nn.Linear(hidden_dim, 1)

    def forward(self, hidden):
        """
        hidden: (B, L, H)
        Returns: slot_counts (B, N_BLOCKS) — integer counts per block summing to MEMORY_SLOTS.
        Also returns alloc_weights (B, N_BLOCKS) for analysis.
        """
        B = hidden.size(0)
        block_summaries = []
        for bl in range(N_BLOCKS):
            start = bl * BLOCK_SIZE
            end   = start + BLOCK_SIZE
            block_h = hidden[:, start:end, :].mean(1)       # (B, H)
            block_summaries.append(block_h)
        block_summaries = torch.stack(block_summaries, dim=1)  # (B, N_BLOCKS, H)

        proj = torch.relu(self.block_proj(block_summaries))   # (B, N_BLOCKS, H)
        logits = self.alloc_head(proj).squeeze(-1)             # (B, N_BLOCKS)
        alloc_weights = torch.softmax(logits, dim=-1)          # (B, N_BLOCKS) — sums to 1

        # Convert to slot counts via Gumbel-softmax straight-through for gradient flow
        # Use soft allocations scaled to MEMORY_SLOTS for training, hard for eval
        soft_counts = alloc_weights * MEMORY_SLOTS             # (B, N_BLOCKS)
        return soft_counts, alloc_weights


# ── Memory selection ──────────────────────────────────────────────────────────

def select_uniform_memory(hidden):
    """Select UNIFORM_SLOTS_PER_BLOCK tokens from each block."""
    B, L, H = hidden.shape
    selected = []
    for bl in range(N_BLOCKS):
        start = bl * BLOCK_SIZE
        end   = start + BLOCK_SIZE
        block_h = hidden[:, start:end, :]           # (B, BLOCK_SIZE, H)
        n = min(UNIFORM_SLOTS_PER_BLOCK, BLOCK_SIZE)
        # Take first n tokens from each block (deterministic)
        selected.append(block_h[:, :n, :])
    memory = torch.cat(selected, dim=1)              # (B, MEMORY_SLOTS, H)
    mask   = torch.ones(B, memory.size(1), device=hidden.device)
    return memory, mask


def select_adaptive_memory(hidden, soft_counts):
    """
    Select tokens from each block proportionally to soft_counts.
    Uses top-k selection within each block.
    """
    B, L, H = hidden.shape
    all_selected = []

    # Compute per-token importance (norm) for selection within each block
    token_scores = hidden.norm(dim=-1)  # (B, L)

    for bl in range(N_BLOCKS):
        start = bl * BLOCK_SIZE
        end   = start + BLOCK_SIZE
        # Round soft count to nearest integer, clip to [0, BLOCK_SIZE]
        n_slots_soft = soft_counts[:, bl]  # (B,)
        # Use floor for simplicity; at least 1 slot if weight > threshold
        n_slots = n_slots_soft.round().clamp(0, BLOCK_SIZE).long()

        block_h = hidden[:, start:end, :]          # (B, BLOCK_SIZE, H)
        block_scores = token_scores[:, start:end]   # (B, BLOCK_SIZE)

        # For each sample in batch, select n_slots[b] tokens
        # Use a soft approach: weighted selection using block scores, gated by soft_counts
        # Differentiable: multiply token embeddings by their soft selection weight
        max_n = min(BLOCK_SIZE, max(1, n_slots.max().item()))
        topk_idx = block_scores.topk(max_n, dim=1).indices  # (B, max_n)
        selected = block_h.gather(1, topk_idx.unsqueeze(-1).expand(-1, -1, H))  # (B, max_n, H)

        # Weight by soft allocation (so gradient flows)
        weight = (n_slots_soft / BLOCK_SIZE).unsqueeze(-1).unsqueeze(-1)  # (B, 1, 1)
        selected = selected * weight

        all_selected.append(selected)

    # Concatenate and take top MEMORY_SLOTS by norm
    all_mem = torch.cat(all_selected, dim=1)    # (B, sum, H)
    mem_norms = all_mem.norm(dim=-1)
    k = min(MEMORY_SLOTS, all_mem.size(1))
    topk_idx = mem_norms.topk(k, dim=1).indices
    memory = all_mem.gather(1, topk_idx.unsqueeze(-1).expand(-1, -1, H))
    mask   = torch.ones(B, k, device=hidden.device)
    return memory, mask


# ── Gini coefficient ──────────────────────────────────────────────────────────

def gini_coefficient(allocations: torch.Tensor) -> float:
    """Compute Gini over a (N,) allocation vector."""
    n = len(allocations)
    if n <= 1:
        return 0.0
    x = allocations.float().sort().values
    idx = torch.arange(1, n + 1, dtype=torch.float)
    return (2.0 * (idx * x).sum() / (n * x.sum()) - (n + 1) / n).item()


# ── Training ──────────────────────────────────────────────────────────────────

def train_uniform(enc, rh):
    opt = Adam(list(enc.parameters()) + list(rh.parameters()), lr=LR)
    enc.train(); rh.train()
    for _ in range(TRAIN_STEPS):
        seq, target = make_assoc_batch_block1(BATCH_SIZE)
        hidden = enc(seq)
        memory, mask = select_uniform_memory(hidden)
        logits = rh(hidden[:, -1, :], memory, mask)
        loss = F.cross_entropy(logits, target)
        opt.zero_grad()
        loss.backward()
        opt.step()


def train_adaptive(enc, rh, allocator):
    opt = Adam(list(enc.parameters()) + list(rh.parameters()) + list(allocator.parameters()), lr=LR)
    enc.train(); rh.train(); allocator.train()
    for _ in range(TRAIN_STEPS):
        seq, target = make_assoc_batch_block1(BATCH_SIZE)
        hidden = enc(seq)
        soft_counts, _ = allocator(hidden)
        memory, mask = select_adaptive_memory(hidden, soft_counts)
        logits = rh(hidden[:, -1, :], memory, mask)
        loss = F.cross_entropy(logits, target)
        opt.zero_grad()
        loss.backward()
        opt.step()


def eval_model(enc, rh, allocator=None, n_batches=50):
    enc.eval(); rh.eval()
    if allocator is not None:
        allocator.eval()
    total_acc = 0.0
    all_weights = []
    with torch.no_grad():
        for _ in range(n_batches):
            seq, target = make_assoc_batch_block1(BATCH_SIZE)
            hidden = enc(seq)
            if allocator is not None:
                soft_counts, alloc_weights = allocator(hidden)
                all_weights.append(alloc_weights)
                memory, mask = select_adaptive_memory(hidden, soft_counts)
            else:
                memory, mask = select_uniform_memory(hidden)
            logits = rh(hidden[:, -1, :], memory, mask)
            total_acc += (logits.argmax(-1) == target).float().mean().item()
    acc = total_acc / n_batches
    if all_weights:
        mean_weights = torch.cat(all_weights, dim=0).mean(0)  # (N_BLOCKS,)
        block1_frac = mean_weights[0].item()
        gini = gini_coefficient(mean_weights)
    else:
        block1_frac = 1.0 / N_BLOCKS
        gini = 0.0
        mean_weights = torch.ones(N_BLOCKS) / N_BLOCKS
    return acc, block1_frac, gini, mean_weights


# ── Experiment class ──────────────────────────────────────────────────────────

class Exp95WriteBudgetBiased(Experiment):
    experiment_id = "exp_9_5"
    hypothesis = (
        "With all KV pairs concentrated in block 1 (positions 0-7 of 32), an adaptive "
        "write budget allocator learns non-uniform allocation (Gini > 0.5, block-1 "
        "fraction > 0.60), outperforming uniform allocation by > 5%."
    )

    def run(self) -> ExperimentResult:
        # ── Uniform baseline ─────────────────────────────────────────────────
        print("  Training uniform baseline ...")
        enc_uni = Encoder().to(DEVICE)
        rh_uni  = ReadHead().to(DEVICE)
        train_uniform(enc_uni, rh_uni)
        uniform_acc, _, _, _ = eval_model(enc_uni, rh_uni, allocator=None)
        print(f"    uniform_acc={uniform_acc:.3f}")

        # ── Adaptive allocator ────────────────────────────────────────────────
        print("  Training adaptive allocator ...")
        enc_adp  = Encoder().to(DEVICE)
        rh_adp   = ReadHead().to(DEVICE)
        allocator = BlockAllocator().to(DEVICE)
        train_adaptive(enc_adp, rh_adp, allocator)
        adaptive_acc, block1_frac, gini, mean_weights = eval_model(enc_adp, rh_adp, allocator)
        print(f"    adaptive_acc={adaptive_acc:.3f}, block1_frac={block1_frac:.3f}, gini={gini:.3f}")
        print(f"    mean_alloc_weights={[round(w.item(),3) for w in mean_weights]}")

        acc_delta = adaptive_acc - uniform_acc

        # ── Outcome ───────────────────────────────────────────────────────────
        if acc_delta > 0.05 and gini > 0.50 and block1_frac > 0.60:
            outcome = OUTCOME_SUPPORTED
        elif acc_delta < 0.01:
            outcome = OUTCOME_REFUTED
        elif acc_delta > 0.01 and gini < 0.50:
            outcome = OUTCOME_INCONCLUSIVE
        else:
            outcome = OUTCOME_INCONCLUSIVE

        metrics = {
            "uniform_acc":           round(uniform_acc, 4),
            "adaptive_acc":          round(adaptive_acc, 4),
            "acc_delta":             round(acc_delta, 4),
            "block1_allocation_frac":round(block1_frac, 4),
            "gini_coefficient":      round(gini, 4),
            "mean_block_weights":    [round(w.item(), 4) for w in mean_weights],
        }
        notes = (
            f"Uniform acc={uniform_acc:.3f}, adaptive acc={adaptive_acc:.3f}, "
            f"delta={acc_delta:.3f}. "
            f"Block-1 fraction={block1_frac:.3f}, Gini={gini:.3f}. "
            f"Block weights: {[round(w.item(),3) for w in mean_weights]}."
        )
        config = {
            "vocab_size": VOCAB_SIZE, "seq_len": SEQ_LEN,
            "hidden_dim": HIDDEN_DIM, "memory_slots": MEMORY_SLOTS,
            "n_blocks": N_BLOCKS, "num_pairs": NUM_PAIRS,
            "train_steps": TRAIN_STEPS, "block_size": BLOCK_SIZE,
        }
        return self.result(outcome, metrics, notes, config)


if __name__ == "__main__":
    Exp95WriteBudgetBiased().execute()
