"""
Experiment 3.7 — Write Budget Allocation

Hypothesis: A controller given a fixed write budget learns to allocate it
non-uniformly in a way that improves performance over uniform allocation.

Setup:
  - SEQ_LEN=32, MEMORY_SLOTS=6 (write budget)
  - 4 equal blocks of 8 tokens each
  - Policy A (uniform): write 1-2 tokens from each block equally
  - Policy B (adaptive): learned allocation predicts slots per block, top-k within each block
  - Measure: distribution of writes across blocks, downstream accuracy
  - Allocation Gini coefficient > 0.3 indicates non-uniform allocation
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

# ── Config ─────────────────────────────────────────────────────────────────────

VOCAB_SIZE    = 64
SEQ_LEN       = 32
HIDDEN_DIM    = 64
MEMORY_SLOTS  = 6
BATCH_SIZE    = 32
TRAIN_STEPS   = 1500
LR            = 3e-4
LOG_EVERY     = 100
DEVICE        = "cpu"

N_BLOCKS      = 4
BLOCK_SIZE    = SEQ_LEN // N_BLOCKS   # = 8
NUM_PAIRS     = 4
QUERY_MARKER  = 2

# Uniform allocation: distribute MEMORY_SLOTS across N_BLOCKS as evenly as possible
def uniform_slots_per_block() -> list[int]:
    base  = MEMORY_SLOTS // N_BLOCKS           # = 1
    extra = MEMORY_SLOTS % N_BLOCKS            # = 2
    # Give extra slots to the last blocks (which tend to have more recent info)
    slots = [base] * N_BLOCKS
    for i in range(extra):
        slots[N_BLOCKS - 1 - i] += 1
    return slots   # e.g. [1, 1, 2, 2]


UNIFORM_SLOTS = uniform_slots_per_block()


# ── Data Generation ────────────────────────────────────────────────────────────

def make_assoc_batch(batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Associative recall on SEQ_LEN=32.
    Key-value pairs are scattered across all 4 blocks to require multi-block coverage.
    Returns (seq, target, kv_block): kv_block[b] = which block contains the queried pair.
    """
    seq      = torch.zeros(batch_size, SEQ_LEN, dtype=torch.long)
    target   = torch.zeros(batch_size, dtype=torch.long)
    kv_block = torch.zeros(batch_size, dtype=torch.long)

    for b in range(batch_size):
        keys = torch.randint(4, 4 + NUM_PAIRS * 3, (NUM_PAIRS,)).unique()[:NUM_PAIRS]
        while len(keys) < NUM_PAIRS:
            keys = torch.cat([keys, torch.randint(4, 32, (1,))])[:NUM_PAIRS]
        vals = torch.randint(32, VOCAB_SIZE, (NUM_PAIRS,))

        # Place one key-value pair per block (first NUM_PAIRS blocks cover the pairs)
        for i in range(NUM_PAIRS):
            block_start = i * BLOCK_SIZE
            seq[b, block_start]     = keys[i]
            seq[b, block_start + 1] = vals[i]
            # Fill rest of block with filler
            for p in range(block_start + 2, block_start + BLOCK_SIZE):
                seq[b, p] = 3

        # Fill any remaining positions before query with filler
        for p in range(NUM_PAIRS * BLOCK_SIZE, SEQ_LEN - 3):
            seq[b, p] = 3

        query_idx           = torch.randint(0, NUM_PAIRS, (1,)).item()
        seq[b, SEQ_LEN - 3] = QUERY_MARKER
        seq[b, SEQ_LEN - 2] = keys[query_idx]
        seq[b, SEQ_LEN - 1] = 0
        target[b]   = vals[query_idx]
        kv_block[b] = query_idx   # which block contains the answer

    return seq, target, kv_block


# ── Encoder ───────────────────────────────────────────────────────────────────

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, HIDDEN_DIM)
        self.ff    = nn.Sequential(
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM * 2), nn.ReLU(),
            nn.Linear(HIDDEN_DIM * 2, HIDDEN_DIM),
        )
        self.norm  = nn.LayerNorm(HIDDEN_DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embed(x)
        return self.norm(h + self.ff(h))


class ReadHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.out    = nn.Linear(HIDDEN_DIM, VOCAB_SIZE)

    def forward(
        self, query_h: torch.Tensor, memory: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        q      = self.q_proj(query_h)
        scores = torch.bmm(memory, q.unsqueeze(-1)).squeeze(-1)
        scores = scores.masked_fill(mask == 0, -1e9)
        attn   = torch.softmax(scores, dim=-1)
        ctx    = (attn.unsqueeze(-1) * memory).sum(1)
        return self.out(ctx)


# ── Allocation Network ─────────────────────────────────────────────────────────

class BlockAllocator(nn.Module):
    """
    Predicts how many of the MEMORY_SLOTS to allocate to each of the N_BLOCKS.
    Uses a softmax over N_BLOCKS to get fractional allocation, then discretizes
    via Straight-Through Gumbel-Softmax for gradient flow.
    """
    def __init__(self):
        super().__init__()
        # Pool sequence into per-block summaries, then predict allocation
        self.block_pool = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.alloc_net  = nn.Sequential(
            nn.Linear(HIDDEN_DIM * N_BLOCKS, HIDDEN_DIM), nn.ReLU(),
            nn.Linear(HIDDEN_DIM, N_BLOCKS),
        )
        # Token scorer: score each token within its block
        self.token_scorer = nn.Sequential(
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM // 2), nn.ReLU(),
            nn.Linear(HIDDEN_DIM // 2, 1),
        )

    def forward(
        self, hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
          memory:        (B, MEMORY_SLOTS, H)
          mask:          (B, MEMORY_SLOTS)
          allocation:    (B, N_BLOCKS) integer slots per block (detached for logging)
        """
        B, L, H = hidden.shape

        # Per-block mean pooling
        block_summaries = []
        for bl in range(N_BLOCKS):
            start = bl * BLOCK_SIZE
            end   = start + BLOCK_SIZE
            blk   = hidden[:, start:end, :]              # (B, BLOCK_SIZE, H)
            pooled = self.block_pool(blk.mean(dim=1))    # (B, H)
            block_summaries.append(pooled)

        block_feats = torch.cat(block_summaries, dim=-1)  # (B, N_BLOCKS * H)
        alloc_logits = self.alloc_net(block_feats)         # (B, N_BLOCKS)

        # Soft allocation via softmax, scaled to MEMORY_SLOTS
        soft_alloc = F.softmax(alloc_logits, dim=-1) * MEMORY_SLOTS  # (B, N_BLOCKS)

        # Discretize: round to integers, ensure sum == MEMORY_SLOTS
        int_alloc = self._discretize_allocation(soft_alloc)  # (B, N_BLOCKS) int

        # Token scores within each block
        token_scores = self.token_scorer(hidden).squeeze(-1)  # (B, L)

        # Gather top-k tokens from each block according to int_alloc
        all_mem  = []
        all_mask = []

        for b in range(B):
            selected = []
            for bl in range(N_BLOCKS):
                k_bl  = int_alloc[b, bl].item()
                start = bl * BLOCK_SIZE
                end   = start + BLOCK_SIZE
                block_scores = token_scores[b, start:end]   # (BLOCK_SIZE,)
                k_bl_safe    = max(0, min(int(k_bl), BLOCK_SIZE))
                if k_bl_safe > 0:
                    _, top_local = torch.topk(block_scores, k_bl_safe)
                    top_global   = top_local + start
                    selected.extend(top_global.tolist())

            # Trim or pad to MEMORY_SLOTS
            selected = selected[:MEMORY_SLOTS]
            if len(selected) == 0:
                selected = list(range(MEMORY_SLOTS))
            while len(selected) < MEMORY_SLOTS:
                selected.append(selected[-1])

            idx_t  = torch.tensor(selected, device=hidden.device)
            mem_b  = hidden[b, idx_t, :]
            msk_b  = torch.ones(MEMORY_SLOTS, device=hidden.device)
            all_mem.append(mem_b)
            all_mask.append(msk_b)

        memory = torch.stack(all_mem,  dim=0)
        mask   = torch.stack(all_mask, dim=0)
        return memory, mask, int_alloc.detach().float()

    @staticmethod
    def _discretize_allocation(soft_alloc: torch.Tensor) -> torch.Tensor:
        """
        Round soft allocations to integers summing to MEMORY_SLOTS.
        Uses floor + distribute remainder to highest fractional parts.
        """
        B = soft_alloc.size(0)
        floored   = soft_alloc.floor().long()         # (B, N_BLOCKS)
        remainder = MEMORY_SLOTS - floored.sum(dim=1) # (B,)
        fractional = soft_alloc - soft_alloc.floor()  # (B, N_BLOCKS)

        result = floored.clone()
        for b in range(B):
            rem = remainder[b].item()
            if rem > 0:
                _, order = fractional[b].sort(descending=True)
                for i in range(int(rem)):
                    result[b, order[i]] += 1
            elif rem < 0:
                # Reduce from smallest allocs
                _, order = result[b].sort(descending=True)
                for i in range(int(-rem)):
                    result[b, order[-(i + 1)]] = max(0, result[b, order[-(i + 1)]].item() - 1)
        return result.clamp(min=0)


# ── Uniform Write Policy ───────────────────────────────────────────────────────

def uniform_write(hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Write UNIFORM_SLOTS[bl] top-norm tokens from each block.
    Returns (memory, mask, alloc_per_block).
    """
    B, L, H = hidden.shape
    norms   = hidden.norm(dim=-1)   # (B, L)
    all_mem  = []
    all_mask = []
    alloc_per_block = torch.zeros(B, N_BLOCKS)

    for b in range(B):
        selected = []
        for bl in range(N_BLOCKS):
            k_bl  = UNIFORM_SLOTS[bl]
            start = bl * BLOCK_SIZE
            end   = start + BLOCK_SIZE
            block_norms  = norms[b, start:end]
            _, top_local = torch.topk(block_norms, k_bl)
            top_global   = top_local + start
            selected.extend(top_global.tolist())
            alloc_per_block[b, bl] = k_bl

        selected = selected[:MEMORY_SLOTS]
        while len(selected) < MEMORY_SLOTS:
            selected.append(selected[-1])

        idx_t  = torch.tensor(selected, device=hidden.device)
        mem_b  = hidden[b, idx_t, :]
        msk_b  = torch.ones(MEMORY_SLOTS, device=hidden.device)
        all_mem.append(mem_b)
        all_mask.append(msk_b)

    memory = torch.stack(all_mem,  dim=0)
    mask   = torch.stack(all_mask, dim=0)
    return memory, mask, alloc_per_block


# ── Gini Coefficient ───────────────────────────────────────────────────────────

def gini_coefficient(values: list[float]) -> float:
    """Compute Gini coefficient of a list of non-negative values."""
    n = len(values)
    if n == 0:
        return 0.0
    sorted_v = sorted(values)
    cumsum   = 0.0
    gini_num = 0.0
    for i, v in enumerate(sorted_v):
        cumsum   += v
        gini_num += (2 * (i + 1) - n - 1) * v
    total = sum(sorted_v)
    if total == 0:
        return 0.0
    return gini_num / (n * total)


# ── Training ───────────────────────────────────────────────────────────────────

def train_uniform() -> tuple[nn.Module, nn.Module]:
    enc       = Encoder().to(DEVICE)
    read_head = ReadHead().to(DEVICE)
    opt       = Adam(list(enc.parameters()) + list(read_head.parameters()), lr=LR)

    for step in range(TRAIN_STEPS):
        seq, target, _ = make_assoc_batch(BATCH_SIZE)
        seq    = seq.to(DEVICE)
        target = target.to(DEVICE)

        hidden = enc(seq)
        memory, mask, _ = uniform_write(hidden)

        query_h   = hidden[:, -1, :]
        logits    = read_head(query_h, memory, mask)
        task_loss = F.cross_entropy(logits, target)

        opt.zero_grad()
        task_loss.backward()
        opt.step()

        if step % LOG_EVERY == 0:
            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                acc   = (preds == target).float().mean().item()
                print(f"  [uniform  ] step={step:4d}  loss={task_loss.item():.3f}  acc={acc:.3f}")

    return enc, read_head


def train_adaptive() -> tuple[nn.Module, nn.Module, BlockAllocator]:
    enc       = Encoder().to(DEVICE)
    allocator = BlockAllocator().to(DEVICE)
    read_head = ReadHead().to(DEVICE)
    opt       = Adam(
        list(enc.parameters()) + list(allocator.parameters()) + list(read_head.parameters()),
        lr=LR,
    )

    for step in range(TRAIN_STEPS):
        seq, target, _ = make_assoc_batch(BATCH_SIZE)
        seq    = seq.to(DEVICE)
        target = target.to(DEVICE)

        hidden = enc(seq)
        memory, mask, _ = allocator(hidden)

        query_h   = hidden[:, -1, :]
        logits    = read_head(query_h, memory, mask)
        task_loss = F.cross_entropy(logits, target)

        opt.zero_grad()
        task_loss.backward()
        opt.step()

        if step % LOG_EVERY == 0:
            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                acc   = (preds == target).float().mean().item()
                print(f"  [adaptive ] step={step:4d}  loss={task_loss.item():.3f}  acc={acc:.3f}")

    return enc, read_head, allocator


def eval_policy(enc, read_head, policy: str, allocator=None) -> tuple[float, list[float]]:
    """Returns (accuracy, mean_alloc_per_block)."""
    total_acc   = 0.0
    block_alloc = [0.0] * N_BLOCKS
    eval_steps  = 20

    with torch.no_grad():
        for _ in range(eval_steps):
            seq, target, _ = make_assoc_batch(BATCH_SIZE)
            seq    = seq.to(DEVICE)
            target = target.to(DEVICE)
            hidden = enc(seq)

            if policy == "uniform":
                memory, mask, alloc = uniform_write(hidden)
            else:
                memory, mask, alloc = allocator(hidden)

            query_h = hidden[:, -1, :]
            logits  = read_head(query_h, memory, mask)
            preds   = logits.argmax(dim=-1)
            total_acc += (preds == target).float().mean().item()

            mean_alloc = alloc.mean(dim=0).tolist()  # (N_BLOCKS,)
            for bl in range(N_BLOCKS):
                block_alloc[bl] += mean_alloc[bl]

    final_acc   = total_acc / eval_steps
    final_alloc = [block_alloc[bl] / eval_steps for bl in range(N_BLOCKS)]
    return final_acc, final_alloc


# ── Experiment ─────────────────────────────────────────────────────────────────

class Exp37WriteBudgetAllocation(Experiment):
    experiment_id = "exp_3_7"
    hypothesis = (
        "A controller given a fixed write budget learns to allocate it non-uniformly "
        "in a way that improves performance over uniform allocation."
    )

    def run(self) -> ExperimentResult:
        torch.manual_seed(42)

        print("\nTraining uniform allocation policy...")
        enc_u, head_u = train_uniform()

        print("\nTraining adaptive allocation policy...")
        enc_a, head_a, allocator = train_adaptive()

        print("\nEvaluating...")
        uniform_acc,  uniform_alloc  = eval_policy(enc_u, head_u, "uniform")
        adaptive_acc, adaptive_alloc = eval_policy(enc_a, head_a, "adaptive", allocator=allocator)

        alloc_gini = gini_coefficient(adaptive_alloc)

        print(f"\nUniform:  acc={uniform_acc:.3f}  alloc={[round(x,2) for x in uniform_alloc]}")
        print(f"Adaptive: acc={adaptive_acc:.3f}  alloc={[round(x,2) for x in adaptive_alloc]}")
        print(f"Adaptive allocation Gini: {alloc_gini:.3f}")

        acc_delta = adaptive_acc - uniform_acc
        is_nonuniform = alloc_gini > 0.3

        if adaptive_acc > uniform_acc and is_nonuniform:
            outcome = OUTCOME_SUPPORTED
        elif not is_nonuniform or adaptive_acc <= uniform_acc:
            outcome = OUTCOME_REFUTED
        else:
            outcome = OUTCOME_INCONCLUSIVE

        metrics = {
            "uniform_acc":           round(uniform_acc, 4),
            "adaptive_acc":          round(adaptive_acc, 4),
            "acc_delta":             round(acc_delta, 4),
            "allocation_gini":       round(alloc_gini, 4),
            "allocation_per_block":  [round(x, 4) for x in adaptive_alloc],
            "uniform_alloc_per_block": [round(x, 4) for x in uniform_alloc],
        }
        notes = (
            f"Adaptive vs uniform accuracy delta: {acc_delta:+.3f}. "
            f"Allocation Gini coefficient: {alloc_gini:.3f} (threshold 0.3). "
            f"Adaptive per-block: {[round(x,2) for x in adaptive_alloc]}. "
            f"Uniform per-block: {[round(x,2) for x in uniform_alloc]} (fixed: {UNIFORM_SLOTS})."
        )
        config = {
            "vocab_size":    VOCAB_SIZE,
            "seq_len":       SEQ_LEN,
            "hidden_dim":    HIDDEN_DIM,
            "memory_slots":  MEMORY_SLOTS,
            "batch_size":    BATCH_SIZE,
            "train_steps":   TRAIN_STEPS,
            "n_blocks":      N_BLOCKS,
            "block_size":    BLOCK_SIZE,
            "uniform_slots": UNIFORM_SLOTS,
            "num_pairs":     NUM_PAIRS,
        }
        return self.result(outcome, metrics, notes, config)


if __name__ == "__main__":
    Exp37WriteBudgetAllocation().execute()
