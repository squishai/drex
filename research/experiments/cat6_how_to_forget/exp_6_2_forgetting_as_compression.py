"""
Experiment 6.2 — Forgetting as Compression

Hypothesis: Graceful degradation via iterative compression outperforms hard eviction
for long-context tasks where storage budget is the binding constraint.

Setup:
  - Long sequences (SEQ_LEN=48). Fixed storage: 4 × 64-dim = 256 floats.
  - Policy A (hard eviction / LRU): always store at full 64-dim, evict oldest.
  - Policy B (compression forgetting): when memory full, compress oldest entry
    from 64-dim → 32-dim, then 32-dim → 16-dim if still needed, before deleting.
  - Task: QA requiring info from early and late in sequence.
  - SUPPORTED if compression_forgetting_acc > lru_eviction_acc
  - REFUTED if LRU wins
  - INCONCLUSIVE if |gap| < 0.02
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

HIDDEN_DIM   = 64
VOCAB_SIZE   = 64
SEQ_LEN      = 48   # long sequences per spec
BATCH_SIZE   = 32
TRAIN_STEPS  = 1500
MEMORY_SLOTS = 4    # hard constraint: 4 × 64-dim = 256 floats
DIM_FULL     = 64
DIM_HALF     = 32
DIM_QUARTER  = 16
EVAL_BATCHES = 200
LR           = 3e-4
DEVICE       = "cpu"


# ── Compressed memory entry ────────────────────────────────────────────────────

class MemEntry:
    """A memory entry that tracks its compression level."""
    LEVELS = [DIM_FULL, DIM_HALF, DIM_QUARTER]

    def __init__(self, vec: torch.Tensor, level: int = 0):
        self.vec   = vec
        self.level = level   # 0 = full, 1 = half, 2 = quarter

    @property
    def dim(self) -> int:
        return self.LEVELS[self.level]

    def can_compress(self) -> bool:
        return self.level < len(self.LEVELS) - 1

    def compressed_dim(self) -> int:
        return self.LEVELS[self.level + 1]


# ── Compression projectors ─────────────────────────────────────────────────────

class Compressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.full_to_half    = nn.Linear(DIM_FULL,    DIM_HALF,    bias=False)
        self.half_to_quarter = nn.Linear(DIM_HALF,    DIM_QUARTER, bias=False)
        # Upproject compressed entries to full dim for the read head
        self.quarter_to_full = nn.Linear(DIM_QUARTER, DIM_FULL,    bias=False)
        self.half_to_full    = nn.Linear(DIM_HALF,    DIM_FULL,    bias=False)

    def compress(self, entry: MemEntry) -> MemEntry:
        with torch.no_grad():
            if entry.level == 0:
                new_vec = self.full_to_half(entry.vec)
                return MemEntry(new_vec, level=1)
            elif entry.level == 1:
                new_vec = self.half_to_quarter(entry.vec)
                return MemEntry(new_vec, level=2)
        return entry

    def to_full(self, entry: MemEntry) -> torch.Tensor:
        if entry.level == 0:
            return entry.vec
        elif entry.level == 1:
            return self.half_to_full(entry.vec)
        else:
            return self.quarter_to_full(entry.vec)


# ── Model definitions ──────────────────────────────────────────────────────────

class BaseModel(nn.Module):
    """Shared embedding and read head used by both policies."""
    def __init__(self):
        super().__init__()
        self.embed     = nn.Embedding(VOCAB_SIZE + 2, HIDDEN_DIM)
        self.read_head = nn.Sequential(
            nn.Linear(HIDDEN_DIM * 2, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, VOCAB_SIZE),
        )

    def encode(self, tok: torch.Tensor) -> torch.Tensor:
        return self.embed(tok)

    def query(self, q_emb: torch.Tensor, mem_summary: torch.Tensor) -> torch.Tensor:
        return self.read_head(torch.cat([q_emb, mem_summary], dim=-1))


class LRUModel(nn.Module):
    """Hard eviction LRU: keeps at most MEMORY_SLOTS full-dim entries."""
    def __init__(self, base: BaseModel):
        super().__init__()
        self.base = base

    def forward(self, seqs: torch.Tensor, query_tok: torch.Tensor) -> torch.Tensor:
        B = seqs.size(0)
        out_logits = []
        for b in range(B):
            mem: list[torch.Tensor] = []
            for t in range(SEQ_LEN - 1):
                emb = self.base.encode(seqs[b, t].unsqueeze(0)).squeeze(0)
                if len(mem) >= MEMORY_SLOTS:
                    mem.pop(0)   # evict oldest (LRU)
                mem.append(emb)
            mem_summary = torch.stack(mem).mean(0) if mem else torch.zeros(HIDDEN_DIM)
            q_emb  = self.base.encode(query_tok[b].unsqueeze(0)).squeeze(0)
            logits = self.base.query(q_emb.unsqueeze(0), mem_summary.unsqueeze(0))
            out_logits.append(logits)
        return torch.cat(out_logits, dim=0)


class CompressionModel(nn.Module):
    """
    Compression forgetting: when memory full, compress oldest entry rather than delete.
    Budget = MEMORY_SLOTS × DIM_FULL = 256 floats.
    We track actual float usage. Compress oldest when budget exceeded.
    """
    def __init__(self, base: BaseModel, compressor: Compressor):
        super().__init__()
        self.base       = base
        self.compressor = compressor

    def _budget_used(self, mem: list[MemEntry]) -> int:
        return sum(e.dim for e in mem)

    def _make_room(self, mem: list[MemEntry]) -> list[MemEntry]:
        budget = MEMORY_SLOTS * DIM_FULL
        # Try to compress oldest compressible entry
        for i in range(len(mem)):
            if mem[i].can_compress():
                mem[i] = self.compressor.compress(mem[i])
                if self._budget_used(mem) <= budget:
                    return mem
        # If still over budget, evict oldest
        if mem:
            mem.pop(0)
        return mem

    def forward(self, seqs: torch.Tensor, query_tok: torch.Tensor) -> torch.Tensor:
        B = seqs.size(0)
        out_logits = []
        budget = MEMORY_SLOTS * DIM_FULL

        compression_levels = []

        for b in range(B):
            mem: list[MemEntry] = []
            for t in range(SEQ_LEN - 1):
                emb = self.base.encode(seqs[b, t].unsqueeze(0)).squeeze(0).detach()
                entry = MemEntry(emb.clone(), level=0)
                mem.append(entry)
                while self._budget_used(mem) > budget:
                    mem = self._make_room(mem)

            # Upsample all entries to full dim for reading
            full_vecs = [self.compressor.to_full(e) for e in mem]
            mem_summary = torch.stack(full_vecs).mean(0) if full_vecs else torch.zeros(HIDDEN_DIM)

            avg_level = sum(e.level for e in mem) / max(len(mem), 1)
            compression_levels.append(avg_level)

            q_emb  = self.base.encode(query_tok[b].unsqueeze(0)).squeeze(0)
            logits = self.base.query(q_emb.unsqueeze(0), mem_summary.unsqueeze(0))
            out_logits.append(logits)

        self._last_compression_level = sum(compression_levels) / max(len(compression_levels), 1)
        return torch.cat(out_logits, dim=0)


# ── Data generator ─────────────────────────────────────────────────────────────

def make_batch(batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """QA task: important pair planted early, noise fills middle, target at end."""
    seqs      = torch.zeros(batch_size, SEQ_LEN, dtype=torch.long)
    query_tok = torch.zeros(batch_size, dtype=torch.long)
    target    = torch.zeros(batch_size, dtype=torch.long)

    for b in range(batch_size):
        # Early key-value pair
        key   = torch.randint(0, 32, (1,)).item()
        value = torch.randint(32, VOCAB_SIZE, (1,)).item()
        early_pos = torch.randint(0, SEQ_LEN // 6, (1,)).item()

        seqs[b, early_pos]     = key
        seqs[b, early_pos + 1] = value

        # Fill rest with distractors
        for i in range(SEQ_LEN):
            if i in (early_pos, early_pos + 1):
                continue
            seqs[b, i] = torch.randint(1, VOCAB_SIZE, (1,)).item()

        query_tok[b] = key
        target[b]    = value

    return seqs, query_tok, target


# ── Evaluation ─────────────────────────────────────────────────────────────────

def evaluate(model: nn.Module, n_batches: int) -> tuple[float, float]:
    """Returns (accuracy, avg_compression_level). compression_level = 0 for LRU."""
    model.eval()
    correct = total = 0
    comp_levels = []

    with torch.no_grad():
        for _ in range(n_batches):
            seqs, query_tok, target = make_batch(BATCH_SIZE)
            logits = model(seqs, query_tok)
            correct += (logits.argmax(-1) == target).sum().item()
            total   += BATCH_SIZE
            if hasattr(model, '_last_compression_level'):
                comp_levels.append(model._last_compression_level)

    model.train()
    avg_comp = sum(comp_levels) / max(len(comp_levels), 1)
    return correct / total, avg_comp


# ── Experiment ──────────────────────────────────────────────────────────────────

class Exp62ForgettingAsCompression(Experiment):
    experiment_id = "exp_6_2"
    hypothesis = (
        "Graceful degradation via iterative compression outperforms hard eviction "
        "for long-context tasks where storage budget is the binding constraint."
    )

    def run(self) -> ExperimentResult:
        torch.manual_seed(42)

        config = {
            "hidden_dim": HIDDEN_DIM, "vocab_size": VOCAB_SIZE,
            "seq_len": SEQ_LEN, "batch_size": BATCH_SIZE,
            "train_steps": TRAIN_STEPS, "memory_slots": MEMORY_SLOTS,
            "budget_floats": MEMORY_SLOTS * DIM_FULL,
        }

        # Shared base; compression model gets its own compressor
        base       = BaseModel().to(DEVICE)
        compressor = Compressor().to(DEVICE)

        lru_model   = LRUModel(base).to(DEVICE)
        comp_model  = CompressionModel(base, compressor).to(DEVICE)

        # Train shared base parameters (embedder + read head) + compressor jointly
        opt = Adam(
            list(base.parameters()) + list(compressor.parameters()),
            lr=LR,
        )

        print("  Training base + compressor...")
        for step in range(TRAIN_STEPS):
            seqs, query_tok, target = make_batch(BATCH_SIZE)

            # Alternate between LRU and compression forward passes
            if step % 2 == 0:
                logits = lru_model(seqs, query_tok)
            else:
                logits = comp_model(seqs, query_tok)

            loss = F.cross_entropy(logits, target)
            opt.zero_grad(); loss.backward(); opt.step()

            if (step + 1) % 500 == 0:
                print(f"    step {step+1}/{TRAIN_STEPS}  loss={loss.item():.3f}")

        # ── Evaluate ──────────────────────────────────────────────────────────
        print("  Evaluating LRU policy...")
        lru_acc, _      = evaluate(lru_model, EVAL_BATCHES)
        print("  Evaluating compression forgetting policy...")
        comp_acc, avg_comp_level = evaluate(comp_model, EVAL_BATCHES)

        print(f"  LRU acc:         {lru_acc:.3f}")
        print(f"  Compression acc: {comp_acc:.3f}")
        print(f"  Avg compression level at query: {avg_comp_level:.3f}")

        gap = comp_acc - lru_acc
        if comp_acc > lru_acc:
            outcome = OUTCOME_SUPPORTED
        elif lru_acc > comp_acc:
            outcome = OUTCOME_REFUTED
        else:
            outcome = OUTCOME_INCONCLUSIVE

        if abs(gap) < 0.02:
            outcome = OUTCOME_INCONCLUSIVE

        return self.result(outcome, {
            "lru_acc":    round(lru_acc,    4),
            "compression_acc": round(comp_acc, 4),
            "gap_compression_minus_lru": round(gap, 4),
            "avg_compression_level_at_query_time": round(avg_comp_level, 4),
        }, notes=(
            f"Compression vs LRU gap: {gap:.3f}. "
            f"Average compression level (0=full, 1=half, 2=quarter): {avg_comp_level:.3f}."
        ), config=config)


if __name__ == "__main__":
    Exp62ForgettingAsCompression().execute()
