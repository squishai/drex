"""
Experiment 6.8 — Memory Consolidation

Hypothesis: Periodic offline consolidation (merging memory entries into higher-level
representations) improves long-horizon recall without active context.

Setup:
  - Memory of 8 slots. Every K=16 steps, run a consolidation pass:
    take all 8 entries, run through a 1-layer transformer, output 4 consolidated
    entries that summarise the 8. Replace original 8 with 4 consolidated + 4 free slots.
  - Baseline: no consolidation (standard LRU with 8 slots).
  - Task: long sequence (SEQ_LEN=64) where multiple entries need to be combined.
  - SUPPORTED if consolidation_acc > no_consolidation_acc.
  - REFUTED if no_consolidation wins.
  - INCONCLUSIVE if |gap| < 0.02.
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

HIDDEN_DIM         = 64
VOCAB_SIZE         = 64
SEQ_LEN            = 64    # long sequences per spec
BATCH_SIZE         = 32
TRAIN_STEPS        = 1500
MEMORY_SLOTS       = 8
CONSOLIDATION_K    = 16    # consolidate every K steps
CONSOLIDATED_OUT   = 4     # output 4 consolidated entries from 8
NHEADS             = 4     # transformer heads for consolidation
EVAL_BATCHES       = 200
LR                 = 3e-4
DEVICE             = "cpu"


# ── Data generator ─────────────────────────────────────────────────────────────

def make_batch(batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Long sequences where an early key-value pair must survive until query time.
    Multiple distractor entries are written throughout.
    Returns seqs (B, SEQ_LEN), query_tok (B,), target (B,).
    """
    seqs      = torch.zeros(batch_size, SEQ_LEN, dtype=torch.long)
    query_tok = torch.zeros(batch_size, dtype=torch.long)
    target    = torch.zeros(batch_size, dtype=torch.long)

    for b in range(batch_size):
        key   = torch.randint(0, 16, (1,)).item()
        value = torch.randint(16, VOCAB_SIZE, (1,)).item()
        # Plant early
        pos   = torch.randint(0, SEQ_LEN // 8, (1,)).item()

        seqs[b, pos]     = key
        seqs[b, pos + 1] = value

        for i in range(SEQ_LEN):
            if i in (pos, pos + 1):
                continue
            seqs[b, i] = torch.randint(1, VOCAB_SIZE, (1,)).item()

        query_tok[b] = key
        target[b]    = value

    return seqs, query_tok, target


# ── Consolidation transformer (1-layer, cross-attention to output slots) ───────

class ConsolidationModule(nn.Module):
    """
    Takes 8 memory entries (8, H), outputs 4 consolidated entries (4, H).
    Uses a 1-layer transformer encoder + a learned query matrix for 4 output slots.
    """
    def __init__(self):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=HIDDEN_DIM, nhead=NHEADS,
            dim_feedforward=HIDDEN_DIM * 2,
            batch_first=True, dropout=0.0,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

        # Learned output query embeddings (4 consolidated slots)
        self.output_queries = nn.Parameter(
            torch.randn(CONSOLIDATED_OUT, HIDDEN_DIM) * 0.02
        )

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=HIDDEN_DIM, num_heads=NHEADS,
            batch_first=True, dropout=0.0,
        )

    def forward(self, mem: torch.Tensor) -> torch.Tensor:
        """
        mem: (M, H) where M = MEMORY_SLOTS (or fewer if not full)
        Returns: (CONSOLIDATED_OUT, H)
        """
        if mem.size(0) == 0:
            return torch.zeros(CONSOLIDATED_OUT, HIDDEN_DIM)

        mem_b = mem.unsqueeze(0)                                     # (1, M, H)
        enc   = self.encoder(mem_b)                                  # (1, M, H)

        queries = self.output_queries.unsqueeze(0)                   # (1, 4, H)
        out, _  = self.cross_attn(queries, enc, enc)                 # (1, 4, H)
        return out.squeeze(0)                                        # (4, H)


# ── No-consolidation baseline (LRU, 8 slots) ──────────────────────────────────

class NoConsolidationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed     = nn.Embedding(VOCAB_SIZE + 2, HIDDEN_DIM)
        self.read_head = nn.Sequential(
            nn.Linear(HIDDEN_DIM * 2, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, VOCAB_SIZE),
        )
        self._last_mean_entries: float = float(MEMORY_SLOTS)

    def forward(self, seqs: torch.Tensor, query_tok: torch.Tensor) -> torch.Tensor:
        B = seqs.size(0)
        out = []
        for b in range(B):
            mem: list[torch.Tensor] = []
            for t in range(SEQ_LEN - 1):
                emb = self.embed(seqs[b, t].unsqueeze(0)).squeeze(0)
                if len(mem) >= MEMORY_SLOTS:
                    mem.pop(0)
                mem.append(emb)

            self._last_mean_entries = len(mem)
            mem_summary = torch.stack(mem).mean(0) if mem else torch.zeros(HIDDEN_DIM)
            q_emb  = self.embed(query_tok[b].unsqueeze(0)).squeeze(0)
            logits = self.read_head(torch.cat([q_emb, mem_summary]).unsqueeze(0))
            out.append(logits)
        return torch.cat(out, dim=0)


# ── Consolidation model ────────────────────────────────────────────────────────

class ConsolidationModel(nn.Module):
    """
    Every CONSOLIDATION_K steps, consolidates the current memory:
      8 entries → 4 consolidated entries, freeing 4 slots.
    """
    def __init__(self, consolidator: ConsolidationModule):
        super().__init__()
        self.embed        = nn.Embedding(VOCAB_SIZE + 2, HIDDEN_DIM)
        self.consolidator = consolidator
        self.read_head    = nn.Sequential(
            nn.Linear(HIDDEN_DIM * 2, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, VOCAB_SIZE),
        )
        self._last_mean_entries:              float = float(MEMORY_SLOTS)
        self._last_consolidation_compression: float = 0.0

    def forward(self, seqs: torch.Tensor, query_tok: torch.Tensor) -> torch.Tensor:
        B = seqs.size(0)
        out = []
        entries_at_query: list[int]   = []
        compressions:     list[float] = []

        for b in range(B):
            mem: list[torch.Tensor] = []

            for t in range(SEQ_LEN - 1):
                emb = self.embed(seqs[b, t].unsqueeze(0)).squeeze(0)

                if len(mem) >= MEMORY_SLOTS:
                    mem.pop(0)
                mem.append(emb)

                # Periodic consolidation
                if (t + 1) % CONSOLIDATION_K == 0 and len(mem) >= MEMORY_SLOTS:
                    before = len(mem)
                    mem_tensor   = torch.stack(mem)            # (8, H)
                    consolidated = self.consolidator(mem_tensor)  # (4, H)
                    mem = [consolidated[i] for i in range(CONSOLIDATED_OUT)]
                    after = len(mem)
                    compressions.append(before / after)

            entries_at_query.append(len(mem))
            mem_summary = torch.stack(mem).mean(0) if mem else torch.zeros(HIDDEN_DIM)
            q_emb  = self.embed(query_tok[b].unsqueeze(0)).squeeze(0)
            logits = self.read_head(torch.cat([q_emb, mem_summary]).unsqueeze(0))
            out.append(logits)

        self._last_mean_entries = (
            sum(entries_at_query) / max(len(entries_at_query), 1)
        )
        self._last_consolidation_compression = (
            sum(compressions) / max(len(compressions), 1)
        )

        return torch.cat(out, dim=0)


# ── Evaluation ─────────────────────────────────────────────────────────────────

def evaluate(model: nn.Module, n_batches: int) -> tuple[float, float, float]:
    """Returns (accuracy, mean_entries_at_query, consolidation_compression_ratio)."""
    model.eval()
    correct = total = 0
    entries = []
    compressions = []

    with torch.no_grad():
        for _ in range(n_batches):
            seqs, query_tok, target = make_batch(BATCH_SIZE)
            logits = model(seqs, query_tok)
            correct += (logits.argmax(-1) == target).sum().item()
            total   += BATCH_SIZE
            if hasattr(model, '_last_mean_entries'):
                entries.append(model._last_mean_entries)
            if hasattr(model, '_last_consolidation_compression'):
                compressions.append(model._last_consolidation_compression)

    model.train()
    avg_entries    = sum(entries)      / max(len(entries),      1)
    avg_comp_ratio = sum(compressions) / max(len(compressions), 1)
    return correct / total, avg_entries, avg_comp_ratio


# ── Experiment ──────────────────────────────────────────────────────────────────

class Exp68MemoryConsolidation(Experiment):
    experiment_id = "exp_6_8"
    hypothesis = (
        "Periodic offline consolidation (merging memory entries into higher-level "
        "representations) improves long-horizon recall without active context."
    )

    def run(self) -> ExperimentResult:
        torch.manual_seed(42)

        config = {
            "hidden_dim": HIDDEN_DIM, "vocab_size": VOCAB_SIZE,
            "seq_len": SEQ_LEN, "batch_size": BATCH_SIZE,
            "train_steps": TRAIN_STEPS, "memory_slots": MEMORY_SLOTS,
            "consolidation_K": CONSOLIDATION_K,
            "consolidated_out": CONSOLIDATED_OUT,
        }

        # Shared consolidator
        consolidator = ConsolidationModule().to(DEVICE)

        # ── Train no-consolidation baseline ───────────────────────────────────
        print("  Training no-consolidation baseline...")
        no_cons = NoConsolidationModel().to(DEVICE)
        opt_nc  = Adam(no_cons.parameters(), lr=LR)

        for step in range(TRAIN_STEPS):
            seqs, query_tok, target = make_batch(BATCH_SIZE)
            loss = F.cross_entropy(no_cons(seqs, query_tok), target)
            opt_nc.zero_grad(); loss.backward(); opt_nc.step()
            if (step + 1) % 500 == 0:
                print(f"    No-cons step {step+1}/{TRAIN_STEPS}  loss={loss.item():.3f}")

        # ── Train consolidation model ──────────────────────────────────────────
        print("  Training consolidation model...")
        cons_model = ConsolidationModel(consolidator).to(DEVICE)
        opt_c = Adam(
            list(cons_model.embed.parameters()) +
            list(cons_model.consolidator.parameters()) +
            list(cons_model.read_head.parameters()),
            lr=LR,
        )

        for step in range(TRAIN_STEPS):
            seqs, query_tok, target = make_batch(BATCH_SIZE)
            loss = F.cross_entropy(cons_model(seqs, query_tok), target)
            opt_c.zero_grad(); loss.backward(); opt_c.step()
            if (step + 1) % 500 == 0:
                print(f"    Cons step {step+1}/{TRAIN_STEPS}  loss={loss.item():.3f}")

        # ── Evaluate ──────────────────────────────────────────────────────────
        no_cons_acc, no_cons_entries, _      = evaluate(no_cons,    EVAL_BATCHES)
        cons_acc,    cons_entries, comp_ratio = evaluate(cons_model, EVAL_BATCHES)

        print(f"  No-consolidation acc:    {no_cons_acc:.3f}  entries={no_cons_entries:.1f}")
        print(f"  Consolidation acc:       {cons_acc:.3f}  entries={cons_entries:.1f}")
        print(f"  Consolidation comp ratio:{comp_ratio:.2f}x")

        gap = cons_acc - no_cons_acc
        if abs(gap) < 0.02:
            outcome = OUTCOME_INCONCLUSIVE
        elif cons_acc > no_cons_acc:
            outcome = OUTCOME_SUPPORTED
        else:
            outcome = OUTCOME_REFUTED

        return self.result(outcome, {
            "no_consolidation_acc":         round(no_cons_acc,  4),
            "consolidation_acc":            round(cons_acc,     4),
            "gap_consolidation_minus_none": round(gap,          4),
            "consolidation_compression_ratio": round(comp_ratio, 4),
            "mean_entries_at_query_time_no_consolidation": round(no_cons_entries, 2),
            "mean_entries_at_query_time_consolidation":    round(cons_entries,    2),
        }, notes=(
            f"Consolidation vs no-consolidation gap: {gap:.3f}. "
            f"Consolidation compresses 8→4 entries ({comp_ratio:.2f}x ratio). "
            f"Mean entries at query: no-cons={no_cons_entries:.1f}, cons={cons_entries:.1f}."
        ), config=config)


if __name__ == "__main__":
    Exp68MemoryConsolidation().execute()
