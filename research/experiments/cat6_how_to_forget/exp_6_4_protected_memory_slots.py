"""
Experiment 6.4 — Protected Memory Slots

Hypothesis: A controller can learn which memories deserve protection (never evict)
without explicit supervision, and performance degrades predictably outside an
optimal protected-set size.

Setup:
  - Memory of 8 slots; K slots are "protected" (cannot be evicted).
  - Test K = 0, 1, 2, 3, 4, 5.
  - Protection assignments are learned (a gate scores entries as protect/evictable).
  - Task: long sequences with critical (query-time needed) and non-critical entries.
  - SUPPORTED if there exists an optimal K (acc peaks at some K < 5, declines at extremes).
  - REFUTED if no protected slots is always best.
  - INCONCLUSIVE if flat.
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

HIDDEN_DIM    = 64
VOCAB_SIZE    = 64
SEQ_LEN       = 24
BATCH_SIZE    = 32
TRAIN_STEPS   = 1500
MEMORY_SLOTS  = 8
K_VALUES      = [0, 1, 2, 3, 4, 5]
CRITICAL_TOK  = 8    # tokens 0..7 are "critical" (planted early, queried later)
EVAL_BATCHES  = 200
LR            = 3e-4
DEVICE        = "cpu"


# ── Data generator ─────────────────────────────────────────────────────────────

def make_batch(batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
      seqs:         (B, SEQ_LEN) tokens
      is_critical:  (B, SEQ_LEN) 1 if token is critical (< CRITICAL_TOK)
      query_tok:    (B,) the critical key to look up
      target:       (B,) value associated with the critical key
    """
    seqs        = torch.zeros(batch_size, SEQ_LEN, dtype=torch.long)
    is_critical = torch.zeros(batch_size, SEQ_LEN, dtype=torch.long)
    query_tok   = torch.zeros(batch_size, dtype=torch.long)
    target      = torch.zeros(batch_size, dtype=torch.long)

    for b in range(batch_size):
        # One critical key-value pair planted in first quarter
        key   = torch.randint(0, CRITICAL_TOK, (1,)).item()
        value = torch.randint(CRITICAL_TOK, VOCAB_SIZE, (1,)).item()
        pos   = torch.randint(0, SEQ_LEN // 4, (1,)).item()

        seqs[b, pos]        = key
        seqs[b, pos + 1]    = value
        is_critical[b, pos] = 1

        # Fill rest with non-critical noise
        for i in range(SEQ_LEN):
            if i in (pos, pos + 1):
                continue
            seqs[b, i] = torch.randint(CRITICAL_TOK, VOCAB_SIZE, (1,)).item()

        query_tok[b] = key
        target[b]    = value

    return seqs, is_critical, query_tok, target


# ── Protection gate ────────────────────────────────────────────────────────────

class ProtectionGate(nn.Module):
    """Scores each memory entry as 'protect' (high score) or 'evictable' (low score)."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(HIDDEN_DIM, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, mem: torch.Tensor) -> torch.Tensor:
        # mem: (slots, H) → scores: (slots,)
        return self.net(mem).squeeze(-1)


# ── Protected memory model ─────────────────────────────────────────────────────

class ProtectedMemModel(nn.Module):
    def __init__(self, K: int):
        super().__init__()
        self.K          = K    # number of protected slots
        self.embed      = nn.Embedding(VOCAB_SIZE + 2, HIDDEN_DIM)
        self.protect_gate = ProtectionGate()
        self.read_head  = nn.Sequential(
            nn.Linear(HIDDEN_DIM * 2, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, VOCAB_SIZE),
        )
        # Track whether protected entries correspond to critical tokens
        self._last_protection_recall: float = 0.0

    def forward(self, seqs: torch.Tensor, query_tok: torch.Tensor,
                is_critical: torch.Tensor | None = None) -> torch.Tensor:
        B = seqs.size(0)
        out = []
        protection_recalls = []

        for b in range(B):
            mem_embs:     list[torch.Tensor] = []
            mem_critical: list[int]          = []   # 1 if this entry came from critical token

            for t in range(SEQ_LEN - 1):
                tok  = seqs[b, t].item()
                crit = 1 if (is_critical is not None and is_critical[b, t].item() == 1) else (
                    1 if tok < CRITICAL_TOK else 0
                )
                emb  = self.embed(seqs[b, t].unsqueeze(0)).squeeze(0)

                if len(mem_embs) >= MEMORY_SLOTS:
                    if self.K == 0:
                        # Standard eviction: pop oldest
                        mem_embs.pop(0)
                        mem_critical.pop(0)
                    else:
                        # Score all entries; protect top-K, evict from the rest
                        mem_tensor = torch.stack(mem_embs)          # (slots, H)
                        scores     = self.protect_gate(mem_tensor)  # (slots,)

                        sorted_idx = scores.argsort(descending=True).tolist()
                        protected  = set(sorted_idx[:self.K])
                        evictable  = [i for i in range(len(mem_embs)) if i not in protected]

                        if evictable:
                            # Evict oldest among evictable
                            evict_idx = evictable[0]
                        else:
                            # All protected, but need to evict — evict lowest scored
                            evict_idx = sorted_idx[-1]

                        mem_embs.pop(evict_idx)
                        mem_critical.pop(evict_idx)

                mem_embs.append(emb)
                mem_critical.append(crit)

            # Compute protection recall (did protected slots contain critical entries?)
            if self.K > 0 and mem_embs:
                mem_tensor = torch.stack(mem_embs)
                scores     = self.protect_gate(mem_tensor)
                sorted_idx = scores.argsort(descending=True).tolist()
                protected  = set(sorted_idx[:self.K])
                critical_in_protected = sum(1 for i in protected if mem_critical[i] == 1)
                total_critical        = sum(mem_critical)
                recall = critical_in_protected / max(total_critical, 1)
                protection_recalls.append(recall)

            mem_summary = torch.stack(mem_embs).mean(0) if mem_embs else torch.zeros(HIDDEN_DIM)
            q_emb  = self.embed(query_tok[b].unsqueeze(0)).squeeze(0)
            logits = self.read_head(torch.cat([q_emb, mem_summary]).unsqueeze(0))
            out.append(logits)

        if protection_recalls:
            self._last_protection_recall = sum(protection_recalls) / len(protection_recalls)

        return torch.cat(out, dim=0)


# ── Evaluation ─────────────────────────────────────────────────────────────────

def evaluate_model(model: ProtectedMemModel, n_batches: int) -> tuple[float, float]:
    model.eval()
    correct = total = 0
    recalls = []
    with torch.no_grad():
        for _ in range(n_batches):
            seqs, is_crit, query_tok, target = make_batch(BATCH_SIZE)
            logits = model(seqs, query_tok, is_critical=is_crit)
            correct += (logits.argmax(-1) == target).sum().item()
            total   += BATCH_SIZE
            recalls.append(model._last_protection_recall)
    model.train()
    avg_recall = sum(recalls) / max(len(recalls), 1)
    return correct / total, avg_recall


# ── Experiment ──────────────────────────────────────────────────────────────────

class Exp64ProtectedMemorySlots(Experiment):
    experiment_id = "exp_6_4"
    hypothesis = (
        "A controller can learn which memories deserve protection (never evict) "
        "without explicit supervision, and performance degrades predictably outside "
        "an optimal protected-set size."
    )

    def run(self) -> ExperimentResult:
        torch.manual_seed(42)

        config = {
            "hidden_dim": HIDDEN_DIM, "vocab_size": VOCAB_SIZE,
            "seq_len": SEQ_LEN, "batch_size": BATCH_SIZE,
            "train_steps": TRAIN_STEPS, "memory_slots": MEMORY_SLOTS,
            "K_values": K_VALUES, "critical_tok_thresh": CRITICAL_TOK,
        }

        acc_at_K: dict[int, float] = {}
        recall_at_K: dict[int, float] = {}

        for K in K_VALUES:
            print(f"  Training K={K} protected slots...")
            model = ProtectedMemModel(K=K).to(DEVICE)
            opt   = Adam(model.parameters(), lr=LR)

            for step in range(TRAIN_STEPS):
                seqs, is_crit, query_tok, target = make_batch(BATCH_SIZE)
                logits = model(seqs, query_tok, is_critical=is_crit)
                loss   = F.cross_entropy(logits, target)
                opt.zero_grad(); loss.backward(); opt.step()
                if (step + 1) % 500 == 0:
                    print(f"    K={K} step {step+1}/{TRAIN_STEPS}  loss={loss.item():.3f}")

            acc, recall = evaluate_model(model, EVAL_BATCHES)
            acc_at_K[K]    = round(acc, 4)
            recall_at_K[K] = round(recall, 4)
            print(f"  K={K}  acc={acc:.3f}  protection_recall={recall:.3f}")

        # Find optimal K
        optimal_K    = max(K_VALUES, key=lambda k: acc_at_K[k])
        optimal_acc  = acc_at_K[optimal_K]
        acc_at_0     = acc_at_K[0]
        acc_at_5     = acc_at_K[5]
        max_acc      = max(acc_at_K.values())

        # Determine if there's a clear peak at some K in 1..4
        interior_peak = any(
            acc_at_K[k] > acc_at_K[0] + 0.01 and acc_at_K[k] > acc_at_K[5] + 0.01
            for k in [1, 2, 3, 4]
        )

        if interior_peak and optimal_K < 5:
            outcome = OUTCOME_SUPPORTED
        elif acc_at_0 >= max_acc - 0.01:
            outcome = OUTCOME_REFUTED
        else:
            outcome = OUTCOME_INCONCLUSIVE

        metrics = {f"acc_at_K_{k}": acc_at_K[k] for k in K_VALUES}
        metrics.update({f"protection_recall_at_K_{k}": recall_at_K[k] for k in K_VALUES})
        metrics["optimal_K"]                   = optimal_K
        metrics["optimal_K_acc"]               = optimal_acc
        metrics["protection_recall_at_optimal_K"] = recall_at_K[optimal_K]

        return self.result(outcome, metrics, notes=(
            f"Optimal K={optimal_K} with acc={optimal_acc:.3f}. "
            f"K=0 acc={acc_at_0:.3f}, K=5 acc={acc_at_5:.3f}. "
            f"Interior peak detected: {interior_peak}."
        ), config=config)


if __name__ == "__main__":
    Exp64ProtectedMemorySlots().execute()
