"""
Experiment 6.1 — Eviction Policy Comparison

Hypothesis: A learned importance-scored eviction policy significantly outperforms
LRU on tasks requiring retention of low-frequency but high-importance information.

Setup:
  - Memory of 8 slots
  - Four eviction policies: LRU, LFU, Random, Learned (MLP scorer)
  - Long sequences (SEQ_LEN=32) with many distractors
  - A few "important" entries (token_id < 10) planted early, rarely repeated
  - Important entries are required at query time
  - SUPPORTED if learned_acc > LRU_acc + 0.03
  - REFUTED if LRU >= learned
  - INCONCLUSIVE if gap < 0.03
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import random
from collections import OrderedDict, defaultdict

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
SEQ_LEN      = 32   # longer than default per spec
BATCH_SIZE   = 32
TRAIN_STEPS  = 1500
MEMORY_SLOTS = 8
IMPORTANT_THRESH = 10   # token_id < 10 → "important"
EVAL_BATCHES = 200
LR           = 3e-4
DEVICE       = "cpu"


# ── Memory implementations ─────────────────────────────────────────────────────

class MemoryLRU:
    """LRU eviction: evict least-recently written."""
    def __init__(self, slots: int):
        self.slots = slots
        self.store: OrderedDict[int, torch.Tensor] = OrderedDict()

    def reset(self):
        self.store = OrderedDict()

    def write(self, key: int, value: torch.Tensor):
        if key in self.store:
            self.store.move_to_end(key)
            self.store[key] = value
            return
        if len(self.store) >= self.slots:
            self.store.popitem(last=False)   # remove oldest
        self.store[key] = value

    def read_all(self) -> torch.Tensor:
        if not self.store:
            return torch.zeros(self.slots, HIDDEN_DIM)
        vals = list(self.store.values())
        padded = vals + [torch.zeros(HIDDEN_DIM)] * (self.slots - len(vals))
        return torch.stack(padded[:self.slots])


class MemoryLFU:
    """LFU eviction: evict least-frequently accessed."""
    def __init__(self, slots: int):
        self.slots = slots
        self.store: dict[int, torch.Tensor] = {}
        self.freq: dict[int, int] = {}
        self.age: dict[int, int] = {}
        self._t = 0

    def reset(self):
        self.store = {}
        self.freq = {}
        self.age = {}
        self._t = 0

    def write(self, key: int, value: torch.Tensor):
        self._t += 1
        if key in self.store:
            self.freq[key] += 1
            self.store[key] = value
            return
        if len(self.store) >= self.slots:
            # evict lowest freq (tie-break: oldest)
            evict_key = min(self.store.keys(), key=lambda k: (self.freq[k], -self.age[k]))
            del self.store[evict_key]
            del self.freq[evict_key]
            del self.age[evict_key]
        self.store[key] = value
        self.freq[key] = 1
        self.age[key] = self._t

    def read_all(self) -> torch.Tensor:
        if not self.store:
            return torch.zeros(self.slots, HIDDEN_DIM)
        vals = list(self.store.values())
        padded = vals + [torch.zeros(HIDDEN_DIM)] * (self.slots - len(vals))
        return torch.stack(padded[:self.slots])


class MemoryRandom:
    """Random eviction."""
    def __init__(self, slots: int):
        self.slots = slots
        self.keys: list[int] = []
        self.store: dict[int, torch.Tensor] = {}

    def reset(self):
        self.keys = []
        self.store = {}

    def write(self, key: int, value: torch.Tensor):
        if key in self.store:
            self.store[key] = value
            return
        if len(self.store) >= self.slots:
            evict = random.choice(self.keys)
            self.keys.remove(evict)
            del self.store[evict]
        self.store[key] = value
        self.keys.append(key)

    def read_all(self) -> torch.Tensor:
        if not self.store:
            return torch.zeros(self.slots, HIDDEN_DIM)
        vals = list(self.store.values())
        padded = vals + [torch.zeros(HIDDEN_DIM)] * (self.slots - len(vals))
        return torch.stack(padded[:self.slots])


# ── Shared model components ────────────────────────────────────────────────────

class Embedder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE + 2, HIDDEN_DIM)

    def forward(self, tok: torch.Tensor) -> torch.Tensor:
        return self.embed(tok)


class LearnedScorer(nn.Module):
    """MLP that scores each memory entry for importance (higher = keep)."""
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


class ReadHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(HIDDEN_DIM * 2, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, VOCAB_SIZE),
        )

    def forward(self, query_emb: torch.Tensor, mem_summary: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([query_emb, mem_summary], dim=-1))


# ── Data generator ─────────────────────────────────────────────────────────────

def make_batch(batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
      seqs:        (B, SEQ_LEN) token ids
      importance:  (B, SEQ_LEN) 1 if important (token < IMPORTANT_THRESH)
      query_tok:   (B,) the query token (always an important token from early seq)
      target:      (B,) the value associated with the queried important token
    """
    seqs       = torch.zeros(batch_size, SEQ_LEN, dtype=torch.long)
    importance = torch.zeros(batch_size, SEQ_LEN, dtype=torch.long)
    query_tok  = torch.zeros(batch_size, dtype=torch.long)
    target     = torch.zeros(batch_size, dtype=torch.long)

    for b in range(batch_size):
        # Plant one important key-value pair early in the sequence
        key   = torch.randint(0, IMPORTANT_THRESH, (1,)).item()
        value = torch.randint(IMPORTANT_THRESH, VOCAB_SIZE, (1,)).item()
        pos   = torch.randint(0, SEQ_LEN // 4, (1,)).item()   # early slot

        seqs[b, pos]       = key
        seqs[b, pos + 1]   = value
        importance[b, pos] = 1

        # Fill rest with distractor tokens (token_id >= IMPORTANT_THRESH)
        for i in range(SEQ_LEN):
            if i == pos or i == pos + 1:
                continue
            seqs[b, i] = torch.randint(IMPORTANT_THRESH, VOCAB_SIZE, (1,)).item()

        query_tok[b] = key
        target[b]    = value

    return seqs, importance, query_tok, target


# ── Policy A/B/C runner (non-differentiable memory) ───────────────────────────

def run_policy_eval(policy: str, embedder: Embedder, read_head: ReadHead,
                    scorer: LearnedScorer | None, n_batches: int) -> float:
    """Evaluate a fixed-memory policy. Returns accuracy."""
    embedder.eval(); read_head.eval()
    if scorer is not None:
        scorer.eval()

    correct = total = 0
    with torch.no_grad():
        for _ in range(n_batches):
            seqs, importance, query_tok, target = make_batch(BATCH_SIZE)
            # Process each item in batch independently through memory
            preds = []
            for b in range(BATCH_SIZE):
                if policy == "lru":
                    mem_obj = MemoryLRU(MEMORY_SLOTS)
                elif policy == "lfu":
                    mem_obj = MemoryLFU(MEMORY_SLOTS)
                elif policy == "random":
                    mem_obj = MemoryRandom(MEMORY_SLOTS)
                else:
                    mem_obj = None   # learned handled separately

                mem_slots: list[torch.Tensor] = []
                mem_keys:  list[int]          = []

                for t in range(SEQ_LEN - 1):
                    tok = seqs[b, t].item()
                    emb = embedder(seqs[b, t].unsqueeze(0)).squeeze(0)   # (H,)

                    if policy == "learned":
                        # Evict lowest-scored entry
                        if len(mem_slots) >= MEMORY_SLOTS:
                            mem_tensor = torch.stack(mem_slots)          # (slots, H)
                            scores = scorer(mem_tensor).squeeze(-1)      # (slots,)
                            evict_idx = scores.argmin().item()
                            mem_slots.pop(evict_idx)
                            mem_keys.pop(evict_idx)
                        mem_slots.append(emb)
                        mem_keys.append(tok)
                    else:
                        mem_obj.write(tok, emb)

                # Build memory summary
                if policy == "learned":
                    if mem_slots:
                        mem_mat = torch.stack(mem_slots)
                        mem_summary = mem_mat.mean(0)
                    else:
                        mem_summary = torch.zeros(HIDDEN_DIM)
                else:
                    mem_mat = mem_obj.read_all()
                    mem_summary = mem_mat.mean(0)

                q_emb  = embedder(query_tok[b].unsqueeze(0)).squeeze(0)
                logits = read_head(q_emb.unsqueeze(0), mem_summary.unsqueeze(0))
                preds.append(logits.argmax(-1).item())

            correct += sum(p == t.item() for p, t in zip(preds, target))
            total   += BATCH_SIZE

    embedder.train(); read_head.train()
    if scorer is not None:
        scorer.train()
    return correct / total


# ── Differentiable training wrapper for learned policy ─────────────────────────

class LearnedEvictionModel(nn.Module):
    """
    Processes a sequence, uses learned eviction to fill 8 memory slots,
    then answers a query.
    """
    def __init__(self):
        super().__init__()
        self.embedder = Embedder()
        self.scorer   = LearnedScorer()
        self.read_head = ReadHead()

    def forward(self, seqs: torch.Tensor, query_tok: torch.Tensor) -> torch.Tensor:
        B = seqs.size(0)
        preds = []

        # Process each sequence; use straight-through for eviction
        for b in range(B):
            mem_slots: list[torch.Tensor] = []
            for t in range(SEQ_LEN - 1):
                emb = self.embedder(seqs[b, t].unsqueeze(0)).squeeze(0)  # (H,)
                if len(mem_slots) >= MEMORY_SLOTS:
                    mem_tensor = torch.stack(mem_slots)
                    scores = self.scorer(mem_tensor).squeeze(-1)
                    evict_idx = scores.argmin().item()
                    mem_slots.pop(int(evict_idx))
                mem_slots.append(emb)

            if mem_slots:
                mem_mat = torch.stack(mem_slots)
                mem_summary = mem_mat.mean(0)
            else:
                mem_summary = torch.zeros(HIDDEN_DIM)

            q_emb  = self.embedder(query_tok[b].unsqueeze(0)).squeeze(0)
            logit  = self.read_head(q_emb.unsqueeze(0), mem_summary.unsqueeze(0))
            preds.append(logit)

        return torch.cat(preds, dim=0)


class SimpleMemModel(nn.Module):
    """
    Soft-attention baseline for non-learned policies (shared embedder + read head).
    Used to pre-train embedder/read_head weights.
    """
    def __init__(self):
        super().__init__()
        self.embedder  = Embedder()
        self.read_head = ReadHead()

    def forward(self, seqs: torch.Tensor, query_tok: torch.Tensor) -> torch.Tensor:
        h          = self.embedder(seqs)                     # (B, L, H)
        mem_summary = h.mean(1)                              # (B, H)
        q_emb      = self.embedder(query_tok)                # (B, H)
        return self.read_head(q_emb, mem_summary)


# ── Experiment ──────────────────────────────────────────────────────────────────

class Exp61EvictionPolicyComparison(Experiment):
    experiment_id = "exp_6_1"
    hypothesis = (
        "A learned importance-scored eviction policy significantly outperforms "
        "LRU on tasks requiring retention of low-frequency but high-importance information."
    )

    def run(self) -> ExperimentResult:
        torch.manual_seed(42)
        random.seed(42)

        config = {
            "hidden_dim": HIDDEN_DIM, "vocab_size": VOCAB_SIZE,
            "seq_len": SEQ_LEN, "batch_size": BATCH_SIZE,
            "train_steps": TRAIN_STEPS, "memory_slots": MEMORY_SLOTS,
            "important_thresh": IMPORTANT_THRESH,
        }

        # ── Train learned eviction model ──────────────────────────────────────
        print("  Training learned eviction model...")
        learned_model = LearnedEvictionModel().to(DEVICE)
        opt_l = Adam(learned_model.parameters(), lr=LR)

        for step in range(TRAIN_STEPS):
            seqs, _, query_tok, target = make_batch(BATCH_SIZE)
            logits = learned_model(seqs, query_tok)
            loss   = F.cross_entropy(logits, target)
            opt_l.zero_grad(); loss.backward(); opt_l.step()
            if (step + 1) % 500 == 0:
                print(f"    step {step+1}/{TRAIN_STEPS}  loss={loss.item():.3f}")

        # ── Train shared model for non-learned policies ───────────────────────
        print("  Training shared embedder/read_head for non-learned policies...")
        shared_model = SimpleMemModel().to(DEVICE)
        opt_s = Adam(shared_model.parameters(), lr=LR)

        for step in range(TRAIN_STEPS):
            seqs, _, query_tok, target = make_batch(BATCH_SIZE)
            logits = shared_model(seqs, query_tok)
            loss   = F.cross_entropy(logits, target)
            opt_s.zero_grad(); loss.backward(); opt_s.step()
            if (step + 1) % 500 == 0:
                print(f"    step {step+1}/{TRAIN_STEPS}  loss={loss.item():.3f}")

        # ── Evaluate all four policies ────────────────────────────────────────
        print("  Evaluating policies...")
        embedder   = shared_model.embedder
        read_head  = shared_model.read_head
        scorer     = learned_model.scorer

        lru_acc    = run_policy_eval("lru",    embedder, read_head, None,   EVAL_BATCHES)
        lfu_acc    = run_policy_eval("lfu",    embedder, read_head, None,   EVAL_BATCHES)
        random_acc = run_policy_eval("random", embedder, read_head, None,   EVAL_BATCHES)

        # For learned, use the learned model's own embedder/scorer/read_head
        l_embed    = learned_model.embedder
        l_rh       = learned_model.read_head
        learned_acc = run_policy_eval("learned", l_embed, l_rh, scorer, EVAL_BATCHES)

        print(f"  LRU acc:    {lru_acc:.3f}")
        print(f"  LFU acc:    {lfu_acc:.3f}")
        print(f"  Random acc: {random_acc:.3f}")
        print(f"  Learned acc:{learned_acc:.3f}")

        ranking = sorted(
            [("lru", lru_acc), ("lfu", lfu_acc), ("random", random_acc), ("learned", learned_acc)],
            key=lambda x: -x[1]
        )
        ranking_str = " > ".join(f"{k}({v:.3f})" for k, v in ranking)

        gap = learned_acc - lru_acc
        if learned_acc > lru_acc + 0.03:
            outcome = OUTCOME_SUPPORTED
        elif lru_acc >= learned_acc:
            outcome = OUTCOME_REFUTED
        else:
            outcome = OUTCOME_INCONCLUSIVE

        return self.result(outcome, {
            "lru_acc":    round(lru_acc,    4),
            "lfu_acc":    round(lfu_acc,    4),
            "random_acc": round(random_acc, 4),
            "learned_acc": round(learned_acc, 4),
            "learned_vs_lru_gap": round(gap, 4),
            "eviction_policy_ranking": ranking_str,
        }, notes=(
            f"Learned vs LRU gap: {gap:.3f}. "
            f"Threshold for SUPPORTED: >0.03. "
            f"Ranking: {ranking_str}."
        ), config=config)


if __name__ == "__main__":
    Exp61EvictionPolicyComparison().execute()
