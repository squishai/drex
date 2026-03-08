"""
Experiment 4.1 — Query Formulation Quality

Hypothesis: A dedicated query formulation module outperforms direct use of the
current hidden state as a retrieval query.

Setup:
  - Memory of 8 slots (64-dim each)
  - Model A (Direct): use current hidden state h directly as retrieval query
  - Model B (Learned): small 2-layer MLP transforms h into a query vector
  - Both trained on associative recall task
  - Learned query MLP trained jointly with the rest of the model
  - Measure accuracy and final loss
  - SUPPORTED if learned_acc > direct_acc + 0.02
  - REFUTED if direct_acc >= learned_acc + 0.02
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
SEQ_LEN      = 24
BATCH_SIZE   = 32
TRAIN_STEPS  = 1500
MEMORY_SLOTS = 8
LR           = 3e-4
DEVICE       = "cpu"
EVAL_STEPS   = 300


# ── Data ───────────────────────────────────────────────────────────────────────

def make_batch(batch_size: int, embed: nn.Embedding) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """
    Associative recall: memory holds (key_token -> value_token) pairs.
    Query presents a key token, model must predict the value token.
    Returns (query_h, memory, target_token, key_tokens).
    """
    # sample distinct key and value tokens for each slot
    key_tokens   = torch.stack([
        torch.randperm(VOCAB_SIZE)[:MEMORY_SLOTS] for _ in range(batch_size)
    ])  # (B, M)
    value_tokens = torch.stack([
        torch.randperm(VOCAB_SIZE)[:MEMORY_SLOTS] for _ in range(batch_size)
    ])  # (B, M)

    # build memory as concat of key + value embeddings projected to HIDDEN_DIM
    key_emb   = embed(key_tokens)    # (B, M, H)
    value_emb = embed(value_tokens)  # (B, M, H)
    memory    = (key_emb + value_emb) / 2.0  # (B, M, H)  — simple sum encoding

    # pick one slot as the query target
    slot_idx = torch.randint(0, MEMORY_SLOTS, (batch_size,))  # (B,)
    query_keys = key_tokens[torch.arange(batch_size), slot_idx]  # (B,)
    targets    = value_tokens[torch.arange(batch_size), slot_idx]  # (B,)

    # query hidden state = embedding of the query key token
    query_h = embed(query_keys)  # (B, H)
    return query_h.detach(), memory.detach(), targets, key_tokens


# ── Models ─────────────────────────────────────────────────────────────────────

class DirectRetrievalModel(nn.Module):
    """Uses hidden state directly as retrieval query (no transformation)."""

    def __init__(self) -> None:
        super().__init__()
        self.embed      = nn.Embedding(VOCAB_SIZE, HIDDEN_DIM)
        self.out_proj   = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.classifier = nn.Linear(HIDDEN_DIM, VOCAB_SIZE)

    def forward(
        self, query_h: torch.Tensor, memory: torch.Tensor
    ) -> torch.Tensor:
        # query = h directly
        q = query_h  # (B, H)
        # dot-product attention over memory
        sims    = torch.bmm(memory, q.unsqueeze(-1)).squeeze(-1)  # (B, M)
        sims    = sims / (HIDDEN_DIM ** 0.5)
        weights = F.softmax(sims, dim=-1).unsqueeze(-1)           # (B, M, 1)
        retrieved = (weights * memory).sum(1)                      # (B, H)
        out = self.out_proj(retrieved)
        return self.classifier(out)  # (B, V)


class LearnedRetrievalModel(nn.Module):
    """Uses a 2-layer MLP to transform h into a query vector before retrieval."""

    def __init__(self) -> None:
        super().__init__()
        self.embed      = nn.Embedding(VOCAB_SIZE, HIDDEN_DIM)
        self.query_mlp  = nn.Sequential(
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
        )
        self.out_proj   = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.classifier = nn.Linear(HIDDEN_DIM, VOCAB_SIZE)

    def forward(
        self, query_h: torch.Tensor, memory: torch.Tensor
    ) -> torch.Tensor:
        # query = MLP(h)
        q = self.query_mlp(query_h)  # (B, H)
        # dot-product attention over memory
        sims    = torch.bmm(memory, q.unsqueeze(-1)).squeeze(-1)  # (B, M)
        sims    = sims / (HIDDEN_DIM ** 0.5)
        weights = F.softmax(sims, dim=-1).unsqueeze(-1)           # (B, M, 1)
        retrieved = (weights * memory).sum(1)                      # (B, H)
        out = self.out_proj(retrieved)
        return self.classifier(out)  # (B, V)


# ── Training helper ────────────────────────────────────────────────────────────

def train_and_eval(
    model: nn.Module,
    embed: nn.Embedding,
    label: str,
) -> tuple[float, float]:
    """Train model, return (final_loss, accuracy)."""
    opt = Adam(model.parameters(), lr=LR)
    model.train()
    final_loss = 0.0

    for step in range(TRAIN_STEPS):
        query_h, memory, targets, _ = make_batch(BATCH_SIZE, embed)
        logits = model(query_h, memory)
        loss   = F.cross_entropy(logits, targets)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step >= TRAIN_STEPS - 50:
            final_loss += loss.item()
        if (step + 1) % 300 == 0:
            print(f"  [{label}] step {step+1}/{TRAIN_STEPS}  loss={loss.item():.4f}")

    final_loss /= 50

    # evaluate
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for _ in range(EVAL_STEPS):
            query_h, memory, targets, _ = make_batch(BATCH_SIZE, embed)
            logits = model(query_h, memory)
            preds  = logits.argmax(dim=-1)
            correct += (preds == targets).sum().item()
            total   += BATCH_SIZE

    acc = correct / total
    return final_loss, acc


# ── Experiment ─────────────────────────────────────────────────────────────────

class Exp41QueryFormulationQuality(Experiment):
    experiment_id = "exp_4_1"
    hypothesis = (
        "A dedicated query formulation module outperforms direct use of the "
        "current hidden state as a retrieval query."
    )

    def run(self) -> ExperimentResult:
        torch.manual_seed(42)

        # shared embedding for data generation (not part of the compared models)
        data_embed = nn.Embedding(VOCAB_SIZE, HIDDEN_DIM)
        nn.init.normal_(data_embed.weight, std=0.1)

        print("\n  Training Direct model...")
        direct_model = DirectRetrievalModel().to(DEVICE)
        direct_loss, direct_acc = train_and_eval(direct_model, data_embed, "Direct")

        torch.manual_seed(42)
        print("\n  Training Learned model...")
        learned_model = LearnedRetrievalModel().to(DEVICE)
        learned_loss, learned_acc = train_and_eval(learned_model, data_embed, "Learned")

        gap = learned_acc - direct_acc

        print(f"\n  Direct  — acc={direct_acc:.4f}  loss={direct_loss:.4f}")
        print(f"  Learned — acc={learned_acc:.4f}  loss={learned_loss:.4f}")
        print(f"  Gap (learned - direct): {gap:+.4f}")

        if gap > 0.02:
            outcome = OUTCOME_SUPPORTED
        elif direct_acc >= learned_acc + 0.02:
            outcome = OUTCOME_REFUTED
        else:
            outcome = OUTCOME_INCONCLUSIVE

        metrics = {
            "direct_acc":   round(direct_acc, 4),
            "learned_acc":  round(learned_acc, 4),
            "direct_loss":  round(direct_loss, 4),
            "learned_loss": round(learned_loss, 4),
            "gap_learned_minus_direct": round(gap, 4),
        }
        notes = (
            f"Learned acc={learned_acc:.4f} vs Direct acc={direct_acc:.4f}, "
            f"gap={gap:+.4f} (threshold ±0.02)."
        )
        config = {
            "hidden_dim":   HIDDEN_DIM,
            "vocab_size":   VOCAB_SIZE,
            "memory_slots": MEMORY_SLOTS,
            "train_steps":  TRAIN_STEPS,
            "batch_size":   BATCH_SIZE,
        }
        return self.result(outcome, metrics, notes, config)


if __name__ == "__main__":
    Exp41QueryFormulationQuality().execute()
