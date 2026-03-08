"""
Experiment 4.9 — Compositional Retrieval

Hypothesis: A learned retrieval mechanism can be trained to retrieve two separate
memory entries and compose them to answer questions neither entry answers alone.

This tests a capability transformers with fixed context genuinely struggle with:
combining non-adjacent stored facts into a novel answer.

Setup:
  - Memory contains (entity, attribute) pairs: ("Alice", "age=32"), ("Bob", "city=Paris")
  - Some queries require composing two facts: "What city does Alice's colleague live in?"
    (requires: Alice→colleague=Bob, Bob→city=Paris)
  - Single-hop baseline: queries answerable from one memory entry
  - Multi-hop test: queries requiring two-entry composition
  - Measure: accuracy on both; gap quantifies compositional capability
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import random

from experiments.base import (
    Experiment, ExperimentResult,
    OUTCOME_SUPPORTED, OUTCOME_REFUTED, OUTCOME_INCONCLUSIVE,
)

# ── Config ────────────────────────────────────────────────────────────────────

NUM_ENTITIES   = 32
NUM_ATTRIBUTES = 16
HIDDEN_DIM     = 64
MEMORY_SIZE    = 20
TRAIN_STEPS    = 3000
EVAL_BATCHES   = 300
BATCH_SIZE     = 32
LR             = 3e-4
DEVICE         = "cpu"
COMPOSITIONAL_THRESHOLD = 0.40  # must exceed random (1/NUM_ATTRIBUTES = 0.0625)


# ── Synthetic Knowledge Base ──────────────────────────────────────────────────
# Entities 0..NUM_ENTITIES-1, Attributes 0..NUM_ATTRIBUTES-1
# Facts:    entity_i has attribute_j  (one attribute per entity)
# Colleague relation: entity_i's colleague is entity_j  (circular shift)

def make_kb():
    torch.manual_seed(0)
    entity_attr  = torch.randint(0, NUM_ATTRIBUTES, (NUM_ENTITIES,))
    colleagues   = torch.arange(NUM_ENTITIES).roll(-1)   # entity i → i+1
    return entity_attr, colleagues


# ── Embeddings ────────────────────────────────────────────────────────────────

class KB_Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.entity_emb = nn.Embedding(NUM_ENTITIES, HIDDEN_DIM)
        self.attr_emb   = nn.Embedding(NUM_ATTRIBUTES, HIDDEN_DIM)
        self.fact_proj  = nn.Linear(HIDDEN_DIM * 2, HIDDEN_DIM)

    def encode_fact(self, entity: torch.Tensor, attr: torch.Tensor) -> torch.Tensor:
        return self.fact_proj(torch.cat([
            self.entity_emb(entity), self.attr_emb(attr)], dim=-1))

    def encode_query(self, entity: torch.Tensor) -> torch.Tensor:
        return self.entity_emb(entity)


class CompositionalRetriever(nn.Module):
    def __init__(self):
        super().__init__()
        self.kb_enc   = KB_Encoder()
        self.hop2_net = nn.Sequential(
            nn.Linear(HIDDEN_DIM * 2, HIDDEN_DIM), nn.ReLU())
        self.classifier = nn.Linear(HIDDEN_DIM, NUM_ATTRIBUTES)

    def retrieve(self, query: torch.Tensor,
                 memory: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Soft attention retrieval over memory."""
        sims = torch.einsum('bh,bmh->bm', query, memory) / (HIDDEN_DIM ** 0.5)
        w    = F.softmax(sims, dim=-1)
        return (w.unsqueeze(-1) * memory).sum(1), w

    def forward_single_hop(
        self, query_entities: torch.Tensor, memory: torch.Tensor
    ) -> torch.Tensor:
        q = self.kb_enc.encode_query(query_entities)
        retrieved, _ = self.retrieve(q, memory)
        return self.classifier(retrieved)

    def forward_two_hop(
        self, query_entities: torch.Tensor, memory: torch.Tensor
    ) -> torch.Tensor:
        # hop 1: find colleague entry
        q1 = self.kb_enc.encode_query(query_entities)
        r1, _ = self.retrieve(q1, memory)

        # hop 2: use retrieved as new query
        q2 = self.hop2_net(torch.cat([q1, r1], dim=-1))
        r2, _ = self.retrieve(q2, memory)
        return self.classifier(r2)


def build_memory_batch(batch_size: int, entity_attr, colleagues, kb_enc) -> torch.Tensor:
    """Build a (B, MEMORY_SIZE, H) memory from all entities."""
    B = batch_size
    # encode all entity-attribute facts
    all_entities = torch.arange(NUM_ENTITIES)
    all_attrs    = entity_attr
    all_facts    = kb_enc.encode_fact(all_entities, all_attrs)   # (E, H)

    # pick MEMORY_SIZE facts per batch item (include all for simplicity)
    M = min(MEMORY_SIZE, NUM_ENTITIES)
    all_facts_exp = all_facts[:M].unsqueeze(0).expand(B, -1, -1)
    return all_facts_exp.detach()


# ── Training & Evaluation ─────────────────────────────────────────────────────

class Exp49CompositionalRetrieval(Experiment):
    experiment_id = "exp_4_9"
    hypothesis = (
        "A learned retrieval mechanism can be trained to retrieve two separate "
        "memory entries and compose them to answer questions neither entry alone "
        "can answer."
    )

    def run(self) -> ExperimentResult:
        torch.manual_seed(42)
        entity_attr, colleagues = make_kb()
        model = CompositionalRetriever().to(DEVICE)
        opt   = Adam(model.parameters(), lr=LR)

        for step in range(TRAIN_STEPS):
            B = BATCH_SIZE
            # mixed training: half single-hop, half two-hop
            q_entities   = torch.randint(0, NUM_ENTITIES, (B,))
            targets_1hop = entity_attr[q_entities]
            colleague_ids = colleagues[q_entities]
            targets_2hop  = entity_attr[colleague_ids]

            with torch.no_grad():
                memory = build_memory_batch(B, entity_attr, colleagues,
                                             model.kb_enc)

            logits_1 = model.forward_single_hop(q_entities, memory)
            logits_2 = model.forward_two_hop(q_entities, memory)
            loss = (F.cross_entropy(logits_1, targets_1hop) +
                    F.cross_entropy(logits_2, targets_2hop))
            opt.zero_grad(); loss.backward(); opt.step()

        # evaluate
        model.eval()
        correct_1 = correct_2 = total = 0
        with torch.no_grad():
            for _ in range(EVAL_BATCHES):
                B = BATCH_SIZE
                q_entities   = torch.randint(0, NUM_ENTITIES, (B,))
                targets_1hop = entity_attr[q_entities]
                colleague_ids = colleagues[q_entities]
                targets_2hop  = entity_attr[colleague_ids]

                memory = build_memory_batch(B, entity_attr, colleagues, model.kb_enc)
                correct_1 += (model.forward_single_hop(q_entities, memory)
                               .argmax(-1) == targets_1hop).sum().item()
                correct_2 += (model.forward_two_hop(q_entities, memory)
                               .argmax(-1) == targets_2hop).sum().item()
                total += B

        acc_1hop = correct_1 / total
        acc_2hop = correct_2 / total
        random_baseline = 1.0 / NUM_ATTRIBUTES

        print(f"\n  Single-hop accuracy: {acc_1hop:.3f}")
        print(f"  Two-hop accuracy:    {acc_2hop:.3f}")
        print(f"  Random baseline:     {random_baseline:.3f}")

        if acc_2hop >= COMPOSITIONAL_THRESHOLD:
            outcome = OUTCOME_SUPPORTED
        elif acc_2hop > random_baseline * 2:
            outcome = OUTCOME_INCONCLUSIVE
        else:
            outcome = OUTCOME_REFUTED

        gap = acc_1hop - acc_2hop
        notes = (
            f"Single-hop={acc_1hop:.3f}, Two-hop={acc_2hop:.3f}, "
            f"Gap={gap:.3f}, Random={random_baseline:.3f}."
        )
        return self.result(outcome, {
            "single_hop_accuracy": acc_1hop,
            "two_hop_accuracy": acc_2hop,
            "random_baseline": random_baseline,
            "compositional_gap": gap,
        }, notes, config={"train_steps": TRAIN_STEPS, "threshold": COMPOSITIONAL_THRESHOLD})


if __name__ == "__main__":
    Exp49CompositionalRetrieval().execute()
