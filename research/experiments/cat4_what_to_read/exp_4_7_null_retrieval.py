"""
Experiment 4.7 — Null Retrieval Learning

Hypothesis: A learned read gate can be trained to return null (no retrieval)
on tasks where most queries have no relevant memory content, without explicit
null supervision.

Setup:
  - Memory contains a mix of relevant and irrelevant content
  - 80% of queries have no matching memory entry (should return null)
  - 20% of queries have a matching entry (should retrieve it)
  - Read gate must learn this distribution from task loss alone
  - Measure: precision/recall of null vs. non-null decisions
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

VOCAB_SIZE     = 128
HIDDEN_DIM     = 64
MEMORY_SIZE    = 16        # number of entries stored
ENTRY_DIM      = 64
P_RELEVANT     = 0.20      # fraction of queries that have a memory match
TRAIN_STEPS    = 2000
EVAL_BATCHES   = 500
BATCH_SIZE     = 64
LR             = 3e-4
DEVICE         = "cpu"
NULL_PRECISION_THRESHOLD = 0.70   # gate should correctly suppress 70%+ of null queries


# ── Model ─────────────────────────────────────────────────────────────────────

class ReadGate(nn.Module):
    def __init__(self):
        super().__init__()
        # takes (query, max_similarity) → scalar gate decision
        self.net = nn.Sequential(
            nn.Linear(HIDDEN_DIM + 1, 32), nn.ReLU(),
            nn.Linear(32, 1), nn.Sigmoid(),
        )

    def forward(self, query: torch.Tensor, max_sim: torch.Tensor) -> torch.Tensor:
        """Returns gate probability (B,)."""
        x = torch.cat([query, max_sim.unsqueeze(-1)], dim=-1)
        return self.net(x).squeeze(-1)


class MemoryRetriever(nn.Module):
    def __init__(self):
        super().__init__()
        self.query_proj  = nn.Linear(HIDDEN_DIM, ENTRY_DIM)
        self.output_proj = nn.Linear(ENTRY_DIM, HIDDEN_DIM)
        self.read_gate   = ReadGate()
        self.null_vec    = nn.Parameter(torch.zeros(HIDDEN_DIM))  # learned null
        self.classifier  = nn.Linear(HIDDEN_DIM, VOCAB_SIZE)

    def forward(
        self, query: torch.Tensor, memory: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        query:  (B, H)
        memory: (B, M, D)  M entries each dim D
        Returns logits (B, V), gate_prob (B,)
        """
        q_proj = self.query_proj(query)              # (B, D)
        sims   = torch.einsum('bd,bmd->bm', q_proj, memory)  # (B, M)
        sims   = sims / (ENTRY_DIM ** 0.5)
        max_sim = sims.max(dim=-1).values             # (B,)

        gate_prob = self.read_gate(query, max_sim)    # (B,)

        # retrieve best match
        weights   = F.softmax(sims, dim=-1).unsqueeze(-1)   # (B, M, 1)
        retrieved = (weights * memory).sum(1)                # (B, D)
        retrieved_h = self.output_proj(retrieved)            # (B, H)

        # gate between retrieved and null
        null = self.null_vec.unsqueeze(0).expand(query.shape[0], -1)
        out  = gate_prob.unsqueeze(-1) * retrieved_h + (1 - gate_prob).unsqueeze(-1) * null
        return self.classifier(out), gate_prob


def make_batch(batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (query_h, memory, target_label, has_match)."""
    # random query embeddings
    query_h = torch.randn(batch_size, HIDDEN_DIM)

    # for each item, MEMORY_SIZE random entries
    memory  = torch.randn(batch_size, MEMORY_SIZE, ENTRY_DIM)

    target  = torch.randint(0, VOCAB_SIZE, (batch_size,))
    has_match = torch.rand(batch_size) < P_RELEVANT

    for b in range(batch_size):
        if has_match[b]:
            # plant a matching entry: memory[0] should respond to query
            memory[b, 0] = query_h[b, :ENTRY_DIM] + torch.randn(ENTRY_DIM) * 0.1

    return query_h, memory, target, has_match


# ── Experiment ────────────────────────────────────────────────────────────────

class Exp47NullRetrievalLearning(Experiment):
    experiment_id = "exp_4_7"
    hypothesis = (
        "A learned read gate can be trained to return null on tasks where most "
        "queries have no relevant memory content, without explicit null supervision."
    )

    def run(self) -> ExperimentResult:
        torch.manual_seed(42)
        model = MemoryRetriever().to(DEVICE)
        opt   = Adam(model.parameters(), lr=LR)

        for step in range(TRAIN_STEPS):
            q, mem, tgt, _ = make_batch(BATCH_SIZE)
            logits, gate_prob = model(q, mem)
            loss = F.cross_entropy(logits, tgt)
            opt.zero_grad(); loss.backward(); opt.step()

        # evaluate
        model.eval()
        tp_null = fp_null = fn_null = tn_null = 0
        with torch.no_grad():
            for _ in range(EVAL_BATCHES):
                q, mem, tgt, has_match = make_batch(BATCH_SIZE)
                _, gate_prob = model(q, mem)
                gate_fires = gate_prob > 0.5  # True = read; False = null

                for b in range(BATCH_SIZE):
                    if has_match[b]:
                        if gate_fires[b]:  tn_null  += 1  # correctly retrieved
                        else:               fn_null += 1  # missed retrieval
                    else:
                        if not gate_fires[b]: tp_null += 1  # correctly suppressed
                        else:                 fp_null += 1  # unnecessary retrieval

        null_precision = tp_null / max(tp_null + fp_null, 1)
        null_recall    = tp_null / max(tp_null + fn_null, 1)
        null_f1        = (2 * null_precision * null_recall /
                          max(null_precision + null_recall, 1e-8))
        ret_rate_on_matches = tn_null / max(tn_null + fn_null, 1)

        print(f"\n  Null precision: {null_precision:.3f}")
        print(f"  Null recall:    {null_recall:.3f}")
        print(f"  Null F1:        {null_f1:.3f}")
        print(f"  Retrieval rate on relevant queries: {ret_rate_on_matches:.3f}")

        if null_precision >= NULL_PRECISION_THRESHOLD:
            outcome = OUTCOME_SUPPORTED
        elif null_precision >= 0.5:
            outcome = OUTCOME_INCONCLUSIVE
        else:
            outcome = OUTCOME_REFUTED

        return self.result(outcome, {
            "null_precision": null_precision,
            "null_recall": null_recall,
            "null_f1": null_f1,
            "retrieval_rate_on_matches": ret_rate_on_matches,
            "p_relevant": P_RELEVANT,
        }, notes=f"Null precision {null_precision:.3f} vs threshold {NULL_PRECISION_THRESHOLD}.",
        config={"train_steps": TRAIN_STEPS, "p_relevant": P_RELEVANT})


if __name__ == "__main__":
    Exp47NullRetrievalLearning().execute()
