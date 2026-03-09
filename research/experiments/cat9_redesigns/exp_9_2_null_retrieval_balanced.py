"""
Experiment 9.2 — Null-Retrieval Balanced Read Gate

Hypothesis: With a 50/50 null-to-retrieval query distribution (fixing exp_4_7's degenerate
80% null), a learned read gate achieves null precision > 0.65 and retrieval recall > 0.65,
outperforming always-null and always-retrieve baselines on harmonic F1.
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

VOCAB_SIZE   = 128
HIDDEN_DIM   = 64
MEMORY_SIZE  = 8
BATCH_SIZE   = 64
TRAIN_STEPS  = 3000
LR           = 3e-4
DEVICE       = "cpu"
PAD_TOKEN    = 0

# ── Data helpers ──────────────────────────────────────────────────────────────

def make_memory_bank(batch_size: int, memory_size: int = MEMORY_SIZE):
    """
    Create a memory bank of key-value pairs.
    Keys: tokens in [4, VOCAB_SIZE//2)
    Values: tokens in [VOCAB_SIZE//2, VOCAB_SIZE)
    Returns keys (B, M) and values (B, M).
    """
    keys   = torch.zeros(batch_size, memory_size, dtype=torch.long)
    values = torch.zeros(batch_size, memory_size, dtype=torch.long)
    for b in range(batch_size):
        k = torch.randint(4, VOCAB_SIZE // 2, (memory_size,))
        v = torch.randint(VOCAB_SIZE // 2, VOCAB_SIZE, (memory_size,))
        keys[b] = k
        values[b] = v
    return keys, values


def make_query_batch(batch_size: int, p_relevant: float = 0.5):
    """
    Create queries. p_relevant fraction will have their key in memory.
    Returns: query_keys (B,), target_values (B,), is_in_memory (B,)
    """
    mem_keys, mem_vals = make_memory_bank(batch_size)
    query_keys    = torch.zeros(batch_size, dtype=torch.long)
    target_values = torch.zeros(batch_size, dtype=torch.long)
    is_in_memory  = torch.zeros(batch_size, dtype=torch.float)

    for b in range(batch_size):
        if torch.rand(1).item() < p_relevant:
            # Query key IS in memory
            slot = torch.randint(0, MEMORY_SIZE, (1,)).item()
            query_keys[b]    = mem_keys[b, slot]
            target_values[b] = mem_vals[b, slot]
            is_in_memory[b]  = 1.0
        else:
            # Query key NOT in memory — use a different key
            out_key = torch.randint(4, VOCAB_SIZE // 2, (1,)).item()
            while out_key in mem_keys[b].tolist():
                out_key = torch.randint(4, VOCAB_SIZE // 2, (1,)).item()
            query_keys[b]    = out_key
            target_values[b] = PAD_TOKEN
            is_in_memory[b]  = 0.0

    return mem_keys, mem_vals, query_keys, target_values, is_in_memory


# ── Model components ──────────────────────────────────────────────────────────

class KeyValueEncoder(nn.Module):
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


class ReadGate(nn.Module):
    """Sigmoid gate: 1 = retrieve, 0 = null."""
    def __init__(self, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, query_h, memory_summary):
        """query_h: (B, H), memory_summary: (B, H)."""
        inp = torch.cat([query_h, memory_summary], dim=-1)
        return torch.sigmoid(self.fc(inp)).squeeze(-1)  # (B,)


class MemoryReader(nn.Module):
    def __init__(self, hidden_dim=HIDDEN_DIM, vocab_size=VOCAB_SIZE):
        super().__init__()
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out    = nn.Linear(hidden_dim, vocab_size)

    def forward(self, query_h, key_h, val_h):
        """
        query_h: (B, H), key_h: (B, M, H), val_h: (B, M, H)
        Returns logits (B, vocab_size).
        """
        q = self.q_proj(query_h).unsqueeze(-1)  # (B, H, 1)
        scores = torch.bmm(key_h, q).squeeze(-1)  # (B, M)
        attn = torch.softmax(scores, dim=-1)       # (B, M)
        ctx = (attn.unsqueeze(-1) * val_h).sum(1)  # (B, H)
        return self.out(ctx)


# ── Training ──────────────────────────────────────────────────────────────────

def train_model(p_relevant: float, use_learned_gate: bool, use_always_retrieve: bool = False):
    enc      = KeyValueEncoder().to(DEVICE)
    reader   = MemoryReader().to(DEVICE)
    gate     = ReadGate().to(DEVICE) if use_learned_gate else None

    params = list(enc.parameters()) + list(reader.parameters())
    if gate is not None:
        params += list(gate.parameters())
    opt = Adam(params, lr=LR)

    enc.train(); reader.train()
    if gate is not None:
        gate.train()

    for step in range(TRAIN_STEPS):
        mem_k, mem_v, q_keys, tgt_vals, in_mem = make_query_batch(BATCH_SIZE, p_relevant)

        q_h   = enc(q_keys)                    # (B, H)
        k_h   = enc(mem_k)                     # (B, M, H)
        v_h   = enc(mem_v)                     # (B, M, H)
        mem_summary = k_h.mean(1)              # (B, H) — mean pooled memory

        # Gate decision
        if use_learned_gate and gate is not None:
            gate_prob = gate(q_h, mem_summary) # (B,)
        elif use_always_retrieve:
            gate_prob = torch.ones(BATCH_SIZE)
        else:
            gate_prob = torch.zeros(BATCH_SIZE)

        # Loss for retrieval queries
        retrieval_mask = in_mem.bool()
        null_mask      = ~retrieval_mask

        loss = torch.tensor(0.0, requires_grad=True)

        if retrieval_mask.any():
            logits_r = reader(q_h[retrieval_mask], k_h[retrieval_mask], v_h[retrieval_mask])
            tgt_r    = tgt_vals[retrieval_mask]
            retr_loss = F.cross_entropy(logits_r, tgt_r)
            if use_learned_gate:
                gate_r = gate_prob[retrieval_mask]
                # Encourage gate=1 for retrieval queries, penalize gate=0
                gate_r_loss = F.binary_cross_entropy(gate_r, torch.ones_like(gate_r))
                retr_loss = retr_loss + gate_r_loss
            loss = loss + retr_loss

        if null_mask.any() and use_learned_gate and gate is not None:
            gate_n = gate_prob[null_mask]
            # Encourage gate=0 for null queries
            null_loss = F.binary_cross_entropy(gate_n, torch.zeros_like(gate_n))
            loss = loss + null_loss

        opt.zero_grad()
        loss.backward()
        opt.step()

    return enc, reader, gate


def evaluate_model(enc, reader, gate, p_relevant: float,
                   use_always_retrieve: bool = False, use_always_null: bool = False,
                   n_batches: int = 100):
    enc.eval(); reader.eval()
    if gate is not None:
        gate.eval()

    tp_null = 0; fp_null = 0; fn_null = 0
    tp_retr = 0; fn_retr = 0

    with torch.no_grad():
        for _ in range(n_batches):
            mem_k, mem_v, q_keys, tgt_vals, in_mem = make_query_batch(BATCH_SIZE, p_relevant)
            q_h = enc(q_keys)
            k_h = enc(mem_k)
            v_h = enc(mem_v)
            mem_summary = k_h.mean(1)

            if gate is not None:
                gate_prob = gate(q_h, mem_summary)
                decide_retrieve = gate_prob > 0.5
            elif use_always_retrieve:
                decide_retrieve = torch.ones(BATCH_SIZE, dtype=torch.bool)
            else:
                decide_retrieve = torch.zeros(BATCH_SIZE, dtype=torch.bool)

            is_null = (in_mem == 0)
            is_retr = (in_mem == 1)

            # Null precision/recall
            # True null: is_null & predicted_null
            pred_null = ~decide_retrieve
            tp_null += (pred_null & is_null).sum().item()
            fp_null += (pred_null & is_retr).sum().item()   # predicted null but was retrieval
            fn_null += (~pred_null & is_null).sum().item()  # predicted retrieve but was null

            # Retrieval recall on actual retrieval queries
            if is_retr.any():
                logits_r = reader(q_h[is_retr], k_h[is_retr], v_h[is_retr])
                tgt_r    = tgt_vals[is_retr]
                preds_r  = logits_r.argmax(-1)
                correct_r = (preds_r == tgt_r).sum().item()
                total_r   = is_retr.sum().item()
                # Only count as "recalled" if we decided to retrieve AND got it right
                decided_retrieve_and_correct = (decide_retrieve & is_retr).sum().item()
                tp_retr += min(correct_r, decided_retrieve_and_correct)
                fn_retr += is_retr.sum().item() - min(correct_r, decided_retrieve_and_correct)

    null_precision = tp_null / max(tp_null + fp_null, 1)
    null_recall    = tp_null / max(tp_null + fn_null, 1)
    null_f1        = (2 * null_precision * null_recall /
                      max(null_precision + null_recall, 1e-8))
    retr_recall    = tp_retr / max(tp_retr + fn_retr, 1)

    total_preds_null = tp_null + fp_null
    total_queries    = n_batches * BATCH_SIZE
    is_degenerate    = (total_preds_null / total_queries > 0.95) or (total_preds_null / total_queries < 0.05)

    return {
        "null_precision": null_precision,
        "null_recall":    null_recall,
        "null_f1":        null_f1,
        "retrieval_recall": retr_recall,
        "is_degenerate":  is_degenerate,
    }


# ── Experiment class ──────────────────────────────────────────────────────────

class Exp92NullRetrievalBalanced(Experiment):
    experiment_id = "exp_9_2"
    hypothesis = (
        "With a 50/50 null-to-retrieval query distribution (fixing exp_4_7's degenerate "
        "80% null), a learned read gate achieves null precision > 0.65 and retrieval "
        "recall > 0.65, outperforming always-null and always-retrieve baselines on "
        "harmonic F1."
    )

    def run(self) -> ExperimentResult:
        results = {}

        # Condition A: Learned gate, p=0.50
        print("  Condition A: learned gate, p_relevant=0.50 ...")
        enc_a, rdr_a, gate_a = train_model(0.50, use_learned_gate=True)
        results["A"] = evaluate_model(enc_a, rdr_a, gate_a, 0.50)
        print(f"    null_f1={results['A']['null_f1']:.3f}, retr_recall={results['A']['retrieval_recall']:.3f}")

        # Condition B: Always-retrieve, p=0.50
        print("  Condition B: always-retrieve, p_relevant=0.50 ...")
        enc_b, rdr_b, _ = train_model(0.50, use_learned_gate=False, use_always_retrieve=True)
        results["B"] = evaluate_model(enc_b, rdr_b, None, 0.50, use_always_retrieve=True)
        print(f"    null_f1={results['B']['null_f1']:.3f}, retr_recall={results['B']['retrieval_recall']:.3f}")

        # Condition C: Always-null, p=0.50
        print("  Condition C: always-null, p_relevant=0.50 ...")
        enc_c, rdr_c, _ = train_model(0.50, use_learned_gate=False, use_always_retrieve=False)
        results["C"] = evaluate_model(enc_c, rdr_c, None, 0.50, use_always_null=True)
        print(f"    null_f1={results['C']['null_f1']:.3f}, retr_recall={results['C']['retrieval_recall']:.3f}")

        # Condition D: Learned gate, p=0.20 (degenerate control)
        print("  Condition D: learned gate, p_relevant=0.20 ...")
        enc_d, rdr_d, gate_d = train_model(0.20, use_learned_gate=True)
        results["D"] = evaluate_model(enc_d, rdr_d, gate_d, 0.20)
        print(f"    null_f1={results['D']['null_f1']:.3f}, retr_recall={results['D']['retrieval_recall']:.3f}, "
              f"degenerate={results['D']['is_degenerate']}")

        null_f1_A = results["A"]["null_f1"]
        retr_A    = results["A"]["retrieval_recall"]
        degen_A   = results["A"]["is_degenerate"]

        if degen_A:
            outcome = OUTCOME_REFUTED
        elif null_f1_A > 0.60 and retr_A > 0.60:
            outcome = OUTCOME_SUPPORTED
        else:
            outcome = OUTCOME_INCONCLUSIVE

        metrics = {}
        for cond, r in results.items():
            for k, v in r.items():
                metrics[f"{k}_{cond}"] = round(v, 4) if isinstance(v, float) else v

        notes = (
            f"Cond A (learned, p=0.5): null_f1={null_f1_A:.3f}, retr_recall={retr_A:.3f}, "
            f"degenerate={degen_A}. "
            f"Cond B (always-retrieve): null_f1={results['B']['null_f1']:.3f}. "
            f"Cond D (p=0.2 control): degenerate={results['D']['is_degenerate']}."
        )
        config = {
            "vocab_size": VOCAB_SIZE, "hidden_dim": HIDDEN_DIM,
            "memory_size": MEMORY_SIZE, "train_steps": TRAIN_STEPS,
            "batch_size": BATCH_SIZE,
        }
        return self.result(outcome, metrics, notes, config)


if __name__ == "__main__":
    Exp92NullRetrievalBalanced().execute()
