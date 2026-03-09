"""
exp_19_1_sparse_hopfield_retrieval.py

Hypothesis: Sparse Hopfield retrieval (sparsemax top-k=2) outperforms standard softmax
attention by >5% precision@1 on 40% interference tasks.
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

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
VOCAB_SIZE = 128
HIDDEN_DIM = 64
MEMORY_SLOTS = 16
SEQ_LEN = 32
NUM_PAIRS = 6
INTERFERENCE_FRACTION = 0.4
STEPS = 400
BATCH = 32
LR = 3e-4
DEVICE = "cpu"


# ---------------------------------------------------------------------------
# Data generator
# ---------------------------------------------------------------------------
def make_assoc_batch(batch_size, seq_len, vocab_size, num_pairs):
    seq = torch.zeros(batch_size, seq_len, dtype=torch.long)
    target = torch.zeros(batch_size, dtype=torch.long)
    for b in range(batch_size):
        keys = torch.randint(4, max(8, vocab_size // 3), (num_pairs * 3,)).unique()[:num_pairs]
        while len(keys) < num_pairs:
            keys = torch.cat([keys, torch.randint(4, vocab_size // 3, (1,))])[:num_pairs]
        vals = torch.randint(vocab_size // 2, vocab_size, (num_pairs,))
        pos = 0
        for i in range(num_pairs):
            if pos + 1 < seq_len - 3:
                seq[b, pos] = keys[i]; seq[b, pos + 1] = vals[i]; pos += 2
        for p in range(pos, seq_len - 3):
            seq[b, p] = 3
        qi = torch.randint(0, num_pairs, (1,)).item()
        seq[b, seq_len - 3] = 2
        seq[b, seq_len - 2] = keys[qi]
        seq[b, seq_len - 1] = 0
        target[b] = vals[qi]
    return seq, target


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------
class Encoder(nn.Module):
    def __init__(self, vocab_size, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        h = self.embed(x)
        return self.norm(h + self.ff(h))


# ---------------------------------------------------------------------------
# Sparse attention helper
# ---------------------------------------------------------------------------
def k_sparse_softmax(scores, k):
    """scores: (B, N) — returns soft weights that sum to 1 but only top-k are non-zero."""
    B, N = scores.shape
    topk_vals, topk_idx = scores.topk(k, dim=-1)       # (B, k)
    sparse_scores = torch.full_like(scores, float('-inf'))
    sparse_scores.scatter_(-1, topk_idx, topk_vals)
    return torch.softmax(sparse_scores, dim=-1)


# ---------------------------------------------------------------------------
# Memory module
# ---------------------------------------------------------------------------
class HopfieldMemory(nn.Module):
    """Write with top-k gate; read with configurable attention type."""

    def __init__(self, hidden_dim, memory_slots, read_mode="softmax", read_k=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.memory_slots = memory_slots
        self.read_mode = read_mode          # "softmax" | "sparse" | "argmax"
        self.read_k = read_k
        # Write gate: selects tokens for memory
        self.write_gate = nn.Linear(hidden_dim, 1)
        # Query / key projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        # Output head
        self.out_head = nn.Linear(hidden_dim, VOCAB_SIZE)

    def forward(self, enc_hidden, query_hidden):
        """
        enc_hidden:   (B, T, H)
        query_hidden: (B, H)   — the last token hidden state (query position)
        Returns: logits (B, VOCAB_SIZE), write_gate_scores (B, T)
        """
        B, T, H = enc_hidden.shape

        # Write gate scores over all token positions
        gate_scores = self.write_gate(enc_hidden).squeeze(-1)   # (B, T)
        gate_probs = torch.sigmoid(gate_scores)                  # (B, T)

        # Top-K write selection: pick top MEMORY_SLOTS tokens
        k_write = min(self.memory_slots, T)
        _, top_write_idx = gate_probs.topk(k_write, dim=-1)     # (B, k_write)

        # Build memory: (B, memory_slots, H) — fill slots with selected tokens
        memory = torch.zeros(B, self.memory_slots, H, device=enc_hidden.device)
        for slot in range(k_write):
            slot_idx = top_write_idx[:, slot]                    # (B,)
            token_h = enc_hidden[torch.arange(B), slot_idx]     # (B, H)
            memory[:, slot] = token_h

        # Read: query attends to memory slots
        q = self.q_proj(query_hidden).unsqueeze(1)               # (B, 1, H)
        k = self.k_proj(memory)                                  # (B, M, H)
        raw_scores = (q @ k.transpose(1, 2)).squeeze(1) / (H ** 0.5)  # (B, M)

        if self.read_mode == "softmax":
            attn = torch.softmax(raw_scores, dim=-1)             # (B, M)
        elif self.read_mode == "sparse":
            attn = k_sparse_softmax(raw_scores, self.read_k)     # (B, M)
        else:  # "argmax" — hard top-1
            best = raw_scores.argmax(dim=-1, keepdim=True)       # (B, 1)
            attn = torch.zeros_like(raw_scores).scatter_(-1, best, 1.0)

        retrieved = (attn.unsqueeze(1) @ memory).squeeze(1)      # (B, H)
        logits = self.out_head(retrieved + query_hidden)          # (B, VOCAB_SIZE)
        return logits, gate_probs, attn


# ---------------------------------------------------------------------------
# Precision@1 helper
# ---------------------------------------------------------------------------
def precision_at_1(attn_weights, memory_contents, query_key_hidden, top_n=1):
    """
    Measures fraction of reads where the most-attended slot contains the
    highest cosine similarity to the query.  Uses the attention argmax.
    attn_weights:     (B, M)
    memory_contents:  (B, M, H)
    query_key_hidden: (B, H)
    """
    best_slot = attn_weights.argmax(dim=-1)               # (B,)
    retrieved_h = memory_contents[torch.arange(attn_weights.size(0)), best_slot]  # (B, H)
    sim = F.cosine_similarity(retrieved_h, query_key_hidden, dim=-1)              # (B,)
    return (sim > 0.3).float().mean().item()


# ---------------------------------------------------------------------------
# Training loop for one condition
# ---------------------------------------------------------------------------
def train_condition(read_mode, read_k=2):
    torch.manual_seed(42)
    encoder = Encoder(VOCAB_SIZE, HIDDEN_DIM).to(DEVICE)
    memory_mod = HopfieldMemory(HIDDEN_DIM, MEMORY_SLOTS, read_mode=read_mode, read_k=read_k).to(DEVICE)
    opt = Adam(list(encoder.parameters()) + list(memory_mod.parameters()), lr=LR)

    correct = 0
    total = 0
    high_conf_correct = 0
    high_conf_total = 0

    for step in range(STEPS):
        seq, target = make_assoc_batch(BATCH, SEQ_LEN, VOCAB_SIZE, NUM_PAIRS)
        seq, target = seq.to(DEVICE), target.to(DEVICE)

        enc_hidden = encoder(seq)   # (B, T, H)

        # Interference: corrupt 40% of items — mix 2 key embeddings
        corrupt_mask = torch.rand(BATCH) < INTERFERENCE_FRACTION
        if corrupt_mask.any():
            for i in range(0, NUM_PAIRS * 2, 2):     # key positions: 0, 2, 4 ...
                j = (i + 2) % (NUM_PAIRS * 2)
                enc_hidden_det = enc_hidden.detach()
                noise = 0.3 * enc_hidden_det[corrupt_mask, j]
                enc_hidden_clone = enc_hidden.clone()
                enc_hidden_clone[corrupt_mask, i] = (
                    0.7 * enc_hidden[corrupt_mask, i] + noise
                )
                enc_hidden = enc_hidden_clone

        query_hidden = enc_hidden[:, SEQ_LEN - 2]   # key query position

        logits, gate_probs, attn = memory_mod(enc_hidden, query_hidden)

        loss = F.cross_entropy(logits, target)
        opt.zero_grad()
        loss.backward()
        opt.step()

        # Accuracy tracking (last quarter of training)
        if step >= STEPS * 3 // 4:
            preds = logits.argmax(dim=-1)
            correct += (preds == target).sum().item()
            total += BATCH
            # High-confidence reads: max attention weight > 0.6
            max_attn = attn.max(dim=-1).values   # (B,)
            hc_mask = max_attn > 0.6
            if hc_mask.any():
                hc_correct = (preds[hc_mask] == target[hc_mask]).sum().item()
                high_conf_correct += hc_correct
                high_conf_total += hc_mask.sum().item()

    acc = correct / max(total, 1)
    precision = high_conf_correct / max(high_conf_total, 1)
    return acc, precision


# ---------------------------------------------------------------------------
# Experiment class
# ---------------------------------------------------------------------------
class Exp191SparseHopfieldRetrieval(Experiment):
    experiment_id = "exp_19_1"
    hypothesis = (
        "Sparse Hopfield retrieval (sparsemax top-k=2) outperforms standard softmax "
        "attention by >5% precision@1 on 40% interference tasks."
    )

    def run(self) -> ExperimentResult:
        config = dict(
            VOCAB_SIZE=VOCAB_SIZE, HIDDEN_DIM=HIDDEN_DIM, MEMORY_SLOTS=MEMORY_SLOTS,
            SEQ_LEN=SEQ_LEN, NUM_PAIRS=NUM_PAIRS, INTERFERENCE_FRACTION=INTERFERENCE_FRACTION,
            STEPS=STEPS, BATCH=BATCH, LR=LR,
        )

        print("Training condition A: standard softmax read...")
        acc_A, precision_A = train_condition("softmax")
        print(f"  acc_A={acc_A:.4f}  precision_A={precision_A:.4f}")

        print("Training condition B: sparse k=2 read...")
        acc_B, precision_B = train_condition("sparse", read_k=2)
        print(f"  acc_B={acc_B:.4f}  precision_B={precision_B:.4f}")

        print("Training condition C: hard top-1 argmax read...")
        acc_C, precision_C = train_condition("argmax")
        print(f"  acc_C={acc_C:.4f}  precision_C={precision_C:.4f}")

        metrics = dict(
            acc_A=round(acc_A, 4),
            acc_B=round(acc_B, 4),
            acc_C=round(acc_C, 4),
            precision_A=round(precision_A, 4),
            precision_B=round(precision_B, 4),
            precision_C=round(precision_C, 4),
            acc_gap_B_minus_A=round(acc_B - acc_A, 4),
            precision_gap_B_minus_A=round(precision_B - precision_A, 4),
        )

        if acc_B > acc_A + 0.05 and precision_B > precision_A + 0.10:
            outcome = OUTCOME_SUPPORTED
            notes = "Sparse k=2 read outperforms softmax on both accuracy and precision on interference tasks."
        elif acc_A >= acc_B or precision_A >= precision_B:
            outcome = OUTCOME_REFUTED
            notes = "Soft attention matches or beats sparse on interference tasks."
        else:
            outcome = OUTCOME_INCONCLUSIVE
            notes = (
                f"Precision improved by {precision_B - precision_A:.4f} but accuracy gap "
                f"({acc_B - acc_A:.4f}) did not exceed 0.05 threshold."
            )

        return self.result(outcome, metrics=metrics, notes=notes, config=config)


if __name__ == "__main__":
    Exp191SparseHopfieldRetrieval().execute()
