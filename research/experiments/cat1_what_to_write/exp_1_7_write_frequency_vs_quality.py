"""
Experiment 1.7 — Write Frequency vs Quality

Hypothesis: For a fixed storage budget, infrequent writes with high compression
outperform frequent writes with low compression on downstream retrieval.

Setup:
  - Fixed budget = 256 floats of memory storage
  - Policy A (freq): write every 4 tokens, compress to dim=16 (16 slots x 16-dim)
  - Policy B (infreq): write every 16 tokens, store full 64-dim (4 slots x 64-dim)
  - Linear encoder/decoder for compression in Policy A
  - Long-sequence (SEQ_LEN=48) associative recall, answer may be at any position
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

VOCAB_SIZE    = 64
HIDDEN_DIM    = 64
SEQ_LEN       = 48          # longer sequence
BATCH_SIZE    = 32
TRAIN_STEPS   = 1500
EVAL_STEPS    = 300
NUM_PAIRS     = 8
LR            = 3e-4
DEVICE        = "cpu"

# Budget = 256 floats
FREQ_STRIDE   = 4            # write every 4 tokens
FREQ_SLOTS    = SEQ_LEN // FREQ_STRIDE   # = 12 slots
FREQ_DIM      = 16           # compressed dim  -> 12 * 16 = 192 (closest to 256)
# We actually use FREQ_SLOTS=16 and FREQ_DIM=16 to hit 256 exactly
FREQ_SLOTS    = 16
FREQ_DIM      = 16           # 16 * 16 = 256

INFREQ_STRIDE = 16           # write every 16 tokens
INFREQ_SLOTS  = SEQ_LEN // INFREQ_STRIDE   # = 3 -> use 4
INFREQ_SLOTS  = 4
INFREQ_DIM    = HIDDEN_DIM   # full 64-dim  -> 4 * 64 = 256

# ── Task ──────────────────────────────────────────────────────────────────────

def make_recall_batch(batch_size: int):
    """Long-sequence associative recall, pairs planted at any position."""
    seqs    = torch.randint(2, VOCAB_SIZE, (batch_size, SEQ_LEN))
    queries = torch.zeros(batch_size, dtype=torch.long)
    targets = torch.zeros(batch_size, dtype=torch.long)
    for b in range(batch_size):
        positions = torch.randperm(SEQ_LEN - 1)[:NUM_PAIRS * 2]
        pairs = []
        for i in range(NUM_PAIRS):
            k_pos = positions[i * 2].item()
            v_pos = positions[i * 2 + 1].item()
            key   = torch.randint(2, VOCAB_SIZE // 2, (1,)).item()
            val   = torch.randint(VOCAB_SIZE // 2, VOCAB_SIZE, (1,)).item()
            seqs[b, k_pos] = key
            seqs[b, v_pos] = val
            pairs.append((k_pos, v_pos))
        idx = torch.randint(NUM_PAIRS, (1,)).item()
        queries[b] = seqs[b, pairs[idx][0]]
        targets[b] = seqs[b, pairs[idx][1]]
    return seqs, queries, targets


# ── Encoder ───────────────────────────────────────────────────────────────────

class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, HIDDEN_DIM)
        self.attn  = nn.MultiheadAttention(HIDDEN_DIM, 2, batch_first=True)
        self.ff    = nn.Sequential(
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM * 2), nn.ReLU(),
            nn.Linear(HIDDEN_DIM * 2, HIDDEN_DIM),
        )
        self.norm1 = nn.LayerNorm(HIDDEN_DIM)
        self.norm2 = nn.LayerNorm(HIDDEN_DIM)

    def forward(self, x):
        h = self.embed(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        h = self.norm1(h + a)
        h = self.norm2(h + self.ff(h))
        return h


# ── Policy A: Frequent writes with compression ────────────────────────────────

class FreqPolicyModel(nn.Module):
    """
    Write every FREQ_STRIDE tokens, compress HIDDEN_DIM -> FREQ_DIM.
    FREQ_SLOTS x FREQ_DIM = 256 floats budget.
    """

    def __init__(self):
        super().__init__()
        self.encoder     = TinyEncoder()
        self.compressor  = nn.Linear(HIDDEN_DIM, FREQ_DIM)
        self.decompressor = nn.Linear(FREQ_DIM, HIDDEN_DIM)
        self.query_proj  = nn.Linear(HIDDEN_DIM, FREQ_DIM)
        self.out_proj    = nn.Linear(HIDDEN_DIM, VOCAB_SIZE)
        self.query_embed = nn.Embedding(VOCAB_SIZE, HIDDEN_DIM)

    def _build_memory(self, hidden):
        """Sample every FREQ_STRIDE-th position, compress."""
        B, L, H = hidden.shape
        positions = list(range(0, L, FREQ_STRIDE))[:FREQ_SLOTS]
        # pad if needed
        while len(positions) < FREQ_SLOTS:
            positions.append(positions[-1])
        pos_t   = torch.tensor(positions, device=hidden.device)
        mem_raw = hidden[:, pos_t, :]              # (B, FREQ_SLOTS, H)
        mem_c   = self.compressor(mem_raw)         # (B, FREQ_SLOTS, FREQ_DIM)
        return mem_c

    def forward(self, seq, query, target):
        hidden = self.encoder(seq)
        mem_c  = self._build_memory(hidden)        # (B, FREQ_SLOTS, FREQ_DIM)
        # decompress for reader
        mem_h  = self.decompressor(mem_c)          # (B, FREQ_SLOTS, H)
        q_h    = self.query_embed(query)           # (B, H)
        q_proj = self.query_proj(q_h).unsqueeze(1) # (B, 1, FREQ_DIM)
        scores  = (q_proj * mem_c).sum(-1) / (FREQ_DIM ** 0.5)
        weights = F.softmax(scores, dim=-1).unsqueeze(-1)
        retrieved = (weights * mem_h).sum(1)       # (B, H)
        logits = self.out_proj(retrieved)
        return F.cross_entropy(logits, target)


# ── Policy B: Infrequent writes, full dimension ───────────────────────────────

class InfreqPolicyModel(nn.Module):
    """
    Write every INFREQ_STRIDE tokens, keep full HIDDEN_DIM vectors.
    INFREQ_SLOTS x HIDDEN_DIM = 256 floats budget.
    """

    def __init__(self):
        super().__init__()
        self.encoder     = TinyEncoder()
        self.query_proj  = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.out_proj    = nn.Linear(HIDDEN_DIM, VOCAB_SIZE)
        self.query_embed = nn.Embedding(VOCAB_SIZE, HIDDEN_DIM)

    def _build_memory(self, hidden):
        B, L, H = hidden.shape
        positions = list(range(0, L, INFREQ_STRIDE))[:INFREQ_SLOTS]
        while len(positions) < INFREQ_SLOTS:
            positions.append(positions[-1])
        pos_t = torch.tensor(positions, device=hidden.device)
        return hidden[:, pos_t, :]   # (B, INFREQ_SLOTS, H)

    def forward(self, seq, query, target):
        hidden = self.encoder(seq)
        mem    = self._build_memory(hidden)         # (B, INFREQ_SLOTS, H)
        q_h    = self.query_embed(query)
        q_proj = self.query_proj(q_h).unsqueeze(1)  # (B, 1, H)
        scores  = (q_proj * mem).sum(-1) / (HIDDEN_DIM ** 0.5)
        weights = F.softmax(scores, dim=-1).unsqueeze(-1)
        retrieved = (weights * mem).sum(1)
        logits = self.out_proj(retrieved)
        return F.cross_entropy(logits, target)


# ── Train / Eval ──────────────────────────────────────────────────────────────

def train_model(model: nn.Module, steps: int, name: str) -> None:
    opt = Adam(model.parameters(), lr=LR)
    model.train()
    for step in range(steps):
        seq, q, t = make_recall_batch(BATCH_SIZE)
        seq, q, t = seq.to(DEVICE), q.to(DEVICE), t.to(DEVICE)
        loss = model(seq, q, t)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (step + 1) % 500 == 0:
            print(f"    [{name}] step {step+1}/{steps}  loss={loss.item():.4f}")


def eval_model(model: nn.Module, steps: int) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for _ in range(steps):
            seq, q, t = make_recall_batch(BATCH_SIZE)
            seq, q, t = seq.to(DEVICE), q.to(DEVICE), t.to(DEVICE)
            hidden     = model.encoder(seq)
            mem        = model._build_memory(hidden)
            q_h        = model.query_embed(q)

            if isinstance(model, FreqPolicyModel):
                q_proj  = model.query_proj(q_h).unsqueeze(1)
                scores  = (q_proj * mem).sum(-1) / (FREQ_DIM ** 0.5)
                weights = F.softmax(scores, dim=-1).unsqueeze(-1)
                mem_h   = model.decompressor(mem)
                retr    = (weights * mem_h).sum(1)
            else:
                q_proj  = model.query_proj(q_h).unsqueeze(1)
                scores  = (q_proj * mem).sum(-1) / (HIDDEN_DIM ** 0.5)
                weights = F.softmax(scores, dim=-1).unsqueeze(-1)
                retr    = (weights * mem).sum(1)

            logits = model.out_proj(retr)
            preds  = logits.argmax(-1)
            correct += (preds == t).sum().item()
            total   += t.shape[0]
    return correct / total


# ── Experiment ────────────────────────────────────────────────────────────────

class Exp17WriteFrequencyVsQuality(Experiment):
    experiment_id = "exp_1_7"
    hypothesis = (
        "For a fixed storage budget, infrequent writes with high compression "
        "outperform frequent writes with low compression on downstream retrieval."
    )

    def run(self) -> ExperimentResult:
        torch.manual_seed(42)

        print("  Training freq policy (write every 4, compress to dim=16) ...")
        freq_model = FreqPolicyModel().to(DEVICE)
        train_model(freq_model, TRAIN_STEPS, "freq")
        freq_acc = eval_model(freq_model, EVAL_STEPS)
        print(f"    freq_acc={freq_acc:.3f}")

        print("  Training infreq policy (write every 16, full dim=64) ...")
        infreq_model = InfreqPolicyModel().to(DEVICE)
        train_model(infreq_model, TRAIN_STEPS, "infreq")
        infreq_acc = eval_model(infreq_model, EVAL_STEPS)
        print(f"    infreq_acc={infreq_acc:.3f}")

        gap = infreq_acc - freq_acc

        if infreq_acc > freq_acc:
            outcome = OUTCOME_SUPPORTED
        elif freq_acc > infreq_acc + 0.02:
            outcome = OUTCOME_REFUTED
        else:
            outcome = OUTCOME_INCONCLUSIVE

        freq_budget    = FREQ_SLOTS * FREQ_DIM
        infreq_budget  = INFREQ_SLOTS * INFREQ_DIM

        metrics = {
            "freq_acc":           round(freq_acc, 4),
            "infreq_acc":         round(infreq_acc, 4),
            "gap_infreq_minus_freq": round(gap, 4),
            "freq_budget_floats": freq_budget,
            "infreq_budget_floats": infreq_budget,
        }
        notes = (
            f"Infreq acc={infreq_acc:.3f} vs freq acc={freq_acc:.3f}. "
            f"Gap={gap:+.4f}. "
            f"Budget: freq={freq_budget} floats, infreq={infreq_budget} floats."
        )
        return self.result(outcome, metrics, notes, config={
            "vocab_size": VOCAB_SIZE, "hidden_dim": HIDDEN_DIM,
            "seq_len": SEQ_LEN, "batch_size": BATCH_SIZE,
            "train_steps": TRAIN_STEPS, "num_pairs": NUM_PAIRS,
            "freq_stride": FREQ_STRIDE, "freq_slots": FREQ_SLOTS,
            "freq_dim": FREQ_DIM, "infreq_stride": INFREQ_STRIDE,
            "infreq_slots": INFREQ_SLOTS, "infreq_dim": INFREQ_DIM,
        })


if __name__ == "__main__":
    Exp17WriteFrequencyVsQuality().execute()
