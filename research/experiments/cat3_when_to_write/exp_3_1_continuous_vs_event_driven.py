"""
Experiment 3.1 — Continuous vs Event-Driven Writing

Hypothesis: Event-driven writing (learned gate) produces better memory coverage
than writing every N tokens for a fixed storage budget.

Setup:
  - MEMORY_SLOTS=6
  - Policy A (continuous): write every N tokens (N = SEQ_LEN // MEMORY_SLOTS)
  - Policy B (event-driven): learned write gate (sigmoid MLP), top-k by gate score
  - Both write the same total number of tokens over a sequence
  - Same downstream retrieval: associative recall
  - Coverage metric: fraction of distinct planted key-value pair keys in memory
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

VOCAB_SIZE    = 64
SEQ_LEN       = 24
HIDDEN_DIM    = 64
MEMORY_SLOTS  = 6
BATCH_SIZE    = 32
TRAIN_STEPS   = 1500
LR            = 3e-4
LOG_EVERY     = 100
DEVICE        = "cpu"

# Number of key-value pairs planted per sequence
NUM_PAIRS     = 4
# Stride for continuous policy
STRIDE        = SEQ_LEN // MEMORY_SLOTS   # = 4

# Special token ids
KEY_TOKEN_START   = 4    # keys use tokens 4..4+NUM_PAIRS*2
QUERY_MARKER      = 2
PAD_TOKEN         = 0


# ── Data Generation ────────────────────────────────────────────────────────────

def make_assoc_batch(batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate associative recall sequences.
    Format: [key1, val1, ..., keyN, valN, <filler>, query_marker, query_key] -> target_val
    Returns: (seq, target, key_positions) where key_positions marks where keys appear.
    """
    seq = torch.zeros(batch_size, SEQ_LEN, dtype=torch.long)
    target = torch.zeros(batch_size, dtype=torch.long)
    key_positions = torch.zeros(batch_size, SEQ_LEN, dtype=torch.float)  # 1 where a key/val is

    for b in range(batch_size):
        # Sample NUM_PAIRS distinct key-value pairs
        keys = torch.randint(4, 4 + NUM_PAIRS * 2, (NUM_PAIRS,))
        keys = keys.unique()[:NUM_PAIRS]
        if len(keys) < NUM_PAIRS:
            extra = torch.randint(4, 32, (NUM_PAIRS - len(keys),))
            keys = torch.cat([keys, extra])[:NUM_PAIRS]
        vals = torch.randint(32, VOCAB_SIZE, (NUM_PAIRS,))

        # Place key-value pairs at the start
        pos = 0
        for i in range(NUM_PAIRS):
            if pos + 1 < SEQ_LEN - 3:
                seq[b, pos]     = keys[i]
                seq[b, pos + 1] = vals[i]
                key_positions[b, pos]     = 1.0
                key_positions[b, pos + 1] = 1.0
                pos += 2

        # Fill remaining with filler tokens (token id 3)
        for p in range(pos, SEQ_LEN - 3):
            seq[b, p] = 3

        # End: query_marker, then a random key
        query_idx = torch.randint(0, NUM_PAIRS, (1,)).item()
        seq[b, SEQ_LEN - 3] = QUERY_MARKER
        seq[b, SEQ_LEN - 2] = keys[query_idx]
        seq[b, SEQ_LEN - 1] = 0  # mask target position
        target[b] = vals[query_idx]

    return seq, target, key_positions


# ── Models ─────────────────────────────────────────────────────────────────────

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, HIDDEN_DIM)
        self.ff    = nn.Sequential(
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM * 2), nn.ReLU(),
            nn.Linear(HIDDEN_DIM * 2, HIDDEN_DIM),
        )
        self.norm  = nn.LayerNorm(HIDDEN_DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embed(x)
        return self.norm(h + self.ff(h))


class WriteGateMLP(nn.Module):
    """Learned write gate: scores each token position."""
    def __init__(self):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM // 2), nn.ReLU(),
            nn.Linear(HIDDEN_DIM // 2, 1),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Returns gate scores (B, L) after sigmoid."""
        return torch.sigmoid(self.gate(hidden)).squeeze(-1)


class ReadHead(nn.Module):
    """Dot-product attention over memory slots -> next token prediction."""
    def __init__(self):
        super().__init__()
        self.query_proj = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.out_proj   = nn.Linear(HIDDEN_DIM, VOCAB_SIZE)

    def forward(self, hidden: torch.Tensor, memory: torch.Tensor, mem_mask: torch.Tensor) -> torch.Tensor:
        """
        hidden: (B, L, H) — use last non-padding token as query
        memory: (B, MEMORY_SLOTS, H)
        mem_mask: (B, MEMORY_SLOTS) — 1 where slot is valid
        """
        query = self.query_proj(hidden[:, -1, :])  # (B, H)
        scores = torch.bmm(memory, query.unsqueeze(-1)).squeeze(-1)  # (B, S)
        scores = scores.masked_fill(mem_mask == 0, -1e9)
        attn   = torch.softmax(scores, dim=-1)                        # (B, S)
        ctx    = (attn.unsqueeze(-1) * memory).sum(1)                 # (B, H)
        return self.out_proj(ctx)


# ── Memory Write Policies ──────────────────────────────────────────────────────

def continuous_write(hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Write every STRIDE-th token. Returns (memory, mask, selected_positions).
    memory: (B, MEMORY_SLOTS, H)
    mask:   (B, MEMORY_SLOTS)
    positions: (B, SEQ_LEN) binary indicating which positions were written
    """
    B, L, H = hidden.shape
    positions = torch.zeros(B, L, device=hidden.device)
    indices = list(range(0, L, STRIDE))[:MEMORY_SLOTS]
    for idx in indices:
        positions[:, idx] = 1.0

    selected_idx = torch.tensor(indices, device=hidden.device)  # (S,)
    memory = hidden[:, selected_idx, :]  # (B, S, H)
    mask   = torch.ones(B, len(indices), device=hidden.device)

    # Pad to MEMORY_SLOTS if needed
    if len(indices) < MEMORY_SLOTS:
        pad_size = MEMORY_SLOTS - len(indices)
        memory = F.pad(memory, (0, 0, 0, pad_size))
        mask   = F.pad(mask,   (0, pad_size))

    return memory, mask, positions


def event_driven_write(
    hidden: torch.Tensor, gate_scores: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Write top-k tokens by gate score.
    Returns (memory, mask, selected_positions).
    """
    B, L, H = hidden.shape
    k = MEMORY_SLOTS
    topk_vals, topk_idx = torch.topk(gate_scores, k, dim=1)  # (B, k)

    # Gather hidden states at top-k positions
    topk_idx_exp = topk_idx.unsqueeze(-1).expand(-1, -1, H)  # (B, k, H)
    memory = torch.gather(hidden, 1, topk_idx_exp)            # (B, k, H)
    mask   = torch.ones(B, k, device=hidden.device)

    positions = torch.zeros(B, L, device=hidden.device)
    positions.scatter_(1, topk_idx, 1.0)

    return memory, mask, positions


# ── Coverage Metric ────────────────────────────────────────────────────────────

def compute_coverage(
    key_positions: torch.Tensor, written_positions: torch.Tensor
) -> float:
    """
    Fraction of key/val positions that were written to memory.
    key_positions: (B, L) binary
    written_positions: (B, L) binary
    """
    covered = (key_positions * written_positions).sum(dim=1)        # (B,)
    total   = key_positions.sum(dim=1).clamp(min=1)                 # (B,)
    return (covered / total).mean().item()


# ── Training ───────────────────────────────────────────────────────────────────

def train_policy(policy: str) -> dict:
    """
    Train a model using the specified write policy ('continuous' or 'event_driven').
    Returns dict with final accuracy, coverage, and trajectory.
    """
    enc      = Encoder().to(DEVICE)
    gate_net = WriteGateMLP().to(DEVICE) if policy == "event_driven" else None
    head     = ReadHead().to(DEVICE)

    params = list(enc.parameters()) + list(head.parameters())
    if gate_net is not None:
        params += list(gate_net.parameters())
    opt = Adam(params, lr=LR)

    acc_log      = []
    coverage_log = []

    for step in range(TRAIN_STEPS):
        seq, target, key_positions = make_assoc_batch(BATCH_SIZE)
        seq      = seq.to(DEVICE)
        target   = target.to(DEVICE)
        key_positions = key_positions.to(DEVICE)

        hidden = enc(seq)  # (B, L, H)

        if policy == "continuous":
            memory, mask, written_pos = continuous_write(hidden)
        else:
            gate_scores = gate_net(hidden)         # (B, L)
            memory, mask, written_pos = event_driven_write(hidden, gate_scores)

        logits    = head(hidden, memory, mask)     # (B, VOCAB_SIZE)
        task_loss = F.cross_entropy(logits, target)

        opt.zero_grad()
        task_loss.backward()
        opt.step()

        if step % LOG_EVERY == 0:
            with torch.no_grad():
                preds    = logits.argmax(dim=-1)
                acc      = (preds == target).float().mean().item()
                coverage = compute_coverage(key_positions, written_pos)
                acc_log.append(acc)
                coverage_log.append(coverage)
                print(f"  [{policy:12s}] step={step:4d}  loss={task_loss.item():.3f}"
                      f"  acc={acc:.3f}  coverage={coverage:.3f}")

    # Final evaluation
    with torch.no_grad():
        total_acc  = 0.0
        total_cov  = 0.0
        eval_steps = 20
        for _ in range(eval_steps):
            seq, target, key_positions = make_assoc_batch(BATCH_SIZE)
            seq      = seq.to(DEVICE)
            target   = target.to(DEVICE)
            key_positions = key_positions.to(DEVICE)
            hidden   = enc(seq)
            if policy == "continuous":
                memory, mask, written_pos = continuous_write(hidden)
            else:
                gate_scores = gate_net(hidden)
                memory, mask, written_pos = event_driven_write(hidden, gate_scores)
            logits  = head(hidden, memory, mask)
            preds   = logits.argmax(dim=-1)
            total_acc += (preds == target).float().mean().item()
            total_cov += compute_coverage(key_positions, written_pos)
        final_acc = total_acc / eval_steps
        final_cov = total_cov / eval_steps

    return {
        "final_acc":      final_acc,
        "final_coverage": final_cov,
        "acc_log":        acc_log,
        "coverage_log":   coverage_log,
    }


# ── Experiment ─────────────────────────────────────────────────────────────────

class Exp31ContinuousVsEventDriven(Experiment):
    experiment_id = "exp_3_1"
    hypothesis = (
        "Event-driven writing (learned gate) produces better memory coverage than "
        "writing every N tokens for a fixed storage budget."
    )

    def run(self) -> ExperimentResult:
        torch.manual_seed(42)

        print("\nTraining continuous policy...")
        cont_res = train_policy("continuous")

        print("\nTraining event-driven policy...")
        ev_res   = train_policy("event_driven")

        continuous_acc      = cont_res["final_acc"]
        event_driven_acc    = ev_res["final_acc"]
        continuous_coverage = cont_res["final_coverage"]
        event_driven_coverage = ev_res["final_coverage"]

        print(f"\nContinuous:   acc={continuous_acc:.3f}  coverage={continuous_coverage:.3f}")
        print(f"Event-driven: acc={event_driven_acc:.3f}  coverage={event_driven_coverage:.3f}")

        acc_delta      = event_driven_acc - continuous_acc
        coverage_delta = event_driven_coverage - continuous_coverage

        if event_driven_acc > continuous_acc or event_driven_coverage > continuous_coverage + 0.05:
            outcome = OUTCOME_SUPPORTED
        elif continuous_acc >= event_driven_acc and continuous_coverage >= event_driven_coverage:
            outcome = OUTCOME_REFUTED
        else:
            outcome = OUTCOME_INCONCLUSIVE

        metrics = {
            "continuous_acc":          round(continuous_acc, 4),
            "event_driven_acc":        round(event_driven_acc, 4),
            "continuous_coverage":     round(continuous_coverage, 4),
            "event_driven_coverage":   round(event_driven_coverage, 4),
            "acc_delta":               round(acc_delta, 4),
            "coverage_delta":          round(coverage_delta, 4),
        }
        notes = (
            f"Event-driven acc delta: {acc_delta:+.3f}. "
            f"Coverage delta: {coverage_delta:+.3f}. "
            f"Stride used for continuous: {STRIDE}. "
            f"Top-k={MEMORY_SLOTS} used for event-driven."
        )
        config = {
            "vocab_size":    VOCAB_SIZE,
            "seq_len":       SEQ_LEN,
            "hidden_dim":    HIDDEN_DIM,
            "memory_slots":  MEMORY_SLOTS,
            "batch_size":    BATCH_SIZE,
            "train_steps":   TRAIN_STEPS,
            "num_pairs":     NUM_PAIRS,
            "stride":        STRIDE,
        }
        return self.result(outcome, metrics, notes, config)


if __name__ == "__main__":
    Exp31ContinuousVsEventDriven().execute()
