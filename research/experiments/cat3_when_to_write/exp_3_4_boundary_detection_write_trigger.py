"""
Experiment 3.4 — Boundary Detection Write Trigger

Hypothesis: Semantic-boundary-triggered writing outperforms fixed-interval writing
on long-document tasks with clear topical structure.

Setup:
  - Sequences with explicit boundary tokens (token_id=1 marks segment start)
  - 3 segments, each with its own key-value pairs
  - Policy A (fixed interval): write every K tokens
  - Policy B (boundary triggered): write when boundary token detected + a few after
  - Both have MEMORY_SLOTS=6 budget
  - Measure per-segment accuracy
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

VOCAB_SIZE     = 64
SEQ_LEN        = 24
HIDDEN_DIM     = 64
MEMORY_SLOTS   = 6
BATCH_SIZE     = 32
TRAIN_STEPS    = 1500
LR             = 3e-4
LOG_EVERY      = 100
DEVICE         = "cpu"

BOUNDARY_TOKEN = 1
N_SEGMENTS     = 3
PAIRS_PER_SEG  = 1    # one key-value pair per segment
# Fixed interval stride
FIXED_STRIDE   = SEQ_LEN // MEMORY_SLOTS   # = 4
# How many tokens after a boundary to write
BOUNDARY_WINDOW = 2


# ── Data Generation ────────────────────────────────────────────────────────────

def make_segmented_batch(batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate sequences with N_SEGMENTS segments separated by BOUNDARY_TOKEN.
    Each segment has one key-value pair.
    Returns:
      seq:      (B, SEQ_LEN)
      targets:  (B, N_SEGMENTS) — target values for each segment's key
      seg_mask: (B, N_SEGMENTS, SEQ_LEN) — which positions belong to each segment
    """
    seq      = torch.zeros(batch_size, SEQ_LEN, dtype=torch.long)
    targets  = torch.zeros(batch_size, N_SEGMENTS, dtype=torch.long)
    seg_mask = torch.zeros(batch_size, N_SEGMENTS, SEQ_LEN, dtype=torch.float)

    # Layout: split SEQ_LEN into N_SEGMENTS+1 parts to leave room for query at end
    # SEQ_LEN=24, N_SEGMENTS=3 -> ~7 tokens per segment, last 3 for query
    seg_len   = (SEQ_LEN - 3) // N_SEGMENTS   # ~7
    query_pos = SEQ_LEN - 3

    for b in range(batch_size):
        seg_keys = torch.randint(4, 4 + N_SEGMENTS * 4, (N_SEGMENTS,))
        seg_vals = torch.randint(32, VOCAB_SIZE, (N_SEGMENTS,))

        for s in range(N_SEGMENTS):
            start = s * seg_len
            end   = start + seg_len
            seq[b, start] = BOUNDARY_TOKEN         # segment boundary marker
            seg_mask[b, s, start:end] = 1.0

            # Place key-value pair 1 token after boundary
            if start + 2 < end:
                seq[b, start + 1] = seg_keys[s]
                seq[b, start + 2] = seg_vals[s]

            # Fill rest of segment with filler
            for p in range(start + 3, min(end, SEQ_LEN - 3)):
                seq[b, p] = 3

            targets[b, s] = seg_vals[s]

        # Query section: pick a random segment to query
        query_seg = torch.randint(0, N_SEGMENTS, (1,)).item()
        seq[b, query_pos]     = 2                     # query marker
        seq[b, query_pos + 1] = seg_keys[query_seg]
        seq[b, query_pos + 2] = 0                     # masked target

        # Override target to be the queried segment's value for main loss
        # (store all segment targets separately for per-seg eval)

    return seq, targets, seg_mask


# ── Encoder ───────────────────────────────────────────────────────────────────

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


class ReadHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.out    = nn.Linear(HIDDEN_DIM, VOCAB_SIZE)

    def forward(self, query_h: torch.Tensor, memory: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        q      = self.q_proj(query_h)                               # (B, H)
        scores = torch.bmm(memory, q.unsqueeze(-1)).squeeze(-1)     # (B, S)
        scores = scores.masked_fill(mask == 0, -1e9)
        attn   = torch.softmax(scores, dim=-1)                      # (B, S)
        ctx    = (attn.unsqueeze(-1) * memory).sum(1)               # (B, H)
        return self.out(ctx)


# ── Write Policies ─────────────────────────────────────────────────────────────

def fixed_interval_write(hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Write every FIXED_STRIDE tokens. Returns (memory, mask)."""
    B, L, H = hidden.shape
    indices  = list(range(0, L, FIXED_STRIDE))[:MEMORY_SLOTS]
    idx_t    = torch.tensor(indices, device=hidden.device)
    memory   = hidden[:, idx_t, :]                     # (B, k, H)
    mask     = torch.ones(B, len(indices), device=hidden.device)
    if len(indices) < MEMORY_SLOTS:
        pad  = MEMORY_SLOTS - len(indices)
        memory = F.pad(memory, (0, 0, 0, pad))
        mask   = F.pad(mask, (0, pad))
    return memory, mask


def boundary_triggered_write(
    hidden: torch.Tensor, seq: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Write tokens immediately after each BOUNDARY_TOKEN, plus BOUNDARY_WINDOW more.
    Falls back to fixed interval if too few boundary-triggered slots.
    Returns (memory, mask).
    """
    B, L, H  = hidden.shape
    slots    = MEMORY_SLOTS
    all_mem  = []
    all_mask = []

    for b in range(B):
        trigger_positions = []
        for pos in range(L):
            if seq[b, pos].item() == BOUNDARY_TOKEN:
                # Write pos+1 .. pos+1+BOUNDARY_WINDOW
                for w in range(1, BOUNDARY_WINDOW + 2):
                    p = pos + w
                    if p < L:
                        trigger_positions.append(p)

        # Deduplicate and trim to budget
        seen = []
        for p in trigger_positions:
            if p not in seen:
                seen.append(p)
        trigger_positions = seen[:slots]

        # If we have fewer than slots, pad with fixed-interval positions
        if len(trigger_positions) < slots:
            fixed = list(range(0, L, FIXED_STRIDE))
            for p in fixed:
                if p not in trigger_positions:
                    trigger_positions.append(p)
                if len(trigger_positions) >= slots:
                    break
        trigger_positions = trigger_positions[:slots]

        idx_t  = torch.tensor(trigger_positions, device=hidden.device)
        mem_b  = hidden[b, idx_t, :]                      # (k, H)
        msk_b  = torch.ones(len(trigger_positions), device=hidden.device)

        if len(trigger_positions) < slots:
            pad   = slots - len(trigger_positions)
            mem_b = F.pad(mem_b, (0, 0, 0, pad))
            msk_b = F.pad(msk_b, (0, pad))

        all_mem.append(mem_b)
        all_mask.append(msk_b)

    memory = torch.stack(all_mem,  dim=0)   # (B, slots, H)
    mask   = torch.stack(all_mask, dim=0)   # (B, slots)
    return memory, mask


# ── Per-Segment Accuracy ───────────────────────────────────────────────────────

def eval_per_segment_acc(
    enc: nn.Module, head: nn.Module, policy: str, n_batches: int = 20
) -> tuple[float, list[float]]:
    """
    Evaluate per-segment retrieval accuracy.
    For each segment, construct a query batch that specifically queries that segment.
    Returns (mean_acc, [acc_seg0, acc_seg1, acc_seg2]).
    """
    seg_accs = []
    for seg_idx in range(N_SEGMENTS):
        total_acc = 0.0
        for _ in range(n_batches):
            # Build a batch specifically querying seg_idx
            seq_raw = torch.zeros(BATCH_SIZE, SEQ_LEN, dtype=torch.long)
            target  = torch.zeros(BATCH_SIZE, dtype=torch.long)
            seg_len = (SEQ_LEN - 3) // N_SEGMENTS

            for b in range(BATCH_SIZE):
                seg_keys = torch.randint(4, 4 + N_SEGMENTS * 4, (N_SEGMENTS,))
                seg_vals = torch.randint(32, VOCAB_SIZE, (N_SEGMENTS,))
                for s in range(N_SEGMENTS):
                    start = s * seg_len
                    end   = start + seg_len
                    seq_raw[b, start] = BOUNDARY_TOKEN
                    if start + 2 < end:
                        seq_raw[b, start + 1] = seg_keys[s]
                        seq_raw[b, start + 2] = seg_vals[s]
                    for p in range(start + 3, min(end, SEQ_LEN - 3)):
                        seq_raw[b, p] = 3
                query_pos = SEQ_LEN - 3
                seq_raw[b, query_pos]     = 2
                seq_raw[b, query_pos + 1] = seg_keys[seg_idx]
                seq_raw[b, query_pos + 2] = 0
                target[b] = seg_vals[seg_idx]

            seq_raw = seq_raw.to(DEVICE)
            target  = target.to(DEVICE)

            with torch.no_grad():
                hidden = enc(seq_raw)
                if policy == "fixed":
                    memory, mask = fixed_interval_write(hidden)
                else:
                    memory, mask = boundary_triggered_write(hidden, seq_raw)
                query_h = hidden[:, -1, :]
                logits  = head(query_h, memory, mask)
                preds   = logits.argmax(dim=-1)
                total_acc += (preds == target).float().mean().item()

        seg_accs.append(total_acc / n_batches)

    mean_acc = sum(seg_accs) / len(seg_accs)
    return mean_acc, seg_accs


# ── Training ───────────────────────────────────────────────────────────────────

def train_policy(policy: str) -> tuple[nn.Module, nn.Module, list]:
    """Train model using specified policy. Returns (enc, head, loss_log)."""
    enc  = Encoder().to(DEVICE)
    head = ReadHead().to(DEVICE)
    opt  = Adam(list(enc.parameters()) + list(head.parameters()), lr=LR)

    loss_log = []

    for step in range(TRAIN_STEPS):
        seq, targets, seg_mask = make_segmented_batch(BATCH_SIZE)
        seq     = seq.to(DEVICE)
        targets = targets.to(DEVICE)

        # Query the last segment for training signal
        query_seg = N_SEGMENTS - 1
        target    = targets[:, query_seg]

        hidden = enc(seq)
        if policy == "fixed":
            memory, mask = fixed_interval_write(hidden)
        else:
            memory, mask = boundary_triggered_write(hidden, seq)

        query_h   = hidden[:, -1, :]
        logits    = head(query_h, memory, mask)
        task_loss = F.cross_entropy(logits, target)

        opt.zero_grad()
        task_loss.backward()
        opt.step()

        if step % LOG_EVERY == 0:
            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                acc   = (preds == target).float().mean().item()
                loss_log.append(task_loss.item())
                print(f"  [{policy:8s}] step={step:4d}  loss={task_loss.item():.3f}  acc={acc:.3f}")

    return enc, head, loss_log


# ── Experiment ─────────────────────────────────────────────────────────────────

class Exp34BoundaryDetectionWriteTrigger(Experiment):
    experiment_id = "exp_3_4"
    hypothesis = (
        "Semantic-boundary-triggered writing outperforms fixed-interval writing "
        "on long-document tasks with clear topical structure."
    )

    def run(self) -> ExperimentResult:
        torch.manual_seed(42)

        print("\nTraining fixed-interval policy...")
        enc_fixed, head_fixed, _ = train_policy("fixed")

        print("\nTraining boundary-triggered policy...")
        enc_boundary, head_boundary, _ = train_policy("boundary")

        print("\nEvaluating per-segment accuracy...")
        fixed_mean, fixed_per_seg      = eval_per_segment_acc(enc_fixed,    head_fixed,    "fixed")
        boundary_mean, boundary_per_seg = eval_per_segment_acc(enc_boundary, head_boundary, "boundary")

        print(f"\nFixed:    mean={fixed_mean:.3f}  per_seg={[round(x,3) for x in fixed_per_seg]}")
        print(f"Boundary: mean={boundary_mean:.3f}  per_seg={[round(x,3) for x in boundary_per_seg]}")

        gap = boundary_mean - fixed_mean
        # Segment balance: lower std = more balanced coverage
        import statistics
        boundary_balance = 1.0 - statistics.stdev(boundary_per_seg) if len(boundary_per_seg) > 1 else 1.0
        fixed_balance    = 1.0 - statistics.stdev(fixed_per_seg)    if len(fixed_per_seg)    > 1 else 1.0

        if boundary_mean > fixed_mean + 0.03:
            outcome = OUTCOME_SUPPORTED
        elif fixed_mean >= boundary_mean:
            outcome = OUTCOME_REFUTED
        else:
            outcome = OUTCOME_INCONCLUSIVE

        metrics = {
            "boundary_acc":             round(boundary_mean, 4),
            "fixed_acc":                round(fixed_mean, 4),
            "boundary_per_segment_acc": [round(x, 4) for x in boundary_per_seg],
            "fixed_per_segment_acc":    [round(x, 4) for x in fixed_per_seg],
            "segment_balance_score":    round(boundary_balance, 4),
            "acc_gap":                  round(gap, 4),
        }
        notes = (
            f"Boundary vs fixed accuracy gap: {gap:+.3f}. "
            f"Boundary per-segment: {[round(x,3) for x in boundary_per_seg]}. "
            f"Fixed per-segment: {[round(x,3) for x in fixed_per_seg]}. "
            f"Boundary balance score: {boundary_balance:.3f}."
        )
        config = {
            "vocab_size":       VOCAB_SIZE,
            "seq_len":          SEQ_LEN,
            "hidden_dim":       HIDDEN_DIM,
            "memory_slots":     MEMORY_SLOTS,
            "batch_size":       BATCH_SIZE,
            "train_steps":      TRAIN_STEPS,
            "n_segments":       N_SEGMENTS,
            "fixed_stride":     FIXED_STRIDE,
            "boundary_window":  BOUNDARY_WINDOW,
        }
        return self.result(outcome, metrics, notes, config)


if __name__ == "__main__":
    Exp34BoundaryDetectionWriteTrigger().execute()
