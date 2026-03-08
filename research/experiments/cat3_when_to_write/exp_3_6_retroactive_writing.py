"""
Experiment 3.6 — Retroactive Writing

Hypothesis: A controller can learn to retroactively write tokens it initially
skipped once later context reveals their importance.

Setup:
  - Two-pass memory controller
  - Pass 1 (forward): process sequence, write gate scores tokens
  - Pass 2 (retroactive): after seeing full sequence, a "revision gate" can
    upgrade previously un-written tokens based on full context
  - Forward-only baseline vs two-pass model
  - Measure: retroactive additions and downstream accuracy
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

NUM_PAIRS     = 4
QUERY_MARKER  = 2
# Forward pass writes at most FORWARD_SLOTS tokens; retroactive pass can add up to
# MEMORY_SLOTS total.
FORWARD_SLOTS = 4


# ── Data Generation ────────────────────────────────────────────────────────────

def make_assoc_batch(batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Standard associative recall sequences."""
    seq    = torch.zeros(batch_size, SEQ_LEN, dtype=torch.long)
    target = torch.zeros(batch_size, dtype=torch.long)

    for b in range(batch_size):
        keys = torch.randint(4, 4 + NUM_PAIRS * 3, (NUM_PAIRS,)).unique()[:NUM_PAIRS]
        while len(keys) < NUM_PAIRS:
            keys = torch.cat([keys, torch.randint(4, 32, (1,))])[:NUM_PAIRS]
        vals = torch.randint(32, VOCAB_SIZE, (NUM_PAIRS,))

        pos = 0
        for i in range(NUM_PAIRS):
            if pos + 1 < SEQ_LEN - 3:
                seq[b, pos]     = keys[i]
                seq[b, pos + 1] = vals[i]
                pos += 2

        for p in range(pos, SEQ_LEN - 3):
            seq[b, p] = 3

        query_idx           = torch.randint(0, NUM_PAIRS, (1,)).item()
        seq[b, SEQ_LEN - 3] = QUERY_MARKER
        seq[b, SEQ_LEN - 2] = keys[query_idx]
        seq[b, SEQ_LEN - 1] = 0
        target[b]           = vals[query_idx]

    return seq, target


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


# ── Write Gates ────────────────────────────────────────────────────────────────

class ForwardGate(nn.Module):
    """Scores each token for initial write decision."""
    def __init__(self):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM // 2), nn.ReLU(),
            nn.Linear(HIDDEN_DIM // 2, 1),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Returns (B, L) gate scores in [0, 1]."""
        return torch.sigmoid(self.gate(hidden)).squeeze(-1)


class RevisionGate(nn.Module):
    """
    Re-scores skipped tokens given full-sequence context.
    Takes: token hidden state + summary of full sequence.
    """
    def __init__(self):
        super().__init__()
        # Summarize full sequence
        self.summary_proj = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        # Score each skipped token using its hidden + global summary
        self.gate = nn.Sequential(
            nn.Linear(HIDDEN_DIM * 2, HIDDEN_DIM), nn.ReLU(),
            nn.Linear(HIDDEN_DIM, 1),
        )

    def forward(
        self, hidden: torch.Tensor, skipped_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        hidden:       (B, L, H)
        skipped_mask: (B, L) — 1 where token was NOT written in forward pass
        Returns revision scores (B, L) in [0, 1] for skipped positions.
        """
        # Global summary: mean pool over all positions
        summary = self.summary_proj(hidden.mean(dim=1))   # (B, H)
        summary_exp = summary.unsqueeze(1).expand_as(hidden)  # (B, L, H)

        combined = torch.cat([hidden, summary_exp], dim=-1)   # (B, L, 2H)
        scores   = torch.sigmoid(self.gate(combined)).squeeze(-1)  # (B, L)

        # Zero out non-skipped positions (they're already written)
        return scores * skipped_mask


class ReadHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.out    = nn.Linear(HIDDEN_DIM, VOCAB_SIZE)

    def forward(
        self, query_h: torch.Tensor, memory: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        q      = self.q_proj(query_h)
        scores = torch.bmm(memory, q.unsqueeze(-1)).squeeze(-1)
        scores = scores.masked_fill(mask == 0, -1e9)
        attn   = torch.softmax(scores, dim=-1)
        ctx    = (attn.unsqueeze(-1) * memory).sum(1)
        return self.out(ctx)


# ── Memory Assembly ────────────────────────────────────────────────────────────

def assemble_memory_forward_only(
    hidden: torch.Tensor, gate_scores: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Top-FORWARD_SLOTS tokens by forward gate score.
    Pads to MEMORY_SLOTS with zeros.
    Returns (memory, mask): (B, MEMORY_SLOTS, H), (B, MEMORY_SLOTS).
    """
    B, L, H  = hidden.shape
    k        = FORWARD_SLOTS
    topk_val, topk_idx = torch.topk(gate_scores, k, dim=1)   # (B, k)
    idx_exp  = topk_idx.unsqueeze(-1).expand(-1, -1, H)
    memory   = torch.gather(hidden, 1, idx_exp)              # (B, k, H)
    mask     = torch.ones(B, k, device=hidden.device)

    pad      = MEMORY_SLOTS - k
    memory   = F.pad(memory, (0, 0, 0, pad))
    mask     = F.pad(mask, (0, pad))
    return memory, mask, topk_idx


def assemble_memory_two_pass(
    hidden: torch.Tensor,
    forward_gate: ForwardGate,
    revision_gate: RevisionGate,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """
    Two-pass assembly: forward selects FORWARD_SLOTS, revision upgrades up to
    MEMORY_SLOTS - FORWARD_SLOTS more from skipped tokens.
    Returns (memory, mask, retroactive_write_rate).
    """
    B, L, H = hidden.shape
    fwd_scores = forward_gate(hidden)   # (B, L)

    # Forward pass: select top FORWARD_SLOTS
    k_fwd = FORWARD_SLOTS
    _, fwd_idx = torch.topk(fwd_scores, k_fwd, dim=1)    # (B, k_fwd)

    # Build skipped mask: positions NOT in top-k forward
    written_mask = torch.zeros(B, L, device=hidden.device)
    written_mask.scatter_(1, fwd_idx, 1.0)
    skipped_mask = 1.0 - written_mask   # (B, L)

    # Revision pass: score skipped tokens
    k_retro = MEMORY_SLOTS - k_fwd
    rev_scores = revision_gate(hidden, skipped_mask)   # (B, L)
    _, rev_idx = torch.topk(rev_scores, k_retro, dim=1)   # (B, k_retro)

    # Assemble final memory: forward + retroactive
    all_mem  = []
    all_mask = []
    retro_rates = []

    for b in range(B):
        fwd_pos  = fwd_idx[b].tolist()
        rev_pos  = rev_idx[b].tolist()

        # Only count retroactive positions that were truly skipped
        retro_actual = [p for p in rev_pos if written_mask[b, p].item() < 0.5]
        retro_rates.append(len(retro_actual) / L)

        all_pos = fwd_pos + rev_pos
        idx_t   = torch.tensor(all_pos, device=hidden.device)
        mem_b   = hidden[b, idx_t, :]           # (MEMORY_SLOTS, H)
        msk_b   = torch.ones(MEMORY_SLOTS, device=hidden.device)
        all_mem.append(mem_b)
        all_mask.append(msk_b)

    memory = torch.stack(all_mem,  dim=0)
    mask   = torch.stack(all_mask, dim=0)
    retro_rate = sum(retro_rates) / len(retro_rates)
    return memory, mask, retro_rate


# ── Training ───────────────────────────────────────────────────────────────────

def train_forward_only() -> tuple[nn.Module, nn.Module, ForwardGate]:
    enc        = Encoder().to(DEVICE)
    fwd_gate   = ForwardGate().to(DEVICE)
    read_head  = ReadHead().to(DEVICE)
    opt        = Adam(
        list(enc.parameters()) + list(fwd_gate.parameters()) + list(read_head.parameters()),
        lr=LR,
    )

    for step in range(TRAIN_STEPS):
        seq, target = make_assoc_batch(BATCH_SIZE)
        seq    = seq.to(DEVICE)
        target = target.to(DEVICE)

        hidden     = enc(seq)
        gate_scores = fwd_gate(hidden)
        memory, mask, _ = assemble_memory_forward_only(hidden, gate_scores)

        query_h   = hidden[:, -1, :]
        logits    = read_head(query_h, memory, mask)
        task_loss = F.cross_entropy(logits, target)

        opt.zero_grad()
        task_loss.backward()
        opt.step()

        if step % LOG_EVERY == 0:
            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                acc   = (preds == target).float().mean().item()
                print(f"  [forward_only] step={step:4d}  loss={task_loss.item():.3f}  acc={acc:.3f}")

    return enc, read_head, fwd_gate


def train_two_pass() -> tuple[nn.Module, nn.Module, ForwardGate, RevisionGate]:
    enc          = Encoder().to(DEVICE)
    fwd_gate     = ForwardGate().to(DEVICE)
    rev_gate     = RevisionGate().to(DEVICE)
    read_head    = ReadHead().to(DEVICE)
    opt          = Adam(
        list(enc.parameters()) + list(fwd_gate.parameters()) +
        list(rev_gate.parameters()) + list(read_head.parameters()),
        lr=LR,
    )

    for step in range(TRAIN_STEPS):
        seq, target = make_assoc_batch(BATCH_SIZE)
        seq    = seq.to(DEVICE)
        target = target.to(DEVICE)

        hidden = enc(seq)
        memory, mask, _ = assemble_memory_two_pass(hidden, fwd_gate, rev_gate)

        query_h   = hidden[:, -1, :]
        logits    = read_head(query_h, memory, mask)
        task_loss = F.cross_entropy(logits, target)

        opt.zero_grad()
        task_loss.backward()
        opt.step()

        if step % LOG_EVERY == 0:
            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                acc   = (preds == target).float().mean().item()
                print(f"  [two_pass    ] step={step:4d}  loss={task_loss.item():.3f}  acc={acc:.3f}")

    return enc, read_head, fwd_gate, rev_gate


def eval_model(enc, read_head, policy: str, fwd_gate=None, rev_gate=None) -> tuple[float, float]:
    """Returns (accuracy, retroactive_write_rate)."""
    total_acc  = 0.0
    total_retro = 0.0
    eval_steps  = 20

    with torch.no_grad():
        for _ in range(eval_steps):
            seq, target = make_assoc_batch(BATCH_SIZE)
            seq    = seq.to(DEVICE)
            target = target.to(DEVICE)
            hidden = enc(seq)

            if policy == "forward_only":
                gate_scores = fwd_gate(hidden)
                memory, mask, _ = assemble_memory_forward_only(hidden, gate_scores)
                retro = 0.0
            else:
                memory, mask, retro = assemble_memory_two_pass(hidden, fwd_gate, rev_gate)

            query_h = hidden[:, -1, :]
            logits  = read_head(query_h, memory, mask)
            preds   = logits.argmax(dim=-1)
            total_acc   += (preds == target).float().mean().item()
            total_retro += retro

    return total_acc / eval_steps, total_retro / eval_steps


# ── Experiment ─────────────────────────────────────────────────────────────────

class Exp36RetroactiveWriting(Experiment):
    experiment_id = "exp_3_6"
    hypothesis = (
        "A controller can learn to retroactively write tokens it initially skipped "
        "once later context reveals their importance."
    )

    def run(self) -> ExperimentResult:
        torch.manual_seed(42)

        print("\nTraining forward-only model...")
        enc_fwd, head_fwd, fwd_gate_fwd = train_forward_only()

        print("\nTraining two-pass model...")
        enc_two, head_two, fwd_gate_two, rev_gate_two = train_two_pass()

        print("\nEvaluating...")
        forward_acc, _          = eval_model(enc_fwd, head_fwd, "forward_only", fwd_gate=fwd_gate_fwd)
        two_pass_acc, retro_rate = eval_model(enc_two, head_two, "two_pass", fwd_gate=fwd_gate_two, rev_gate=rev_gate_two)

        gap = two_pass_acc - forward_acc
        print(f"\nForward-only acc: {forward_acc:.3f}")
        print(f"Two-pass acc:     {two_pass_acc:.3f}  (gap={gap:+.3f})")
        print(f"Retroactive write rate: {retro_rate:.3f}")

        if two_pass_acc > forward_acc + 0.02:
            outcome = OUTCOME_SUPPORTED
        elif forward_acc >= two_pass_acc:
            outcome = OUTCOME_REFUTED
        else:
            outcome = OUTCOME_INCONCLUSIVE

        metrics = {
            "forward_acc":          round(forward_acc, 4),
            "two_pass_acc":         round(two_pass_acc, 4),
            "acc_gap":              round(gap, 4),
            "retroactive_write_rate": round(retro_rate, 4),
        }
        notes = (
            f"Two-pass vs forward-only accuracy gap: {gap:+.3f}. "
            f"Retroactive write rate (fraction of tokens upgraded): {retro_rate:.3f}. "
            f"Forward pass writes {FORWARD_SLOTS} slots; revision adds up to {MEMORY_SLOTS - FORWARD_SLOTS} more."
        )
        config = {
            "vocab_size":    VOCAB_SIZE,
            "seq_len":       SEQ_LEN,
            "hidden_dim":    HIDDEN_DIM,
            "memory_slots":  MEMORY_SLOTS,
            "forward_slots": FORWARD_SLOTS,
            "batch_size":    BATCH_SIZE,
            "train_steps":   TRAIN_STEPS,
            "num_pairs":     NUM_PAIRS,
        }
        return self.result(outcome, metrics, notes, config)


if __name__ == "__main__":
    Exp36RetroactiveWriting().execute()
