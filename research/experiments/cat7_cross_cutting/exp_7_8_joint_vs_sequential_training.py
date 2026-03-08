"""
Experiment 7.8 — Joint vs Curriculum (Sequential) Training

Hypothesis: Curriculum training (one controller component at a time) produces
more stable controller behavior than joint training from the start.

Setup:
  - Full controller with 3 components: write gate, compressor, read gate.
  - Policy A (joint): train all three simultaneously from step 0.
  - Policy B (curriculum):
      steps    0-500:  write gate only (compressor + read gate frozen)
      steps  500-1000: write gate + compressor (read gate frozen)
      steps 1000-1500: all three trained jointly
  - Measure: final accuracy, training stability (loss variance last 200 steps),
    gate collapse (write or read gate collapses to near-uniform output).
  - SUPPORTED if curriculum_acc >= joint_acc AND
    curriculum_loss_variance < joint_loss_variance.
  - REFUTED if joint wins on both.
  - INCONCLUSIVE otherwise.
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

VOCAB_SIZE     = 64
HIDDEN_DIM     = 64
COMPRESS_DIM   = 32
SEQ_LEN        = 24
MEMORY_SLOTS   = 8
BATCH_SIZE     = 32
TRAIN_STEPS    = 1500     # total steps for both policies
EVAL_BATCHES   = 200
STABILITY_WINDOW = 200    # last N steps for loss variance
LR             = 3e-4
DEVICE         = "cpu"

# Curriculum schedule boundaries
PHASE1_END   = 500
PHASE2_END   = 1000
# PHASE3_END = TRAIN_STEPS (1500)

# Gate collapse: std of sigmoid(gate logits) < threshold considered collapsed
COLLAPSE_THRESHOLD = 0.02


# ── Models ────────────────────────────────────────────────────────────────────

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, HIDDEN_DIM)
        self.ff    = nn.Sequential(
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM * 2),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM * 2, HIDDEN_DIM),
        )
        self.norm = nn.LayerNorm(HIDDEN_DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embed(x)
        return self.norm(h + self.ff(h))


class WriteGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, 1),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.net(hidden).squeeze(-1)   # (B, L)

    def gate_std(self, hidden: torch.Tensor) -> float:
        with torch.no_grad():
            return torch.sigmoid(self.forward(hidden)).std().item()


class Compressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.encode = nn.Linear(HIDDEN_DIM, COMPRESS_DIM)
        self.decode = nn.Linear(COMPRESS_DIM, HIDDEN_DIM)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.decode(F.relu(self.encode(h)))


class ReadGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj  = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.out     = nn.Linear(HIDDEN_DIM, VOCAB_SIZE)
        self.query_e = nn.Embedding(VOCAB_SIZE, HIDDEN_DIM)

    def forward(self, query: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        q    = self.q_proj(self.query_e(query)).unsqueeze(1)
        sims = (q * memory).sum(-1) / (HIDDEN_DIM ** 0.5)
        w    = F.softmax(sims, dim=-1).unsqueeze(-1)
        return self.out((w * memory).sum(1))

    def attention_std(self, query: torch.Tensor, memory: torch.Tensor) -> float:
        with torch.no_grad():
            q    = self.q_proj(self.query_e(query)).unsqueeze(1)
            sims = (q * memory).sum(-1) / (HIDDEN_DIM ** 0.5)
            w    = F.softmax(sims, dim=-1)
        return w.std().item()


# ── Data ──────────────────────────────────────────────────────────────────────

def make_batch(batch_size: int):
    seq   = torch.randint(0, VOCAB_SIZE, (batch_size, SEQ_LEN))
    k_pos = torch.randint(0, SEQ_LEN // 2, (batch_size,))
    v_pos = torch.randint(SEQ_LEN // 2, SEQ_LEN, (batch_size,))
    keys  = torch.randint(0, VOCAB_SIZE // 2, (batch_size,))
    vals  = torch.randint(VOCAB_SIZE // 2, VOCAB_SIZE, (batch_size,))
    for b in range(batch_size):
        seq[b, k_pos[b]] = keys[b]
        seq[b, v_pos[b]] = vals[b]
    return seq, keys, vals


# ── Forward pass ─────────────────────────────────────────────────────────────

def forward_step(
    enc: Encoder,
    write: WriteGate,
    comp: Compressor,
    read: ReadGate,
    seq: torch.Tensor,
    keys: torch.Tensor,
    vals: torch.Tensor,
) -> torch.Tensor:
    h       = enc(seq)
    logits  = write(h)
    top_idx = logits.topk(MEMORY_SLOTS, dim=-1).indices
    h_sel   = h.gather(1, top_idx.unsqueeze(-1).expand(-1, -1, HIDDEN_DIM))
    h_comp  = comp(h_sel)
    out     = read(keys, h_comp)
    return F.cross_entropy(out, vals)


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(enc: Encoder, write: WriteGate, comp: Compressor,
             read: ReadGate) -> float:
    enc.eval(); write.eval(); comp.eval(); read.eval()
    correct = total = 0
    with torch.no_grad():
        for _ in range(EVAL_BATCHES):
            seq, keys, vals = make_batch(BATCH_SIZE)
            h       = enc(seq)
            top_idx = write(h).topk(MEMORY_SLOTS, dim=-1).indices
            h_sel   = h.gather(1, top_idx.unsqueeze(-1).expand(-1, -1, HIDDEN_DIM))
            h_comp  = comp(h_sel)
            out     = read(keys, h_comp)
            correct += (out.argmax(-1) == vals).sum().item()
            total   += BATCH_SIZE
    enc.train(); write.train(); comp.train(); read.train()
    return correct / total


def detect_collapse(
    enc: Encoder, write: WriteGate, read: ReadGate,
) -> tuple[bool, bool]:
    """Return (write_collapsed, read_collapsed)."""
    enc.eval(); write.eval(); read.eval()
    with torch.no_grad():
        seq, keys, _ = make_batch(BATCH_SIZE)
        h = enc(seq)
        wg_std = torch.sigmoid(write(h)).std().item()

        # For read gate: build random memory and measure attention entropy
        top_idx = write(h).topk(MEMORY_SLOTS, dim=-1).indices
        h_sel   = h.gather(1, top_idx.unsqueeze(-1).expand(-1, -1, HIDDEN_DIM))
        q       = read.q_proj(read.query_e(keys)).unsqueeze(1)
        sims    = (q * h_sel).sum(-1) / (HIDDEN_DIM ** 0.5)
        w       = F.softmax(sims, dim=-1)
        rg_std  = w.std().item()

    enc.train(); write.train(); read.train()
    write_collapsed = wg_std < COLLAPSE_THRESHOLD
    read_collapsed  = rg_std < COLLAPSE_THRESHOLD
    return write_collapsed, read_collapsed


# ── Policy A: Joint training ──────────────────────────────────────────────────

def train_joint() -> dict:
    torch.manual_seed(42)
    enc   = Encoder()
    write = WriteGate()
    comp  = Compressor()
    read  = ReadGate()

    all_params = (
        list(enc.parameters())
        + list(write.parameters())
        + list(comp.parameters())
        + list(read.parameters())
    )
    opt = Adam(all_params, lr=LR)

    losses: list[float] = []
    for step in range(TRAIN_STEPS):
        seq, keys, vals = make_batch(BATCH_SIZE)
        loss = forward_step(enc, write, comp, read, seq, keys, vals)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(all_params, 1.0)
        opt.step()
        losses.append(loss.item())

    acc          = evaluate(enc, write, comp, read)
    loss_var     = torch.tensor(losses[-STABILITY_WINDOW:]).var().item()
    wc, rc       = detect_collapse(enc, write, read)

    print(f"  Joint:      acc={acc:.3f}  loss_var={loss_var:.4f}  "
          f"write_collapse={wc}  read_collapse={rc}")
    return {
        "acc":              acc,
        "loss_variance":    loss_var,
        "write_collapsed":  wc,
        "read_collapsed":   rc,
        "gate_collapse":    wc or rc,
    }


# ── Policy B: Curriculum training ─────────────────────────────────────────────

def train_curriculum() -> dict:
    torch.manual_seed(42)
    enc   = Encoder()
    write = WriteGate()
    comp  = Compressor()
    read  = ReadGate()

    losses: list[float] = []

    for step in range(TRAIN_STEPS):
        # Determine which components are active this step
        if step < PHASE1_END:
            # Phase 1: write gate only
            active_params = list(enc.parameters()) + list(write.parameters())
            for m in (comp, read):
                for p in m.parameters():
                    p.requires_grad_(False)
            for p in write.parameters():
                p.requires_grad_(True)

        elif step < PHASE2_END:
            # Phase 2: write gate + compressor
            active_params = (
                list(enc.parameters())
                + list(write.parameters())
                + list(comp.parameters())
            )
            for p in comp.parameters():
                p.requires_grad_(True)
            for p in read.parameters():
                p.requires_grad_(False)

        else:
            # Phase 3: all components
            active_params = (
                list(enc.parameters())
                + list(write.parameters())
                + list(comp.parameters())
                + list(read.parameters())
            )
            for m in (write, comp, read):
                for p in m.parameters():
                    p.requires_grad_(True)

        opt = Adam(active_params, lr=LR)
        seq, keys, vals = make_batch(BATCH_SIZE)
        loss = forward_step(enc, write, comp, read, seq, keys, vals)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(active_params, 1.0)
        opt.step()
        losses.append(loss.item())

    # Re-enable all params for evaluation
    for m in (enc, write, comp, read):
        for p in m.parameters():
            p.requires_grad_(True)

    acc       = evaluate(enc, write, comp, read)
    loss_var  = torch.tensor(losses[-STABILITY_WINDOW:]).var().item()
    wc, rc    = detect_collapse(enc, write, read)

    print(f"  Curriculum: acc={acc:.3f}  loss_var={loss_var:.4f}  "
          f"write_collapse={wc}  read_collapse={rc}")
    return {
        "acc":              acc,
        "loss_variance":    loss_var,
        "write_collapsed":  wc,
        "read_collapsed":   rc,
        "gate_collapse":    wc or rc,
    }


# ── Experiment ────────────────────────────────────────────────────────────────

class Exp78JointVsSequentialTraining(Experiment):
    experiment_id = "exp_7_8"
    hypothesis = (
        "Curriculum training (one controller component at a time) produces "
        "more stable controller behavior than joint training from the start."
    )

    def run(self) -> ExperimentResult:
        torch.manual_seed(42)

        print("  Running Policy A (joint)...")
        joint      = train_joint()
        print("  Running Policy B (curriculum)...")
        curriculum = train_curriculum()

        curriculum_wins_acc  = curriculum["acc"]          >= joint["acc"]
        curriculum_more_stable = curriculum["loss_variance"] < joint["loss_variance"]
        joint_wins_both = (
            joint["acc"] > curriculum["acc"]
            and joint["loss_variance"] < curriculum["loss_variance"]
        )

        if curriculum_wins_acc and curriculum_more_stable:
            outcome = OUTCOME_SUPPORTED
        elif joint_wins_both:
            outcome = OUTCOME_REFUTED
        else:
            outcome = OUTCOME_INCONCLUSIVE

        metrics = {
            "joint_acc":                  joint["acc"],
            "curriculum_acc":             curriculum["acc"],
            "joint_loss_variance":        joint["loss_variance"],
            "curriculum_loss_variance":   curriculum["loss_variance"],
            "joint_gate_collapse":        joint["gate_collapse"],
            "curriculum_gate_collapse":   curriculum["gate_collapse"],
            "joint_write_collapsed":      joint["write_collapsed"],
            "joint_read_collapsed":       joint["read_collapsed"],
            "curriculum_write_collapsed": curriculum["write_collapsed"],
            "curriculum_read_collapsed":  curriculum["read_collapsed"],
        }
        notes = (
            f"Curriculum acc >= joint acc: {curriculum_wins_acc}. "
            f"Curriculum loss_var < joint loss_var: {curriculum_more_stable}. "
            f"Joint acc={joint['acc']:.3f}, Curriculum acc={curriculum['acc']:.3f}. "
            f"Joint var={joint['loss_variance']:.5f}, "
            f"Curriculum var={curriculum['loss_variance']:.5f}."
        )
        return self.result(outcome, metrics, notes, config={
            "train_steps":       TRAIN_STEPS,
            "phase1_end":        PHASE1_END,
            "phase2_end":        PHASE2_END,
            "memory_slots":      MEMORY_SLOTS,
            "stability_window":  STABILITY_WINDOW,
            "collapse_threshold": COLLAPSE_THRESHOLD,
        })


if __name__ == "__main__":
    Exp78JointVsSequentialTraining().execute()
