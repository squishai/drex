"""
Experiment 3.3 — Write Timing vs Content Depth

Hypothesis: Writing later in context (more processed representations) outperforms
early writing (raw representations) for inferential tasks.

Setup:
  - MEMORY_SLOTS=6, 2-layer transformer
  - Policy A (early write): write after layer 1
  - Policy B (late write): write after layer 2
  - Both store the same positional tokens (top-k by attention entropy)
  - Task: sequences encode a rule + examples, question requires applying the rule
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
N_HEADS       = 4

# Inferential task: rule token + examples
RULE_OFFSET   = 2    # rule tokens: 2..5  (4 possible rules)
EXAMPLE_START = 6    # example input tokens
N_RULES       = 4
N_EXAMPLES    = 3    # number of (input, output) example pairs per sequence


# ── Inferential Task Data ──────────────────────────────────────────────────────

def make_inferential_batch(batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate sequences of the form:
      [rule_token, ex_in1, ex_out1, ex_in2, ex_out2, ex_in3, ex_out3, <filler>, query_in] -> query_out
    The rule defines a simple mapping: out = (in + rule_offset) % N_OUTPUT_VALS
    where rule_offset is 1..N_RULES and N_OUTPUT_VALS = 8.

    rule_token   : 2..2+N_RULES
    example inputs: 10..17 (N_OUTPUT_VALS=8 distinct)
    example outputs: 20..27
    query input  : same range as example inputs
    """
    N_OUTPUT_VALS = 8
    seq    = torch.zeros(batch_size, SEQ_LEN, dtype=torch.long)
    target = torch.zeros(batch_size, dtype=torch.long)

    for b in range(batch_size):
        rule_id     = torch.randint(0, N_RULES, (1,)).item()          # 0..3
        rule_offset = rule_id + 1                                       # 1..4
        rule_token  = RULE_OFFSET + rule_id                             # 2..5

        seq[b, 0] = rule_token

        pos = 1
        example_inputs = torch.randint(0, N_OUTPUT_VALS, (N_EXAMPLES,))
        for i in range(N_EXAMPLES):
            in_tok  = 10 + example_inputs[i].item()
            out_tok = 20 + (example_inputs[i].item() + rule_offset) % N_OUTPUT_VALS
            if pos + 1 < SEQ_LEN - 2:
                seq[b, pos]     = in_tok
                seq[b, pos + 1] = out_tok
                pos += 2

        # Filler
        for p in range(pos, SEQ_LEN - 2):
            seq[b, p] = 3  # filler token

        # Query
        query_in   = torch.randint(0, N_OUTPUT_VALS, (1,)).item()
        query_out  = (query_in + rule_offset) % N_OUTPUT_VALS
        seq[b, SEQ_LEN - 2] = 10 + query_in
        seq[b, SEQ_LEN - 1] = 0   # mask
        target[b] = 20 + query_out

    return seq, target


# ── Transformer Blocks ─────────────────────────────────────────────────────────

class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn  = nn.MultiheadAttention(HIDDEN_DIM, N_HEADS, batch_first=True)
        self.ff    = nn.Sequential(
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM * 2), nn.ReLU(),
            nn.Linear(HIDDEN_DIM * 2, HIDDEN_DIM),
        )
        self.norm1 = nn.LayerNorm(HIDDEN_DIM)
        self.norm2 = nn.LayerNorm(HIDDEN_DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ff(x))
        return x


class TwoLayerTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed  = nn.Embedding(VOCAB_SIZE, HIDDEN_DIM)
        self.pos    = nn.Embedding(SEQ_LEN, HIDDEN_DIM)
        self.layer1 = TransformerBlock()
        self.layer2 = TransformerBlock()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (h1, h2): hidden states after layer 1 and layer 2."""
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        h = self.embed(x) + self.pos(positions)
        h1 = self.layer1(h)
        h2 = self.layer2(h1)
        return h1, h2


class MemoryReadHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.out    = nn.Linear(HIDDEN_DIM, VOCAB_SIZE)

    def forward(self, query_h: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        """
        query_h: (B, H) — last token representation
        memory:  (B, MEMORY_SLOTS, H)
        """
        q      = self.q_proj(query_h).unsqueeze(1)          # (B, 1, H)
        scores = torch.bmm(q, memory.transpose(1, 2))        # (B, 1, S)
        attn   = torch.softmax(scores, dim=-1)               # (B, 1, S)
        ctx    = torch.bmm(attn, memory).squeeze(1)          # (B, H)
        return self.out(ctx)


def select_top_k_tokens(hidden: torch.Tensor, k: int) -> torch.Tensor:
    """Select top-k tokens by L2 norm (a proxy for information content)."""
    norms    = hidden.norm(dim=-1)        # (B, L)
    _, topk  = torch.topk(norms, k, dim=1)  # (B, k)
    expanded = topk.unsqueeze(-1).expand(-1, -1, hidden.size(-1))
    return torch.gather(hidden, 1, expanded)  # (B, k, H)


# ── Training ───────────────────────────────────────────────────────────────────

def train_write_timing(write_layer: str) -> dict:
    """
    Train model with write_layer = 'early' (after layer 1) or 'late' (after layer 2).
    """
    transformer = TwoLayerTransformer().to(DEVICE)
    read_head   = MemoryReadHead().to(DEVICE)

    params = list(transformer.parameters()) + list(read_head.parameters())
    opt    = Adam(params, lr=LR)

    acc_log  = []
    loss_log = []

    for step in range(TRAIN_STEPS):
        seq, target = make_inferential_batch(BATCH_SIZE)
        seq    = seq.to(DEVICE)
        target = target.to(DEVICE)

        h1, h2 = transformer(seq)

        # Choose which representation to write to memory
        write_repr = h1 if write_layer == "early" else h2
        memory     = select_top_k_tokens(write_repr, MEMORY_SLOTS)   # (B, S, H)

        # Query is always the final-layer representation of the last token
        query_h = h2[:, -1, :]   # (B, H)

        logits    = read_head(query_h, memory)   # (B, VOCAB_SIZE)
        task_loss = F.cross_entropy(logits, target)

        opt.zero_grad()
        task_loss.backward()
        opt.step()

        if step % LOG_EVERY == 0:
            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                acc   = (preds == target).float().mean().item()
                acc_log.append(acc)
                loss_log.append(task_loss.item())
                print(f"  [{write_layer:5s}] step={step:4d}  loss={task_loss.item():.3f}  acc={acc:.3f}")

    # Final evaluation
    with torch.no_grad():
        total_acc  = 0.0
        total_loss = 0.0
        eval_steps = 20
        for _ in range(eval_steps):
            seq, target = make_inferential_batch(BATCH_SIZE)
            seq    = seq.to(DEVICE)
            target = target.to(DEVICE)
            h1, h2 = transformer(seq)
            write_repr = h1 if write_layer == "early" else h2
            memory     = select_top_k_tokens(write_repr, MEMORY_SLOTS)
            query_h    = h2[:, -1, :]
            logits     = read_head(query_h, memory)
            loss       = F.cross_entropy(logits, target)
            preds      = logits.argmax(dim=-1)
            total_acc  += (preds == target).float().mean().item()
            total_loss += loss.item()
        final_acc  = total_acc / eval_steps
        final_loss = total_loss / eval_steps

    return {
        "final_acc":  final_acc,
        "final_loss": final_loss,
        "acc_log":    acc_log,
        "loss_log":   loss_log,
    }


# ── Experiment ─────────────────────────────────────────────────────────────────

class Exp33WriteTimingVsContent(Experiment):
    experiment_id = "exp_3_3"
    hypothesis = (
        "Writing later in context (more processed representations) outperforms "
        "early writing (raw representations) for inferential tasks."
    )

    def run(self) -> ExperimentResult:
        torch.manual_seed(42)

        print("\nTraining early-write policy (layer 1 representations)...")
        early_res = train_write_timing("early")

        print("\nTraining late-write policy (layer 2 representations)...")
        late_res  = train_write_timing("late")

        early_write_acc   = early_res["final_acc"]
        late_write_acc    = late_res["final_acc"]
        early_final_loss  = early_res["final_loss"]
        late_final_loss   = late_res["final_loss"]

        gap = late_write_acc - early_write_acc
        print(f"\nEarly write: acc={early_write_acc:.3f}  loss={early_final_loss:.3f}")
        print(f"Late write:  acc={late_write_acc:.3f}  loss={late_final_loss:.3f}")
        print(f"Gap (late - early): {gap:+.3f}")

        if late_write_acc > early_write_acc + 0.02:
            outcome = OUTCOME_SUPPORTED
        elif early_write_acc >= late_write_acc:
            outcome = OUTCOME_REFUTED
        else:
            outcome = OUTCOME_INCONCLUSIVE

        metrics = {
            "early_write_acc":   round(early_write_acc, 4),
            "late_write_acc":    round(late_write_acc, 4),
            "early_final_loss":  round(early_final_loss, 4),
            "late_final_loss":   round(late_final_loss, 4),
            "acc_gap":           round(gap, 4),
        }
        notes = (
            f"Late vs early accuracy gap: {gap:+.3f}. "
            f"Threshold for SUPPORTED: >0.02. "
            f"Task: inferential rule application ({N_RULES} rules, {N_EXAMPLES} examples)."
        )
        config = {
            "vocab_size":    VOCAB_SIZE,
            "seq_len":       SEQ_LEN,
            "hidden_dim":    HIDDEN_DIM,
            "memory_slots":  MEMORY_SLOTS,
            "batch_size":    BATCH_SIZE,
            "train_steps":   TRAIN_STEPS,
            "n_heads":       N_HEADS,
            "n_rules":       N_RULES,
            "n_examples":    N_EXAMPLES,
        }
        return self.result(outcome, metrics, notes, config)


if __name__ == "__main__":
    Exp33WriteTimingVsContent().execute()
