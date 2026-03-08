"""
Experiment 1.5 — Write Signal Ablation

Hypothesis: A learned write gate outperforms random write, attention-weighted
write, and surprise-driven write on associative recall tasks.

Setup:
  - Associative recall task: sequences of (key, value) token pairs
  - 4 write policies compete with identical storage budgets and retrieval
  - Measure downstream retrieval accuracy for each policy
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

VOCAB_SIZE   = 64
SEQ_LEN      = 32
HIDDEN_DIM   = 64
NUM_PAIRS    = 8          # key-value pairs per sequence
MEMORY_SLOTS = 8          # fixed storage budget for all policies
TRAIN_STEPS  = 2000
EVAL_STEPS   = 500
BATCH_SIZE   = 32
LR           = 3e-4
DEVICE       = "cpu"

# ── Task: Associative Recall ──────────────────────────────────────────────────

def make_batch(batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
        seqs:    (B, SEQ_LEN) int64 token ids
        queries: (B,)         int64 query key ids
        targets: (B,)         int64 target value ids
    """
    seqs    = torch.randint(2, VOCAB_SIZE, (batch_size, SEQ_LEN))
    queries = torch.zeros(batch_size, dtype=torch.long)
    targets = torch.zeros(batch_size, dtype=torch.long)

    for b in range(batch_size):
        # scatter NUM_PAIRS key-value pairs into sequence
        positions = torch.randperm(SEQ_LEN - 1)[:NUM_PAIRS * 2]
        for i in range(NUM_PAIRS):
            k_pos = positions[i * 2].item()
            v_pos = positions[i * 2 + 1].item()
            key   = torch.randint(2, VOCAB_SIZE // 2, (1,)).item()
            val   = torch.randint(VOCAB_SIZE // 2, VOCAB_SIZE, (1,)).item()
            seqs[b, k_pos] = key
            seqs[b, v_pos] = val

        # pick one pair as the query
        idx = torch.randint(NUM_PAIRS, (1,)).item()
        queries[b] = seqs[b, positions[idx * 2].item()]
        targets[b] = seqs[b, positions[idx * 2 + 1].item()]

    return seqs, queries, targets


# ── Tiny Encoder ─────────────────────────────────────────────────────────────

class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, HIDDEN_DIM)
        self.attn  = nn.MultiheadAttention(HIDDEN_DIM, 2, batch_first=True)
        self.ff    = nn.Sequential(nn.Linear(HIDDEN_DIM, HIDDEN_DIM * 2), nn.ReLU(),
                                    nn.Linear(HIDDEN_DIM * 2, HIDDEN_DIM))
        self.norm1 = nn.LayerNorm(HIDDEN_DIM)
        self.norm2 = nn.LayerNorm(HIDDEN_DIM)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns hidden states (B,L,H) and attention weights (B,L,L)."""
        h = self.embed(x)
        attn_out, attn_w = self.attn(h, h, h, need_weights=True)
        h = self.norm1(h + attn_out)
        h = self.norm2(h + self.ff(h))
        return h, attn_w


# ── Write Policies ────────────────────────────────────────────────────────────

def policy_random(hidden: torch.Tensor, attn: torch.Tensor,
                  loss_per_token: torch.Tensor | None) -> torch.Tensor:
    """Write the first MEMORY_SLOTS tokens."""
    B, L, H = hidden.shape
    mask = torch.zeros(B, L, device=hidden.device, dtype=torch.bool)
    mask[:, :MEMORY_SLOTS] = True
    return mask


def policy_attention(hidden: torch.Tensor, attn: torch.Tensor,
                     loss_per_token: torch.Tensor | None) -> torch.Tensor:
    """Write tokens with highest mean attention received."""
    B, L, H = hidden.shape
    # attn shape: (B, L, L) — attn[b, q, k] = how much q attends to k
    importance = attn.mean(dim=1)          # (B, L) — avg attention received
    topk = importance.topk(MEMORY_SLOTS, dim=1).indices   # (B, K)
    mask = torch.zeros(B, L, device=hidden.device, dtype=torch.bool)
    mask.scatter_(1, topk, True)
    return mask


def policy_surprise(hidden: torch.Tensor, attn: torch.Tensor,
                    loss_per_token: torch.Tensor | None) -> torch.Tensor:
    """Write tokens with highest prediction loss (surprise)."""
    B, L, H = hidden.shape
    if loss_per_token is None:
        return policy_random(hidden, attn, None)
    topk = loss_per_token.topk(MEMORY_SLOTS, dim=1).indices
    mask = torch.zeros(B, L, device=hidden.device, dtype=torch.bool)
    mask.scatter_(1, topk, True)
    return mask


class LearnedWriteGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = nn.Linear(HIDDEN_DIM, 1)

    def forward(self, hidden: torch.Tensor, attn: torch.Tensor,
                loss_per_token: torch.Tensor | None) -> torch.Tensor:
        """Differentiable hard-top-k via straight-through."""
        B, L, H = hidden.shape
        scores = self.gate(hidden).squeeze(-1)       # (B, L)
        topk_scores, topk_idx = scores.topk(MEMORY_SLOTS, dim=1)
        mask = torch.zeros(B, L, device=hidden.device, dtype=torch.bool)
        mask.scatter_(1, topk_idx, True)
        return mask


# ── Memory Read ───────────────────────────────────────────────────────────────

class MemoryReader(nn.Module):
    def __init__(self):
        super().__init__()
        self.query_proj = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.out_proj   = nn.Linear(HIDDEN_DIM, VOCAB_SIZE)

    def forward(self, query_hidden: torch.Tensor,
                memory_keys: torch.Tensor, memory_vals: torch.Tensor) -> torch.Tensor:
        """
        query_hidden: (B, H)
        memory_keys:  (B, K, H)
        memory_vals:  (B, K, H)
        Returns logits (B, VOCAB_SIZE)
        """
        q = self.query_proj(query_hidden).unsqueeze(1)   # (B, 1, H)
        scores = (q * memory_keys).sum(-1) / (HIDDEN_DIM ** 0.5)  # (B, K)
        weights = F.softmax(scores, dim=-1).unsqueeze(-1)          # (B, K, 1)
        retrieved = (weights * memory_vals).sum(1)                 # (B, H)
        return self.out_proj(retrieved)


# ── Full Model ────────────────────────────────────────────────────────────────

class MemoryController(nn.Module):
    def __init__(self, write_policy_fn=None, learned_gate: LearnedWriteGate | None = None):
        super().__init__()
        self.encoder      = TinyEncoder()
        self.reader       = MemoryReader()
        self.lm_head      = nn.Linear(HIDDEN_DIM, VOCAB_SIZE)
        self.write_policy = write_policy_fn
        self.learned_gate = learned_gate
        self.query_embed  = nn.Embedding(VOCAB_SIZE, HIDDEN_DIM)

    def forward(self, seq: torch.Tensor, query: torch.Tensor,
                target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (task_loss, token_losses_for_surprise)."""
        hidden, attn_w = self.encoder(seq)     # (B, L, H), (B, L, L)

        # per-token LM loss for surprise signal
        lm_logits = self.lm_head(hidden)       # (B, L, V)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_targets = seq[:, 1:].contiguous()
        loss_per_token = F.cross_entropy(
            shift_logits.view(-1, VOCAB_SIZE),
            shift_targets.view(-1),
            reduction='none',
        ).view(shift_logits.shape[0], -1)  # (B, L-1)
        # pad to L
        pad = torch.zeros(loss_per_token.shape[0], 1, device=loss_per_token.device)
        loss_per_token = torch.cat([loss_per_token, pad], dim=1)

        # apply write policy
        if self.learned_gate is not None:
            write_mask = self.learned_gate(hidden, attn_w, loss_per_token)
        else:
            write_mask = self.write_policy(hidden, attn_w, loss_per_token)

        # extract written slots — use zeros for unwritten
        B, L, H = hidden.shape
        mem_h = hidden.clone()
        mem_h[~write_mask] = 0.0

        # take top MEMORY_SLOTS written per batch item
        written_idx = write_mask.float().topk(MEMORY_SLOTS, dim=1).indices  # (B, K)
        memory_keys = mem_h.gather(
            1, written_idx.unsqueeze(-1).expand(-1, -1, H))
        memory_vals = hidden.gather(
            1, written_idx.unsqueeze(-1).expand(-1, -1, H))

        # form query from query token
        q_h = self.query_embed(query)          # (B, H)
        logits = self.reader(q_h, memory_keys, memory_vals)  # (B, V)

        task_loss = F.cross_entropy(logits, target)
        return task_loss, loss_per_token


# ── Training Loop ─────────────────────────────────────────────────────────────

def train_and_eval(policy_name: str, model: MemoryController) -> dict:
    opt = Adam(model.parameters(), lr=LR)
    train_losses = []

    for step in range(TRAIN_STEPS):
        seq, q, t = make_batch(BATCH_SIZE)
        seq, q, t = seq.to(DEVICE), q.to(DEVICE), t.to(DEVICE)
        loss, _ = model(seq, q, t)
        opt.zero_grad()
        loss.backward()
        opt.step()
        train_losses.append(loss.item())

    # evaluation
    correct = 0
    total   = 0
    model.eval()
    with torch.no_grad():
        for _ in range(EVAL_STEPS):
            seq, q, t = make_batch(BATCH_SIZE)
            seq, q, t = seq.to(DEVICE), q.to(DEVICE), t.to(DEVICE)
            # rebuild mask and logits
            hidden, attn_w = model.encoder(seq)
            lm_logits = model.lm_head(hidden)
            shift_logits = lm_logits[:, :-1, :].contiguous()
            shift_targets = seq[:, 1:].contiguous()
            loss_per_token = F.cross_entropy(
                shift_logits.view(-1, VOCAB_SIZE),
                shift_targets.view(-1),
                reduction='none',
            ).view(shift_logits.shape[0], -1)
            pad = torch.zeros(loss_per_token.shape[0], 1)
            loss_per_token = torch.cat([loss_per_token, pad], dim=1)

            if model.learned_gate is not None:
                write_mask = model.learned_gate(hidden, attn_w, loss_per_token)
            else:
                write_mask = model.write_policy(hidden, attn_w, loss_per_token)

            B, L, H = hidden.shape
            written_idx = write_mask.float().topk(MEMORY_SLOTS, dim=1).indices
            memory_keys = hidden.gather(1, written_idx.unsqueeze(-1).expand(-1,-1,H))
            memory_vals = hidden.gather(1, written_idx.unsqueeze(-1).expand(-1,-1,H))
            q_h = model.query_embed(q)
            logits = model.reader(q_h, memory_keys, memory_vals)
            preds = logits.argmax(dim=-1)
            correct += (preds == t).sum().item()
            total   += t.shape[0]

    acc = correct / total
    final_loss = sum(train_losses[-100:]) / 100
    print(f"  {policy_name:20s}  acc={acc:.3f}  final_loss={final_loss:.4f}")
    return {"accuracy": acc, "final_loss": final_loss}


# ── Experiment ────────────────────────────────────────────────────────────────

class Exp15WriteSignalAblation(Experiment):
    experiment_id = "exp_1_5"
    hypothesis = (
        "A learned write gate outperforms random write, attention-weighted write, "
        "and surprise-driven write on associative recall tasks."
    )

    def run(self) -> ExperimentResult:
        torch.manual_seed(42)

        policies = {
            "random":    (policy_random, None),
            "attention": (policy_attention, None),
            "surprise":  (policy_surprise, None),
            "learned":   (None, LearnedWriteGate()),
        }

        results = {}
        for name, (fn, gate) in policies.items():
            model = MemoryController(write_policy_fn=fn, learned_gate=gate).to(DEVICE)
            results[name] = train_and_eval(name, model)

        accs = {k: v["accuracy"] for k, v in results.items()}
        learned_is_best = accs["learned"] == max(accs.values())
        learned_beats_all = all(
            accs["learned"] >= accs[k] for k in ["random", "attention", "surprise"]
        )

        if learned_is_best and learned_beats_all:
            outcome = OUTCOME_SUPPORTED
        elif accs["learned"] < max(accs["random"], accs["attention"], accs["surprise"]):
            outcome = OUTCOME_REFUTED
        else:
            outcome = OUTCOME_INCONCLUSIVE

        metrics = {f"acc_{k}": v["accuracy"] for k, v in results.items()}
        metrics.update({f"loss_{k}": v["final_loss"] for k, v in results.items()})
        metrics["ranking"] = sorted(accs.keys(), key=lambda k: accs[k], reverse=True)

        gap = accs["learned"] - max(accs["random"], accs["attention"], accs["surprise"])
        notes = (
            f"Learned gate delta over best baseline: {gap:+.3f}. "
            f"Ranking: {metrics['ranking']}."
        )

        return self.result(outcome, metrics, notes, config={
            "vocab_size": VOCAB_SIZE, "seq_len": SEQ_LEN,
            "memory_slots": MEMORY_SLOTS, "train_steps": TRAIN_STEPS,
        })


if __name__ == "__main__":
    Exp15WriteSignalAblation().execute()
