"""
Experiment 5.7 — Attention-Memory Arbitration

Hypothesis: When local attention and external memory produce conflicting
predictions, a learned arbitration policy outperforms both fixed-priority
policies.

Setup:
  - Task with two conflicting signals:
    (a) Local pattern: last 3 tokens suggest one answer (via attention)
    (b) Stored distant fact: memory suggests different answer
  - Ground truth alternates which signal is correct (50/50)
  - Three policies:
    (1) always-trust-attention
    (2) always-trust-memory
    (3) learned arbitrator MLP gating between attention and memory output
  - SUPPORTED if learned_arb_acc > both fixed policies by > 0.05
  - REFUTED if one fixed policy wins
  - INCONCLUSIVE otherwise
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
MEMORY_SLOTS  = 8
SEQ_LEN       = 24
BATCH_SIZE    = 32
TRAIN_STEPS   = 1500
LR            = 3e-4
DEVICE        = "cpu"
LOCAL_WINDOW  = 3    # last N tokens for local attention signal


# ── Shared Utilities ──────────────────────────────────────────────────────────

class TokenEmbedder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, HIDDEN_DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embed(x)


def local_attention_signal(
    embeddings: torch.Tensor,    # (B, L, H)
    query: torch.Tensor,         # (B, H)
    window: int = LOCAL_WINDOW,
) -> torch.Tensor:
    """Soft attention over last `window` token embeddings. Returns (B, H)."""
    B, L, H = embeddings.shape
    local   = embeddings[:, -window:, :]              # (B, W, H)
    q       = query.unsqueeze(1)                       # (B, 1, H)
    scores  = (q * local).sum(-1) / H ** 0.5          # (B, W)
    w       = F.softmax(scores, dim=-1).unsqueeze(-1)  # (B, W, 1)
    return (w * local).sum(1)                          # (B, H)


def memory_signal(
    query: torch.Tensor,         # (B, H)
    memory: torch.Tensor,        # (B, M, H)
    q_proj: nn.Linear,
    k_proj: nn.Linear,
    v_proj: nn.Linear,
) -> torch.Tensor:
    """Soft attention over memory slots. Returns (B, H)."""
    q = q_proj(query).unsqueeze(1)
    k = k_proj(memory)
    v = v_proj(memory)
    w = F.softmax((q * k).sum(-1) / HIDDEN_DIM ** 0.5, dim=-1).unsqueeze(-1)
    return (w * v).sum(1)


# ── Policy 1: Always Trust Attention ─────────────────────────────────────────

class AlwaysAttentionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedder = TokenEmbedder()
        self.gru      = nn.GRU(HIDDEN_DIM, HIDDEN_DIM, batch_first=True)
        self.head     = nn.Linear(HIDDEN_DIM, VOCAB_SIZE)

    def forward(
        self, seq: torch.Tensor, memory: torch.Tensor
    ) -> torch.Tensor:
        emb = self.embedder(seq)                         # (B, L, H)
        _, h_n = self.gru(emb)
        query  = h_n.squeeze(0)                          # (B, H)
        attn   = local_attention_signal(emb, query)      # (B, H)
        return self.head(attn)


# ── Policy 2: Always Trust Memory ─────────────────────────────────────────────

class AlwaysMemoryModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedder = TokenEmbedder()
        self.gru      = nn.GRU(HIDDEN_DIM, HIDDEN_DIM, batch_first=True)
        self.q_proj   = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.k_proj   = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.v_proj   = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.head     = nn.Linear(HIDDEN_DIM, VOCAB_SIZE)

    def forward(
        self, seq: torch.Tensor, memory: torch.Tensor
    ) -> torch.Tensor:
        emb = self.embedder(seq)
        _, h_n = self.gru(emb)
        query  = h_n.squeeze(0)
        mem_out = memory_signal(query, memory,
                                self.q_proj, self.k_proj, self.v_proj)
        return self.head(mem_out)


# ── Policy 3: Learned Arbitrator ─────────────────────────────────────────────

class ArbitratedModel(nn.Module):
    """
    Computes both attention signal and memory signal, then uses an MLP
    to gate between them. Gate is conditioned on both signals + query.
    """

    def __init__(self):
        super().__init__()
        self.embedder = TokenEmbedder()
        self.gru      = nn.GRU(HIDDEN_DIM, HIDDEN_DIM, batch_first=True)

        self.q_proj = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.k_proj = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.v_proj = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)

        # Arbitrator: takes [query, attn_signal, mem_signal] -> gate scalar
        self.arbitrator = nn.Sequential(
            nn.Linear(HIDDEN_DIM * 3, 64), nn.ReLU(),
            nn.Linear(64, 32),             nn.ReLU(),
            nn.Linear(32, 1),              nn.Sigmoid(),
        )

        self.head = nn.Linear(HIDDEN_DIM, VOCAB_SIZE)

    def forward(
        self, seq: torch.Tensor, memory: torch.Tensor
    ) -> torch.Tensor:
        emb = self.embedder(seq)
        _, h_n = self.gru(emb)
        query  = h_n.squeeze(0)                           # (B, H)

        attn_out = local_attention_signal(emb, query)     # (B, H)
        mem_out  = memory_signal(query, memory,
                                 self.q_proj, self.k_proj, self.v_proj)  # (B, H)

        gate_inp = torch.cat([query, attn_out, mem_out], dim=-1)  # (B, 3H)
        gate     = self.arbitrator(gate_inp)              # (B, 1)

        fused = gate * mem_out + (1 - gate) * attn_out   # (B, H)
        return self.head(fused)


# ── Task ──────────────────────────────────────────────────────────────────────

def make_conflict_batch(batch_size: int):
    """
    Conflict task:
    - local_answer  = (seq[-1] * 3 + 7) % VOCAB_SIZE  (from last token)
    - memory_answer = stored in memory slot 0
    - ground_truth alternates: even indices -> local is correct,
                               odd indices  -> memory is correct
    - label (which_is_correct): 0 = attention, 1 = memory

    Returns: seq, target, memory, which_is_correct
    """
    B = batch_size
    seq = torch.randint(1, VOCAB_SIZE, (B, SEQ_LEN))

    local_answer  = (seq[:, -1] * 3 + 7) % VOCAB_SIZE   # (B,)
    memory_answer = torch.randint(2, VOCAB_SIZE, (B,))   # (B,)

    # Ensure local != memory answer
    same_mask = (local_answer == memory_answer)
    memory_answer[same_mask] = (memory_answer[same_mask] + 1) % VOCAB_SIZE

    # Which signal is correct: 50/50
    which = torch.randint(0, 2, (B,))   # 0 = attn, 1 = memory
    target = torch.where(which == 0, local_answer, memory_answer)

    # Encode memory answer in slot 0
    memory = torch.randn(B, MEMORY_SLOTS, HIDDEN_DIM) * 0.1
    for b in range(B):
        memory[b, 0, :] = 0.0
        memory[b, 0, memory_answer[b].item() % HIDDEN_DIM] = 2.0

    return seq, target, memory, which


# ── Train / Eval ──────────────────────────────────────────────────────────────

def train_model(model: nn.Module) -> None:
    opt = Adam(model.parameters(), lr=LR)
    for _ in range(TRAIN_STEPS):
        seq, tgt, mem, _ = make_conflict_batch(BATCH_SIZE)
        logits = model(seq, mem)
        loss   = F.cross_entropy(logits, tgt)
        opt.zero_grad(); loss.backward(); opt.step()


def eval_model(model: nn.Module, n_batches: int = 50) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for _ in range(n_batches):
            seq, tgt, mem, _ = make_conflict_batch(BATCH_SIZE)
            logits = model(seq, mem)
            preds  = logits.argmax(-1)
            correct += (preds == tgt).sum().item()
            total   += tgt.size(0)
    model.train()
    return correct / total


# ── Experiment ────────────────────────────────────────────────────────────────

class Exp57AttentionMemoryArbitration(Experiment):
    experiment_id = "exp_5_7"
    hypothesis = (
        "When local attention and external memory produce conflicting "
        "predictions, a learned arbitration policy outperforms both "
        "fixed-priority policies."
    )

    def run(self) -> ExperimentResult:
        torch.manual_seed(42)

        config = {
            "vocab_size":   VOCAB_SIZE,
            "hidden_dim":   HIDDEN_DIM,
            "memory_slots": MEMORY_SLOTS,
            "seq_len":      SEQ_LEN,
            "batch_size":   BATCH_SIZE,
            "train_steps":  TRAIN_STEPS,
            "local_window": LOCAL_WINDOW,
        }

        print("\n  Training always-attention model...")
        attn_model = AlwaysAttentionModel().to(DEVICE)
        train_model(attn_model)
        attn_acc = eval_model(attn_model)
        print(f"    attn_only_acc={attn_acc:.3f}")

        print("  Training always-memory model...")
        mem_model = AlwaysMemoryModel().to(DEVICE)
        train_model(mem_model)
        mem_acc = eval_model(mem_model)
        print(f"    mem_only_acc={mem_acc:.3f}")

        print("  Training learned arbitrator model...")
        arb_model = ArbitratedModel().to(DEVICE)
        train_model(arb_model)
        arb_acc = eval_model(arb_model)
        print(f"    arbitrated_acc={arb_acc:.3f}")

        best_fixed    = max(attn_acc, mem_acc)
        arb_advantage = arb_acc - best_fixed

        metrics = {
            "attn_only_acc":       round(attn_acc, 4),
            "mem_only_acc":        round(mem_acc, 4),
            "arbitrated_acc":      round(arb_acc, 4),
            "arbitration_advantage": round(arb_advantage, 4),
            "best_fixed_policy_acc": round(best_fixed, 4),
        }

        if arb_acc > attn_acc + 0.05 and arb_acc > mem_acc + 0.05:
            outcome = OUTCOME_SUPPORTED
        elif attn_acc > arb_acc or mem_acc > arb_acc:
            outcome = OUTCOME_REFUTED
        else:
            outcome = OUTCOME_INCONCLUSIVE

        notes = (
            f"attn_only_acc={attn_acc:.3f}, mem_only_acc={mem_acc:.3f}, "
            f"arbitrated_acc={arb_acc:.3f}. "
            f"Arbitration advantage over best fixed: {arb_advantage:.4f}. "
            f"SUPPORTED requires advantage > 0.05 over BOTH fixed policies."
        )

        return self.result(outcome, metrics, notes, config)


if __name__ == "__main__":
    Exp57AttentionMemoryArbitration().execute()
