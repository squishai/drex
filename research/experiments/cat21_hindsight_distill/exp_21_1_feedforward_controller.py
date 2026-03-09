"""
Experiment 21.1 — Feedforward Controller vs LSTM Controller

Hypothesis: A feedforward-only memory controller achieves higher external memory
utilization than an LSTM controller at equal parameter count.
"""

from __future__ import annotations

import math
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
HIDDEN_DIM    = 32
SEQ_LEN       = 24
MEMORY_SLOTS  = 8
NUM_PAIRS     = 4
STEPS         = 400
BATCH         = 32
LR            = 3e-4
DEVICE        = "cpu"

# ── Data generator ────────────────────────────────────────────────────────────

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

# ── Shared read head ──────────────────────────────────────────────────────────

class ReadHead(nn.Module):
    """Attention-based read from memory slots."""
    def __init__(self, hidden_dim, memory_slots):
        super().__init__()
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj   = nn.Linear(hidden_dim, hidden_dim)
        self.scale      = math.sqrt(hidden_dim)

    def forward(self, query: torch.Tensor, memory: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        query:  (B, H)
        memory: (B, S, H)
        returns: read_vec (B, H), attn_weights (B, S)
        """
        q = self.query_proj(query).unsqueeze(1)          # (B, 1, H)
        k = self.key_proj(memory)                         # (B, S, H)
        scores = torch.bmm(q, k.transpose(1, 2)) / self.scale  # (B, 1, S)
        attn   = F.softmax(scores, dim=-1).squeeze(1)    # (B, S)
        read   = torch.bmm(attn.unsqueeze(1), memory).squeeze(1)  # (B, H)
        return read, attn

# ── LSTM controller ───────────────────────────────────────────────────────────

class LSTMController(nn.Module):
    """
    Processes the full sequence with an LSTM; derives write gates from hidden states.
    Writes top-k tokens (by gate score) into memory using soft Gumbel selection.
    """
    def __init__(self, vocab_size, hidden_dim, memory_slots, num_pairs):
        super().__init__()
        self.hidden_dim    = hidden_dim
        self.memory_slots  = memory_slots
        self.num_pairs     = num_pairs
        half = hidden_dim // 2

        self.embed      = nn.Embedding(vocab_size, hidden_dim)
        self.lstm       = nn.LSTM(hidden_dim, half, batch_first=True)
        self.write_gate = nn.Linear(half, 1)
        self.val_proj   = nn.Linear(hidden_dim, hidden_dim)
        self.read_head  = ReadHead(hidden_dim, memory_slots)
        self.out        = nn.Linear(hidden_dim, vocab_size)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        B, L = seq.shape
        emb = self.embed(seq)                              # (B, L, H)
        h_lstm, _ = self.lstm(emb)                        # (B, L, H//2)
        gate_scores = torch.sigmoid(self.write_gate(h_lstm)).squeeze(-1)  # (B, L)

        # Soft top-k selection via Gumbel straight-through
        memory = self._soft_topk_write(emb, gate_scores)  # (B, S, H)

        # Query from last non-pad token (position seq_len-2: query key)
        query = emb[:, -2, :]                             # (B, H)
        read_vec, _ = self.read_head(query, memory)       # (B, H)
        logits = self.out(read_vec)                       # (B, V)
        return logits

    def _soft_topk_write(self, emb: torch.Tensor, gate_scores: torch.Tensor) -> torch.Tensor:
        """
        Soft top-k: use Gumbel noise + straight-through to select num_pairs tokens
        and write them into memory_slots (padding remaining slots with zeros).
        """
        B, L, H = emb.shape
        K = min(self.num_pairs, self.memory_slots)

        # Gumbel straight-through selection
        gumbel_noise = -torch.log(-torch.log(torch.clamp(
            torch.rand_like(gate_scores), 1e-10, 1.0
        )))
        perturbed = gate_scores + 0.1 * gumbel_noise     # (B, L)

        # Hard top-k indices
        _, topk_idx = perturbed.topk(K, dim=-1)           # (B, K)

        # Soft weights via softmax over perturbed scores at selected positions
        gathered = perturbed.gather(1, topk_idx)           # (B, K)
        soft_weights = F.softmax(gathered, dim=-1)         # (B, K)

        # Weighted sum of selected embeddings via value projection
        vals = self.val_proj(emb)                          # (B, L, H)
        topk_vals = vals.gather(
            1, topk_idx.unsqueeze(-1).expand(B, K, H)
        )                                                  # (B, K, H)

        # Pad to memory_slots
        memory = torch.zeros(B, self.memory_slots, H, device=emb.device)
        # Write each selected token as a weighted slot entry
        for k_i in range(K):
            slot_vec = topk_vals[:, k_i, :] * soft_weights[:, k_i:k_i+1]  # (B, H)
            memory[:, k_i, :] = slot_vec

        return memory

# ── Feedforward controller ────────────────────────────────────────────────────

class FeedforwardController(nn.Module):
    """
    2-layer MLP processes each token independently (no recurrence).
    Write gates are computed position-wise without cross-token state.
    """
    def __init__(self, vocab_size, hidden_dim, memory_slots, num_pairs):
        super().__init__()
        self.hidden_dim   = hidden_dim
        self.memory_slots = memory_slots
        self.num_pairs    = num_pairs

        self.embed      = nn.Embedding(vocab_size, hidden_dim)
        # 2-layer MLP applied position-wise
        self.mlp        = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.write_gate = nn.Linear(hidden_dim, 1)
        self.val_proj   = nn.Linear(hidden_dim, hidden_dim)
        self.read_head  = ReadHead(hidden_dim, memory_slots)
        self.out        = nn.Linear(hidden_dim, vocab_size)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        B, L = seq.shape
        emb = self.embed(seq)                              # (B, L, H)
        h   = self.mlp(emb)                               # (B, L, H)  position-wise
        gate_scores = torch.sigmoid(self.write_gate(h)).squeeze(-1)  # (B, L)

        memory = self._soft_topk_write(emb, gate_scores)  # (B, S, H)

        query = emb[:, -2, :]
        read_vec, _ = self.read_head(query, memory)
        logits = self.out(read_vec)
        return logits

    def _soft_topk_write(self, emb: torch.Tensor, gate_scores: torch.Tensor) -> torch.Tensor:
        B, L, H = emb.shape
        K = min(self.num_pairs, self.memory_slots)

        gumbel_noise = -torch.log(-torch.log(torch.clamp(
            torch.rand_like(gate_scores), 1e-10, 1.0
        )))
        perturbed = gate_scores + 0.1 * gumbel_noise

        _, topk_idx = perturbed.topk(K, dim=-1)
        gathered     = perturbed.gather(1, topk_idx)
        soft_weights = F.softmax(gathered, dim=-1)

        vals      = self.val_proj(emb)
        topk_vals = vals.gather(1, topk_idx.unsqueeze(-1).expand(B, K, H))

        memory = torch.zeros(B, self.memory_slots, H, device=emb.device)
        for k_i in range(K):
            memory[:, k_i, :] = topk_vals[:, k_i, :] * soft_weights[:, k_i:k_i+1]

        return memory

# ── Training ──────────────────────────────────────────────────────────────────

def train_model(model: nn.Module, steps: int) -> None:
    opt = Adam(model.parameters(), lr=LR)
    model.train()
    for step in range(steps):
        seq, target = make_assoc_batch(BATCH, SEQ_LEN, VOCAB_SIZE, NUM_PAIRS)
        seq, target = seq.to(DEVICE), target.to(DEVICE)
        logits = model(seq)
        loss   = F.cross_entropy(logits, target)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (step + 1) % 500 == 0:
            print(f"    step {step+1}/{steps}  loss={loss.item():.4f}")

# ── Evaluation ────────────────────────────────────────────────────────────────

def eval_model_with_utilization(model: nn.Module, n_batches: int = 100) -> tuple[float, float]:
    """
    Returns (accuracy, slot_utilization_entropy).
    Utilization measured as entropy of the slot-access distribution.
    Higher entropy -> more even usage -> better utilization.
    """
    model.eval()
    correct = 0
    total   = 0
    slot_counts = torch.zeros(MEMORY_SLOTS)

    with torch.no_grad():
        for _ in range(n_batches):
            seq, target = make_assoc_batch(BATCH, SEQ_LEN, VOCAB_SIZE, NUM_PAIRS)
            seq, target = seq.to(DEVICE), target.to(DEVICE)
            B, L = seq.shape
            H = model.hidden_dim

            # Reproduce forward pass to capture attention weights
            emb = model.embed(seq)

            if isinstance(model, LSTMController):
                h_lstm, _ = model.lstm(emb)
                gate_scores = torch.sigmoid(model.write_gate(h_lstm)).squeeze(-1)
            else:
                h   = model.mlp(emb)
                gate_scores = torch.sigmoid(model.write_gate(h)).squeeze(-1)

            memory = model._soft_topk_write(emb, gate_scores)

            query = emb[:, -2, :]
            _, attn = model.read_head(query, memory)      # (B, S)

            # Accumulate argmax slot accesses
            accessed = attn.argmax(dim=-1)                # (B,)
            for s in accessed:
                slot_counts[s.item()] += 1

            logits = model.out(
                torch.bmm(attn.unsqueeze(1), memory).squeeze(1)
            )
            preds   = logits.argmax(dim=-1)
            correct += (preds == target).sum().item()
            total   += B

    acc = correct / total

    # Entropy of slot access distribution
    probs = slot_counts / slot_counts.sum().clamp(min=1e-8)
    entropy = -(probs * torch.log(probs + 1e-8)).sum().item()

    return acc, entropy

# ── Experiment ────────────────────────────────────────────────────────────────

class Exp211FeedforwardController(Experiment):
    experiment_id = "exp_21_1"
    hypothesis = (
        "A feedforward-only memory controller achieves higher external memory "
        "utilization than an LSTM controller at equal parameter count."
    )

    def run(self) -> ExperimentResult:
        config = dict(
            vocab_size=VOCAB_SIZE, hidden_dim=HIDDEN_DIM, seq_len=SEQ_LEN,
            memory_slots=MEMORY_SLOTS, num_pairs=NUM_PAIRS, steps=STEPS, batch=BATCH,
        )

        # ── Condition A: LSTM controller ──────────────────────────────────────
        print("  Training Condition A: LSTM controller ...")
        model_A = LSTMController(VOCAB_SIZE, HIDDEN_DIM, MEMORY_SLOTS, NUM_PAIRS).to(DEVICE)
        train_model(model_A, STEPS)
        acc_A, util_A = eval_model_with_utilization(model_A)
        params_A = sum(p.numel() for p in model_A.parameters())
        print(f"    acc_A={acc_A:.4f}  util_A={util_A:.4f}  params={params_A}")

        # ── Condition B: Feedforward controller ───────────────────────────────
        print("  Training Condition B: Feedforward controller ...")
        model_B = FeedforwardController(VOCAB_SIZE, HIDDEN_DIM, MEMORY_SLOTS, NUM_PAIRS).to(DEVICE)
        train_model(model_B, STEPS)
        acc_B, util_B = eval_model_with_utilization(model_B)
        params_B = sum(p.numel() for p in model_B.parameters())
        print(f"    acc_B={acc_B:.4f}  util_B={util_B:.4f}  params={params_B}")

        acc_gap  = acc_B - acc_A
        util_gap = util_B - util_A
        print(f"  acc_gap={acc_gap:.4f}  util_gap={util_gap:.4f}")

        # ── Outcome ───────────────────────────────────────────────────────────
        if acc_B > acc_A + 0.01 and util_B > util_A + 0.15:
            outcome = OUTCOME_SUPPORTED
        elif acc_A >= acc_B - 0.02 and abs(util_gap) < 0.15:
            outcome = OUTCOME_REFUTED
        else:
            outcome = OUTCOME_INCONCLUSIVE

        metrics = dict(
            acc_A=round(acc_A, 4),
            acc_B=round(acc_B, 4),
            util_A=round(util_A, 4),
            util_B=round(util_B, 4),
            acc_gap=round(acc_gap, 4),
            util_gap=round(util_gap, 4),
            params_A=params_A,
            params_B=params_B,
        )
        notes = (
            f"LSTM: acc={acc_A:.4f} util={util_A:.4f}. "
            f"FF: acc={acc_B:.4f} util={util_B:.4f}. "
            f"acc_gap={acc_gap:.4f} util_gap={util_gap:.4f}."
        )
        return self.result(outcome, metrics, notes, config)


if __name__ == "__main__":
    Exp211FeedforwardController().execute()
