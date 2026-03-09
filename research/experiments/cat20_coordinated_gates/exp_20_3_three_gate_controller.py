"""
exp_20_3_three_gate_controller.py

Hypothesis: Three-gate controller with all auxiliary losses combined outperforms
any single-auxiliary system.
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
VOCAB_SIZE = 64
HIDDEN_DIM = 64
SEQ_LEN = 24
MEMORY_SLOTS = 8
NUM_PAIRS = 4
STEPS = 400
BATCH = 32
LR = 3e-4
DEVICE = "cpu"
LAMBDA_W = 0.1      # write sparsity
LAMBDA_R = 0.1      # read accuracy
LAMBDA_F = 0.05     # forget usefulness
K_WRITE = 4


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
# Three-gate memory controller
# ---------------------------------------------------------------------------
class ThreeGateMemory(nn.Module):
    """
    Three independent gates:
      - write_gate:  selects tokens to store in memory (top-k)
      - read_gate:   decides whether to use memory or pass query through
      - forget_gate: scores slots — lowest score gets evicted before writing
    Slot ages tracked as integer counters (not parameters).
    """

    def __init__(self, hidden_dim, memory_slots, k_write):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.memory_slots = memory_slots
        self.k_write = k_write

        self.write_gate = nn.Linear(hidden_dim, 1)
        self.read_gate = nn.Linear(hidden_dim, 1)
        self.forget_gate = nn.Linear(hidden_dim, memory_slots)  # scores each slot

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_head = nn.Linear(hidden_dim, VOCAB_SIZE)

    def forward(self, enc_hidden, query_hidden, slot_ages):
        """
        enc_hidden:   (B, T, H)
        query_hidden: (B, H)
        slot_ages:    (B, M)  integer tensor — steps since each slot was written

        Returns: logits, write_scores, read_score, forget_scores, memory
        """
        B, T, H = enc_hidden.shape
        M = self.memory_slots

        # --- Write gate ---
        write_logits = self.write_gate(enc_hidden).squeeze(-1)   # (B, T)
        write_scores = torch.sigmoid(write_logits)               # (B, T)
        k = min(self.k_write, M, T)
        _, top_write_idx = write_scores.topk(k, dim=-1)          # (B, k)

        # --- Forget gate: score each slot from query perspective ---
        forget_scores = torch.sigmoid(self.forget_gate(query_hidden))  # (B, M)
        evict_slot = forget_scores.argmin(dim=-1)                       # (B,)  slot to evict

        # --- Build memory ---
        memory = torch.zeros(B, M, H, device=enc_hidden.device)
        for slot in range(k):
            tok_idx = top_write_idx[:, slot]
            mem_slot = slot % M
            memory[:, mem_slot] = enc_hidden[torch.arange(B), tok_idx]

        # Eviction: zero out the evicted slot (simulate forget)
        for b in range(B):
            memory[b, evict_slot[b]] = 0.0

        # --- Read gate ---
        read_score = torch.sigmoid(self.read_gate(query_hidden))  # (B, 1)

        # --- Attend over memory ---
        q = self.q_proj(query_hidden).unsqueeze(1)                # (B, 1, H)
        keys = self.k_proj(memory)                                 # (B, M, H)
        raw = (q @ keys.transpose(1, 2)).squeeze(1) / (H ** 0.5)  # (B, M)
        attn = torch.softmax(raw, dim=-1)
        retrieved = (attn.unsqueeze(1) @ memory).squeeze(1)       # (B, H)

        # Blend: read_gate interpolates between memory retrieval and passthrough
        fused = read_score * retrieved + (1 - read_score) * query_hidden
        logits = self.out_head(fused)

        return logits, write_scores, read_score, forget_scores, memory


# ---------------------------------------------------------------------------
# Oracle memory for read auxiliary
# ---------------------------------------------------------------------------
def build_oracle_memory(enc_hidden, num_pairs, memory_slots):
    B, T, H = enc_hidden.shape
    memory = torch.zeros(B, memory_slots, H, device=enc_hidden.device)
    n_write = min(num_pairs * 2, memory_slots, T - 3)
    for i in range(n_write):
        slot = i % memory_slots
        memory[:, slot] = enc_hidden[:, i]
    return memory


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train_condition(use_write_aux, use_read_aux, use_forget_aux, label=""):
    torch.manual_seed(42)
    encoder = Encoder(VOCAB_SIZE, HIDDEN_DIM).to(DEVICE)
    mem_ctrl = ThreeGateMemory(HIDDEN_DIM, MEMORY_SLOTS, K_WRITE).to(DEVICE)
    read_head_oracle = nn.Sequential(
        nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
        nn.ReLU(),
        nn.Linear(HIDDEN_DIM, VOCAB_SIZE),
    ).to(DEVICE)
    opt = Adam(
        list(encoder.parameters()) +
        list(mem_ctrl.parameters()) +
        list(read_head_oracle.parameters()),
        lr=LR,
    )

    # Slot ages: (BATCH, MEMORY_SLOTS) — reset each episode
    slot_ages = torch.zeros(BATCH, MEMORY_SLOTS, dtype=torch.long, device=DEVICE)

    correct = total = 0

    for step in range(STEPS):
        seq, target = make_assoc_batch(BATCH, SEQ_LEN, VOCAB_SIZE, NUM_PAIRS)
        seq, target = seq.to(DEVICE), target.to(DEVICE)

        # Reset slot ages each batch (each sequence is independent)
        slot_ages = torch.zeros(BATCH, MEMORY_SLOTS, dtype=torch.long, device=DEVICE)

        enc_h = encoder(seq)                                     # (B, T, H)
        query_h = enc_h[:, SEQ_LEN - 2]                         # (B, H)

        logits, write_scores, read_score, forget_scores, memory = \
            mem_ctrl(enc_h, query_h, slot_ages)

        task_loss = F.cross_entropy(logits, target)
        total_loss = task_loss

        # Auxiliary 1: Write sparsity L1
        if use_write_aux:
            write_aux = write_scores.mean()
            total_loss = total_loss + LAMBDA_W * write_aux

        # Auxiliary 2: Read accuracy — oracle read supervision
        # Gradient is blocked from flowing back to encoder/write gate via .detach()
        if use_read_aux:
            oracle_mem = build_oracle_memory(enc_h.detach(), NUM_PAIRS, MEMORY_SLOTS)
            # Use mem_ctrl's attention infrastructure via a simple MLP oracle head
            B_o, M_o, H_o = oracle_mem.shape
            q_o = mem_ctrl.q_proj(query_h.detach())              # (B, H)
            k_o = mem_ctrl.k_proj(oracle_mem.detach())           # (B, M, H)
            raw_o = (q_o.unsqueeze(1) @ k_o.transpose(1, 2)).squeeze(1) / (H_o ** 0.5)
            attn_o = torch.softmax(raw_o, dim=-1)
            retr_o = (attn_o.unsqueeze(1) @ oracle_mem.detach()).squeeze(1)  # (B, H)
            oracle_logits = mem_ctrl.out_head(retr_o + query_h.detach())
            read_aux = F.cross_entropy(oracle_logits, target)
            total_loss = total_loss + LAMBDA_R * read_aux

        # Auxiliary 3: Forget usefulness — penalize forgetting recently-accessed slots
        # recent_mask: slots with age < 4 (just written) should not be evicted
        if use_forget_aux:
            recent_mask = (slot_ages < 4).float().to(DEVICE)    # (B, M)
            forget_usefulness_loss = (forget_scores * recent_mask).mean()
            total_loss = total_loss + LAMBDA_F * forget_usefulness_loss

        opt.zero_grad()
        total_loss.backward()
        opt.step()

        # Increment ages for written slots (approximate: add 1 for all, then
        # reset the top-k written slots to 0)
        with torch.no_grad():
            slot_ages = slot_ages + 1
            _, top_write_idx = write_scores.topk(min(K_WRITE, MEMORY_SLOTS, SEQ_LEN), dim=-1)
            for s in range(top_write_idx.size(1)):
                mem_slot = s % MEMORY_SLOTS
                slot_ages[:, mem_slot] = 0

        if step >= STEPS * 3 // 4:
            correct += (logits.argmax(-1) == target).sum().item()
            total += BATCH

    return correct / max(total, 1)


# ---------------------------------------------------------------------------
# Experiment class
# ---------------------------------------------------------------------------
class Exp203ThreeGateController(Experiment):
    experiment_id = "exp_20_3"
    hypothesis = (
        "Three-gate controller with all auxiliary losses combined outperforms "
        "any single-auxiliary system."
    )

    def run(self) -> ExperimentResult:
        config = dict(
            VOCAB_SIZE=VOCAB_SIZE, HIDDEN_DIM=HIDDEN_DIM, SEQ_LEN=SEQ_LEN,
            MEMORY_SLOTS=MEMORY_SLOTS, NUM_PAIRS=NUM_PAIRS, STEPS=STEPS,
            BATCH=BATCH, K_WRITE=K_WRITE, LR=LR,
            LAMBDA_W=LAMBDA_W, LAMBDA_R=LAMBDA_R, LAMBDA_F=LAMBDA_F,
        )

        conditions = [
            ("A", False, False, False, "task only"),
            ("B", True,  False, False, "write sparsity only"),
            ("C", False, True,  False, "read accuracy only"),
            ("D", False, False, True,  "forget usefulness only"),
            ("E", True,  True,  True,  "all three combined"),
        ]

        results = {}
        for label, uw, ur, uf, desc in conditions:
            print(f"Training condition {label}: {desc}...")
            acc = train_condition(uw, ur, uf, label)
            results[label] = acc
            print(f"  acc_{label}={acc:.4f}")

        acc_A, acc_B, acc_C, acc_D, acc_E = (
            results["A"], results["B"], results["C"], results["D"], results["E"]
        )
        best_single = max(acc_B, acc_C, acc_D)

        metrics = dict(
            acc_A=round(acc_A, 4),
            acc_B=round(acc_B, 4),
            acc_C=round(acc_C, 4),
            acc_D=round(acc_D, 4),
            acc_E=round(acc_E, 4),
            best_single_aux=round(best_single, 4),
            gap_E_minus_best_single=round(acc_E - best_single, 4),
            gap_E_minus_A=round(acc_E - acc_A, 4),
        )

        if acc_E > best_single + 0.02:
            outcome = OUTCOME_SUPPORTED
            notes = (
                f"Full three-gate (acc_E={acc_E:.4f}) beats best single-aux "
                f"({best_single:.4f}) by {acc_E - best_single:.4f}."
            )
        elif acc_A >= acc_E - 0.01:
            outcome = OUTCOME_REFUTED
            notes = (
                f"No-auxiliary baseline (acc_A={acc_A:.4f}) nearly matches "
                f"full system (acc_E={acc_E:.4f})."
            )
        else:
            outcome = OUTCOME_INCONCLUSIVE
            notes = (
                f"Full system improves over baseline by {acc_E - acc_A:.4f} but "
                f"gap over best single ({acc_E - best_single:.4f}) < 0.02."
            )

        return self.result(outcome, metrics=metrics, notes=notes, config=config)


if __name__ == "__main__":
    Exp203ThreeGateController().execute()
