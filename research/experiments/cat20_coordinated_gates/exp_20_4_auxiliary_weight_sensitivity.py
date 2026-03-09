"""
exp_20_4_auxiliary_weight_sensitivity.py

Hypothesis: Optimal write sparsity auxiliary weight is in [0.01, 0.1] — outside
this range gate collapses or task signal drowns.
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
MEMORY_SLOTS = 6
NUM_PAIRS = 4
STEPS = 500
BATCH = 32
LR = 3e-4
DEVICE = "cpu"
LAMBDA_VALUES = [0.0, 0.01, 0.1, 1.0]


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
# Write gate memory
# ---------------------------------------------------------------------------
class WriteGateMemory(nn.Module):
    def __init__(self, hidden_dim, memory_slots, k_write=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.memory_slots = memory_slots
        self.k_write = k_write
        self.write_gate = nn.Linear(hidden_dim, 1)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_head = nn.Linear(hidden_dim, VOCAB_SIZE)

    def forward(self, enc_hidden, query_hidden):
        B, T, H = enc_hidden.shape
        gate_logits = self.write_gate(enc_hidden).squeeze(-1)    # (B, T)
        gate_scores = torch.sigmoid(gate_logits)                 # (B, T)

        k = min(self.k_write, self.memory_slots, T)
        _, top_idx = gate_scores.topk(k, dim=-1)

        memory = torch.zeros(B, self.memory_slots, H, device=enc_hidden.device)
        for slot in range(k):
            tok_idx = top_idx[:, slot]
            memory[:, slot] = enc_hidden[torch.arange(B), tok_idx]

        q = self.q_proj(query_hidden).unsqueeze(1)
        keys = self.k_proj(memory)
        scores = (q @ keys.transpose(1, 2)).squeeze(1) / (H ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        retrieved = (attn.unsqueeze(1) @ memory).squeeze(1)
        logits = self.out_head(retrieved + query_hidden)

        return logits, gate_scores


# ---------------------------------------------------------------------------
# Collapse check
# ---------------------------------------------------------------------------
def is_collapsed(rate: float) -> bool:
    """Gate is degenerate if always off (<2%) or always on (>95%)."""
    return rate < 0.02 or rate > 0.95


# ---------------------------------------------------------------------------
# Training loop for a single lambda
# ---------------------------------------------------------------------------
def train_one_lambda(lam: float):
    torch.manual_seed(42)
    encoder = Encoder(VOCAB_SIZE, HIDDEN_DIM).to(DEVICE)
    mem = WriteGateMemory(HIDDEN_DIM, MEMORY_SLOTS, k_write=4).to(DEVICE)
    opt = Adam(list(encoder.parameters()) + list(mem.parameters()), lr=LR)

    correct = total = 0
    write_rates = []

    for step in range(STEPS):
        seq, target = make_assoc_batch(BATCH, SEQ_LEN, VOCAB_SIZE, NUM_PAIRS)
        seq, target = seq.to(DEVICE), target.to(DEVICE)

        enc_h = encoder(seq)
        query_h = enc_h[:, SEQ_LEN - 2]
        logits, gate_scores = mem(enc_h, query_h)

        task_loss = F.cross_entropy(logits, target)
        aux_loss = gate_scores.mean()     # L1 on sigmoid scores (promotes sparsity)
        total_loss = task_loss + lam * aux_loss

        opt.zero_grad()
        total_loss.backward()
        opt.step()

        if step >= STEPS * 3 // 4:
            correct += (logits.argmax(-1) == target).sum().item()
            total += BATCH
            write_rates.append(gate_scores.mean().item())

    acc = correct / max(total, 1)
    avg_wr = sum(write_rates) / max(len(write_rates), 1)
    return acc, avg_wr


# ---------------------------------------------------------------------------
# Experiment class
# ---------------------------------------------------------------------------
class Exp204AuxiliaryWeightSensitivity(Experiment):
    experiment_id = "exp_20_4"
    hypothesis = (
        "Optimal write sparsity auxiliary weight is in [0.01, 0.1] — outside "
        "this range gate collapses or task signal drowns."
    )

    def run(self) -> ExperimentResult:
        config = dict(
            VOCAB_SIZE=VOCAB_SIZE, HIDDEN_DIM=HIDDEN_DIM, SEQ_LEN=SEQ_LEN,
            MEMORY_SLOTS=MEMORY_SLOTS, NUM_PAIRS=NUM_PAIRS, STEPS=STEPS,
            BATCH=BATCH, LR=LR, LAMBDA_VALUES=LAMBDA_VALUES,
        )

        acc_by_lam = {}
        wr_by_lam = {}
        collapsed_by_lam = {}

        for lam in LAMBDA_VALUES:
            print(f"  lambda={lam} ...")
            acc, wr = train_one_lambda(lam)
            col = is_collapsed(wr)
            acc_by_lam[lam] = acc
            wr_by_lam[lam] = wr
            collapsed_by_lam[lam] = col
            print(f"    acc={acc:.4f}  write_rate={wr:.4f}  collapsed={col}")

        # Find peak among non-collapsed models
        non_collapsed = {lam: acc for lam, acc in acc_by_lam.items()
                         if not collapsed_by_lam[lam]}
        if non_collapsed:
            peak_lambda = max(non_collapsed, key=non_collapsed.__getitem__)
            peak_acc = non_collapsed[peak_lambda]
        else:
            # All collapsed — use lambda=0 baseline
            peak_lambda = 0.0
            peak_acc = acc_by_lam[0.0]

        peak_in_range = 0.01 <= peak_lambda <= 0.1

        # Check for interior peak pattern
        acc_001 = acc_by_lam.get(0.001, peak_acc)
        acc_1_0 = acc_by_lam.get(1.0, peak_acc)
        below_left = acc_001 < peak_acc - 0.005
        below_right = acc_1_0 < peak_acc - 0.005

        # Check monotone decay (acc strictly decreasing with lambda)
        sorted_lams = sorted(LAMBDA_VALUES)
        sorted_accs = [acc_by_lam[l] for l in sorted_lams]
        monotone_decay = all(
            sorted_accs[i] >= sorted_accs[i + 1]
            for i in range(len(sorted_accs) - 1)
        )

        metrics = {}
        for lam in LAMBDA_VALUES:
            key = str(lam).replace(".", "_")
            metrics[f"acc_lam_{key}"] = round(acc_by_lam[lam], 4)
            metrics[f"wr_lam_{key}"] = round(wr_by_lam[lam], 4)
            metrics[f"collapsed_lam_{key}"] = collapsed_by_lam[lam]
        metrics["peak_lambda"] = peak_lambda
        metrics["peak_acc"] = round(peak_acc, 4)
        metrics["peak_in_range"] = peak_in_range
        metrics["below_left_of_range"] = below_left
        metrics["below_right_of_range"] = below_right
        metrics["monotone_decay"] = monotone_decay

        if peak_in_range and below_left and below_right:
            outcome = OUTCOME_SUPPORTED
            notes = (
                f"Peak accuracy {peak_acc:.4f} at lambda={peak_lambda} (in [0.01, 0.1]); "
                f"lower (lambda=0.001: {acc_001:.4f}) and higher (lambda=1.0: {acc_1_0:.4f}) "
                "both degrade."
            )
        elif acc_by_lam[0.0] >= peak_acc - 0.005:
            outcome = OUTCOME_REFUTED
            notes = (
                f"Best accuracy achieved at lambda=0 ({acc_by_lam[0.0]:.4f}); "
                "auxiliary never helps."
            )
        elif monotone_decay:
            outcome = OUTCOME_INCONCLUSIVE
            notes = (
                "Accuracy monotonically decays with lambda — "
                "no interior peak found in tested range."
            )
        else:
            outcome = OUTCOME_INCONCLUSIVE
            notes = (
                f"Peak at lambda={peak_lambda} (in_range={peak_in_range}); "
                "boundary conditions not fully met."
            )

        return self.result(outcome, metrics=metrics, notes=notes, config=config)


if __name__ == "__main__":
    Exp204AuxiliaryWeightSensitivity().execute()
