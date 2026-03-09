"""
Experiment 9.4 — Protected Slots Extended: Interior Optimum

Hypothesis: With MEMORY_SLOTS=12 and K extended to 0-10, an interior optimum exists at
K=3-6 — fewer protected slots are insufficient to cover all 3 critical items, more wastes
capacity on non-critical entries, creating a U-shaped accuracy curve.
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
SEQ_LEN       = 32
MEMORY_SLOTS  = 12
NUM_PAIRS     = 5
NUM_CRITICAL  = 3
BATCH_SIZE    = 32
TRAIN_STEPS   = 500
LR            = 3e-4
DEVICE        = "cpu"
K_VALUES      = [0, 2, 4, 6, 8, 10]   # even sweep from 0 to 10

# ── Data ──────────────────────────────────────────────────────────────────────

def make_assoc_batch_protected(batch_size, seq_len=SEQ_LEN, vocab_size=VOCAB_SIZE,
                                num_pairs=NUM_PAIRS, num_critical=NUM_CRITICAL):
    """
    Critical KV pairs placed at positions 0-5 (positions 0,1,2,3,4,5).
    Non-critical KV pairs placed after that.
    Returns seq, target, critical_mask, query_is_critical.
    """
    seq = torch.zeros(batch_size, seq_len, dtype=torch.long)
    target = torch.zeros(batch_size, dtype=torch.long)
    critical_mask = torch.zeros(batch_size, seq_len)
    query_is_critical = torch.zeros(batch_size, dtype=torch.bool)

    for b in range(batch_size):
        keys = torch.randint(4, vocab_size // 2, (num_pairs * 2,)).unique()[:num_pairs]
        while len(keys) < num_pairs:
            keys = torch.cat([keys, torch.randint(4, vocab_size // 2, (1,))])[:num_pairs]
        vals = torch.randint(vocab_size // 2, vocab_size, (num_pairs,))

        pos = 0
        for i in range(num_pairs):
            if pos + 1 < seq_len - 3:
                seq[b, pos]     = keys[i]
                seq[b, pos + 1] = vals[i]
                if i < num_critical:
                    critical_mask[b, pos]     = 1.0
                    critical_mask[b, pos + 1] = 1.0
                pos += 2

        for p in range(pos, seq_len - 3):
            seq[b, p] = 3

        qi = torch.randint(0, num_pairs, (1,)).item()
        seq[b, seq_len - 3] = 2
        seq[b, seq_len - 2] = keys[qi]
        seq[b, seq_len - 1] = 0
        target[b] = vals[qi]
        query_is_critical[b] = (qi < num_critical)

    return seq, target, critical_mask, query_is_critical


# ── Model components ──────────────────────────────────────────────────────────

class Encoder(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2), nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        h = self.embed(x)
        return self.norm(h + self.ff(h))


class WriteGate(nn.Module):
    def __init__(self, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, h):
        return torch.sigmoid(self.fc(h)).squeeze(-1)


class ReadHead(nn.Module):
    def __init__(self, hidden_dim=HIDDEN_DIM, vocab_size=VOCAB_SIZE, memory_slots=MEMORY_SLOTS):
        super().__init__()
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out    = nn.Linear(hidden_dim, vocab_size)

    def forward(self, query_h, memory, mask):
        q = self.q_proj(query_h)
        scores = torch.bmm(memory, q.unsqueeze(-1)).squeeze(-1)
        scores = scores.masked_fill(mask == 0, -1e9)
        attn = torch.softmax(scores, dim=-1)
        ctx  = (attn.unsqueeze(-1) * memory).sum(1)
        return self.out(ctx)


# ── Memory with K protected slots ─────────────────────────────────────────────

def select_memory_slots(hidden, write_scores, critical_mask, k_protected):
    """
    Select MEMORY_SLOTS tokens into memory:
    - First K slots reserved for top-K critical tokens
    - Remaining (MEMORY_SLOTS - K) slots from non-critical tokens by write score
    """
    B, L, H = hidden.shape
    total_slots = min(MEMORY_SLOTS, L)
    k_open = total_slots - k_protected

    all_indices = []
    for b in range(B):
        critical_positions = (critical_mask[b] > 0.5).nonzero(as_tuple=True)[0]
        non_critical_pos   = (critical_mask[b] <= 0.5).nonzero(as_tuple=True)[0]

        # Protected slots: top-k_protected critical positions by write score
        if k_protected > 0 and len(critical_positions) > 0:
            crit_scores = write_scores[b, critical_positions]
            n_prot = min(k_protected, len(critical_positions))
            top_crit = crit_scores.topk(n_prot).indices
            prot_idx = critical_positions[top_crit]
        else:
            prot_idx = torch.tensor([], dtype=torch.long)

        # Open slots: top-k_open remaining positions by write score
        if k_open > 0 and len(non_critical_pos) > 0:
            nc_scores = write_scores[b, non_critical_pos]
            n_open = min(k_open, len(non_critical_pos))
            top_nc = nc_scores.topk(n_open).indices
            open_idx = non_critical_pos[top_nc]
        else:
            open_idx = torch.tensor([], dtype=torch.long)

        combined = torch.cat([prot_idx, open_idx])
        # Pad to total_slots if needed
        if len(combined) < total_slots:
            # Fill remaining from any position
            remaining = total_slots - len(combined)
            all_pos = torch.arange(L)
            already = combined
            mask_used = torch.zeros(L, dtype=torch.bool)
            mask_used[already] = True
            unused = all_pos[~mask_used][:remaining]
            combined = torch.cat([combined, unused])
        all_indices.append(combined[:total_slots])

    indices = torch.stack(all_indices)  # (B, total_slots)
    memory = hidden.gather(1, indices.unsqueeze(-1).expand(-1, -1, H))
    mask_out = torch.ones(B, total_slots, device=DEVICE)
    return memory, mask_out


# ── Training and eval ─────────────────────────────────────────────────────────

def train_and_eval_k(k_protected: int) -> dict:
    enc  = Encoder().to(DEVICE)
    gate = WriteGate().to(DEVICE)
    rh   = ReadHead().to(DEVICE)
    opt  = Adam(list(enc.parameters()) + list(gate.parameters()) + list(rh.parameters()), lr=LR)

    enc.train(); gate.train(); rh.train()
    for step in range(TRAIN_STEPS):
        seq, target, crit_mask, _ = make_assoc_batch_protected(BATCH_SIZE)
        hidden = enc(seq)
        ws     = gate(hidden)
        memory, mask = select_memory_slots(hidden, ws, crit_mask, k_protected)
        logits = rh(hidden[:, -1, :], memory, mask)
        loss   = F.cross_entropy(logits, target)
        opt.zero_grad()
        loss.backward()
        opt.step()

    # Eval
    enc.eval(); gate.eval(); rh.eval()
    total_acc = 0.0
    crit_acc  = 0.0
    nc_acc    = 0.0
    crit_n = 0; nc_n = 0
    n_eval = 50

    with torch.no_grad():
        for _ in range(n_eval):
            seq, target, crit_mask, q_is_crit = make_assoc_batch_protected(BATCH_SIZE)
            hidden = enc(seq)
            ws     = gate(hidden)
            memory, mask = select_memory_slots(hidden, ws, crit_mask, k_protected)
            logits = rh(hidden[:, -1, :], memory, mask)
            preds  = logits.argmax(-1)
            correct = (preds == target)

            total_acc += correct.float().mean().item()
            crit_correct = correct[q_is_crit]
            nc_correct   = correct[~q_is_crit]
            if len(crit_correct) > 0:
                crit_acc += crit_correct.float().mean().item()
                crit_n   += 1
            if len(nc_correct) > 0:
                nc_acc += nc_correct.float().mean().item()
                nc_n   += 1

    return {
        "acc":      total_acc / n_eval,
        "crit_acc": crit_acc / max(crit_n, 1),
        "nc_acc":   nc_acc / max(nc_n, 1),
    }


# ── Experiment class ──────────────────────────────────────────────────────────

class Exp94ProtectedSlotsExtended(Experiment):
    experiment_id = "exp_9_4"
    hypothesis = (
        "With MEMORY_SLOTS=12 and K extended to 0-10, an interior optimum exists at "
        "K=3-6 — fewer protected slots are insufficient to cover all 3 critical items, "
        "more wastes capacity on non-critical entries, creating a U-shaped accuracy curve."
    )

    def run(self) -> ExperimentResult:
        accs = []
        crit_accs = []
        nc_accs   = []

        for k in K_VALUES:
            print(f"  K={k} protected slots ...")
            res = train_and_eval_k(k)
            accs.append(res["acc"])
            crit_accs.append(res["crit_acc"])
            nc_accs.append(res["nc_acc"])
            print(f"    acc={res['acc']:.3f}, crit_acc={res['crit_acc']:.3f}, nc_acc={res['nc_acc']:.3f}")

        acc_tensor = torch.tensor(accs)
        max_acc    = acc_tensor.max().item()
        min_acc    = acc_tensor.min().item()
        k_opt      = acc_tensor.argmax().item()

        # Interior peak: optimal K is in {2..8}, and both endpoints are lower
        interior_peak_exists = (
            2 <= k_opt <= 8
            and accs[0]  < max_acc - 0.02
            and accs[-1] < max_acc - 0.02
        )

        # Monotone: acc[10] is max
        is_monotone = (k_opt == 10)
        # Flat
        is_flat = (max_acc - min_acc) < 0.03

        if interior_peak_exists:
            outcome = OUTCOME_SUPPORTED
        elif is_monotone:
            outcome = OUTCOME_REFUTED
        else:
            outcome = OUTCOME_INCONCLUSIVE

        metrics = {
            "accuracies":            [round(a, 4) for a in accs],
            "critical_accuracies":   [round(a, 4) for a in crit_accs],
            "noncritical_accuracies":[round(a, 4) for a in nc_accs],
            "k_opt":                 int(k_opt),
            "max_acc":               round(max_acc, 4),
            "min_acc":               round(min_acc, 4),
            "acc_range":             round(max_acc - min_acc, 4),
            "interior_peak_exists":  interior_peak_exists,
            "is_monotone":           is_monotone,
            "is_flat":               is_flat,
        }
        notes = (
            f"K values {K_VALUES}, accs {[round(a,3) for a in accs]}. "
            f"Optimal K={k_opt}, max_acc={max_acc:.3f}. "
            f"Interior peak: {interior_peak_exists}."
        )
        config = {
            "vocab_size": VOCAB_SIZE, "hidden_dim": HIDDEN_DIM,
            "seq_len": SEQ_LEN, "memory_slots": MEMORY_SLOTS,
            "num_pairs": NUM_PAIRS, "num_critical": NUM_CRITICAL,
            "train_steps": TRAIN_STEPS, "k_values": K_VALUES,
        }
        return self.result(outcome, metrics, notes, config)


if __name__ == "__main__":
    Exp94ProtectedSlotsExtended().execute()
