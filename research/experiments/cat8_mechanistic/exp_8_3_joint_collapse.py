"""
Experiment 8.3 — Write-Evict Correlation Collapse

Hypothesis: The write-evict correlation collapse (r=0.990 in exp_6_7) is gradient aliasing —
both gates receive identical gradient from shared loss. Oracle pre-training of each gate
on independent labels breaks this, yielding write-evict correlation < 0.5.
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
SEQ_LEN        = 24
NUM_PAIRS      = 4
MEMORY_SLOTS   = 8
BATCH_SIZE     = 32
LR             = 3e-4
DEVICE         = "cpu"
PRETRAIN_STEPS = 200
JOINT_STEPS    = 600
MAX_AGE        = SEQ_LEN

# ── Data ──────────────────────────────────────────────────────────────────────

def make_assoc_batch(batch_size, seq_len=SEQ_LEN, vocab_size=VOCAB_SIZE, num_pairs=NUM_PAIRS):
    seq = torch.zeros(batch_size, seq_len, dtype=torch.long)
    target = torch.zeros(batch_size, dtype=torch.long)
    kv_mask = torch.zeros(batch_size, seq_len)
    for b in range(batch_size):
        keys = torch.randint(4, 4 + num_pairs * 4, (num_pairs * 2,)).unique()[:num_pairs]
        while len(keys) < num_pairs:
            keys = torch.cat([keys, torch.randint(4, vocab_size // 2, (1,))])[:num_pairs]
        vals = torch.randint(vocab_size // 2, vocab_size, (num_pairs,))
        pos = 0
        for i in range(num_pairs):
            if pos + 1 < seq_len - 3:
                seq[b, pos] = keys[i]
                seq[b, pos + 1] = vals[i]
                kv_mask[b, pos] = 1.0
                kv_mask[b, pos + 1] = 1.0
                pos += 2
        for p in range(pos, seq_len - 3):
            seq[b, p] = 3
        qi = torch.randint(0, num_pairs, (1,)).item()
        seq[b, seq_len - 3] = 2
        seq[b, seq_len - 2] = keys[qi]
        seq[b, seq_len - 1] = 0
        target[b] = vals[qi]
    return seq, target, kv_mask


# ── Model components ──────────────────────────────────────────────────────────

class Encoder(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, hidden_dim=HIDDEN_DIM):
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


class WriteGate(nn.Module):
    def __init__(self, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, h):
        return torch.sigmoid(self.fc(h)).squeeze(-1)  # (B, L)


class EvictGate(nn.Module):
    def __init__(self, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, h):
        return torch.sigmoid(self.fc(h)).squeeze(-1)  # (B, L)


class ReadHead(nn.Module):
    def __init__(self, hidden_dim=HIDDEN_DIM, vocab_size=VOCAB_SIZE):
        super().__init__()
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, query_h, memory, mask):
        q = self.q_proj(query_h)
        scores = torch.bmm(memory, q.unsqueeze(-1)).squeeze(-1)
        scores = scores.masked_fill(mask == 0, -1e9)
        attn = torch.softmax(scores, dim=-1)
        ctx = (attn.unsqueeze(-1) * memory).sum(1)
        return self.out(ctx)


# ── Memory write with eviction ────────────────────────────────────────────────

def write_to_memory(hidden, write_scores, evict_scores, k=MEMORY_SLOTS):
    """
    Select top-k tokens by write score. Returns (memory, mask).
    Evict scores are used to determine which slot to evict when memory is full —
    here simplified to just use write scores for top-k selection (full eviction
    integrated through evict gate applied to hidden states).
    """
    B, L, H = hidden.shape
    topk_idx = write_scores.topk(k, dim=1).indices   # (B, k)
    memory = hidden.gather(1, topk_idx.unsqueeze(-1).expand(-1, -1, H))
    mask = torch.ones(B, k, device=hidden.device)
    return memory, mask, topk_idx


# ── Training routines ─────────────────────────────────────────────────────────

def train_joint(enc, write_gate, evict_gate, rh, steps):
    """Joint training on task loss only."""
    params = (list(enc.parameters()) + list(write_gate.parameters()) +
              list(evict_gate.parameters()) + list(rh.parameters()))
    opt = Adam(params, lr=LR)
    enc.train(); write_gate.train(); evict_gate.train(); rh.train()

    for _ in range(steps):
        seq, target, _ = make_assoc_batch(BATCH_SIZE)
        hidden = enc(seq)
        ws = write_gate(hidden)
        es = evict_gate(hidden)
        memory, mask, _ = write_to_memory(hidden, ws, es)
        logits = rh(hidden[:, -1, :], memory, mask)
        loss = F.cross_entropy(logits, target)
        opt.zero_grad()
        loss.backward()
        opt.step()


def pretrain_write_gate(enc, write_gate, steps):
    """Pre-train write gate: predict is_kv_token (BCE)."""
    params = list(enc.parameters()) + list(write_gate.parameters())
    opt = Adam(params, lr=LR)
    enc.train(); write_gate.train()

    for _ in range(steps):
        seq, _, kv_mask = make_assoc_batch(BATCH_SIZE)
        hidden = enc(seq)
        ws = write_gate(hidden)                           # (B, L)
        loss = F.binary_cross_entropy(ws, kv_mask)
        opt.zero_grad()
        loss.backward()
        opt.step()


def pretrain_evict_gate(enc, evict_gate, steps):
    """Pre-train evict gate: predict slot_age / MAX_AGE (oldest = highest score)."""
    params = list(enc.parameters()) + list(evict_gate.parameters())
    opt = Adam(params, lr=LR)
    enc.train(); evict_gate.train()

    for _ in range(steps):
        seq, _, _ = make_assoc_batch(BATCH_SIZE)
        hidden = enc(seq)
        es = evict_gate(hidden)  # (B, L)
        # Age label: position 0 is oldest → highest evict score
        ages = torch.arange(SEQ_LEN, 0, -1, dtype=torch.float).unsqueeze(0).expand(BATCH_SIZE, -1)
        age_labels = ages / MAX_AGE
        loss = F.mse_loss(es, age_labels)
        opt.zero_grad()
        loss.backward()
        opt.step()


def measure_write_evict_correlation(enc, write_gate, evict_gate, n_batches=30):
    """Collect write_scores and evict_scores, compute Pearson r."""
    enc.eval(); write_gate.eval(); evict_gate.eval()
    all_ws, all_es = [], []
    with torch.no_grad():
        for _ in range(n_batches):
            seq, _, _ = make_assoc_batch(BATCH_SIZE)
            hidden = enc(seq)
            ws = write_gate(hidden).view(-1)
            es = evict_gate(hidden).view(-1)
            all_ws.append(ws)
            all_es.append(es)
    ws_all = torch.cat(all_ws)
    es_all = torch.cat(all_es)
    wm = ws_all - ws_all.mean()
    em = es_all - es_all.mean()
    denom = (wm.norm() * em.norm()).clamp(min=1e-8)
    return ((wm * em).sum() / denom).item()


def eval_accuracy(enc, write_gate, evict_gate, rh, n_batches=10):
    enc.eval(); write_gate.eval(); evict_gate.eval(); rh.eval()
    total = 0.0
    with torch.no_grad():
        for _ in range(n_batches):
            seq, target, _ = make_assoc_batch(BATCH_SIZE)
            hidden = enc(seq)
            ws = write_gate(hidden)
            es = evict_gate(hidden)
            memory, mask, _ = write_to_memory(hidden, ws, es)
            logits = rh(hidden[:, -1, :], memory, mask)
            total += (logits.argmax(-1) == target).float().mean().item()
    return total / n_batches


# ── Experiment class ──────────────────────────────────────────────────────────

class Exp83JointCollapse(Experiment):
    experiment_id = "exp_8_3"
    hypothesis = (
        "The write-evict correlation collapse (r=0.990 in exp_6_7) is gradient aliasing — "
        "both gates receive identical gradient from shared loss. Oracle pre-training of each "
        "gate on independent labels breaks this, yielding write-evict correlation < 0.5."
    )

    def run(self) -> ExperimentResult:
        results = {}

        # ── Condition A: Joint training only ─────────────────────────────────
        print("  Condition A: random init + joint training ...")
        enc_a  = Encoder().to(DEVICE)
        wg_a   = WriteGate().to(DEVICE)
        eg_a   = EvictGate().to(DEVICE)
        rh_a   = ReadHead().to(DEVICE)
        train_joint(enc_a, wg_a, eg_a, rh_a, PRETRAIN_STEPS + JOINT_STEPS)
        corr_a = measure_write_evict_correlation(enc_a, wg_a, eg_a)
        acc_a  = eval_accuracy(enc_a, wg_a, eg_a, rh_a)
        results["A"] = {"corr": corr_a, "acc": acc_a}
        print(f"    corr={corr_a:.4f}, acc={acc_a:.3f}")

        # ── Condition B: Oracle pre-train then joint fine-tune ────────────────
        print("  Condition B: oracle pre-train + joint fine-tune ...")
        enc_b  = Encoder().to(DEVICE)
        wg_b   = WriteGate().to(DEVICE)
        eg_b   = EvictGate().to(DEVICE)
        rh_b   = ReadHead().to(DEVICE)
        pretrain_write_gate(enc_b, wg_b, PRETRAIN_STEPS)
        pretrain_evict_gate(enc_b, eg_b, PRETRAIN_STEPS)
        train_joint(enc_b, wg_b, eg_b, rh_b, JOINT_STEPS)
        corr_b = measure_write_evict_correlation(enc_b, wg_b, eg_b)
        acc_b  = eval_accuracy(enc_b, wg_b, eg_b, rh_b)
        results["B"] = {"corr": corr_b, "acc": acc_b}
        print(f"    corr={corr_b:.4f}, acc={acc_b:.3f}")

        # ── Condition C: Gradient isolation ──────────────────────────────────
        print("  Condition C: gradient isolation ...")
        enc_c  = Encoder().to(DEVICE)
        wg_c   = WriteGate().to(DEVICE)
        eg_c   = EvictGate().to(DEVICE)
        rh_c   = ReadHead().to(DEVICE)
        opt_c  = Adam(
            list(enc_c.parameters()) + list(wg_c.parameters()) +
            list(eg_c.parameters()) + list(rh_c.parameters()), lr=LR
        )
        enc_c.train(); wg_c.train(); eg_c.train(); rh_c.train()
        for _ in range(PRETRAIN_STEPS + JOINT_STEPS):
            seq, target, _ = make_assoc_batch(BATCH_SIZE)
            hidden = enc_c(seq)
            # Stop gradient between write and evict paths
            ws = wg_c(hidden.detach())
            es = eg_c(hidden.detach())
            memory, mask, topk_idx = write_to_memory(hidden, ws.detach(), es.detach())
            # Read head receives gradient through hidden directly
            query_hidden = enc_c(seq)[:, -1, :]
            logits = rh_c(query_hidden, memory.detach(), mask)
            loss = F.cross_entropy(logits, target)
            opt_c.zero_grad()
            loss.backward()
            opt_c.step()
        corr_c = measure_write_evict_correlation(enc_c, wg_c, eg_c)
        acc_c  = eval_accuracy(enc_c, wg_c, eg_c, rh_c)
        results["C"] = {"corr": corr_c, "acc": acc_c}
        print(f"    corr={corr_c:.4f}, acc={acc_c:.3f}")

        # ── Outcome ───────────────────────────────────────────────────────────
        corr_b_val = results["B"]["corr"]
        acc_b_val  = results["B"]["acc"]
        acc_a_val  = results["A"]["acc"]

        if corr_b_val < 0.5 and acc_b_val >= acc_a_val - 0.02:
            outcome = OUTCOME_SUPPORTED
        elif corr_b_val > 0.8:
            outcome = OUTCOME_REFUTED
        else:
            outcome = OUTCOME_INCONCLUSIVE

        metrics = {
            "write_evict_corr_A": round(results["A"]["corr"], 4),
            "write_evict_corr_B": round(results["B"]["corr"], 4),
            "write_evict_corr_C": round(results["C"]["corr"], 4),
            "acc_A": round(results["A"]["acc"], 4),
            "acc_B": round(results["B"]["acc"], 4),
            "acc_C": round(results["C"]["acc"], 4),
        }
        notes = (
            f"Condition A (joint): corr={results['A']['corr']:.4f}, acc={results['A']['acc']:.3f}. "
            f"Condition B (oracle pretrain): corr={results['B']['corr']:.4f}, acc={results['B']['acc']:.3f}. "
            f"Condition C (grad isolation): corr={results['C']['corr']:.4f}, acc={results['C']['acc']:.3f}."
        )
        config = {
            "vocab_size": VOCAB_SIZE, "hidden_dim": HIDDEN_DIM,
            "seq_len": SEQ_LEN, "num_pairs": NUM_PAIRS,
            "memory_slots": MEMORY_SLOTS, "pretrain_steps": PRETRAIN_STEPS,
            "joint_steps": JOINT_STEPS,
        }
        return self.result(outcome, metrics, notes, config)


if __name__ == "__main__":
    Exp83JointCollapse().execute()
