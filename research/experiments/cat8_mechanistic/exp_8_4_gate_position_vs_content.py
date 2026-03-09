"""
Experiment 8.4 — Gate Uses Position vs Content

Hypothesis: The learned write gate's small advantage over random selection (exp_1_5 +0.003)
comes from exploiting token position as proxy for importance, not from detecting semantic
content.
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
SEQ_LEN       = 24
NUM_PAIRS     = 4
MEMORY_SLOTS  = 6
BATCH_SIZE    = 32
TRAIN_STEPS   = 2000
LR            = 3e-4
DEVICE        = "cpu"

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


# ── Model variants ────────────────────────────────────────────────────────────

class EncoderFull(nn.Module):
    """Standard encoder: token embedding (content + position via embedding table)."""
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, HIDDEN_DIM)
        self.pos_embed = nn.Embedding(SEQ_LEN, HIDDEN_DIM)
        self.ff = nn.Sequential(
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM * 2), nn.ReLU(),
            nn.Linear(HIDDEN_DIM * 2, HIDDEN_DIM),
        )
        self.norm = nn.LayerNorm(HIDDEN_DIM)

    def forward(self, x):
        pos = torch.arange(x.size(1), device=x.device).unsqueeze(0).expand_as(x)
        h = self.embed(x) + self.pos_embed(pos)
        return self.norm(h + self.ff(h))


class EncoderPositionBlind(nn.Module):
    """Position-blind: no positional embedding, only token content."""
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, HIDDEN_DIM)
        self.ff = nn.Sequential(
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM * 2), nn.ReLU(),
            nn.Linear(HIDDEN_DIM * 2, HIDDEN_DIM),
        )
        self.norm = nn.LayerNorm(HIDDEN_DIM)

    def forward(self, x):
        h = self.embed(x)
        return self.norm(h + self.ff(h))


class EncoderPositionOnly(nn.Module):
    """Position-only: gate sees only positional embedding, not token identity."""
    def __init__(self):
        super().__init__()
        # Content encoder (used by read head but not by gate)
        self.embed = nn.Embedding(VOCAB_SIZE, HIDDEN_DIM)
        self.ff = nn.Sequential(
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM * 2), nn.ReLU(),
            nn.Linear(HIDDEN_DIM * 2, HIDDEN_DIM),
        )
        self.norm = nn.LayerNorm(HIDDEN_DIM)
        # Position-only representation for gate
        self.pos_embed = nn.Embedding(SEQ_LEN, HIDDEN_DIM)

    def forward(self, x):
        h = self.embed(x)
        h = self.norm(h + self.ff(h))
        pos = torch.arange(x.size(1), device=x.device).unsqueeze(0).expand_as(x)
        pos_h = self.pos_embed(pos)
        return h, pos_h  # content hidden, position hidden


class WriteGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(HIDDEN_DIM, 1)

    def forward(self, h):
        return torch.sigmoid(self.fc(h)).squeeze(-1)  # (B, L)


class ReadHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.out = nn.Linear(HIDDEN_DIM, VOCAB_SIZE)

    def forward(self, query_h, memory, mask):
        q = self.q_proj(query_h)
        scores = torch.bmm(memory, q.unsqueeze(-1)).squeeze(-1)
        scores = scores.masked_fill(mask == 0, -1e9)
        attn = torch.softmax(scores, dim=-1)
        ctx = (attn.unsqueeze(-1) * memory).sum(1)
        return self.out(ctx)


# ── Pearson r ─────────────────────────────────────────────────────────────────

def pearson_r(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.float().view(-1)
    y = y.float().view(-1)
    xm = x - x.mean()
    ym = y - y.mean()
    denom = (xm.norm() * ym.norm()).clamp(min=1e-8)
    return ((xm * ym).sum() / denom).item()


# ── Training and eval ─────────────────────────────────────────────────────────

def train_and_eval(condition: str):
    """Train condition A/B/C and return (acc, pos_corr, content_corr)."""
    gate = WriteGate().to(DEVICE)
    rh   = ReadHead().to(DEVICE)

    if condition == "A":
        enc = EncoderFull().to(DEVICE)
        def get_hidden(x):
            return enc(x), enc(x)  # (content, gate input)
        def gate_input(x):
            return enc(x)
        params = list(enc.parameters()) + list(gate.parameters()) + list(rh.parameters())
    elif condition == "B":
        enc = EncoderPositionBlind().to(DEVICE)
        def gate_input(x):
            return enc(x)
        params = list(enc.parameters()) + list(gate.parameters()) + list(rh.parameters())
    else:  # C
        enc = EncoderPositionOnly().to(DEVICE)
        def gate_input(x):
            _, pos_h = enc(x)
            return pos_h
        params = list(enc.parameters()) + list(gate.parameters()) + list(rh.parameters())

    opt = Adam(params, lr=LR)

    for step in range(TRAIN_STEPS):
        seq, target, _ = make_assoc_batch(BATCH_SIZE)

        if condition in ("A", "B"):
            h = gate_input(seq)
            ws = gate(h)
        else:
            content_h, pos_h = enc(seq)
            ws = gate(pos_h)
            h = content_h

        k = min(MEMORY_SLOTS, SEQ_LEN)
        topk_idx = ws.topk(k, dim=1).indices
        B, L, H = h.shape
        memory = h.gather(1, topk_idx.unsqueeze(-1).expand(-1, -1, H))
        mask = torch.ones(B, k, device=DEVICE)

        logits = rh(h[:, -1, :], memory, mask)
        loss = F.cross_entropy(logits, target)
        opt.zero_grad()
        loss.backward()
        opt.step()

    # Eval
    if condition in ("A", "B"):
        enc.eval()
    else:
        enc.eval()
    gate.eval(); rh.eval()

    all_ws, all_pos_idx, all_kv_mask = [], [], []
    total_acc = 0.0
    n_eval = 50

    with torch.no_grad():
        for _ in range(n_eval):
            seq, target, kv_mask = make_assoc_batch(BATCH_SIZE)
            pos_indices = torch.arange(SEQ_LEN, dtype=torch.float).unsqueeze(0).expand(BATCH_SIZE, -1)

            if condition == "A":
                h = gate_input(seq)
                ws = gate(h)
            elif condition == "B":
                h = gate_input(seq)
                ws = gate(h)
            else:
                content_h, pos_h = enc(seq)
                ws = gate(pos_h)
                h = content_h

            all_ws.append(ws)
            all_pos_idx.append(pos_indices)
            all_kv_mask.append(kv_mask)

            k = min(MEMORY_SLOTS, SEQ_LEN)
            topk_idx = ws.topk(k, dim=1).indices
            B, L, H = h.shape
            memory = h.gather(1, topk_idx.unsqueeze(-1).expand(-1, -1, H))
            mask = torch.ones(B, k, device=DEVICE)
            logits = rh(h[:, -1, :], memory, mask)
            total_acc += (logits.argmax(-1) == target).float().mean().item()

    ws_all  = torch.cat(all_ws)
    pos_all = torch.cat(all_pos_idx)
    kv_all  = torch.cat(all_kv_mask)

    pos_corr     = pearson_r(ws_all, pos_all)
    content_corr = pearson_r(ws_all, kv_all)
    acc          = total_acc / n_eval

    return acc, pos_corr, content_corr


# ── Experiment class ──────────────────────────────────────────────────────────

class Exp84GatePositionVsContent(Experiment):
    experiment_id = "exp_8_4"
    hypothesis = (
        "The learned write gate's small advantage over random selection (exp_1_5 +0.003) "
        "comes from exploiting token position as proxy for importance, not from detecting "
        "semantic content."
    )

    def run(self) -> ExperimentResult:
        res = {}
        for cond in ("A", "B", "C"):
            print(f"  Condition {cond} ...")
            acc, pos_corr, content_corr = train_and_eval(cond)
            res[cond] = {"acc": acc, "pos_corr": pos_corr, "content_corr": content_corr}
            print(f"    acc={acc:.3f}, pos_corr={pos_corr:.4f}, content_corr={content_corr:.4f}")

        acc_a = res["A"]["acc"]
        acc_b = res["B"]["acc"]
        pos_corr_a = res["A"]["pos_corr"]

        if acc_a - acc_b > 0.005 and pos_corr_a > 0.2:
            outcome = OUTCOME_SUPPORTED
        elif acc_b >= acc_a - 0.002:
            outcome = OUTCOME_REFUTED
        else:
            outcome = OUTCOME_INCONCLUSIVE

        metrics = {
            "acc_A":           round(res["A"]["acc"], 4),
            "acc_B":           round(res["B"]["acc"], 4),
            "acc_C":           round(res["C"]["acc"], 4),
            "pos_corr_A":      round(res["A"]["pos_corr"], 4),
            "pos_corr_B":      round(res["B"]["pos_corr"], 4),
            "pos_corr_C":      round(res["C"]["pos_corr"], 4),
            "content_corr_A":  round(res["A"]["content_corr"], 4),
            "content_corr_B":  round(res["B"]["content_corr"], 4),
            "content_corr_C":  round(res["C"]["content_corr"], 4),
        }
        notes = (
            f"A (full): acc={res['A']['acc']:.3f}, pos_r={res['A']['pos_corr']:.4f}. "
            f"B (pos-blind): acc={res['B']['acc']:.3f}. "
            f"C (pos-only): acc={res['C']['acc']:.3f}. "
            f"acc_A - acc_B = {acc_a - acc_b:.4f}."
        )
        config = {
            "vocab_size": VOCAB_SIZE, "hidden_dim": HIDDEN_DIM,
            "seq_len": SEQ_LEN, "num_pairs": NUM_PAIRS,
            "memory_slots": MEMORY_SLOTS, "train_steps": TRAIN_STEPS,
        }
        return self.result(outcome, metrics, notes, config)


if __name__ == "__main__":
    Exp84GatePositionVsContent().execute()
