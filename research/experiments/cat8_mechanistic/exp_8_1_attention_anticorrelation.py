"""
Experiment 8.1 — Attention Anti-Correlation Mechanistic Analysis

Hypothesis: The attention-importance anti-correlation (r=-0.503 from exp_1_1) is caused
by softmax normalization forcing zero-sum redistribution, not a semantic mismatch —
removing normalization will produce positive or near-zero correlation.
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

VOCAB_SIZE  = 64
HIDDEN_DIM  = 32
SEQ_LEN     = 24
NUM_PAIRS   = 4
BATCH_SIZE  = 32
TRAIN_STEPS = 1000
LR          = 3e-4
DEVICE      = "cpu"

# ── Data ──────────────────────────────────────────────────────────────────────

def make_assoc_batch(batch_size, seq_len=SEQ_LEN, vocab_size=VOCAB_SIZE, num_pairs=NUM_PAIRS):
    seq = torch.zeros(batch_size, seq_len, dtype=torch.long)
    target = torch.zeros(batch_size, dtype=torch.long)
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
                pos += 2
        for p in range(pos, seq_len - 3):
            seq[b, p] = 3  # padding token
        qi = torch.randint(0, num_pairs, (1,)).item()
        seq[b, seq_len - 3] = 2  # query marker
        seq[b, seq_len - 2] = keys[qi]
        seq[b, seq_len - 1] = 0  # blank to predict
        target[b] = vals[qi]
    return seq, target


def make_importance_mask(seq, vocab_size=VOCAB_SIZE, num_pairs=NUM_PAIRS):
    """
    Returns a (B, L) binary mask: 1 where the token is a key or value
    (i.e., in the first 2*num_pairs positions and not padding/special).
    """
    B, L = seq.shape
    mask = torch.zeros(B, L)
    kv_end = num_pairs * 2
    for b in range(B):
        for pos in range(min(kv_end, L - 3)):
            tok = seq[b, pos].item()
            if tok >= 4:  # not special token
                mask[b, pos] = 1.0
    return mask


# ── Model ─────────────────────────────────────────────────────────────────────

class SmallEncoder(nn.Module):
    """2-head cross-attention over the sequence; query = last token."""
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, HIDDEN_DIM)
        self.ff = nn.Sequential(
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM * 2),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM * 2, HIDDEN_DIM),
        )
        self.norm = nn.LayerNorm(HIDDEN_DIM)
        self.q_proj = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.k_proj = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.v_proj = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.out_proj = nn.Linear(HIDDEN_DIM, VOCAB_SIZE)
        self.num_heads = 2
        self.head_dim = HIDDEN_DIM // self.num_heads

    def forward(self, x):
        # x: (B, L)
        h = self.embed(x)               # (B, L, H)
        h = self.norm(h + self.ff(h))   # (B, L, H)

        # Query = last token, keys/values = all positions
        query = h[:, -1:, :]            # (B, 1, H)
        B, L, H = h.shape

        Q = self.q_proj(query)          # (B, 1, H)
        K = self.k_proj(h)              # (B, L, H)
        V = self.v_proj(h)              # (B, L, H)

        # Raw dot product scores: (B, 1, L)
        raw_scores = torch.bmm(Q, K.transpose(1, 2)) / (H ** 0.5)  # (B, 1, L)
        raw_scores = raw_scores.squeeze(1)  # (B, L)

        # Softmax attention weights
        softmax_attn = torch.softmax(raw_scores, dim=-1)  # (B, L)

        # Context for output
        ctx = (softmax_attn.unsqueeze(-1) * V).sum(1)  # (B, H)
        logits = self.out_proj(ctx)                     # (B, vocab)
        return logits, raw_scores, softmax_attn


# ── Pearson correlation ───────────────────────────────────────────────────────

def pearson_r(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.float().view(-1)
    y = y.float().view(-1)
    xm = x - x.mean()
    ym = y - y.mean()
    denom = (xm.norm() * ym.norm()).clamp(min=1e-8)
    return ((xm * ym).sum() / denom).item()


def entropy_normalized(attn: torch.Tensor) -> torch.Tensor:
    """Normalize each row of attn by its per-sequence entropy (avoid div-by-zero)."""
    eps = 1e-8
    # attn: (B, L)
    ent = -(attn * (attn + eps).log()).sum(-1, keepdim=True)  # (B, 1)
    ent = ent.clamp(min=eps)
    return attn / ent


# ── Experiment class ──────────────────────────────────────────────────────────

class Exp81AttentionAntiCorrelation(Experiment):
    experiment_id = "exp_8_1"
    hypothesis = (
        "The attention-importance anti-correlation (r=-0.503 from exp_1_1) is caused by "
        "softmax normalization forcing zero-sum redistribution, not a semantic mismatch — "
        "removing normalization will produce positive or near-zero correlation."
    )

    def run(self) -> ExperimentResult:
        model = SmallEncoder().to(DEVICE)
        opt = Adam(model.parameters(), lr=LR)

        print("  Training encoder on associative recall ...")
        model.train()
        for step in range(TRAIN_STEPS):
            seq, target = make_assoc_batch(BATCH_SIZE)
            logits, _, _ = model(seq)
            loss = F.cross_entropy(logits, target)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if (step + 1) % 200 == 0:
                print(f"    step {step+1}/{TRAIN_STEPS}  loss={loss.item():.4f}")

        # ── Measure correlations ──────────────────────────────────────────────
        print("  Measuring correlations ...")
        model.eval()
        all_softmax, all_raw, all_entropic, all_imp = [], [], [], []
        with torch.no_grad():
            for _ in range(100):
                seq, _ = make_assoc_batch(BATCH_SIZE)
                imp = make_importance_mask(seq)  # (B, L)
                _, raw_scores, softmax_attn = model(seq)
                entropic = entropy_normalized(softmax_attn)

                all_softmax.append(softmax_attn)
                all_raw.append(raw_scores)
                all_entropic.append(entropic)
                all_imp.append(imp)

        softmax_all = torch.cat(all_softmax, dim=0)   # (N, L)
        raw_all     = torch.cat(all_raw, dim=0)        # (N, L)
        entropic_all= torch.cat(all_entropic, dim=0)   # (N, L)
        imp_all     = torch.cat(all_imp, dim=0)        # (N, L)

        r_softmax  = pearson_r(softmax_all, imp_all)
        r_raw      = pearson_r(raw_all, imp_all)
        r_entropic = pearson_r(entropic_all, imp_all)

        print(f"  Pearson r — softmax={r_softmax:.4f}, raw={r_raw:.4f}, entropic={r_entropic:.4f}")

        # ── Outcome ───────────────────────────────────────────────────────────
        if r_raw > 0.05 and r_softmax < -0.10:
            outcome = OUTCOME_SUPPORTED
        elif r_raw < -0.10:
            outcome = OUTCOME_REFUTED
        elif abs(r_raw - r_softmax) < 0.10:
            outcome = OUTCOME_INCONCLUSIVE
        else:
            outcome = OUTCOME_INCONCLUSIVE

        metrics = {
            "pearson_r_softmax":  round(r_softmax, 4),
            "pearson_r_raw":      round(r_raw, 4),
            "pearson_r_entropic": round(r_entropic, 4),
        }
        notes = (
            f"Softmax r={r_softmax:.4f}, raw dot-product r={r_raw:.4f}, "
            f"entropy-normalized r={r_entropic:.4f}. "
            f"Hypothesis: raw should be > 0.05 while softmax < -0.10."
        )
        config = {
            "vocab_size": VOCAB_SIZE, "hidden_dim": HIDDEN_DIM,
            "seq_len": SEQ_LEN, "num_pairs": NUM_PAIRS,
            "train_steps": TRAIN_STEPS, "batch_size": BATCH_SIZE,
        }
        return self.result(outcome, metrics, notes, config)


if __name__ == "__main__":
    Exp81AttentionAntiCorrelation().execute()
