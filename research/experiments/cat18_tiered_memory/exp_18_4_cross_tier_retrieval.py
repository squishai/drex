"""
exp_18_4_cross_tier_retrieval.py

Hypothesis: Simultaneous cross-tier retrieval outperforms cascaded sequential
retrieval by >3%.
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

# ── Constants ──────────────────────────────────────────────────────────────────
DEVICE      = "cpu"
LR          = 3e-4
VOCAB_SIZE  = 64
HIDDEN_DIM  = 64
FAST_SLOTS  = 16
SLOW_SLOTS  = 48
SEQ_LEN     = 64
NUM_PAIRS   = 6
STEPS       = 400
BATCH       = 32
CASCADE_T   = 0.3    # cascade threshold


# ── Data generator ─────────────────────────────────────────────────────────────
def make_long_assoc_batch(batch_size, seq_len, vocab_size, num_pairs):
    seq    = torch.zeros(batch_size, seq_len, dtype=torch.long)
    target = torch.zeros(batch_size, dtype=torch.long)
    for b in range(batch_size):
        keys = torch.randint(4, max(8, vocab_size // 3), (num_pairs * 3,)).unique()[:num_pairs]
        while len(keys) < num_pairs:
            keys = torch.cat([keys, torch.randint(4, vocab_size // 3, (1,))])[:num_pairs]
        vals = torch.randint(vocab_size // 2, vocab_size, (num_pairs,))
        pos  = 0
        for i in range(num_pairs):
            if pos + 1 < seq_len - 3:
                seq[b, pos] = keys[i]; seq[b, pos + 1] = vals[i]; pos += 2
        for p in range(pos, seq_len - 3):
            seq[b, p] = torch.randint(4, vocab_size // 3, (1,)).item()
        num_critical        = max(1, num_pairs // 2)
        qi                  = torch.randint(0, num_critical, (1,)).item()
        seq[b, seq_len - 3] = 2
        seq[b, seq_len - 2] = keys[qi]
        seq[b, seq_len - 1] = 0
        target[b]           = vals[qi]
    return seq, target


# ── Encoder ────────────────────────────────────────────────────────────────────
class Encoder(nn.Module):
    def __init__(self, vocab_size, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.ff    = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm  = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        h = self.embed(x)
        return self.norm(h + self.ff(h))


# ── Shared write mechanism (same for all three read strategies) ────────────────
class SharedWriter(nn.Module):
    """
    Fills fast (first FAST_SLOTS) and slow (next SLOW_SLOTS) using LRU + learned
    demotion.  Returns (fast_mem, slow_mem, fast_mask, slow_mask).
    """

    def __init__(self, hidden_dim, fast_slots, slow_slots):
        super().__init__()
        self.hidden_dim   = hidden_dim
        self.fast_slots   = fast_slots
        self.slow_slots   = slow_slots
        self.write_gate   = nn.Linear(hidden_dim, 1)
        self.demotion_net = nn.Linear(hidden_dim, 1)

    def forward(self, h):
        B = h.shape[0]; T = h.shape[1]
        fast_mem  = torch.zeros(B, self.fast_slots, self.hidden_dim)
        slow_mem  = torch.zeros(B, self.slow_slots, self.hidden_dim)
        fast_age  = torch.zeros(B, self.fast_slots, dtype=torch.long)
        slow_age  = torch.zeros(B, self.slow_slots, dtype=torch.long)
        fast_used = torch.zeros(B, self.fast_slots, dtype=torch.bool)
        slow_used = torch.zeros(B, self.slow_slots, dtype=torch.bool)

        for t in range(T - 3):
            tok_h = h[:, t, :]
            ws    = torch.sigmoid(self.write_gate(tok_h)).squeeze(-1)
            fast_age = fast_age + fast_used.long()
            slow_age = slow_age + slow_used.long()

            for b in range(B):
                if ws[b].item() < 0.4:
                    continue
                ff = (~fast_used[b]).nonzero(as_tuple=False)
                if ff.numel() > 0:
                    sl = ff[0, 0].item()
                    fast_mem[b, sl]  = tok_h[b].detach()
                    fast_age[b, sl]  = 0
                    fast_used[b, sl] = True
                else:
                    ds  = self.demotion_net(fast_mem[b]).squeeze(-1)
                    dem = ds.argmin().item()
                    dh  = fast_mem[b, dem].clone()
                    sf  = (~slow_used[b]).nonzero(as_tuple=False)
                    ss  = sf[0, 0].item() if sf.numel() > 0 else slow_age[b].argmax().item()
                    slow_mem[b, ss]  = dh
                    slow_age[b, ss]  = 0
                    slow_used[b, ss] = True
                    fast_mem[b, dem]  = tok_h[b].detach()
                    fast_age[b, dem]  = 0
                    fast_used[b, dem] = True

        return (fast_mem, slow_mem,
                fast_used.float(), slow_used.float())


# ── Three read strategies ──────────────────────────────────────────────────────
def attention_read(q_proj_net, out_net, query_h, memory, mask):
    q      = q_proj_net(query_h)
    scores = torch.bmm(memory, q.unsqueeze(-1)).squeeze(-1)
    scores = scores.masked_fill(mask == 0, -1e9)
    attn   = torch.softmax(scores, dim=-1)
    ctx    = (attn.unsqueeze(-1) * memory).sum(1)
    return out_net(ctx), attn


# ── Strategy A: simultaneous ───────────────────────────────────────────────────
class SimultaneousModel(nn.Module):
    def __init__(self, vocab_size, hidden_dim, fast_slots, slow_slots):
        super().__init__()
        self.encoder  = Encoder(vocab_size, hidden_dim)
        self.writer   = SharedWriter(hidden_dim, fast_slots, slow_slots)
        self.q_proj   = nn.Linear(hidden_dim, hidden_dim)
        self.out      = nn.Linear(hidden_dim, vocab_size)

    def forward(self, seq):
        B    = seq.shape[0]
        h    = self.encoder(seq)
        fm, sm, fmask, smask = self.writer(h)
        all_mem  = torch.cat([fm, sm], dim=1)
        all_mask = torch.cat([fmask, smask], dim=1)
        logits, _ = attention_read(self.q_proj, self.out, h[:, -1, :], all_mem, all_mask)
        return logits, None


# ── Strategy B: cascade ────────────────────────────────────────────────────────
class CascadeModel(nn.Module):
    """
    Query fast first.  If max attention < CASCADE_T, also query slow.
    Final output is weighted average.
    """

    def __init__(self, vocab_size, hidden_dim, fast_slots, slow_slots, threshold=CASCADE_T):
        super().__init__()
        self.encoder   = Encoder(vocab_size, hidden_dim)
        self.writer    = SharedWriter(hidden_dim, fast_slots, slow_slots)
        self.q_proj    = nn.Linear(hidden_dim, hidden_dim)
        self.out       = nn.Linear(hidden_dim, vocab_size)
        self.threshold = threshold
        self._slow_accesses = []

    def forward(self, seq):
        B    = seq.shape[0]
        h    = self.encoder(seq)
        fm, sm, fmask, smask = self.writer(h)
        query_h = h[:, -1, :]

        # fast read
        q      = self.q_proj(query_h)
        f_scores = torch.bmm(fm, q.unsqueeze(-1)).squeeze(-1)
        f_scores = f_scores.masked_fill(fmask == 0, -1e9)
        f_attn   = torch.softmax(f_scores, dim=-1)
        f_ctx    = (f_attn.unsqueeze(-1) * fm).sum(1)          # (B, H)

        max_attn = f_attn.max(dim=-1).values  # (B,)

        # slow read (where max_attn below threshold)
        s_scores = torch.bmm(sm, q.unsqueeze(-1)).squeeze(-1)
        s_scores = s_scores.masked_fill(smask == 0, -1e9)
        s_attn   = torch.softmax(s_scores, dim=-1)
        s_ctx    = (s_attn.unsqueeze(-1) * sm).sum(1)          # (B, H)

        need_slow = (max_attn < self.threshold).float().unsqueeze(-1)  # (B, 1)
        self._slow_accesses.append(need_slow.mean().item())

        ctx    = f_ctx + need_slow * s_ctx                     # (B, H)
        logits = self.out(ctx)
        return logits, need_slow.squeeze(-1)


# ── Strategy C: always-sequential ─────────────────────────────────────────────
class AlwaysSequentialModel(nn.Module):
    """
    Query fast, query slow, concatenate outputs, project to vocab.
    """

    def __init__(self, vocab_size, hidden_dim, fast_slots, slow_slots):
        super().__init__()
        self.encoder = Encoder(vocab_size, hidden_dim)
        self.writer  = SharedWriter(hidden_dim, fast_slots, slow_slots)
        self.q_proj  = nn.Linear(hidden_dim, hidden_dim)
        self.out     = nn.Linear(hidden_dim * 2, vocab_size)

    def forward(self, seq):
        B    = seq.shape[0]
        h    = self.encoder(seq)
        fm, sm, fmask, smask = self.writer(h)
        q    = self.q_proj(h[:, -1, :])

        f_scores = torch.bmm(fm, q.unsqueeze(-1)).squeeze(-1)
        f_scores = f_scores.masked_fill(fmask == 0, -1e9)
        f_attn   = torch.softmax(f_scores, dim=-1)
        f_ctx    = (f_attn.unsqueeze(-1) * fm).sum(1)

        s_scores = torch.bmm(sm, q.unsqueeze(-1)).squeeze(-1)
        s_scores = s_scores.masked_fill(smask == 0, -1e9)
        s_attn   = torch.softmax(s_scores, dim=-1)
        s_ctx    = (s_attn.unsqueeze(-1) * sm).sum(1)

        logits = self.out(torch.cat([f_ctx, s_ctx], dim=-1))
        return logits, None


# ── Train/eval helpers ─────────────────────────────────────────────────────────
def train_model(model, steps):
    opt = Adam(model.parameters(), lr=LR)
    for _ in range(steps):
        seq, target = make_long_assoc_batch(BATCH, SEQ_LEN, VOCAB_SIZE, NUM_PAIRS)
        logits, _ = model(seq)
        loss = F.cross_entropy(logits, target)
        opt.zero_grad(); loss.backward(); opt.step()
    return model


def eval_model(model, steps=400):
    correct = 0
    slow_rate_sum = 0.0
    slow_rate_count = 0
    for _ in range(steps):
        seq, target = make_long_assoc_batch(BATCH, SEQ_LEN, VOCAB_SIZE, NUM_PAIRS)
        with torch.no_grad():
            logits, slow_indicator = model(seq)
        correct += (logits.argmax(-1) == target).sum().item()
        if slow_indicator is not None:
            slow_rate_sum   += slow_indicator.mean().item()
            slow_rate_count += 1

    acc       = correct / (steps * BATCH)
    slow_rate = slow_rate_sum / slow_rate_count if slow_rate_count > 0 else 0.0
    return acc, slow_rate


# ── Experiment class ───────────────────────────────────────────────────────────
class Exp184CrossTierRetrieval(Experiment):
    experiment_id = "exp_18_4"
    hypothesis    = (
        "Simultaneous cross-tier retrieval outperforms cascaded sequential "
        "retrieval by >3%."
    )

    def run(self) -> ExperimentResult:
        config = dict(
            VOCAB_SIZE=VOCAB_SIZE, HIDDEN_DIM=HIDDEN_DIM,
            FAST_SLOTS=FAST_SLOTS, SLOW_SLOTS=SLOW_SLOTS,
            SEQ_LEN=SEQ_LEN, NUM_PAIRS=NUM_PAIRS,
            STEPS=STEPS, BATCH=BATCH, CASCADE_T=CASCADE_T,
        )

        print("Training Strategy A: simultaneous …")
        model_a = train_model(SimultaneousModel(VOCAB_SIZE, HIDDEN_DIM, FAST_SLOTS, SLOW_SLOTS), STEPS)
        acc_a, _ = eval_model(model_a)
        print(f"  acc_A = {acc_a:.4f}")

        print("Training Strategy B: cascade …")
        model_b = train_model(CascadeModel(VOCAB_SIZE, HIDDEN_DIM, FAST_SLOTS, SLOW_SLOTS), STEPS)
        acc_b, slow_rate_b = eval_model(model_b)
        print(f"  acc_B = {acc_b:.4f}  slow_access_rate = {slow_rate_b:.4f}")

        print("Training Strategy C: always-sequential …")
        model_c = train_model(AlwaysSequentialModel(VOCAB_SIZE, HIDDEN_DIM, FAST_SLOTS, SLOW_SLOTS), STEPS)
        acc_c, _ = eval_model(model_c)
        print(f"  acc_C = {acc_c:.4f}")

        best_seq  = max(acc_b, acc_c)
        gap_a_vs_b = acc_a - acc_b
        gap_a_vs_c = acc_a - acc_c
        gap_a_vs_best = acc_a - best_seq

        metrics = dict(
            acc_A=round(acc_a, 4), acc_B=round(acc_b, 4), acc_C=round(acc_c, 4),
            gap_A_vs_B=round(gap_a_vs_b, 4), gap_A_vs_C=round(gap_a_vs_c, 4),
            gap_A_vs_best_seq=round(gap_a_vs_best, 4),
            slow_access_rate_B=round(slow_rate_b, 4),
        )

        if acc_a > best_seq + 0.03:
            outcome = OUTCOME_SUPPORTED
            notes   = f"Simultaneous beats best sequential by {gap_a_vs_best:.3f}"
        elif min(gap_a_vs_b, gap_a_vs_c) >= -0.02:
            # any sequential within 0.02 of simultaneous
            outcome = OUTCOME_REFUTED
            notes   = f"Sequential within 0.02 of simultaneous (gap_best={gap_a_vs_best:.3f})"
        else:
            outcome = OUTCOME_INCONCLUSIVE
            notes   = f"Simultaneous best but gap={gap_a_vs_best:.3f} < 0.03"

        return self.result(outcome, metrics, notes, config)


if __name__ == "__main__":
    Exp184CrossTierRetrieval().execute()
