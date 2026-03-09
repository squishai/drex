"""
exp_18_2_demotion_policy_interpretability.py

Hypothesis: Learned demotion controller discovers frequency-not-recency policy
(corr_access > 0.15, corr_recency < -0.10).
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
SEQ_LEN     = 64
FAST_SLOTS  = 16
SLOW_SLOTS  = 64
NUM_PAIRS   = 6
STEPS       = 500
BATCH       = 32
EVAL_SEQS   = 100


# ── Data generator (same as exp_18_1) ─────────────────────────────────────────
def make_long_assoc_batch(batch_size, seq_len, vocab_size, num_pairs):
    seq    = torch.zeros(batch_size, seq_len, dtype=torch.long)
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
            seq[b, p] = torch.randint(4, vocab_size // 3, (1,)).item()

        num_critical        = max(1, num_pairs // 2)
        qi                  = torch.randint(0, num_critical, (1,)).item()
        seq[b, seq_len - 3] = 2
        seq[b, seq_len - 2] = keys[qi]
        seq[b, seq_len - 1] = 0
        target[b]           = vals[qi]

    return seq, target


# ── Encoder / ReadHead ─────────────────────────────────────────────────────────
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


class ReadHead(nn.Module):
    def __init__(self, hidden_dim, vocab_size):
        super().__init__()
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out    = nn.Linear(hidden_dim, vocab_size)

    def forward(self, query_h, memory, mask):
        q      = self.q_proj(query_h)
        scores = torch.bmm(memory, q.unsqueeze(-1)).squeeze(-1)
        scores = scores.masked_fill(mask == 0, -1e9)
        attn   = torch.softmax(scores, dim=-1)
        ctx    = (attn.unsqueeze(-1) * memory).sum(1)
        return self.out(ctx)


# ── Tiered model (same architecture as exp_18_1 condition B) ─────────────────
class TieredModel(nn.Module):
    def __init__(self, vocab_size, hidden_dim, fast_slots, slow_slots):
        super().__init__()
        self.encoder      = Encoder(vocab_size, hidden_dim)
        self.write_gate   = nn.Linear(hidden_dim, 1)
        self.demotion_net = nn.Linear(hidden_dim, 1)
        self.read_head    = ReadHead(hidden_dim, vocab_size)
        self.hidden_dim   = hidden_dim
        self.fast_slots   = fast_slots
        self.slow_slots   = slow_slots

    def _process(self, h):
        B = h.shape[0]; T = h.shape[1]
        fast_mem  = torch.zeros(B, self.fast_slots, self.hidden_dim)
        slow_mem  = torch.zeros(B, self.slow_slots, self.hidden_dim)
        fast_age  = torch.zeros(B, self.fast_slots, dtype=torch.long)
        slow_age  = torch.zeros(B, self.slow_slots, dtype=torch.long)
        fast_used = torch.zeros(B, self.fast_slots, dtype=torch.bool)
        slow_used = torch.zeros(B, self.slow_slots, dtype=torch.bool)

        for t in range(T - 3):
            tok_h       = h[:, t, :]
            write_score = torch.sigmoid(self.write_gate(tok_h)).squeeze(-1)
            fast_age    = fast_age + fast_used.long()
            slow_age    = slow_age + slow_used.long()

            for b in range(B):
                if write_score[b].item() < 0.4:
                    continue
                fast_free = (~fast_used[b]).nonzero(as_tuple=False)
                if fast_free.numel() > 0:
                    slot = fast_free[0, 0].item()
                    fast_mem[b, slot]  = tok_h[b].detach()
                    fast_age[b, slot]  = 0
                    fast_used[b, slot] = True
                else:
                    dem_scores = self.demotion_net(fast_mem[b]).squeeze(-1)
                    dem_slot   = dem_scores.argmin().item()
                    demoted_h  = fast_mem[b, dem_slot].clone()
                    slow_free  = (~slow_used[b]).nonzero(as_tuple=False)
                    ss         = slow_free[0, 0].item() if slow_free.numel() > 0 else slow_age[b].argmax().item()
                    slow_mem[b, ss]  = demoted_h
                    slow_age[b, ss]  = 0
                    slow_used[b, ss] = True
                    fast_mem[b, dem_slot]  = tok_h[b].detach()
                    fast_age[b, dem_slot]  = 0
                    fast_used[b, dem_slot] = True

        all_mem  = torch.cat([fast_mem, slow_mem], dim=1)
        all_used = torch.cat([fast_used, slow_used], dim=1).float()
        return all_mem, all_used

    def forward(self, seq):
        h       = self.encoder(seq)
        mem, mask = self._process(h)
        return self.read_head(h[:, -1, :], mem, mask)


# ── Training ───────────────────────────────────────────────────────────────────
def train_model(steps):
    model = TieredModel(VOCAB_SIZE, HIDDEN_DIM, FAST_SLOTS, SLOW_SLOTS)
    opt   = Adam(model.parameters(), lr=LR)
    for _ in range(steps):
        seq, target = make_long_assoc_batch(BATCH, SEQ_LEN, VOCAB_SIZE, NUM_PAIRS)
        loss = F.cross_entropy(model(seq), target)
        opt.zero_grad(); loss.backward(); opt.step()
    return model


# ── Analysis: track per-slot metadata during eval ─────────────────────────────
def analyze_demotion(model, eval_seqs):
    """
    Run eval_seqs individual sequences (batch_size=1).
    Tracks fast-slot metadata and records demotion events.

    Returns: demotion_scores, access_counts, recency_values, content_norms
    """
    demotion_scores_list = []
    access_counts_list   = []
    recency_list         = []
    content_norms_list   = []

    global_step = 0

    for _ in range(eval_seqs):
        seq, target = make_long_assoc_batch(1, SEQ_LEN, VOCAB_SIZE, NUM_PAIRS)
        B = 1; T = SEQ_LEN

        with torch.no_grad():
            h = model.encoder(seq)  # (1, T, H)

        fast_mem          = torch.zeros(1, FAST_SLOTS, HIDDEN_DIM)
        fast_age          = torch.zeros(1, FAST_SLOTS, dtype=torch.long)
        fast_used         = torch.zeros(1, FAST_SLOTS, dtype=torch.bool)
        fast_access_count = torch.zeros(1, FAST_SLOTS, dtype=torch.long)
        fast_last_access  = torch.full((1, FAST_SLOTS), -1, dtype=torch.long)
        slow_mem          = torch.zeros(1, SLOW_SLOTS, HIDDEN_DIM)
        slow_used         = torch.zeros(1, SLOW_SLOTS, dtype=torch.bool)
        slow_age          = torch.zeros(1, SLOW_SLOTS, dtype=torch.long)

        for t in range(T - 3):
            tok_h       = h[:, t, :]
            write_score = torch.sigmoid(model.write_gate(tok_h)).squeeze(-1)
            fast_age    = fast_age + fast_used.long()
            slow_age    = slow_age + slow_used.long()

            # Simulate an access: query reads from fast memory
            if fast_used[0].any():
                q      = model.read_head.q_proj(tok_h[0])
                scores = fast_mem[0] @ q                    # (fast_slots,)
                scores = scores.masked_fill(~fast_used[0], -1e9)
                attn   = torch.softmax(scores, dim=-1)
                accessed_slot = attn.argmax().item()
                fast_access_count[0, accessed_slot] += 1
                fast_last_access[0, accessed_slot]   = global_step

            if write_score[0].item() < 0.4:
                global_step += 1
                continue

            fast_free = (~fast_used[0]).nonzero(as_tuple=False)
            if fast_free.numel() > 0:
                slot = fast_free[0, 0].item()
                fast_mem[0, slot]          = tok_h[0].detach()
                fast_age[0, slot]          = 0
                fast_used[0, slot]         = True
                fast_access_count[0, slot] = 0
                fast_last_access[0, slot]  = global_step
            else:
                dem_scores_raw = model.demotion_net(fast_mem[0]).squeeze(-1)  # (fast_slots,)
                dem_slot       = dem_scores_raw.argmin().item()

                # Record this demotion event
                ds  = dem_scores_raw[dem_slot].item()
                ac  = fast_access_count[0, dem_slot].item()
                la  = fast_last_access[0, dem_slot].item()
                rec = global_step - la if la >= 0 else global_step
                cn  = fast_mem[0, dem_slot].norm().item()

                demotion_scores_list.append(ds)
                access_counts_list.append(ac)
                recency_list.append(rec)
                content_norms_list.append(cn)

                # Demote and write new
                demoted_h = fast_mem[0, dem_slot].clone()
                sf        = (~slow_used[0]).nonzero(as_tuple=False)
                ss        = sf[0, 0].item() if sf.numel() > 0 else slow_age[0].argmax().item()
                slow_mem[0, ss]  = demoted_h
                slow_age[0, ss]  = 0
                slow_used[0, ss] = True

                fast_mem[0, dem_slot]          = tok_h[0].detach()
                fast_age[0, dem_slot]          = 0
                fast_used[0, dem_slot]         = True
                fast_access_count[0, dem_slot] = 0
                fast_last_access[0, dem_slot]  = global_step

            global_step += 1

    return (demotion_scores_list, access_counts_list,
            recency_list, content_norms_list)


# ── Pearson correlation ────────────────────────────────────────────────────────
def pearson_r(xs, ys):
    if len(xs) < 3:
        return 0.0
    x = torch.tensor(xs, dtype=torch.float32)
    y = torch.tensor(ys, dtype=torch.float32)
    mx = x.mean(); my = y.mean()
    sx = x.std(unbiased=False); sy = y.std(unbiased=False)
    if sx < 1e-8 or sy < 1e-8:
        return 0.0
    return (((x - mx) * (y - my)).mean() / (sx * sy)).item()


# ── Experiment class ───────────────────────────────────────────────────────────
class Exp182DemotionPolicyInterpretability(Experiment):
    experiment_id = "exp_18_2"
    hypothesis    = (
        "Learned demotion controller discovers frequency-not-recency policy "
        "(corr_access > 0.15, corr_recency < -0.10)."
    )

    def run(self) -> ExperimentResult:
        config = dict(
            VOCAB_SIZE=VOCAB_SIZE, HIDDEN_DIM=HIDDEN_DIM, SEQ_LEN=SEQ_LEN,
            FAST_SLOTS=FAST_SLOTS, SLOW_SLOTS=SLOW_SLOTS,
            NUM_PAIRS=NUM_PAIRS, STEPS=STEPS, BATCH=BATCH, EVAL_SEQS=EVAL_SEQS,
        )

        print("Training tiered memory model (same as exp_18_1 condition B) …")
        model = train_model(STEPS)

        print("Running demotion analysis …")
        dem_scores, access_counts, recency, content_norms = analyze_demotion(model, EVAL_SEQS)
        n_events = len(dem_scores)
        print(f"  Recorded {n_events} demotion events")

        if n_events < 5:
            return self.result(
                OUTCOME_INCONCLUSIVE,
                {"n_demotion_events": n_events},
                "Too few demotion events recorded for reliable correlation",
                config,
            )

        corr_access  = pearson_r(dem_scores, access_counts)
        corr_recency = pearson_r(dem_scores, recency)
        corr_norm    = pearson_r(dem_scores, content_norms)

        print(f"  corr(dem_score, access_count) = {corr_access:.4f}")
        print(f"  corr(dem_score, recency)      = {corr_recency:.4f}")
        print(f"  corr(dem_score, content_norm) = {corr_norm:.4f}")

        metrics = dict(
            n_demotion_events=n_events,
            corr_access=round(corr_access, 4),
            corr_recency=round(corr_recency, 4),
            corr_content_norm=round(corr_norm, 4),
        )

        if corr_access > 0.15 and corr_recency < -0.10:
            outcome = OUTCOME_SUPPORTED
            notes   = (f"Frequency-not-recency confirmed: "
                       f"corr_access={corr_access:.3f}, corr_recency={corr_recency:.3f}")
        elif all(abs(c) < 0.05 for c in [corr_access, corr_recency, corr_norm]):
            outcome = OUTCOME_REFUTED
            notes   = "All correlations < 0.05 — demotion appears random"
        else:
            outcome = OUTCOME_INCONCLUSIVE
            notes   = (f"corr_access={corr_access:.3f} > 0.15 but "
                       f"corr_recency={corr_recency:.3f} > -0.10")

        return self.result(outcome, metrics, notes, config)


if __name__ == "__main__":
    Exp182DemotionPolicyInterpretability().execute()
