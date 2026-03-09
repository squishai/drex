from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from experiments.base import (Experiment, ExperimentResult, OUTCOME_SUPPORTED, OUTCOME_REFUTED, OUTCOME_INCONCLUSIVE)

experiment_id = "exp_10_1"
hypothesis = "The retroactive writing benefit decays to <5% accuracy gain when the revision gate's lookahead window is fewer than 6 tokens."

DEVICE = "cpu"
LR = 3e-4
VOCAB_SIZE = 64
SEQ_LEN = 24
HIDDEN_DIM = 64
MEMORY_SLOTS = 6
FORWARD_SLOTS = 4
RETRO_SLOTS = 2
NUM_PAIRS = 4
TRAIN_STEPS = 500
BATCH_SIZE = 32
WINDOW_SIZES = [0, 2, 4, 6, 8, 24]


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
        seq[b, seq_len - 3] = 2; seq[b, seq_len - 2] = keys[qi]; seq[b, seq_len - 1] = 0
        target[b] = vals[qi]
    return seq, target


class Encoder(nn.Module):
    def __init__(self, vocab_size, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.ff = nn.Sequential(nn.Linear(hidden_dim, hidden_dim * 2), nn.ReLU(), nn.Linear(hidden_dim * 2, hidden_dim))
        self.norm = nn.LayerNorm(hidden_dim)
    def forward(self, x):
        h = self.embed(x); return self.norm(h + self.ff(h))


class ReadHead(nn.Module):
    def __init__(self, hidden_dim, vocab_size):
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


class LookaheadModel(nn.Module):
    def __init__(self, vocab_size, hidden_dim, memory_slots, forward_slots, retro_slots, window):
        super().__init__()
        self.encoder = Encoder(vocab_size, hidden_dim)
        self.forward_gate = nn.Linear(hidden_dim, 1)
        self.read_head = ReadHead(hidden_dim, vocab_size)
        self.window = window
        self.forward_slots = forward_slots
        self.retro_slots = retro_slots
        self.memory_slots = memory_slots
        self.hidden_dim = hidden_dim
        if window > 0:
            self.revision_gate = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            )

    def forward(self, seq):
        B, T = seq.shape
        hidden = self.encoder(seq)  # (B, T, H)
        fwd_scores = self.forward_gate(hidden[:, :-3, :]).squeeze(-1)  # (B, T-3)
        n_candidates = fwd_scores.shape[1]
        k = min(self.forward_slots, n_candidates)
        _, fwd_idx = fwd_scores.topk(k, dim=1)
        fwd_idx_sorted, _ = fwd_idx.sort(dim=1)

        # Build forward memory
        memory_list = []
        fwd_set = []
        for b in range(B):
            slots = hidden[b][fwd_idx_sorted[b]]  # (k, H)
            fwd_set.append(set(fwd_idx_sorted[b].tolist()))
            memory_list.append(slots)

        if self.window > 0:
            retro_memory_list = []
            for b in range(B):
                skipped = [i for i in range(n_candidates) if i not in fwd_set[b]]
                if not skipped:
                    retro_memory_list.append(memory_list[b])
                    continue
                # VECTORIZED: build all gate_inputs at once for this batch item
                gate_inputs_b = []
                for t in skipped:
                    t_end = min(t + 1 + self.window, T - 3)
                    if t_end > t + 1:
                        ctx = hidden[b, t + 1:t_end].mean(0)
                    else:
                        ctx = hidden[b, t]
                    gate_inputs_b.append(torch.cat([hidden[b, t], ctx], dim=-1))
                gate_inputs_b = torch.stack(gate_inputs_b)  # (n_skipped, 2H)
                scores = self.revision_gate(gate_inputs_b).squeeze(-1)  # (n_skipped,)
                top_idx = scores.topk(min(self.retro_slots, len(skipped))).indices
                retro_hidden = torch.stack([hidden[b, skipped[i]] for i in top_idx.tolist()])
                combined = torch.cat([memory_list[b], retro_hidden], dim=0)
                retro_memory_list.append(combined)
            memory_list = retro_memory_list

        # Pad to fixed memory_slots
        padded = []
        masks = []
        for b in range(B):
            m = memory_list[b]
            n = m.shape[0]
            if n < self.memory_slots:
                pad = torch.zeros(self.memory_slots - n, self.hidden_dim, device=seq.device)
                m = torch.cat([m, pad], dim=0)
                mask = torch.cat([torch.ones(n), torch.zeros(self.memory_slots - n)]).to(seq.device)
            else:
                m = m[:self.memory_slots]
                mask = torch.ones(self.memory_slots, device=seq.device)
            padded.append(m)
            masks.append(mask)

        memory = torch.stack(padded)   # (B, slots, H)
        mask = torch.stack(masks)      # (B, slots)
        query_h = hidden[:, -2, :]     # query token is seq[-2]
        logits = self.read_head(query_h, memory, mask)
        return logits


def train_and_eval(window, seed_offset=0):
    torch.manual_seed(42 + seed_offset)
    model = LookaheadModel(VOCAB_SIZE, HIDDEN_DIM, MEMORY_SLOTS, FORWARD_SLOTS, RETRO_SLOTS, window).to(DEVICE)
    opt = Adam(model.parameters(), lr=LR)
    model.train()
    for step in range(TRAIN_STEPS):
        seq, target = make_assoc_batch(BATCH_SIZE, SEQ_LEN, VOCAB_SIZE, NUM_PAIRS)
        seq, target = seq.to(DEVICE), target.to(DEVICE)
        logits = model(seq)
        loss = F.cross_entropy(logits, target)
        opt.zero_grad(); loss.backward(); opt.step()

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for _ in range(20):
            seq, target = make_assoc_batch(BATCH_SIZE, SEQ_LEN, VOCAB_SIZE, NUM_PAIRS)
            seq, target = seq.to(DEVICE), target.to(DEVICE)
            logits = model(seq)
            preds = logits.argmax(dim=-1)
            correct += (preds == target).sum().item()
            total += target.shape[0]
    return correct / total


class Exp101RetroactiveLookahead(Experiment):
    experiment_id = "exp_10_1"
    hypothesis = "The retroactive writing benefit decays to <5% accuracy gain when the revision gate's lookahead window is fewer than 6 tokens."

    def run(self) -> ExperimentResult:
        config = dict(VOCAB_SIZE=VOCAB_SIZE, SEQ_LEN=SEQ_LEN, HIDDEN_DIM=HIDDEN_DIM,
                      MEMORY_SLOTS=MEMORY_SLOTS, FORWARD_SLOTS=FORWARD_SLOTS,
                      RETRO_SLOTS=RETRO_SLOTS, NUM_PAIRS=NUM_PAIRS,
                      TRAIN_STEPS=TRAIN_STEPS, BATCH_SIZE=BATCH_SIZE,
                      WINDOW_SIZES=WINDOW_SIZES)
        accs = {}
        for i, w in enumerate(WINDOW_SIZES):
            print(f"  Training window={w} ...")
            acc = train_and_eval(w, seed_offset=i)
            accs[w] = acc
            print(f"    acc={acc:.4f}")

        baseline = accs[0]
        gaps = {w: accs[w] - baseline for w in WINDOW_SIZES}

        gap_at_24 = gaps[24]
        gap_at_4 = gaps[4]
        max_gap = max(gaps.values())
        min_gap = min(gaps.values())

        metrics = {f"acc_w{w}": round(accs[w], 4) for w in WINDOW_SIZES}
        metrics.update({f"gap_w{w}": round(gaps[w], 4) for w in WINDOW_SIZES})
        metrics["gap_at_24"] = round(gap_at_24, 4)
        metrics["gap_at_4"] = round(gap_at_4, 4)
        metrics["max_gap_minus_min_gap"] = round(max_gap - min_gap, 4)

        if gap_at_24 > 0.08 and gap_at_4 < 0.04:
            outcome = OUTCOME_SUPPORTED
            notes = f"gap@24={gap_at_24:.3f}>0.08 and gap@4={gap_at_4:.3f}<0.04 — hypothesis supported."
        elif (max_gap - min_gap) < 0.03:
            outcome = OUTCOME_REFUTED
            notes = f"Gap is flat (max-min={max_gap - min_gap:.3f}<0.03) — no benefit from lookahead."
        else:
            outcome = OUTCOME_INCONCLUSIVE
            notes = f"gap@24={gap_at_24:.3f}, gap@4={gap_at_4:.3f} — pattern inconclusive."

        return self.result(outcome=outcome, metrics=metrics, notes=notes, config=config)


if __name__ == "__main__":
    Exp101RetroactiveLookahead().execute()
