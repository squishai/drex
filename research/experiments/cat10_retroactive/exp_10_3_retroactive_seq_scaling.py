from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from experiments.base import (Experiment, ExperimentResult, OUTCOME_SUPPORTED, OUTCOME_REFUTED, OUTCOME_INCONCLUSIVE)

experiment_id = "exp_10_3"
hypothesis = "The retroactive writing accuracy gain scales with sequence length (Pearson r > 0.8 across seq_len 24, 32, 48, 64)."

DEVICE = "cpu"
LR = 3e-4
VOCAB_SIZE = 64
HIDDEN_DIM = 48
TRAIN_STEPS = 500
BATCH_SIZE = 32
SEQ_LENS = [24, 32, 48, 64]


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


class ScalingModel(nn.Module):
    def __init__(self, vocab_size, hidden_dim, memory_slots, forward_slots, retroactive):
        super().__init__()
        self.encoder = Encoder(vocab_size, hidden_dim)
        self.forward_gate = nn.Linear(hidden_dim, 1)
        self.read_head = ReadHead(hidden_dim, vocab_size)
        self.retroactive = retroactive
        self.forward_slots = forward_slots
        self.retro_slots = memory_slots - forward_slots
        self.memory_slots = memory_slots
        self.hidden_dim = hidden_dim
        if retroactive:
            self.revision_gate = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            )

    def forward(self, seq):
        B, T = seq.shape
        hidden = self.encoder(seq)
        candidate_hidden = hidden[:, :-3, :]
        n_cand = candidate_hidden.shape[1]
        fwd_scores = self.forward_gate(candidate_hidden).squeeze(-1)
        k = min(self.forward_slots, n_cand)
        _, fwd_idx = fwd_scores.topk(k, dim=1)
        fwd_idx_sorted, _ = fwd_idx.sort(dim=1)

        memory_list = []
        masks_list = []

        for b in range(B):
            fwd_set = set(fwd_idx_sorted[b].tolist())
            fwd_h = hidden[b][fwd_idx_sorted[b]]
            context = hidden[b].mean(0)

            if self.retroactive:
                skipped = [i for i in range(n_cand) if i not in fwd_set]
                if skipped:
                    # VECTORIZED: build all gate_inputs at once
                    gate_inputs_b = torch.stack([
                        torch.cat([hidden[b, t], context], dim=-1) for t in skipped
                    ])  # (n_skipped, 2H)
                    scores = self.revision_gate(gate_inputs_b).squeeze(-1)  # (n_skipped,)
                    top_idx = scores.topk(min(self.retro_slots, len(skipped))).indices.tolist()
                    selected_t = [skipped[i] for i in top_idx]
                    retro_h = torch.stack([hidden[b, t] for t in selected_t])
                    slots = torch.cat([fwd_h, retro_h], dim=0)
                else:
                    slots = fwd_h
            else:
                slots = fwd_h

            n = slots.shape[0]
            if n < self.memory_slots:
                pad = torch.zeros(self.memory_slots - n, self.hidden_dim, device=seq.device)
                slots = torch.cat([slots, pad], dim=0)
                mask = torch.cat([torch.ones(n), torch.zeros(self.memory_slots - n)]).to(seq.device)
            else:
                slots = slots[:self.memory_slots]
                mask = torch.ones(self.memory_slots, device=seq.device)

            memory_list.append(slots)
            masks_list.append(mask)

        memory = torch.stack(memory_list)
        mask = torch.stack(masks_list)
        query_h = hidden[:, -2, :]
        return self.read_head(query_h, memory, mask)


def pearson_r(xs, ys):
    n = len(xs)
    mx = sum(xs) / n; my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denom_x = (sum((x - mx) ** 2 for x in xs)) ** 0.5
    denom_y = (sum((y - my) ** 2 for y in ys)) ** 0.5
    if denom_x < 1e-9 or denom_y < 1e-9:
        return 0.0
    return num / (denom_x * denom_y)


def train_and_eval_scaling(seq_len, retroactive, seed_offset=0):
    torch.manual_seed(42 + seed_offset)
    num_pairs = max(4, seq_len // 6)
    memory_slots = seq_len // 4
    forward_slots = memory_slots - 2

    model = ScalingModel(VOCAB_SIZE, HIDDEN_DIM, memory_slots, forward_slots, retroactive).to(DEVICE)
    opt = Adam(model.parameters(), lr=LR)
    model.train()
    for step in range(TRAIN_STEPS):
        seq, target = make_assoc_batch(BATCH_SIZE, seq_len, VOCAB_SIZE, num_pairs)
        seq, target = seq.to(DEVICE), target.to(DEVICE)
        logits = model(seq)
        loss = F.cross_entropy(logits, target)
        opt.zero_grad(); loss.backward(); opt.step()

    model.eval()
    correct = 0; total = 0
    with torch.no_grad():
        for _ in range(20):
            seq, target = make_assoc_batch(BATCH_SIZE, seq_len, VOCAB_SIZE, num_pairs)
            seq, target = seq.to(DEVICE), target.to(DEVICE)
            logits = model(seq)
            preds = logits.argmax(dim=-1)
            correct += (preds == target).sum().item()
            total += target.shape[0]
    return correct / total


class Exp103RetroactiveSeqScaling(Experiment):
    experiment_id = "exp_10_3"
    hypothesis = "The retroactive writing accuracy gain scales with sequence length (Pearson r > 0.8 across seq_len 24, 32, 48, 64)."

    def run(self) -> ExperimentResult:
        config = dict(VOCAB_SIZE=VOCAB_SIZE, HIDDEN_DIM=HIDDEN_DIM,
                      TRAIN_STEPS=TRAIN_STEPS, BATCH_SIZE=BATCH_SIZE, SEQ_LENS=SEQ_LENS)
        gaps = []
        acc_fwd_all = {}
        acc_retro_all = {}

        for i, sl in enumerate(SEQ_LENS):
            print(f"  seq_len={sl}: training forward ...")
            acc_fwd = train_and_eval_scaling(sl, retroactive=False, seed_offset=i * 2)
            print(f"    acc_fwd={acc_fwd:.4f}")
            print(f"  seq_len={sl}: training retroactive ...")
            acc_retro = train_and_eval_scaling(sl, retroactive=True, seed_offset=i * 2 + 1)
            print(f"    acc_retro={acc_retro:.4f}")
            gap = acc_retro - acc_fwd
            gaps.append(gap)
            acc_fwd_all[sl] = acc_fwd
            acc_retro_all[sl] = acc_retro
            print(f"    gap={gap:.4f}")

        r = pearson_r(SEQ_LENS, gaps)
        gap_at_24 = gaps[0]
        gap_at_64 = gaps[-1]

        metrics = {f"acc_fwd_{sl}": round(acc_fwd_all[sl], 4) for sl in SEQ_LENS}
        metrics.update({f"acc_retro_{sl}": round(acc_retro_all[sl], 4) for sl in SEQ_LENS})
        metrics.update({f"gap_{sl}": round(g, 4) for sl, g in zip(SEQ_LENS, gaps)})
        metrics["pearson_r"] = round(r, 4)

        if r > 0.8:
            outcome = OUTCOME_SUPPORTED
            notes = f"Pearson r={r:.3f}>0.8 — gain scales with sequence length."
        elif gap_at_64 < gap_at_24:
            outcome = OUTCOME_REFUTED
            notes = f"gap@64={gap_at_64:.3f} < gap@24={gap_at_24:.3f} — gain does not scale."
        else:
            outcome = OUTCOME_INCONCLUSIVE
            notes = f"Pearson r={r:.3f} — scaling relationship inconclusive."

        return self.result(outcome=outcome, metrics=metrics, notes=notes, config=config)


if __name__ == "__main__":
    Exp103RetroactiveSeqScaling().execute()
