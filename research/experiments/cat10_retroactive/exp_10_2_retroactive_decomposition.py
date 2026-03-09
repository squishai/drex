from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from experiments.base import (Experiment, ExperimentResult, OUTCOME_SUPPORTED, OUTCOME_REFUTED, OUTCOME_INCONCLUSIVE)

experiment_id = "exp_10_2"
hypothesis = "The retroactive writing benefit comes primarily (>80%) from adding new entries never written in the forward pass, not from re-encoding existing forward-pass entries with full context."

DEVICE = "cpu"
LR = 3e-4
VOCAB_SIZE = 64
SEQ_LEN = 24
HIDDEN_DIM = 64
MEMORY_SLOTS = 6
FORWARD_SLOTS = 4
NUM_PAIRS = 4
TRAIN_STEPS = 600
BATCH_SIZE = 32


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


class DecompModel(nn.Module):
    """
    condition: 'A' forward-only, 'B' new-write retro, 'C' overwrite retro, 'D' combined
    """
    def __init__(self, vocab_size, hidden_dim, memory_slots, forward_slots, condition):
        super().__init__()
        self.encoder = Encoder(vocab_size, hidden_dim)
        self.forward_gate = nn.Linear(hidden_dim, 1)
        self.read_head = ReadHead(hidden_dim, vocab_size)
        self.condition = condition
        self.forward_slots = forward_slots
        self.memory_slots = memory_slots
        self.hidden_dim = hidden_dim
        self.retro_slots = memory_slots - forward_slots  # 2

        if condition in ('B', 'D'):
            self.new_write_gate = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            )
        if condition in ('C', 'D'):
            self.re_encoder_attn = nn.MultiheadAttention(hidden_dim, num_heads=1, batch_first=True)

    def forward(self, seq):
        B, T = seq.shape
        hidden = self.encoder(seq)  # (B, T, H)
        candidate_hidden = hidden[:, :-3, :]  # exclude query tokens
        n_cand = candidate_hidden.shape[1]

        fwd_scores = self.forward_gate(candidate_hidden).squeeze(-1)  # (B, n_cand)
        k = min(self.forward_slots, n_cand)
        _, fwd_idx = fwd_scores.topk(k, dim=1)
        fwd_idx_sorted, _ = fwd_idx.sort(dim=1)

        memory_list = []
        masks_list = []

        for b in range(B):
            fwd_set = set(fwd_idx_sorted[b].tolist())
            fwd_hidden_b = hidden[b][fwd_idx_sorted[b]]  # (k, H)

            # Context: mean pool of all hidden states
            context = hidden[b].mean(0)  # (H,)

            if self.condition == 'A':
                slots = fwd_hidden_b
            elif self.condition == 'B':
                # New write: score skipped tokens using context
                skipped = [i for i in range(n_cand) if i not in fwd_set]
                if skipped:
                    # VECTORIZED: build all gate_inputs at once
                    gate_inputs_b = torch.stack([
                        torch.cat([hidden[b, t], context], dim=-1) for t in skipped
                    ])  # (n_skipped, 2H)
                    scores = self.new_write_gate(gate_inputs_b).squeeze(-1)  # (n_skipped,)
                    top_idx = scores.topk(min(self.retro_slots, len(skipped))).indices.tolist()
                    selected_t = [skipped[i] for i in top_idx]
                    retro_h = torch.stack([hidden[b, t] for t in selected_t])
                    slots = torch.cat([fwd_hidden_b, retro_h], dim=0)
                else:
                    slots = fwd_hidden_b
            elif self.condition == 'C':
                # Re-encode forward slots with full-context attention
                all_h = hidden[b].unsqueeze(0)  # (1, T, H)
                query = fwd_hidden_b.unsqueeze(0)  # (1, k, H)
                re_encoded, _ = self.re_encoder_attn(query, all_h, all_h)
                slots = re_encoded.squeeze(0)  # (k, H)
            elif self.condition == 'D':
                # Both: re-encode forward slots + add new writes
                all_h = hidden[b].unsqueeze(0)
                query = fwd_hidden_b.unsqueeze(0)
                re_encoded, _ = self.re_encoder_attn(query, all_h, all_h)
                re_slots = re_encoded.squeeze(0)

                skipped = [i for i in range(n_cand) if i not in fwd_set]
                if skipped:
                    # VECTORIZED: build all gate_inputs at once
                    gate_inputs_b = torch.stack([
                        torch.cat([hidden[b, t], context], dim=-1) for t in skipped
                    ])  # (n_skipped, 2H)
                    scores = self.new_write_gate(gate_inputs_b).squeeze(-1)  # (n_skipped,)
                    top_idx = scores.topk(min(self.retro_slots, len(skipped))).indices.tolist()
                    selected_t = [skipped[i] for i in top_idx]
                    retro_h = torch.stack([hidden[b, t] for t in selected_t])
                    slots = torch.cat([re_slots, retro_h], dim=0)
                else:
                    slots = re_slots
            else:
                raise ValueError(f"Unknown condition {self.condition}")

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
        logits = self.read_head(query_h, memory, mask)
        return logits


def train_condition(condition, seed_offset=0):
    torch.manual_seed(42 + seed_offset)
    model = DecompModel(VOCAB_SIZE, HIDDEN_DIM, MEMORY_SLOTS, FORWARD_SLOTS, condition).to(DEVICE)
    opt = Adam(model.parameters(), lr=LR)
    model.train()
    for step in range(TRAIN_STEPS):
        seq, target = make_assoc_batch(BATCH_SIZE, SEQ_LEN, VOCAB_SIZE, NUM_PAIRS)
        seq, target = seq.to(DEVICE), target.to(DEVICE)
        logits = model(seq)
        loss = F.cross_entropy(logits, target)
        opt.zero_grad(); loss.backward(); opt.step()

    model.eval()
    correct = 0; total = 0
    with torch.no_grad():
        for _ in range(20):
            seq, target = make_assoc_batch(BATCH_SIZE, SEQ_LEN, VOCAB_SIZE, NUM_PAIRS)
            seq, target = seq.to(DEVICE), target.to(DEVICE)
            logits = model(seq)
            preds = logits.argmax(dim=-1)
            correct += (preds == target).sum().item()
            total += target.shape[0]
    return correct / total


class Exp102RetroactiveDecomposition(Experiment):
    experiment_id = "exp_10_2"
    hypothesis = "The retroactive writing benefit comes primarily (>80%) from adding new entries never written in the forward pass, not from re-encoding existing forward-pass entries with full context."

    def run(self) -> ExperimentResult:
        config = dict(VOCAB_SIZE=VOCAB_SIZE, SEQ_LEN=SEQ_LEN, HIDDEN_DIM=HIDDEN_DIM,
                      MEMORY_SLOTS=MEMORY_SLOTS, FORWARD_SLOTS=FORWARD_SLOTS,
                      NUM_PAIRS=NUM_PAIRS, TRAIN_STEPS=TRAIN_STEPS, BATCH_SIZE=BATCH_SIZE)
        results = {}
        for i, cond in enumerate(['A', 'B', 'C', 'D']):
            print(f"  Training condition {cond} ...")
            acc = train_condition(cond, seed_offset=i)
            results[cond] = acc
            print(f"    acc_{cond}={acc:.4f}")

        acc_A, acc_B, acc_C, acc_D = results['A'], results['B'], results['C'], results['D']
        new_write_fraction = (acc_B - acc_A) / max((acc_D - acc_A), 0.001)

        metrics = {
            "acc_A": round(acc_A, 4),
            "acc_B": round(acc_B, 4),
            "acc_C": round(acc_C, 4),
            "acc_D": round(acc_D, 4),
            "new_write_fraction": round(new_write_fraction, 4),
            "gain_B_over_A": round(acc_B - acc_A, 4),
            "gain_C_over_A": round(acc_C - acc_A, 4),
            "gain_D_over_A": round(acc_D - acc_A, 4),
        }

        if new_write_fraction > 0.80:
            outcome = OUTCOME_SUPPORTED
            notes = f"New-write fraction={new_write_fraction:.3f}>0.80 — benefit is primarily from new entries."
        elif (acc_C - acc_A) > (acc_B - acc_A) + 0.03:
            outcome = OUTCOME_REFUTED
            notes = f"Overwrite gain ({acc_C - acc_A:.3f}) exceeds new-write gain ({acc_B - acc_A:.3f}) by >0.03."
        elif 0.40 <= new_write_fraction <= 0.80:
            outcome = OUTCOME_INCONCLUSIVE
            notes = f"New-write fraction={new_write_fraction:.3f} between 0.4 and 0.8 — inconclusive."
        else:
            outcome = OUTCOME_INCONCLUSIVE
            notes = f"New-write fraction={new_write_fraction:.3f} — inconclusive."

        return self.result(outcome=outcome, metrics=metrics, notes=notes, config=config)


if __name__ == "__main__":
    Exp102RetroactiveDecomposition().execute()
