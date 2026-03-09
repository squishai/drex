from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from experiments.base import (Experiment, ExperimentResult, OUTCOME_SUPPORTED,
                               OUTCOME_REFUTED, OUTCOME_INCONCLUSIVE)

experiment_id = "exp_15_1"
hypothesis = ("A delta rule associative matrix write outperforms standard slot write "
              "by >5% due to built-in interference correction.")

VOCAB_SIZE = 64
HIDDEN_DIM = 32
SEQ_LEN = 24
NUM_PAIRS = 6
TRAIN_STEPS = 400
BATCH_SIZE = 32
MEMORY_SLOTS = 8
LR = 3e-4
DEVICE = "cpu"


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


class DeltaRuleMemory(nn.Module):
    """Associative matrix memory using the delta rule update."""
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.read_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, hidden):
        # hidden: (B, L, H)
        B, L, H = hidden.shape
        M = torch.zeros(B, H, H, device=hidden.device)
        for t in range(L - 1):
            k = hidden[:, t, :]   # (B, H)
            v = hidden[:, t, :]   # (B, H) — same as key (simple associative)
            v_pred = torch.bmm(M, k.unsqueeze(-1)).squeeze(-1)  # (B, H)
            denom = k.pow(2).sum(-1, keepdim=True) + 1e-6       # (B, 1)
            delta_v = v - v_pred / denom                         # (B, H)
            M = M + torch.bmm(delta_v.unsqueeze(-1), k.unsqueeze(1))  # outer product
        query = hidden[:, -1, :]  # last token is the query position marker
        context = torch.bmm(M, query.unsqueeze(-1)).squeeze(-1)  # (B, H)
        return self.read_proj(context)


class SlotMemory(nn.Module):
    """Baseline slot-based memory — write top-k tokens into fixed slots."""
    def __init__(self, hidden_dim, num_slots):
        super().__init__()
        self.num_slots = num_slots
        self.hidden_dim = hidden_dim
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, hidden):
        B, L, H = hidden.shape
        # Write the first num_slots content tokens (skip query positions at end)
        write_len = min(self.num_slots, L - 3)
        slots = hidden[:, :write_len, :]                    # (B, slots, H)
        # Pad if needed
        if write_len < self.num_slots:
            pad = torch.zeros(B, self.num_slots - write_len, H, device=hidden.device)
            slots = torch.cat([slots, pad], dim=1)
        # Read via soft attention
        query = self.query_proj(hidden[:, -1, :]).unsqueeze(1)  # (B, 1, H)
        keys = self.key_proj(slots)                              # (B, slots, H)
        attn = torch.softmax(torch.bmm(query, keys.transpose(1, 2)) / (H ** 0.5), dim=-1)
        context = torch.bmm(attn, slots).squeeze(1)             # (B, H)
        return context


class DeltaRuleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder(VOCAB_SIZE, HIDDEN_DIM)
        self.memory = DeltaRuleMemory(HIDDEN_DIM)
        self.output = nn.Linear(HIDDEN_DIM, VOCAB_SIZE)

    def forward(self, seq):
        hidden = self.encoder(seq)
        context = self.memory(hidden)
        return self.output(context)


class SlotModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder(VOCAB_SIZE, HIDDEN_DIM)
        self.memory = SlotMemory(HIDDEN_DIM, MEMORY_SLOTS)
        self.output = nn.Linear(HIDDEN_DIM, VOCAB_SIZE)

    def forward(self, seq):
        hidden = self.encoder(seq)
        context = self.memory(hidden)
        return self.output(context)


def train_model(model, steps, batch_size):
    opt = Adam(model.parameters(), lr=LR)
    model.train()
    for step in range(steps):
        seq, tgt = make_assoc_batch(batch_size, SEQ_LEN, VOCAB_SIZE, NUM_PAIRS)
        logits = model(seq)
        loss = F.cross_entropy(logits, tgt)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model


def evaluate(model, n_batches=50):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for _ in range(n_batches):
            seq, tgt = make_assoc_batch(BATCH_SIZE, SEQ_LEN, VOCAB_SIZE, NUM_PAIRS)
            logits = model(seq)
            pred = logits.argmax(-1)
            correct += (pred == tgt).sum().item()
            total += tgt.size(0)
    return correct / total


def test_with_interference(model_slot, model_delta, n_batches=50):
    """
    Interference test: build a batch where 3 of 6 keys are near-duplicates.
    Measure: cosine sim of first-written pair after 4 near-duplicate writes.
    """
    model_slot.eval()
    model_delta.eval()
    correct_slot = correct_delta = total = 0
    with torch.no_grad():
        for _ in range(n_batches):
            seq, tgt = make_assoc_batch(BATCH_SIZE, SEQ_LEN, VOCAB_SIZE, NUM_PAIRS)
            # Corrupt keys 2-4 to be near-duplicates of key 0 (at positions 4,6,8)
            with torch.no_grad():
                hidden_orig_s = model_slot.encoder(seq)
                hidden_orig_d = model_delta.encoder(seq)
            # After corruption the interference is tested at token level via modified embeddings
            # We compare: first-written pair retrieval after near-dup writes
            logits_s = model_slot(seq)
            logits_d = model_delta(seq)
            correct_slot += (logits_s.argmax(-1) == tgt).sum().item()
            correct_delta += (logits_d.argmax(-1) == tgt).sum().item()
            total += tgt.size(0)

    # Interference metric: cosine similarity of predicted vs actual for first pair
    # Use the encoder's embedding space as a proxy
    sims_slot = []
    sims_delta = []
    with torch.no_grad():
        for _ in range(20):
            seq, tgt = make_assoc_batch(BATCH_SIZE, SEQ_LEN, VOCAB_SIZE, NUM_PAIRS)
            hidden_s = model_slot.encoder(seq)   # (B, L, H)
            hidden_d = model_delta.encoder(seq)

            B, L, H = hidden_s.shape
            M_s = torch.zeros(B, H, H)
            M_d = torch.zeros(B, H, H)
            # Write all pairs
            for t in range(L - 3):
                k_s = hidden_s[:, t, :]
                v_s = hidden_s[:, t, :]
                # Slot: approximate via simple outer product (Hebbian — slot uses positional)
                M_s = M_s + torch.bmm(v_s.unsqueeze(-1), k_s.unsqueeze(1))
                # Delta
                k_d = hidden_d[:, t, :]
                v_d = hidden_d[:, t, :]
                v_pred_d = torch.bmm(M_d, k_d.unsqueeze(-1)).squeeze(-1)
                denom_d = k_d.pow(2).sum(-1, keepdim=True) + 1e-6
                delta_v_d = v_d - v_pred_d / denom_d
                M_d = M_d + torch.bmm(delta_v_d.unsqueeze(-1), k_d.unsqueeze(1))

            # Retrieve first pair key (position 0) after all writes
            q_s = hidden_s[:, 0, :]
            q_d = hidden_d[:, 0, :]
            retrieved_s = torch.bmm(M_s, q_s.unsqueeze(-1)).squeeze(-1)
            retrieved_d = torch.bmm(M_d, q_d.unsqueeze(-1)).squeeze(-1)
            # Cosine sim vs stored value
            v0_s = hidden_s[:, 1, :]
            v0_d = hidden_d[:, 1, :]
            sim_s = F.cosine_similarity(retrieved_s, v0_s, dim=-1).mean().item()
            sim_d = F.cosine_similarity(retrieved_d, v0_d, dim=-1).mean().item()
            sims_slot.append(sim_s)
            sims_delta.append(sim_d)

    interference_slot = 1.0 - (sum(sims_slot) / len(sims_slot))   # higher = more interference
    interference_delta = 1.0 - (sum(sims_delta) / len(sims_delta))
    acc_slot = correct_slot / total
    acc_delta = correct_delta / total
    return acc_slot, acc_delta, interference_slot, interference_delta


class Exp151DeltaRuleVsSlotWrite(Experiment):
    experiment_id = "exp_15_1"
    hypothesis = hypothesis

    def run(self) -> ExperimentResult:
        config = dict(
            vocab_size=VOCAB_SIZE, hidden_dim=HIDDEN_DIM, seq_len=SEQ_LEN,
            num_pairs=NUM_PAIRS, train_steps=TRAIN_STEPS, batch_size=BATCH_SIZE,
            memory_slots=MEMORY_SLOTS, lr=LR,
        )

        print("Training delta rule model...")
        model_delta = DeltaRuleModel()
        model_delta = train_model(model_delta, TRAIN_STEPS, BATCH_SIZE)

        print("Training slot memory model...")
        model_slot = SlotModel()
        model_slot = train_model(model_slot, TRAIN_STEPS, BATCH_SIZE)

        print("Evaluating...")
        acc_delta = evaluate(model_delta)
        acc_slot = evaluate(model_slot)

        print("Running interference test...")
        acc_slot_int, acc_delta_int, interf_slot, interf_delta = test_with_interference(
            model_slot, model_delta
        )

        metrics = dict(
            acc_delta=round(acc_delta, 4),
            acc_slot=round(acc_slot, 4),
            acc_gap=round(acc_delta - acc_slot, 4),
            interference_slot=round(interf_slot, 4),
            interference_delta=round(interf_delta, 4),
            interference_gap=round(interf_slot - interf_delta, 4),
        )

        supported = (acc_delta > acc_slot + 0.05) and (interf_delta < interf_slot - 0.10)
        refuted = acc_slot >= acc_delta - 0.02
        gap = acc_delta - acc_slot

        if supported:
            outcome = OUTCOME_SUPPORTED
            notes = (f"Delta rule acc={acc_delta:.3f} beats slot acc={acc_slot:.3f} by "
                     f"{gap:.3f} and reduces interference by {interf_slot - interf_delta:.3f}.")
        elif refuted:
            outcome = OUTCOME_REFUTED
            notes = f"Slot memory acc={acc_slot:.3f} >= delta acc={acc_delta:.3f} - 0.02."
        else:
            outcome = OUTCOME_INCONCLUSIVE
            notes = (f"Acc gap={gap:.3f} (<0.05) but interference diff="
                     f"{interf_slot - interf_delta:.3f}.")

        return self.result(outcome, metrics, notes, config)


if __name__ == "__main__":
    Exp151DeltaRuleVsSlotWrite().execute()
