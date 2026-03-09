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

experiment_id = "exp_16_3"
hypothesis = ("Parametric memory scales more gracefully with seq_len than slot memory "
              "(higher accuracy retention at 4x length).")

VOCAB_SIZE = 64
HIDDEN_DIM = 32
SEQ_LENS = [24, 48]
NUM_PAIRS_PER_LEN = [4, 8]
MEMORY_SLOTS = 8
STEPS = 300
BATCH = 8
INFERENCE_LR = 0.01
LR = 3e-4
DEVICE = "cpu"
MLP_INNER = 8


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


# ---- Slot Memory ----

class SlotMemoryModel(nn.Module):
    def __init__(self, num_slots=MEMORY_SLOTS):
        super().__init__()
        self.encoder = Encoder(VOCAB_SIZE, HIDDEN_DIM)
        self.query_proj = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.key_proj = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.output = nn.Linear(HIDDEN_DIM, VOCAB_SIZE)
        self.num_slots = num_slots

    def forward(self, seq):
        hidden = self.encoder(seq)
        B, L, H = hidden.shape
        write_len = min(self.num_slots, L - 3)
        slots = hidden[:, :write_len, :]
        if write_len < self.num_slots:
            pad = torch.zeros(B, self.num_slots - write_len, H, device=hidden.device)
            slots = torch.cat([slots, pad], dim=1)
        query = self.query_proj(hidden[:, -1, :]).unsqueeze(1)
        keys = self.key_proj(slots)
        attn = torch.softmax(
            torch.bmm(query, keys.transpose(1, 2)) / (H ** 0.5), dim=-1
        )
        context = torch.bmm(attn, slots).squeeze(1)
        return self.output(context)


# ---- Parametric Memory ----

class InnerMLP(nn.Module):
    def __init__(self, hidden_dim=HIDDEN_DIM, inner=MLP_INNER):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, inner)
        self.fc2 = nn.Linear(inner, hidden_dim)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


class ParametricMemoryModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder(VOCAB_SIZE, HIDDEN_DIM)
        self.base_mlp = InnerMLP(HIDDEN_DIM, MLP_INNER)
        self.output = nn.Linear(HIDDEN_DIM, VOCAB_SIZE)

    def forward(self, seq):
        hidden = self.encoder(seq)
        B, L, H = hidden.shape
        contexts = []
        for b in range(B):
            mlp = InnerMLP(HIDDEN_DIM, MLP_INNER)
            mlp.load_state_dict(self.base_mlp.state_dict())
            inner_opt = torch.optim.SGD(mlp.parameters(), lr=INFERENCE_LR)
            for t in range(0, L - 3, 2):
                key_h = hidden[b, t, :].detach()
                val_h = hidden[b, t + 1, :].detach()
                with torch.enable_grad():
                    inner_opt.zero_grad()
                    pred = mlp(key_h.unsqueeze(0))
                    loss = F.mse_loss(pred, val_h.unsqueeze(0))
                    loss.backward()
                inner_opt.step()
            query_h = hidden[b, -1, :].detach()
            with torch.no_grad():
                context = mlp(query_h.unsqueeze(0)).squeeze(0)
            contexts.append(context)
        context_batch = torch.stack(contexts, dim=0)
        return self.output(context_batch)


def train_model(model, steps, batch_size, seq_len, num_pairs):
    if isinstance(model, ParametricMemoryModel):
        opt = Adam(
            list(model.encoder.parameters()) +
            list(model.base_mlp.parameters()) +
            list(model.output.parameters()),
            lr=LR,
        )
    else:
        opt = Adam(model.parameters(), lr=LR)
    model.train()
    for step in range(steps):
        seq, tgt = make_assoc_batch(batch_size, seq_len, VOCAB_SIZE, num_pairs)
        logits = model(seq)
        loss = F.cross_entropy(logits, tgt)
        opt.zero_grad(); loss.backward(); opt.step()
        if (step + 1) % 500 == 0:
            print(f"    step {step+1}/{steps}, loss={loss.item():.4f}")
    return model


def evaluate(model, seq_len, num_pairs, n_batches=40):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for _ in range(n_batches):
            seq, tgt = make_assoc_batch(BATCH, seq_len, VOCAB_SIZE, num_pairs)
            logits = model(seq)
            pred = logits.argmax(-1)
            correct += (pred == tgt).sum().item()
            total += tgt.size(0)
    return correct / total


class Exp163ParametricSeqScaling(Experiment):
    experiment_id = "exp_16_3"
    hypothesis = hypothesis

    def run(self) -> ExperimentResult:
        config = dict(
            vocab_size=VOCAB_SIZE, hidden_dim=HIDDEN_DIM, seq_lens=SEQ_LENS,
            num_pairs_per_len=NUM_PAIRS_PER_LEN, memory_slots=MEMORY_SLOTS,
            steps=STEPS, batch=BATCH, inference_lr=INFERENCE_LR, lr=LR,
        )

        acc_slot = {}
        acc_param = {}

        for seq_len, num_pairs in zip(SEQ_LENS, NUM_PAIRS_PER_LEN):
            print(f"\n--- seq_len={seq_len}, num_pairs={num_pairs} ---")

            print(f"  Training slot memory (A)...")
            model_A = SlotMemoryModel(MEMORY_SLOTS)
            model_A = train_model(model_A, STEPS, BATCH, seq_len, num_pairs)
            a = evaluate(model_A, seq_len, num_pairs)
            acc_slot[seq_len] = round(a, 4)
            print(f"  Slot acc={a:.3f}")

            print(f"  Training parametric memory (B)...")
            model_B = ParametricMemoryModel()
            model_B = train_model(model_B, STEPS, BATCH, seq_len, num_pairs)
            b = evaluate(model_B, seq_len, num_pairs)
            acc_param[seq_len] = round(b, 4)
            print(f"  Parametric acc={b:.3f}")

        retention_A = acc_slot[48] / max(acc_slot[24], 0.001)
        retention_B = acc_param[48] / max(acc_param[24], 0.001)
        retention_diff = retention_B - retention_A

        metrics = dict(
            acc_slot=acc_slot,
            acc_parametric=acc_param,
            retention_slot=round(retention_A, 4),
            retention_parametric=round(retention_B, 4),
            retention_diff=round(retention_diff, 4),
        )

        if retention_diff > 0.15:
            outcome = OUTCOME_SUPPORTED
            notes = (f"Parametric retention={retention_B:.3f} vs slot retention={retention_A:.3f}; "
                     f"diff={retention_diff:.3f} > 0.15. Scales more gracefully.")
        elif retention_diff < 0.05:
            outcome = OUTCOME_REFUTED
            notes = f"Retention difference={retention_diff:.3f} < 0.05."
        else:
            outcome = OUTCOME_INCONCLUSIVE
            notes = f"Retention diff={retention_diff:.3f} between 0.05 and 0.15."

        return self.result(outcome, metrics, notes, config)


if __name__ == "__main__":
    Exp163ParametricSeqScaling().execute()
