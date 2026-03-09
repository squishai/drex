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

experiment_id = "exp_16_1"
hypothesis = ("Parametric MLP memory (1 gradient step per token) outperforms "
              "fixed-slot memory at matched parameter count.")

VOCAB_SIZE = 64
HIDDEN_DIM = 32
SEQ_LEN = 24
NUM_PAIRS = 5
MEMORY_SLOTS = 8
TRAIN_STEPS = 300
BATCH_SIZE = 8
INFERENCE_LR = 0.01
LR = 3e-4
DEVICE = "cpu"
MLP_INNER = 8   # Linear(32,8) -> ReLU -> Linear(8,32): 32*8+8+8*32+32 = 296 params


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


# ---- Slot Memory (Model A) ----

class SlotMemoryModel(nn.Module):
    """8-slot associative memory with soft-attention read."""
    def __init__(self):
        super().__init__()
        self.encoder = Encoder(VOCAB_SIZE, HIDDEN_DIM)
        self.query_proj = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.key_proj = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.output = nn.Linear(HIDDEN_DIM, VOCAB_SIZE)
        # param count: 8 slots * 32 dim = 256 values stored
        self.num_slots = MEMORY_SLOTS

    def forward(self, seq):
        hidden = self.encoder(seq)   # (B, L, H)
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


# ---- Parametric MLP Memory (Model B) ----

class InnerMLP(nn.Module):
    """Small MLP: Linear(H, inner) -> ReLU -> Linear(inner, H). ~296 params."""
    def __init__(self, hidden_dim=HIDDEN_DIM, inner=MLP_INNER):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, inner)
        self.fc2 = nn.Linear(inner, hidden_dim)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))

    def param_count(self):
        return sum(p.numel() for p in self.parameters())


class ParametricMemoryModel(nn.Module):
    """
    Per-sequence MLP memory updated online during sequence processing.
    Uses a single shared MLP per forward pass; weights are reset between sequences.
    Inner updates use SGD with INFERENCE_LR.
    """
    def __init__(self):
        super().__init__()
        self.encoder = Encoder(VOCAB_SIZE, HIDDEN_DIM)
        self.base_mlp = InnerMLP(HIDDEN_DIM, MLP_INNER)
        self.output = nn.Linear(HIDDEN_DIM, VOCAB_SIZE)

    def forward(self, seq):
        hidden = self.encoder(seq)   # (B, L, H)
        B, L, H = hidden.shape

        # Process each batch item with its own MLP copy
        contexts = []
        for b in range(B):
            # Clone base MLP weights for this sequence
            mlp = InnerMLP(HIDDEN_DIM, MLP_INNER)
            mlp.load_state_dict(self.base_mlp.state_dict())
            inner_opt = torch.optim.SGD(mlp.parameters(), lr=INFERENCE_LR)

            # Online updates: at key positions (even positions before query zone)
            for t in range(0, L - 3, 2):
                key_h = hidden[b, t, :].detach()
                val_h = hidden[b, t + 1, :].detach()
                with torch.enable_grad():
                    inner_opt.zero_grad()
                    pred = mlp(key_h.unsqueeze(0))  # (1, H)
                    loss = F.mse_loss(pred, val_h.unsqueeze(0))
                    loss.backward()
                inner_opt.step()

            # Read at query position
            query_h = hidden[b, -1, :].detach()
            with torch.no_grad():
                context = mlp(query_h.unsqueeze(0)).squeeze(0)  # (H,)
            contexts.append(context)

        context_batch = torch.stack(contexts, dim=0)  # (B, H)
        return self.output(context_batch)


def train_slot(steps, batch_size):
    model = SlotMemoryModel()
    opt = Adam(model.parameters(), lr=LR)
    model.train()
    for _ in range(steps):
        seq, tgt = make_assoc_batch(batch_size, SEQ_LEN, VOCAB_SIZE, NUM_PAIRS)
        logits = model(seq)
        loss = F.cross_entropy(logits, tgt)
        opt.zero_grad(); loss.backward(); opt.step()
    return model


def train_parametric(steps, batch_size):
    model = ParametricMemoryModel()
    # Only train encoder and output; inner MLP is updated online during forward
    opt = Adam(
        list(model.encoder.parameters()) +
        list(model.base_mlp.parameters()) +
        list(model.output.parameters()),
        lr=LR,
    )
    model.train()
    for step in range(steps):
        seq, tgt = make_assoc_batch(batch_size, SEQ_LEN, VOCAB_SIZE, NUM_PAIRS)
        logits = model(seq)
        loss = F.cross_entropy(logits, tgt)
        opt.zero_grad(); loss.backward(); opt.step()
        if (step + 1) % 500 == 0:
            print(f"  step {step+1}/{steps}, loss={loss.item():.4f}")
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


class Exp161ParametricVsSlotMemory(Experiment):
    experiment_id = "exp_16_1"
    hypothesis = hypothesis

    def run(self) -> ExperimentResult:
        config = dict(
            vocab_size=VOCAB_SIZE, hidden_dim=HIDDEN_DIM, seq_len=SEQ_LEN,
            num_pairs=NUM_PAIRS, memory_slots=MEMORY_SLOTS,
            train_steps=TRAIN_STEPS, batch_size=BATCH_SIZE,
            inference_lr=INFERENCE_LR, lr=LR, mlp_inner=MLP_INNER,
        )

        # Parameter counts
        slot_model_temp = SlotMemoryModel()
        param_count_A = MEMORY_SLOTS * HIDDEN_DIM  # conceptual slot storage
        param_mlp = InnerMLP().param_count()

        print("Training slot memory (A)...")
        model_A = train_slot(TRAIN_STEPS, BATCH_SIZE)
        acc_A = evaluate(model_A)
        print(f"  Slot memory acc={acc_A:.3f}")

        print("Training parametric memory (B)...")
        model_B = train_parametric(TRAIN_STEPS, BATCH_SIZE)
        acc_B = evaluate(model_B)
        print(f"  Parametric memory acc={acc_B:.3f}")

        gap = acc_B - acc_A
        metrics = dict(
            acc_slot=round(acc_A, 4),
            acc_parametric=round(acc_B, 4),
            acc_gap=round(gap, 4),
            param_count_slot_storage=param_count_A,
            param_count_mlp=param_mlp,
        )

        if gap > 0.05:
            outcome = OUTCOME_SUPPORTED
            notes = (f"Parametric MLP acc={acc_B:.3f} beats slot acc={acc_A:.3f} "
                     f"by {gap:.3f} > 0.05.")
        elif acc_A >= acc_B - 0.02:
            outcome = OUTCOME_REFUTED
            notes = f"Slot acc={acc_A:.3f} >= parametric acc={acc_B:.3f} - 0.02."
        else:
            outcome = OUTCOME_INCONCLUSIVE
            notes = f"Gap={gap:.3f} is between -0.02 and 0.05."

        return self.result(outcome, metrics, notes, config)


if __name__ == "__main__":
    Exp161ParametricVsSlotMemory().execute()
