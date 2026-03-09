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

experiment_id = "exp_15_4"
hypothesis = ("Delta rule outperforms Larimar outer-product write on overwrite tasks "
              "(same key, updated value).")

VOCAB_SIZE = 64
HIDDEN_DIM = 32
SEQ_LEN = 24
NUM_PAIRS = 4
UPDATE_FRACTION = 0.5
STEPS = 400
BATCH = 32
LR = 3e-4
DEVICE = "cpu"


def make_assoc_batch_update(batch_size, seq_len, vocab_size, num_pairs, update_fraction):
    """
    For update_fraction of sequences: key[0] appears at pos 0 and pos 8 with different values.
    Query at end asks for key[0] -> should answer val_new (second occurrence).
    Returns seq, target, is_update_mask.
    """
    seq = torch.zeros(batch_size, seq_len, dtype=torch.long)
    target = torch.zeros(batch_size, dtype=torch.long)
    is_update = torch.zeros(batch_size, dtype=torch.bool)

    for b in range(batch_size):
        keys = torch.randint(4, max(8, vocab_size // 3), (num_pairs * 3,)).unique()[:num_pairs]
        while len(keys) < num_pairs:
            keys = torch.cat([keys, torch.randint(4, vocab_size // 3, (1,))])[:num_pairs]
        vals = torch.randint(vocab_size // 2, vocab_size, (num_pairs,))

        do_update = torch.rand(1).item() < update_fraction

        if do_update and seq_len >= 12:
            # key[0] at pos 0 -> val_old, key[0] at pos 8 -> val_new
            val_old = vals[0].clone()
            val_new = torch.randint(vocab_size // 2, vocab_size, (1,))[0]
            while val_new == val_old:
                val_new = torch.randint(vocab_size // 2, vocab_size, (1,))[0]

            seq[b, 0] = keys[0]; seq[b, 1] = val_old
            # Fill remaining pairs at positions 2..
            pos = 2
            for i in range(1, num_pairs):
                if pos + 1 < 7:
                    seq[b, pos] = keys[i]; seq[b, pos + 1] = vals[i]; pos += 2
            for p in range(pos, 8):
                seq[b, p] = 3
            # Second occurrence of key[0] at pos 8
            seq[b, 8] = keys[0]; seq[b, 9] = val_new
            for p in range(10, seq_len - 3):
                seq[b, p] = 3
            seq[b, seq_len - 3] = 2
            seq[b, seq_len - 2] = keys[0]
            seq[b, seq_len - 1] = 0
            target[b] = val_new
            is_update[b] = True
        else:
            # Normal associative recall — no duplicates
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
            is_update[b] = False

    return seq, target, is_update


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


def process_with_matrix(hidden, method="delta"):
    """
    Process sequence using given write method.
    method: 'delta' | 'larimar' (outer product, no correction)
    Returns: context (B, H)
    """
    B, L, H = hidden.shape
    M = torch.zeros(B, H, H, device=hidden.device)
    for t in range(L - 1):
        k = hidden[:, t, :]
        v = hidden[:, t, :]
        if method == "delta":
            v_pred = torch.bmm(M, k.unsqueeze(-1)).squeeze(-1)
            denom = k.pow(2).sum(-1, keepdim=True) + 1e-6
            delta_v = v - v_pred / denom
            M = M + torch.bmm(delta_v.unsqueeze(-1), k.unsqueeze(1))
        elif method == "larimar":
            M = M + torch.bmm(v.unsqueeze(-1), k.unsqueeze(1))
    query = hidden[:, -1, :]
    context = torch.bmm(M, query.unsqueeze(-1)).squeeze(-1)
    return context


class MemoryModel(nn.Module):
    def __init__(self, method: str):
        super().__init__()
        self.method = method
        self.encoder = Encoder(VOCAB_SIZE, HIDDEN_DIM)
        self.read_proj = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.output = nn.Linear(HIDDEN_DIM, VOCAB_SIZE)

    def forward(self, seq):
        hidden = self.encoder(seq)
        context = process_with_matrix(hidden, self.method)
        return self.output(self.read_proj(context))


def train_model(model, steps, batch_size):
    opt = Adam(model.parameters(), lr=LR)
    model.train()
    for step in range(steps):
        seq, tgt, _ = make_assoc_batch_update(
            batch_size, SEQ_LEN, VOCAB_SIZE, NUM_PAIRS, UPDATE_FRACTION
        )
        logits = model(seq)
        loss = F.cross_entropy(logits, tgt)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model


def evaluate_split(model, n_batches=60):
    """Returns (acc_update, acc_normal, acc_overall)."""
    model.eval()
    correct_up = total_up = 0
    correct_no = total_no = 0
    with torch.no_grad():
        for _ in range(n_batches):
            seq, tgt, is_up = make_assoc_batch_update(
                BATCH, SEQ_LEN, VOCAB_SIZE, NUM_PAIRS, UPDATE_FRACTION
            )
            logits = model(seq)
            pred = logits.argmax(-1)
            correct = pred == tgt
            if is_up.any():
                correct_up += correct[is_up].sum().item()
                total_up += is_up.sum().item()
            if (~is_up).any():
                correct_no += correct[~is_up].sum().item()
                total_no += (~is_up).sum().item()

    acc_up = correct_up / max(total_up, 1)
    acc_no = correct_no / max(total_no, 1)
    acc_all = (correct_up + correct_no) / max(total_up + total_no, 1)
    return acc_up, acc_no, acc_all


class Exp154DeltaVsLarimar(Experiment):
    experiment_id = "exp_15_4"
    hypothesis = hypothesis

    def run(self) -> ExperimentResult:
        config = dict(
            vocab_size=VOCAB_SIZE, hidden_dim=HIDDEN_DIM, seq_len=SEQ_LEN,
            num_pairs=NUM_PAIRS, update_fraction=UPDATE_FRACTION,
            steps=STEPS, batch=BATCH, lr=LR,
        )

        print("Training delta rule model...")
        model_delta = MemoryModel("delta")
        model_delta = train_model(model_delta, STEPS, BATCH)

        print("Training Larimar (outer product) model...")
        model_larimar = MemoryModel("larimar")
        model_larimar = train_model(model_larimar, STEPS, BATCH)

        print("Evaluating split by update vs non-update queries...")
        acc_up_A, acc_no_A, acc_all_A = evaluate_split(model_delta)
        acc_up_B, acc_no_B, acc_all_B = evaluate_split(model_larimar)

        print(f"  Delta:   update={acc_up_A:.3f}, normal={acc_no_A:.3f}, all={acc_all_A:.3f}")
        print(f"  Larimar: update={acc_up_B:.3f}, normal={acc_no_B:.3f}, all={acc_all_B:.3f}")

        gap_update = acc_up_A - acc_up_B

        metrics = dict(
            delta_acc_update=round(acc_up_A, 4),
            delta_acc_normal=round(acc_no_A, 4),
            delta_acc_overall=round(acc_all_A, 4),
            larimar_acc_update=round(acc_up_B, 4),
            larimar_acc_normal=round(acc_no_B, 4),
            larimar_acc_overall=round(acc_all_B, 4),
            update_gap_delta_minus_larimar=round(gap_update, 4),
        )

        if gap_update > 0.10:
            outcome = OUTCOME_SUPPORTED
            notes = (f"Delta outperforms Larimar on update queries by {gap_update:.3f} > 0.10. "
                     "Correction term enables proper key overwrite.")
        elif acc_up_B >= acc_up_A - 0.02:
            outcome = OUTCOME_REFUTED
            notes = (f"Larimar acc_update={acc_up_B:.3f} >= delta "
                     f"acc_update={acc_up_A:.3f} - 0.02.")
        else:
            outcome = OUTCOME_INCONCLUSIVE
            notes = f"Both degrade similarly on update queries; gap={gap_update:.3f}."

        return self.result(outcome, metrics, notes, config)


if __name__ == "__main__":
    Exp154DeltaVsLarimar().execute()
