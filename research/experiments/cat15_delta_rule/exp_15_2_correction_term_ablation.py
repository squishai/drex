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

experiment_id = "exp_15_2"
hypothesis = ("The correction term in the delta rule is essential — "
              "Hebbian M += v*k^T degrades by >10% on key-interference tasks.")

VOCAB_SIZE = 64
HIDDEN_DIM = 32
SEQ_LEN = 32
NUM_PAIRS = 6
INTERFERENCE_FRACTION = 0.5
STEPS = 400
BATCH = 32
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


# ---- Three memory methods ----

def delta_rule_read(hidden, method="delta"):
    """
    Process sequence with given update rule and return read context.
    hidden: (B, L, H)
    method: 'delta' | 'hebbian' | 'norm_hebbian'
    Returns: (B, H)
    """
    B, L, H = hidden.shape
    M = torch.zeros(B, H, H, device=hidden.device)
    for t in range(L - 1):
        k = hidden[:, t, :]  # (B, H)
        v = hidden[:, t, :]  # (B, H)
        if method == "delta":
            v_pred = torch.bmm(M, k.unsqueeze(-1)).squeeze(-1)
            denom = k.pow(2).sum(-1, keepdim=True) + 1e-6
            delta_v = v - v_pred / denom
            M = M + torch.bmm(delta_v.unsqueeze(-1), k.unsqueeze(1))
        elif method == "hebbian":
            M = M + torch.bmm(v.unsqueeze(-1), k.unsqueeze(1))
        elif method == "norm_hebbian":
            v_n = F.normalize(v, dim=-1)
            k_n = F.normalize(k, dim=-1)
            M = M + torch.bmm(v_n.unsqueeze(-1), k_n.unsqueeze(1))
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

    def forward(self, seq, corrupt_mask=None, enc_out=None):
        if enc_out is not None:
            hidden = enc_out
        else:
            hidden = self.encoder(seq)
        context = delta_rule_read(hidden, self.method)
        return self.output(self.read_proj(context))


def apply_interference(enc_out, interference_fraction=0.5):
    """
    For interference_fraction of batch items, corrupt pairs 0 and 1:
    mix their key embeddings so they interfere.
    enc_out: (B, L, H) — already encoded
    Returns corrupted enc_out (detached copy).
    """
    B, L, H = enc_out.shape
    out = enc_out.clone()
    n_corrupt = int(B * interference_fraction)
    for b in range(n_corrupt):
        # key i=0 at pos 0, key i=1 at pos 2 — mix them
        pos_i, pos_j = 0, 2
        orig_i = out[b, pos_i, :].clone()
        orig_j = out[b, pos_j, :].clone()
        out[b, pos_i, :] = 0.7 * orig_i + 0.3 * orig_j
        out[b, pos_j, :] = 0.7 * orig_j + 0.3 * orig_i
    return out


def train_model(model, steps, batch_size):
    opt = Adam(model.parameters(), lr=LR)
    model.train()
    for step in range(steps):
        seq, tgt = make_assoc_batch(batch_size, SEQ_LEN, VOCAB_SIZE, NUM_PAIRS)
        with torch.no_grad():
            enc = model.encoder(seq)
        enc = enc.detach().requires_grad_(False)
        # Re-encode through graph for gradient
        enc_grad = model.encoder(seq)
        # Apply interference at embedding level
        enc_int = apply_interference(enc_grad, INTERFERENCE_FRACTION)
        context = delta_rule_read(enc_int, model.method)
        logits = model.output(model.read_proj(context))
        loss = F.cross_entropy(logits, tgt)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model


def evaluate_split(model, n_batches=50):
    """Returns (acc_clean, acc_interfered)."""
    model.eval()
    correct_clean = total_clean = 0
    correct_int = total_int = 0
    with torch.no_grad():
        for _ in range(n_batches):
            seq, tgt = make_assoc_batch(BATCH, SEQ_LEN, VOCAB_SIZE, NUM_PAIRS)
            enc = model.encoder(seq)
            n_int = int(BATCH * INTERFERENCE_FRACTION)

            # Clean batch (no corruption)
            context_clean = delta_rule_read(enc, model.method)
            logits_clean = model.output(model.read_proj(context_clean))
            correct_clean += (logits_clean.argmax(-1) == tgt).sum().item()
            total_clean += tgt.size(0)

            # Interfered batch
            enc_int = apply_interference(enc, INTERFERENCE_FRACTION)
            context_int = delta_rule_read(enc_int, model.method)
            logits_int = model.output(model.read_proj(context_int))
            # Only count the corrupted items
            correct_int += (logits_int.argmax(-1)[:n_int] == tgt[:n_int]).sum().item()
            total_int += n_int

    acc_clean = correct_clean / max(total_clean, 1)
    acc_int = correct_int / max(total_int, 1)
    return acc_clean, acc_int


class Exp152CorrectionTermAblation(Experiment):
    experiment_id = "exp_15_2"
    hypothesis = hypothesis

    def run(self) -> ExperimentResult:
        config = dict(
            vocab_size=VOCAB_SIZE, hidden_dim=HIDDEN_DIM, seq_len=SEQ_LEN,
            num_pairs=NUM_PAIRS, interference_fraction=INTERFERENCE_FRACTION,
            steps=STEPS, batch=BATCH, lr=LR,
        )

        results = {}
        for method in ["delta", "hebbian", "norm_hebbian"]:
            print(f"Training method={method}...")
            model = MemoryModel(method)
            model = train_model(model, STEPS, BATCH)
            acc_clean, acc_int = evaluate_split(model)
            results[method] = dict(acc_clean=round(acc_clean, 4),
                                   acc_int=round(acc_int, 4))
            print(f"  {method}: clean={acc_clean:.3f}, interfered={acc_int:.3f}")

        acc_A_int = results["delta"]["acc_int"]
        acc_B_int = results["hebbian"]["acc_int"]
        gap = acc_A_int - acc_B_int

        metrics = dict(
            delta_acc_clean=results["delta"]["acc_clean"],
            delta_acc_interfered=results["delta"]["acc_int"],
            hebbian_acc_clean=results["hebbian"]["acc_clean"],
            hebbian_acc_interfered=results["hebbian"]["acc_int"],
            norm_hebbian_acc_clean=results["norm_hebbian"]["acc_clean"],
            norm_hebbian_acc_interfered=results["norm_hebbian"]["acc_int"],
            delta_vs_hebbian_gap_on_interference=round(gap, 4),
        )

        if gap > 0.10:
            outcome = OUTCOME_SUPPORTED
            notes = (f"Delta rule beats Hebbian by {gap:.3f} on interference subset. "
                     "Correction term is essential.")
        elif acc_B_int >= acc_A_int - 0.03:
            outcome = OUTCOME_REFUTED
            notes = (f"Hebbian acc_int={acc_B_int:.3f} within 0.03 of delta "
                     f"acc_int={acc_A_int:.3f}.")
        else:
            outcome = OUTCOME_INCONCLUSIVE
            notes = f"Gap={gap:.3f} between 0.03 and 0.10 on interference task."

        return self.result(outcome, metrics, notes, config)


if __name__ == "__main__":
    Exp152CorrectionTermAblation().execute()
