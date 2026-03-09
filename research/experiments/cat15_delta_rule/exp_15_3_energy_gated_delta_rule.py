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

experiment_id = "exp_15_3"
hypothesis = ("Energy-gated delta rule (write only when delta_E < 0) achieves "
              ">90% accuracy of continuous write at <70% write rate.")

VOCAB_SIZE = 64
HIDDEN_DIM = 32
SEQ_LEN = 24
NUM_PAIRS = 5
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


def hopfield_energy(M):
    """Simplified energy: -0.5 * ||M||_F^2 (negative Frobenius norm squared)."""
    return -0.5 * (M * M).sum(dim=(-2, -1))  # (B,)


def delta_update(M, k, v):
    """Compute delta rule update delta M. Returns dM (B, H, H)."""
    v_pred = torch.bmm(M, k.unsqueeze(-1)).squeeze(-1)      # (B, H)
    denom = k.pow(2).sum(-1, keepdim=True) + 1e-6           # (B, 1)
    delta_v = v - v_pred / denom                            # (B, H)
    dM = torch.bmm(delta_v.unsqueeze(-1), k.unsqueeze(1))   # (B, H, H)
    return dM


class ContinuousDeltaModel(nn.Module):
    """Condition A: write every token."""
    def __init__(self):
        super().__init__()
        self.encoder = Encoder(VOCAB_SIZE, HIDDEN_DIM)
        self.read_proj = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.output = nn.Linear(HIDDEN_DIM, VOCAB_SIZE)

    def forward(self, seq):
        hidden = self.encoder(seq)
        B, L, H = hidden.shape
        M = torch.zeros(B, H, H, device=hidden.device)
        for t in range(L - 1):
            k = hidden[:, t, :]
            v = hidden[:, t, :]
            M = M + delta_update(M, k, v)
        query = hidden[:, -1, :]
        context = torch.bmm(M, query.unsqueeze(-1)).squeeze(-1)
        return self.output(self.read_proj(context)), None


class EnergyGatedDeltaModel(nn.Module):
    """Condition B: write only when delta_E < 0."""
    def __init__(self):
        super().__init__()
        self.encoder = Encoder(VOCAB_SIZE, HIDDEN_DIM)
        self.read_proj = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.output = nn.Linear(HIDDEN_DIM, VOCAB_SIZE)

    def forward(self, seq):
        hidden = self.encoder(seq)
        B, L, H = hidden.shape
        M = torch.zeros(B, H, H, device=hidden.device)
        total_writes = 0
        total_tokens = 0
        for t in range(L - 1):
            k = hidden[:, t, :]
            v = hidden[:, t, :]
            dM = delta_update(M, k, v)
            E_before = hopfield_energy(M)          # (B,)
            E_after = hopfield_energy(M + dM)      # (B,)
            # Write where energy decreases (delta_E < 0)
            gate = (E_after < E_before).float().unsqueeze(-1).unsqueeze(-1)  # (B,1,1)
            M = M + gate * dM
            total_writes += gate.sum().item()
            total_tokens += B
        write_rate = total_writes / max(total_tokens, 1)
        query = hidden[:, -1, :]
        context = torch.bmm(M, query.unsqueeze(-1)).squeeze(-1)
        return self.output(self.read_proj(context)), write_rate


class LearnedGateDeltaModel(nn.Module):
    """Condition C: learned gate (Linear(H,1) + sigmoid) controls writes."""
    def __init__(self):
        super().__init__()
        self.encoder = Encoder(VOCAB_SIZE, HIDDEN_DIM)
        self.gate_net = nn.Sequential(nn.Linear(HIDDEN_DIM, 1), nn.Sigmoid())
        self.read_proj = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.output = nn.Linear(HIDDEN_DIM, VOCAB_SIZE)

    def forward(self, seq):
        hidden = self.encoder(seq)
        B, L, H = hidden.shape
        M = torch.zeros(B, H, H, device=hidden.device)
        total_writes = 0.0
        total_tokens = 0
        for t in range(L - 1):
            k = hidden[:, t, :]
            v = hidden[:, t, :]
            g = self.gate_net(k)               # (B, 1)
            dM = delta_update(M.detach(), k, v)
            # Soft gate for gradient flow; hard threshold for write_rate metric
            with torch.no_grad():
                hard_gate = (g > 0.5).float()
                total_writes += hard_gate.sum().item()
                total_tokens += B
            gate_expand = g.unsqueeze(-1)      # (B, 1, 1)
            M = M + gate_expand * dM
        write_rate = total_writes / max(total_tokens, 1)
        query = hidden[:, -1, :]
        context = torch.bmm(M, query.unsqueeze(-1)).squeeze(-1)
        return self.output(self.read_proj(context)), write_rate


def train_model(model, steps, batch_size):
    opt = Adam(model.parameters(), lr=LR)
    model.train()
    for step in range(steps):
        seq, tgt = make_assoc_batch(batch_size, SEQ_LEN, VOCAB_SIZE, NUM_PAIRS)
        logits, _ = model(seq)
        loss = F.cross_entropy(logits, tgt)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model


def evaluate(model, n_batches=50):
    model.eval()
    correct = total = 0
    write_rates = []
    with torch.no_grad():
        for _ in range(n_batches):
            seq, tgt = make_assoc_batch(BATCH, SEQ_LEN, VOCAB_SIZE, NUM_PAIRS)
            logits, wr = model(seq)
            pred = logits.argmax(-1)
            correct += (pred == tgt).sum().item()
            total += tgt.size(0)
            if wr is not None:
                write_rates.append(wr)
    acc = correct / total
    avg_wr = sum(write_rates) / len(write_rates) if write_rates else 1.0
    return acc, avg_wr


class Exp153EnergyGatedDeltaRule(Experiment):
    experiment_id = "exp_15_3"
    hypothesis = hypothesis

    def run(self) -> ExperimentResult:
        config = dict(
            vocab_size=VOCAB_SIZE, hidden_dim=HIDDEN_DIM, seq_len=SEQ_LEN,
            num_pairs=NUM_PAIRS, steps=STEPS, batch=BATCH, lr=LR,
        )

        print("Training Condition A: continuous delta rule...")
        model_A = ContinuousDeltaModel()
        model_A = train_model(model_A, STEPS, BATCH)
        acc_A, wr_A = evaluate(model_A)
        print(f"  A: acc={acc_A:.3f}, write_rate={wr_A:.3f}")

        print("Training Condition B: energy-gated delta rule...")
        model_B = EnergyGatedDeltaModel()
        model_B = train_model(model_B, STEPS, BATCH)
        acc_B, wr_B = evaluate(model_B)
        print(f"  B: acc={acc_B:.3f}, write_rate={wr_B:.3f}")

        print("Training Condition C: learned gate delta rule...")
        model_C = LearnedGateDeltaModel()
        model_C = train_model(model_C, STEPS, BATCH)
        acc_C, wr_C = evaluate(model_C)
        print(f"  C: acc={acc_C:.3f}, write_rate={wr_C:.3f}")

        acc_ratio_B = acc_B / max(acc_A, 1e-6)
        acc_ratio_C = acc_C / max(acc_A, 1e-6)

        metrics = dict(
            acc_A_continuous=round(acc_A, 4),
            write_rate_A=round(wr_A, 4),
            acc_B_energy_gated=round(acc_B, 4),
            write_rate_B=round(wr_B, 4),
            acc_ratio_B=round(acc_ratio_B, 4),
            acc_C_learned_gate=round(acc_C, 4),
            write_rate_C=round(wr_C, 4),
            acc_ratio_C=round(acc_ratio_C, 4),
        )

        # B must achieve >90% of A's accuracy at <70% write rate
        supported = acc_ratio_B > 0.90 and wr_B < 0.70
        refuted = wr_B > 0.85 or (acc_A - acc_B) / max(acc_A, 1e-6) > 0.10

        if supported:
            outcome = OUTCOME_SUPPORTED
            notes = (f"Energy gate: acc_ratio={acc_ratio_B:.3f}>0.90, "
                     f"write_rate={wr_B:.3f}<0.70. Hypothesis confirmed.")
        elif refuted:
            outcome = OUTCOME_REFUTED
            notes = (f"Energy gate write_rate={wr_B:.3f}>0.85 or "
                     f"accuracy drop > 10% (ratio={acc_ratio_B:.3f}).")
        else:
            outcome = OUTCOME_INCONCLUSIVE
            notes = (f"Near-equal acc (ratio={acc_ratio_B:.3f}) but "
                     f"write rate={wr_B:.3f} does not drop below 0.70.")

        return self.result(outcome, metrics, notes, config)


if __name__ == "__main__":
    Exp153EnergyGatedDeltaRule().execute()
