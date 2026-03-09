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

experiment_id = "exp_16_2"
hypothesis = ("Skipping MLP memory updates for low-surprise tokens achieves "
              "same accuracy with >40% fewer update steps.")

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
MLP_INNER = 8

S_VALUES = [0.0, 0.3, 0.6, 1.0, 1.5]


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


class InnerMLP(nn.Module):
    def __init__(self, hidden_dim=HIDDEN_DIM, inner=MLP_INNER):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, inner)
        self.fc2 = nn.Linear(inner, hidden_dim)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


def compute_surprise(hidden_t, seen_hiddens):
    """
    Surprise proxy: L2 distance of hidden[t] from mean of previously seen hiddens.
    If no hiddens seen yet, returns large surprise (always write).
    """
    if len(seen_hiddens) == 0:
        return float("inf")
    mean_h = torch.stack(seen_hiddens, dim=0).mean(dim=0)
    return (hidden_t - mean_h).pow(2).sum().sqrt().item()


class SurpriseGatedModel(nn.Module):
    def __init__(self, surprise_threshold: float):
        super().__init__()
        self.threshold = surprise_threshold
        self.encoder = Encoder(VOCAB_SIZE, HIDDEN_DIM)
        self.base_mlp = InnerMLP(HIDDEN_DIM, MLP_INNER)
        self.output = nn.Linear(HIDDEN_DIM, VOCAB_SIZE)

    def forward(self, seq, track_write_rate=False):
        hidden = self.encoder(seq)   # (B, L, H)
        B, L, H = hidden.shape
        contexts = []
        total_writes = 0
        total_possible = 0

        for b in range(B):
            mlp = InnerMLP(HIDDEN_DIM, MLP_INNER)
            mlp.load_state_dict(self.base_mlp.state_dict())
            inner_opt = torch.optim.SGD(mlp.parameters(), lr=INFERENCE_LR)
            seen = []

            for t in range(0, L - 3, 2):
                key_h = hidden[b, t, :].detach()
                val_h = hidden[b, t + 1, :].detach()
                total_possible += 1

                surprise = compute_surprise(key_h, seen)
                seen.append(key_h)

                if surprise > self.threshold:
                    with torch.enable_grad():
                        inner_opt.zero_grad()
                        pred = mlp(key_h.unsqueeze(0))
                        loss = F.mse_loss(pred, val_h.unsqueeze(0))
                        loss.backward()
                    inner_opt.step()
                    total_writes += 1

            query_h = hidden[b, -1, :].detach()
            with torch.no_grad():
                context = mlp(query_h.unsqueeze(0)).squeeze(0)
            contexts.append(context)

        context_batch = torch.stack(contexts, dim=0)
        logits = self.output(context_batch)
        write_rate = total_writes / max(total_possible, 1)
        return logits, write_rate


def train_model(model, steps, batch_size):
    opt = Adam(
        list(model.encoder.parameters()) +
        list(model.base_mlp.parameters()) +
        list(model.output.parameters()),
        lr=LR,
    )
    model.train()
    for _ in range(steps):
        seq, tgt = make_assoc_batch(batch_size, SEQ_LEN, VOCAB_SIZE, NUM_PAIRS)
        logits, _ = model(seq)
        loss = F.cross_entropy(logits, tgt)
        opt.zero_grad(); loss.backward(); opt.step()
    return model


def evaluate(model, n_batches=50):
    model.eval()
    correct = total = 0
    write_rates = []
    with torch.no_grad():
        for _ in range(n_batches):
            seq, tgt = make_assoc_batch(BATCH_SIZE, SEQ_LEN, VOCAB_SIZE, NUM_PAIRS)
            logits, wr = model(seq, track_write_rate=True)
            pred = logits.argmax(-1)
            correct += (pred == tgt).sum().item()
            total += tgt.size(0)
            write_rates.append(wr)
    acc = correct / total
    avg_wr = sum(write_rates) / len(write_rates)
    return acc, avg_wr


class Exp162SurpriseGatedMemoryUpdate(Experiment):
    experiment_id = "exp_16_2"
    hypothesis = hypothesis

    def run(self) -> ExperimentResult:
        config = dict(
            vocab_size=VOCAB_SIZE, hidden_dim=HIDDEN_DIM, seq_len=SEQ_LEN,
            num_pairs=NUM_PAIRS, train_steps=TRAIN_STEPS, batch_size=BATCH_SIZE,
            inference_lr=INFERENCE_LR, lr=LR, s_values=S_VALUES,
        )

        # First train S=0.0 (baseline, always updates) and share encoder/mlp init
        print("Training baseline (S=0.0)...")
        baseline = SurpriseGatedModel(surprise_threshold=0.0)
        baseline = train_model(baseline, TRAIN_STEPS, BATCH_SIZE)
        acc_baseline, wr_baseline = evaluate(baseline)
        print(f"  S=0.0: acc={acc_baseline:.3f}, write_rate={wr_baseline:.3f}")

        per_s_results = {}
        per_s_results[0.0] = dict(acc=round(acc_baseline, 4), write_rate=round(wr_baseline, 4))

        for S in S_VALUES[1:]:
            print(f"Training S={S}...")
            model_s = SurpriseGatedModel(surprise_threshold=S)
            # Initialize from baseline encoder/mlp weights for fair comparison
            model_s.encoder.load_state_dict(baseline.encoder.state_dict())
            model_s.base_mlp.load_state_dict(baseline.base_mlp.state_dict())
            model_s.output.load_state_dict(baseline.output.state_dict())
            model_s = train_model(model_s, TRAIN_STEPS // 2, BATCH_SIZE)
            acc_s, wr_s = evaluate(model_s)
            per_s_results[S] = dict(acc=round(acc_s, 4), write_rate=round(wr_s, 4))
            print(f"  S={S}: acc={acc_s:.3f}, write_rate={wr_s:.3f}")

        # Check Pareto criterion: write_rate < 0.60 AND acc within 0.02 of baseline
        pareto_found = False
        pareto_S = None
        for S, res in per_s_results.items():
            if S == 0.0:
                continue
            if res["write_rate"] < 0.60 and abs(res["acc"] - acc_baseline) <= 0.02:
                pareto_found = True
                pareto_S = S
                break

        # Check refuted: accuracy drops > 5% before write_rate reaches 0.60
        early_drop = False
        for S, res in per_s_results.items():
            if S == 0.0:
                continue
            if res["write_rate"] > 0.60 and (acc_baseline - res["acc"]) > 0.05:
                early_drop = True
                break

        metrics = dict(
            baseline_acc=round(acc_baseline, 4),
            baseline_write_rate=round(wr_baseline, 4),
            per_threshold=per_s_results,
            pareto_found=pareto_found,
            pareto_threshold=pareto_S,
        )

        if pareto_found:
            outcome = OUTCOME_SUPPORTED
            best = per_s_results[pareto_S]
            notes = (f"Pareto criterion met at S={pareto_S}: "
                     f"write_rate={best['write_rate']:.3f} < 0.60 and "
                     f"acc={best['acc']:.3f} within 0.02 of baseline {acc_baseline:.3f}.")
        elif early_drop:
            outcome = OUTCOME_REFUTED
            notes = "Accuracy drops > 5% before write_rate reaches 0.60."
        else:
            outcome = OUTCOME_INCONCLUSIVE
            notes = "Smooth degradation curve; no clear pareto knee found."

        return self.result(outcome, metrics, notes, config)


if __name__ == "__main__":
    Exp162SurpriseGatedMemoryUpdate().execute()
