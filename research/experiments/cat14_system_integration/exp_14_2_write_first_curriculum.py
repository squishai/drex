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

experiment_id = "exp_14_2"
hypothesis = (
    "Write-first curriculum (train write gate before enabling reads) outperforms "
    "joint training from step 0."
)

DEVICE = "cpu"
LR = 3e-4
VOCAB_SIZE = 64
SEQ_LEN = 24
HIDDEN_DIM = 64
MEMORY_SLOTS = 6
NUM_PAIRS = 4
PHASE1 = 250
PHASE2 = 250
BATCH = 32
K_WRITE = 4


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
        self.ff = nn.Sequential(nn.Linear(hidden_dim, hidden_dim * 2), nn.ReLU(),
                                nn.Linear(hidden_dim * 2, hidden_dim))
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        h = self.embed(x)
        return self.norm(h + self.ff(h))


class WriteFirstModel(nn.Module):
    def __init__(self, vocab_size, hidden_dim, num_slots):
        super().__init__()
        self.encoder = Encoder(vocab_size, hidden_dim)
        self.write_gate = nn.Linear(hidden_dim, num_slots)
        self.read_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, vocab_size)
        self.hidden_dim = hidden_dim
        self.num_slots = num_slots

    def write_to_memory(self, hiddens):
        # hiddens: (B, T, D)
        B, T, D = hiddens.shape
        gate_scores = self.write_gate(hiddens).mean(-1)  # (B, T)
        k_w = min(K_WRITE, T)
        _, top_idx = gate_scores.topk(k_w, dim=-1)  # (B, K)

        # Vectorized: gather selected hidden states and scatter to slots
        top_idx_exp = top_idx.unsqueeze(-1).expand(-1, -1, D)        # (B, K, D)
        gathered = hiddens.gather(1, top_idx_exp)                    # (B, K, D)
        mem = torch.zeros(B, self.num_slots, D, device=hiddens.device)
        slot_indices = (torch.arange(k_w, device=hiddens.device)
                        .unsqueeze(0).expand(B, -1) % self.num_slots)  # (B, K)
        slot_indices_exp = slot_indices.unsqueeze(-1).expand(-1, -1, D)  # (B, K, D)
        mem.scatter_(1, slot_indices_exp, gathered)
        return mem, top_idx

    def forward(self, seq):
        hiddens = self.encoder(seq)  # (B, T, D)
        context_hiddens = hiddens[:, :-1, :]
        mem, _ = self.write_to_memory(context_hiddens)
        query_h = hiddens[:, -1, :]
        q = self.read_proj(query_h)
        scores = torch.bmm(mem, q.unsqueeze(-1)).squeeze(-1)
        attn = torch.softmax(scores, dim=-1)
        ctx = (attn.unsqueeze(-1) * mem).sum(1)
        return self.out(ctx)

    def write_quality(self, seq):
        """Compute mean cosine_sim(memory_slots, written_token_hiddens) over a batch."""
        with torch.no_grad():
            hiddens = self.encoder(seq)
            context_hiddens = hiddens[:, :-1, :]
            B, T, D = context_hiddens.shape
            gate_scores = self.write_gate(context_hiddens).mean(-1)
            k_w = min(K_WRITE, T)
            _, top_idx = gate_scores.topk(k_w, dim=-1)

            # Vectorized gather of written hiddens
            top_idx_exp = top_idx.unsqueeze(-1).expand(-1, -1, D)      # (B, K, D)
            written_hiddens = context_hiddens.gather(1, top_idx_exp)   # (B, K, D)

            mem = torch.zeros(B, self.num_slots, D, device=hiddens.device)
            slot_indices = (torch.arange(k_w, device=hiddens.device)
                            .unsqueeze(0).expand(B, -1) % self.num_slots)  # (B, K)
            slot_indices_exp = slot_indices.unsqueeze(-1).expand(-1, -1, D)
            mem.scatter_(1, slot_indices_exp, written_hiddens)

            # Vectorized cosim: mem slots vs written hiddens
            mem_slots = mem[:, slot_indices[0], :]                     # (B, K, D)
            cs = F.cosine_similarity(
                mem_slots.reshape(-1, D), written_hiddens.reshape(-1, D), dim=-1
            ).mean().item()
            return cs


def eval_accuracy(model, num_batches=20):
    model.eval()
    correct = 0; total = 0
    with torch.no_grad():
        for _ in range(num_batches):
            seq, target = make_assoc_batch(BATCH, SEQ_LEN, VOCAB_SIZE, NUM_PAIRS)
            seq, target = seq.to(DEVICE), target.to(DEVICE)
            preds = model(seq).argmax(-1)
            correct += (preds == target).sum().item()
            total += BATCH
    model.train()
    return correct / total


class Exp142WriteFirstCurriculum(Experiment):
    experiment_id = "exp_14_2"
    hypothesis = hypothesis

    def run(self) -> ExperimentResult:
        # Condition A: joint training from step 0
        print("  Training condition A (joint from step 0)...")
        torch.manual_seed(42)
        model_A = WriteFirstModel(VOCAB_SIZE, HIDDEN_DIM, MEMORY_SLOTS).to(DEVICE)
        opt_A = Adam(model_A.parameters(), lr=LR)
        wq_A_1000 = None

        for step in range(PHASE1 + PHASE2):
            seq, target = make_assoc_batch(BATCH, SEQ_LEN, VOCAB_SIZE, NUM_PAIRS)
            seq, target = seq.to(DEVICE), target.to(DEVICE)
            logits = model_A(seq)
            loss = F.cross_entropy(logits, target)
            opt_A.zero_grad(); loss.backward(); opt_A.step()
            if step == PHASE1 - 1:
                wq_A_1000 = model_A.write_quality(seq)

        acc_A = eval_accuracy(model_A)

        # Condition B: phase 1 = write gate + encoder reconstruction, phase 2 = joint
        print("  Training condition B (write-first curriculum)...")
        torch.manual_seed(42)
        model_B = WriteFirstModel(VOCAB_SIZE, HIDDEN_DIM, MEMORY_SLOTS).to(DEVICE)

        # Phase 1: train only encoder + write_gate on slot reconstruction MSE
        phase1_params = list(model_B.encoder.parameters()) + list(model_B.write_gate.parameters())
        opt_B1 = Adam(phase1_params, lr=LR)
        wq_B_1000 = None

        for step in range(PHASE1):
            seq, _ = make_assoc_batch(BATCH, SEQ_LEN, VOCAB_SIZE, NUM_PAIRS)
            seq = seq.to(DEVICE)
            hiddens = model_B.encoder(seq)
            context_hiddens = hiddens[:, :-1, :]
            B_, T_, D_ = context_hiddens.shape
            gate_scores = model_B.write_gate(context_hiddens).mean(-1)
            _, top_idx = gate_scores.topk(min(K_WRITE, T_), dim=-1)

            # Vectorized MSE loss: gather all selected slot hiddens at once
            k_w = min(K_WRITE, T_)
            top_idx_exp = top_idx.unsqueeze(-1).expand(-1, -1, D_)    # (B_, K, D)
            slot_hiddens = context_hiddens.gather(1, top_idx_exp)     # (B_, K, D)
            target_h_all = (context_hiddens.mean(dim=1, keepdim=True)
                            .detach().expand(-1, k_w, -1))             # (B_, K, D)
            loss = F.mse_loss(slot_hiddens, target_h_all)

            opt_B1.zero_grad(); loss.backward(); opt_B1.step()

        with torch.no_grad():
            seq_eval, _ = make_assoc_batch(BATCH, SEQ_LEN, VOCAB_SIZE, NUM_PAIRS)
            wq_B_1000 = model_B.write_quality(seq_eval.to(DEVICE))

        # Phase 2: unfreeze all, train jointly on task loss
        opt_B2 = Adam(model_B.parameters(), lr=LR)
        for step in range(PHASE2):
            seq, target = make_assoc_batch(BATCH, SEQ_LEN, VOCAB_SIZE, NUM_PAIRS)
            seq, target = seq.to(DEVICE), target.to(DEVICE)
            logits = model_B(seq)
            loss = F.cross_entropy(logits, target)
            opt_B2.zero_grad(); loss.backward(); opt_B2.step()

        acc_B = eval_accuracy(model_B)

        metrics = {
            "acc_A": round(acc_A, 4),
            "acc_B": round(acc_B, 4),
            "write_quality_at_1000_A": round(wq_A_1000, 4) if wq_A_1000 is not None else 0.0,
            "write_quality_at_1000_B": round(wq_B_1000, 4) if wq_B_1000 is not None else 0.0,
            "acc_diff_B_minus_A": round(acc_B - acc_A, 4),
            "wq_diff_B_minus_A": round((wq_B_1000 or 0) - (wq_A_1000 or 0), 4),
        }

        config = dict(VOCAB_SIZE=VOCAB_SIZE, SEQ_LEN=SEQ_LEN, HIDDEN_DIM=HIDDEN_DIM,
                      MEMORY_SLOTS=MEMORY_SLOTS, NUM_PAIRS=NUM_PAIRS,
                      PHASE1=PHASE1, PHASE2=PHASE2, BATCH=BATCH)

        wq_diff = (wq_B_1000 or 0) - (wq_A_1000 or 0)
        acc_diff = acc_B - acc_A

        if acc_diff > 0.02 and wq_diff > 0.05:
            outcome = OUTCOME_SUPPORTED
            notes = f"Write-first B beats A by {acc_diff:.3f} acc and {wq_diff:.3f} write quality."
        elif acc_A >= acc_B - 0.01:
            outcome = OUTCOME_REFUTED
            notes = f"Joint training A >= curriculum B - 0.01: acc_A={acc_A:.3f}, acc_B={acc_B:.3f}."
        else:
            outcome = OUTCOME_INCONCLUSIVE
            notes = f"acc_diff={acc_diff:.3f}, wq_diff={wq_diff:.3f} — partial improvement."

        return self.result(outcome, metrics, notes, config)


if __name__ == "__main__":
    Exp142WriteFirstCurriculum().execute()
