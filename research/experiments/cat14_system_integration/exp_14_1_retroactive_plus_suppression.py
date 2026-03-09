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

experiment_id = "exp_14_1"
hypothesis = (
    "Combining retroactive write and read confidence suppression yields "
    "super-additive accuracy gains."
)

DEVICE = "cpu"
LR = 3e-4
VOCAB_SIZE = 64
SEQ_LEN = 24
HIDDEN_DIM = 64
MEMORY_SLOTS = 6
NUM_PAIRS = 4
STEPS = 500
BATCH = 32
CONF_THRESHOLD = 0.8
FORWARD_SLOTS = 4
REVISION_SLOTS = 2


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


class MemoryController(nn.Module):
    """Configurable memory with optional retroactive write and read suppression."""
    def __init__(self, hidden_dim, vocab_size, num_slots,
                 use_retroactive=False, use_suppression=False):
        super().__init__()
        self.encoder = Encoder(vocab_size, hidden_dim)
        self.write_gate = nn.Linear(hidden_dim, num_slots)
        self.read_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, vocab_size)
        self.memory = nn.Parameter(torch.zeros(1, num_slots, hidden_dim))
        self.use_retroactive = use_retroactive
        self.use_suppression = use_suppression
        self.hidden_dim = hidden_dim
        self.num_slots = num_slots
        self.vocab_size = vocab_size
        self.conf_threshold = CONF_THRESHOLD

    def write_memory(self, hiddens):
        # hiddens: (B, T, D)
        B, T, D = hiddens.shape
        gate_scores = self.write_gate(hiddens)  # (B, T, S)
        # Forward pass: select top FORWARD_SLOTS positions per batch
        # Average gate score across positions: (B, T, S) -> pick top FORWARD_SLOTS timesteps
        slot_importance = gate_scores.mean(-1)  # (B, T)
        _, top_idx = slot_importance.topk(min(FORWARD_SLOTS, T), dim=-1)  # (B, F)
        mem = self.memory.expand(B, -1, -1).clone()  # (B, S, D)

        # Vectorized forward write: gather selected hidden states and scatter to slots
        F_size = top_idx.shape[1]
        fwd_idx_exp = top_idx.unsqueeze(-1).expand(-1, -1, D)       # (B, F, D)
        fwd_hidden = hiddens.gather(1, fwd_idx_exp)                  # (B, F, D)
        fwd_slots = (torch.arange(F_size, device=hiddens.device)
                     .unsqueeze(0).expand(B, -1) % self.num_slots)   # (B, F)
        fwd_slots_exp = fwd_slots.unsqueeze(-1).expand(-1, -1, D)    # (B, F, D)
        mem = mem.scatter(1, fwd_slots_exp, fwd_hidden)

        if self.use_retroactive:
            # Revision: per-batch loop, but vectorized inner selection
            for b in range(B):
                skipped_b = list(set(range(min(T, 2 * NUM_PAIRS))) - set(top_idx[b].tolist()))
                if len(skipped_b) > 0:
                    skipped_t = torch.tensor(skipped_b, device=hiddens.device)
                    rev_scores = slot_importance[b][skipped_t]
                    n_rev = min(REVISION_SLOTS, len(skipped_b))
                    _, rev_local = rev_scores.topk(n_rev)
                    rev_t = skipped_t[rev_local]
                    rev_slots = ((FORWARD_SLOTS + torch.arange(n_rev, device=hiddens.device))
                                 % self.num_slots)
                    rev_hidden = hiddens[b][rev_t]                   # (n_rev, D)
                    rev_slots_exp = rev_slots.unsqueeze(-1).expand(-1, D)  # (n_rev, D)
                    mem_b = mem[b].scatter(0, rev_slots_exp, rev_hidden)   # (S, D)
                    mem = torch.cat([mem[:b], mem_b.unsqueeze(0), mem[b+1:]], dim=0)
        return mem

    def forward(self, seq):
        # seq: (B, T)
        hiddens = self.encoder(seq)  # (B, T, D)
        mem = self.write_memory(hiddens[:, :-1, :])

        # Query from last token
        query_h = hiddens[:, -1, :]  # (B, D)

        if self.use_suppression:
            # Compute a preliminary output without memory
            direct_logits = self.out(query_h)
            confidence = torch.softmax(direct_logits, dim=-1).max(dim=-1).values  # (B,)
            # If confident, skip memory read
            use_mem = (confidence < self.conf_threshold).float().unsqueeze(-1)  # (B, 1)

        q = self.read_proj(query_h)  # (B, D)
        scores = torch.bmm(mem, q.unsqueeze(-1)).squeeze(-1)  # (B, S)
        attn = torch.softmax(scores, dim=-1)  # (B, S)
        ctx = (attn.unsqueeze(-1) * mem).sum(1)  # (B, D)

        if self.use_suppression:
            ctx = ctx * use_mem + query_h * (1 - use_mem)

        return self.out(ctx)


def train_condition(use_retroactive: bool, use_suppression: bool, seed_offset: int) -> float:
    torch.manual_seed(42 + seed_offset)
    model = MemoryController(HIDDEN_DIM, VOCAB_SIZE, MEMORY_SLOTS,
                             use_retroactive=use_retroactive,
                             use_suppression=use_suppression).to(DEVICE)
    opt = Adam(model.parameters(), lr=LR)

    for step in range(STEPS):
        seq, target = make_assoc_batch(BATCH, SEQ_LEN, VOCAB_SIZE, NUM_PAIRS)
        seq, target = seq.to(DEVICE), target.to(DEVICE)
        logits = model(seq)
        loss = F.cross_entropy(logits, target)
        opt.zero_grad(); loss.backward(); opt.step()

    # Eval
    model.eval()
    correct = 0; total = 0
    with torch.no_grad():
        for _ in range(20):
            seq, target = make_assoc_batch(BATCH, SEQ_LEN, VOCAB_SIZE, NUM_PAIRS)
            seq, target = seq.to(DEVICE), target.to(DEVICE)
            preds = model(seq).argmax(-1)
            correct += (preds == target).sum().item()
            total += BATCH
    return correct / total


class Exp141RetroactivePlusSuppression(Experiment):
    experiment_id = "exp_14_1"
    hypothesis = hypothesis

    def run(self) -> ExperimentResult:
        print("  Training condition A (baseline)...")
        acc_A = train_condition(False, False, 0)
        print("  Training condition B (retroactive write)...")
        acc_B = train_condition(True, False, 1)
        print("  Training condition C (read suppression)...")
        acc_C = train_condition(False, True, 2)
        print("  Training condition D (both)...")
        acc_D = train_condition(True, True, 3)

        gap_B = acc_B - acc_A
        gap_C = acc_C - acc_A
        gap_D = acc_D - acc_A
        super_additive = gap_D > gap_B + gap_C + 0.01

        metrics = {
            "acc_A": round(acc_A, 4),
            "acc_B": round(acc_B, 4),
            "acc_C": round(acc_C, 4),
            "acc_D": round(acc_D, 4),
            "gap_B": round(gap_B, 4),
            "gap_C": round(gap_C, 4),
            "gap_D": round(gap_D, 4),
            "super_additive": super_additive,
        }

        config = dict(VOCAB_SIZE=VOCAB_SIZE, SEQ_LEN=SEQ_LEN, HIDDEN_DIM=HIDDEN_DIM,
                      MEMORY_SLOTS=MEMORY_SLOTS, NUM_PAIRS=NUM_PAIRS,
                      STEPS=STEPS, BATCH=BATCH)

        if super_additive and gap_B > 0.02 and gap_C > 0.01:
            outcome = OUTCOME_SUPPORTED
            notes = (f"Super-additive: gap_D={gap_D:.3f} > gap_B+gap_C={gap_B+gap_C:.3f}+0.01; "
                     f"gap_B={gap_B:.3f}>0.02, gap_C={gap_C:.3f}>0.01.")
        elif gap_D < max(gap_B, gap_C) + 0.005:
            outcome = OUTCOME_REFUTED
            notes = f"gap_D={gap_D:.3f} < max(gap_B,gap_C)+0.005={max(gap_B,gap_C)+0.005:.3f}."
        else:
            outcome = OUTCOME_INCONCLUSIVE
            notes = f"Additive but not super-additive: gap_D={gap_D:.3f}, gap_B+gap_C={gap_B+gap_C:.3f}."

        return self.result(outcome, metrics, notes, config)


if __name__ == "__main__":
    Exp141RetroactivePlusSuppression().execute()
