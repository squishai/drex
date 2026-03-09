from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from experiments.base import (Experiment, ExperimentResult, OUTCOME_SUPPORTED, OUTCOME_REFUTED, OUTCOME_INCONCLUSIVE)

experiment_id = "exp_11_1"
hypothesis = "A two-step query former (linear -> cross-attention over last 4 hidden states -> linear) shifts the bottleneck away from read accuracy (identified in exp_7_7)."

DEVICE = "cpu"
LR = 3e-4
VOCAB_SIZE = 64
HIDDEN_DIM = 64
SEQ_LEN = 24
MEMORY_SLOTS = 8
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


class WriteGate(nn.Module):
    def __init__(self, hidden_dim, memory_slots):
        super().__init__()
        self.gate = nn.Linear(hidden_dim, 1)
        self.memory_slots = memory_slots
        self.hidden_dim = hidden_dim

    def forward(self, hidden):
        # hidden: (B, T, H); returns memory (B, slots, H), mask (B, slots)
        B, T, H = hidden.shape
        scores = self.gate(hidden).squeeze(-1)  # (B, T)
        k = min(self.memory_slots, T)
        _, top_idx = scores.topk(k, dim=1)
        top_idx_sorted, _ = top_idx.sort(dim=1)
        memory_list = []
        mask_list = []
        for b in range(B):
            slots = hidden[b][top_idx_sorted[b]]  # (k, H)
            n = slots.shape[0]
            if n < self.memory_slots:
                pad = torch.zeros(self.memory_slots - n, H, device=hidden.device)
                slots = torch.cat([slots, pad], dim=0)
                mask = torch.cat([torch.ones(n), torch.zeros(self.memory_slots - n)]).to(hidden.device)
            else:
                mask = torch.ones(self.memory_slots, device=hidden.device)
            memory_list.append(slots)
            mask_list.append(mask)
        return torch.stack(memory_list), torch.stack(mask_list)


class QueryFormerA(nn.Module):
    """Single linear projection."""
    def __init__(self, hidden_dim):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, hidden_dim)
    def forward(self, hidden):
        # hidden: (B, T, H) — use last token
        return self.proj(hidden[:, -2, :])  # (B, H)


class QueryFormerB(nn.Module):
    """Two-step: linear -> cross-attn over last 4 -> linear."""
    def __init__(self, hidden_dim):
        super().__init__()
        self.pre_proj = nn.Linear(hidden_dim, hidden_dim)
        self.mha = nn.MultiheadAttention(hidden_dim, num_heads=1, batch_first=True)
        self.post_proj = nn.Linear(hidden_dim, hidden_dim)
    def forward(self, hidden):
        B, T, H = hidden.shape
        last = hidden[:, -2, :].unsqueeze(1)      # (B, 1, H)
        ctx = hidden[:, max(0, T - 4):, :]        # (B, <=4, H) — last 4 positions
        pre = self.pre_proj(last)
        attn_out, _ = self.mha(pre, ctx, ctx)      # (B, 1, H)
        return self.post_proj(attn_out.squeeze(1)) # (B, H)


class QueryFormerC(nn.Module):
    """Three-step: same as B + MLP after attention."""
    def __init__(self, hidden_dim):
        super().__init__()
        self.pre_proj = nn.Linear(hidden_dim, hidden_dim)
        self.mha = nn.MultiheadAttention(hidden_dim, num_heads=1, batch_first=True)
        self.post_proj = nn.Linear(hidden_dim, hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
    def forward(self, hidden):
        B, T, H = hidden.shape
        last = hidden[:, -2, :].unsqueeze(1)
        ctx = hidden[:, max(0, T - 4):, :]
        pre = self.pre_proj(last)
        attn_out, _ = self.mha(pre, ctx, ctx)
        mid = self.post_proj(attn_out.squeeze(1))
        return self.mlp(mid)


class MultistepModel(nn.Module):
    def __init__(self, vocab_size, hidden_dim, memory_slots, query_former_type):
        super().__init__()
        self.encoder = Encoder(vocab_size, hidden_dim)
        self.write_gate = WriteGate(hidden_dim, memory_slots)
        self.read_head = ReadHead(hidden_dim, vocab_size)
        self.hidden_dim = hidden_dim
        if query_former_type == 'A':
            self.query_former = QueryFormerA(hidden_dim)
        elif query_former_type == 'B':
            self.query_former = QueryFormerB(hidden_dim)
        elif query_former_type == 'C':
            self.query_former = QueryFormerC(hidden_dim)

    def forward(self, seq):
        hidden = self.encoder(seq)
        memory, mask = self.write_gate(hidden[:, :-3, :])
        query_h = self.query_former(hidden)
        return self.read_head(query_h, memory, mask)


def make_oracle_memory(seq, target, hidden, memory_slots, vocab_size):
    """Force the correct target token into memory for oracle read accuracy measurement."""
    B, T, H = hidden.shape
    memory_list = []
    mask_list = []
    for b in range(B):
        # Put the target embedding at slot 0, fill rest with other hidden states
        tgt_val = target[b].item()
        # Find position of val token in seq
        val_positions = (seq[b] == tgt_val).nonzero(as_tuple=True)[0]
        if len(val_positions) > 0:
            pos = val_positions[0].item()
            anchor = hidden[b, pos].unsqueeze(0)
        else:
            anchor = hidden[b, 0].unsqueeze(0)
        # Fill remaining slots
        remaining = min(memory_slots - 1, T - 1)
        others = hidden[b, :remaining]
        slots = torch.cat([anchor, others], dim=0)
        n = slots.shape[0]
        if n < memory_slots:
            pad = torch.zeros(memory_slots - n, H, device=hidden.device)
            slots = torch.cat([slots, pad], dim=0)
            mask = torch.cat([torch.ones(n), torch.zeros(memory_slots - n)]).to(hidden.device)
        else:
            mask = torch.ones(memory_slots, device=hidden.device)
        memory_list.append(slots)
        mask_list.append(mask)
    return torch.stack(memory_list), torch.stack(mask_list)


def train_model(qtype, seed_offset=0):
    torch.manual_seed(42 + seed_offset)
    model = MultistepModel(VOCAB_SIZE, HIDDEN_DIM, MEMORY_SLOTS, qtype).to(DEVICE)
    opt = Adam(model.parameters(), lr=LR)

    checkpoints = {
        int(0.2 * TRAIN_STEPS): None,
        int(0.5 * TRAIN_STEPS): None,
        int(1.0 * TRAIN_STEPS): None,
    }

    model.train()
    for step in range(1, TRAIN_STEPS + 1):
        seq, target = make_assoc_batch(BATCH_SIZE, SEQ_LEN, VOCAB_SIZE, NUM_PAIRS)
        seq, target = seq.to(DEVICE), target.to(DEVICE)
        logits = model(seq)
        loss = F.cross_entropy(logits, target)
        opt.zero_grad(); loss.backward(); opt.step()

        if step in checkpoints:
            checkpoints[step] = {k: v.clone() for k, v in model.state_dict().items()}

    return model, checkpoints


def eval_oracle_acc(model, checkpoint_state):
    """Evaluate oracle read accuracy using a saved checkpoint."""
    model.load_state_dict(checkpoint_state)
    model.eval()
    correct = 0; total = 0
    with torch.no_grad():
        for _ in range(20):
            seq, target = make_assoc_batch(BATCH_SIZE, SEQ_LEN, VOCAB_SIZE, NUM_PAIRS)
            seq, target = seq.to(DEVICE), target.to(DEVICE)
            hidden = model.encoder(seq)
            memory, mask = make_oracle_memory(seq, target, hidden, MEMORY_SLOTS, VOCAB_SIZE)
            query_h = model.query_former(hidden)
            logits = model.read_head(query_h, memory, mask)
            preds = logits.argmax(dim=-1)
            correct += (preds == target).sum().item()
            total += target.shape[0]
    return correct / total


def eval_normal_acc(model):
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


class Exp111MultistepQueryFormer(Experiment):
    experiment_id = "exp_11_1"
    hypothesis = "A two-step query former (linear -> cross-attention over last 4 hidden states -> linear) shifts the bottleneck away from read accuracy (identified in exp_7_7)."

    def run(self) -> ExperimentResult:
        config = dict(VOCAB_SIZE=VOCAB_SIZE, HIDDEN_DIM=HIDDEN_DIM, SEQ_LEN=SEQ_LEN,
                      MEMORY_SLOTS=MEMORY_SLOTS, NUM_PAIRS=NUM_PAIRS,
                      TRAIN_STEPS=TRAIN_STEPS, BATCH_SIZE=BATCH_SIZE)
        metrics = {}
        oracle_finals = {}

        for i, qtype in enumerate(['A', 'B', 'C']):
            print(f"  Training query former {qtype} ...")
            model, checkpoints = train_model(qtype, seed_offset=i)
            step_keys = sorted(checkpoints.keys())

            oracle_accs = {}
            for step in step_keys:
                if checkpoints[step] is not None:
                    oacc = eval_oracle_acc(model, checkpoints[step])
                    oracle_accs[step] = oacc
                    print(f"    [{qtype}] step={step} oracle_read_acc={oacc:.4f}")

            # Restore final weights for normal acc
            final_step = max(step_keys)
            if checkpoints[final_step] is not None:
                model.load_state_dict(checkpoints[final_step])
            normal_acc = eval_normal_acc(model)
            print(f"    [{qtype}] final normal_acc={normal_acc:.4f}")

            oracle_final = oracle_accs.get(final_step, 0.0)
            oracle_finals[qtype] = oracle_final
            metrics[f"oracle_acc_{qtype}_final"] = round(oracle_final, 4)
            metrics[f"normal_acc_{qtype}"] = round(normal_acc, 4)
            for step, oacc in oracle_accs.items():
                frac = round(step / TRAIN_STEPS, 1)
                metrics[f"oracle_acc_{qtype}_step{frac}"] = round(oacc, 4)

        oracle_acc_A = oracle_finals['A']
        oracle_acc_B = oracle_finals['B']
        improvement = oracle_acc_B - oracle_acc_A
        metrics["oracle_acc_B_minus_A"] = round(improvement, 4)

        if improvement > 0.10:
            outcome = OUTCOME_SUPPORTED
            notes = f"oracle_read_acc_B={oracle_acc_B:.3f} > oracle_read_acc_A={oracle_acc_A:.3f} by {improvement:.3f}>0.10."
        elif improvement <= 0.02:
            outcome = OUTCOME_REFUTED
            notes = f"oracle_read_acc_B={oracle_acc_B:.3f} not > oracle_read_acc_A={oracle_acc_A:.3f} by 0.02."
        else:
            outcome = OUTCOME_INCONCLUSIVE
            notes = f"Improvement={improvement:.3f} is between 0.02 and 0.10 — inconclusive."

        return self.result(outcome=outcome, metrics=metrics, notes=notes, config=config)


if __name__ == "__main__":
    Exp111MultistepQueryFormer().execute()
