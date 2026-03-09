from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from experiments.base import (Experiment, ExperimentResult, OUTCOME_SUPPORTED, OUTCOME_REFUTED, OUTCOME_INCONCLUSIVE)

experiment_id = "exp_11_2"
hypothesis = "Read-before-write duplicate suppression (skip write if cosine similarity to any existing memory slot > 0.8) improves retrieval F1 by >3% without reducing recall by >5%."

DEVICE = "cpu"
LR = 3e-4
VOCAB_SIZE = 64
HIDDEN_DIM = 64
SEQ_LEN = 24
MEMORY_SLOTS = 8
NUM_PAIRS = 4
TRAIN_STEPS = 600
BATCH_SIZE = 32
DEDUP_THRESHOLD = 0.8


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


class RBWModel(nn.Module):
    """Read-Before-Write model."""
    def __init__(self, vocab_size, hidden_dim, memory_slots, dedup, threshold=0.8):
        super().__init__()
        self.encoder = Encoder(vocab_size, hidden_dim)
        self.write_gate = nn.Linear(hidden_dim, 1)
        self.read_head = ReadHead(hidden_dim, vocab_size)
        self.dedup = dedup
        self.threshold = threshold
        self.memory_slots = memory_slots
        self.hidden_dim = hidden_dim

    def _build_memory(self, hidden, seq_len_candidates):
        """Build memory with or without dedup. Returns memory, mask, write_count."""
        B = hidden.shape[0]
        memory_list = []
        mask_list = []
        write_counts = []

        # Vectorize: compute gate scores for all batch items at once
        scores_all = self.write_gate(hidden[:, :seq_len_candidates, :]).squeeze(-1)  # (B, n)
        for b in range(B):
            scores = scores_all[b]  # (n,) — already computed
            # Rank tokens by gate score
            ranked_idx = scores.argsort(descending=True)
            slots = []
            for t in ranked_idx.tolist():
                if len(slots) >= self.memory_slots:
                    break
                h_t = hidden[b, t]
                if self.dedup and len(slots) > 0:
                    existing = torch.stack(slots)  # (k, H)
                    h_norm = F.normalize(h_t.unsqueeze(0), dim=-1)
                    e_norm = F.normalize(existing, dim=-1)
                    cos_sims = (e_norm * h_norm).sum(-1)
                    if cos_sims.max().item() > self.threshold:
                        continue  # skip — too similar
                slots.append(h_t)

            write_counts.append(len(slots))
            n = len(slots)
            if n == 0:
                slots_tensor = torch.zeros(self.memory_slots, self.hidden_dim, device=hidden.device)
                mask = torch.zeros(self.memory_slots, device=hidden.device)
            else:
                slots_tensor = torch.stack(slots)
                if n < self.memory_slots:
                    pad = torch.zeros(self.memory_slots - n, self.hidden_dim, device=hidden.device)
                    slots_tensor = torch.cat([slots_tensor, pad], dim=0)
                    mask = torch.cat([torch.ones(n), torch.zeros(self.memory_slots - n)]).to(hidden.device)
                else:
                    mask = torch.ones(self.memory_slots, device=hidden.device)
            memory_list.append(slots_tensor)
            mask_list.append(mask)

        memory = torch.stack(memory_list)
        mask_out = torch.stack(mask_list)
        return memory, mask_out, write_counts

    def forward(self, seq):
        B, T = seq.shape
        hidden = self.encoder(seq)
        n_cand = T - 3
        memory, mask, _ = self._build_memory(hidden, n_cand)
        query_h = hidden[:, -2, :]
        return self.read_head(query_h, memory, mask)

    def forward_with_memory(self, seq):
        B, T = seq.shape
        hidden = self.encoder(seq)
        n_cand = T - 3
        memory, mask, write_counts = self._build_memory(hidden, n_cand)
        query_h = hidden[:, -2, :]
        logits = self.read_head(query_h, memory, mask)
        return logits, memory, mask, write_counts


def train_model(dedup, seed_offset=0):
    torch.manual_seed(42 + seed_offset)
    model = RBWModel(VOCAB_SIZE, HIDDEN_DIM, MEMORY_SLOTS, dedup=dedup, threshold=DEDUP_THRESHOLD).to(DEVICE)
    opt = Adam(model.parameters(), lr=LR)
    model.train()
    for step in range(TRAIN_STEPS):
        seq, target = make_assoc_batch(BATCH_SIZE, SEQ_LEN, VOCAB_SIZE, NUM_PAIRS)
        seq, target = seq.to(DEVICE), target.to(DEVICE)
        logits = model(seq)
        loss = F.cross_entropy(logits, target)
        opt.zero_grad(); loss.backward(); opt.step()
    return model


def eval_metrics(model, n_batches=50):
    """Compute precision@1, recall@1, f1, write_rate."""
    model.eval()
    total = 0
    precision_correct = 0
    recall_correct = 0
    total_writes = 0
    total_possible = 0

    with torch.no_grad():
        for _ in range(n_batches):
            seq, target = make_assoc_batch(BATCH_SIZE, SEQ_LEN, VOCAB_SIZE, NUM_PAIRS)
            seq, target = seq.to(DEVICE), target.to(DEVICE)
            logits, memory, mask, write_counts = model.forward_with_memory(seq)
            B = target.shape[0]

            # precision@1: does top-1 logit match target?
            preds = logits.argmax(dim=-1)
            precision_correct += (preds == target).sum().item()

            # recall@1: is target anywhere in memory slots?
            encoder = model.encoder
            target_embeds = encoder.embed(target)  # (B, H)
            for b in range(B):
                tgt_embed = target_embeds[b]
                mem_b = memory[b]  # (slots, H)
                mask_b = mask[b]   # (slots,)
                active_slots = mem_b[mask_b > 0]
                if active_slots.shape[0] > 0:
                    cos_sims = F.cosine_similarity(active_slots, tgt_embed.unsqueeze(0).expand_as(active_slots), dim=-1)
                    recall_correct += int(cos_sims.max().item() > 0.5)
                # else: target not in memory

            n_cand = SEQ_LEN - 3
            total_writes += sum(write_counts)
            total_possible += B * n_cand
            total += B

    p = precision_correct / total
    r = recall_correct / total
    f1 = 2 * p * r / (p + r + 1e-8)
    write_rate = total_writes / total_possible
    return p, r, f1, write_rate


class Exp112ReadBeforeWrite(Experiment):
    experiment_id = "exp_11_2"
    hypothesis = "Read-before-write duplicate suppression (skip write if cosine similarity to any existing memory slot > 0.8) improves retrieval F1 by >3% without reducing recall by >5%."

    def run(self) -> ExperimentResult:
        config = dict(VOCAB_SIZE=VOCAB_SIZE, HIDDEN_DIM=HIDDEN_DIM, SEQ_LEN=SEQ_LEN,
                      MEMORY_SLOTS=MEMORY_SLOTS, NUM_PAIRS=NUM_PAIRS,
                      TRAIN_STEPS=TRAIN_STEPS, BATCH_SIZE=BATCH_SIZE,
                      DEDUP_THRESHOLD=DEDUP_THRESHOLD)

        print("  Training condition A (standard write) ...")
        model_A = train_model(dedup=False, seed_offset=0)
        p_A, r_A, f1_A, wr_A = eval_metrics(model_A)
        print(f"    P={p_A:.4f} R={r_A:.4f} F1={f1_A:.4f} write_rate={wr_A:.4f}")

        print("  Training condition B (read-before-write) ...")
        model_B = train_model(dedup=True, seed_offset=1)
        p_B, r_B, f1_B, wr_B = eval_metrics(model_B)
        print(f"    P={p_B:.4f} R={r_B:.4f} F1={f1_B:.4f} write_rate={wr_B:.4f}")

        f1_gain = f1_B - f1_A
        recall_drop = r_A - r_B

        metrics = {
            "precision_A": round(p_A, 4),
            "recall_A": round(r_A, 4),
            "f1_A": round(f1_A, 4),
            "write_rate_A": round(wr_A, 4),
            "precision_B": round(p_B, 4),
            "recall_B": round(r_B, 4),
            "f1_B": round(f1_B, 4),
            "write_rate_B": round(wr_B, 4),
            "f1_gain": round(f1_gain, 4),
            "recall_drop": round(recall_drop, 4),
        }

        if f1_B > f1_A + 0.03 and recall_drop <= 0.05:
            outcome = OUTCOME_SUPPORTED
            notes = f"F1 gain={f1_gain:.3f}>0.03 and recall drop={recall_drop:.3f}<=0.05."
        elif f1_B <= f1_A + 0.01:
            outcome = OUTCOME_REFUTED
            notes = f"F1_B={f1_B:.3f} not > F1_A={f1_A:.3f} by 0.01."
        else:
            outcome = OUTCOME_INCONCLUSIVE
            notes = f"F1 gain={f1_gain:.3f}, recall drop={recall_drop:.3f} — inconclusive."

        return self.result(outcome=outcome, metrics=metrics, notes=notes, config=config)


if __name__ == "__main__":
    Exp112ReadBeforeWrite().execute()
