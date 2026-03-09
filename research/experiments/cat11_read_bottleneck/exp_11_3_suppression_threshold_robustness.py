from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from experiments.base import (Experiment, ExperimentResult, OUTCOME_SUPPORTED, OUTCOME_REFUTED, OUTCOME_INCONCLUSIVE)

experiment_id = "exp_11_3"
hypothesis = "The optimal read suppression threshold T varies systematically by task type — factual QA, pattern matching, and sequence completion each have different optimal thresholds (differ by >0.15)."

DEVICE = "cpu"
LR = 3e-4
VOCAB_SIZE = 64
HIDDEN_DIM = 64
SEQ_LEN = 24
MEMORY_SLOTS = 6
TRAIN_STEPS = 400
BATCH_SIZE = 32
T_VALUES = [0.2, 0.4, 0.6, 0.8, 0.95]


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


def make_pattern_batch(batch_size, seq_len, vocab_size):
    """Pattern task: [A,B,A,B,...] query=A -> predict B."""
    seq = torch.zeros(batch_size, seq_len, dtype=torch.long)
    target = torch.zeros(batch_size, dtype=torch.long)
    for b in range(batch_size):
        token_a = torch.randint(4, 16, (1,)).item()
        token_b = torch.randint(32, min(63, vocab_size - 1), (1,)).item()
        for i in range(seq_len - 3):
            seq[b, i] = token_a if i % 2 == 0 else token_b
        seq[b, seq_len - 3] = 2    # marker
        seq[b, seq_len - 2] = token_a
        seq[b, seq_len - 1] = 0    # blank
        target[b] = token_b
    return seq, target


def make_copy_batch(batch_size, seq_len, vocab_size):
    """Copy task: first half random, query=position -> predict first_half[position]."""
    half = seq_len // 2  # 12
    seq = torch.zeros(batch_size, seq_len, dtype=torch.long)
    target = torch.zeros(batch_size, dtype=torch.long)
    for b in range(batch_size):
        first_half = torch.randint(4, vocab_size, (half,))
        seq[b, :half] = first_half
        # Positions half..seq_len-3 are zeros (blank)
        pos_query = torch.randint(0, half, (1,)).item()
        # Encode query position as token (offset by 4 to avoid special tokens)
        seq[b, seq_len - 3] = 2
        seq[b, seq_len - 2] = pos_query + 4  # position token
        seq[b, seq_len - 1] = 0
        target[b] = first_half[pos_query]
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


class ConfidenceGatedModel(nn.Module):
    """
    Model with confidence-gated reading:
    - Always compute direct logits from hidden[-2]
    - If confidence (max softmax) > T: use direct output
    - Else: read from memory and use that output
    During training: use full output (direct + memory combined) for gradient flow.
    During eval: apply threshold gating.
    """
    def __init__(self, vocab_size, hidden_dim, memory_slots):
        super().__init__()
        self.encoder = Encoder(vocab_size, hidden_dim)
        self.write_gate = nn.Linear(hidden_dim, 1)
        self.read_head = ReadHead(hidden_dim, vocab_size)
        self.direct_out = nn.Linear(hidden_dim, vocab_size)
        self.memory_slots = memory_slots
        self.hidden_dim = hidden_dim

    def build_memory(self, hidden, n_cand):
        B = hidden.shape[0]
        memory_list = []
        mask_list = []
        scores = self.write_gate(hidden[:, :n_cand, :]).squeeze(-1)  # (B, n_cand)
        k = min(self.memory_slots, n_cand)
        _, top_idx = scores.topk(k, dim=1)
        top_idx_sorted, _ = top_idx.sort(dim=1)
        for b in range(B):
            slots = hidden[b][top_idx_sorted[b]]
            n = slots.shape[0]
            if n < self.memory_slots:
                pad = torch.zeros(self.memory_slots - n, self.hidden_dim, device=hidden.device)
                slots = torch.cat([slots, pad], dim=0)
                mask = torch.cat([torch.ones(n), torch.zeros(self.memory_slots - n)]).to(hidden.device)
            else:
                mask = torch.ones(self.memory_slots, device=hidden.device)
            memory_list.append(slots)
            mask_list.append(mask)
        return torch.stack(memory_list), torch.stack(mask_list)

    def forward(self, seq, threshold=None):
        B, T = seq.shape
        hidden = self.encoder(seq)
        n_cand = T - 3
        memory, mask = self.build_memory(hidden, n_cand)
        query_h = hidden[:, -2, :]
        direct_logits = self.direct_out(query_h)
        mem_logits = self.read_head(query_h, memory, mask)

        if threshold is None or not self.training is False:
            # Training: combine both (average) for gradient flow
            return (direct_logits + mem_logits) / 2.0

        # Eval: confidence gating per sample
        confidence = torch.softmax(direct_logits, dim=-1).max(dim=-1).values  # (B,)
        use_direct = confidence > threshold  # (B,) bool
        output = torch.where(use_direct.unsqueeze(-1), direct_logits, mem_logits)
        suppressed = use_direct.float().mean().item()
        return output, suppressed


def get_batch_fn(task_name):
    num_pairs = 4
    if task_name == 'factual':
        return lambda bs: make_assoc_batch(bs, SEQ_LEN, VOCAB_SIZE, num_pairs)
    elif task_name == 'pattern':
        return lambda bs: make_pattern_batch(bs, SEQ_LEN, VOCAB_SIZE)
    elif task_name == 'copy':
        return lambda bs: make_copy_batch(bs, SEQ_LEN, VOCAB_SIZE)
    else:
        raise ValueError(f"Unknown task: {task_name}")


def train_and_eval_task_threshold(task_name, threshold, seed_offset=0):
    torch.manual_seed(42 + seed_offset)
    batch_fn = get_batch_fn(task_name)
    model = ConfidenceGatedModel(VOCAB_SIZE, HIDDEN_DIM, MEMORY_SLOTS).to(DEVICE)
    opt = Adam(model.parameters(), lr=LR)
    model.train()
    for step in range(TRAIN_STEPS):
        seq, target = batch_fn(BATCH_SIZE)
        seq, target = seq.to(DEVICE), target.to(DEVICE)
        logits = model(seq, threshold=None)  # training: no gating
        loss = F.cross_entropy(logits, target)
        opt.zero_grad(); loss.backward(); opt.step()

    model.eval()
    correct = 0; total = 0; total_suppressed = 0
    with torch.no_grad():
        for _ in range(20):
            seq, tgt = batch_fn(BATCH_SIZE)
            seq, tgt = seq.to(DEVICE), tgt.to(DEVICE)
            result = model(seq, threshold=threshold)
            if isinstance(result, tuple):
                output, suppressed = result
                total_suppressed += suppressed
            else:
                output = result
                total_suppressed += 0
            preds = output.argmax(dim=-1)
            correct += (preds == tgt).sum().item()
            total += tgt.shape[0]
    acc = correct / total
    supp_rate = total_suppressed / 20
    return acc, supp_rate


class Exp113SuppressionThresholdRobustness(Experiment):
    experiment_id = "exp_11_3"
    hypothesis = "The optimal read suppression threshold T varies systematically by task type — factual QA, pattern matching, and sequence completion each have different optimal thresholds (differ by >0.15)."

    def run(self) -> ExperimentResult:
        config = dict(VOCAB_SIZE=VOCAB_SIZE, HIDDEN_DIM=HIDDEN_DIM, SEQ_LEN=SEQ_LEN,
                      MEMORY_SLOTS=MEMORY_SLOTS, TRAIN_STEPS=TRAIN_STEPS,
                      BATCH_SIZE=BATCH_SIZE, T_VALUES=T_VALUES)

        tasks = ['factual', 'pattern', 'copy']
        # For each task, train once per threshold (since model weights differ by threshold behavior
        # only at eval; train without gating). To be rigorous, train one model per (task, threshold)
        # pair so threshold can influence training distribution. But since gating only applies at eval,
        # we train one model per task and eval across all thresholds.

        task_results = {}
        seed = 0
        for task in tasks:
            print(f"  Task: {task}")
            task_acc = {}
            task_supp = {}
            for T in T_VALUES:
                acc, supp = train_and_eval_task_threshold(task, T, seed_offset=seed)
                task_acc[T] = acc
                task_supp[T] = supp
                print(f"    T={T:.2f}: acc={acc:.4f}, suppression_rate={supp:.4f}")
                seed += 1
            task_results[task] = (task_acc, task_supp)

        metrics = {}
        optimal_T = {}
        for task in tasks:
            task_acc, task_supp = task_results[task]
            # Penalized score: acc * (1 - 0.1 * suppression_rate)
            best_T = None
            best_score = -1.0
            for T in T_VALUES:
                score = task_acc[T] * (1.0 - 0.1 * task_supp[T])
                if score > best_score:
                    best_score = score
                    best_T = T
            optimal_T[task] = best_T
            metrics[f"optimal_T_{task}"] = best_T
            for T in T_VALUES:
                metrics[f"acc_{task}_T{T}"] = round(task_acc[T], 4)
                metrics[f"supp_{task}_T{T}"] = round(task_supp[T], 4)

        # Check max pairwise difference
        opt_vals = list(optimal_T.values())
        max_diff = 0.0
        for i in range(len(opt_vals)):
            for j in range(i + 1, len(opt_vals)):
                diff = abs(opt_vals[i] - opt_vals[j])
                if diff > max_diff:
                    max_diff = diff
        metrics["max_pairwise_optimal_T_diff"] = round(max_diff, 4)
        metrics["optimal_T_factual"] = optimal_T['factual']
        metrics["optimal_T_pattern"] = optimal_T['pattern']
        metrics["optimal_T_copy"] = optimal_T['copy']

        all_within_005 = all(
            abs(opt_vals[i] - opt_vals[j]) <= 0.05
            for i in range(len(opt_vals))
            for j in range(i + 1, len(opt_vals))
        )

        if max_diff > 0.15:
            outcome = OUTCOME_SUPPORTED
            notes = (f"Max pairwise T difference={max_diff:.3f}>0.15. "
                     f"Optimal T: factual={optimal_T['factual']}, pattern={optimal_T['pattern']}, copy={optimal_T['copy']}.")
        elif all_within_005:
            outcome = OUTCOME_REFUTED
            notes = f"All optimal T values within 0.05 (max_diff={max_diff:.3f})."
        else:
            outcome = OUTCOME_INCONCLUSIVE
            notes = f"Max pairwise diff={max_diff:.3f} — between 0.05 and 0.15, inconclusive."

        return self.result(outcome=outcome, metrics=metrics, notes=notes, config=config)


if __name__ == "__main__":
    Exp113SuppressionThresholdRobustness().execute()
