"""
Experiment 5.2 — Read Frequency vs Task Performance

Hypothesis: Optimal read frequency is task-dependent and cannot be determined
by a single fixed schedule across task types.

Setup:
  - Fixed read frequencies: every 1, 2, 4, 8 tokens
  - Memory of 8 slots, retrieval by soft attention
  - Three task types:
    (1) Factual QA: retrieve stored facts from memory
    (2) Sequence completion: predict next token from context
    (3) Pattern matching: detect repeated sequences
  - For each task, measure accuracy at each read frequency
  - SUPPORTED if optimal read frequency differs across task types
  - REFUTED if one frequency is optimal for all tasks
  - INCONCLUSIVE if differences are small
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from experiments.base import (
    Experiment, ExperimentResult,
    OUTCOME_SUPPORTED, OUTCOME_REFUTED, OUTCOME_INCONCLUSIVE,
)

# ── Config ────────────────────────────────────────────────────────────────────

VOCAB_SIZE   = 64
HIDDEN_DIM   = 64
MEMORY_SLOTS = 8
SEQ_LEN      = 24
BATCH_SIZE   = 32
TRAIN_STEPS  = 1500
LR           = 3e-4
DEVICE       = "cpu"

READ_FREQS   = [1, 2, 4, 8]   # read every N tokens


# ── Memory Module ─────────────────────────────────────────────────────────────

class SoftAttentionMemory(nn.Module):
    """Read from memory slots via soft attention over stored keys."""

    def __init__(self):
        super().__init__()
        self.key_proj   = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.val_proj   = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.query_proj = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)

    def forward(self, query: torch.Tensor,
                memory: torch.Tensor) -> torch.Tensor:
        """
        query:  (B, H)
        memory: (B, M, H)
        returns retrieved: (B, H)
        """
        q = self.query_proj(query).unsqueeze(1)           # (B, 1, H)
        k = self.key_proj(memory)                          # (B, M, H)
        v = self.val_proj(memory)                          # (B, M, H)
        scores = (q * k).sum(-1) / (HIDDEN_DIM ** 0.5)    # (B, M)
        w = F.softmax(scores, dim=-1).unsqueeze(-1)        # (B, M, 1)
        return (w * v).sum(1)                              # (B, H)


# ── Shared Model ──────────────────────────────────────────────────────────────

class FixedFreqModel(nn.Module):
    """
    Token-by-token GRU that reads from memory every `read_freq` steps.
    The memory is fixed (not written during inference) and provided externally.
    """

    def __init__(self, read_freq: int):
        super().__init__()
        self.read_freq  = read_freq
        self.embed      = nn.Embedding(VOCAB_SIZE, HIDDEN_DIM)
        self.gru        = nn.GRUCell(HIDDEN_DIM + HIDDEN_DIM, HIDDEN_DIM)
        self.memory_mod = SoftAttentionMemory()
        self.head       = nn.Linear(HIDDEN_DIM, VOCAB_SIZE)

    def forward(
        self,
        seq: torch.Tensor,        # (B, L)
        memory: torch.Tensor,     # (B, M, H)
        target_pos: int = -1,     # position whose logits are returned
    ) -> torch.Tensor:
        B, L = seq.shape
        h = torch.zeros(B, HIDDEN_DIM, device=seq.device)
        retrieved = torch.zeros(B, HIDDEN_DIM, device=seq.device)

        for t in range(L):
            if t % self.read_freq == 0:
                retrieved = self.memory_mod(h, memory)
            x_t = self.embed(seq[:, t])
            inp = torch.cat([x_t, retrieved], dim=-1)
            h   = self.gru(inp, h)

        pos = target_pos if target_pos >= 0 else L - 1
        _ = pos  # use last hidden state regardless
        return self.head(h)  # (B, V)


# ── Task Generators ───────────────────────────────────────────────────────────

def make_factual_qa_batch(batch_size: int):
    """
    Task 1 — Factual QA.
    A fact (token id) is stored in memory slot 0.
    The sequence contains a 'query' trigger at position 0 (token 1).
    Target = the stored fact.
    """
    B = batch_size
    facts   = torch.randint(2, VOCAB_SIZE, (B,))           # the answer
    seq     = torch.randint(2, VOCAB_SIZE, (B, SEQ_LEN))
    seq[:, 0] = 1                                           # query trigger
    # Store fact in memory slot 0 (first 4 dims encode the fact id)
    memory  = torch.randn(B, MEMORY_SLOTS, HIDDEN_DIM) * 0.1
    for b in range(B):
        memory[b, 0, :] = 0.0
        memory[b, 0, facts[b].item() % HIDDEN_DIM] = 2.0
    return seq, facts, memory


def make_seq_completion_batch(batch_size: int):
    """
    Task 2 — Sequence completion.
    The sequence is an arithmetic progression mod VOCAB_SIZE.
    Target = next token in progression (no memory needed).
    """
    B     = batch_size
    start = torch.randint(0, VOCAB_SIZE, (B,))
    step  = torch.randint(1, 5, (B,))
    seq   = torch.stack(
        [(start + step * t) % VOCAB_SIZE for t in range(SEQ_LEN)], dim=1
    )
    target = (start + step * SEQ_LEN) % VOCAB_SIZE
    memory = torch.randn(B, MEMORY_SLOTS, HIDDEN_DIM) * 0.1   # irrelevant
    return seq, target, memory


def make_pattern_match_batch(batch_size: int):
    """
    Task 3 — Pattern matching.
    A pattern of length 4 is repeated every 4 tokens.
    Target = the token that should appear at SEQ_LEN (continuation).
    Memory holds the pattern explicitly in slot 0.
    """
    B = batch_size
    pattern = torch.randint(2, VOCAB_SIZE, (B, 4))
    seq     = pattern.repeat(1, SEQ_LEN // 4)[:, :SEQ_LEN]   # (B, L)
    target  = pattern[:, SEQ_LEN % 4]
    # Store pattern in memory
    memory  = torch.randn(B, MEMORY_SLOTS, HIDDEN_DIM) * 0.1
    for b in range(B):
        for i in range(4):
            memory[b, i, :] = 0.0
            memory[b, i, pattern[b, i].item() % HIDDEN_DIM] = 2.0
    return seq, target, memory


TASKS = {
    "factual_qa":       make_factual_qa_batch,
    "seq_completion":   make_seq_completion_batch,
    "pattern_matching": make_pattern_match_batch,
}


# ── Train / Eval ──────────────────────────────────────────────────────────────

def train_model(read_freq: int, task_fn) -> nn.Module:
    model = FixedFreqModel(read_freq).to(DEVICE)
    opt   = Adam(model.parameters(), lr=LR)

    for _ in range(TRAIN_STEPS):
        seq, tgt, mem = task_fn(BATCH_SIZE)
        logits = model(seq, mem)
        loss   = F.cross_entropy(logits, tgt)
        opt.zero_grad(); loss.backward(); opt.step()

    return model


def eval_accuracy(model: nn.Module, task_fn, n_batches: int = 50) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for _ in range(n_batches):
            seq, tgt, mem = task_fn(BATCH_SIZE)
            logits = model(seq, mem)
            preds  = logits.argmax(-1)
            correct += (preds == tgt).sum().item()
            total   += tgt.size(0)
    model.train()
    return correct / total


# ── Experiment ────────────────────────────────────────────────────────────────

class Exp52ReadFrequencyVsTaskPerformance(Experiment):
    experiment_id = "exp_5_2"
    hypothesis = (
        "Optimal read frequency is task-dependent and cannot be determined "
        "by a single fixed schedule across task types."
    )

    def run(self) -> ExperimentResult:
        torch.manual_seed(42)

        config = {
            "vocab_size":   VOCAB_SIZE,
            "hidden_dim":   HIDDEN_DIM,
            "memory_slots": MEMORY_SLOTS,
            "seq_len":      SEQ_LEN,
            "batch_size":   BATCH_SIZE,
            "train_steps":  TRAIN_STEPS,
            "read_freqs":   READ_FREQS,
        }

        # acc_matrix[task][freq_idx] = accuracy
        task_names = list(TASKS.keys())
        acc_matrix: dict[str, list[float]] = {t: [] for t in task_names}
        optimal_freq: dict[str, int] = {}

        for task_name, task_fn in TASKS.items():
            print(f"\n  Task: {task_name}")
            best_acc  = -1.0
            best_freq = READ_FREQS[0]

            for freq in READ_FREQS:
                model = train_model(freq, task_fn)
                acc   = eval_accuracy(model, task_fn)
                acc_matrix[task_name].append(acc)
                print(f"    freq={freq}  acc={acc:.3f}")

                if acc > best_acc:
                    best_acc  = acc
                    best_freq = freq

            optimal_freq[task_name] = best_freq
            print(f"    => optimal freq: {best_freq}")

        # Flatten acc_matrix for JSON serialisation
        acc_matrix_flat = {
            f"{task}_freq{READ_FREQS[i]}": acc_matrix[task][i]
            for task in task_names
            for i in range(len(READ_FREQS))
        }

        opt_f1 = optimal_freq["factual_qa"]
        opt_f2 = optimal_freq["seq_completion"]
        opt_f3 = optimal_freq["pattern_matching"]
        freqs_differ = len({opt_f1, opt_f2, opt_f3}) > 1

        metrics = {
            "optimal_freq_task1": opt_f1,
            "optimal_freq_task2": opt_f2,
            "optimal_freq_task3": opt_f3,
            "frequencies_differ": freqs_differ,
            **acc_matrix_flat,
        }

        # Check if differences in accuracy are meaningful (> 0.02 between best
        # and second-best frequency for at least one task)
        meaningful_diff = False
        for task in task_names:
            sorted_accs = sorted(acc_matrix[task], reverse=True)
            if len(sorted_accs) >= 2 and (sorted_accs[0] - sorted_accs[1]) > 0.02:
                meaningful_diff = True
                break

        if freqs_differ and meaningful_diff:
            outcome = OUTCOME_SUPPORTED
        elif not freqs_differ:
            outcome = OUTCOME_REFUTED
        else:
            outcome = OUTCOME_INCONCLUSIVE

        notes = (
            f"Optimal frequencies — task1(factual_qa):{opt_f1}, "
            f"task2(seq_completion):{opt_f2}, "
            f"task3(pattern_matching):{opt_f3}. "
            f"Frequencies differ: {freqs_differ}. "
            f"Meaningful accuracy gap between frequencies: {meaningful_diff}."
        )

        return self.result(outcome, metrics, notes, config)


if __name__ == "__main__":
    Exp52ReadFrequencyVsTaskPerformance().execute()
