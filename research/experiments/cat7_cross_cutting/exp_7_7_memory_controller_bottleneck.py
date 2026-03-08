"""
Experiment 7.7 — Memory Controller Bottleneck

Hypothesis: Write quality (not read quality, compression ratio, or eviction
policy) is the first performance bottleneck encountered during controller
training.

Setup:
  - Full controller: write gate + compression + retrieval + eviction.
  - Train jointly; track 4 metrics at each logging step:
      (1) write_quality:        cosine sim of written entries to originals
      (2) read_accuracy:        retrieval acc@1 given perfect memory
      (3) compression_fidelity: cosine sim after compression
      (4) eviction_correctness: did it evict the right entry
  - SUPPORTED if write_quality has the lowest value for first 40% of training.
  - REFUTED if read_accuracy is the first bottleneck.
  - INCONCLUSIVE if no clear ordering.
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

VOCAB_SIZE     = 64
HIDDEN_DIM     = 64
COMPRESS_DIM   = 32      # compressed slot dimension
SEQ_LEN        = 24
MEMORY_SLOTS   = 8
BATCH_SIZE     = 32
TRAIN_STEPS    = 1500
LOG_EVERY      = 50      # log metrics every N steps
LR             = 3e-4
DEVICE         = "cpu"


# ── Models ────────────────────────────────────────────────────────────────────

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, HIDDEN_DIM)
        self.ff    = nn.Sequential(
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM * 2),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM * 2, HIDDEN_DIM),
        )
        self.norm = nn.LayerNorm(HIDDEN_DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embed(x)
        return self.norm(h + self.ff(h))


class WriteGate(nn.Module):
    """Decide which tokens to write into memory."""
    def __init__(self):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, 1),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.gate(hidden).squeeze(-1)   # (B, L)


class Compressor(nn.Module):
    """Compress HIDDEN_DIM -> COMPRESS_DIM then reconstruct."""
    def __init__(self):
        super().__init__()
        self.encode = nn.Linear(HIDDEN_DIM, COMPRESS_DIM)
        self.decode = nn.Linear(COMPRESS_DIM, HIDDEN_DIM)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.decode(F.relu(self.encode(h)))

    def compress(self, h: torch.Tensor) -> torch.Tensor:
        return F.relu(self.encode(h))          # (B, M, COMPRESS_DIM)


class ReadGate(nn.Module):
    """Attention-based retrieval."""
    def __init__(self):
        super().__init__()
        self.q_proj  = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.out     = nn.Linear(HIDDEN_DIM, VOCAB_SIZE)
        self.query_e = nn.Embedding(VOCAB_SIZE, HIDDEN_DIM)

    def forward(self, query: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        q    = self.q_proj(self.query_e(query)).unsqueeze(1)
        sims = (q * memory).sum(-1) / (HIDDEN_DIM ** 0.5)
        w    = F.softmax(sims, dim=-1).unsqueeze(-1)
        return self.out((w * memory).sum(1))


class EvictionPolicy(nn.Module):
    """Score each memory slot for eviction (lower = evict first)."""
    def __init__(self):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM // 2),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM // 2, 1),
        )

    def forward(self, memory: torch.Tensor) -> torch.Tensor:
        return self.scorer(memory).squeeze(-1)   # (B, M)


# ── Data ──────────────────────────────────────────────────────────────────────

def make_batch(batch_size: int):
    seq   = torch.randint(0, VOCAB_SIZE, (batch_size, SEQ_LEN))
    k_pos = torch.randint(0, SEQ_LEN // 2, (batch_size,))
    v_pos = torch.randint(SEQ_LEN // 2, SEQ_LEN, (batch_size,))
    keys  = torch.randint(0, VOCAB_SIZE // 2, (batch_size,))
    vals  = torch.randint(VOCAB_SIZE // 2, VOCAB_SIZE, (batch_size,))
    for b in range(batch_size):
        seq[b, k_pos[b]] = keys[b]
        seq[b, v_pos[b]] = vals[b]
    return seq, keys, vals


# ── Metric helpers ────────────────────────────────────────────────────────────

def cosine_sim_batch(a: torch.Tensor, b: torch.Tensor) -> float:
    """Mean cosine similarity between corresponding vectors in (B, M, H) tensors."""
    a_n = F.normalize(a.reshape(-1, a.shape[-1]), dim=-1)
    b_n = F.normalize(b.reshape(-1, b.shape[-1]), dim=-1)
    return (a_n * b_n).sum(-1).mean().item()


@torch.no_grad()
def measure_metrics(
    enc: Encoder,
    write_gate: WriteGate,
    compressor: Compressor,
    read_gate: ReadGate,
    eviction: EvictionPolicy,
) -> dict[str, float]:
    enc.eval(); write_gate.eval(); compressor.eval()
    read_gate.eval(); eviction.eval()

    n_batches   = 20
    wq_total    = 0.0   # write quality: cosine sim written vs original
    ra_total    = 0.0   # read accuracy
    cf_total    = 0.0   # compression fidelity
    ec_total    = 0.0   # eviction correctness

    for _ in range(n_batches):
        seq, keys, vals = make_batch(BATCH_SIZE)
        h = enc(seq)                              # (B, L, H)

        # -- Write quality: select top-k, measure cosine sim compressed vs raw --
        logits_w = write_gate(h)                   # (B, L)
        top_idx  = logits_w.topk(MEMORY_SLOTS, dim=-1).indices
        h_sel    = h.gather(1, top_idx.unsqueeze(-1).expand(-1, -1, HIDDEN_DIM))  # (B, M, H)
        h_comp   = compressor(h_sel)               # (B, M, H) reconstructed
        wq_total += cosine_sim_batch(h_comp, h_sel)

        # -- Compression fidelity: reconstructed vs original selected vectors --
        cf_total += cosine_sim_batch(h_comp, h_sel)   # same as write quality here

        # -- Read accuracy: use raw (uncompressed) memory as "perfect" memory --
        out_perfect = read_gate(keys, h_sel)
        ra_total   += (out_perfect.argmax(-1) == vals).float().mean().item()

        # -- Eviction correctness: does eviction target the oldest/worst slot? --
        # We define "correct" eviction = evicting the slot with lowest write logit
        # (the slot least relevant to recent input).
        evict_scores = eviction(h_sel)             # (B, M)
        evict_idx    = evict_scores.argmin(-1)     # (B,) index to evict
        # Ground truth: evict the slot with lowest write logit value
        gt_evict_idx = logits_w.gather(1, top_idx).argmin(-1)  # (B,)
        ec_total    += (evict_idx == gt_evict_idx).float().mean().item()

    enc.train(); write_gate.train(); compressor.train()
    read_gate.train(); eviction.train()

    return {
        "write_quality":        wq_total   / n_batches,
        "read_accuracy":        ra_total   / n_batches,
        "compression_fidelity": cf_total   / n_batches,
        "eviction_correctness": ec_total   / n_batches,
    }


# ── Training ──────────────────────────────────────────────────────────────────

def train_full_controller() -> tuple[dict, list[dict]]:
    torch.manual_seed(42)
    enc        = Encoder()
    write_gate = WriteGate()
    compressor = Compressor()
    read_gate  = ReadGate()
    eviction   = EvictionPolicy()

    all_params = (
        list(enc.parameters())
        + list(write_gate.parameters())
        + list(compressor.parameters())
        + list(read_gate.parameters())
        + list(eviction.parameters())
    )
    opt = Adam(all_params, lr=LR)

    log_history: list[dict] = []

    for step in range(1, TRAIN_STEPS + 1):
        seq, keys, vals = make_batch(BATCH_SIZE)
        h = enc(seq)

        # Write gate selects slots
        logits_w = write_gate(h)
        top_idx  = logits_w.topk(MEMORY_SLOTS, dim=-1).indices
        h_sel    = h.gather(1, top_idx.unsqueeze(-1).expand(-1, -1, HIDDEN_DIM))

        # Compress selected slots
        h_comp   = compressor(h_sel)               # reconstruct

        # Compression loss: reconstructed should match original
        loss_compress = F.mse_loss(h_comp, h_sel.detach())

        # Read from compressed memory
        out_logits   = read_gate(keys, h_comp)
        loss_read    = F.cross_entropy(out_logits, vals)

        # Eviction: should score the "stale" slot lowest
        evict_scores = eviction(h_comp)
        gt_evict     = logits_w.gather(1, top_idx).argmin(-1)
        loss_evict   = F.cross_entropy(-evict_scores, gt_evict)

        loss = loss_read + 0.5 * loss_compress + 0.5 * loss_evict

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(all_params, 1.0)
        opt.step()

        if step % LOG_EVERY == 0:
            m = measure_metrics(enc, write_gate, compressor, read_gate, eviction)
            m["step"] = step
            log_history.append(m)
            print(f"  step={step:4d}  wq={m['write_quality']:.3f}  "
                  f"ra={m['read_accuracy']:.3f}  cf={m['compression_fidelity']:.3f}  "
                  f"ec={m['eviction_correctness']:.3f}")

    final = measure_metrics(enc, write_gate, compressor, read_gate, eviction)
    return final, log_history


# ── Analysis helpers ──────────────────────────────────────────────────────────

METRIC_KEYS = ["write_quality", "read_accuracy", "compression_fidelity", "eviction_correctness"]


def avg_in_range(history: list[dict], frac_lo: float, frac_hi: float) -> dict[str, float]:
    """Average each metric over a fraction of the training log."""
    n     = len(history)
    lo    = int(n * frac_lo)
    hi    = int(n * frac_hi)
    sub   = history[lo:hi] if hi > lo else history[:1]
    return {k: sum(d[k] for d in sub) / len(sub) for k in METRIC_KEYS}


def bottleneck_of(avgs: dict[str, float]) -> str:
    """Return the metric with the lowest average value (= biggest bottleneck)."""
    return min(METRIC_KEYS, key=lambda k: avgs[k])


# ── Experiment ────────────────────────────────────────────────────────────────

class Exp77MemoryControllerBottleneck(Experiment):
    experiment_id = "exp_7_7"
    hypothesis = (
        "Write quality (not read quality, compression ratio, or eviction policy) "
        "is the first performance bottleneck encountered during controller training."
    )

    def run(self) -> ExperimentResult:
        torch.manual_seed(42)
        final, history = train_full_controller()

        # Bottleneck at ~20% and ~50% of training
        bn_20 = bottleneck_of(avg_in_range(history, 0.0, 0.40))
        bn_50 = bottleneck_of(avg_in_range(history, 0.0, 0.50))

        # Ordering: which metric is lowest in first 40% of training steps
        early_avgs  = avg_in_range(history, 0.0, 0.40)
        bottleneck_order = sorted(METRIC_KEYS, key=lambda k: early_avgs[k])

        write_is_bottleneck_early = (bn_20 == "write_quality")
        read_is_bottleneck_early  = (bn_20 == "read_accuracy")

        if write_is_bottleneck_early:
            outcome = OUTCOME_SUPPORTED
        elif read_is_bottleneck_early:
            outcome = OUTCOME_REFUTED
        else:
            outcome = OUTCOME_INCONCLUSIVE

        metrics = {
            "bottleneck_order":            bottleneck_order,
            "bottleneck_at_20pct_training": bn_20,
            "bottleneck_at_50pct_training": bn_50,
            "final_write_quality":         final["write_quality"],
            "final_read_accuracy":         final["read_accuracy"],
            "final_compression_fidelity":  final["compression_fidelity"],
            "final_eviction_correctness":  final["eviction_correctness"],
            "early_averages":              early_avgs,
        }
        notes = (
            f"Bottleneck at 20%: {bn_20}. "
            f"Bottleneck at 50%: {bn_50}. "
            f"Order (lowest first): {bottleneck_order}."
        )
        return self.result(outcome, metrics, notes, config={
            "train_steps":   TRAIN_STEPS,
            "log_every":     LOG_EVERY,
            "compress_dim":  COMPRESS_DIM,
            "memory_slots":  MEMORY_SLOTS,
        })


if __name__ == "__main__":
    Exp77MemoryControllerBottleneck().execute()
