"""
Experiment 2.3 — Lossy vs Lossless Storage (Mixed-Precision Memory)

Hypothesis: A controller can learn without supervision which information should be
stored exactly (numbers, names) vs approximately (context, themes).

Setup:
  - EXACT tokens (IDs 0-31): require lossless storage
  - APPROX tokens (IDs 32-127): can tolerate lossy storage
  - Gated memory: gate MLP sees the token embedding and outputs a precision decision
  - High-precision path: full dim=64 (no compression)
  - Low-precision path: compress dim=64 → dim=16 → dim=64 (lossy)
  - Training loss weights: exact tokens penalized more heavily for reconstruction error
  - Precision budget penalty encourages the gate to prefer low-precision by default
  - Gate does NOT receive the exact/approx label — must learn from the embeddings
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

VOCAB_SIZE          = 256
HIDDEN_DIM          = 64
LOW_DIM             = 16          # bottleneck for low-precision path
EXACT_TOKEN_MAX     = 32          # tokens 0–31 are "exact"
APPROX_TOKEN_MIN    = 32          # tokens 32–127 are "approximate"
APPROX_TOKEN_MAX    = 128
BATCH_SIZE          = 32
TRAIN_STEPS         = 2000
EVAL_BATCHES        = 200
LR                  = 1e-3
DEVICE              = "cpu"
EXACT_LOSS_WEIGHT   = 8.0         # high penalty for reconstruction error on exact tokens
APPROX_LOSS_WEIGHT  = 1.0         # low penalty for approx tokens
PRECISION_PENALTY   = 0.05        # L1 regularizer discouraging high-precision usage
INCONCLUSIVE_DELTA  = 0.05        # |diff| < this → INCONCLUSIVE


# ── Model ─────────────────────────────────────────────────────────────────────

class GatedMemory(nn.Module):
    """
    Mixed-precision memory controlled by a learned gate.

    Gate MLP takes a token embedding and outputs g ∈ [0, 1]:
      g → 1  means route through the high-precision path (full dim, low loss)
      g → 0  means route through the low-precision path (64→16→64, lossy)

    The gate never sees the exact/approx label — it sees only the embedding.
    """

    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, HIDDEN_DIM)

        # Gate: embedding → precision probability (no label supervision)
        self.gate_mlp = nn.Sequential(
            nn.Linear(HIDDEN_DIM, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

        # Low-precision path: 64 → 16 → 64
        self.lp_encoder = nn.Linear(HIDDEN_DIM, LOW_DIM)
        self.lp_decoder = nn.Linear(LOW_DIM, HIDDEN_DIM)

    def forward(
        self, token_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            out      : (B, HIDDEN_DIM)  gated reconstruction
            g        : (B,)             gate value (1 = high-precision)
            x        : (B, HIDDEN_DIM)  original embedding (reconstruction target)
        """
        x  = self.embedding(token_ids)         # (B, HIDDEN_DIM)
        g  = self.gate_mlp(x).squeeze(-1)      # (B,)  in [0, 1]

        # High-precision path: store at full dimensionality
        hp_out = x                              # (B, HIDDEN_DIM)

        # Low-precision path: compress then expand (lossy)
        lp_out = self.lp_decoder(self.lp_encoder(x))   # (B, HIDDEN_DIM)

        # Soft gated combination
        out = g.unsqueeze(-1) * hp_out + (1.0 - g.unsqueeze(-1)) * lp_out
        return out, g, x


# ── Data helpers ──────────────────────────────────────────────────────────────

def sample_exact_ids(batch_size: int) -> torch.Tensor:
    return torch.randint(0, EXACT_TOKEN_MAX, (batch_size,))


def sample_approx_ids(batch_size: int) -> torch.Tensor:
    return torch.randint(APPROX_TOKEN_MIN, APPROX_TOKEN_MAX, (batch_size,))


def sample_mixed_ids(batch_size: int) -> torch.Tensor:
    """Half exact, half approx (randomly interleaved)."""
    exact  = sample_exact_ids(batch_size // 2)
    approx = sample_approx_ids(batch_size - batch_size // 2)
    ids    = torch.cat([exact, approx])
    perm   = torch.randperm(ids.size(0))
    return ids[perm]


# ── Training ──────────────────────────────────────────────────────────────────

def train(model: GatedMemory) -> None:
    opt = Adam(model.parameters(), lr=LR)

    for step in range(TRAIN_STEPS):
        token_ids = sample_mixed_ids(BATCH_SIZE)
        out, g, x = model(token_ids)

        # Per-token reconstruction error (B,)
        recon_err = F.mse_loss(out, x.detach(), reduction="none").mean(dim=-1)

        # Weight by token type (exact tokens cost more to reconstruct badly)
        is_exact = (token_ids < EXACT_TOKEN_MAX).float()
        weights  = is_exact * EXACT_LOSS_WEIGHT + (1.0 - is_exact) * APPROX_LOSS_WEIGHT
        weighted_recon = (recon_err * weights).mean()

        # Precision budget penalty: discourages always using high-precision
        precision_cost = PRECISION_PENALTY * g.mean()

        loss = weighted_recon + precision_cost

        opt.zero_grad()
        loss.backward()
        opt.step()

        if (step + 1) % 500 == 0:
            print(f"    step {step+1}/{TRAIN_STEPS}  "
                  f"loss={loss.item():.4f}  "
                  f"g_exact={g[is_exact.bool()].mean().item():.3f}  "
                  f"g_approx={g[~is_exact.bool()].mean().item():.3f}")


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(model: GatedMemory) -> dict[str, float]:
    model.eval()
    exact_errors:   list[float] = []
    approx_errors:  list[float] = []
    hp_rate_exact:  list[float] = []
    hp_rate_approx: list[float] = []

    with torch.no_grad():
        for _ in range(EVAL_BATCHES):
            # evaluate on exact tokens
            eids = sample_exact_ids(BATCH_SIZE)
            out, g, x = model(eids)
            err = F.mse_loss(out, x, reduction="none").mean(dim=-1)
            exact_errors.append(err.mean().item())
            hp_rate_exact.append(g.mean().item())

            # evaluate on approx tokens
            aids = sample_approx_ids(BATCH_SIZE)
            out, g, x = model(aids)
            err = F.mse_loss(out, x, reduction="none").mean(dim=-1)
            approx_errors.append(err.mean().item())
            hp_rate_approx.append(g.mean().item())

    avg_err_exact  = sum(exact_errors)  / len(exact_errors)
    avg_err_approx = sum(approx_errors) / len(approx_errors)
    hp_rate_e = sum(hp_rate_exact)  / len(hp_rate_exact)
    hp_rate_a = sum(hp_rate_approx) / len(hp_rate_approx)
    error_ratio = avg_err_approx / avg_err_exact if avg_err_exact > 1e-9 else float("inf")

    return {
        "avg_error_exact":               avg_err_exact,
        "avg_error_approx":              avg_err_approx,
        "error_ratio":                   error_ratio,
        "high_precision_rate_for_exact": hp_rate_e,
        "high_precision_rate_for_approx": hp_rate_a,
    }


# ── Experiment ────────────────────────────────────────────────────────────────

class Exp23LossyVsLossless(Experiment):
    experiment_id = "exp_2_3"
    hypothesis = (
        "A controller can learn without supervision which information should be stored "
        "exactly (numbers, names) vs approximately (context, themes)."
    )

    def run(self) -> ExperimentResult:
        torch.manual_seed(42)

        model = GatedMemory().to(DEVICE)
        print(f"\nTraining gated memory for {TRAIN_STEPS} steps...")
        train(model)

        print("\nEvaluating...")
        m = evaluate(model)

        print(f"  avg_error_exact={m['avg_error_exact']:.4f}  "
              f"avg_error_approx={m['avg_error_approx']:.4f}")
        print(f"  error_ratio={m['error_ratio']:.3f}  "
              f"hp_rate_exact={m['high_precision_rate_for_exact']:.3f}  "
              f"hp_rate_approx={m['high_precision_rate_for_approx']:.3f}")

        diff = m["avg_error_approx"] - m["avg_error_exact"]
        if m["avg_error_exact"] < m["avg_error_approx"]:
            if abs(diff) < INCONCLUSIVE_DELTA:
                outcome = OUTCOME_INCONCLUSIVE
            else:
                outcome = OUTCOME_SUPPORTED
        else:
            if abs(diff) < INCONCLUSIVE_DELTA:
                outcome = OUTCOME_INCONCLUSIVE
            else:
                outcome = OUTCOME_REFUTED

        notes = (
            f"avg_error_exact={m['avg_error_exact']:.4f}, "
            f"avg_error_approx={m['avg_error_approx']:.4f}, "
            f"diff (approx-exact)={diff:+.4f}. "
            f"HP rate: exact={m['high_precision_rate_for_exact']:.3f}, "
            f"approx={m['high_precision_rate_for_approx']:.3f}. "
            f"System learned to protect exact tokens: {m['avg_error_exact'] < m['avg_error_approx']}."
        )
        config = {
            "vocab_size": VOCAB_SIZE, "hidden_dim": HIDDEN_DIM, "low_dim": LOW_DIM,
            "exact_token_max": EXACT_TOKEN_MAX, "approx_token_max": APPROX_TOKEN_MAX,
            "exact_loss_weight": EXACT_LOSS_WEIGHT, "approx_loss_weight": APPROX_LOSS_WEIGHT,
            "precision_penalty": PRECISION_PENALTY, "train_steps": TRAIN_STEPS,
        }
        return self.result(outcome, m, notes, config)


if __name__ == "__main__":
    Exp23LossyVsLossless().execute()
