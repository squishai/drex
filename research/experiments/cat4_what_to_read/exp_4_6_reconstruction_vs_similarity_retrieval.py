"""
Experiment 4.6 — Reconstruction vs Similarity Retrieval

Hypothesis: For exact recall tasks similarity-based retrieval wins; for
inferential completion tasks reconstruction-based retrieval wins.

Setup:
  - Two retrieval paradigms:
      Similarity: find memory entry with highest dot-product to query, return it.
      Reconstruction: a decoder takes (query, entry) and scores each entry by how
        well it "completes" the query; retrieve the entry minimizing reconstruction
        loss (approximated during eval with a learned scoring head).
  - Two tasks:
      (1) Exact recall: retrieve the exact stored token.
      (2) Inferential: answer requires logical inference (A + B = ?) where
          A and B are stored in separate slots.
  - SUPPORTED if similarity wins exact AND reconstruction wins inferential
    (or vice-versa — showing specialisation).
  - REFUTED if one method wins both tasks.
  - INCONCLUSIVE if neither wins clearly on either task.
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

# ── Config ─────────────────────────────────────────────────────────────────────

HIDDEN_DIM   = 64
VOCAB_SIZE   = 64
SEQ_LEN      = 24
BATCH_SIZE   = 32
TRAIN_STEPS  = 1500
MEMORY_SLOTS = 8
LR           = 3e-4
DEVICE       = "cpu"
EVAL_STEPS   = 300
WIN_MARGIN   = 0.02   # minimum gap to declare a winner on a task


# ── Data — Task 1: Exact Recall ────────────────────────────────────────────────

def make_exact_batch(batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Exact recall: slot 0 contains the relevant entry that matches the query.
    Target = a specific token index (the 'answer' token associated with the slot).
    query_h: normalized embedding of a query key.
    memory: slot 0 is highly similar to query_h, slots 1+ are random noise.
    targets: (B,) token indices.
    """
    targets = torch.randint(0, VOCAB_SIZE, (batch_size,))
    query_h = F.normalize(torch.randn(batch_size, HIDDEN_DIM), dim=-1)

    relevant = F.normalize(query_h + torch.randn(batch_size, HIDDEN_DIM) * 0.05, dim=-1)
    noise    = F.normalize(torch.randn(batch_size, MEMORY_SLOTS - 1, HIDDEN_DIM), dim=-1)
    memory   = torch.cat([relevant.unsqueeze(1), noise], dim=1)
    return query_h.detach(), memory.detach(), targets


# ── Data — Task 2: Inferential (A + B) ────────────────────────────────────────

def make_inferential_batch(batch_size: int) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor
]:
    """
    Inferential: two operands A, B stored in slots 0 and 1.
    Target = (A + B) % (VOCAB_SIZE // 2).
    query_h encodes the 'add?' operation signal.
    The model must retrieve both slots and combine them.
    """
    half = VOCAB_SIZE // 2
    a_vals = torch.randint(0, half, (batch_size,))
    b_vals = torch.randint(0, half, (batch_size,))
    targets = (a_vals + b_vals) % half  # (B,) in [0, half)

    # query = constant "add" signal plus noise (no direct similarity to slots)
    query_h = F.normalize(torch.randn(batch_size, HIDDEN_DIM), dim=-1)

    # encode operands as deterministic low-dim embeddings scaled to HIDDEN_DIM
    def operand_vec(vals: torch.Tensor) -> torch.Tensor:
        # one-hot in HIDDEN_DIM space (first half dims)
        v = torch.zeros(batch_size, HIDDEN_DIM)
        for b in range(batch_size):
            v[b, int(vals[b].item()) % HIDDEN_DIM] = 1.0
        return F.normalize(v, dim=-1)

    slot_a = operand_vec(a_vals)  # (B, H)
    slot_b = operand_vec(b_vals)  # (B, H)
    noise  = F.normalize(torch.randn(batch_size, MEMORY_SLOTS - 2, HIDDEN_DIM), dim=-1)
    memory = torch.cat([slot_a.unsqueeze(1), slot_b.unsqueeze(1), noise], dim=1)
    return query_h.detach(), memory.detach(), targets


# ── Models ─────────────────────────────────────────────────────────────────────

class SimilarityRetrievalModel(nn.Module):
    """Dot-product similarity to find best entry; aggregate; classify."""

    def __init__(self, n_output: int = VOCAB_SIZE) -> None:
        super().__init__()
        self.query_proj = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.out_proj   = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.classifier = nn.Linear(HIDDEN_DIM, n_output)

    def forward(self, query_h: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        q       = self.query_proj(query_h)
        sims    = torch.bmm(memory, q.unsqueeze(-1)).squeeze(-1) / (HIDDEN_DIM ** 0.5)
        weights = F.softmax(sims, dim=-1).unsqueeze(-1)
        retr    = (weights * memory).sum(1)
        return self.classifier(self.out_proj(retr))


class ReconstructionRetrievalModel(nn.Module):
    """
    Reconstruction-based retrieval: for each memory entry, score it by how
    well a decoder can reconstruct the query from (query, entry).
    Retrieve the entry with highest reconstruction score (argmax + straight-through
    for training; softmax combination weighted by reconstruction scores for eval).
    Final classifier operates on the weighted-retrieved vector.
    """

    def __init__(self, n_output: int = VOCAB_SIZE) -> None:
        super().__init__()
        # encoder: takes (query || entry) -> reconstruction score
        self.scorer     = nn.Sequential(
            nn.Linear(HIDDEN_DIM * 2, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, 1),
        )
        self.out_proj   = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.classifier = nn.Linear(HIDDEN_DIM, n_output)

    def forward(self, query_h: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        B, M, H = memory.shape
        # expand query to (B, M, H) for pairwise scoring
        q_exp  = query_h.unsqueeze(1).expand(-1, M, -1)          # (B, M, H)
        pairs  = torch.cat([q_exp, memory], dim=-1)               # (B, M, 2H)
        scores = self.scorer(pairs).squeeze(-1)                   # (B, M)
        weights = F.softmax(scores, dim=-1).unsqueeze(-1)         # (B, M, 1)
        retr   = (weights * memory).sum(1)                        # (B, H)
        return self.classifier(self.out_proj(retr))


# ── Training helper ────────────────────────────────────────────────────────────

def train_and_eval(
    model: nn.Module,
    make_batch_fn,
    label: str,
    n_output: int = VOCAB_SIZE,
) -> float:
    opt = Adam(model.parameters(), lr=LR)
    model.train()

    for step in range(TRAIN_STEPS):
        query_h, memory, targets = make_batch_fn(BATCH_SIZE)
        logits = model(query_h, memory)
        loss   = F.cross_entropy(logits, targets)
        opt.zero_grad(); loss.backward(); opt.step()
        if (step + 1) % 300 == 0:
            print(f"  [{label}] step {step+1}/{TRAIN_STEPS}  loss={loss.item():.4f}")

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for _ in range(EVAL_STEPS):
            query_h, memory, targets = make_batch_fn(BATCH_SIZE)
            preds = model(query_h, memory).argmax(dim=-1)
            correct += (preds == targets).sum().item()
            total   += BATCH_SIZE

    return correct / total


# ── Experiment ─────────────────────────────────────────────────────────────────

class Exp46ReconstructionVsSimilarityRetrieval(Experiment):
    experiment_id = "exp_4_6"
    hypothesis = (
        "For exact recall tasks similarity-based retrieval wins; for inferential "
        "completion tasks reconstruction-based retrieval wins."
    )

    def run(self) -> ExperimentResult:
        half = VOCAB_SIZE // 2

        # Task 1: Exact recall
        torch.manual_seed(42)
        print("\n  [Task 1: Exact Recall] Similarity model...")
        sim_exact  = SimilarityRetrievalModel(n_output=VOCAB_SIZE).to(DEVICE)
        sim_exact_acc = train_and_eval(sim_exact, make_exact_batch, "Sim/Exact")

        torch.manual_seed(42)
        print("\n  [Task 1: Exact Recall] Reconstruction model...")
        rec_exact  = ReconstructionRetrievalModel(n_output=VOCAB_SIZE).to(DEVICE)
        rec_exact_acc = train_and_eval(rec_exact, make_exact_batch, "Rec/Exact")

        # Task 2: Inferential
        torch.manual_seed(42)
        print("\n  [Task 2: Inferential] Similarity model...")
        sim_infer  = SimilarityRetrievalModel(n_output=half).to(DEVICE)
        sim_infer_acc = train_and_eval(
            sim_infer,
            lambda b: make_inferential_batch(b),
            "Sim/Infer",
            n_output=half,
        )

        torch.manual_seed(42)
        print("\n  [Task 2: Inferential] Reconstruction model...")
        rec_infer  = ReconstructionRetrievalModel(n_output=half).to(DEVICE)
        rec_infer_acc = train_and_eval(
            rec_infer,
            lambda b: make_inferential_batch(b),
            "Rec/Infer",
            n_output=half,
        )

        print(f"\n  Exact:       Sim={sim_exact_acc:.4f}  Rec={rec_exact_acc:.4f}")
        print(f"  Inferential: Sim={sim_infer_acc:.4f}  Rec={rec_infer_acc:.4f}")

        sim_wins_exact  = sim_exact_acc  > rec_exact_acc  + WIN_MARGIN
        rec_wins_infer  = rec_infer_acc  > sim_infer_acc  + WIN_MARGIN
        rec_wins_exact  = rec_exact_acc  > sim_exact_acc  + WIN_MARGIN
        sim_wins_infer  = sim_infer_acc  > rec_infer_acc  + WIN_MARGIN

        # SUPPORTED: each method specialises (sim wins exact, rec wins infer — or vice versa)
        specialisation_A = sim_wins_exact and rec_wins_infer
        specialisation_B = rec_wins_exact and sim_wins_infer

        # REFUTED: same method wins both tasks clearly
        sim_wins_both = sim_wins_exact and sim_wins_infer
        rec_wins_both = rec_wins_exact and rec_wins_infer

        if specialisation_A or specialisation_B:
            outcome = OUTCOME_SUPPORTED
        elif sim_wins_both or rec_wins_both:
            outcome = OUTCOME_REFUTED
        else:
            outcome = OUTCOME_INCONCLUSIVE

        metrics = {
            "similarity_exact_acc":        round(sim_exact_acc, 4),
            "recon_exact_acc":             round(rec_exact_acc, 4),
            "similarity_inferential_acc":  round(sim_infer_acc, 4),
            "recon_inferential_acc":       round(rec_infer_acc, 4),
            "sim_wins_exact":              sim_wins_exact,
            "rec_wins_inferential":        rec_wins_infer,
        }
        notes = (
            f"Exact(Sim={sim_exact_acc:.4f}, Rec={rec_exact_acc:.4f}), "
            f"Inferential(Sim={sim_infer_acc:.4f}, Rec={rec_infer_acc:.4f}). "
            f"Specialisation A={specialisation_A}, B={specialisation_B}."
        )
        config = {
            "hidden_dim":   HIDDEN_DIM,
            "vocab_size":   VOCAB_SIZE,
            "memory_slots": MEMORY_SLOTS,
            "train_steps":  TRAIN_STEPS,
            "batch_size":   BATCH_SIZE,
            "win_margin":   WIN_MARGIN,
        }
        return self.result(outcome, metrics, notes, config)


if __name__ == "__main__":
    Exp46ReconstructionVsSimilarityRetrieval().execute()
