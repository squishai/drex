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

experiment_id = "exp_13_2"
hypothesis = (
    "Three-hop compositional retrieval retains >50% of two-hop accuracy "
    "(degradation_ratio > 0.50) at hidden_dim=64."
)

DEVICE = "cpu"
LR = 3e-4
KB_SIZE = 32
HIDDEN_DIM = 64
VOCAB_SIZE = 128
STEPS = 500
BATCH = 32


def build_chains():
    """Build A->B, B->C, C->D mappings for KB_SIZE chains."""
    # A, B, C, D are entity classes; each of size KB_SIZE
    # A: indices 0..31, B: 32..63, C: 64..95, D (output tokens): 96..127
    A = torch.arange(0, KB_SIZE)
    B = torch.randperm(KB_SIZE) + KB_SIZE        # 32-63
    C = torch.randperm(KB_SIZE) + 2 * KB_SIZE    # 64-95
    D = torch.randint(3 * KB_SIZE, VOCAB_SIZE, (KB_SIZE,))  # 96-127
    return A, B, C, D


class MultiHopMemory(nn.Module):
    """Soft memory that supports multiple hops."""
    def __init__(self, hidden_dim, num_slots, max_hops):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, hidden_dim)
        # Separate memory banks for each hop level
        self.mem_keys = nn.ParameterList([
            nn.Parameter(torch.randn(num_slots, hidden_dim)) for _ in range(max_hops)
        ])
        self.mem_vals = nn.ParameterList([
            nn.Parameter(torch.randn(num_slots, hidden_dim)) for _ in range(max_hops)
        ])
        self.hop_projs = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(max_hops)
        ])
        self.out = nn.Linear(hidden_dim, VOCAB_SIZE)
        self.max_hops = max_hops

    def read_hop(self, query, hop_idx):
        k = F.normalize(self.mem_keys[hop_idx], dim=-1)
        q = F.normalize(self.hop_projs[hop_idx](query), dim=-1)
        scores = torch.mm(q, k.t())
        attn = torch.softmax(scores, dim=-1)
        return torch.mm(attn, self.mem_vals[hop_idx])

    def forward(self, start_ids, num_hops):
        h = self.embed(start_ids)
        for i in range(num_hops):
            h = self.read_hop(h, i)
        return self.out(h)


class Exp132ThreehopChain(Experiment):
    experiment_id = "exp_13_2"
    hypothesis = hypothesis

    def run(self) -> ExperimentResult:
        A, B, C, D = build_chains()
        # Build lookup tables
        # A[i] -> B[i] -> C[i] -> D[i]
        a_to_b = {A[i].item(): B[i].item() for i in range(KB_SIZE)}
        b_to_c = {B[i].item(): C[i].item() for i in range(KB_SIZE)}
        c_to_d = {C[i].item(): D[i].item() for i in range(KB_SIZE)}

        # Single-hop target: A -> B
        # Two-hop target: A -> B -> C
        # Three-hop target: A -> B -> C -> D
        a_to_c = {a: b_to_c[b] for a, b in a_to_b.items()}
        a_to_d = {a: c_to_d[c] for a, c in a_to_c.items()}

        model = MultiHopMemory(HIDDEN_DIM, num_slots=KB_SIZE * 3, max_hops=3).to(DEVICE)
        opt = Adam(model.parameters(), lr=LR)

        a_tensor = A.to(DEVICE)
        b_targets = B.to(DEVICE)
        c_targets = C.to(DEVICE)
        d_targets = D.to(DEVICE)

        for step in range(STEPS):
            idx = torch.randint(0, KB_SIZE, (BATCH,))
            task = torch.randint(0, 3, (1,)).item()  # 0=1hop, 1=2hop, 2=3hop

            a_ids = A[idx].to(DEVICE)

            if task == 0:
                targets = B[idx].to(DEVICE)
                logits = model(a_ids, num_hops=1)
            elif task == 1:
                targets = C[idx].to(DEVICE)
                logits = model(a_ids, num_hops=2)
            else:
                targets = D[idx].to(DEVICE)
                logits = model(a_ids, num_hops=3)

            loss = F.cross_entropy(logits, targets)
            opt.zero_grad(); loss.backward(); opt.step()

        # Evaluation
        model.eval()
        with torch.no_grad():
            # Single-hop: A -> B
            logits_1 = model(a_tensor, num_hops=1)
            acc_single = (logits_1.argmax(-1) == b_targets).float().mean().item()

            # Two-hop: A -> C
            logits_2 = model(a_tensor, num_hops=2)
            acc_two = (logits_2.argmax(-1) == c_targets).float().mean().item()

            # Three-hop: A -> D
            logits_3 = model(a_tensor, num_hops=3)
            acc_three = (logits_3.argmax(-1) == d_targets).float().mean().item()

        degradation_ratio = acc_three / max(acc_two, 0.001)

        metrics = {
            "acc_single": round(acc_single, 4),
            "acc_two": round(acc_two, 4),
            "acc_three": round(acc_three, 4),
            "degradation_ratio": round(degradation_ratio, 4),
        }

        config = dict(KB_SIZE=KB_SIZE, HIDDEN_DIM=HIDDEN_DIM,
                      VOCAB_SIZE=VOCAB_SIZE, STEPS=STEPS, BATCH=BATCH)

        if degradation_ratio > 0.50:
            outcome = OUTCOME_SUPPORTED
            notes = f"Three-hop retains {degradation_ratio:.2f} of two-hop accuracy (>0.50)."
        elif degradation_ratio < 0.30:
            outcome = OUTCOME_REFUTED
            notes = f"Three-hop retains only {degradation_ratio:.2f} of two-hop accuracy (<0.30)."
        else:
            outcome = OUTCOME_INCONCLUSIVE
            notes = f"Degradation ratio {degradation_ratio:.2f} between 0.30 and 0.50."

        return self.result(outcome, metrics, notes, config)


if __name__ == "__main__":
    Exp132ThreehopChain().execute()
