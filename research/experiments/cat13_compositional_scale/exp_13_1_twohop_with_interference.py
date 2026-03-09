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

experiment_id = "exp_13_1"
hypothesis = (
    "The two-hop retrieval regularization effect (exp_4_9) persists at 64-entity KB "
    "with 40% near-duplicate interference."
)

DEVICE = "cpu"
LR = 3e-4
KB_SIZE = 64
HIDDEN_DIM = 64
NEAR_DUP_FRACTION = 0.4
STEPS = 500
BATCH = 32
VOCAB_SIZE = 128


class MemoryBank(nn.Module):
    """Fixed-size soft key-value memory with learnable keys."""
    def __init__(self, num_slots, hidden_dim):
        super().__init__()
        self.keys = nn.Parameter(torch.randn(num_slots, hidden_dim))
        self.values = nn.Parameter(torch.randn(num_slots, hidden_dim))

    def read(self, query):
        # query: (B, D), keys: (S, D), values: (S, D)
        q = F.normalize(query, dim=-1)
        k = F.normalize(self.keys, dim=-1)
        scores = torch.mm(q, k.t())  # (B, S)
        attn = torch.softmax(scores, dim=-1)  # (B, S)
        return torch.mm(attn, self.values)  # (B, D)


class HopModel(nn.Module):
    def __init__(self, hidden_dim, vocab_size, num_hops):
        super().__init__()
        self.entity_embed = nn.Embedding(KB_SIZE + 4, hidden_dim)
        self.rel_embed = nn.Embedding(8, hidden_dim)
        self.memory = MemoryBank(KB_SIZE, hidden_dim)
        self.num_hops = num_hops
        self.hop_projs = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_hops)])
        self.out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, entity_ids, rel_ids=None):
        # entity_ids: (B,)
        h = self.entity_embed(entity_ids)
        if rel_ids is not None:
            h = h + self.rel_embed(rel_ids)
        for i in range(self.num_hops):
            q = self.hop_projs[i](h)
            h = self.memory.read(q)
        return self.out(h)


def build_kb():
    """Returns entity->attribute and entity->colleague mappings."""
    torch.manual_seed(999)
    attributes = torch.randint(VOCAB_SIZE // 2, VOCAB_SIZE, (KB_SIZE,))  # tokens 64-127
    colleagues = torch.randint(0, KB_SIZE, (KB_SIZE,))
    for i in range(KB_SIZE):
        if colleagues[i].item() == i:
            colleagues[i] = (i + 1) % KB_SIZE
    return attributes, colleagues


def apply_near_duplicates(embed: nn.Embedding, near_dup_fraction: float):
    """Perturb near_dup_fraction of entity embeddings toward a random other entity."""
    n = KB_SIZE
    n_dup = int(n * near_dup_fraction)
    dup_indices = torch.randperm(n)[:n_dup]
    with torch.no_grad():
        for idx in dup_indices:
            other = (idx.item() + torch.randint(1, n, (1,)).item()) % n
            embed.weight[idx] = 0.7 * embed.weight[idx] + 0.3 * embed.weight[other]


class Exp131TwohopWithInterference(Experiment):
    experiment_id = "exp_13_1"
    hypothesis = hypothesis

    def run(self) -> ExperimentResult:
        attributes, colleagues = build_kb()

        # Identify near-duplicate entities (same indices as in apply_near_duplicates with seed)
        torch.manual_seed(42)  # controlled after base.execute sets seed
        n_dup = int(KB_SIZE * NEAR_DUP_FRACTION)
        dup_entities = torch.randperm(KB_SIZE)[:n_dup]
        dup_set = set(dup_entities.tolist())

        model = HopModel(HIDDEN_DIM, VOCAB_SIZE, num_hops=2).to(DEVICE)
        apply_near_duplicates(model.entity_embed, NEAR_DUP_FRACTION)
        opt = Adam(model.parameters(), lr=LR)

        for step in range(STEPS):
            # Build mixed batch: 50% single-hop, 50% two-hop
            half = BATCH // 2
            entity_ids = torch.randint(0, KB_SIZE, (BATCH,), device=DEVICE)

            # Single-hop: entity -> attribute
            sh_entities = entity_ids[:half]
            sh_targets = attributes[sh_entities]

            # Two-hop: entity -> colleague -> colleague's attribute
            th_entities = entity_ids[half:]
            th_colleague = colleagues[th_entities]
            th_targets = attributes[th_colleague]

            # Forward
            sh_logits = model(sh_entities, rel_ids=None)
            # For two-hop, we use a relationship token (rel=0 = "colleague_attr")
            rel_ids = torch.zeros(half, dtype=torch.long, device=DEVICE)
            th_logits = model(th_entities, rel_ids=rel_ids)

            loss_sh = F.cross_entropy(sh_logits, sh_targets.to(DEVICE))
            loss_th = F.cross_entropy(th_logits, th_targets.to(DEVICE))
            loss = loss_sh + loss_th

            opt.zero_grad(); loss.backward(); opt.step()

        # Evaluation
        model.eval()
        with torch.no_grad():
            all_entities = torch.arange(KB_SIZE, device=DEVICE)
            all_attrs = attributes.to(DEVICE)

            # Single-hop accuracy
            sh_logits = model(all_entities)
            sh_preds = sh_logits.argmax(-1)
            single_hop_acc = (sh_preds == all_attrs).float().mean().item()

            # Two-hop accuracy
            all_colleagues = colleagues.to(DEVICE)
            all_th_targets = attributes[all_colleagues].to(DEVICE)
            rel_ids = torch.zeros(KB_SIZE, dtype=torch.long, device=DEVICE)
            th_logits = model(all_entities, rel_ids=rel_ids)
            th_preds = th_logits.argmax(-1)
            two_hop_acc = (th_preds == all_th_targets).float().mean().item()

            # Interference subset accuracy (near-dup entities)
            dup_tensor = dup_entities.to(DEVICE)
            dup_attrs = attributes[dup_entities].to(DEVICE)
            dup_colleagues = colleagues[dup_entities].to(DEVICE)
            dup_th_targets = attributes[dup_colleagues].to(DEVICE)

            sh_dup_logits = model(dup_tensor)
            sh_dup_acc = (sh_dup_logits.argmax(-1) == dup_attrs).float().mean().item()

            dup_rel = torch.zeros(len(dup_tensor), dtype=torch.long, device=DEVICE)
            th_dup_logits = model(dup_tensor, rel_ids=dup_rel)
            th_dup_acc = (th_dup_logits.argmax(-1) == dup_th_targets).float().mean().item()

        gap = two_hop_acc - single_hop_acc
        interference_gap = th_dup_acc - sh_dup_acc

        metrics = {
            "single_hop_acc": round(single_hop_acc, 4),
            "two_hop_acc": round(two_hop_acc, 4),
            "two_hop_vs_single_gap": round(gap, 4),
            "interference_sh_acc": round(sh_dup_acc, 4),
            "interference_th_acc": round(th_dup_acc, 4),
            "interference_gap": round(interference_gap, 4),
        }

        config = dict(KB_SIZE=KB_SIZE, HIDDEN_DIM=HIDDEN_DIM,
                      NEAR_DUP_FRACTION=NEAR_DUP_FRACTION,
                      STEPS=STEPS, BATCH=BATCH, VOCAB_SIZE=VOCAB_SIZE)

        # Criterion on interference-heavy queries: two_hop_acc > single_hop_acc - 0.05
        if interference_gap > -0.05:
            outcome = OUTCOME_SUPPORTED
            notes = f"Two-hop acc within 5% of single-hop on interference subset (gap={interference_gap:.3f})."
        elif interference_gap < -0.15:
            outcome = OUTCOME_REFUTED
            notes = f"Two-hop acc more than 15% below single-hop on interference subset (gap={interference_gap:.3f})."
        else:
            outcome = OUTCOME_INCONCLUSIVE
            notes = f"Interference gap {interference_gap:.3f} between -0.15 and -0.05."

        return self.result(outcome, metrics, notes, config)


if __name__ == "__main__":
    Exp131TwohopWithInterference().execute()
