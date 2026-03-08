"""
Experiment 6.7 — Write-Evict Coupling

Hypothesis: Joint optimization of write and evict decisions outperforms treating
them as independent operations when storage pressure is constant.

Setup:
  - Constant storage pressure: memory always full (MEMORY_SLOTS=4, writing every step).
  - Policy A (independent): separate write gate and evict gate, trained with task loss only.
  - Policy B (joint): single network outputs (write_vector, evict_index) simultaneously.
  - Task: associative recall with high write pressure.
  - SUPPORTED if joint_acc > independent_acc + 0.02.
  - REFUTED if independent wins.
  - INCONCLUSIVE if |gap| < 0.02.
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
MEMORY_SLOTS = 4    # constant pressure: memory always full
EVAL_BATCHES = 200
LR           = 3e-4
DEVICE       = "cpu"


# ── Data generator ─────────────────────────────────────────────────────────────

def make_batch(batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    High write-pressure associative recall.
    Each step writes a new token. One key-value pair is critical and must survive.
    """
    seqs      = torch.zeros(batch_size, SEQ_LEN, dtype=torch.long)
    query_tok = torch.zeros(batch_size, dtype=torch.long)
    target    = torch.zeros(batch_size, dtype=torch.long)

    for b in range(batch_size):
        key   = torch.randint(0, 16, (1,)).item()
        value = torch.randint(16, VOCAB_SIZE, (1,)).item()
        pos   = torch.randint(0, SEQ_LEN // 4, (1,)).item()

        seqs[b, pos]     = key
        seqs[b, pos + 1] = value

        for i in range(SEQ_LEN):
            if i in (pos, pos + 1):
                continue
            seqs[b, i] = torch.randint(1, VOCAB_SIZE, (1,)).item()

        query_tok[b] = key
        target[b]    = value

    return seqs, query_tok, target


# ── Independent policy (separate write gate + evict gate) ─────────────────────

class IndependentPolicy(nn.Module):
    """
    Write gate: decides how much of new token to write (scalar gate on embedding).
    Evict gate: independently scores each slot for eviction (softmin selection).
    Each is trained independently via task loss only.
    """
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE + 2, HIDDEN_DIM)

        # Write gate: takes new token embedding → scalar in [0,1]
        self.write_gate = nn.Sequential(
            nn.Linear(HIDDEN_DIM, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        # Evict gate: takes (new_token_emb, slot_emb) → eviction score
        self.evict_gate = nn.Sequential(
            nn.Linear(HIDDEN_DIM * 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        self.read_head = nn.Sequential(
            nn.Linear(HIDDEN_DIM * 2, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, VOCAB_SIZE),
        )

        # Track write-evict correlation (Pearson r between write score and evict score)
        self._last_write_evict_corr: float = 0.0

    def forward(self, seqs: torch.Tensor, query_tok: torch.Tensor,
                track_corr: bool = False) -> torch.Tensor:
        B = seqs.size(0)
        out = []
        corr_data: list[tuple[float, float]] = []

        for b in range(B):
            # Initialise memory with zeros
            mem = torch.zeros(MEMORY_SLOTS, HIDDEN_DIM)

            for t in range(SEQ_LEN - 1):
                new_emb  = self.embed(seqs[b, t].unsqueeze(0)).squeeze(0)  # (H,)
                w_score  = self.write_gate(new_emb.unsqueeze(0)).squeeze()  # scalar

                # Evict: score each slot independently of write decision
                evict_scores = []
                for s in range(MEMORY_SLOTS):
                    inp = torch.cat([new_emb, mem[s]]).unsqueeze(0)
                    evict_scores.append(self.evict_gate(inp).squeeze())

                evict_logits = torch.stack(evict_scores)           # (slots,)
                evict_idx    = evict_logits.argmax().item()

                if track_corr:
                    corr_data.append((w_score.item(), evict_logits[evict_idx].item()))

                # Write gated embedding into evicted slot
                new_mem = mem.clone()
                new_mem[evict_idx] = new_emb * w_score
                mem = new_mem

            mem_summary = mem.mean(0)
            q_emb  = self.embed(query_tok[b].unsqueeze(0)).squeeze(0)
            logits = self.read_head(torch.cat([q_emb, mem_summary]).unsqueeze(0))
            out.append(logits)

        if track_corr and len(corr_data) > 1:
            ws = torch.tensor([x[0] for x in corr_data])
            es = torch.tensor([x[1] for x in corr_data])
            ws_c = ws - ws.mean(); es_c = es - es.mean()
            denom = (ws_c.norm() * es_c.norm()).item()
            self._last_write_evict_corr = (
                (ws_c * es_c).sum().item() / denom if denom > 1e-8 else 0.0
            )

        return torch.cat(out, dim=0)


# ── Joint policy (single network, shared gradient) ────────────────────────────

class JointPolicy(nn.Module):
    """
    Single network takes (new_token_emb, current_memory) and outputs:
      - write_vector: what to write (full HIDDEN_DIM vector)
      - evict_logits: which slot to evict (MEMORY_SLOTS logits)
    Trained end-to-end with shared gradients.
    """
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE + 2, HIDDEN_DIM)

        # Joint network input: new_emb + flattened memory
        joint_input_dim = HIDDEN_DIM + MEMORY_SLOTS * HIDDEN_DIM

        self.joint_net = nn.Sequential(
            nn.Linear(joint_input_dim, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),
        )

        self.write_head  = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)   # write vector
        self.evict_head  = nn.Linear(HIDDEN_DIM, MEMORY_SLOTS) # evict logits

        self.read_head = nn.Sequential(
            nn.Linear(HIDDEN_DIM * 2, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, VOCAB_SIZE),
        )

        self._last_write_evict_corr: float = 0.0

    def forward(self, seqs: torch.Tensor, query_tok: torch.Tensor,
                track_corr: bool = False) -> torch.Tensor:
        B = seqs.size(0)
        out = []
        corr_data: list[tuple[float, float]] = []

        for b in range(B):
            mem = torch.zeros(MEMORY_SLOTS, HIDDEN_DIM)

            for t in range(SEQ_LEN - 1):
                new_emb  = self.embed(seqs[b, t].unsqueeze(0)).squeeze(0)  # (H,)
                mem_flat = mem.flatten().unsqueeze(0)                        # (1, slots*H)
                joint_in = torch.cat([new_emb.unsqueeze(0), mem_flat], dim=-1)  # (1, input_dim)

                h = self.joint_net(joint_in)                     # (1, H)
                write_vec    = self.write_head(h).squeeze(0)     # (H,)
                evict_logits = self.evict_head(h).squeeze(0)     # (slots,)

                evict_idx = evict_logits.argmax().item()

                if track_corr:
                    w_norm = write_vec.norm().item()
                    e_val  = evict_logits[evict_idx].item()
                    corr_data.append((w_norm, e_val))

                new_mem = mem.clone()
                new_mem[evict_idx] = write_vec
                mem = new_mem

            mem_summary = mem.mean(0)
            q_emb  = self.embed(query_tok[b].unsqueeze(0)).squeeze(0)
            logits = self.read_head(torch.cat([q_emb, mem_summary]).unsqueeze(0))
            out.append(logits)

        if track_corr and len(corr_data) > 1:
            ws = torch.tensor([x[0] for x in corr_data])
            es = torch.tensor([x[1] for x in corr_data])
            ws_c = ws - ws.mean(); es_c = es - es.mean()
            denom = (ws_c.norm() * es_c.norm()).item()
            self._last_write_evict_corr = (
                (ws_c * es_c).sum().item() / denom if denom > 1e-8 else 0.0
            )

        return torch.cat(out, dim=0)


# ── Evaluation ─────────────────────────────────────────────────────────────────

def evaluate(model: nn.Module, n_batches: int) -> tuple[float, float]:
    """Returns (accuracy, mean_write_evict_correlation)."""
    model.eval()
    correct = total = 0
    corrs = []
    with torch.no_grad():
        for _ in range(n_batches):
            seqs, query_tok, target = make_batch(BATCH_SIZE)
            logits = model(seqs, query_tok, track_corr=True)
            correct += (logits.argmax(-1) == target).sum().item()
            total   += BATCH_SIZE
            if hasattr(model, '_last_write_evict_corr'):
                corrs.append(model._last_write_evict_corr)
    model.train()
    avg_corr = sum(corrs) / max(len(corrs), 1)
    return correct / total, avg_corr


# ── Experiment ──────────────────────────────────────────────────────────────────

class Exp67WriteEvictCoupling(Experiment):
    experiment_id = "exp_6_7"
    hypothesis = (
        "Joint optimization of write and evict decisions outperforms treating "
        "them as independent operations when storage pressure is constant."
    )

    def run(self) -> ExperimentResult:
        torch.manual_seed(42)

        config = {
            "hidden_dim": HIDDEN_DIM, "vocab_size": VOCAB_SIZE,
            "seq_len": SEQ_LEN, "batch_size": BATCH_SIZE,
            "train_steps": TRAIN_STEPS, "memory_slots": MEMORY_SLOTS,
        }

        # ── Train independent policy ──────────────────────────────────────────
        print("  Training independent policy...")
        indep = IndependentPolicy().to(DEVICE)
        opt_i = Adam(indep.parameters(), lr=LR)

        for step in range(TRAIN_STEPS):
            seqs, query_tok, target = make_batch(BATCH_SIZE)
            loss = F.cross_entropy(indep(seqs, query_tok), target)
            opt_i.zero_grad(); loss.backward(); opt_i.step()
            if (step + 1) % 500 == 0:
                print(f"    Indep step {step+1}/{TRAIN_STEPS}  loss={loss.item():.3f}")

        # ── Train joint policy ────────────────────────────────────────────────
        print("  Training joint policy...")
        joint = JointPolicy().to(DEVICE)
        opt_j = Adam(joint.parameters(), lr=LR)

        for step in range(TRAIN_STEPS):
            seqs, query_tok, target = make_batch(BATCH_SIZE)
            loss = F.cross_entropy(joint(seqs, query_tok), target)
            opt_j.zero_grad(); loss.backward(); opt_j.step()
            if (step + 1) % 500 == 0:
                print(f"    Joint step {step+1}/{TRAIN_STEPS}  loss={loss.item():.3f}")

        # ── Evaluate ──────────────────────────────────────────────────────────
        indep_acc, corr_indep = evaluate(indep, EVAL_BATCHES)
        joint_acc, corr_joint = evaluate(joint, EVAL_BATCHES)

        print(f"  Independent acc:  {indep_acc:.3f}  write-evict corr={corr_indep:.3f}")
        print(f"  Joint acc:        {joint_acc:.3f}  write-evict corr={corr_joint:.3f}")

        gap = joint_acc - indep_acc
        if abs(gap) < 0.02:
            outcome = OUTCOME_INCONCLUSIVE
        elif joint_acc > indep_acc + 0.02:
            outcome = OUTCOME_SUPPORTED
        else:
            outcome = OUTCOME_REFUTED

        return self.result(outcome, {
            "independent_acc":                  round(indep_acc,   4),
            "joint_acc":                        round(joint_acc,   4),
            "gap_joint_minus_independent":      round(gap,         4),
            "write_evict_correlation_independent": round(corr_indep, 4),
            "write_evict_correlation_joint":    round(corr_joint,  4),
        }, notes=(
            f"Joint vs independent gap: {gap:.3f}. "
            f"Write-evict correlation — independent: {corr_indep:.3f}, joint: {corr_joint:.3f}."
        ), config=config)


if __name__ == "__main__":
    Exp67WriteEvictCoupling().execute()
