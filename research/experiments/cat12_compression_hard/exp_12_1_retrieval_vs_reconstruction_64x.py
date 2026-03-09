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

experiment_id = "exp_12_1"
hypothesis = (
    "At 64x compression with 100-way gallery discrimination, retrieval-objective "
    "compressor achieves >=15% higher Acc@1 than reconstruction-objective compressor."
)

DEVICE = "cpu"
LR = 3e-4
EMBED_DIM = 32
N_TOKENS = 16
FLAT_DIM = EMBED_DIM * N_TOKENS  # 512
BOTTLENECK = 8  # 64x compression
TRAIN_STEPS = 3000
BATCH_SIZE = 64
GALLERY_SIZE = 100
TEMPERATURE = 0.1


class Compressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(FLAT_DIM, 256), nn.ReLU(), nn.Linear(256, BOTTLENECK)
        )
        self.decoder = nn.Sequential(
            nn.Linear(BOTTLENECK, 256), nn.ReLU(), nn.Linear(256, FLAT_DIM)
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)


def make_batch(batch_size):
    return torch.randn(batch_size, FLAT_DIM)


def infonce_loss(q, k_pos, k_all, temperature):
    # q: (B, D), k_pos: (B, D), k_all: (B, D) — all compressed
    q = F.normalize(q, dim=-1)
    k_pos = F.normalize(k_pos, dim=-1)
    k_all = F.normalize(k_all, dim=-1)
    # sim(q_i, k_all_j) => (B, B)
    sim = torch.mm(q, k_all.t()) / temperature
    labels = torch.arange(batch_size := q.size(0), device=q.device)
    return F.cross_entropy(sim, labels)


class Exp121RetrievalVsReconstruction64x(Experiment):
    experiment_id = "exp_12_1"
    hypothesis = hypothesis

    def run(self) -> ExperimentResult:
        # --- Condition A: MSE autoencoder ---
        model_A = Compressor().to(DEVICE)
        opt_A = Adam(model_A.parameters(), lr=LR)

        for step in range(TRAIN_STEPS):
            x = make_batch(BATCH_SIZE).to(DEVICE)
            z = model_A.encode(x)
            x_hat = model_A.decode(z)
            loss = F.mse_loss(x_hat, x)
            opt_A.zero_grad(); loss.backward(); opt_A.step()

        # --- Condition B: InfoNCE contrastive ---
        model_B = Compressor().to(DEVICE)
        opt_B = Adam(model_B.parameters(), lr=LR)

        for step in range(TRAIN_STEPS):
            x = make_batch(BATCH_SIZE).to(DEVICE)
            x_pos = x + 0.01 * torch.randn_like(x)
            q = model_B.encode(x)
            k_pos = model_B.encode(x_pos)
            # k_all = k_pos for in-batch negatives
            loss = infonce_loss(q, k_pos, k_pos, TEMPERATURE)
            opt_B.zero_grad(); loss.backward(); opt_B.step()

        # --- Evaluation ---
        model_A.eval(); model_B.eval()
        with torch.no_grad():
            gallery = make_batch(GALLERY_SIZE).to(DEVICE)  # (100, 512)

            # Compress gallery for each model
            gallery_z_A = model_A.encode(gallery)   # (100, 8)
            gallery_z_B = model_B.encode(gallery)   # (100, 8)

            # Reconstruct gallery for model A (for recon_cosim)
            gallery_recon_A = model_A.decode(gallery_z_A)  # (100, 512)
            gallery_recon_B = model_A.decode(model_A.encode(gallery))  # reuse A decoder

            # Acc@1: noisy queries
            correct_A = 0; correct_B = 0
            for i in range(GALLERY_SIZE):
                query = gallery[i:i+1] + 0.05 * torch.randn(1, FLAT_DIM, device=DEVICE)
                qz_A = model_A.encode(query)
                qz_B = model_B.encode(query)

                sim_A = F.cosine_similarity(qz_A.expand(GALLERY_SIZE, -1), gallery_z_A)
                sim_B = F.cosine_similarity(qz_B.expand(GALLERY_SIZE, -1), gallery_z_B)

                if sim_A.argmax().item() == i:
                    correct_A += 1
                if sim_B.argmax().item() == i:
                    correct_B += 1

            acc1_A = correct_A / GALLERY_SIZE
            acc1_B = correct_B / GALLERY_SIZE
            retrieval_gap = acc1_B - acc1_A

            # recon cosim for A
            recon_cosim_A = F.cosine_similarity(gallery_recon_A, gallery).mean().item()

            # For B: use B's encoder and A's decoder as a proxy for recon quality
            gallery_z_B2 = model_B.encode(gallery)
            # B has no decoder; estimate by projecting B codes back — not directly comparable
            # Instead compute cosim in compressed space: model_B has no explicit recon cosim
            # Per spec: recon_cosim_B = mean cosine_sim(decoded_B, original) — B has no decoder
            # Use model_A decoder on model_B codes as a proxy
            gallery_recon_B2 = model_A.decode(gallery_z_B2)
            recon_cosim_B = F.cosine_similarity(gallery_recon_B2, gallery).mean().item()

        metrics = {
            "acc1_A": round(acc1_A, 4),
            "acc1_B": round(acc1_B, 4),
            "retrieval_gap": round(retrieval_gap, 4),
            "recon_cosim_A": round(recon_cosim_A, 4),
            "recon_cosim_B": round(recon_cosim_B, 4),
        }

        config = dict(EMBED_DIM=EMBED_DIM, N_TOKENS=N_TOKENS, FLAT_DIM=FLAT_DIM,
                      BOTTLENECK=BOTTLENECK, TRAIN_STEPS=TRAIN_STEPS,
                      BATCH_SIZE=BATCH_SIZE, GALLERY_SIZE=GALLERY_SIZE)

        if retrieval_gap > 0.15 and recon_cosim_A > recon_cosim_B + 0.05:
            outcome = OUTCOME_SUPPORTED
            notes = "Retrieval gap >15% and A recon cosim dominates B."
        elif (acc1_A > 0.80 and acc1_B > 0.80) or retrieval_gap < 0.05:
            outcome = OUTCOME_REFUTED
            notes = "Both models >80% Acc@1 or gap <5%."
        else:
            outcome = OUTCOME_INCONCLUSIVE
            notes = f"Gap {retrieval_gap:.3f} between 0.05 and 0.15."

        return self.result(outcome, metrics, notes, config)


if __name__ == "__main__":
    Exp121RetrievalVsReconstruction64x().execute()
