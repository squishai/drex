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

experiment_id = "exp_12_2"
hypothesis = (
    "The 2x-8x compression training failure is gradient starvation from wide bottleneck, "
    "not structural — LR warmup restores training."
)

DEVICE = "cpu"
LR = 3e-4
INPUT_DIM = 64
TRAIN_STEPS = 300
BATCH_SIZE = 64
COMPRESSION_RATIOS = [2, 4, 8, 16]
BOTTLENECKS = [32, 16, 8, 4]
MEASURE_STEPS = [100, 200, 300]


def make_autoencoder(bottleneck: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(INPUT_DIM, 128), nn.ReLU(), nn.Linear(128, bottleneck),
        nn.Linear(bottleneck, 128), nn.ReLU(), nn.Linear(128, INPUT_DIM)
    )


def get_encoder_params(model: nn.Sequential):
    # First 3 modules are encoder (Linear, ReLU, Linear)
    return list(model[0].parameters()) + list(model[2].parameters())


def train_one_run(bottleneck: int, schedule: str) -> dict:
    model = make_autoencoder(bottleneck).to(DEVICE)

    if schedule == "C":
        opt = Adam(model.parameters(), lr=1e-5)
    else:
        opt = Adam(model.parameters(), lr=LR)

    cosim_log = {}
    grad_norm_log = {}

    for step in range(1, TRAIN_STEPS + 1):
        # LR schedule
        if schedule == "B":
            if step <= 500:
                new_lr = LR * step / 500
                for pg in opt.param_groups:
                    pg["lr"] = new_lr
            else:
                for pg in opt.param_groups:
                    pg["lr"] = LR

        x = torch.randn(BATCH_SIZE, INPUT_DIM, device=DEVICE)
        x_hat = model(x)
        loss = F.mse_loss(x_hat, x)
        opt.zero_grad()
        loss.backward()

        # Measure grad norm of encoder params
        enc_params = get_encoder_params(model)
        total_norm = 0.0
        for p in enc_params:
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5

        opt.step()

        if step in MEASURE_STEPS:
            model.eval()
            with torch.no_grad():
                x_eval = torch.randn(256, INPUT_DIM, device=DEVICE)
                x_hat_eval = model(x_eval)
                cosim = F.cosine_similarity(x_hat_eval, x_eval).mean().item()
            model.train()
            cosim_log[step] = round(cosim, 4)
            grad_norm_log[step] = round(total_norm, 6)

    return {"cosim": cosim_log, "grad_norm": grad_norm_log}


class Exp122CompressionGradientStarvation(Experiment):
    experiment_id = "exp_12_2"
    hypothesis = hypothesis

    def run(self) -> ExperimentResult:
        results = {}
        for ratio, bottleneck in zip(COMPRESSION_RATIOS, BOTTLENECKS):
            results[ratio] = {}
            for schedule in ["A", "B", "C"]:
                key = f"ratio{ratio}_sched{schedule}"
                print(f"  Training {key} (bottleneck={bottleneck})...")
                r = train_one_run(bottleneck, schedule)
                results[ratio][schedule] = r

        # Determine outcome: does warmup (B) fix failures where A fails at 2x and 4x?
        warmup_helps_count = 0
        a_fails_count = 0
        b_still_bad_count = 0

        flat_metrics = {}
        for ratio in COMPRESSION_RATIOS:
            for sched in ["A", "B", "C"]:
                r = results[ratio][sched]
                flat_metrics[f"ratio{ratio}_sched{sched}_cosim_final"] = r["cosim"].get(300, 0.0)
                flat_metrics[f"ratio{ratio}_sched{sched}_cosim_100"] = r["cosim"].get(100, 0.0)

        # Check 2x and 4x specifically
        for ratio in [2, 4]:
            cosim_A = results[ratio]["A"]["cosim"].get(300, 0.0)
            cosim_B = results[ratio]["B"]["cosim"].get(300, 0.0)
            if cosim_A < 0.1:
                a_fails_count += 1
                if cosim_B > 0.3:
                    warmup_helps_count += 1
                elif cosim_B < 0.2:
                    b_still_bad_count += 1

        metrics = flat_metrics
        metrics["a_fails_at_2x_4x"] = a_fails_count
        metrics["warmup_restores_at_2x_4x"] = warmup_helps_count
        metrics["b_still_bad_count"] = b_still_bad_count

        config = dict(INPUT_DIM=INPUT_DIM, COMPRESSION_RATIOS=COMPRESSION_RATIOS,
                      BOTTLENECKS=BOTTLENECKS, TRAIN_STEPS=TRAIN_STEPS,
                      BATCH_SIZE=BATCH_SIZE, MEASURE_STEPS=MEASURE_STEPS)

        if warmup_helps_count > 0 and a_fails_count > 0:
            outcome = OUTCOME_SUPPORTED
            notes = (f"Schedule A fails (cosim<0.1) at {a_fails_count} of 2 ratios; "
                     f"warmup (B) restores {warmup_helps_count} of those.")
        elif b_still_bad_count == a_fails_count and a_fails_count > 0:
            outcome = OUTCOME_REFUTED
            notes = "All schedules fail at 2x-4x; warmup does not help."
        elif warmup_helps_count > 0:
            cosim_B_2x = results[2]["B"]["cosim"].get(300, 0.0)
            cosim_B_4x = results[4]["B"]["cosim"].get(300, 0.0)
            if cosim_B_2x < 0.2 and cosim_B_4x < 0.2:
                outcome = OUTCOME_INCONCLUSIVE
                notes = "Warmup helps but cosim still <0.2 at 2x-4x."
            else:
                outcome = OUTCOME_INCONCLUSIVE
                notes = "Warmup partially helps but A does not clearly fail."
        else:
            outcome = OUTCOME_INCONCLUSIVE
            notes = "A does not fail at 2x-4x; hypothesis conditions not triggered."

        return self.result(outcome, metrics, notes, config)


if __name__ == "__main__":
    Exp122CompressionGradientStarvation().execute()
