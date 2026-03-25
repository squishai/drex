#!/usr/bin/env bash
# Sprint 2b — Transformer + ESN Episodic Memory (no Mamba, MPS-native)
#
# Rationale: Sprint 2 (Mamba backbone) runs at ~200 tok/s on Apple Silicon MPS
# due to missing hardware-accelerated scan kernels. This sprint isolates the ESN
# episodic memory contribution on top of the baseline transformer — runs at full
# MPS speed (~15k–30k tok/s), completing in ~30–60 min for 3 seeds.
#
# Experiment: Transformer L1 (SWA) + L2 (InfiniAttention) + ESN episodic memory
# Hypothesis: ESN zero-training-cost associative memory improves baseline ppl
# Success criterion: median val_ppl ≤ Sprint 1 median (1.12) + 0.20 = 1.32
#           AND ideally median val_ppl < 1.12 (beats baseline outright)
#
# Usage: bash scripts/run_poc_sprint2b.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RESULTS_DIR="${REPO_ROOT}/results/poc"
CKPT_ROOT="${REPO_ROOT}/checkpoints"
LOG_DIR="${RESULTS_DIR}"

mkdir -p "${RESULTS_DIR}" "${CKPT_ROOT}"

echo "=== Sprint 2b: Transformer + ESN Episodic Memory ===" 
echo "Repo: ${REPO_ROOT}"
echo "Logs: ${LOG_DIR}"
echo ""

for SEED in 42 43 44; do
    CKPT_DIR="${CKPT_ROOT}/poc_2b_s${SEED}"
    LOG_FILE="${LOG_DIR}/sprint2b_seed${SEED}.log"

    echo "--- Seed ${SEED} ---"
    echo "Checkpoint: ${CKPT_DIR}"
    echo "Log:        ${LOG_FILE}"
    echo ""

    python3 "${REPO_ROOT}/scripts/train.py" \
        --d-model 128 \
        --n-heads 4 \
        --n-layers 4 \
        --segment-len 128 \
        --use-episodic-memory \
        --use-esn-memory \
        --esn-reservoir-mult 4 \
        --esn-spectral-radius 0.95 \
        --steps 10000 \
        --batch-size 8 \
        --val-every 500 \
        --log-every 100 \
        --save-every 5000 \
        --ckpt-dir "${CKPT_DIR}" \
        --seed "${SEED}" \
        2>&1 | tee "${LOG_FILE}"

    echo ""
    echo "Seed ${SEED} complete. Log saved to ${LOG_FILE}"
    echo ""
done

echo "=== Sprint 2b complete. All 3 seeds done. ==="
echo "Results: ${LOG_DIR}/sprint2b_seed{42,43,44}.log"
echo ""
echo "NEXT: run scripts/record_sprint2b_results.py (or manually read logs)"
echo "Gate: median val_ppl at step 10000 <= 1.32"
