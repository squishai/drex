#!/usr/bin/env bash
# Sprint 2 — Mamba SSM Backbone (exp_poc_b)
# Replace L1 sliding-window attention with Mamba-1 SSM.
# Hypothesis: Mamba's selective state-space dynamics match or beat SWA, with O(n) complexity.
#
# Success criterion: median val_ppl(Sprint 2) ≤ median val_ppl(Sprint 1) + 0.20
#   Sprint 1 baseline median = 1.12  →  gate = ≤ 1.32
#
# Usage:
#   cd /path/to/drex
#   bash scripts/run_poc_sprint2.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

mkdir -p results/poc

for SEED in 42 43 44; do
    echo ""
    echo "======================================================="
    echo "  Sprint 2 (Mamba backbone) — seed ${SEED}"
    echo "======================================================="
    echo ""

    python3 scripts/train.py \
        --d-model 128 --n-heads 4 --n-layers 4 --ff-mult 4 \
        --segment-len 128 \
        --use-mamba --mamba-d-state 16 --mamba-d-conv 4 --mamba-expand 2 \
        --steps 10000 --batch-size 8 \
        --val-every 500 --log-every 100 --save-every 5000 \
        --ckpt-dir "checkpoints/poc_b_s${SEED}" \
        --seed "${SEED}" \
        2>&1 | tee "results/poc/sprint2_seed${SEED}.log"

    echo ""
    echo "  seed ${SEED} done — last val_ppl:"
    grep "\[val\]" "results/poc/sprint2_seed${SEED}.log" | tail -1
    echo ""
done

echo "======================================================"
echo "  Sprint 2 complete — all seeds done."
echo "  Gate: median val_ppl ≤ 1.32 (Sprint 1 + 0.20)"
echo ""
echo "  val_ppl at step 10k per seed:"
for SEED in 42 43 44; do
    ppl=$(grep "\[val\] step  10000" "results/poc/sprint2_seed${SEED}.log" 2>/dev/null | grep -oE "val_ppl +[0-9.]+" | awk '{print $2}' || echo "n/a")
    echo "    seed ${SEED}: ${ppl}"
done
echo "======================================================"
