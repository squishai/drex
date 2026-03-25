# Sprint 2b — Transformer + ESN Episodic Memory (MPS-Native)

**Experiment:** exp_poc_2b  
**Date started:** 2026-03-24  
**Platform:** M3 MPS (Apple Silicon) — runs at full transformer speed (~15k–30k tok/s)  
**Config:** d=128, 4L, 4H, segment_len=128, batch=8, 10k steps, 3 seeds  
**Flags:** `--use-episodic-memory --use-esn-memory --esn-reservoir-mult 4 --esn-spectral-radius 0.95`

## Why This Sprint

Sprint 2 (Mamba backbone) stalled locally: ~200 tok/s on MPS due to no hardware-accelerated
scan kernel for Apple Silicon. Sprint 2b isolates the ESN episodic memory contribution on
the **baseline transformer** backbone — no Mamba required, runs at full speed locally.

This is a clean single-variable ablation:
- Sprint 1: Transformer (baseline) → median ppl **1.12**
- Sprint 2b: Transformer + ESN memory → **this run**
- Sprint 2: Transformer → Mamba (cloud) → TBD
- Sprint 3: Mamba + ESN (cloud) → TBD

## Gate

**Pass:** median val_ppl at step 10k ≤ **1.32** (Sprint 1 median + 0.20)  
**Bonus:** median val_ppl < **1.12** (strictly beats baseline — ESN clearly helps)

## Results

| Seed | val_ppl @ 10k | Trainable Params | Status |
|------|---------------|-----------------|--------|
| 42   | —             | —               | 🔄 RUNNING |
| 43   | —             | —               | 🔲 TODO |
| 44   | —             | —               | 🔲 TODO |

**Median val_ppl:** —  
**Gate result:** PENDING

## Write Rate (wr) Diagnostics

Expected range: [0.10, 0.85] at each log step.

| Seed | wr @ step 2000 | wr @ step 5000 | wr @ step 10000 | Note |
|------|----------------|----------------|-----------------|------|
| 42   | —              | —              | —               |      |
| 43   | —              | —              | —               |      |
| 44   | —              | —              | —               |      |

## Notes

- ESN reservoir (mult=4, sr=0.95, c=default) has **zero trainable parameters**
- Net parameter increase over baseline: only the linear readout and write gate
- NaN skip behavior expected ~every 12 steps (InfiniAttention L2 on MPS) — normal
- If wr > 0.85: rerun with `--episodic-gate-thresh 0.50`
- If wr < 0.10: rerun with `--episodic-gate-thresh 0.40`

## Logs

- `results/poc/sprint2b_seed42.log`
- `results/poc/sprint2b_seed43.log`
- `results/poc/sprint2b_seed44.log`
