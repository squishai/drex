# Sprint 4 — Mamba + ESN + HDC Encoder (exp_poc_d)

**Goal:** Add HDC encoder on top of Sprint 3 (Mamba + ESN). Full DREX-UNIFIED core stack.

**Config:**
- d_model=128, n_heads=4, n_layers=4, ff_mult=4
- segment_len=128, batch_size=8, steps=10,000
- `--use-mamba --mamba-d-state 16 --mamba-d-conv 4 --mamba-expand 2`
- `--use-episodic-memory --use-esn-memory --esn-reservoir-mult 4 --esn-spectral-radius 0.95`
- `--use-hdc-encoder --hdc-dim 512 --hdc-seed 0`
- `--tbptt-reset-every 12`
- optimizer: AdamW lr=3e-4, wd=0.1, warmup=500
- Platform: Kaggle T4 (14.6 GB VRAM)

**Sprint 3 baseline (cloud CUDA):** median val_ppl = 2.88 (seeds 42/43/44)

**Gate:** median val_ppl ≤ 2.88 (Sprint 3 median)

---

## Results

| seed | val_ppl (step 10k) | tok/s   | elapsed | status |
|------|--------------------|---------|---------|--------|
| 42   | 3.15               | ~1,720  | 7821s   | ✅ OK  |
| 43   | 3.15               | ~1,730  | 7965s   | ✅ OK  |
| 44   | 3.16               | ~1,738  | 7545s   | ✅ OK  |

**Median val_ppl at step 10k:** 3.15

**Gate:** median ≤ 2.88 → ❌ FAIL (regression: +0.27 ppl over Sprint 3)

> **Note:** The Kaggle session gate check reported ✅ PASS — this is a **false positive**.
> Sprint 3's summary file was absent from the fresh Kaggle session, causing the gate to
> evaluate `3.15 ≤ 9999` (default). The true result is a regression.

---

## Root Cause: hdc_dim / d_model Mismatch

**hdc_dim=512, d_model=128 → 4:1 ratio.** The HDC hypervectors are linearly projected
512→128 before entering Mamba. This collapses the high-dimensional symbolic geometry that
makes HDC useful — the model pays compute cost for the HDC projection with no benefit.
At hdc_dim ≤ d_model, the geometry is preserved in the downstream space.

Additional signal:
- **NaN cascades onset shifted earlier:** step ~9600 (Sprint 4) vs ~9900 (Sprint 3).
  HDC destabilizes training dynamics ~300 steps earlier than ESN alone.
- **Variance near-zero:** 3.15, 3.15, 3.16 (±0.005). The regression is real and
  consistent across all seeds — not noise.
- **Throughput: ~1,720 tok/s** — slightly faster than Sprint 3 (~1,580 tok/s) because
  HDC projection is a cheap linear op vs the ESN reservoir recurrence.

---

## Sprint 4b Ablation (recommended)

Run Sprint 4 again with `--hdc-dim 128` (matching d_model). Expected outcome: HDC
contribution should be neutral-to-positive when the projection preserves dimensionality.
This isolates whether the regression is architecture-level or dimension-tuning.

```bash
# Proposed Sprint 4b flags (add to run_poc_cloud.py or run manually)
--use-mamba --mamba-d-state 16 --mamba-d-conv 4 --mamba-expand 2
--use-episodic-memory --use-esn-memory --esn-reservoir-mult 4 --esn-spectral-radius 0.95
--use-hdc-encoder --hdc-dim 128 --hdc-seed 0   # ← matched to d_model
--tbptt-reset-every 12 --batch-size 8
```

---

## Sprint progression (CUDA cloud baseline)

| Sprint | Config                        | Median val_ppl | Δ vs prev | Gate   |
|--------|-------------------------------|----------------|-----------|--------|
| 2      | Mamba backbone                | 3.03           | —         | ✅ PASS |
| 3      | + ESN episodic memory         | 2.88           | −0.15     | ✅ PASS |
| 4      | + HDC encoder (hdc_dim=512)   | **3.15**       | **+0.27** | ❌ FAIL |
| 4b     | + HDC encoder (hdc_dim=128)   | TBD            | TBD       | pending |
| 5      | Scale (d=256, 8L, 50k, s42)   | TBD            | TBD       | pending |
