# Sprint 2 — Mamba SSM Backbone (exp_poc_b)

**Goal:** Replace L1 SWA with Mamba-1 SSM. Test the core backbone swap.

**Config:**
- d_model=128, n_heads=4, n_layers=4, ff_mult=4
- segment_len=128, batch_size=8, steps=10,000
- `--use-mamba --mamba-d-state 16 --mamba-d-conv 4 --mamba-expand 2`
- optimizer: AdamW lr=3e-4, wd=0.1, warmup=500
- val_every=500, log_every=100

**Sprint 1 baseline:** median val_ppl = 1.12 (seeds 42/43/44)

**Gate:** median val_ppl ≤ 1.32 (Sprint 1 + 0.20 tolerance)

**Command:**
```bash
bash scripts/run_poc_sprint2.sh
```

---

## Results

| seed | val_ppl (step 10k) | tok/s | status |
|------|--------------------|-------|--------|
| 42   | —                  | —     | RUNNING |
| 43   | —                  | —     | PENDING |
| 44   | —                  | —     | PENDING |

**Median val_ppl at step 10k:** TBD

**Gate:** val_ppl ≤ 1.32 → TBD

**vs baseline:** TBD (target: ≤ +0.20 over Sprint 1 median 1.12)

---

## Diagnostic to watch

- If Mamba val_ppl ≥ 1.62 (≥+0.50 over baseline), check that `log_A` gradient is
  flowing — the selective scan must learn, not just pass state unchanged (D-skip dominated).
- Fallback: reduce `--mamba-d-state 8`, increase `--mamba-expand 4`, retry.

---

## Logs

- `results/poc/sprint2_seed42.log`
- `results/poc/sprint2_seed43.log`
- `results/poc/sprint2_seed44.log`

## Checkpoints

- `checkpoints/poc_b_s42/step_0010000_final.safetensors`
- `checkpoints/poc_b_s43/step_0010000_final.safetensors`
- `checkpoints/poc_b_s44/step_0010000_final.safetensors`

---

## Notes

Fill in after runs complete:
- Convergence behaviour:
- Wall clock time per seed:
- NaN skip rate (compare to Sprint 1 — Mamba may be more stable on MPS):
- Any anomalies:
