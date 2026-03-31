# Changelog

All notable changes to this project will be documented in this file.

Format: [Conventional Commits](https://www.conventionalcommits.org/) · [Semantic Versioning](https://semver.org/)

---

## [Unreleased]

---

## Sprint 2b Complete — 2026-05-22

Transformer + ESN episodic memory 3-seed run complete. Gate PASS with bonus.

**Results:**
- Seed 42: val_ppl = 1.10 @ step 10k
- Seed 43: val_ppl = 1.12 @ step 10k
- Seed 44: val_ppl = 1.08 @ step 10k
- **Median: 1.10 — PASS (≤1.32) — Bonus PASS (median < 1.12, beats Sprint 1 baseline)**

**Config:** d=128, 4L, 4H, segment_len=128, batch=8, ESN mult=4, sr=0.95, c=0.01

**Files added/updated:**
- `results/poc/sprint2b_seed42.log` — complete, 10k steps
- `results/poc/sprint2b_seed43.log` — complete, 10k steps
- `results/poc/sprint2b_seed44.log` — complete, 10k steps
- `results/poc/sprint2b_esn.md` — results recorded, trajectory table added
- `DREX_UNIFIED_PLAN.md` — Sprint Checklist Sprint 2b updated to ✅ DONE
- `checkpoints/poc_2b_s42/`, `checkpoints/poc_2b_s43/`, `checkpoints/poc_2b_s44/` — final checkpoints saved

---

## Phase 1 Complete — 2026-03-30

All nine DREX-UNIFIED Phase 1 components implemented and unit-validated.
**136/136 tests passing.** Commit range: `02630a7..82fa74c`.

| Component | File | Tests | Key Contract |
|-----------|------|-------|--------------|
| HDC Token Encoder (Obj 0) | src/hdc/encoder.py | 18 | cosine_sim(A,A)>0.999; mean cross-pair<0.05 at d_hdc=4096 |
| Mamba SSM Backbone (Obj 1) | src/backbone/mamba.py | 12 | all PC layer losses decrease; causality verified |
| DREX Controller (Obj 3) | src/controller/policy.py | 10 | REINFORCE beats random; routing collapse detection live |
| ESN Working Memory (Obj 2a) | src/memory/reservoir.py | 17 | spectral_radius<1.0 for ρ∈{0.90,0.95,0.99}; feedback extends recall |
| Episodic Memory (Obj 2b) | src/memory/episodic.py | 17 | EMA stability; α=0.90 optimal; force_overwrite verified |
| NoProp Semantic Memory (Obj 2c) | src/memory/semantic.py | 13 | block independence CI passes; optimizer isolation verified |
| KAN Readout (Obj 5) | src/readout/kan.py | 5 | MLP parity within 2%; param overhead 8.86x<18x bound |
| CI — Gradient Leak | test_gradient_leak_ci.py | 9 | PCN cross-layer=0; NoProp cross-block=0; ESN reservoir grad=0 |
| CI — Dtype Contracts | test_dtype_contracts_ci.py | 14 | HDC→float32; Mamba→bfloat16; Controller→int32/float32/bool |
| CI — Shape Contracts | test_shape_contracts_ci.py | 21 | all component shape boundaries verified |

---

### Wave 6 — KAN Readout (feat(readout))

- **Added** `src/readout/kan.py`: `BSplineKANLayer` (Cox–de Boor B-spline recursion, extended knot vector, SiLU residual base weight) and `KANReadout` (2-layer stack, hidden=max(√(d_in·d_out), 32)), float32 throughout (DREX dtype contract satisfied).
- **Added** `tests/python/test_kan.py`: 5 validation tests — MLP-parity approximation, spline-variation post-fitting, parameter-count scaling bound, forward timing (<5 s), deterministic regression snapshot.
- **Added** `tests/python/fixtures/kan_regression_snapshot.npy`: regression reference output (shape (20,32) float32, seed 42).
- **Updated** `DREX_UNIFIED_PLAN.md`: Wave 6 row marked ✅ COMPLETE.
- **Test count**: 136/136 passing.

### Wave 5 — Controller RL (feat(controller))

- `src/controller/policy.py`, `src/controller/reward.py`
- 136 tests total (10 added in this wave), commit `2bdf9e1`

### Wave 4 — Semantic Memory NoProp (feat(memory))

- `src/memory/semantic.py`
- 131 tests total (13 added), commit `b15d086`

### Wave 3 — Mamba Backbone (feat(backbone))

- `src/backbone/mamba.py`
- 108 tests total (12 added), commit `a80b95d`

### Wave 2 — HDC + ESN (feat(hdc,esn))

- 96 tests total (34 added), commit `1eca043`

### Wave 1 — Episodic Memory + Sparse Router (feat(memory,router))

- 62 tests total (18 added), commit `02630a7`

### Wave 0 — Foundation (feat(foundation))

- Initial 44 tests, project scaffold, conftest, pyproject.toml
