# Changelog

All notable changes to this project will be documented in this file.

Format: [Conventional Commits](https://www.conventionalcommits.org/) · [Semantic Versioning](https://semver.org/)

---

## [Unreleased]

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
