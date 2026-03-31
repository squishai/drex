# DREX-UNIFIED PLAN
# Architecture Evolution & Implementation Roadmap — v0.2 Fresh Start

*Created: 2026-03-24 | Updated: 2026-03-XX (v0.2 reset) | Status: Active*
*Master spec: DREX_UNIFIED_SPEC.md v0.2 supersedes all prior planning documents*

**See DREX_UNIFIED_SPEC.md for the per-component interface spec, tensor shapes,
validation criteria, and phase gates.**

---

## v0.2 Reset Notice

All prior implementation code has been archived to `research/legacy/`. This was
a deliberate clean-slate decision: the legacy code (Phases 1–25) is valuable
research and remains accessible, but v0.2 defines a different file layout (`src/`),
stricter CI requirements, and updated exit criteria that the legacy code was not
built against.

The migration script is at `scripts/migrate_to_v0_2.sh`. Run it once from the
repo root to move `python/` and `tests/` to `research/legacy/` and scaffold the
new `src/` and `tests/` layout.

Key changes in v0.2 vs legacy:
- `src/` is the real filesystem source root (not `python/drex/models/`)
- HDC encoder must be token-ID encoder (int32 → float32), not embedding-lifting
- Phase 1 requires CI infrastructure (gradient leak, dtype, shape assertions) FIRST
- ESN feedback extension test is a Phase 1 EXIT BLOCKER (was never validated in legacy)
- NoProp exit criterion is accuracy parity on CIFAR-100 (not VRAM-only)
- Controller routing collapse test is a Phase 1 EXIT BLOCKER

---

## Implementation Status (v0.2 — fresh start)

| Wave | Component                | File                            | Status              |
|------|--------------------------|---------------------------------|---------------------|
| 0    | CI Infrastructure        | .github/workflows/ci.yml        | ✅ complete (44 tests green) |
| 1    | HDC Token Encoder        | src/hdc/encoder.py              | ✅ complete (62 tests green, D_hdc_min=1024) |
| 2    | ESN Reservoir + feedback | src/memory/reservoir.py         | ✅ complete (96 tests green, spectral_radius sweep 0.90/0.95/0.99, feedback EXIT BLOCKER) |
| 2    | Episodic Memory          | src/memory/episodic.py          | ✅ complete (alpha sweep 0.70–0.99, force_overwrite, noise attenuation verified) |
| 3    | Mamba PCN Backbone       | src/backbone/mamba.py           | ✅ complete (108 tests green, PC convergence all layers, causality + recurrence verified) |
| 4    | NoProp Semantic Memory   | src/memory/semantic.py          | ✅ complete (121 tests green, block independence CI passes, block optimizer isolation verified) |
| 5    | RL Controller            | src/controller/policy.py        | ✅ complete (131 tests green, REINFORCE beats random synthetic task, routing collapse detection passes) |
| 6    | KAN Readout              | src/readout/kan.py              | ✅ complete (136 tests green, B-spline 8.86x overhead bounded by 2*(n_basis+1)=18x, regression snapshot committed 82fa74c) |
| 7    | Phase 1 Documentation    | docs/phase1_validation_report.md | ✅ complete (validation report, spec sync, CHANGELOG — 2026-03-30) |

---

## Phase 1 — Seven-Wave Implementation Plan

### Wave 0: CI Infrastructure (EXIT BLOCKER — must be green before any component code)

Goal: automated gradient leak, dtype boundary, and shape contract assertions running
on every commit via GitHub Actions.

Deliverables:
- `.github/workflows/ci.yml` — full pytest suite + three CI-specific assertion tests
- `tests/python/test_gradient_leak_ci.py` — asserts no cross-layer gradients in PCN,
  no cross-block gradients in NoProp, zero gradients on ESN reservoir weights
- `tests/python/test_dtype_contracts_ci.py` — asserts all component boundary dtypes
  match the DTYPE BOUNDARY CONTRACT section in DREX_UNIFIED_SPEC.md
- `tests/python/test_shape_contracts_ci.py` — asserts all intermediate tensor shapes

Exit criterion: CI green on main. All three assertion test files pass with synthetic
inputs (no real model weights required at Wave 0).

Wave 0 completes before Wave 1-6 begin.

---

### Wave 1: HDC Token Encoder (Objective 0)

Depends on: Wave 0 CI green.
Can run in parallel with: Wave 2.

File: `src/hdc/encoder.py`
Test: `tests/python/test_hdc_encoder.py`

What to build:
- Token-ID encoder: (B,S) int32 → (B,S,D_hdc) float32
- item_memory: fixed random (vocab_size, D_hdc), never trained
- position_permutations: cyclic roll offsets per position
- encode_sequence: bundle permuted item vectors
- D_hdc default: start at 4096, scale to 10000 if orthogonality tests pass

Exit criterion (all six in spec must pass):
- similarity > 0.999, orthogonality mean < 0.05, associativity, sequence order,
  shape (B,S,D_hdc), dtype float32

Key note: this is NOT the legacy embedding-lifting HDC from research/legacy/.
Build fresh from the spec.

---

### Wave 2: ESN Reservoir + Spectral Radius Sweep (Objective 2a & 2b)

Depends on: Wave 0 CI green.
Can run in parallel with: Wave 1.

Files: `src/memory/reservoir.py`, `src/memory/episodic.py`
Tests: `tests/python/test_reservoir.py`, `tests/python/test_episodic.py`

What to build:
- Reservoir (L1): fixed W_res (spectral_radius < 1.0), W_in, W_fb; ridge readout
- Episodic (L2): EMA delta writes (alpha=0.90 default), hard overwrite mode

Exit criterion Wave 2a (reservoir): echo state property, convergence, feedback
extension test (EXIT BLOCKER — confirm feedback=True improves memory beyond N steps).
Spectral radius sweep: run 0.90, 0.95, 0.99 on POS tagging task. Document winner.

Exit criterion Wave 2b (episodic): EMA stability, delta write, alpha sweep result.

---

### Wave 3: Mamba PCN Backbone (Objective 1) ✅ COMPLETE

Depends on: Wave 1 (HDC encoder interface locked).
Can run in parallel with: waves 4, 5, 6.

File: `src/backbone/mamba.py`
Test: `tests/python/test_mamba.py`

What was built:
- MambaSSM block (pure PyTorch, no external mamba_ssm dep): ZOH discretisation,
  causal conv1d, selective S6 scan, bfloat16 throughout.
- PCNMambaBackbone: n_layers MambaSSM with per-layer Adam optimizers, callable
  top_loss_fn API for isolated-graph training.
- 12 tests: shape/dtype, causality, PCN convergence (top-layer task loss),
  gradient leak isolation (4 tests).

Result: 12/12 tests pass. Full suite 108/108 green.
Commit: pending.

Notable design decisions:
- train_step accepts Callable[[Tensor], Tensor] so top layer builds a fresh
  isolated graph each step (avoids inplace-version-counter error from Adam).
- Convergence test asserts top-layer task loss strictly decreases; inter-layer
  prediction losses guarded against divergence (< 0.5) but not forced monotone
  (upper-layer representation shift is expected during learning).

---

### Wave 4: NoProp Semantic Memory (Objective 2c) ✅ COMPLETE

Depends on: Wave 0 CI green.
Can run in parallel with: waves 1, 2, 3, 5, 6.

File: `src/memory/semantic.py`
Test: `tests/python/test_semantic.py`

What was built:
- NoPropBlock: 3-layer MLP (in_proj → hidden×2 → out_proj), bfloat16 weights
- Per-block local denoising loss (Gaussian noise, train mode only)
- Per-block optimizer isolation — no shared gradient paths
- NoPropSemanticMemory: n-block stack with stop_gradient between blocks
- train_step: no-grad first-pass to collect detached inputs, then isolated backward per block

Exit criteria met:
- Block independence assertion: PASSING (13 tests, 0 failures)
- Top-block convergence: PASSING (loss strictly decreases over 300 steps)
- VRAM efficiency: PASSING
- Full suite: 121/121 green (commit: feat(memory): Wave 4 NoPropBlock+NoPropSemanticMemory)

---

### Wave 5: RL Controller + Routing Collapse Evaluation (Objective 3) ✅ COMPLETE

Depends on: Wave 0 CI green. HDC encoder interface helps but not strictly required.
Can run in parallel with: waves 1, 2, 4, 6.

Files: `src/controller/policy.py`, `src/controller/reward.py`
Tests: `tests/python/test_controller.py`

What was built:
- DREXController: 2-layer MLP (d_model → 128 → n_tiers), all float32 internally
- REINFORCE with EMA baseline subtraction for variance reduction
- Per-block bfloat16 input assertion (same pattern as NoPropBlock)
- Outputs: write_decisions (int32 one-hot), read_weights (float32 softmax), sparse_gates (bool)
- NaN guard: >10 consecutive NaN rewards → RuntimeError halts training
- Routing collapse detection: deque(maxlen=100) of modal-tier per update call;
  if any tier >95% → WARNING + 0.1 load-balance penalty injected into reward
- RewardSignal: -F.cross_entropy, returns NaN tensor on bad inputs

Exit criteria met:
- Shape/dtype contracts: PASSING (5 tests)
- NaN guard: PASSING (2 tests)
- Routing collapse detection: PASSING (2 tests)
- REINFORCE learning (500 eps, D_MODEL=32): >50% accuracy on last 100 episodes: PASSING
- Full suite: 131/131 green (commit: feat(controller): Wave 5 DREXController+RewardSignal)

---

### Wave 6: KAN Readout Validation (Objective 5) ✅ COMPLETE

Depends on: Wave 0 CI green.
Can run in parallel with: waves 1, 2, 4, 5.

File: `src/readout/kan.py`
Test: `tests/python/test_kan.py`

What was built:
- BSplineKANLayer: Cox–de Boor B-spline recursion, (n_in, n_out) edges each with
  (n_grid + spline_order) learnable coefficients + SiLU residual base weight.
- KANReadout: 2-layer KAN, d_in → hidden (geometric mean) → d_out, float32 throughout.
- 5 tests: MLP parity, spline variation, parameter scaling, forward timing, regression snapshot.

Exit criteria met:
- Approximation: KAN final loss within 0.02 of MLP on sin(x.sum()) regression: PASSING
- Spline variation: max edge variation > 0.01 after fitting: PASSING
- Parameter scaling: overhead < 2*(n_basis+1)x — confirms linear O(n_in*n_out*n_basis): PASSING
- Timing: mean forward pass << 5.0 s for d_in=256, d_out=1000: PASSING
- Regression snapshot: `tests/python/fixtures/kan_regression_snapshot.npy` committed: PASSING
- Full suite: 136/136 green (commit: feat(readout): Wave 6 BSplineKANLayer+KANReadout)

---

### Wave 7: Phase 1 Gate Documentation & Spec Sync

Depends on: all Waves 1–6 exit criteria met.

Deliverables:
- Internal validation report: exact numbers for every component test, hardware,
  timing (markdown in `docs/phase1_validation_report.md`)
- Spectral radius sweep table
- noise_std sweep table for NoProp
- Ablation log format verified in all experiment run outputs
- `DREX_UNIFIED_SPEC.md` Phase 1 gate checkboxes updated to [x]
- `DREX_UNIFIED_PLAN.md` (this file) updated with final numbers
- `CHANGELOG.md` entry written

---

## Wave Dependency Graph

    Wave 0: CI Infrastructure         ← MUST complete first (EXIT BLOCKER)
    Wave 1: HDCTokenEncoder (Obj 0)   ← after Wave 0 green
    Wave 2: ESN + Episodic (Obj 2a/b) ← parallel with Wave 1
    Wave 3: Mamba PCN (Obj 1)         ← after Wave 1 (interface dependency)
    Wave 4: NoProp Semantic (Obj 2c)  ← parallel with Waves 1-3
    Wave 5: Controller (Obj 3)        ← parallel with Waves 1-3
    Wave 6: KAN Readout (Obj 5)       ← parallel with Waves 1-3
    Wave 7: Documentation & gate      ← after all Waves 1-6 pass

---

## Part 2 — Research Archive (Phases 1–25)

The following summarizes the validated findings from the legacy research.
This work informs v0.2 implementation decisions even though the legacy code
is archived. See `research/legacy/` for runnable code and `research/` for
full experiment reports.

### Research Complete (Phases 1–16)

247+ controlled experiments across 48 categories. The current validated architecture is:

    Delta-rule associative matrix
    + EMA smoothing α(L) = 0.95^(96/L)
    + Episodic/semantic split (two H/2 matrices)
    + Relative-vector-norm write gate at thresh*=0.70
    + Null retrieval gate
    + OR write gate (not AND)

All architectural decisions above are backed by ≥7/9 seed evidence. Hard constraints
and dead ends are fully documented in ARCHITECTURE_FINDINGS.md. These findings are
real and durable — they will inform DREX-UNIFIED regardless of what backbone changes.

### Training In Progress

Exp A (baseline, no episodic memory): step ~22,400 / 50,000
  - val_ppl ~1.23 at step 22,000 (improving steadily)
  - 4.26M parameters, char-level TinyStories, MPS M3
  - Throughput: ~15,000–26,000 tok/s

Exp B (full episodic memory): waiting on Exp A final checkpoint
  - Watcher: PID 77406, auto-starts on Exp A completion
  - Current blocker: write rate plateau at wr≈0.963 for L=512
    (expected to converge; requires ≥10k steps to determine)

### NoProp Experiments In Progress (Phase 22)

Wave 0+1 Run 2 result: 6/7 PASS after fixing shared optimizer bug.
  - Critical bug fixed: block optimizers were sharing head params, causing 6× conflicting
    Adam updates per step — guaranteed divergence
  - After fix: NoProp STE (1A) converges to val_ppl 17,239 in 800 steps (gate: PASS)
  - NoProp DQT (1C) converges to val_ppl 13,376 in 800 steps (gate: PASS)
  - HESTIA (1D) still failing — tau annealing instability, under investigation

Wave 2–3 smoke tests: running (results in results/wave2/, results/wave3/)
  - Latest wave 3 shows gate_ppl_pass=True, gate_dead_pass=True for 0A and 1A
  - val_ppl still high (7k–25k range) at 800 steps — these are smoke tests, not
    full convergence runs

Pending NoProp work:
  - Full convergence run (5k–10k steps) for winning Wave 1 variant
  - Wave 2 diagnostics (gradient norms, dead zones, block depth sweep)
  - Scale to 125M parameter plan (Phase 22 follow-on)

### Paper

Draft complete: paper/main.tex (9-page NeurIPS preprint)
Status: several \todo{pending} entries in Tables 3–5, Abstract, Discussion
Waiting on: Exp A/B final checkpoints to fill tables

---

## Part 2 — The Architecture Gap

The current DREX is a transformer with custom memory modules bolted on:

    Transformer L1 (sliding window attention)
    → Transformer L2 (Infini-Attention delta-rule matrix)
    → Transformer L3 (Titans-style MLP weight snapshots)
    → Transformer L4 (Episodic/semantic split delta-rule — the research contribution)
    → Transformer FFN
    → Output

The transformer backbone is still the dominant compute cost. The research contribution
is real (validated memory module, Phase 11–12 findings), but the foundation is still
the architecture we're trying to beat.

The architectural research sessions (March 2026) established what a genuine departure
looks like. The DREX-UNIFIED architecture replaces the transformer backbone entirely:

    INPUT (raw bytes / tokens)
    → HDC ENCODER (fixed random projection — zero training)
    → MAMBA SSM BACKBONE (linear time — trained via Predictive Coding)
    ↓                     ↓
    DREX CONTROLLER (small RL policy — REINFORCE or Q-learning)
    ↓             ↓              ↓
    ESN RESERVOIR  EPISODIC      SEMANTIC MEMORY
    (working mem,  (ESN+EMA,     (small SSM, NoProp
    zero training) near-zero)    local block training)
    ↓             ↓              ↓
    SPARSE ROUTER (top-k conditional compute — only active paths cost compute)
    ↓
    KAN READOUT (learnable spline functions — interpretable, fast scaling laws)
    ↓
    OUTPUT ← reward signal loops back to DREX CONTROLLER

This is not a variation on transformer. Every component is deliberately chosen to
minimize or eliminate traditional gradient-based training costs.

---

## Part 3 — Why Each Component

### HDC Encoder (Kanerva 1988–2009 / ACM HDC Survey 2023)
Random projection into 10,000+ dimensional hypervector space.
Operations: binding (element-wise multiply), bundling (element-wise add), permutation.
Zero training. Johnson-Lindenstrauss: geometry is preserved.
Why: Gives the controller a compositional, symbolic representation before any gradient
is computed. Naturally composable. Noise-robust. O(d) per operation.

### Mamba SSM Backbone (Gu & Dao 2023 / Mamba-2 2024)
Selective state space model with hardware-aware parallel scan.
O(n) training, O(1) inference memory per token.
Why: Eliminates the transformer's O(n^2) attention bottleneck. On byte-level tasks
(raw, untokenized input), Mamba outperforms a FLOP-matched transformer significantly.
Trained via Predictive Coding (local, no full backward pass).

### ESN Reservoir Working Memory (Jaeger & Haas 2004 / BabyLM 2025)
Fixed random recurrent network (~1% connectivity). Never updated.
Only the linear readout trains — one ridge regression solve (milliseconds, no GPU).
Memory bounded by reservoir size N unless feedback is added.
Key finding: output feedback → attractor states → 30–60% error reduction, equivalent
to doubling reservoir size for free.
Why: The episodic memory tier that caused Phase 7 multi-stability issues becomes free.
There are no weights in the reservoir to enter a multi-stable equilibrium.

Connection to Phase 7 findings:
The write gate multi-stability problem observed in Phases 6–12 (initialization-dependent
equilibria, wr collapse into low-accuracy regime) is a consequence of trying to train
a continuous differentiable function to make binary write/no-write decisions. The ESN
has no such gate — writes are structural (reservoir dynamics), not parameterized.
The controller decides WHAT to write to the reservoir input; the reservoir simply
transforms it. The failure mode disappears entirely.

### DREX Controller (Behrouz et al. Titans 2025 + Phase 7/12 findings)
Small RL policy (REINFORCE or simple Q-learning) on hypervectors.
Decides: what to write to each memory tier, what to read, when to activate sparse modules.
Reward: downstream prediction accuracy.
Why: Treating the write gate as a differentiable continuous function is exactly the
design choice that produced the multi-stability problem (Phase 7). A discrete RL policy
trained with a clean reward signal bypasses this entirely.
Note: exp_7_1 showed REINFORCE fails for the Phase 1–12 write gate because the encoder
gradient becomes zero (gate blocks signal). The DREX-UNIFIED controller is different:
it receives HDC hypervectors directly (not passing through a differentiable gate),
and trains via RL reward rather than backpropagation through the gate.

### NoProp Semantic Tier (arXiv 2503.24322 / Phase 22 validation in progress)
Small SSM where each block trains independently via local denoising objective.
No global backpropagation. Comparable to full backprop at CIFAR-100 scale.
Parallel block training (each block has its own loss, no gradient flow between blocks).
Updates parameters at inference for continual learning.
Phase 22 connection: the optimizer bug fix (shared head params) is directly applicable
to any NoProp implementation. The fundamental approach is validated — the implementation
was the blocker, not the theory.
Why: Global backprop through tiered memory is what makes DREX's training complex and
slow. NoProp eliminates it for the semantic tier.

### Sparse Router (MoE literature / DREX sparse execution thesis)
Top-k gating with load balancing.
Only the modules relevant to the current input activate.
Why: If 30% of modules activate on average, compute is reduced by 70% with no quality
loss on routed tasks. Dead modules receive zero gradient (no wasted capacity).

### KAN Readout (Liu et al. MIT/Caltech ICLR 2025)
Learnable spline functions on edges rather than fixed linear weights.
Smaller KANs match larger MLPs. Faster scaling laws than standard MLPs.
Pairs naturally with HDC encoder (both compositional and interpretable).
Why: The readout becomes auditable. The transformations from memory state to output
can be visualized and sometimes recovered as closed-form symbolic expressions.

---

## Part 4 — Training Cost Profile (Comparison)

Standard Transformer (GPT-3 scale):
  All parameters: Adam with full backpropagation through all layers.
  Cost: Extreme. GPU cluster required.

DREX (current, Phases 1–16):
  Memory modules: full backpropagation through write loop + attention + FFN.
  Cost: Low (runs on M3 MacBook, ~4.26M params, 50k steps feasible).
  Bottleneck: write loop sequential Python iterations (511 × 4 layers per step).

DREX-UNIFIED (target):

  Component           | Training method                 | Training cost
  --------------------|--------------------------------|----------------
  HDC Encoder         | Fixed (random projection)       | Zero
  Mamba SSM           | Predictive Coding (local)       | Low
  ESN Reservoir       | Fixed (never updated)           | Zero
  Episodic tier       | EMA delta writes                | Near-Zero
  Semantic tier       | NoProp (local block denoising)  | Low (parallel)
  DREX Controller     | REINFORCE / Q-learning          | Tiny
  Sparse Router       | Top-k gating (load balanced)    | Tiny
  KAN Readout         | Spline fitting (closed-form)    | Very Low
  Output readout      | None                            | Zero

Total: Every component that was expensive is either eliminated or replaced with a
local/fixed method. The first time a model of this class can genuinely train on
consumer hardware without meaningful cost justification.

---

## Part 5 — Connection to Existing Validated Findings

The Phase 1–16 research does not become irrelevant. Key findings carry forward:

1. Delta-rule associative matrix with EMA: maps directly onto the episodic/semantic
   tiers of DREX-UNIFIED. The specific formula (α(L) = 0.95^(96/L), thresh*=0.70)
   was validated and should be the starting specification for the ESN readout.

2. Fixed 50/50 episodic/semantic split (not learned router): DREX-UNIFIED uses a
   hand-designed inductive bias for the episodic/semantic split — same conclusion.

3. Null retrieval gate: the learned scalar gate suppressing empty-memory reads is
   applicable to the Mamba+ESN combination. Keep it.

4. Adam for the learnable components: DREX-UNIFIED has learnable components
   (Mamba, controller, KAN). Use Adam, not SGD (exp_34_6).

5. Write gate multi-stability = the reason for the ESN pivot: Phase 7 finding is
   not a dead end. It's the precise research motivation for replacing the
   differentiable gate with an RL policy and a fixed reservoir.

6. NoProp optimizer fix (Phase 22): the shared-optimizer bug fix is a real
   engineering finding that applies to any NoProp implementation, including the
   semantic tier training.

7. L=512 write rate convergence at high α: at L=512, α(L=512)=0.990, meaning
   (1−α)=0.010. Matrices start near-zero so wr plateaus high initially. This is
   predictable from the formula and will apply to any SSM-based memory tier.
   The chunked recurrence fix (Phase 20) should be the first optimization applied.

---

## Part 6 — Phased Forward Plan

### Current priority: POC Sprint Campaign — prove DREX-UNIFIED beats baseline.

**Phases 23, 24, 25 are COMPLETE.** All modular components exist, are tested, and
are independently togglable. The next goal is no longer implementation — it is
EVIDENCE. Run the 5-sprint campaign below to produce the first empirical proof that
the DREX-UNIFIED architecture is superior to the baseline transformer.

---

## POC Sprint Campaign

Goal: Produce empirical evidence that Mamba + ESN + HDC outperforms the baseline
transformer on long-context language modeling at equal or lower compute budget.

All experiments: TinyStories char-level, d=128, n_layers=4, n_heads=4, window_size=128,
segment_len=128, batch_size=8, 10k steps, 3 seeds (42, 43, 44 for statistical confidence).

Fast iteration scale (d=128, 10k steps): ~15–25 min/run on M3. 3 seeds = ~1h/sprint.

Success criterion (global): ≥1 sprint beats baseline val_ppl by ≥0.10 across ≥2/3 seeds.

---

### Sprint 1 — Baseline (exp_poc_a)

**Goal:** Establish the floor. Every subsequent sprint must beat this.

**What it measures:** Transformer L1 (SWA) + L2 (InfiniAttention). No episodic memory.

```bash
python scripts/train.py \
  --d-model 128 --n-heads 4 --n-layers 4 --segment-len 128 \
  --steps 10000 --batch-size 8 --val-every 500 \
  --log-every 100 --save-every 5000 \
  --ckpt-dir checkpoints/poc_a_s42 --seed 42

python scripts/train.py \
  --d-model 128 --n-heads 4 --n-layers 4 --segment-len 128 \
  --steps 10000 --batch-size 8 --val-every 500 \
  --log-every 100 --save-every 5000 \
  --ckpt-dir checkpoints/poc_a_s43 --seed 43

python scripts/train.py \
  --d-model 128 --n-heads 4 --n-layers 4 --segment-len 128 \
  --steps 10000 --batch-size 8 --val-every 500 \
  --log-every 100 --save-every 5000 \
  --ckpt-dir checkpoints/poc_a_s44 --seed 44
```

**Record:** median val_ppl at step 10k across 3 seeds → `results/poc_sprint1.md`

**Gate to proceed:** runs converge (val_ppl < 2.5 at step 10k)

---

### Sprint 2 — Mamba Backbone (exp_poc_b = exp_57)

**Goal:** Replace L1 SWA with Mamba SSM. Test the core backbone swap.

**Hypothesis:** Mamba's selective state-space dynamics give similar or better
perplexity vs SWA, with O(n) complexity at any segment length.

```bash
python scripts/train.py \
  --d-model 128 --n-heads 4 --n-layers 4 --segment-len 128 \
  --use-mamba --mamba-d-state 16 --mamba-d-conv 4 --mamba-expand 2 \
  --steps 10000 --batch-size 8 --val-every 500 \
  --log-every 100 --save-every 5000 \
  --ckpt-dir checkpoints/poc_b_s42 --seed 42
# (repeat for seeds 43, 44)
```

**Success criterion:** median val_ppl(Sprint 2) ≤ median val_ppl(Sprint 1) + 0.20
  (within 0.20 of baseline is acceptable; better than baseline is the target)

**Diagnostic:** if Mamba is ≥0.5 worse, check that log_A gradient is flowing — the
selective scan must be learning, not just passing state unchanged (D-skip over-dominates)

**If fails:** reduce mamba_d_state to 8, increase mamba_expand to 4, retry.

**Platform note:** Mamba on Apple Silicon MPS has no hardware-accelerated scan kernel →
~200 tok/s (150× slower than baseline). Run Sprint 2 on Colab or Kaggle (T4/P100)
using `notebooks/platforms/colab_drex_poc.ipynb` with `SPRINT = 2`.

---

### Sprint 2b — Transformer + ESN Episodic Memory (exp_poc_2b) — MPS-NATIVE

**Goal:** Isolate the ESN episodic memory contribution on top of the baseline
transformer *without* Mamba. Runs at full MPS speed (~15k–30k tok/s) on Apple Silicon.

**Why this sprint exists:** Sprint 2 is stalled locally due to Mamba/MPS incompatibility.
Sprint 2b provides a clean ablation (ESN vs. baseline) that runs in ~1h locally and
produces useful evidence regardless of whether Mamba is added later.

**What it measures:** Transformer L1 (SWA) + L2 (InfiniAttention) + ESN reservoir
episodic memory. The ESN reservoir itself has zero trainable parameters — only the
linear readout and write gate train. Net parameter increase is small.

**Hypothesis:** ESN working memory reduces perplexity vs. the bare transformer by
storing and retrieving recent context patterns that SWA window cannot reach.

```bash
bash scripts/run_poc_sprint2b.sh
```

Or manually:

```bash
python scripts/train.py \
  --d-model 128 --n-heads 4 --n-layers 4 --segment-len 128 \
  --use-episodic-memory --use-esn-memory \
  --esn-reservoir-mult 4 --esn-spectral-radius 0.95 \
  --steps 10000 --batch-size 8 --val-every 500 \
  --log-every 100 --save-every 5000 \
  --ckpt-dir checkpoints/poc_2b_s42 --seed 42
# (repeat for seeds 43, 44)
```

**Success criterion:** median val_ppl(Sprint 2b) ≤ 1.32 (Sprint 1 + 0.20)
  Bonus: median val_ppl(Sprint 2b) < 1.12 (strictly beats baseline)

**Diagnostic:** monitor `wr` (write rate) at each log step — must be in [0.10, 0.85].
  If wr > 0.85: decrease `--episodic-gate-thresh` from 0.70 to 0.50.
  If wr < 0.10: decrease to 0.40.

**Record:** `results/poc/sprint2b_esn.md`

---

### Sprint 3 — Mamba + ESN Episodic Memory (exp_poc_c = exp_58)

**Goal:** Add zero-training-cost associative memory on top of the Mamba backbone.

**Hypothesis:** ESN working memory provides the episodic recall that Mamba's SSM
state cannot hold at O(1) memory — combination should beat either alone.

```bash
python scripts/train.py \
  --d-model 128 --n-heads 4 --n-layers 4 --segment-len 128 \
  --use-mamba --mamba-d-state 16 --mamba-d-conv 4 --mamba-expand 2 \
  --use-episodic-memory --use-esn-memory \
  --esn-reservoir-mult 4 --esn-spectral-radius 0.95 \
  --steps 10000 --batch-size 8 --val-every 500 \
  --log-every 100 --save-every 5000 \
  --ckpt-dir checkpoints/poc_c_s42 --seed 42
# (repeat for seeds 43, 44)
```

**Success criterion:** median val_ppl(Sprint 3) < median val_ppl(Sprint 2)

**Diagnostic:** print `wr` (write rate) at each log step — must be in [0.10, 0.85].
If wr > 0.85 (over-writing), decrease `--episodic-gate-thresh` from 0.70 to 0.50.
If wr < 0.10 (under-writing), decrease to 0.40 (hard floor from exp_43_1).

**Also run** passkey eval after training:
```bash
python -m drex.eval.passkey --checkpoint checkpoints/poc_c_s42/step_0010000.safetensors \
  --max-context 1024
```

---

### Sprint 4 — Full DREX-UNIFIED Core: Mamba + ESN + HDC (exp_poc_d)

**Goal:** Add HDC encoder as the zero-training-cost input lifter.

**Hypothesis:** HDC compositional encoding gives the ESN reservoir richer structure
to write into — the three zero-cost or near-zero-cost components together form a
synergistic system.

```bash
python scripts/train.py \
  --d-model 128 --n-heads 4 --n-layers 4 --segment-len 128 \
  --use-mamba --mamba-d-state 16 --mamba-d-conv 4 --mamba-expand 2 \
  --use-episodic-memory --use-esn-memory \
  --esn-reservoir-mult 4 --esn-spectral-radius 0.95 \
  --use-hdc-encoder --hdc-dim 512 --hdc-seed 0 \
  --steps 10000 --batch-size 8 --val-every 500 \
  --log-every 100 --save-every 5000 \
  --ckpt-dir checkpoints/poc_d_s42 --seed 42
# (repeat for seeds 43, 44)
```

**Success criterion:** median val_ppl(Sprint 4) ≤ median val_ppl(Sprint 3)
  (adding HDC must not hurt, and ideally helps by ≥0.05 ppl)

**Diagnostic:** if HDC hurts, try hdc_dim=256 (closer to d_model). If still hurts,
the problem may be that hdc_dim >> d_model creates too large a readdown bottleneck —
test with hdc_dim=256 and hdc_normalize=False.

**Count trainable params:**
```bash
# Should show model with ≥50% fewer trainable params vs Sprint 1 baseline
# (ESN reservoir + HDC projections are all frozen buffers)
```

---

### Sprint 5 — Scale + Proof (exp_poc_e)

**Goal:** Take the best config from Sprints 2–4 and scale to d=256, 8-layer,
50k steps, longer context. This is the "architecture kicks ass" run.

```bash
# Use best Sprint config + increase scale:
python scripts/train.py \
  --d-model 256 --n-heads 4 --n-layers 8 --segment-len 512 \
  --use-mamba --mamba-d-state 16 --mamba-d-conv 4 --mamba-expand 2 \
  --use-episodic-memory --use-esn-memory \
  --esn-reservoir-mult 4 --esn-spectral-radius 0.95 \
  --use-hdc-encoder --hdc-dim 1024 \
  --steps 50000 --batch-size 8 --val-every 1000 \
  --log-every 200 --save-every 5000 \
  --ckpt-dir checkpoints/poc_e_s42 --seed 42
```

**Run evaluations:**
```bash
python -m drex.eval.passkey \
  --checkpoint checkpoints/poc_e_s42/step_0050000_final.safetensors \
  --max-context 4096

python -m drex.eval.babilong \
  --checkpoint checkpoints/poc_e_s42/step_0050000_final.safetensors
```

**POC success criteria (all 3 must hold):**
1. val_ppl ≤ Sprint 1 baseline (transformer + no memory) at equal step budget
2. passkey retrieval depth ≥ 2× Sprint 1 baseline (memory is doing something)
3. Model has ≤ training_cost_score of baseline (measure: track-record tok/s * param count)

**If all 3 pass:** the paper is writing itself. Update results/TRAINING_RUNS.md and
submit to arXiv. **This is the experimental proof that DREX-UNIFIED works.**

---

## Sprint Checklist & Tracking

| Sprint | Config                              | Status               | Median val_ppl | Notes |
|--------|-------------------------------------|----------------------|----------------|-------|
| 1      | Baseline transformer                | ✅ DONE (gate ✅)    | **1.12**       | s42=1.11, s43=1.48, s44=1.12 |
| 2      | + Mamba backbone                    | ⏸️ PAUSED (cloud)   | —              | MPS: ~200 tok/s (150× slower). Run on Colab/Kaggle T4. |
| 2b     | Transformer + ESN memory (no Mamba) | ✅ DONE (gate ✅)    | **1.10**       | s42=1.10, s43=1.12, s44=1.08. Bonus PASS (median < 1.12). |
| 3      | Mamba + ESN episodic memory         | 🔲 TODO (cloud)      | —              | seeds 42, 43, 44 |
| 4      | Mamba + ESN + HDC encoder           | 🔲 TODO (cloud)      | —              | seeds 42, 43, 44 |
| 5      | Scale: d=256, 8L, 50k steps         | 🔲 TODO (cloud)      | —              | seed 42 only     |

Update this table as each sprint completes.

---

### Phase 17 — Results Integration & arXiv (PENDING — code ready)

Trigger: Exp A/B 50k-step final checkpoints.

- [ ] Extract val_ppl + wr trajectory; update results/TRAINING_RUNS.md
- [ ] Run eval_passkey.py for both checkpoints
- [ ] Run eval_babilong.py for both checkpoints
- [ ] Fill all \todo{pending} in paper/main.tex
- [ ] Recompile paper (pdflatex x3 + bibtex); verify zero \todo{}
- [ ] Submit to arXiv (cs.LG + cs.CL)

---

### Phase 18 — Write-Rate Convergence (PENDING — trigger: Exp B log)

Read wr from results/exp_b_train.log at steps 5k, 10k, 20k, 30k, 50k.

Outcome table:
  - Converges (wr ∈ [0.10, 0.85] by step 30k): document, mark resolved
  - Slow convergence: log time-to-convergence, revisit α formula for L>256
  - Does not converge (wr > 0.85 at 50k): run exp_49, α warmup schedule

exp_49 (conditional): 3 seeds × 10k steps × {no warmup, linear α warmup, step α warmup}

---

### Phase 19 — Ablation Completeness (PENDING — code infrastructure ready)

Scale: d_model=256, seg_len=512, 10k steps, 3 seeds.

exp_50: Full-sequence residual at 10k steps (resolve INCONCLUSIVE from Phase 16)
exp_51: Recency weight ablation (first controlled test of w_t=(t+1)/L benefit)
exp_52: L2 vs L4 interaction (are Infini-Attention and MemoryModule complementary?)

---

### Phase 20 — Throughput Optimization (PENDING — trigger: Phases 17–19 done)

Option B (implement first): Chunked recurrence in MemoryModule.
  chunk_size=32 at L=512: 15 iterations instead of 511 (~34× fewer Python calls)
  Target: 5× throughput improvement over current 2,310 tok/s, approaching Exp A baseline

Option A (follow-on): Parallel scan (Heinsen 2023, arXiv:2311.06281)

---

### Phase 21 — Scale & Broader Evaluation (PENDING — trigger: Phase 20)

- Exp C: 512d/8L model (~18M parameters)
- Multi-dataset training: TinyStories + Wikipedia
- BABILong distractor density sweep
- Passkey recall to 32k context

---

### Phase 22 — NoProp x Ternary Validation (IN PROGRESS)

Status: Wave 0+1 complete (6/7 PASS, optimizer bug fixed).
Wave 2–3 smoke tests running.

Remaining work:
- [ ] Full convergence run for best Wave 1 variant (5k–10k steps, WikiText-2)
- [ ] Wave 2 diagnostics (grad norms, dead zone mapping, block depth sweep)
- [ ] Decision gate: if ≥1 Wave 1 variant converges, proceed to 125M scale plan
- [ ] If no Wave 1 variant converges near 10k steps, investigate gradient amplification
      (2E) and hybrid fallback (NoProp mid-layers, backprop edge layers)

---

### Phase 23 — ESN Reservoir Proof of Concept (NEW — start after Phase 22)

Goal: Validate that a fixed ESN reservoir can match or exceed the current L4
MemoryModule on the associative recall benchmark, at zero training cost.

Design:
  - Replace M_sem and M_epi (trainable delta-rule matrices) with ESN reservoirs
  - Reservoir size N = d_model × 4 (e.g., 1024 for d_model=256)
  - Connectivity: ~1% sparsity, spectral radius ρ = 0.95
  - Keep readout (linear + null gate) — this is the ONLY trained component
  - Readout training: ridge regression (one-shot) or kept as trainable Linear

Implementation target: python/drex/models/memory_esn.py
  Drop-in replacement: same interface as MemoryModule
  Flag: --use-esn-memory

Experiments:
  exp_53: ESN reservoir vs current MemoryModule (same hyperparams otherwise)
    3 seeds × 10k steps × {baseline MemoryModule, ESN variant}
    Success criterion: ESN val_ppl within +0.10 of baseline MemoryModule (≥2/3 seeds)

  exp_54: Controller feedback to reservoir
    Add output feedback: reservoir_input_t = concat(x_t, last_read_output)
    Expected: 30–60% error reduction at zero additional training cost
    3 seeds × same config

  If exp_53 SUPPORTED: ESN becomes standard. If REFUTED: document why and return to
  trained delta-rule (keeping Phase 1–16 finding, not discarding it).

---

### Phase 24 — HDC Encoder Integration (COMPLETE — 2026-03-24)

Goal: Add a fixed HDC projection layer before the main model.
Input representation switches from raw byte embeddings to HDC hypervectors.

Implementation (DONE):
  - python/drex/models/hdc_encoder.py: HDCEncoder class + hdc_bind/bundle/permute prims
    - Fixed random projection lift (d_model → hdc_dim) + readdown (hdc_dim → d_model)
    - All projection weights frozen as buffers — zero trainable parameters (only LayerNorm)
    - Training mode: tanh thresholding (differentiable). Eval mode: sign (hard bipolar)
    - Residual merge + LayerNorm output: preserves original embedding + HDC structure
    - hdc_dim must be strictly > d_model (enforced in constructor)
  - transformer.py: DrexConfig.use_hdc_encoder, hdc_dim, hdc_normalize, hdc_seed fields
    - DrexTransformer.hdc_encoder created when use_hdc_encoder=True (None otherwise)
    - Applied in forward() after embedding sum, before transformer layers
  - scripts/train.py: --use-hdc-encoder, --hdc-dim, --no-hdc-normalize, --hdc-seed flags
  - tests/python/test_hdc_encoder.py: 44 tests, 100% coverage of hdc_encoder.py
  - pyproject.toml: added pythonpath=["python"] to pytest config

Experiments (PENDING — ready to run):
  exp_55: Byte embedding baseline vs HDC encoder
    Success criterion: val_ppl maintained (within ±0.05) or improved
    Focus: does compositional structure of HDC encoding benefit downstream memory?

  exp_56: HDC controller representation
    If the DREX Controller uses HDC features directly, does routing quality improve?

Status: CODE DONE. Experiments exp_55/56 pending (trigger: Exp A/B baselines available).

---

### Phase 25 — Mamba SSM Backbone (NEW — trigger: Phase 23 validated)

Goal: Replace the current transformer attention layers with Mamba SSM layers.
This is the single highest-leverage backbone change.

Design:
  - Keep all existing memory modules (L2, L3, L4 / ESN variant)
  - Replace L1 sliding-window attention with Mamba selective SSM layer
  - Use Mamba-2 (state space duality) if available; fall back to Mamba-1 (Gu & Dao 2023)
  - Training: standard backprop first; Predictive Coding explored as follow-on

Key compatibility checks:
  - TBPTT state management: Mamba has its own hidden state — needs same boundary-reset
    logic as current LayerState
  - Gradient checkpointing: Mamba layers support this
  - Segment length: Mamba is O(n) in practice at any segment length; no L=512 cost cliff

Experiments:
  exp_57: Mamba backbone vs transformer baseline (Exp A equivalent with Mamba)
    Goal: match Exp A val_ppl at equal or lower compute budget

  exp_58: Mamba + ESN episodic memory (Mamba replacing transformer in Exp B)
    Goal: match or beat Exp B val_ppl at significantly higher throughput

---

### Phase 26 — RL Controller (NEW — trigger: Phase 25 validated)

Goal: Replace the differentiable write gate with a small RL policy.

Design:
  - Controller input: concatenation of Mamba state + HDC hypervector of last input
  - Controller output: discrete actions (write to ESN, read from ESN, read from semantic,
    activate module k, suppress module k)
  - Training: REINFORCE with reward = improvement in next-token prediction accuracy
  - Controller architecture: 2-layer MLP with tanh activations, hidden dim 128

Why this is different from exp_7_1 failure:
  exp_7_1 found that REINFORCE fails when the write gate is positioned between the
  encoder and the loss — the encoder gradient becomes zero because the gate blocks
  the backprop signal. In DREX-UNIFIED, the controller does NOT sit in the gradient
  path. It receives detached representations and acts via RL reward, not backprop.
  The failure mode from exp_7_1 is structurally absent.

Experiments:
  exp_59: RL controller vs fixed write policy baseline
    Fixed baseline: always write (wr=1.0) vs RL policy
    Success criterion: RL policy achieves similar or better recall at <40% write rate

---

### Phase 27 — NoProp Semantic Tier (NEW — trigger: Phase 22 full convergence + Phase 25)

Goal: Replace the L3 Titans-style MLP (Adam gradient step training) with a NoProp-trained
SSM block as the semantic memory tier.

Design:
  - Small SSM (2–4 layers, d=128) as semantic memory
  - Each block trains via local denoising objective (NoProp-DT, validated Phase 22)
  - Block optimizers own only block-specific params (fix from Phase 22 applies directly)
  - Shared head optimizer updated once per global step (Phase 22 fix)
  - Updates parameters during inference for continual learning (zero catastrophic forgetting)

Experiments:
  exp_60: NoProp semantic tier vs Adam-trained MLP at same capacity
    Success criterion: val_ppl within ±0.15 ppl using NoProp vs Adam (≥2/3 seeds)
    Training speed target: NoProp blocks must be ≥2× faster to train per step

---

### Phase 28 — KAN Readout (NEW — trigger: Phase 25)

Goal: Replace the final linear output projection with a KAN layer.

Design:
  - 2-layer KAN replacing the linear readout (out_proj in MemoryModule + lm_head)
  - Spline degree: 3 (cubic). Grid size: 5 knots.
  - Training: standard autograd (KAN gradients are well-behaved)

Why KAN pairs with the ESN: The ESN reservoir output is a high-dimensional state vector
that may contain interpretable geometric structure. KAN can learn to extract it via
learnable spline functions rather than a fixed dot product. The result is auditable.

Experiments:
  exp_61: KAN readout vs linear readout
    Success criterion: val_ppl maintained (within ±0.05) or improved at ≤2× parameter count

---

### Phase 29 — Sparse Execution Integration (NEW — trigger: Phase 26)

Goal: Wire the RL controller to enable conditional module execution.
Only the modules relevant to the current input activate.

Design:
  - Controller outputs a bitmask: {read_esn, read_episodic, read_semantic, activate_ffn_k}
  - Inactive modules: forward pass skipped entirely (torch.zeros fallback)
  - Load balancing auxiliary loss: push controller toward uniform module utilization
  - Target: 30–50% of modules active per step on average

Experiments:
  exp_62: Sparse execution at 50% activation rate vs always-on baseline
    Success criterion: ppl within ±0.10 at 50% projected compute cost

---

### Phase 30 — DREX-UNIFIED Full Benchmark (NEW — trigger: Phases 23–29)

Goal: End-to-end benchmark of DREX-UNIFIED vs:
  - Baseline transformer (Exp A equivalent)
  - Current DREX (Exp B, Phase 17 results)
  - Published comparators: Titans (Behrouz et al. 2025), Mamba-pure, RWKV

Evaluation suite:
  - Passkey recall: 512/1k/2k/4k/8k/16k/32k context lengths
  - BABILong: Tasks 1–5, 2k/4k/8k context
  - TinyStories: val_ppl at convergence
  - Training cost comparison: tok/s, total GPU hours, peak memory

---

## Part 7 — Decision Tree

Some phases are conditional. Here is the gating logic:

Phase 23 (ESN reservoir) SUPPORTED:
  → ESN becomes the standard episodic memory. Phase 24+ use ESN.
Phase 23 REFUTED:
  → Keep trained delta-rule. Document the specific failure mode.
  → Phase 24 can still proceed with HDC encoder (does not depend on ESN).

Phase 25 (Mamba backbone) SUPPORTED:
  → Replace transformer backbone. All subsequent phases use Mamba.
Phase 25 REFUTED:
  → Use RWKV as fallback (same O(1) inference, different architecture).
  → If RWKV also fails: keep transformer backbone and focus on memory/training
    improvements only. DREX becomes a memory-augmented transformer paper, not
    a full post-transformer replacement.

Phase 26 (RL controller) SUPPORTED:
  → Phases 28 (KAN) and 29 (sparse execution) activate.
Phase 26 REFUTED:
  → Maintain fixed write policies from Phase 1–16.
  → Sparse execution still possible with simpler top-k gating (no learned controller).

Phase 22 (NoProp) full convergence SUPPORTED:
  → Phase 27 (NoProp semantic tier) activates.
Phase 22 REFUTED:
  → Use gradient-isolated local contrastive loss as alternative semantic tier training.
  → The learned delta-rule remains as fallback (Phase 1–16 validated).

---

## Part 8 — Open Questions Driving Future Research

1. Does ESN reservoir match trained delta-rule on associative recall?
   This is the central question. Current evidence says similar quality on BabyLM-scale
   language tasks (2025). Whether it holds in the DREX context (combined with attention,
   delta rule L2, controller) is not known. Phase 23 answers this directly.

2. Does the write gate multi-stability disappear with RL controller?
   Phase 7 finding is the key motivator. The hypothesis: moving from a differentiable
   gate to a discrete RL policy removes the multi-stable loss landscape. Phase 26
   answers this directly. If it does disappear, Phase 7 is publishable as a finding
   in its own right — the first controlled characterization of write gate instability
   in associative memory networks.

3. Does NoProp scale to the semantic tier?
   Phase 22 validates NoProp at 6-layer, d=256, WikiText-2 scale. Phase 27 asks
   whether it holds for a smaller 2–4 layer semantic SSM. The evidence so far
   (Wave 1 convergence after optimizer fix) is positive.

4. What is the actual throughput of DREX-UNIFIED on M3?
   Current Exp B: 2,310 tok/s (after Phase 16 CPU backend fix).
   With chunked recurrence (Phase 20): projected ~10,000 tok/s.
   With Mamba backbone (Phase 25): measured data needed; theoretical ceiling is
   much higher due to elimination of O(L) write loop and O(n^2) attention.

5. Can the reservoir be designed rather than randomly initialized?
   Deep reservoir computing literature shows spectral properties can be tuned to
   match task memory requirements. For DREX, the episode length distribution
   (typical TinyStories story ≈ 500–2000 tokens) determines the optimal τ/L.
   A designed reservoir with τ/L ≈ 0.21 (same calibration as the Phase 11 EMA fix)
   is the hypothesis.

6. Is DREX-UNIFIED publishable before it beats transformers at scale?
   Yes. The architectural decisions — ESN for working memory, RL controller,
   NoProp semantic tier — each individually constitute research contributions.
   The Phase 7 write gate stability finding is publishable standalone. The Phase 11
   EMA bootstrap finding is in the current paper. DREX-UNIFIED can be a second paper
   ("DREX-UNIFIED: component-by-component construction of a post-transformer hybrid")
   submitted after the current arXiv paper is out.

---

## Part 9 — Practical Sequence for March–May 2026

### March 2026 (current state)
- Let Exp A/B continue to completion (don't interrupt)
- Continue NoProp Wave 2–3 diagnostics
- Begin Phase 23 design spec and ESN implementation (memory_esn.py)
  → Can be done in parallel with Exp A/B running — different code path

### April 2026
- Phase 22 full convergence runs (winning variant, 5k–10k steps)
- Phase 23 exp_53/54 (ESN proof of concept)
- Phase 24 HDC encoder (simple version, 1–2 days implementation)
- Phase 17 results integration (when Exp A/B checkpoints are ready)
- arXiv submission

### May 2026
- Phase 25 Mamba backbone (most complex, allow 2–3 weeks)
- Phase 19 ablation completeness (can run in background)
- Phase 20 chunked recurrence (needed before Mamba benchmarking)
- Begin Phase 26 RL controller design

### June 2026 onward
- Phases 26–30 (RL controller, NoProp semantic tier, KAN, sparse, full benchmark)
- DREX-UNIFIED paper first draft

---

## Part 10 — Files to Create

When implementing, create these files in this order:

1. python/drex/models/memory_esn.py     (Phase 23 — ESN reservoir module)
2. python/drex/models/encoder_hdc.py    (Phase 24 — HDC encoder)
3. tests/python/test_memory_esn.py      (Phase 23 — must have 100% coverage)
4. tests/python/test_encoder_hdc.py     (Phase 24 — must have 100% coverage)
5. python/drex/models/backbone_mamba.py (Phase 25 — Mamba backbone wrapper)
6. python/drex/models/controller_rl.py  (Phase 26 — RL controller)
7. python/drex/models/memory_noprop.py  (Phase 27 — NoProp semantic tier)
8. python/drex/models/readout_kan.py    (Phase 28 — KAN readout)
9. python/drex/models/router_sparse.py  (Phase 29 — sparse execution router)
10. python/drex/models/drex_unified.py  (Phase 30 — full DREX-UNIFIED assembly)

---

## Part 11 — Non-Negotiable Implementation Rules (carry forward from PLAN.md)

These constraints from Phase 1–16 research must propagate into all future phases:

1. gate_thresh >= 0.40 wherever a norm-based write gate is used (exp_43_1)
2. α(L) = 0.95^(96/L) for EMA decay in any delta-rule memory (exp_47_1/3)
3. Fixed 50/50 episodic/semantic split — no learned router (exp_38_1)
4. Adam (not SGD) for any trained components (exp_34_6)
5. F.normalize(k, dim=-1, eps=1e-6) — not the default eps=1e-12 (Phase 15 NaN bug)
6. NaN loss guard before backward pass (Phase 15, esp. small models)
7. No shared optimizers across block-local and global parameters in NoProp
   (Phase 22 optimizer bug — block opts own only block-specific params)
8. validate write_rate ∈ [0.10, 0.85] after any change to write mechanism (all phases)
9. TBPTT document-boundary contamination: use reset_on_boundary for streaming data
   (Phase 15)

---

*This document should be updated after each phase completes. Phases 17–22 update
PLAN.md. Phases 23+ update this document. When a phase moves from PENDING to COMPLETE,
move its checklist items to ARCHITECTURE_FINDINGS.md under a new section.*
