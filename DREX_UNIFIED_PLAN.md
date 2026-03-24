# DREX-UNIFIED PLAN
# Architecture Evolution & Forward Research Roadmap

*Created: 2026-03-24 | Status: Active planning document*
*Synthesizes: Phases 1–22 findings + architectural research from March 2026 sessions*

---

## Part 1 — Where We Are (State of the Repo)

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

### Current priority: Complete Phases 17–22 without interruption.

Reason: Exp A/B produce the first published benchmark for the current architecture.
That benchmark is the baseline against which DREX-UNIFIED will be measured.
NoProp Phase 22 validates the training method for the semantic tier.
Do not fork or abandon Exp A/B — let them run to completion.

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
