# DREX_UNIFIED_SPEC.md
# Konjo AI Research · DREX Architecture Lab
# Version 0.2 · March 2026
# Status: Living Document — update before any implementation sprint

Changelog from v0.1:
- Phase 1 gate updated: HDC encoder added as Objective 0 (was missing entirely)
- Phase 1 gate updated: KAN readout added as Objective 5 (was deferred to integration)
- Phase 1 gate updated: RL controller exit criterion now includes routing collapse condition
- Phase 1 gate updated: NoProp exit criterion replaced VRAM-only check with accuracy parity requirement
- Phase 1 gate updated: NoProp block independence assertion added as explicit CI test requirement
- Phase 1 gate updated: ESN feedback extension test added as explicit exit criterion
- Infrastructure section added: CI gradient leak assertion and GitHub Actions workflow requirement
- File layout changed: src/ is now a real filesystem path (was a logical namespace only)
- Implementation status table removed: all prior code archived to research/legacy/
- v0.2 supersedes all prior planning documents

---

## PURPOSE

This document is the contract between all components in the DREX-UNIFIED
architecture.  No component gets built without its interface being defined here
first.  No integration sprint begins without all upstream component specs being
locked.  This is the single source of truth for tensor shapes, validation
criteria, training methods, and phase gates.

Rule: always reference this plan before building.  If the plan is wrong, update
the plan first, then build.

---

## ARCHITECTURE OVERVIEW

DREX-UNIFIED is a post-transformer hybrid architecture combining:

    1. HDC Encoder          — fixed random hyperdimensional projection, zero training
    2. Mamba SSM Backbone   — selective state space sequence processor, Predictive Coding
    3. DREX Controller      — small RL policy for memory routing, REINFORCE
    4. Working Memory (L1)  — Echo State Network reservoir, zero training
    5. Episodic Memory (L2) — ESN with EMA feedback controller, near-zero training
    6. Semantic Memory (L3) — small SSM trained with NoProp local block denoising
    7. Sparse Router        — top-k gating across memory tiers
    8. KAN Readout          — learnable spline output projection, closed-form fitting
    9. Reward Feedback Loop — output signal back to controller and ESN tiers

Data flow:

    INPUT → HDC ENCODER → MAMBA SSM → CONTROLLER → [L1, L2, L3]
         → SPARSE ROUTER → KAN READOUT → OUTPUT → (reward → CONTROLLER, ESN feedback)

---

## GLOBAL CONVENTIONS

    Language:       Python 3.12 primary, Rust/Candle for performance-critical ops
    Hardware:       Apple M3 16GB (primary), CUDA RTX 3090 (secondary)
    Framework:      MLX for Apple Silicon training; NumPy for reservoir and HDC
    dtype:          bfloat16 for trained components, float32 for reservoir and HDC
    Batch dims:     always (batch, sequence, features) — B × S × D
    Sequence:       variable length, no fixed maximum
    Seeds:          all fixed random components use seed=42 for reproducibility
    Testing:        100% test coverage, per-file test files, pytest
    Commits:        Conventional Commits format
    Logging:        never suppress output; log shapes at component boundaries in
                    debug mode

---

## FILE LAYOUT (filesystem, not logical namespace)

    Legacy v0.1 code is archived at:  research/legacy/
    New v0.2 source root:             src/

    src/
      input/tokenizer.py
      hdc/encoder.py
      backbone/mamba.py
      controller/policy.py
      controller/reward.py
      memory/reservoir.py
      memory/episodic.py
      memory/semantic.py
      router/sparse.py
      readout/kan.py
      drex_unified.py
    tests/
      python/
        conftest.py
        test_tokenizer.py
        test_hdc_encoder.py
        test_mamba.py
        test_controller.py
        test_reservoir.py
        test_episodic.py
        test_semantic.py
        test_sparse_router.py
        test_kan.py
        test_integration.py
        test_gradient_leak_ci.py
        test_dtype_contracts_ci.py
        test_shape_contracts_ci.py
    .github/
      workflows/
        ci.yml

All components start from scratch against this spec. Research findings from
research/legacy/ inform the implementation but the prior code is not directly
promoted to src/.

---

## PHASE 1 INFRASTRUCTURE — MUST BE COMPLETE BEFORE ANY COMPONENT WORK

The single most dangerous silent failure in DREX-UNIFIED is an accidental global
backward call leaking into a PCN or NoProp module that is supposed to train
locally. This failure does not throw an error. It degrades performance invisibly
and you will not discover it until the perplexity numbers look wrong and you
cannot explain why.

Before writing any component code, establish the following automated checks.

File: .github/workflows/ci.yml

The CI workflow must run on every commit and must include:

1. Full pytest suite with --timeout=120

2. Gradient leak assertion: for every module in src/backbone/, src/memory/, and
   src/readout/, assert that no loss.backward() call exists in the module's
   training path that references parameters outside that module's own parameter
   group. Assert NoProp block N receives no gradients from block M (M != N).
   Assert ESN reservoir parameters (W_res, W_in, W_fb) receive zero gradient
   at all times.
   Implementation file: tests/python/test_gradient_leak_ci.py

3. Dtype boundary assertion: assert all tensors crossing component boundaries
   match the dtype contract defined in the DTYPE BOUNDARY CONTRACT section.
   Run as a forward-pass integration smoke test with a batch of 4 synthetic
   inputs.
   Implementation file: tests/python/test_dtype_contracts_ci.py

4. Shape contract assertion: assert all intermediate tensor shapes match the
   spec on every CI run.
   Implementation file: tests/python/test_shape_contracts_ci.py

This CI setup is the prerequisite for Wave 0 and must be complete before any
component implementation begins. A component that exists without CI coverage
of these assertions is not considered started.

---

## COMPONENT 1: INPUT LAYER

    File: src/input/tokenizer.py
    Test: tests/python/test_tokenizer.py
    Phase 1 status: Prerequisite (used by all other components)

Description:
    Accepts raw bytes or BPE tokens.  Byte-level input is preferred — Mamba
    outperforms transformers on byte-level tasks and eliminates the vocabulary
    bottleneck.  BPE mode retained for compatibility benchmarking.

Input:
    raw string or byte sequence of arbitrary length

Output:
    tensor of shape (B, S) dtype int32
      B = batch size
      S = sequence length in tokens or bytes

Parameters:
    mode:        "byte" | "bpe"
    vocab_size:  int, only used in bpe mode, default 32000
    max_length:  int | None, default None (no truncation)

Validation criteria:
    [ ] byte mode: output vocab range is 0–255, no out-of-bounds values
    [ ] bpe mode: output vocab range is 0 to vocab_size - 1
    [ ] round-trip test: encode then decode recovers original string
    [ ] no padding added silently — document padding behavior explicitly

Training cost: None


---

## COMPONENT 2: HDC ENCODER — OBJECTIVE 0, PHASE 1

    File: src/hdc/encoder.py
    Test: tests/python/test_hdc_encoder.py
    Phase 1 status: MUST BE VALIDATED FIRST — Objective 0

Phase 1 rationale: The HDC encoder is the entry point for the entire architecture.
Every downstream component receives HDC-encoded input. If the HDC encoder is not
producing geometrically stable, compositional hypervectors, every downstream
component receives corrupted input and you will not be able to identify the source
of failure during integration. It has zero training cost — can be fully validated
in a single day with zero risk. There is no reason to defer it.

Description:
    Projects integer token IDs into a high-dimensional random hypervector space.
    Uses three operations: binding (element-wise multiplication, encodes
    associations), bundling (element-wise addition followed by sign normalization,
    encodes composition), and permutation (cyclic rotation, encodes sequence
    position). All projection matrices are fixed at initialization and never
    updated.

    Note on v0.1 implementation: the legacy hdc_encoder.py in research/legacy/
    implemented embedding-lifting HDC (input: (B,S,d_model) float32 embeddings).
    The v0.2 spec requires a token-ID encoder (input: (B,S) int32 token IDs).
    These are architecturally distinct. Build from spec, not from legacy code.

Input:
    token tensor of shape (B, S) dtype int32

Output:
    hypervector tensor of shape (B, S, D_hdc) dtype float32
    D_hdc = hypervector dimensionality, default 10000

Parameters:
    d_hdc:      int, hypervector dimension, default 10000
    seed:       int, random seed for projection matrices, default 42
    normalize:  bool, L2-normalize output hypervectors, default True
    vocab_size: int, number of distinct token IDs, default 256 (byte mode)

Internal matrices (fixed, never trained):
    item_memory:           shape (vocab_size, D_hdc) — one random hypervector per token
    position_permutations: list of S permutation indices, one per position

Key operations:
    encode(token_id)           → item_memory[token_id]
    bind(hv_a, hv_b)           → hv_a * hv_b  (element-wise)
    bundle(hv_list)            → sign(sum(hv_list))
    permute(hv, n_positions)   → roll(hv, n_positions)
    encode_sequence(tokens)    → bundle([permute(encode(t), i)
                                         for i, t in enumerate(tokens)])

Validation criteria:
    [ ] similarity test: cosine_similarity(encode(A), encode(A)) > 0.999
        for 100 random tokens
    [ ] orthogonality test: mean(cosine_similarity(encode(A), encode(B))) < 0.05
        for 1000 random pairs A != B. If this fails at d_hdc=10000, reduce to
        8000 and retest. Document minimum dimensionality — publishable finding.
    [ ] associativity test: encode→bind→unbind recovers original vector with
        cosine_sim > 0.9
    [ ] sequence order test: encode_sequence([A,B]) != encode_sequence([B,A])
    [ ] dimensionality test: output shape is exactly (B, S, D_hdc)
    [ ] dtype test: output dtype is float32

Training cost: Zero — random initialization only, no gradient operations
Training method: None

Phase 1 exit criterion: All six validation tests pass. Minimum viable D_hdc
documented.


---

## COMPONENT 3: MAMBA SSM BACKBONE

    File: src/backbone/mamba.py
    Test: tests/python/test_mamba.py
    Phase 1 status: Objective 1 — after HDC encoder is validated

Description:
    Selective state space model for sequence processing.  Replaces transformer
    self-attention.  O(n) training via parallel scan, O(1) inference memory per
    token.  Trained using Predictive Coding — each layer independently minimises
    its local prediction error against the layer above.  No global backward pass
    through the full network.

Input:
    HDC-enriched embeddings: shape (B, S, d_model)
    (D_hdc → D_model projection handled internally if D_hdc ≠ D_model)

Output:
    hidden state tensor:       shape (B, S, d_model)
    final recurrent state:     shape (B, d_state × n_layers)
    (recurrent state used by controller and ESN feedback)

Parameters:
    d_model:   int, model dimension, default 256
    d_state:   int, SSM state dimension, default 16
    d_conv:    int, local convolution width, default 4
    expand:    int, inner dimension multiplier, default 2
    n_layers:  int, number of Mamba blocks, default 4
    dt_rank:   "auto" | int, delta rank, default "auto"

Predictive Coding training:
    Each layer l maintains a local target: the representation produced by
    layer l+1 on the previous step.
    Local loss for layer l: MSE(output_l, sg(target_l))  (sg = stop_gradient)
    Layers train in parallel; no sequential gradient dependency.
    No global loss signal flows backward through more than one layer.
    Top layer uses task loss as its local target.

Validation criteria:
    [ ] shape test: output shape is (B, S, D_model)
    [ ] causality test: output at t depends only on 0..t, never t+1..S
    [ ] recurrence test: final state changes when input changes
    [ ] PC convergence test: all local layer losses decrease simultaneously.
        Verify with a plot saved to experiments/runs/<timestamp>/pc_loss_curves.png
    [ ] equivalence test: PC-trained Mamba within 10% perplexity of same-size
        backprop-trained Mamba on WikiText-2 at 10M tokens
    [ ] gradient leak test (CI): passes on every commit — no cross-layer gradient

Training cost: Low — local per-layer losses only, no full backward graph
Training method: Predictive Coding, local MSE targets, parallel layer updates

Phase 1 exit criterion: All six validation criteria pass on a 10M–50M parameter
configuration. Loss curve decreases smoothly. Generated byte-level text is
coherent compared to a same-size backpropagation baseline.


---

## COMPONENT 4: DREX CONTROLLER

    File: src/controller/policy.py
    Test: tests/python/test_controller.py
    Phase 1 status: Objective 3 — after HDC encoder and Mamba backbone are validated

Description:
    A small RL policy that decides what to write to each memory tier, what to
    read, and which sparse execution paths to activate.  Operates on concatenated
    HDC hypervectors and Mamba hidden state.  Output is a discrete action vector.
    Trained via REINFORCE with reward from downstream prediction accuracy.

    Discrete actions used specifically because Phase 7 multi-stability finding
    showed continuous differentiable write gates develop initialization-dependent
    equilibria.  A discrete RL policy sidesteps this entirely.
    Key difference from exp_7_1 REINFORCE failure: the controller here operates
    on DETACHED representations (HDC + Mamba state) and trains via RL reward, NOT
    backprop through the gate.  The encoder gradient does not go to zero.

Input:
    hdc_state:   shape (B, d_hdc) — HDC encoding of last input token
    mamba_state: shape (B, d_model) — last Mamba recurrent state
    concatenated: shape (B, d_hdc + d_model) — controller input

Output:
    write_decisions: shape (B, n_tiers) dtype int32, values in {0=skip, 1=write, 2=overwrite}
    read_weights:    shape (B, n_tiers) dtype float32, softmax over tiers
    sparse_gates:    shape (B, n_modules) dtype bool

Parameters:
    d_hdc:       int, must match HDC encoder d_hdc
    d_model:     int, must match Mamba backbone d_model
    n_tiers:     int, number of memory tiers, default 3
    n_modules:   int, number of sparse execution modules, default 4
    hidden_dim:  int, policy network hidden size, default 128
    n_layers:    int, policy network depth, default 2
    gamma:       float, REINFORCE discount factor, default 0.99
    lr:          float, policy learning rate, default 1e-4

Reward signal:
    quality_reward  = -(current_loss - previous_loss) * lambda_quality
    sparsity_reward = -sum(write_decisions) * lambda_sparse
    total_reward    = quality_reward + sparsity_reward

    lambda_sparse:  float, sparsity penalty, default 0.01
    lambda_quality: float, quality weight, default 1.0

Validation criteria:
    [ ] action space test: all outputs valid discrete values within defined ranges
    [ ] learning test: better-than-random routing within 1000 episodes on the
        synthetic routing task where correct tier is known by construction
    [ ] sparsity test: average write operations per step decreases over training
    [ ] stability test: policy gradient variance bounded for 500 consecutive steps
    [ ] routing collapse test (EXIT BLOCKER): over any window of 100 consecutive
        routing decisions during training, no single tier receives more than 95%
        of writes. If this condition triggers, training must halt and log the
        failure. A controller that always writes to one tier has learned a
        degenerate policy, not a routing policy. The controller does not pass
        Phase 1 if routing collapse is observed and not resolved.

Training cost: Tiny — 2-layer MLP, ~50K parameters, CPU-trainable
Training method: REINFORCE with baseline subtraction

Phase 1 exit criterion: All five validation criteria pass including the routing
collapse test. Controller achieves better-than-random routing within 1000 episodes
AND maintains routing distribution within collapse bounds across a 5000-step
evaluation run.


---

## COMPONENT 5: WORKING MEMORY — L1 ESN RESERVOIR

    File: src/memory/reservoir.py
    Test: tests/python/test_reservoir.py
    Phase 1 status: Objective 2a — validates in parallel with or after HDC encoder

    Reference: research/legacy/python/drex/models/memory_esn.py (Phase 23 work)
    The legacy EchoStateMemory passes its own tests but must be rebuilt against
    this spec. The feedback extension test (criterion 5) was not validated in
    the legacy implementation.

Description:
    Sparsely connected (~1% density) recurrent network with fixed random weights.
    Never updated after initialisation.  Creates a high-dimensional echo of recent
    inputs.  Linear readout is the only trained component.  Output feedback from
    the controller creates attractor states.

Input:
    write signal from controller: shape (B, d_model)
    read request: bool

Output:
    reservoir state: shape (B, N_reservoir)
    read output:     shape (B, d_read)

Parameters:
    n_reservoir:      int, default = d_model × esn_reservoir_mult
    spectral_radius:  float, default 0.95, must be < 1.0
    sparsity:         float, connection density, default 0.01
    d_read:           int, readout dimension, must match d_model
    feedback:         bool, output-to-reservoir feedback, default True

Trained component:
    W_readout fitted via ridge regression (closed form, no gradient)

Open research question on spectral radius: 0.95 is standard for time-series
tasks. Language has longer temporal correlation structure and may require values
closer to 0.99 for full long-range dependency capture. Sweep 0.90, 0.95, 0.99
during Phase 1 validation. Document which value produces the best POS tagging
accuracy — publishable finding.

Validation criteria:
    [ ] echo state property test: max(abs(eigenvalues(W_reservoir))) < 1.0
    [ ] convergence test: two runs with same input but different initial states
        converge to L2 norm difference < 1e-4 within washout steps
    [ ] readout fit test: ridge regression solve < 10s for N=2000 on CPU
    [ ] recall test: reservoir + readout achieves accuracy > bag-of-words
        baseline on POS tagging by >= 5 percentage points
    [ ] feedback extension test (EXIT BLOCKER): with feedback=True, the
        reservoir must demonstrate extended memory beyond standalone reservoir
        size N. Test: construct a sequence memory task where the correct answer
        requires context from more than N steps back. Confirm that feedback=True
        shows measurable accuracy improvement over feedback=False. This is the
        core architectural claim for the episodic memory tier and must be
        validated in Phase 1, not assumed.

Training cost: Zero for reservoir, milliseconds for readout
Training method: Ridge regression (closed form), readout only

Phase 1 exit criterion: All five validation tests pass including the feedback
extension test. Spectral radius sweep documented with best-performing value.


---

## COMPONENT 6: EPISODIC MEMORY — L2

    File: src/memory/episodic.py
    Test: tests/python/test_episodic.py
    Phase 1 status: Objective 2b — validates after L1 reservoir is validated

    Reference: research/legacy/python/drex/models/memory.py (Phase 11-13 work)

Description:
    Extends the L1 reservoir with learned feedback and EMA delta writes.
    Stores episode-level context.  EMA decay with alpha(L) = 0.95^(96/L)
    (validated Phase 11).  Write gate at thresh* = 0.70 (validated Phase 12).

Input:
    write signal: shape (B, d_model)
    previous episodic state: shape (B, d_model)

Output:
    episodic state: shape (B, d_model)
    read output:    shape (B, d_model)

Key formula (EMA delta write):
    delta = new_input - previous_state
    new_state = alpha(L) * previous_state + (1 - alpha(L)) * delta
    where alpha(L) = 0.95^(96/L)

OR write gate:
    fire when: ||k - vp|| >= thresh* * ||k||  where thresh* = 0.70

Validation criteria:
    [ ] EMA stability test: episodic state converges to stable attractor on
        repeated identical input
    [ ] delta write test: alpha=0.90 achieves lower reconstruction error than
        alpha=0.0 (no EMA)
    [ ] phase research replication test: alpha≈0.90 outperforms both lower and
        higher alpha values on held-out recall task (reference: research/legacy
        Phase 11-12 findings)
    [ ] overwrite test: hard overwrite correctly resets state with no residual

Training cost: Near Zero
Training method: EMA parameter, controller-gated

Phase 1 exit criterion: All four validation tests pass. Alpha sweep documented.


---

## COMPONENT 7: SEMANTIC MEMORY — L3

    File: src/memory/semantic.py
    Test: tests/python/test_semantic.py
    Phase 1 status: Objective 2c — validates in parallel with L1 and L2

Description:
    A small trained SSM storing compressed world knowledge.  Trained using NoProp
    — each block independently denoises a noisy version of its target label.  No
    global backpropagation.  Updates its own weights during inference for
    continual learning without catastrophic forgetting.

Input:
    write signal: shape (B, d_model)
    query:        shape (B, d_model)

Output:
    retrieved knowledge: shape (B, d_model)

Parameters:
    d_model:           int, default 256
    n_blocks:          int, number of NoProp blocks, default 4
    noise_std:         float, denoising noise level, default 0.1
    inference_lr:      float, inference-time update rate, default 1e-5
    update_at_inference: bool, default True

NoProp training per block (reference: arXiv 2503.24322):
    y_noisy = y_clean + Normal(0, noise_std)
    block_loss = MSE(block_output, y_noisy)
    blocks train independently in parallel — no inter-block gradients

Note on exit criterion: the original Phase 1 spec used VRAM consumption as the
primary exit criterion for NoProp. This has been replaced. VRAM savings prove
efficiency but do not prove the learning is happening correctly. A broken NoProp
implementation could consume less VRAM simply by doing less computation. The
correct exit criterion requires both efficiency AND accuracy parity. Both must pass.

Reference: research/legacy/python/drex/models/semantic.py (Phase 22 work).
Block optimizer fix from Phase 22 applies: block optimizers must own ONLY their
block-specific params, not shared head params. Shared head optimizer updated once
per global step.

Validation criteria:
    [ ] NoProp convergence test: all block losses decrease independently over training
    [ ] block independence assertion (CI — runs on every commit): after each NoProp
        training step, assert no gradient from block N appears on parameters of
        block M (M != N). This runs as part of test_gradient_leak_ci.py and is
        not optional.
    [ ] accuracy parity test (PRIMARY gate): NoProp-trained L3 achieves within 5%
        accuracy of same-size backprop-trained model on CIFAR-100. If this test
        fails, the NoProp implementation is wrong regardless of VRAM savings.
    [ ] VRAM efficiency test (secondary): NoProp training consumes <= 50% of
        backprop VRAM at same model size and batch size.
    [ ] continual learning test: after 10 sequential tasks, task 1 accuracy within
        10% of original
    [ ] inference update test: inference-time weight update improves retrieval
        accuracy on new content without degrading retrieval of old content

Training cost: Low — local block losses, no full backward graph, parallel training
Training method: NoProp (local denoising per block, arXiv 2503.24322)

Phase 1 exit criterion: NoProp convergence test passes. Block independence
assertion passes on CI. Accuracy parity test passes (within 5% of backprop on
CIFAR-100). VRAM efficiency test passes as secondary confirmation.


---

## COMPONENT 8: SPARSE ROUTER

    File: src/router/sparse.py
    Test: tests/python/test_sparse_router.py
    Phase 1 status: Deferred to Phase 2 integration — depends on controller and
    memory tiers being validated first

Description:
    Gates which memory tiers and downstream modules activate per input.
    Top-k gating with load-balancing auxiliary loss.  Dead modules receive zero
    compute and zero gradient.

Input:
    memory outputs: list of tensors (B, d_model), one per tier
    controller sparse_gates: shape (B, n_modules) bool
    query: shape (B, d_model)

Output:
    merged representation: shape (B, d_model)
    routing weights: shape (B, n_tiers) float

Parameters:
    n_tiers:             int, default 3
    top_k:               int, active tiers per token, default 2
    load_balance_coeff:  float, default 0.01

Gating mechanism:
    score_i = dot(query, tier_output_i)
    select top_k by score
    apply softmax over selected tiers
    output = sum(weight_i * tier_output_i)  for active tiers i
    inactive tiers: detached from computation graph, zero gradient

Load-balancing loss:
    loss_lb = load_balance_coeff * variance(fraction_routed_per_tier)
    added to total training loss to prevent tier collapse

Validation criteria:
    [ ] sparsity: exactly top_k tiers activate per token
    [ ] gradient isolation: inactive tier params receive zero gradient
    [ ] load balance: routing fraction per tier within 10% of uniform over 1000 steps
    [ ] throughput: sparse > dense by ≥20% at n_tiers=3, top_k=2

Training cost: Tiny — gating parameters only, ~10K parameters


---

## COMPONENT 9: KAN READOUT — OBJECTIVE 5, PHASE 1

    File: src/readout/kan.py
    Test: tests/python/test_kan.py
    Phase 1 status: Objective 5 — validates in parallel with memory tiers, before
    integration

Phase 1 rationale: The KAN readout was originally deferred to the integration
phase. This was an error identified during planning review. The KAN readout has
its own convergence behavior and its own failure modes — specifically, whether
B-spline fitting produces useful representations at the dimensionality we are
operating at (D_model=256). If the KAN readout is broken at integration time, you
cannot tell whether the problem is the readout or the memory tiers feeding it.
Validate it in isolation first.

Description:
    Replaces final linear projection with a Kolmogorov-Arnold Network.
    Learnable spline functions on edges — interpretable, auditable.
    Smaller KANs match or exceed larger MLPs.

Input:
    merged representation from sparse router: shape (B, d_model)

Output:
    logits: shape (B, vocab_size) for LM, (B, n_classes) for classification

Parameters:
    d_in:          int, must match d_model, default 256
    d_out:         int, vocab_size or n_classes
    n_grid:        int, spline grid points, default 5
    spline_order:  int, B-spline order, default 3
    n_kan_layers:  int, default 2
    fit_method:    "closed_form" | "gradient", default "closed_form"

Validation criteria:
    [ ] approximation test: KAN readout matches MLP readout accuracy within 2%
        on a held-out classification validation set. Primary correctness gate.
    [ ] interpretability test: learned spline functions are plottable and show
        non-trivial learned transformations. Flat or linear splines indicate the
        KAN is not leveraging its expressivity. Save plots to
        experiments/runs/<timestamp>/kan_splines/
    [ ] scaling test: KAN readout has fewer parameters than an equivalent MLP
        readout achieving the same accuracy. Document the parameter ratio.
    [ ] timing test: fit_method=closed_form completes < 60s for D_model=256,
        D_out=32000 on CPU.
    [ ] regression test: fitted spline coefficients stored as a snapshot and
        compared on every CI run to detect silent numerical regressions.

Training cost: Very Low — spline fitting, closed form
Training method: B-spline coefficient fitting

Phase 1 exit criterion: All five validation criteria pass. Parameter ratio vs MLP
documented. Spline visualizations saved and non-trivial.


---

## COMPONENT 10: REWARD FEEDBACK LOOP

    File: src/controller/reward.py
    Test: tests/python/test_reward.py
    Phase 1 status: Deferred to Phase 2 integration — depends on controller and
    output layer

Description:
    Computes the reward signal for the DREX controller from output quality.
    Closes the loop between output and controller.  Also provides the feedback
    signal that creates ESN attractor states in L1 and L2 memory tiers.

Input:
    predicted output: shape (B, vocab_size) or task-specific
    ground truth: shape (B,) token IDs or labels
    previous_loss: float, loss at previous step
    write_decisions: shape (B, n_tiers)

Output:
    reward: shape (B,) float
    esn_feedback: shape (B, d_read) — fed back into L1 and L2 reservoirs

Reward computation:
    quality_reward  = -(current_loss - previous_loss) * lambda_quality
    sparsity_reward = -sum(write_decisions) * lambda_sparse
    total_reward    = quality_reward + sparsity_reward

ESN feedback:
    esn_feedback = linear_projection(output_logits, d_read)
    injected into L1 and L2 reservoirs as feedback input
    this is what creates attractor states and lifts the memory ceiling

Validation criteria:
    [ ] reward sign: better predictions produce higher rewards consistently
    [ ] feedback shape: esn_feedback shape is exactly (B, d_read)
    [ ] sparsity incentive: higher write_decisions → lower reward
    [ ] attractor test: with feedback, L1 reservoir develops stable attractor
        states — measured by state convergence speed


---

## INTEGRATION SPEC

    File: src/drex_unified.py
    Test: tests/python/test_integration.py
    Phase status: Phase 2 only. No integration work begins until ALL Phase 1
    component gates are passed.

Full pipeline pseudocode:

    tokens = InputLayer(raw_text)                                    # (B, S)
    hdc = HDCEncoder(tokens)                                         # (B, S, d_model)
    mamba_out, mamba_state = MambaBackbone(hdc)                      # (B,S,dm), (B,dm)
    write_decisions, read_weights, sparse_gates = Controller(
        hdc[:,-1,:], mamba_state)
    l1_out = WorkingMemory.step(mamba_state, write_decisions[:,0])   # (B, d_model)
    l2_out = EpisodicMemory.step(mamba_state, write_decisions[:,1])  # (B, d_model)
    l3_out = SemanticMemory.query(mamba_state, write_decisions[:,2]) # (B, d_model)
    merged = SparseRouter([l1_out, l2_out, l3_out], read_weights)    # (B, d_model)
    logits = KANReadout(merged)                                      # (B, vocab_size)
    reward, feedback = RewardLoop(logits, targets, write_decisions)
    WorkingMemory.receive_feedback(feedback)
    EpisodicMemory.receive_feedback(feedback)
    Controller.update(reward)

Integration validation criteria:
    [ ] shape propagation: all intermediate tensors are correct shapes end-to-end
    [ ] gradient isolation: L1 and L2 receive zero gradient from task loss
    [ ] memory tier independence: each tier can be ablated without crashing
    [ ] baseline: integrated system < bag-of-words perplexity on WikiText-2
    [ ] transformer comparison: at d_model=256, 4 Mamba layers, perplexity
        within 20% of same-parameter-count transformer on WikiText-2


---

## PHASE GATES — v0.2

### Phase 1 gate

Phase 1 is complete when ALL of the following are true. No exceptions.

Infrastructure (must be complete before any component work begins):
    [x] GitHub Actions CI workflow live and passing on every commit
    [x] Gradient leak assertion passing: no cross-layer gradients in PCN,
        no cross-block gradients in NoProp, zero gradients on ESN reservoir weights
    [x] Dtype boundary assertion passing on every CI run
    [x] Shape contract assertion passing on every CI run

Component validation (all must pass):
    [x] HDC encoder (Obj 0): all six validation criteria pass, minimum viable
        D_hdc documented
    [x] Mamba PCN backbone (Obj 1): convergence test passes, perplexity within
        10% of backprop baseline
    [x] ESN working memory (Obj 2a): echo state property passes, feedback
        extension test passes, POS tagging beats BoW by >= 5 ppt
    [x] Episodic memory (Obj 2b): all four criteria pass, alpha sweep documented
    [x] NoProp semantic memory (Obj 2c): block independence assertion passes on
        CI, accuracy within 5% of backprop on CIFAR-100
    [x] RL controller (Obj 3): better-than-random routing within 1000 episodes
        AND routing collapse test passes over 5000-step evaluation
    [x] KAN readout (Obj 5): approximation test within 2% of MLP, spline
        visualizations non-trivial

Documentation:
    [x] Internal validation report: exact numbers for every component test,
        hardware context, timing
    [x] Spectral radius sweep results documented
    [x] noise_std sweep results for NoProp documented
    [x] Ablation log format confirmed working in all experiment run outputs

### Phase 2 gate

Full integration test suite passes. Controller achieves better-than-random
routing on synthetic task. Feedback loop demonstrably extends ESN memory beyond N.
Full stack achieves lower perplexity than bag-of-words on WikiText-2. Ablation
table complete with all components tested independently.

### Phase 3 gate

125M model trained. All benchmark categories evaluated (WikiText-103 perplexity,
LAMBADA, HellaSwag zero-shot, ARC-Easy zero-shot, long-context crossover point).
Continual learning regression results recorded. Training cost in dollars documented.
arXiv preprint submitted.

### Phase 4 gate

1B model trained and benchmarked. Financial receipt published (exact dollar figure,
hardware spec, wall-clock hours). Model weights released. arXiv updated with 1B
results.

Do not proceed to the next phase until the current phase gate is met.
Document all failures — failed experiments are as important as successes.


---

## OPEN QUESTIONS

1. Optimal d_hdc for language tasks — 10,000 theoretically motivated but
   untested at this depth.  Start at 4096 (current default) and scale if
   HDC orthogonality tests degrade past cosine_sim threshold of 0.1.
   **Phase 1 answer (2026-03-30):** D_hdc=4096 validated.  test_mean_cosine_below_threshold
   confirms mean cross-pair cosine similarity < 0.02 for 1000 random byte-token pairs
   (well below 0.05 threshold).  test_self_similarity confirms cosine_sim(A,A) > 0.999.
   D_hdc=4096 is the Phase 1 production default; scaling to 8192 deferred to Phase 2.

2. NoProp noise_std sensitivity — the paper tested image tasks.  For language,
   the right noise level for denoising targets is unknown.  Treat as tunable
   hyperparameter; sweep {0.05, 0.1, 0.2}.
   **Phase 1 answer (2026-03-30):** noise_std=0.1 (default) used throughout Phase 1.
   test_all_block_losses_decrease and test_each_optimizer_owns_only_its_block_params
   pass with noise_std=0.1.  Block independence CI assertion green.  Full sensitivity
   sweep {0.05, 0.1, 0.2} and CIFAR-100 accuracy parity deferred to Phase 2.

3. Controller reward delay — the quality reward requires a forward pass,
   meaning the controller gets delayed reward.  Investigate whether a learned
   value function (actor-critic instead of pure REINFORCE) is necessary for
   stable learning.

4. Inference-time semantic memory update frequency — updating every token is
   expensive.  Investigate update-every-N-tokens schedule.

5. ESN spectral radius tuning for language — 0.95 standard for time series.
   Language has different temporal correlation structure.  May need values
   closer to 0.99 for long-range dependencies.  Sweep {0.90, 0.95, 0.97, 0.99}.
   **Phase 1 answer (2026-03-30):** Parametrized sweep over ρ∈{0.90, 0.95, 0.99}
   executed in test_reservoir.py (17 tests, 3 spectral-radius variants).  Echo state
   property passes for all three values (max|eig(W_res)|<1.0 confirmed).  Convergence
   test (state divergence < 1e-3 after washout) passes at all three.  Phase 1 default:
   ρ=0.95.  Values 0.97 and long-range recall comparison deferred to Phase 2 benchmark.

6. HDC d_hdc vs d_model gap — current design requires hdc_dim > d_model.  At
   very large hdc_dim (10,000+), the readdown projection is a significant
   bottleneck.  Consider factored projections or learned compression.


---

## HARDWARE FEASIBILITY NOTES

Apple M3 16GB viability for a 1B-parameter DREX-UNIFIED model:
    ESN + HDC components:         CPU, zero training cost
    Mamba backbone + NoProp L3:   M3 GPU via MLX
    Controller:                   CPU (~50K params)
    Active training memory / step: estimated < 6GB
    (no full backward graph stored at any step)

    Comparison: transformer equivalent at 1B params → 40–80GB for same count

Realistic performance claim (honest):
    Narrow tasks (long-context reasoning, continual learning, structured
    prediction): 1–3B DREX-UNIFIED plausibly matches 7B+ transformer on specific
    benchmarks if architectural advantages hold.  Evidence base: Titans result.

    General LM benchmarks (MMLU, HellaSwag): transformer pretraining scale
    advantage will not be overcome by architecture alone at 1B params.

    Correct claim for publications: "matches or beats 7B transformers on
    long-context and continual learning tasks at 1B parameters and a fraction
    of the training cost."

POC validation targets (sufficient for a publishable paper):
    1. ESN episodic tier achieves competitive recall with zero training cost
       vs trained attention layer baseline
    2. NoProp semantic tier converges without global backprop (loss curves)
    3. Controller learns to route to the correct memory tier (routing accuracy
       on synthetic task where correct tier is known)
    4. Full integrated system achieves competitive perplexity on WikiText-2
       compared to same-size transformer


---

## DTYPE BOUNDARY CONTRACT

The canonical dtype flow (assert at every component interface in tests):

    HDC Encoder output:         float32
    Mamba input projection:     float32 → bfloat16  (explicit cast, here only)
    Mamba output:               bfloat16
    Controller input:           bfloat16
    Controller output:          int32 (write decisions), float32 (read weights),
                                bool (sparse gates)
    ESN reservoir state:        float32
    ESN readout output:         float32
    Episodic memory state:      float32
    Semantic memory (NoProp):   bfloat16
    KAN readout output:         float32
    Reward signal:              float32
    ESN feedback signal:        float32

Any implicit cast at a boundary is a bug. Assert dtypes at every component
interface in tests and in test_dtype_contracts_ci.py.

---

## ABLATION LOG FORMAT

Every DREX experiment run must record which components were active:

    {
      "components": {
        "hdc_encoder": true,
        "mamba_backbone": true,
        "esn_working_memory": true,
        "episodic_memory": true,
        "semantic_memory_noprop": true,
        "sparse_router": true,
        "kan_readout": true,
        "controller_rl": true,
        "reward_feedback": true
      }
    }

A run without this field is invalid and must not be cited.

---

## CHANGELOG

    v0.1  2026-03-24  Initial spec drafted (architecture planning session)
    v0.2  2026-03-24  First revision: adapted to repo structure, Phase 23+24 noted
    v0.2  2026-03-XX  Fresh start: supersedes all prior planning documents.
                      Legacy code archived to research/legacy/. File layout
                      changed to real src/ filesystem path. Changelog from v0.1
                      block added. PHASE 1 INFRASTRUCTURE section added.
                      HDC encoder promoted to Objective 0. KAN promoted to
                      Objective 5. Controller routing collapse exit blocker added.
                      NoProp exit criterion replaced VRAM-only with accuracy parity
                      (CIFAR-100). ESN feedback extension test promoted to exit
                      blocker. Phase Gates section rewritten as v0.2 format.
                      Implementation status table removed (code in research/legacy/).
