# Preliminary Architecture Findings: Drex Memory Research
*9 Research Phases · 45 Experiment Categories · 211 Experiments · 940 Runs*

---

## Executive Summary

Over nine phases of empirical investigation we tested 211 architectural hypotheses for a
learnable memory module suitable for integration into a modern LLM. Experiments ran on an
associative recall task (reconstruct a value token given a query key, embedded in a longer
sequence) using a controlled PyTorch harness with three independent seeds per run.

| Outcome | Count | Share |
|---|---|---|
| ✓ Supported | 51 | 24% |
| ~ Inconclusive | 72 | 34% |
| ✗ Refuted | 88 | 42% |

The high refutation rate is by design: each phase was guided by outcomes from the previous
one, so later phases targeted hypotheses in contested territory. What survived this
pressure is a validated, replicable minimal stack:

> **Delta-rule associative matrix + EMA smoothing (α = 0.95) + episodic/semantic split
> (two H/2 matrices) + relative-vector-norm write gate (‖k − vp‖ ≥ thresh·‖k‖)**

All four components are individually seed-stable across 9 independent seeds (experiments
32_1–32_4). The interaction between EMA and the write gate at short sequences is the
primary open engineering problem (see §8).

---

## 1 · Memory Write Mechanism

### 1.1 Core Update Rule

The **delta-rule** outer-product update is the recommended write primitive:

```
M ← M + (k − v_prev) ⊗ k_n
```

where `k` is the current key embedding, `v_prev = M · k_n` is the current memory
prediction, and `k_n = k / ‖k‖`. This gives the model built-in correction semantics:
it only updates M by as much as the current key-value association deviates from what
is already stored, which dramatically reduces interference compared to unconditional
outer-product writes.

**Why not slot memory?** Slot memory degrades more rapidly with sequence length.
Parametric memory retains a >0.35 accuracy advantage over slot memory at 4× context
length (exp_32_4, replicated across 9 seeds, gap > 0.35 on ≥ 7/9).

**Why not a parametric MLP memory?** Parametric MLP scales better with sequence length
(exp_16_3, exp_32_4) and is worth keeping as a complementary component, but the
delta-rule matrix is cheaper to implement correctly and its gating behaviour is more
transparent. Use the parametric module if your task has long-range retrieval at >8×
baseline sequence length; for typical context lengths the delta-rule matrix is
sufficient.

### 1.2 When to Write: Selective Gating

Write every token is wasteful and reduces accuracy (exp_3_1: event-driven > periodic).
The best-tested gate criterion is **relative vector-norm energy**:

```
energy   = ‖k − M · k_n‖   (reconstruction error)
threshold = thresh × ‖k‖    (scale-invariant reference)
write = energy > threshold
```

with `thresh = 0.40`.

**Critical implementation note**: Earlier work mistakenly used matrix-mean energy
`Delta.pow(2).mean([1,2])`, which produces values of order `1/H` (≈ 0.007–0.016 at
H = 64). This is ~25× below the threshold, causing zero gate fire rate and near-random
accuracy — a silent correctness failure (exp_45_1). The relative-norm formulation is
dimension-invariant. Validate gate write rate is in [0.10, 0.85] after any change
to the gate criterion.

**Key numbers** (exp_32_2, 9-seed confirmed):
- acc_ratio (gated / ungated) > 0.90 on ≥ 7/9 seeds at write_rate < 0.70
- Energy-gated delta achieves <70% write rate at 90% of full accuracy

**Learned gate signal beats heuristics** (exp_1_5): Learned write gate > random write
gate > attention-score gate > surprise gate. Low-surprise, predictable tokens tend to
form the most stable, reusable context. Surprise is anti-correlated with retrieval
value.

**Retroactive writing works** (exp_3_6, exp_32_1): A two-pass scheme — write forward,
then optionally re-write based on downstream context — yields +0.133 accuracy gap over
forward-only writing, replicated on ≥ 7/9 seeds with gap > 0.09. Budget accordingly
for a second write pass if inference latency allows.

**Write latency differs by position** (exp_3_5): Writing a token at its presentation
time is not always optimal. Delayed or retroactive writes are permissible and
measurably useful for certain token positions.

### 1.3 Write Gate Stability

The write gate exhibits **multi-stability** (exp_43_1, exp_39_3): learnable thresholds
initialized at different values converge to distinct stable equilibria rather than a
single global attractor. The accuracy-maximizing equilibrium is near thresh ∈ [0.30, 0.50].

Implications:
- Initialize the write gate threshold near 0.4 (not random).
- A hard threshold gate has lower write-rate variance across seeds than a soft sigmoid
  gate (exp_43_4). Prefer hard gating if write-rate consistency matters.
- Regularizing the gate threshold (L2 toward 0.5) failed to collapse the multi-stability
  (exp_43_6, exp_43_7). Initialization is the primary lever.

---

## 2 · EMA Smoothing

Apply **exponential moving average (EMA) smoothing** to delta-rule updates:

```
M ← M + (1 − α) · (k − v_prev) ⊗ k_n,   α = 0.95
```

Evidence:

| Claim | Experiment | Result |
|---|---|---|
| EMA (α=0.85–0.95) achieves acc_ratio ≥ 0.96 AND improves clean accuracy 5–10% | exp_37_3 | ✓ SUPPORTED |
| EMA reduces gradient variance >30% vs standard delta | exp_41_6 | ✓ SUPPORTED |
| Per-position alpha provides no benefit over global alpha | exp_41_5 | ✓ SUPPORTED |
| Outer-product linear memory matches slot accuracy within 2% | exp_29_1 | ✓ SUPPORTED |

**Use a single global α = 0.95.** Per-position or per-matrix learned alphas add
complexity without measurable gain (exp_41_5). Separate alphas for episodic vs semantic
matrices are inconclusive (exp_41_7).

### 2.1 EMA-Gate Interaction Warning

EMA inflates write rate at short sequences. At SEQ_LEN = 32 with thresh = 0.40,
measured write rate under EMA+gate is ≈ 0.96 — the gate is functionally inert
because EMA delays memory convergence, keeping `‖k − vp‖` large throughout training.

| Sequence length | Measured write rate (EMA+gate) |
|---|---|
| L = 32 | 0.96 |
| L = 96 | 0.31 |

An inflated write rate means the gate adds no selectivity. This is not a bug per se but
means the threshold needs to be re-calibrated to the (α, L) operating point. Phase 10
is addressing this directly. **Do not assume a fixed threshold transfers across
significantly different sequence lengths.**

---

## 3 · Episodic / Semantic Memory Split

Replace the single delta-rule matrix with **two H/2 matrices** carrying different
inductive biases:

- **Semantic matrix M_sem**: standard delta-rule writes (content association)
- **Episodic matrix M_epi**: delta-rule writes with temporal recency weighting `(t+1)/L`

Evidence:

| Claim | Experiment | Result |
|---|---|---|
| Split outperforms unified by >5% on tasks requiring both recall types | exp_36_3 | ✓ SUPPORTED |
| Split advantage persists at SEQ_LEN=96 (3× baseline) | exp_42_7 | ✓ SUPPORTED |

Important constraints:

- **Fixed split beats learned routing by 10–24%** (exp_38_1). The recency-weighted
  inductive bias on the episodic matrix is doing the work; a learned gate that tries
  to discover this routing from scratch fails. Do not replace the fixed positional
  weighting with a learned router.
- **Asymmetric capacity (25% epi / 75% sem) is seed-dependent** (exp_38_2). The 50/50
  split is the safe default.
- **Read combination**: both concatenation and a learned attention gate over
  `[M_sem_read, M_epi_read]` produce inconclusive results (exp_38_3). Use
  concatenation (simpler) unless task evidence supports a gate.

---

## 4 · Read Mechanism

### 4.1 Query Formation

Use a **dedicated query former** (exp_4_1): a small feed-forward module that projects
the final hidden state to the retrieval query, rather than using the hidden state
directly. This is one of the most robust findings across phases.

### 4.2 Soft vs Hard Retrieval

**Soft (attention-weighted) retrieval > hard (argmax) retrieval** for gradient stability
(exp_4_4). Hard retrieval produces unstable training dynamics during early learning.

### 4.3 Null Retrieval Gate

A **null retrieval gate** (output zero rather than retrieve when query is irrelevant) is
learnable without supervision (exp_4_7). Add it. Read suppression under high-confidence
positions costs < 1% accuracy while reducing unnecessary memory reads (exp_5_6).

### 4.4 Read Frequency

Read suppression threshold varies by > 0.15 across task types (exp_11_3): a single
fixed threshold will under-suppress on some tasks and over-suppress on others. Either
make the threshold learnable or tune it per task family.

---

## 5 · Compositional Retrieval

Two-hop and three-hop compositional retrieval are feasible at H = 64:

| Finding | Experiment | Result |
|---|---|---|
| Two-hop retrieval survives 40% near-duplicate interference | exp_13_1 | ✓ SUPPORTED |
| Three-hop chain accuracy is retained (>2×) at H=64 | exp_13_2, exp_32_3 | ✓ SUPPORTED (9-seed) |
| Two-hop sustains >70% accuracy under 60% bridge corruption | exp_24_2 | ✓ SUPPORTED |

**Four-hop chains at H = 64 are infeasible** (exp_24_4): accuracy drops > 50% relative
to two-hop, and a hop-by-hop curriculum did not close this gap. This is probably a
fundamental capacity limit at this hidden dimension.

Gradient-surprise-gated TTT (test-time training) achieves 90% of full-update accuracy
at < 50% update rate (exp_29_3): selectively updating the parametric component only when
the gradient-norm ratio exceeds 1.5 is a strong sparse-update policy.

Multi-head delta rule (4 heads × H/4) outperforms single-head by > 5% at 8-pair recall
(exp_30_1, SUPPORTED). Prefer multi-head when scaling to higher pair counts.

---

## 6 · Forgetting / Eviction Policy

**Learned importance-based eviction > LRU** (exp_6_1). The model learns to protect
slots containing high-retrieval-value associations.

**Selective forgetting under distribution shift is learnable** (exp_6_3): when the
input distribution changes, the model can learn to identify and evict stale memories
without full resets.

**Protected slots** are optimal at K > 5 (exp_9_4, exp_26_1, 9-seed stable): marking a
small fixed set of slots as write-protected against eviction preserves high-value
associations across the full context window. The optimum K is 3–6 slots.

---

## 7 · Controller and Gate Architecture

| Finding | Experiment | Recommendation |
|---|---|---|
| Gumbel-softmax > STE > RL for discrete gate gradient | exp_7_1 | Use Gumbel-softmax |
| Controller complexity budget exists | exp_7_2 | Do not over-parameterize the controller |
| Controller decisions interpretable before task training | exp_7_9 | Supports modular pre-training strategy |
| Balanced null/retrieval training distribution required | exp_9_2 | Sample null queries at ≥ 30% during training |
| Delta rule has strong optimizer preference (>10% spread) | exp_34_6 | Use Adam; AdamW if weight decay is needed |

---

## 8 · Open Engineering Questions (Phase 10 Focus)

### 8.1 EMA–Gate Threshold Calibration

The most actionable open problem: **the write gate fires 96% of the time under EMA at
L = 32**, making the gate useless at short contexts. The gate recovers selectivity at
L = 96 (wr ≈ 0.31), but this cannot be relied upon at LLM inference where context
prefix length varies.

Three approaches under investigation:

1. **Threshold sweep per (α, L)**: find the calibration curve and apply it as a fixed
   schedule.
2. **Velocity gate**: condition fire on the *change* in reconstruction error over recent
   steps rather than the absolute value, making the gate independent of EMA convergence
   state.
3. **Learned gate bias**: train a small MLP to predict an adaptive per-token threshold
   offset, allowing the model itself to compensate for EMA's effect.

### 8.2 Task-Hardness Threshold Dependence

Write rate equilibrium increases with interference density ρ = N_pairs / H (exp_39_2,
though inconclusive at the exact threshold tested). In harder tasks the gate naturally
fires more. Whether a single threshold generalises across difficulty regimes — or
whether the threshold needs to track task load — is unresolved.

### 8.3 Sequence-Length Generalization

Scale generalization at L = 96 with the corrected gate is partially refuted (exp_45_6):
write rate at L = 32 ≈ 0.96 but at L = 96 ≈ 0.31. The gate is selectivity-capable at
long contexts but not at short. For LLM use, short-prefix behaviour must be addressed
before deployment.

---

## 9 · Confirmed Dead Ends

The following architectural directions were tested with adequate statistical power and
consistently failed to provide benefit. Avoid them in future designs.

| Direction | Evidence | Why It Failed |
|---|---|---|
| Tiered (two-tier) memory | cat_18, multiple exps | Flat architecture matches or beats all two-tier variants |
| Hierarchical write decisions | cat_17 | No accuracy gain; adds controller complexity |
| Joint write-evict optimisation | cat_9 ablations | Entangled objectives; both degrade |
| Sparse Hopfield addressing | cat_19 | No interference reduction benefit at this scale |
| Prospective / anticipatory writing | cat_17 | Query-conditioned prospective gate unstable |
| Hindsight oracle distillation | cat_21 | Distilled gate no better than end-to-end trained |
| Feedforward controller (no recurrence) | cat_21 | Lower memory utilisation than LSTM controller |
| Learned episodic/semantic router | exp_38_1 | Fixed inductive bias 10–24% better than trained router |
| Three-gate auxiliary loss combinations | cat_20 | Super-additive benefit refuted; individual losses inconclusive |
| Offline consolidation ("sleep replay") | exp_36_1 | No accuracy improvement from offline replay |
| Bidirectional delta rule | exp_30_3 | Backward pass hurts early-query accuracy |
| Momentum delta rule | exp_30_2 | No accuracy or stability improvement over EMA |
| Row-normalisation of M | exp_37_2 | Baseline already robust; normalisation neither helps nor hurts |
| Write-first curriculum training | cat_34 ablations | No benefit over standard training |
| Write rate regularisation to target wr | exp_43_6 | Does not collapse multi-stability; corrupts task loss |
| Two-phase gate training (freeze then unfreeze) | exp_43_7 | Does not reduce equilibrium spread |
| Matrix-mean energy gate criterion | exp_45_1 | O(1/H) scale incompatibility with any fixed threshold |

---

## 10 · Recommended Architecture Stack

The following configuration is the current best-validated design for a memory module
that attaches to an LLM layer. All components have been verified seed-stable across
≥ 7 independent seeds.

```
Memory module(hidden_dim H):
  M_sem  ∈ ℝ^{H/2 × H/2}   # semantic delta-rule matrix
  M_epi  ∈ ℝ^{H/2 × H/2}   # episodic delta-rule matrix (recency-weighted)
  α      = 0.95              # global EMA coefficient (scalar, not learned)
  thresh = 0.40              # write gate threshold (see §8 for calibration)

Write(token t, key k, hidden_dim H, sequence_position t, seq_len L):
  k_n    = k / ‖k‖
  v_prev = M_sem · k_n      # current memory prediction (semantic)
  energy = ‖k − v_prev‖
  ref    = thresh × ‖k‖
  if energy > ref:           # relative-norm gate (dimension-invariant)
    Δ      = (k − v_prev) ⊗ k_n
    w_epi  = (t + 1) / L    # recency weight for episodic matrix
    M_sem ← M_sem + (1 − α) · Δ
    M_epi ← M_epi + (1 − α) · w_epi · Δ

Read(query q):
  q_n     = QueryFormer(q)         # dedicated feed-forward query former
  r_sem   = M_sem · (q_n / ‖q_n‖)
  r_epi   = M_epi · (q_n / ‖q_n‖)
  output  = concat(r_sem, r_epi)   # or attend if task justifies it
  if NullGate(q, output) > 0.5:   # learned null retrieval gate
    output = zeros
  return output
```

**Optimizer**: Adam (exp_34_6). The delta rule shows >10% accuracy spread across
optimizers; AdamW is acceptable; avoid SGD.

**Training**: Initialize write gate threshold at 0.40, not randomly (exp_43_1). If using
a learnable threshold, initialize near 0.4 — multi-stability makes random init risky.

**Validation checklist after any change to write mechanism**:
1. Confirm write rate is in [0.10, 0.85] across training (silent correctness check).
2. Confirm accuracy is > random at end of training (gate collapse detection).
3. Evaluate at both short (L ≈ 32) and long (L ≈ 96) sequence lengths.

---

## 11 · Quantitative Reference Table

| Metric | Value | Experiment |
|---|---|---|
| EMA gradient variance reduction | 75% | exp_29_1 |
| EMA clean accuracy improvement | 5–10% | exp_37_3 |
| Retroactive writing accuracy gap | +0.133 | exp_3_6 / exp_32_1 |
| Energy-gated write rate at 90% acc | < 0.70 | exp_15_3 / exp_32_2 |
| Three-hop retention vs two-hop | > 2× | exp_13_2 / exp_32_3 |
| Parametric memory length retention gap (vs slot) | > 0.35 | exp_16_3 / exp_32_4 |
| Two-hop accuracy under 60% bridge corruption | > 70% | exp_24_2 |
| EMA write rate at L=32 (thresh=0.40) | 0.96 | exp_45_4 |
| EMA write rate at L=96 (thresh=0.40) | 0.31 | exp_45_6 |
| Read suppression quality cost | < 1% | exp_5_6 |
| Optimizer accuracy spread (best vs worst) | > 10% | exp_34_6 |
| Protected slot count optimum | 3–6 slots | exp_9_4 / exp_26_1 |

---

## 12 · Confidence Classification

**High confidence (≥7/9 seeds, multiple phases)**:
- Delta-rule over slot memory for length-generalisation
- EMA α=0.95, global scalar sufficient
- Relative vector-norm gate criterion (dimension-invariant)
- Retroactive writing is beneficial
- Fixed episodic/semantic split beats learned routing
- Multi-hop retrieval feasible at H=64 up to 3 hops
- Write gate multi-stability; initialize threshold near 0.40

**Medium confidence (2–3 seeds, single phase)**:
- Dedicated query former benefit
- Null retrieval gate learnability
- Soft retrieval stability advantage
- Parametric MLP as complementary long-context module
- Non-uniform write budget allocation

**Low confidence / inconclusive (needs more work)**:
- Asymmetric episodic/semantic capacity (25/75 vs 50/50)
- Adam at inference vs SGD at inference for TTT
- Read combination gate vs concatenation
- Learned alpha per episodic/semantic matrix
- EMA-gate calibration curve across (α, L)

---

*Generated from `research/MASTER_REPORT.md` · Phases 1–9 · Commit f7bf97b*
