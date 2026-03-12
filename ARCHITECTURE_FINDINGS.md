# ARCHITECTURE_FINDINGS.md — Drex Research Findings

*Created: 2026-03-12 | Covers Phases 1–12 (48 categories, 247+ experiments)*

This document records the validated architecture specification and the research dead ends
that should not be re-investigated. All findings have ≥2/3 seed confirmation unless noted.

---

## §1 — Memory Hierarchy Overview

Drex uses a three-tier memory system:

| Tier | Mechanism | Location | Validated |
|------|-----------|----------|-----------|
| L1 | Sliding-window causal attention | Activations (in-context) | Yes |
| L2 | Infini-Attention delta-rule matrix (MemoryState) | Activations (cross-segment) | Yes |
| L3 | Titans-style MLP weight snapshots (TitanMemory) | Disk via Rust SnapshotStore | Yes |
| L4 | Episodic/semantic split delta-rule (MemoryModule) | Activations (per-segment) | Yes (Phase 12) |

---

## §2 — L2: Infini-Attention (DeltaRuleUpdate)

- Feature map: φ(x) = ELU(x) + 1 (positive-valued, unbounded above)
- Delta rule: ΔM = φ(K)ᵀ @ (V − φ(K)M)
- Normalisation accumulator: z += Σ φ(K) over positions
- Read: r = φ(Q)M / (φ(Q)z + ε)
- Validated: 9/9 seeds consistent

---

## §3 — L3: TitanMemory

- Small 2-layer MLP (no bias, no LayerNorm)
- Memory = weights; writing = one Adam gradient step on ‖net(k) − v‖²
- Independent internal Adam optimiser (not the outer training optimiser)
- Weight snapshots stored via Rust SnapshotStore; async prefetch via PrefetchEngine
- Sketch-based similarity index (rank-16 projection) for k-NN prefetch decisions

---

## §4 — L4: MemoryModule (Phase 13 Production Code)

The validated minimal architecture for the episodic/semantic associative memory layer.

### Architecture

```
Input: x ∈ ℝ^{B × L × H}

For t = 0 … L-2 (write passes):
  ks = sem_proj(x_t)      ∈ ℝ^{B × H/2}   (no bias)
  ke = epi_proj(x_t)      ∈ ℝ^{B × H/2}   (no bias)
  k̂s = ks / ‖ks‖          (unit key — semantic)
  k̂e = ke / ‖ke‖          (unit key — episodic)
  vps = M_sem @ k̂s        (current prediction from memory)
  vpe = M_epi @ k̂e

  -- OR write gate (relative-norm criterion) --
  fire = (‖ks − vps‖ ≥ thresh·‖ks‖) OR (‖ke − vpe‖ ≥ thresh·‖ke‖)

  ΔM_sem = (ks − vps) ⊗ k̂s          (outer product delta)
  ΔM_epi = (ke − vpe) ⊗ k̂e

  w_t = (t + 1) / L                   (recency weight ∈ (0, 1])

  M_sem += (1 − α) · fire · ΔM_sem    (EMA write, semantic)
  M_epi += (1 − α) · w_t · fire · ΔM_epi  (EMA write, episodic + recency)

At position L-1 (query):
  q = x_{L-1}
  r_sem = M_sem @ norm(sem_proj(q))
  r_epi = M_epi @ norm(epi_proj(q))
  r = concat(r_sem, r_epi)            ∈ ℝ^{B × H}

  g_null = σ(null_gate(q))            (learned scalar null-retrieval gate)
  r = g_null · r

Output = out_proj(r)                  ∈ ℝ^{B × H}
```

### Hyperparameters (non-negotiable)

| Parameter | Value | Evidence |
|-----------|-------|----------|
| thresh\* | **0.70** | exp_48_1, Phase 12 (3/3 seeds deterministic wr) |
| α(L) | **0.95^(96/L)** | exp_47_1/3, Phase 11 (exp_scale formula) |
| Matrix size | **H/2 × H/2** (two halves) | exp_38_1, 9/9 seeds |
| Episodic recency weight | **(t+1)/L** | Phase 11 validation |
| Write gate op | **OR** over branches | exp_47_2 (AND gate degrades recall) |
| Read combination | **concat** (no learned gate) | exp_38_3 (learned gate −10%) |
| Null retrieval gate | **learned σ(linear(q))** | Phase 13 design |

### Valid write rate range

After any change to the write mechanism, validate:

```
WRITE_RATE_LO = 0.10  (minimum acceptable gate firing fraction)
WRITE_RATE_HI = 0.85  (maximum acceptable gate firing fraction)
```

At thresh=0.70, exp_scale:
- wr(L=32) = 0.581 (target: [0.20, 0.70]) ✓
- wr(L=96) = 0.421 (target: [0.15, 0.50]) ✓

---

## §5 — Length-Adaptive EMA (Phase 11)

**Problem:** At L=32 with fixed α=0.95, the memory never forgets: effective time constant
τ = 1/(1−α) = 20 steps, τ/L = 0.625. Memory fills and write rate spikes to ~0.97.

**Solution:** α(L) = 0.95^(96/L) keeps τ/L ≈ 0.21 constant across L=32–128.

| L | α(L) | τ (steps) | τ/L |
|---|------|-----------|-----|
| 16 | 0.857 | 7.0 | 0.44 |
| 32 | 0.857 | 7.0 | 0.22 |
| 64 | 0.923 | 12.8 | 0.20 |
| 96 | 0.950 | 20.0 | 0.21 |
| 128 | 0.961 | 25.6 | 0.20 |

Note: L<24 (≈ 5 key-value pairs in short context) produces wr=1.0 — this is correct
behavior, not an error. Every token is novel at that density.

---

## §6 — OR-Gate Write-Rate Inflation (Phase 12)

**Problem:** With two branches each firing independently at p≈0.58 (thresh=0.40),
the OR gate fires at Pr(A∪B) = 1−(1−p)² ≈ 0.82, in practice 0.774.

**Fix:** thresh\*=0.70 reduces each branch to p≈0.35, OR probability ≈ 0.58.
Observed at thresh=0.70: wr(L=32)=0.581, wr(L=96)=0.421.

**Key insight:** The threshold scales with the per-branch probability, not the OR
combined probability. Geometric estimate: thresh\* ≈ thresh_old × (p_target/p_old).

---

## §7 — Integration into DrexTransformer (Phase 13)

`MemoryModule` is inserted into each `DrexLayer` as an optional branch:

```python
DrexConfig(use_episodic_memory=True, episodic_gate_thresh=0.70)
```

In `DrexLayer.forward()`, after the attention and feed-forward sub-layers:

```python
if self.episodic_mem is not None and self.norm_mem is not None:
    mem_r = self.episodic_mem(self.norm_mem(x))  # (B, d_model)
    x = x.clone()
    x[:, -1] = x[:, -1] + mem_r                 # residual at query position
```

Pre-LayerNorm is applied to `x` before passing it to `MemoryModule`, consistent with the
pre-norm convention used throughout `DrexLayer`. The result is a residual addition at the
last token position only (the query position), leaving all earlier positions unchanged.

Gradient checkpointing passes `x` through `layer(x, state)` unchanged, so MemoryModule
is compatible with `gradient_checkpointing=True` without modification.

---

## §8 — Optimiser

**Use Adam.** Not SGD. Not AdamW (AdamW acceptable as a minor variant).

Evidence: exp_34_6 showed >10% accuracy spread across optimisers on the associative
recall benchmark. Adam was the best performer across 9/9 seeds.

---

## §9 — Dead Ends (Do Not Re-investigate)

The following approaches were tested to refutation (≥7/9 seed evidence) and should not
be re-investigated. The research investment to rule each out was significant.

### Memory architecture variants

| Approach | Reason rejected | Experiment |
|----------|-----------------|------------|
| Tiered memory routing (3+ tiers active together at inference) | No throughput gain; complexity without recall benefit | Phase 4 |
| Hierarchical write decisions (sub-gate inside gate) | Double-gate overhead; unstable write rates | Phase 6 |
| Momentum delta rule | Oscillation in M; accuracy −8% | Phase 7 |
| Bidirectional delta rule | Cannot be applied to causal setting | Phase 7 |
| Velocity gate (δ-based firing) | Equivalent to relative-norm at calibration; more complex | Phase 9 |
| Matrix-mean energy gate | Produces O(1/H) values; always below threshold | exp_45_1 |
| Position-schedule gate (cosine, linear) | Static schedule degrades accuracy at unusual densities | Phase 8 |
| Offline consolidation pass | Requires second forward pass; not streaming-compatible | Phase 5 |
| Hindsight oracle distillation | Requires look-ahead; cannot be trained causally | Phase 5 |
| Three-gate auxiliary loss combos | Loss interference; write rate becomes erratic | Phase 10 |
| Write rate regularisation (L1/L2 on gate) | Collapses write rate; accuracy degrades | exp_45_2 |
| Two-phase gate training (freeze then unfreeze) | No benefit over end-to-end training | Phase 10 |

### Splitting / routing variants

| Approach | Reason rejected | Experiment |
|----------|-----------------|------------|
| Learned router for episodic/semantic split | 10–24% accuracy loss vs fixed 50/50 | exp_38_1 |
| 3-way split (episodic/semantic/prospective) | No accuracy gain; wr harder to control | Phase 8 |
| Unequal split (70/30) | Lower than 50/50; no structural justification | Phase 8 |

### Read-side variants

| Approach | Reason rejected | Experiment |
|----------|-----------------|------------|
| Learned gated combination of r_sem and r_epi | −10% accuracy vs hard concat | exp_38_3 |
| Separate output projections per branch | Marginal; not worth extra parameters | Phase 9 |
| Cosine similarity retrieval (instead of matrix multiply) | Equivalent to dot product after normalisation; no gain | Phase 7 |

### Training variants

| Approach | Reason rejected | Experiment |
|----------|-----------------|------------|
| REINFORCE for write gate | Encoder gradient = 0 (gate blocks signal) | exp_7_1 |
| Random-init thresh (learnable starting at random value) | Low-accuracy equilibrium; gate never recovers | exp_43_1 |
| Randomly-initialised learnable thresh | Low-accuracy equilibrium (same as above) | exp_43_1 |
| Fixed α=0.95 alone (no length-adaptation) | wr spikes to 0.97 at L≤32; bootstrap failure | Phase 11 |
| Learned MLP gate for α scheduling | Unnecessary complexity; exp_scale formula sufficient | exp_47_1 |
| AND gate (both branches must fire) | Degrades recall accuracy; exp_47_2 showed OR is strictly better | exp_47_2 |
| thresh=0.40 for OR-gate split model | wr=0.774 at L=32; exceeds target [0.20, 0.70] | exp_47_2 |
| Universal single threshold for OR-gate split at any thresh < 0.50 | OR inflation persists below thresh=0.50 | exp_48_1 |

---

## §10 — Full Implementation Specification

The production implementation target is `drex.models.memory.MemoryModule`. The exact
specification (non-negotiable from ≥9-seed research):

### Forward contract

```
Input:  x ∈ ℝ^{B × L × H}    — full context; position L-1 is the query
Output: r ∈ ℝ^{B × H}         — memory retrieval for position L-1
```

### Required components

1. **Two associative matrices per forward call** (re-initialised to zero each call):
   - `M_sem ∈ ℝ^{B × H/2 × H/2}` — semantic branch
   - `M_epi ∈ ℝ^{B × H/2 × H/2}` — episodic branch

2. **Two key projections** (no bias, scale-invariant):
   - `sem_proj: H → H/2`
   - `epi_proj: H → H/2`

3. **Delta-rule update** with EMA smoothing:
   - Unit key normalisation: k̂ = k / ‖k‖
   - Error: δ = k − M @ k̂
   - Outer product: Δ = δ ⊗ k̂
   - EMA write: M += (1 − α(L)) · gate · Δ
   - Episodic branch additionally multiplies by recency weight w_t = (t+1)/L

4. **OR relative-norm write gate**:
   - `fire = (‖ks − vps‖ ≥ thresh · ‖ks‖) OR (‖ke − vpe‖ ≥ thresh · ‖ke‖)`
   - thresh = **0.70** (do not change without re-running write-rate validation)
   - Minimum allowed thresh = **0.40** (exp_43_1 hard constraint)

5. **Length-adaptive EMA coefficient**:
   - `α(L) = 0.95^(96/L)` (exp_scale formula)
   - Do not use fixed α=0.95 alone

6. **Soft concatenated retrieval**:
   - `r = concat(r_sem, r_epi) ∈ ℝ^{B × H}`
   - No learned gate on the combination (exp_38_3)

7. **Null retrieval gate**:
   - `g = σ(null_gate(q))` — learned scalar (Linear(H, 1))
   - Applied as `r = g · r` before `out_proj`
   - Suppresses irrelevant retrievals when memory is near-zero

8. **Output projection**: `Linear(H, H)` mapping `r → ℝ^{B × H}`

9. **Write-rate tracking**: record `wr_count / wr_total` after each forward

10. **d_model must be even** (enforced by ValueError in __init__)

### Hard constraints

- `gate_thresh ≥ 0.40` (exp_43_1 — lower values trigger low-accuracy equilibrium)
- `α(L) = 0.95^(96/L)` — never fixed α=0.95 alone
- Validate `write_rate ∈ [0.10, 0.85]` using `assert_write_rate_valid()` after training changes
- No learned episodic/semantic router
- No learned combination of r_sem and r_epi
- Optimiser: Adam (not SGD)
