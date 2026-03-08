# Research Log

This is a living document. Every experiment run gets an entry. Failures get
as much detail as successes — more, if the failure was interesting.

Format per entry:
- Date
- Experiment ID
- Outcome (SUPPORTED / REFUTED / INCONCLUSIVE / ERROR)
- What actually happened (not what was expected)
- The most surprising thing
- What it changes about the next question to ask

---

## Log

---

### 2026-03-08 | exp_7_9 | SUPPORTED (marginally)
**Interpretability Baseline**

The gate is non-random before training (std=0.146). A single linear projection
establishes token-type preferences from embedding geometry alone. Numeric tokens
are the most written (+0.076 trained correlation). Punctuation is avoided (−0.047).
Position has near-zero signal.

**Surprising:** Training barely changed the correlations (0.055 → 0.059 interpretability
score). The preferences are established in the embedding geometry, not learned from
the task. This means a single linear gate's "intelligence" lives in the embedding
space, not the gate weights.

**Changes:** As controllers get more complex, check whether the added layers are
learning task-specific signals or just amplifying embedding biases. The baseline
is now established.

---

### 2026-03-08 | exp_7_1 | SUPPORTED (strongly)
**Differentiability: STE vs Gumbel vs REINFORCE**

Ranking: Gumbel > STE > REINFORCE on both accuracy and stability.
REINFORCE loss variance: 2.696 (380× worse than Gumbel 0.007).
REINFORCE encoder gradient norm: 0.000 — the encoder does not learn.

**Surprising:** REINFORCE's gradient norm is literally zero. The policy gradient
signal does not flow back to the encoder at all in this setup. REINFORCE trains the
policy over a static, randomly-initialized feature space. This isn't a subtle
degradation — it's a complete training failure for the encoder component.

**Changes:** REINFORCE is ruled out for encoder-memory joint training. Gumbel-softmax
with temperature annealing is the default training mechanism going forward. This
constrains the design space substantially.

*Note: original implementation had a shape bug in the REINFORCE log_prob computation.
Fixed and re-run. See commit history.*

---

### 2026-03-08 | exp_6_6 | INCONCLUSIVE
**Controller Catastrophic Forgetting**

Domain A accuracy fell from 22.7% to 1.9% after domain B training (−0.208).
EWC reduced this by only 1.8% — negligible protection.

**Surprising:** The EWC failure isn't surprising given the low baseline (22.7%).
EWC protects weights with high Fisher information on domain A, but if those weights
were themselves weakly trained, there is little to protect. The forgetting may be
real but is hard to interpret at this performance level.

**Changes:** Re-run with domain A baseline above 70% before introducing domain B.
The forgetting phenomenon needs a stronger signal to work from. Also: investigate
whether the controller's policy (not just its output layer) is what's being forgotten.
The policy could be forgotten even when the task performance metric stays constant.

---

### 2026-03-08 | exp_5_1 | REFUTED
**Read Gate Collapse Detection**

Regime A (task only) stabilized at 15.6% read rate — no collapse.
Regime B (sparsity regularizer) collapsed to 9.4% (never).
Regime D (confidence) stable at 65.6%.

**Surprising:** Same pattern as exp_3_2. The regime designed to prevent collapse
(sparsity) caused the opposite collapse. The unconstrained gate found a stable
~16% read rate naturally. Regime A and regime C (coverage penalty) settled at
identical rates (15.6%), suggesting the task loss already determines the equilibrium
and additional regularization provides no marginal benefit.

**Changes:** The "collapse prevention" problem is not the main challenge. The
challenge is designing tasks that require selective memory use so the equilibrium
rate reflects genuine capability, not just distributional convenience.

---

### 2026-03-08 | exp_4_9 | SUPPORTED (strongly)
**Compositional Retrieval**

Single-hop: 0.968. Two-hop: 1.000. Two-hop *outperforms* single-hop.
The gap is −0.032 in favor of two-hop — the opposite of the expected direction.

**Surprising:** The two-hop architecture is more accurate than single-hop on this
task. The intermediate aggregation step may be acting as a regularizer that prevents
overfitting to spurious correlations in the direct pattern-match. The model is
forced to form a meaningful intermediate representation.

**Changes:** Follow up on why two-hop generalizes better. Is it the architecture or
the task structure? Run this on a larger KB with interference to see if the advantage
holds.

---

### 2026-03-08 | exp_4_7 | SUPPORTED* (degenerate)
**Null Retrieval Learning**

Null precision: 1.000. Null recall: 0.800. Retrieval on matches: 0.000.
The gate learned to never fire. "Always null" achieves 80% accuracy when 80% of
queries are null.

**Surprising:** The shortcut is perfectly rational. The model correctly learned the
task distribution and found the highest-reward policy — which happens to be useless.
The hypothesis was technically supported but the experiment was gamed.

**Changes:** Redesign with 50/50 null/retrieval split, and separate loss terms for
missed retrievals vs false retrievals. The degenerate solution reveals a task design
flaw, not a controller capability.

---

### 2026-03-08 | exp_3_2 | REFUTED
**Write Gate Collapse Detection**

Regime A (no signal): stable at 19.3% — no collapse.
Regime D (anti-collapse penalty): collapsed to 95.1% ALWAYS-write.

**Surprising:** The anti-collapse penalty caused the opposite collapse. The
unconstrained gate found a stable natural equilibrium. The assumption that gates
collapse without explicit anti-collapse signal was wrong, at this scale.

**Changes:** Don't add gate regularizers before understanding the natural equilibrium.
Investigate why the unconstrained gate settles at ~20% — is this a property of the
task or the model size?

---

### 2026-03-08 | exp_2_9 | REFUTED
**Retrieval vs Storage Compression Objectives**

Both compressors: acc@1 = 1.000. Reconstruction slightly better for A (0.414 vs 0.397).

**Surprising:** The task didn't discriminate between objectives. 5% query noise was
too easy for 8x compression to fail on. Both objectives trained perfectly adequate
retrievers.

**Changes:** Re-run at 64x compression where the ratio curve showed real degradation.
The objective discrimination likely only manifests when compression is severe enough
to require genuine tradeoffs.

---

### 2026-03-08 | exp_2_1 | INCONCLUSIVE
**Compression Ratio Curve**

The curve is non-monotonic. 2x–8x: cosine sim ~0.052 (near zero).
16x: cosine sim 0.219 (peak — more compression is better).
Gradual decline from 16x to 100x. No catastrophic cliff.

**Surprising:** Low compression ratios train poorly. Large bottlenecks give the
autoencoder enough room to be lazy — high-capacity models with insufficient training
budget fail to learn. The 16x sweet spot shows that moderate bottleneck constraints
act as useful regularization.

**Changes:** The question is not "where does quality cliff?" but "why does
underconstrained compression fail to learn?" Investigate the training dynamics at
2x–8x: are the gradients too sparse or too small? Does more training or a lower LR
fix the low-compression regime?

---

### 2026-03-08 | exp_1_5 | SUPPORTED (weakly)
**Write Signal Ablation**

Ranking: learned (0.124) > attention (0.121) > random (0.120) > surprise (0.118).
All policies clustered between 11.8–12.4%. Delta of +0.003 for learned.

**Surprising:** Surprise-weighted writing is the worst policy. High perplexity
tokens — the ones with the most "information" by many definitions — are not the
ones most worth remembering for downstream retrieval. The surprise signal is
anticorrelated with retrieval value.

**Changes:** Investigate the surprise anti-correlation further. Does this hold across
task types? If surprise is reliably anti-predictive of retrieval value, it could be
used as a *negative* signal — write the low-surprise, predictable tokens that form
the stable context. This inverts a common assumption.

---

## Meta-Observations (across all phase 1 experiments)

**The natural equilibrium:** Both write and read gates settle at ~15–20% activity
rates when trained on task loss alone. This wasn't predicted and deserves focused
investigation. Is ~16% the right rate for these tasks, or is it a model-size artifact?

**Regularizers are counterproductive:** Anti-collapse penalties, sparsity regularizers,
and coverage bonuses all produced worse behavior than no regularization in these
experiments. The correct approach may be task design rather than loss engineering.

**REINFORCE is ruled out:** Zero gradient norm through the encoder. This narrows
the differentiability design space to Gumbel-softmax and STE. Gumbel is preferred.

**Degenerate solutions dominate:** When a shortcut exists, the model finds it.
Task design is the primary tool for obtaining meaningful results, not loss design.
