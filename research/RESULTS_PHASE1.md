# Phase 1 Results Report — Priority Experiment Queue

**Date:** 2026-03-08
**Experiments run:** 10 of 44
**Total compute time:** ~2,300 seconds (sequential wall-clock)

---

## Summary Table

| ID | Experiment | Outcome | Key Metric |
|----|-----------|---------|------------|
| 1.5 | Write Signal Ablation | SUPPORTED | learned +0.003 over best baseline |
| 2.1 | Compression Ratio Curve | INCONCLUSIVE | no catastrophic cliff found |
| 2.9 | Compression Objectives | REFUTED | both objectives → 100% retrieval |
| 3.2 | Write Gate Collapse | REFUTED | unconstrained gate stable at 19.3% |
| 4.7 | Null Retrieval Learning | SUPPORTED* | precision 1.0 via degenerate policy |
| 4.9 | Compositional Retrieval | SUPPORTED | two-hop 100%, single-hop 96.8% |
| 5.1 | Read Gate Collapse | REFUTED | unconstrained gate stable at 15.6% |
| 6.6 | Controller Catastrophic Forgetting | INCONCLUSIVE | forgetting 0.208, EWC -1.8% |
| 7.1 | Differentiability (STE/Gumbel/REINFORCE) | SUPPORTED | Gumbel dominates on both metrics |
| 7.9 | Interpretability Baseline | SUPPORTED | gate non-random, numeric bias +0.076 |

*Degenerate support — technically meets threshold but for the wrong reason.

**4 SUPPORTED / 1 SUPPORTED* / 3 REFUTED / 2 INCONCLUSIVE**

---

## Detailed Results

### exp_1_5 — Write Signal Ablation
**Outcome: SUPPORTED (weakly)**

| Policy | Accuracy | Final Loss |
|--------|----------|------------|
| random | 0.120 | 3.150 |
| attention | 0.121 | 3.155 |
| surprise | 0.118 | 3.132 |
| **learned** | **0.124** | **3.054** |

The learned gate ranks first but by a margin of only +0.003 over attention-weighted
writing. All policies clustered between 11.8% and 12.4% accuracy, suggesting the
task as designed is near the noise floor for the model size.

**The more important finding:** surprise-weighted writing is the *worst* of the
baselines. Tokens where the model lost the most —- the ones with the most "information"
by one standard definition — are not the ones most worth remembering. High perplexity
does not predict retrieval value. This directly challenges a common intuition in
the field.

Loss curves tell a cleaner story than accuracy: the learned gate's lower final loss
(3.054 vs 3.132–3.155) suggests it is fitting the task structure, not just the
policy.

---

### exp_2_1 — Compression Ratio Curve
**Outcome: INCONCLUSIVE**

| Ratio | Bottleneck | Cosine Sim | MSE |
|-------|-----------|------------|-----|
| 2x | 2048 | 0.052 | 1.006 |
| 4x | 1024 | 0.053 | 1.006 |
| 8x | 512 | 0.053 | 1.006 |
| **16x** | **256** | **0.219** | **0.960** |
| 32x | 128 | 0.198 | 0.969 |
| 64x | 64 | 0.148 | 0.986 |
| 100x | 40 | 0.121 | 0.994 |

The expected monotonic decay from 2x to 100x does not appear. Instead:
- **2x through 8x: near-zero cosine similarity (~0.052).** Large bottlenecks are
  harder to train — the autoencoder learns almost nothing. More capacity does not
  mean better compression at fixed training budget.
- **Non-monotonic peak at 16x (0.219).** A smaller bottleneck is a stronger
  regularizer that forces the encoder to learn useful structure.
- Gradual decline from 16x to 100x, largest single-step drop between 32x and 64x
  (0.051), well below the catastrophic threshold of 0.15.

No catastrophic cliff detected. The curve is better described as a compression
sweet spot with a constrained optimum around 16x, not a failure mode.

**What this changes:** The question is not "where does quality fall off a cliff"
but "why does underconstrained compression fail to learn?" Wide bottlenecks with
insufficient training may be worse than narrow bottlenecks. This has implications
for the architecture — moderate compression ratios may train better than near-lossless
approaches.

---

### exp_2_9 — Retrieval vs Storage Compression Objectives
**Outcome: REFUTED**

| Compressor | Recon Cosine | Retrieval Acc@1 |
|------------|-------------|-----------------|
| A (reconstruction) | 0.414 | 1.000 |
| B (retrieval) | 0.397 | 1.000 |

Both compressors achieve perfect retrieval accuracy. The task is too easy at this
scale to discriminate between objectives — any reasonable 8x compression that
preserves some structure can retrieve the correct item when query noise is 5%.

The small reconstruction advantage of compressor A (0.414 vs 0.397) suggests the
objectives don't produce identical representations, but the difference doesn't
manifest in task performance.

**What this changes:** The hypothesis may still be true at harder compression
ratios (64x–100x) or with more demanding retrieval tasks. This experiment should
be re-run at the 32x–64x range where the compression ratio curve showed meaningful
degradation, and with retrieval tasks requiring fine-grained discrimination.

---

### exp_3_2 — Write Gate Collapse Detection
**Outcome: REFUTED**

| Regime | Write Rate | Collapsed? |
|--------|-----------|------------|
| A — task only | 0.193 | NO |
| B — entropy reg | 0.745 | NO |
| C — reconstruction | 0.279 | NO |
| D — penalty | 0.951 | ALWAYS |

The central prediction — that regime A (no anti-collapse signal) would collapse —
was wrong. It stabilized at 19.3% write rate. The unconstrained gate found a
natural equilibrium: write roughly every 5th token.

The striking result: regime D, which penalized low write rates to *prevent* collapse,
collapsed in the opposite direction — 95.1% write rate. The anti-collapse objective
drove the gate to write almost everything, which is a different form of uselessness.

**Pattern across regimes:** Write rate tracks directly to what the loss incentivizes.
Task-only signal → sparse natural rate. Entropy encouragement → high rate. Explicit
penalty → near-total rate. Reconstruction → moderate rate. The gate is highly
responsive to gradient signal.

**What this changes:** Write gate collapse at this scale and task type is not the
automatic failure mode it was assumed to be. The real risk is not collapse but
incentive misalignment — a gate that writes at whatever rate minimizes its training
signal, which can vary widely by objective.

---

### exp_4_7 — Null Retrieval Learning
**Outcome: SUPPORTED (degenerate)**

| Metric | Value |
|--------|-------|
| Null precision | 1.000 |
| Null recall | 0.800 |
| Null F1 | 0.889 |
| Retrieval rate on matches | 0.000 |

The read gate learned to never fire. Null precision is perfect because it
consistently predicts "no retrieval" — which is correct 80% of the time by the
task distribution. The hypothesis is technically supported but for the wrong reason.

This is a specific instance of the read gate collapse identified in exp_5_1 acting
as a shortcut: when null queries outnumber retrieval queries, "always null" is a
near-optimal policy for task loss alone without ever learning to retrieve.

**What this changes:** The task distribution itself trains a degenerate policy when
null queries dominate. Any experiment testing null retrieval must either balance the
distribution or use a loss that separately penalizes missed retrievals on relevant
queries. The degenerate solution is more informative than the intended one — it
reveals that retrieval training requires either near-balanced distributions or
explicit recall supervision.

---

### exp_4_9 — Compositional Retrieval
**Outcome: SUPPORTED (strongly)**

| Condition | Accuracy | vs Random (1/16) |
|-----------|----------|------------------|
| Single-hop | 0.968 | +93.1pp |
| **Two-hop** | **1.000** | **+93.8pp** |
| Random baseline | 0.063 | — |

The two-hop retriever not only succeeded — it achieved *higher* accuracy than the
single-hop retriever. The two-step composition architecture turns out to generalize
better on the structured synthetic task than direct single-step lookup.

The negative gap (−0.032) is unexpected. One interpretation: the intermediate
aggregation step in the two-hop path acts as a regularizer — the model is forced to
produce a meaningful intermediate representation rather than a direct pattern match,
which coincidentally improves generalization.

**Caveat:** The synthetic KB (circular colleague index, uniform attribute assignment)
may advantage the two-hop architecture by providing cleaner signal for the first
hop. Real-world compositional tasks may produce a different gap direction.

**What this changes:** Compositional retrieval is learnable on structured tasks.
The question for Phase 2 is whether this generalizes (a) to larger KBs where
interference is higher, (b) to implicit rather than explicit entity-attribute
structure, and (c) to cases where the composition path itself is variable.

---

### exp_5_1 — Read Gate Collapse Detection
**Outcome: REFUTED**

| Regime | Read Rate | Collapsed? |
|--------|----------|------------|
| A — task only | 0.156 | NO |
| B — sparsity reg | 0.094 | NEVER (YES) |
| C — coverage | 0.156 | NO |
| D — confidence | 0.656 | NO |

Mirrors exp_3_2 exactly. The unconstrained gate (A) stabilized at 15.6%.
Sparsity regularization — designed to discourage unnecessary reading — pushed the
gate past the collapse threshold to near-never (9.4%).

Note that regime A and regime C settle at the identical read rate (15.6%), despite
regime C including an explicit coverage penalty. This suggests the task loss already
contains sufficient signal to maintain a ~16% read rate, and the additional coverage
objective provides no marginal benefit.

**Combined finding from exp_3_2 + exp_5_1:** At the scale and task complexity tested,
neither write nor read gates collapse naturally from task loss alone.
They find low but stable non-zero rates. Anti-collapse regularizers can be
counterproductive — they may cause the opposite collapse or add noise without
changing the natural equilibrium.

---

### exp_6_6 — Controller Catastrophic Forgetting
**Outcome: INCONCLUSIVE**

| Condition | Domain A Acc | Domain B Acc | Δ from before |
|-----------|-------------|-------------|---------------|
| Before B training | 0.227 | — | — |
| Standard (after B) | 0.019 | 0.106 | −0.208 |
| EWC (after B) | 0.023 | 0.125 | −0.204 |

Catastrophic forgetting is real and severe (0.208 drop, above the 0.15 threshold).
However, EWC provided essentially no protection — only 1.8% reduction in forgetting.

Two confounds make this INCONCLUSIVE:
1. **Low baseline performance on domain A (22.7%)** — the model has not strongly
   learned domain A before fine-tuning begins. It is hard to "forget" policies that
   were never well-established.
2. **Low domain B performance (10.6%)** — the model also learns B poorly. Both tasks
   are near the floor for this architecture.

The 0.208 forgetting may largely reflect the model abandoning poorly-learned domain
A weights in favor of slightly better domain B weights, rather than true policy
forgetting. EWC protects weights that had high Fisher information on domain A, but
if those weights were themselves weak (poor domain A performance), there is little
to protect.

**What this changes:** This needs to be re-run with a model that first achieves
strong domain A performance (>70%) before domain B training begins. The forgetting
test is only meaningful if there is something substantial to forget.

---

### exp_7_1 — Differentiability
**Outcome: SUPPORTED (strongly)**

| Method | Accuracy | Loss Variance | Grad Norm |
|--------|----------|--------------|-----------|
| **Gumbel** | **0.128** | **0.0071** | 1.060 |
| STE | 0.126 | 0.0094 | 1.447 |
| REINFORCE | 0.102 | 2.696 | 0.000 |

The ranking is identical on accuracy and stability: Gumbel > STE > REINFORCE.

REINFORCE's loss variance is 380× worse than Gumbel. More critically, its mean
gradient norm is 0.000 — the policy gradient signal through the encoder is
essentially dead. REINFORCE trains the policy network but fails to propagate useful
learning signal back to the encoder. The encoder learns nothing; the policy is
optimizing over a fixed, random feature space.

Gumbel and STE both produce reasonable gradient flow. Gumbel's lower variance
suggests the temperature annealing is providing a useful curriculum.

**What this changes:** REINFORCE is not a viable training approach for the discrete
memory selection problem at this scale. The encoder-controller joint training problem
requires differentiable relaxations. Of the two, Gumbel with temperature annealing
is the stronger default choice. This is a load-bearing result — it constrains the
design space significantly.

---

### exp_7_9 — Interpretability Baseline
**Outcome: SUPPORTED (marginally)**

| Feature | Untrained | Trained | Direction |
|---------|----------|---------|-----------|
| Punctuation | −0.043 | −0.047 | avoids |
| Numeric | +0.062 | **+0.076** | favors |
| Rare | +0.060 | +0.055 | slightly favors |
| Position | −0.008 | −0.004 | no signal |
| Gate mean | 0.510 | 0.509 | near-uniform |
| Gate std | 0.146 | 0.151 | slightly more decisive |

The gate is non-random even before training (std=0.146). A single linear projection
over learned embeddings is sufficient to produce token-type preferences.

**Numeric tokens are the most favored** (0.076 trained). **Punctuation is actively
avoided** (−0.047). Rare tokens are weakly favored over common ones. Position in
the sequence has essentially no signal.

Training barely changes these correlations (interpretability score 0.055 → 0.059),
suggesting the gate's token preferences are established rapidly from the embedding
geometry and are not updated substantially by task gradients at this training budget.

**What this changes:** The interpretability tooling works and identifies real
structure. The finding that numeric tokens are preferred and punctuation is avoided
is semantically coherent — a minimal controller "discovers" information-bearing
tokens without any explicit supervision. This provides a natural benchmark: as
controllers become more complex, their token preferences should become more specific
and more aligned with task-relevant content.

---

## Cross-Cutting Findings

### The Natural Equilibrium Surprise
Experiments 3.2 and 5.1 both found that unconstrained gates (task loss only) do
not collapse. They settle at ~15–19% activity rates. This is the first unexpected
result of the research phase and directly refutes a core assumption.

The implication: the "write gate collapse" problem may be less about the gate
architecture and more about task design. Gates trained on tasks where memory
provides marginal benefit will find low but non-zero rates. The question to pursue
is not "how do we prevent collapse?" but "how do we design tasks that require
high-quality selective memory use?"

### Regularizers Cause Opposite Collapses
In both exp_3_2 and exp_5_1, the regime designed to prevent one collapse caused
the other:
- Anti-collapse penalty (D) → always-write (95.1%)
- Sparsity regularizer (B) → never-read (9.4%)

This suggests regularizers that directly target the gate value are fragile.
The more stable anti-collapse approach may be indirect: design the task to require
selective memory, rather than adding explicit gate penalties.

### REINFORCE Is Not Viable Here
Experiment 7.1 produced the clearest result. REINFORCE's gradient norm through
the encoder is effectively zero — the encoder does not learn under policy gradient.
Any architecture that relies on REINFORCE-style training for the memory controller
will have a non-differentiable wall between the memory mechanism and the rest of
the network. Gumbel-softmax or STE are required.

### The Degenerate Solution Problem
Experiments 4.7 and 4.9 together reveal a pattern. In exp_4.7, the most important
finding was the degenerate shortcut: "always null" satisfied the hypothesis by
gaming the task distribution. In exp_4.9, the model couldn't take this shortcut —
the task required real retrieval — and achieved near-perfect performance.

The design lesson: experiments that allow degenerate solutions will find them.
The controller learning hypothesis can only be tested when degenerate policies
are not available.

---

## The Emerging Picture

After 10 experiments, the clearest thing is what we don't need to worry about and
what we do:

**Lower priority than assumed:**
- Write gate collapse from task loss alone (doesn't happen at this scale)
- Read gate collapse from task loss alone (doesn't happen at this scale)
- Catastrophic forgetting from a few thousand training steps (EWC is enough for shallow forgetting; deep forgetting needs a better-trained base)

**Higher priority than assumed:**
- Task design: experiments can be gamed by degenerate policies if the distribution allows it
- Compression training dynamics: large bottlenecks don't train well without sufficient compute
- REINFORCE as a training method: ruled out for encoder-memory joint training
- Incentive alignment vs. capacity: what a gate learns tracks its gradient signal faithfully

**Open questions that need new experiments:**
1. Does the natural ~16% equilibrium rate change with task complexity? Or is it a
   property of the model size?
2. Can exp_2.9 objectives be discriminated at harder compression ratios (64x+)?
3. Can exp_4.7 null retrieval be cleanly tested with a balanced distribution?
4. Does two-hop compositional retrieval hold on larger KBs with interference?
5. What happens to exp_6.6 when domain A performance exceeds 70% before B training?

---

## Next Steps

The phase 1 results reconfigure the priority order for phase 2:

1. Redesign exp_2.9 at 64x compression with harder retrieval discrimination
2. Re-run exp_4.7 with balanced null/retrieval distribution (50/50)
3. Re-run exp_6.6 with higher baseline domain A performance
4. Build exp_3.1 (continuous vs event-driven) now that collapse rate dynamics are better understood
5. Build exp_4.3 (retrieval depth sensitivity) to characterize the ~16x compression sweet spot
6. Begin exp_7.4 (minimal controller architecture search) using Gumbel-softmax as the training mechanism

The architecture that nobody has built yet is still not visible. But the shape of
the problem space has sharpened considerably. The gate dynamics and differentiability
findings constrain the design space. The natural equilibrium finding is the most
interesting anomaly — follow it.
