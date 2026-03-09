# Hypotheses

Each experiment has a single falsifiable hypothesis. A result is only useful if
it can prove the hypothesis false. Negative results are recorded with the same
rigor as positive ones.

---

## Category 1: What To Write

**H-1.1 — Relevance Signal Baseline**
Attention weight is a valid proxy for memory importance and correlates with what
a human would consider "worth remembering."
*Status: UNTESTED*

**H-1.2 — Surprise as a Write Signal**
A memory built from high-perplexity tokens supports better downstream retrieval
than one built from attention weights.
*Status: UNTESTED*

**H-1.3 — Gradient Magnitude as Write Signal**
Storing representations where gradient magnitude is highest produces a memory
store that generalizes better than attention-based selection.
*Status: UNTESTED*

**H-1.4 — Contrastive Write Selection**
Diversity-driven storage (maximally different entries) outperforms importance-driven
storage on recall tasks.
*Status: UNTESTED*

**H-1.5 — Write Signal Ablation**
A learned write gate outperforms random write, attention-weighted write, and
surprise-driven write on associative recall tasks. (PRIORITY)
*Status: UNTESTED*

**H-1.6 — Semantic Deduplication at Write Time**
Deduplication via cosine similarity improves retrieval precision without causing
information loss significant enough to hurt downstream task performance.
*Status: UNTESTED*

**H-1.7 — Write Frequency vs Write Quality**
For a fixed storage budget, infrequent writes with high compression outperform
frequent writes with low compression on long-document QA tasks.
*Status: UNTESTED*

**H-1.8 — Hierarchical Write Decisions**
A two-stage write decision (coarse gate then fine-grained selection) outperforms
a single write gate.
*Status: UNTESTED*

---

## Category 2: How To Write (Compression)

**H-2.1 — Compression Ratio vs Recall Fidelity Curve**
There exists a compression ratio threshold beyond which recall fidelity degrades
catastrophically rather than gracefully. (PRIORITY)
*Status: UNTESTED*

**H-2.2 — Autoencoder vs Attention-based Compression**
Attention-based compression produces more retrievable representations than
autoencoder compression on inferential recall tasks (though results may diverge
by task type).
*Status: UNTESTED*

**H-2.3 — Lossy vs Lossless Memory Representations**
A controller can learn without supervision which information should be stored
exactly (numbers, names) vs. approximately (themes, sentiment).
*Status: UNTESTED*

**H-2.4 — Chunk Size Sensitivity**
There exists an optimal chunk size for compression beyond which quality degrades
independent of compression ratio.
*Status: UNTESTED*

**H-2.5 — Structured vs Unstructured Compression**
Compressing into a structured representation (key-value, slot-based) improves
retrieval over compressing into a flat dense vector.
*Status: UNTESTED*

**H-2.6 — Compression Generalization**
A compressor trained on domain A produces meaningfully worse retrieval on domain B
than a domain-B-trained compressor, indicating compression overfits to domain.
*Status: UNTESTED*

**H-2.7 — Iterative Compression**
A hierarchy of increasingly abstract memory levels can be built by iterative
compression without catastrophic information loss at each stage.
*Status: UNTESTED*

**H-2.8 — Compression Under Distribution Shift**
A compressor degrades gracefully (not catastrophically) when input distribution
shifts significantly mid-context.
*Status: UNTESTED*

**H-2.9 — Retrieval-Oriented vs Storage-Oriented Compression**
Minimizing reconstruction loss and maximizing downstream retrieval accuracy are
fundamentally different objectives and produce measurably different representations.
(PRIORITY)
*Status: UNTESTED*

---

## Category 3: When To Write

**H-3.1 — Continuous vs Event-Driven Writing**
Event-driven writing (learned gate) produces better memory coverage than writing
every N tokens for a fixed storage budget.
*Status: UNTESTED*

**H-3.2 — Write Gate Collapse**
A learned write gate trained without explicit anti-collapse objectives will learn
to never write within N training steps on standard tasks. (PRIORITY)
*Status: UNTESTED*

**H-3.3 — Write Timing vs Content Quality**
Writing later in a context (more processed representations) outperforms writing
early (raw representations) for inferential downstream tasks.
*Status: UNTESTED*

**H-3.4 — Boundary Detection as Write Trigger**
Semantic-boundary-triggered writing outperforms fixed-interval writing on
long-document tasks with clear topical structure.
*Status: UNTESTED*

**H-3.5 — Write Latency Sensitivity**
Downstream retrieval quality degrades measurably when write latency exceeds a
specific token distance threshold.
*Status: UNTESTED*

**H-3.6 — Retroactive Writing**
A controller can learn to retroactively write tokens it initially skipped once
later context reveals their importance.
*Status: UNTESTED*

**H-3.7 — Write Budget Allocation**
A controller given a fixed write budget per context learns to allocate that budget
non-uniformly in a way that improves performance over uniform allocation.
*Status: UNTESTED*

---

## Category 4: What To Read

**H-4.1 — Query Formulation Quality**
A dedicated query formulation module outperforms direct use of the current hidden
state as a retrieval query.
*Status: UNTESTED*

**H-4.2 — Single vs Multi-Vector Retrieval**
Multi-vector retrieval captures more relevant memory content than single-vector
retrieval on tasks with multi-faceted information needs.
*Status: UNTESTED*

**H-4.3 — Retrieval Depth Sensitivity**
There exists an optimal retrieval depth (top-k) beyond which additional retrieved
entries introduce more noise than signal.
*Status: UNTESTED*

**H-4.4 — Soft vs Hard Retrieval**
Soft retrieval (weighted average) produces more stable training than hard retrieval
(discrete selection), though hard retrieval may achieve higher peak task performance.
*Status: UNTESTED*

**H-4.5 — Cross-Level Retrieval**
Simultaneous cross-tier retrieval achieves better recall than sequential cascading
retrieval (working → episodic → semantic).
*Status: UNTESTED*

**H-4.6 — Retrieval by Reconstruction vs Similarity**
For tasks requiring exact recall, similarity-based retrieval outperforms
reconstruction-based retrieval. For inferential completion, the relationship inverts.
*Status: UNTESTED*

**H-4.7 — Null Retrieval Learning**
A learned read gate can be trained to return null (no retrieval) on tasks where
most queries have no relevant memory content, without explicit null supervision.
(PRIORITY)
*Status: UNTESTED*

**H-4.8 — Retrieval Interference**
Retrieval quality degrades non-linearly as the number of near-duplicate entries
in memory increases, with a specific saturation point.
*Status: UNTESTED*

**H-4.9 — Compositional Retrieval**
A learned retrieval mechanism can be trained to retrieve two separate memory entries
and compose them to answer questions neither entry answers alone. (PRIORITY)
*Status: UNTESTED*

---

## Category 5: When To Read

**H-5.1 — Read Gate Collapse**
A learned read gate trained without explicit anti-collapse objectives will learn
a degenerate policy (always read or never read) within N training steps. (PRIORITY)
*Status: UNTESTED*

**H-5.2 — Read Frequency vs Task Performance**
Optimal read frequency is task-dependent and cannot be determined by a single
fixed schedule across task types.
*Status: UNTESTED*

**H-5.3 — Predictive Read Triggering**
Anticipatory retrieval (predicting retrieval need before it arises) improves
end-to-end latency without measurably hurting task quality.
*Status: UNTESTED*

**H-5.4 — Read vs Recompute Decision**
A controller can learn to prefer recomputation over retrieval for information
that is cheap to recompute and prefer retrieval for information that is expensive.
*Status: UNTESTED*

**H-5.5 — Cascading Read Depth**
Confidence-gated cascading retrieval (shallow first, deeper only if low confidence)
matches full-depth retrieval quality at significantly lower average compute cost.
*Status: UNTESTED*

**H-5.6 — Read Suppression Under High Confidence**
Suppressing memory reads when next-token prediction confidence exceeds a threshold
costs less than 1% task quality on standard benchmarks.
*Status: UNTESTED*

**H-5.7 — Attention-Memory Arbitration**
When local attention and external memory produce conflicting predictions, a learned
arbitration policy outperforms both fixed-priority policies (always prefer attention,
always prefer memory).
*Status: UNTESTED*

---

## Category 6: How To Forget

**H-6.1 — Eviction Policy Comparison**
A learned importance-scored eviction policy significantly outperforms LRU on
tasks requiring retention of low-frequency but high-importance information.
*Status: UNTESTED*

**H-6.2 — Forgetting as Compression**
Graceful degradation via iterative compression outperforms hard eviction for
long-context tasks where storage budget is the binding constraint.
*Status: UNTESTED*

**H-6.3 — Selective Forgetting Under Distribution Shift**
A controller can learn to evict domain-mismatched memories when input distribution
shifts, without explicit domain labels.
*Status: UNTESTED*

**H-6.4 — Protected Memory Slots**
A controller can learn which memories deserve protection (never evict) without
explicit supervision, and performance degrades predictably as protected set size
grows beyond an optimal threshold.
*Status: UNTESTED*

**H-6.5 — Forgetting Curve Mimicry**
A biologically-inspired memory decay function (Ebbinghaus curve) improves
long-horizon task performance compared to instant eviction.
*Status: UNTESTED*

**H-6.6 — Catastrophic Forgetting of the Controller Itself**
The memory controller itself (as a neural network) suffers measurable catastrophic
forgetting of its learned policies when exposed to a new domain. (PRIORITY)
*Status: UNTESTED*

**H-6.7 — Write-Evict Coupling**
Joint optimization of write and evict decisions outperforms treating them as
independent operations on tasks where storage pressure is constant.
*Status: UNTESTED*

**H-6.8 — Memory Consolidation**
Periodic offline consolidation (merging multiple entries into one higher-level
representation) improves long-horizon performance without active context.
*Status: UNTESTED*

---

## Category 7: Cross-Cutting

**H-7.1 — End-to-End Controller Differentiability**
Gumbel-softmax relaxation produces more stable training than straight-through
estimators, and both outperform RL-based approaches for discrete memory selection.
(PRIORITY)
*Status: UNTESTED*

**H-7.2 — Controller Overhead Budget**
There exists a maximum controller complexity (measured in FLOPs) beyond which
the controller's overhead exceeds its efficiency contribution.
*Status: UNTESTED*

**H-7.3 — Controller Generalization Across Task Types**
A controller trained on factual QA learns memory management policies that
generalize to reasoning tasks but not to generation tasks.
*Status: UNTESTED*

**H-7.4 — Minimal Controller Architecture Search**
Meaningful memory management behavior requires at minimum two layers of
non-linearity in the controller network.
*Status: UNTESTED*

**H-7.5 — Controller Stability Under Scale**
A controller's learned policy trained at 100M parameters does not transfer
directly to a 1B parameter model without additional fine-tuning.
*Status: UNTESTED*

**H-7.6 — Adversarial Memory Probing**
The memory controller is measurably vulnerable to inputs designed to maximize
write activity, and this vulnerability does not self-correct during training.
*Status: UNTESTED*

**H-7.7 — Memory Controller as Bottleneck Identification**
Write quality (not read quality, compression ratio, or eviction policy) is the
first performance bottleneck encountered during controller training.
*Status: UNTESTED*

**H-7.8 — Joint vs Sequential Controller Training**
Curriculum training (one component at a time) produces more stable controller
behavior than joint training from the start.
*Status: UNTESTED*

**H-7.9 — Controller Interpretability Baseline**
The controller's write and read decisions are interpretable (non-random, correlating
with human-meaningful features) in their simplest form before any task-specific
training. (PRIORITY)
*Status: UNTESTED*

---

## Category 8: Mechanistic Investigations (Phase 2)

**H-8.1 — Attention Anti-Correlation is Normalization Artifact**
The Pearson r = −0.503 correlation between attention weight and write-gate importance
is an artifact of softmax normalization; raw pre-softmax dot products will show
positive or near-zero correlation with the same importance signal.
*Status: UNTESTED*

**H-8.2 — Gate Equilibrium Scales With Task Demand**
The natural ~16–20% write-gate equilibrium observed across 3 independent experiments
scales with task memory demand: higher KV-pair density drives higher equilibrium
write rate (Pearson r > 0.8 across difficulty levels).
*Status: UNTESTED*

**H-8.3 — Write-Evict Collapse is Gradient Aliasing**
The write-evict collapse (r = 0.990) observed in exp_6_7 is caused by gradient
aliasing from a shared task loss; oracle pre-training on independent labels before
joint fine-tuning breaks the collapse and produces r < 0.5.
*Status: UNTESTED*

**H-8.4 — Write Gate Exploits Position as Content Proxy**
The learned write gate relies on token position as a proxy for content importance;
removing positional embeddings degrades gate quality significantly (accuracy drop
> 0.5%) while position correlation drops below 0.05.
*Status: UNTESTED*

---

## Category 9: Inconclusive Redesigns (Phase 2)

**H-9.1 — Compression Objective Gap at 64× Compression**
At 64× compression with a 100-way gallery, an InfoNCE retrieval objective produces
>15% absolute Acc@1 gain over an MSE reconstruction objective, confirming that
the Phase 1 result was inconclusive only because 8× was too easy.
*Status: UNTESTED*

**H-9.2 — Null Retrieval With Balanced Distribution**
A memory controller trained on a 50/50 null/retrieval split achieves F1 > 0.60
on both null detection and genuine retrieval, demonstrating real selectivity rather
than distributional shortcut exploitation.
*Status: UNTESTED*

**H-9.3 — EWC Effectiveness With Strong Domain-A Baseline**
Elastic Weight Consolidation reduces forward transfer forgetting by > 50% relative
to unregularized sequential training when the domain-A baseline accuracy exceeds 70%.
*Status: UNTESTED*

**H-9.4 — Protected Slots Interior Optimum Above K=5**
The write-accuracy peak for protected memory slots lies at K > 5 when MEMORY_SLOTS = 12;
the Phase 1 result (K=5 best, at K_max) was truncated and does not represent a true
interior optimum.
*Status: UNTESTED*

**H-9.5 — Write Budget Non-Uniform Allocation on Biased Task**
A learned write budget produces measurably non-uniform slot allocation (Gini > 0.50)
and accuracy gain > 5% over uniform allocation when the task structure requires
concentrated writes in one segment of the sequence.
*Status: UNTESTED*

---

## Category 10: Retroactive Writing Mechanism (Phase 2)

**H-10.1 — Retroactive Benefit Decays With Lookahead Window**
The accuracy benefit of retroactive writing falls below 5% of its full-sequence
value when the lookahead window is restricted to fewer than 6 tokens of right context.
*Status: UNTESTED*

**H-10.2 — Retroactive Gain is Primarily New-Write, Not Re-Encoding**
More than 80% of the combined retroactive accuracy gain comes from writing previously
skipped tokens (new-write), not from re-encoding already-written tokens with updated
context (overwrite).
*Status: UNTESTED*

**H-10.3 — Retroactive Gain Scales With Sequence Length**
The absolute accuracy benefit of retroactive writing increases monotonically with
sequence length (Pearson r > 0.8) across seq_len ∈ {24, 32, 48, 64}.
*Status: UNTESTED*

---

## Category 11: Read Bottleneck Interventions (Phase 2)

**H-11.1 — Two-Step Query Former Reduces Read Bottleneck**
A two-step query former (linear projection followed by cross-attention over the
last 4 hidden states) improves oracle read accuracy by > 10% compared to a single
linear query projection, directly addressing the bottleneck identified in exp_7_7.
*Status: UNTESTED*

**H-11.2 — Read-Before-Write Deduplication Improves Retrieval**
Suppressing writes for tokens with cosine similarity > 0.8 to existing memory
improves retrieval F1 by > 3% without reducing recall by more than 5%.
*Status: UNTESTED*

**H-11.3 — Optimal Suppression Threshold Varies By Task Type**
The optimal read-suppression confidence threshold from exp_5_6 varies by more than
0.15 across factual, pattern, and completion retrieval tasks, indicating that a
single universal threshold cannot be optimal.
*Status: UNTESTED*

---

## Category 12: Compression Hard Regimes (Phase 2)

**H-12.1 — Retrieval Objective Dominates at 64× Compression**
At 64× token compression with a 100-way gallery, a retrieval (InfoNCE) training
objective outperforms a reconstruction (MSE) objective by > 15% absolute Acc@1.
*Status: UNTESTED*

**H-12.2 — Extended LR Warmup Rescues Gradient-Starved Compression**
The 2×–8× compression failure in exp_2_1 is gradient starvation from a too-wide
bottleneck; a cosine LR schedule with extended warmup recovers > 10% training
accuracy relative to constant LR at the same step budget.
*Status: UNTESTED*

---

## Category 13: Compositional Retrieval at Scale (Phase 2)

**H-13.1 — Two-Hop Regularization Persists With Interference**
The accuracy advantage of two-hop compositional retrieval (exp_4_9) persists
at a 64-entity knowledge base with 40% near-duplicate interference, with
the gap remaining ≥ the original 2-hop advantage on the interference subset.
*Status: UNTESTED*

**H-13.2 — Three-Hop Chain Is Feasible At Hidden Dim 64**
Three-hop chain retrieval is achievable with hidden_dim=64; accuracy degrades
less than 50% relative to two-hop accuracy on the same knowledge base.
*Status: UNTESTED*

---

## Category 14: System Integration (Phase 2)

**H-14.1 — Retroactive Write and Read Suppression Are Super-Additive**
Combining retroactive writing (cat10) and read confidence suppression (cat11)
produces accuracy gains that are super-additive (combined gain > sum of individual
gains + 1%), because each mechanism targets an independent bottleneck.
*Status: UNTESTED*

**H-14.2 — Write-First Curriculum Outperforms Joint Training**
Training the write gate for 1000 steps before adding read-head training produces
higher final accuracy than joint training from step 0, because the write gate first
learns a stable policy before the read head introduces competing gradients.
*Status: UNTESTED*

**H-14.3 — Cosine Gumbel Temperature Annealing Improves Accuracy**
Annealing Gumbel-softmax temperature from 1.0 to 0.1 via cosine schedule improves
final accuracy by > 2% compared to a constant temperature of 0.5, because gradual
discretization prevents early collapse to a near-deterministic policy.
*Status: UNTESTED*

---

## Category 15: Delta Rule / Associative Matrix Writes (Phase 3)

**H-15.1 — Delta Rule Outperforms Slot Write Via Interference Correction**
Delta rule matrix writes (M += (v − Mk^T/‖k‖²)k^T) outperform standard fixed-slot
writes by > 5% accuracy due to explicit interference correction, which prevents
over-writing when multiple keys share similarity.
*Status: UNTESTED*

**H-15.2 — Correction Term is Critical for Key-Interference Tasks**
Removing the correction term from the delta rule (degrading to Hebbian M += vk^T)
causes accuracy to drop > 10% on tasks where 50% of keys are deliberately set to
interfering (near-duplicate) values.
*Status: UNTESTED*

**H-15.3 — Energy-Gated Delta Rule Achieves Sparse Writes at Low Cost**
An energy-gated delta rule (write only when ΔE < 0) achieves > 90% of the accuracy
of continuous writes while reducing write rate to < 70%, because the energy
criterion naturally gates out redundant or interference-increasing writes.
*Status: UNTESTED*

**H-15.4 — Delta Rule Outperforms Outer-Product Write on Overwrite Tasks**
On tasks requiring key overwrite (same key appearing twice with different values),
delta rule writes outperform Larimar-style outer-product writes (M += vk^T) by
> 10% accuracy, because the correction term explicitly handles prior associations.
*Status: UNTESTED*

---

## Category 16: Online Gradient Descent Memory / Titans-Style (Phase 3)

**H-16.1 — Parametric MLP Memory Outperforms Slot Memory at Matched Budget**
A parametric MLP memory updated via 1 SGD step per token outperforms a fixed-slot
memory array at a matched parameter budget (≈256 params each) on associative recall,
because gradient steps enable finer-grained interference management than slot-level
writes.
*Status: UNTESTED*

**H-16.2 — Surprise-Gated Updates Recover Accuracy at Reduced Step Count**
A surprise-gated parametric memory (update skipped for low-surprise tokens, as
measured by L2 distance from running mean) achieves accuracy within 2% of full
updates while reducing update steps by > 40% at the optimal surprise threshold.
*Status: UNTESTED*

**H-16.3 — Parametric Memory Scales Better With Sequence Length Than Slot Memory**
Parametric MLP memory retains a higher fraction of peak accuracy as sequence length
scales from 24 to 96 tokens compared to slot-based memory (accuracy retention ratio
difference > 15%), because gradient steps adapt the representation to sequence length.
*Status: UNTESTED*

---

## Category 17: Prospective / Query-Conditioned Writing (Phase 3)

**H-17.1 — Query-Conditioned Write Gate Outperforms Context-Only Gate**
A write gate conditioned on a predicted future query type outperforms a context-only
gate by > 5% on multi-query-type tasks, because anticipating query structure enables
proactive selection of task-relevant tokens.
*Status: UNTESTED*

**H-17.2 — K-Token Lookahead Async Write Has an Optimal Lookahead Distance**
An asynchronous write gate that decides K steps before the write token uses lookahead
context to outperform a same-time gate by > 3% at some K ∈ {2, 4, 6}.
*Status: UNTESTED*

**H-17.3 — Prospective and Retroactive Writing are Redundant Mechanisms**
Combining prospective write (query-conditioned gate) and retroactive write (revision
gate) yields less than 1.5× the accuracy gain of either mechanism alone, because both
mechanisms compensate for the same forward-pass gate limitation.
*Status: UNTESTED*

**H-17.4 — Query-Conditioned Write Gain Scales With Query Predictability**
The accuracy advantage of query-conditioned writing correlates linearly (r > 0.85)
with the predictability of query type from context, because the write gate can only
exploit query information it can reliably predict.
*Status: UNTESTED*

---

## Category 18: Tiered Memory Architecture (Phase 3)

**H-18.1 — Two-Tier Memory Outperforms Flat Memory on Long-Context Tasks**
A two-tier memory (16-slot fast + 64-slot slow with learned demotion) outperforms
a flat 64-slot memory by > 5% on tasks where critical pairs are seeded early and
must survive 64-token contexts.
*Status: UNTESTED*

**H-18.2 — Learned Demotion Controller Discovers Frequency Over Recency**
A trained demotion controller learns a frequency-based policy rather than a
recency-based policy, producing positive correlation (> 0.15) between access
count and demotion probability and negative correlation (< −0.10) with recency.
*Status: UNTESTED*

**H-18.3 — Tiered Memory Has a Capacity Crossover Point**
Flat memory is more parameter-efficient below ~32 total slots; tiered architecture
outperforms flat above that threshold, with a crossover point in the range [16, 64].
*Status: UNTESTED*

**H-18.4 — Simultaneous Cross-Tier Retrieval Outperforms Sequential**
Parallel attention over both memory tiers simultaneously outperforms cascaded
sequential retrieval (fast first, then slow if confidence low) by > 3% accuracy,
because sequential retrieval introduces greedy commitment errors.
*Status: UNTESTED*

---

## Category 19: Sparse Hopfield Addressing (Phase 3)

**H-19.1 — Sparse Hopfield Retrieval Outperforms Soft Attention on Interference**
Sparse Hopfield retrieval (top-k=2 SparseMAP-style masking) outperforms standard
soft attention by > 5% precision@1 on tasks with 40% near-duplicate interference,
because hard zeros suppress spurious activation from interfering patterns.
*Status: UNTESTED*

**H-19.2 — Energy Write Criterion Produces Sparse Writes at Accuracy Gain**
Writing to Hopfield memory only when the energy change ΔE < 0 produces a write
rate < 35% while improving accuracy by > 3%, because the energy criterion
filters writes that would increase interference.
*Status: UNTESTED*

**H-19.3 — Sparse Hopfield Sustains Capacity Longer Before Cliff**
Sparse Hopfield addressing maintains retrieval accuracy for at least 2 more
patterns than dense soft attention before catastrophic capacity cliff, as measured
by the pattern count at which accuracy drops below 50%.
*Status: UNTESTED*

---

## Category 20: Three-Gate Coordinated Controller (Phase 3)

**H-20.1 — Write Sparsity Auxiliary Loss Improves Accuracy and Gate Health**
An L1 auxiliary loss targeting ~15% write rate equilibrium improves downstream
task accuracy while keeping write rate in the healthy range [5%, 35%], avoiding
the degenerate modes observed in exp_3_2.
*Status: UNTESTED*

**H-20.2 — Read Accuracy Auxiliary Loss Reduces Read Bottleneck**
An explicit cross-entropy auxiliary loss on oracle read accuracy (with detached
gradients to the write gate) reduces the read bottleneck identified in exp_7_7
more than implicit task-loss signal alone (oracle read accuracy gain > 5%).
*Status: UNTESTED*

**H-20.3 — Combined Auxiliary Losses Outperform Any Single Auxiliary**
A three-gate controller trained with all auxiliary losses (write sparsity + read
accuracy + forget usefulness) outperforms any single-auxiliary or no-auxiliary
baseline by > 2% accuracy, because each loss targets an independent failure mode.
*Status: UNTESTED*

**H-20.4 — Optimal Auxiliary Weight Range is [0.01, 0.1]**
Task accuracy is maximized at auxiliary loss weight λ ∈ [0.01, 0.1]; at λ < 0.01
the auxiliary has no effect, and at λ > 0.1 it dominates the task signal.
*Status: UNTESTED*

---

## Category 21: Feedforward Controller + Hindsight Distillation (Phase 3)

**H-21.1 — Feedforward Controller Achieves Higher Memory Utilization**
A feedforward-only controller (no recurrence) achieves higher external memory
utilization (measured by slot-access entropy) than an LSTM controller at equal
parameter count, because recurrent controllers can short-circuit memory access
via their hidden state.
*Status: UNTESTED*

**H-21.2 — Hindsight Oracle Labels Improve Write Gate Training Signal**
A write gate trained with hindsight oracle labels (retroactively marking which
writes were causally necessary for correct predictions) achieves higher task
accuracy than end-to-end gradient training on the same task and step budget.
*Status: UNTESTED*

**H-21.3 — Distilled Gate Outperforms End-to-End Learned Gate**
A write gate distilled from oracle hindsight labels via alternating BCE + task
training achieves higher accuracy than an end-to-end trained gate, because the
oracle supervision provides a cleaner gradient signal for sparse discrete decisions.
*Status: UNTESTED*

**H-21.4 — Feedforward Controller + Hindsight Distillation is Strongest Write Policy**
Combining a feedforward controller with hindsight oracle distillation produces
higher accuracy than any single-mechanism baseline (LSTM+task, FF+task, LSTM+oracle),
representing the strongest overall write policy tested in Phase 3.
*Status: UNTESTED*
