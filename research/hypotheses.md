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
