# drex Research — Master Results Report

**Generated:** 2026-03-08 20:04 UTC
**Experiments:** 57  |  **Seeds per experiment:** 42, 123, 777
**Total runs evaluated:** 171

## Overall Scoreboard

| Outcome | Count | % |
|---------|-------|---|
| ✓ SUPPORTED    | 16    | 28% |
| ~ INCONCLUSIVE | 21 | 37% |
| ✗ REFUTED      | 20      | 35% |
| ! ERROR        | 0        | 0% |

**Seed consistency:** 57/57 experiments gave the same verdict across all seeds. 0 inconsistent.

## Summary Table

| ID | Outcome | Consistent | Key Metric (mean ± std) | Notes |
|----|---------|------------|------------------------|-------|
| exp_1_1 | ✗ REFUTED | ✓ | attention_correlation=-0.503±0.000 | Pearson r=-0.5026. Attention acc=0.088 vs random=0.087 vs or… |
| exp_1_2 | ✗ REFUTED | ✓ | attention_acc=0.144±0.000 | Surprise acc=0.139, attention acc=0.144, no-memory baseline=… |
| exp_1_3 | ~ INCONCLUSIVE | ✓ | attention_acc=0.090±0.000 | Gradient acc=0.107, attention=0.090, random=0.109. Gap=+0.01… |
| exp_1_4 | ~ INCONCLUSIVE | ✓ | diversity_acc=0.138±0.000 | Diversity acc=0.138 vs importance acc=0.144. Gap=-0.0067. Di… |
| exp_1_5 | ✓ SUPPORTED | ✓ | acc_attention=0.121±0.000 | Learned gate delta over best baseline: +0.003. Ranking: ['le… |
| exp_1_6 | ~ INCONCLUSIVE | ✓ | coverage_delta=0.000±0.000 | Dedup: precision=0.104 (std=0.120), coverage=1.000 (std=1.00… |
| exp_1_7 | ~ INCONCLUSIVE | ✓ | freq_acc=0.066±0.000 | Infreq acc=0.058 vs freq acc=0.066. Gap=-0.0076. Budget: fre… |
| exp_1_8 | ✗ REFUTED | ✓ | gap_two_minus_single=-0.015±0.000 | Two-stage acc=0.131 vs single-stage acc=0.147. Gap=-0.0151. … |
| exp_2_1 | ~ INCONCLUSIVE | ✓ | cliff_after_ratio=64.000±0.000 | Largest quality drop: 0.051 between 32x and 64x compression.… |
| exp_2_2 | ✗ REFUTED | ✓ | exact_cosim_attention=0.321±0.000 | Inferential recall: attention=0.990, autoenc=0.998, gain=-0.… |
| exp_2_3 | ~ INCONCLUSIVE | ✓ | avg_error_approx=0.000±0.000 | avg_error_exact=0.0000, avg_error_approx=0.0004, diff (appro… |
| exp_2_4 | ✗ REFUTED | ✓ | cosine_sim.16=0.340±0.000 | Quality scores [0.642, 0.473, 0.34, 0.227, 0.153] for chunk … |
| exp_2_5 | ✗ REFUTED | ✓ | acc_gain_structured_over_flat=-0.313±0.000 | Structured acc=0.056, flat acc=0.369, gain=-0.313. Structure… |
| exp_2_6 | ✓ SUPPORTED | ✓ | comp_a_on_a=0.610±0.000 | Comp_A: in-domain=0.610, cross-domain=0.221, drop=0.390 (thr… |
| exp_2_7 | ~ INCONCLUSIVE | ✓ | cosim_stage1=0.487±0.000 | Cosine similarities after iterative compression: stage1=0.48… |
| exp_2_8 | ~ INCONCLUSIVE | ✓ | quality_a_in_mixed_seq=0.609±0.000 | Domain A (in-distribution): 0.605. Domain B (shifted): 0.290… |
| exp_2_9 | ✗ REFUTED | ✓ | recon_a_cosine=0.413±0.000 | Reconstruction A=0.414 vs B=0.397 (gap 0.016). Retrieval A=1… |
| exp_3_1 | ✓ SUPPORTED | ✓ | acc_delta=0.005±0.000 | Event-driven acc delta: +0.005. Coverage delta: +0.059. Stri… |
| exp_3_2 | ✗ REFUTED | ✓ | A_no_signal.final_write_rate=0.193±0.000 | Regime A collapsed: False. Other regimes collapsed: True. Wr… |
| exp_3_3 | ✗ REFUTED | ✓ | acc_gap=0.000±0.000 | Late vs early accuracy gap: +0.000. Threshold for SUPPORTED:… |
| exp_3_4 | ✗ REFUTED | ✓ | acc_gap=-0.173±0.000 | Boundary vs fixed accuracy gap: -0.173. Boundary per-segment… |
| exp_3_5 | ✓ SUPPORTED | ✓ | acc_at_latency_0=0.216±0.000 | Baseline (L=0): 0.216. Accuracy range across latencies: 0.20… |
| exp_3_6 | ✓ SUPPORTED | ✓ | acc_gap=0.133±0.000 | Two-pass vs forward-only accuracy gap: +0.133. Retroactive w… |
| exp_3_7 | ✗ REFUTED | ✓ | acc_delta=0.027±0.000 | Adaptive vs uniform accuracy delta: +0.027. Allocation Gini … |
| exp_4_1 | ✓ SUPPORTED | ✓ | direct_acc=0.047±0.000 | Learned acc=0.0709 vs Direct acc=0.0469, gap=+0.0241 (thresh… |
| exp_4_2 | ✗ REFUTED | ✓ | gap_multi_minus_single=-0.020±0.000 | Multi acc=0.9545 vs Single acc=0.9744, gap=-0.0199 (threshol… |
| exp_4_3 | ~ INCONCLUSIVE | ✓ | acc_at_k_1=0.016±0.000 | Accuracy is flat across k values (range=0.0026).… |
| exp_4_4 | ✓ SUPPORTED | ✓ | hard_final_acc=0.017±0.000 | Soft var=0.000033 vs Hard var=0.000076; Soft acc=0.0164 vs H… |
| exp_4_5 | ~ INCONCLUSIVE | ✓ | avg_tiers_queried_sequential=3.000±0.000 | Simultaneous acc=0.0163 vs Sequential acc=0.0156, gap=+0.000… |
| exp_4_6 | ~ INCONCLUSIVE | ✓ | recon_exact_acc=0.015±0.000 | Exact(Sim=0.0163, Rec=0.0145), Inferential(Sim=0.0289, Rec=0… |
| exp_4_7 | ✓ SUPPORTED | ✓ | null_f1=0.889±0.000 | Null precision 1.000 vs threshold 0.7.… |
| exp_4_8 | ~ INCONCLUSIVE | ✓ | acc_at_N_0=0.017±0.000 | Accuracy is flat across N values; no interference detected.… |
| exp_4_9 | ✓ SUPPORTED | ✓ | compositional_gap=-0.032±0.000 | Single-hop=0.968, Two-hop=1.000, Gap=-0.032, Random=0.062.… |
| exp_5_1 | ✗ REFUTED | ✓ | A_task_only.final_read_rate=0.156±0.000 | Regime A collapsed: False (stable). Read rates: A_task_only=… |
| exp_5_2 | ✓ SUPPORTED | ✓ | factual_qa_freq1=0.996±0.000 | Optimal frequencies — task1(factual_qa):4, task2(seq_complet… |
| exp_5_3 | ~ INCONCLUSIVE | ✓ | acc_gap=0.006±0.000 | Reactive: acc=0.990, read_rate=0.125. Predictive: acc=0.984,… |
| exp_5_4 | ✗ REFUTED | ✓ | rate_diff=0.000±0.000 | Type A (cheap/recompute): retrieval_rate=0.000, acc=1.000. T… |
| exp_5_5 | ✗ REFUTED | ✓ | cascading_acc=0.396±0.000 | Full-depth: acc=0.961, tiers=3.00. Cascading: acc=0.396, tie… |
| exp_5_6 | ✓ SUPPORTED | ✓ | acc_at_T_50=0.972±0.000 | Baseline acc (no suppression): 0.997. Optimal T=0.8, quality… |
| exp_5_7 | ✗ REFUTED | ✓ | arbitrated_acc=0.482±0.000 | attn_only_acc=0.511, mem_only_acc=0.501, arbitrated_acc=0.48… |
| exp_6_1 | ✓ SUPPORTED | ✓ | learned_acc=0.059±0.000 | Learned vs LRU gap: 0.040. Threshold for SUPPORTED: >0.03. R… |
| exp_6_2 | ~ INCONCLUSIVE | ✓ | avg_compression_level_at_query_time=1.933±0.000 | Compression vs LRU gap: -0.004. Average compression level (0… |
| exp_6_3 | ✓ SUPPORTED | ✓ | gap_selective_minus_lru=0.076±0.000 | Selective vs LRU gap on phase-2 queries: 0.076. Eviction rat… |
| exp_6_4 | ~ INCONCLUSIVE | ✓ | acc_at_K_0=0.018±0.000 | Optimal K=5 with acc=0.063. K=0 acc=0.018, K=5 acc=0.063. In… |
| exp_6_5 | ~ INCONCLUSIVE | ✓ | ebbinghaus_acc=0.024±0.000 | Ebbinghaus vs LRU gap: 0.004. Learned S=11.367 steps. Mean r… |
| exp_6_6 | ~ INCONCLUSIVE | ✓ | acc_a_after_ewc=0.023±0.000 | Standard forgetting: 0.208. EWC forgetting: 0.204. Significa… |
| exp_6_7 | ✗ REFUTED | ✓ | gap_joint_minus_independent=-0.035±0.000 | Joint vs independent gap: -0.035. Write-evict correlation — … |
| exp_6_8 | ~ INCONCLUSIVE | ✓ | consolidation_acc=0.021±0.000 | Consolidation vs no-consolidation gap: 0.003. Consolidation … |
| exp_7_1 | ✓ SUPPORTED | ✓ | Gumbel.accuracy=0.128±0.000 | Gumbel most stable: True. Both beat REINFORCE: True. Acc: ST… |
| exp_7_2 | ✓ SUPPORTED | ✓ | acc_per_complexity.Large=0.108±0.000 | Peak efficiency at: Medium. Efficiency ratios: {'Tiny': 0.0,… |
| exp_7_3 | ~ INCONCLUSIVE | ✓ | factual_acc=0.094±0.000 | Reasoning gap=0.040 (threshold <0.15: True). Generation gap=… |
| exp_7_4 | ~ INCONCLUSIVE | ✓ | acc_per_depth.0=0.122±0.000 | Min depth for meaningful behavior: None. depth=0 meaningful:… |
| exp_7_5 | ✗ REFUTED | ✓ | finetuned_transfer_acc=0.101±0.000 | Transfer gap (fresh - zero_shot): 0.028. Supported threshold… |
| exp_7_6 | ✗ REFUTED | ✓ | adversarial_ratio=1.015±0.000 | Adversarial ratio: 1.015 (threshold >1.5: False). Self-corre… |
| exp_7_7 | ✗ REFUTED | ✓ | early_averages.compression_fidelity=0.520±0.000 | Bottleneck at 20%: read_accuracy. Bottleneck at 50%: read_ac… |
| exp_7_8 | ~ INCONCLUSIVE | ✓ | curriculum_acc=0.075±0.000 | Curriculum acc >= joint acc: False. Curriculum loss_var < jo… |
| exp_7_9 | ✓ SUPPORTED | ✓ | interpretability_score_trained=0.059±0.000 | Interp score: trained=0.059 untrained=0.055. Gate is non-ran… |

---

## Detailed Results by Category

### Category 1 — What To Write
*1 supported / 3 refuted / 4 inconclusive / 0 error*

#### exp_1_1  ✗ REFUTED
**Hypothesis:** Attention weight correlates positively with memory importance and attention-based memory outperforms random memory on retrieval tasks.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 128s

**Metrics (mean ± std across seeds):**

- `attention_correlation` = **-0.5026**  *(stable across seeds)*
- `attention_memory_acc` = **0.0877**  *(stable across seeds)*
- `oracle_memory_acc` = **0.0869**  *(stable across seeds)*
- `random_memory_acc` = **0.0874**  *(stable across seeds)*

**Notes:** Pearson r=-0.5026. Attention acc=0.088 vs random=0.087 vs oracle=0.087.

---
#### exp_1_2  ✗ REFUTED
**Hypothesis:** A memory built from high-surprise (high-perplexity) tokens supports better retrieval than attention-based memory.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 59s

**Metrics (mean ± std across seeds):**

- `attention_acc` = **0.1442**  *(stable across seeds)*
- `baseline_acc` = **0.0306**  *(stable across seeds)*
- `gap_surprise_minus_attention` = **-0.0048**  *(stable across seeds)*
- `surprise_acc` = **0.1394**  *(stable across seeds)*

**Notes:** Surprise acc=0.139, attention acc=0.144, no-memory baseline=0.031. Gap=-0.0048.

---
#### exp_1_3  ~ INCONCLUSIVE
**Hypothesis:** Storing tokens where gradient magnitude is highest produces memories that generalize better than attention-selected memories.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 56s

**Metrics (mean ± std across seeds):**

- `attention_acc` = **0.0900**  *(stable across seeds)*
- `gap_gradient_minus_attention` = **0.0168**  *(stable across seeds)*
- `gradient_acc` = **0.1068**  *(stable across seeds)*
- `random_acc` = **0.1086**  *(stable across seeds)*

**Notes:** Gradient acc=0.107, attention=0.090, random=0.109. Gap=+0.0168.

---
#### exp_1_4  ~ INCONCLUSIVE
**Hypothesis:** Diversity-driven storage (maximally dissimilar entries) outperforms importance-driven storage on recall tasks.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 45s

**Metrics (mean ± std across seeds):**

- `diversity_acc` = **0.1377**  *(stable across seeds)*
- `diversity_mean_pairwise_sim` = **0.3732**  *(stable across seeds)*
- `gap_diversity_minus_importance` = **-0.0067**  *(stable across seeds)*
- `importance_acc` = **0.1444**  *(stable across seeds)*
- `importance_mean_pairwise_sim` = **0.5095**  *(stable across seeds)*

**Notes:** Diversity acc=0.138 vs importance acc=0.144. Gap=-0.0067. Diversity sim=0.3732 vs importance sim=0.5095.

---
#### exp_1_5  ✓ SUPPORTED
**Hypothesis:** A learned write gate outperforms random write, attention-weighted write, and surprise-driven write on associative recall tasks.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 151s

**Metrics (mean ± std across seeds):**

- `acc_attention` = **0.1210**  *(stable across seeds)*
- `acc_learned` = **0.1242**  *(stable across seeds)*
- `acc_random` = **0.1201**  *(stable across seeds)*
- `acc_surprise` = **0.1176**  *(stable across seeds)*
- `loss_attention` = **3.1545**  *(stable across seeds)*
- `loss_learned` = **3.0535**  *(stable across seeds)*
- `loss_random` = **3.1498**  *(stable across seeds)*
- `loss_surprise` = **3.1317**  *(stable across seeds)*
- `ranking` = [['learned', 'attention', 'random', 'surprise'], ['learned', 'attention', 'random', 'surprise'], ['learned', 'attention', 'random', 'surprise']]

**Notes:** Learned gate delta over best baseline: +0.003. Ranking: ['learned', 'attention', 'random', 'surprise'].

---
#### exp_1_6  ~ INCONCLUSIVE
**Hypothesis:** Cosine-similarity deduplication at write time improves retrieval precision without dangerous information loss.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 75s

**Metrics (mean ± std across seeds):**

- `coverage_delta` = **0.0000**  *(stable across seeds)*
- `dedup_coverage` = **1.0000**  *(stable across seeds)*
- `dedup_precision` = **0.1036**  *(stable across seeds)*
- `precision_delta` = **-0.0160**  *(stable across seeds)*
- `standard_coverage` = **1.0000**  *(stable across seeds)*
- `standard_precision` = **0.1197**  *(stable across seeds)*

**Notes:** Dedup: precision=0.104 (std=0.120), coverage=1.000 (std=1.000). Threshold=0.7.

---
#### exp_1_7  ~ INCONCLUSIVE
**Hypothesis:** For a fixed storage budget, infrequent writes with high compression outperform frequent writes with low compression on downstream retrieval.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 55s

**Metrics (mean ± std across seeds):**

- `freq_acc` = **0.0658**  *(stable across seeds)*
- `freq_budget_floats` = **256.0000**  *(stable across seeds)*
- `gap_infreq_minus_freq` = **-0.0076**  *(stable across seeds)*
- `infreq_acc` = **0.0582**  *(stable across seeds)*
- `infreq_budget_floats` = **256.0000**  *(stable across seeds)*

**Notes:** Infreq acc=0.058 vs freq acc=0.066. Gap=-0.0076. Budget: freq=256 floats, infreq=256 floats.

---
#### exp_1_8  ✗ REFUTED
**Hypothesis:** A two-stage write decision (coarse filter then fine ranking) outperforms a single-stage write gate.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 49s

**Metrics (mean ± std across seeds):**

- `gap_two_minus_single` = **-0.0151**  *(stable across seeds)*
- `single_stage_acc` = **0.1466**  *(stable across seeds)*
- `two_stage_acc` = **0.1315**  *(stable across seeds)*

**Notes:** Two-stage acc=0.131 vs single-stage acc=0.147. Gap=-0.0151. Stage1 threshold=0.5.

---

### Category 2 — How To Write (Compression)
*1 supported / 4 refuted / 4 inconclusive / 0 error*

#### exp_2_1  ~ INCONCLUSIVE
**Hypothesis:** There exists a compression ratio threshold beyond which recall fidelity degrades catastrophically rather than gracefully.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 1879s

**Metrics (mean ± std across seeds):**

- `catastrophic_cliff_detected` = [False, False, False]
- `cliff_after_ratio` = **64.0000**  *(stable across seeds)*
- `cliff_at_ratio` = **32.0000**  *(stable across seeds)*
- `cosine_sims.100` = **0.1211**  *(stable across seeds)*
- `cosine_sims.16` = **0.2188**  *(stable across seeds)*
- `cosine_sims.2` = **0.0515**  *(stable across seeds)*
- `cosine_sims.32` = **0.1983**  *(stable across seeds)*
- `cosine_sims.4` = **0.0529**  *(stable across seeds)*
- `cosine_sims.64` = **0.1475**  *(stable across seeds)*
- `cosine_sims.8` = **0.0530**  *(stable across seeds)*
- `max_single_step_drop` = **0.0508**  *(stable across seeds)*
- `mse_values.100` = **0.9937**  *(stable across seeds)*
- `mse_values.16` = **0.9603**  *(stable across seeds)*
- `mse_values.2` = **1.0064**  *(stable across seeds)*
- `mse_values.32` = **0.9689**  *(stable across seeds)*
- `mse_values.4` = **1.0058**  *(stable across seeds)*
- `mse_values.64` = **0.9863**  *(stable across seeds)*
- `mse_values.8` = **1.0058**  *(stable across seeds)*

**Notes:** Largest quality drop: 0.051 between 32x and 64x compression. Cosine sim at 2x=0.052, at 100x=0.121.

---
#### exp_2_2  ✗ REFUTED
**Hypothesis:** Attention-based compression produces more retrievable representations than autoencoder compression, especially on inferential recall tasks.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 19s

**Metrics (mean ± std across seeds):**

- `exact_cosim_attention` = **0.3209**  *(stable across seeds)*
- `exact_cosim_autoenc` = **0.4716**  *(stable across seeds)*
- `fuzzy_acc_attention` = **0.0000**  *(stable across seeds)*
- `fuzzy_acc_autoenc` = **0.2047**  *(stable across seeds)*
- `inferential_acc_attention` = **0.9905**  *(stable across seeds)*
- `inferential_acc_autoenc` = **0.9978**  *(stable across seeds)*
- `inferential_gain_attn_over_autoenc` = **-0.0073**  *(stable across seeds)*

**Notes:** Inferential recall: attention=0.990, autoenc=0.998, gain=-0.007. Attention wins on inferential task by >5pp: False. Autoencoder wins all three tasks: True.

---
#### exp_2_3  ~ INCONCLUSIVE
**Hypothesis:** A controller can learn without supervision which information should be stored exactly (numbers, names) vs approximately (context, themes).

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 3s

**Metrics (mean ± std across seeds):**

- `avg_error_approx` = **0.0004**  *(stable across seeds)*
- `avg_error_exact` = **0.0000**  *(stable across seeds)*
- `error_ratio` = **33.8341**  *(stable across seeds)*
- `high_precision_rate_for_approx` = **0.9852**  *(stable across seeds)*
- `high_precision_rate_for_exact` = **0.9969**  *(stable across seeds)*

**Notes:** avg_error_exact=0.0000, avg_error_approx=0.0004, diff (approx-exact)=+0.0004. HP rate: exact=0.997, approx=0.985. System learned to protect exact tokens: True.

---
#### exp_2_4  ✗ REFUTED
**Hypothesis:** There exists an optimal chunk size for compression beyond which quality degrades independent of compression ratio.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 326s

**Metrics (mean ± std across seeds):**

- `clear_peak_detected` = [False, False, False]
- `cosine_sim.16` = **0.3402**  *(stable across seeds)*
- `cosine_sim.32` = **0.2274**  *(stable across seeds)*
- `cosine_sim.4` = **0.6425**  *(stable across seeds)*
- `cosine_sim.64` = **0.1531**  *(stable across seeds)*
- `cosine_sim.8` = **0.4728**  *(stable across seeds)*
- `is_flat` = [False, False, False]
- `is_monotone_decrease` = [True, True, True]
- `optimal_chunk_size` = **4.0000**  *(stable across seeds)*
- `quality_at_16` = **0.3402**  *(stable across seeds)*
- `quality_at_32` = **0.2274**  *(stable across seeds)*
- `quality_at_4` = **0.6425**  *(stable across seeds)*
- `quality_at_64` = **0.1531**  *(stable across seeds)*
- `quality_at_8` = **0.4728**  *(stable across seeds)*
- `quality_range` = **0.4894**  *(stable across seeds)*

**Notes:** Quality scores [0.642, 0.473, 0.34, 0.227, 0.153] for chunk sizes [4, 8, 16, 32, 64]. Best quality at chunk_size=4 (index 0). Clear peak at middle: False. Monotone decrease: True. Flat: False.

---
#### exp_2_5  ✗ REFUTED
**Hypothesis:** Compressing into a structured key-value representation improves retrieval over compressing into a flat dense vector.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 17s

**Metrics (mean ± std across seeds):**

- `acc_gain_structured_over_flat` = **-0.3127**  *(stable across seeds)*
- `flat_acc` = **0.3686**  *(stable across seeds)*
- `flat_cosim` = **0.4555**  *(stable across seeds)*
- `structured_acc` = **0.0559**  *(stable across seeds)*
- `structured_cosim` = **0.3111**  *(stable across seeds)*

**Notes:** Structured acc=0.056, flat acc=0.369, gain=-0.313. Structured beats flat by >3pp: False. Flat beats structured: True.

---
#### exp_2_6  ✓ SUPPORTED
**Hypothesis:** A compressor trained on domain A produces meaningfully worse retrieval on domain B, indicating compression overfits to domain.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 12s

**Metrics (mean ± std across seeds):**

- `comp_a_on_a` = **0.6104**  *(stable across seeds)*
- `comp_a_on_b` = **0.2206**  *(stable across seeds)*
- `comp_ab_on_a` = **0.5300**  *(stable across seeds)*
- `comp_ab_on_b` = **0.5337**  *(stable across seeds)*
- `comp_b_on_a` = **0.2342**  *(stable across seeds)*
- `comp_b_on_b` = **0.6136**  *(stable across seeds)*
- `drop_a_cross_domain` = **0.3897**  *(stable across seeds)*
- `drop_b_cross_domain` = **0.3794**  *(stable across seeds)*

**Notes:** Comp_A: in-domain=0.610, cross-domain=0.221, drop=0.390 (threshold=0.1). Comp_B: in-domain=0.614, cross-domain=0.234, drop=0.379. Mixed compressor: A=0.530, B=0.534.

---
#### exp_2_7  ~ INCONCLUSIVE
**Hypothesis:** A hierarchy of increasingly abstract memory levels can be built by iterative compression without catastrophic information loss at each stage.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 11s

**Metrics (mean ± std across seeds):**

- `cosim_stage1` = **0.4873**  *(stable across seeds)*
- `cosim_stage2` = **0.2303**  *(stable across seeds)*
- `cosim_stage3` = **0.0911**  *(stable across seeds)*
- `info_retention_pct_per_stage.stage1` = **0.4873**  *(stable across seeds)*
- `info_retention_pct_per_stage.stage2` = **0.4727**  *(stable across seeds)*
- `info_retention_pct_per_stage.stage3` = **0.3957**  *(stable across seeds)*
- `total_compression_stage1` = **4.0000**  *(stable across seeds)*
- `total_compression_stage2` = **16.0000**  *(stable across seeds)*
- `total_compression_stage3` = **64.0000**  *(stable across seeds)*

**Notes:** Cosine similarities after iterative compression: stage1=0.487 (4x), stage2=0.230 (16x), stage3=0.091 (64x). Stage2 above support threshold 0.3: False. Stage2 below refute threshold 0.1: False.

---
#### exp_2_8  ~ INCONCLUSIVE
**Hypothesis:** A compressor trained on a fixed distribution degrades gracefully (not catastrophically) under mid-context distribution shift.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 8s

**Metrics (mean ± std across seeds):**

- `catastrophic` = [False, False, False]
- `quality_a_in_mixed_seq` = **0.6087**  *(stable across seeds)*
- `quality_domain_a` = **0.6050**  *(stable across seeds)*
- `quality_domain_b_shifted` = **0.2901**  *(stable across seeds)*
- `quality_drop` = **0.3149**  *(stable across seeds)*

**Notes:** Domain A (in-distribution): 0.605. Domain B (shifted): 0.290. Drop: 0.315 vs catastrophic threshold 0.2. Graceful degradation (drop < threshold): False. Catastrophic failure: False.

---
#### exp_2_9  ✗ REFUTED
**Hypothesis:** Minimizing reconstruction loss and maximizing downstream retrieval accuracy are fundamentally different objectives and produce measurably different compressed representations.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 246s

**Metrics (mean ± std across seeds):**

- `objectives_diverge` = [False, False, False]
- `recon_a_cosine` = **0.4135**  *(stable across seeds)*
- `recon_b_cosine` = **0.3972**  *(stable across seeds)*
- `recon_gap` = **0.0163**  *(stable across seeds)*
- `retrieval_acc1_a` = **1.0000**  *(stable across seeds)*
- `retrieval_acc1_b` = **1.0000**  *(stable across seeds)*
- `retrieval_gap` = **0.0000**  *(stable across seeds)*

**Notes:** Reconstruction A=0.414 vs B=0.397 (gap 0.016). Retrieval A=1.000 vs B=1.000 (gap 0.000). Diverge=False.

---

### Category 3 — When To Write
*3 supported / 4 refuted / 0 inconclusive / 0 error*

#### exp_3_1  ✓ SUPPORTED
**Hypothesis:** Event-driven writing (learned gate) produces better memory coverage than writing every N tokens for a fixed storage budget.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 38s

**Metrics (mean ± std across seeds):**

- `acc_delta` = **0.0047**  *(stable across seeds)*
- `continuous_acc` = **0.0266**  *(stable across seeds)*
- `continuous_coverage` = **0.2500**  *(stable across seeds)*
- `coverage_delta` = **0.0588**  *(stable across seeds)*
- `event_driven_acc` = **0.0312**  *(stable across seeds)*
- `event_driven_coverage` = **0.3088**  *(stable across seeds)*

**Notes:** Event-driven acc delta: +0.005. Coverage delta: +0.059. Stride used for continuous: 4. Top-k=6 used for event-driven.

---
#### exp_3_2  ✗ REFUTED
**Hypothesis:** A learned write gate trained without explicit anti-collapse objectives will learn to never write (or always write) within N training steps on standard tasks.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 22s

**Metrics (mean ± std across seeds):**

- `A_no_signal.collapsed_always` = [False, False, False]
- `A_no_signal.collapsed_zero` = [False, False, False]
- `A_no_signal.final_write_rate` = **0.1927**  *(stable across seeds)*
- `B_entropy.collapsed_always` = [False, False, False]
- `B_entropy.collapsed_zero` = [False, False, False]
- `B_entropy.final_write_rate` = **0.7448**  *(stable across seeds)*
- `C_reconstruction.collapsed_always` = [False, False, False]
- `C_reconstruction.collapsed_zero` = [False, False, False]
- `C_reconstruction.final_write_rate` = **0.2786**  *(stable across seeds)*
- `D_penalty.collapsed_always` = [True, True, True]
- `D_penalty.collapsed_zero` = [False, False, False]
- `D_penalty.final_write_rate` = **0.9505**  *(stable across seeds)*

**Notes:** Regime A collapsed: False. Other regimes collapsed: True. Write rates: A_no_signal=0.19, B_entropy=0.74, C_reconstruction=0.28, D_penalty=0.95.

---
#### exp_3_3  ✗ REFUTED
**Hypothesis:** Writing later in context (more processed representations) outperforms early writing (raw representations) for inferential tasks.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 88s

**Metrics (mean ± std across seeds):**

- `acc_gap` = **0.0000**  *(stable across seeds)*
- `early_final_loss` = **0.0410**  *(stable across seeds)*
- `early_write_acc` = **1.0000**  *(stable across seeds)*
- `late_final_loss` = **0.0054**  *(stable across seeds)*
- `late_write_acc` = **1.0000**  *(stable across seeds)*

**Notes:** Late vs early accuracy gap: +0.000. Threshold for SUPPORTED: >0.02. Task: inferential rule application (4 rules, 3 examples).

---
#### exp_3_4  ✗ REFUTED
**Hypothesis:** Semantic-boundary-triggered writing outperforms fixed-interval writing on long-document tasks with clear topical structure.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 43s

**Metrics (mean ± std across seeds):**

- `acc_gap` = **-0.1729**  *(stable across seeds)*
- `boundary_acc` = **0.0359**  *(stable across seeds)*
- `boundary_per_segment_acc` = [[0.0391, 0.0406, 0.0281], [0.0391, 0.0406, 0.0281], [0.0391, 0.0406, 0.0281]]
- `fixed_acc` = **0.2089**  *(stable across seeds)*
- `fixed_per_segment_acc` = [[0.0234, 0.0141, 0.5891], [0.0234, 0.0141, 0.5891], [0.0234, 0.0141, 0.5891]]
- `segment_balance_score` = **0.9932**  *(stable across seeds)*

**Notes:** Boundary vs fixed accuracy gap: -0.173. Boundary per-segment: [0.039, 0.041, 0.028]. Fixed per-segment: [0.023, 0.014, 0.589]. Boundary balance score: 0.993.

---
#### exp_3_5  ✓ SUPPORTED
**Hypothesis:** Downstream retrieval quality degrades measurably when write latency exceeds a specific token distance threshold.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 132s

**Metrics (mean ± std across seeds):**

- `acc_at_latency_0` = **0.2156**  *(stable across seeds)*
- `acc_at_latency_16` = **0.0312**  *(stable across seeds)*
- `acc_at_latency_2` = **0.2375**  *(stable across seeds)*
- `acc_at_latency_4` = **0.2016**  *(stable across seeds)*
- `acc_at_latency_8` = **0.0406**  *(stable across seeds)*
- `acc_range` = **0.2062**  *(stable across seeds)*
- `latency_threshold` = **8.0000**  *(stable across seeds)*

**Notes:** Baseline (L=0): 0.216. Accuracy range across latencies: 0.206. First threshold where drop >0.05: 8. Flat (no degradation): False.

---
#### exp_3_6  ✓ SUPPORTED
**Hypothesis:** A controller can learn to retroactively write tokens it initially skipped once later context reveals their importance.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 41s

**Metrics (mean ± std across seeds):**

- `acc_gap` = **0.1328**  *(stable across seeds)*
- `forward_acc` = **0.0391**  *(stable across seeds)*
- `retroactive_write_rate` = **0.0833**  *(stable across seeds)*
- `two_pass_acc` = **0.1719**  *(stable across seeds)*

**Notes:** Two-pass vs forward-only accuracy gap: +0.133. Retroactive write rate (fraction of tokens upgraded): 0.083. Forward pass writes 4 slots; revision adds up to 2 more.

---
#### exp_3_7  ✗ REFUTED
**Hypothesis:** A controller given a fixed write budget learns to allocate it non-uniformly in a way that improves performance over uniform allocation.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 63s

**Metrics (mean ± std across seeds):**

- `acc_delta` = **0.0266**  *(stable across seeds)*
- `adaptive_acc` = **0.2437**  *(stable across seeds)*
- `allocation_gini` = **0.1639**  *(stable across seeds)*
- `allocation_per_block` = [[1.9859, 1.0, 1.0234, 1.9906], [1.9859, 1.0, 1.0234, 1.9906], [1.9859, 1.0, 1.0234, 1.9906]]
- `uniform_acc` = **0.2172**  *(stable across seeds)*
- `uniform_alloc_per_block` = [[1.0, 1.0, 2.0, 2.0], [1.0, 1.0, 2.0, 2.0], [1.0, 1.0, 2.0, 2.0]]

**Notes:** Adaptive vs uniform accuracy delta: +0.027. Allocation Gini coefficient: 0.164 (threshold 0.3). Adaptive per-block: [1.99, 1.0, 1.02, 1.99]. Uniform per-block: [1.0, 1.0, 2.0, 2.0] (fixed: [1, 1, 2, 2]).

---

### Category 4 — What To Read
*4 supported / 1 refuted / 4 inconclusive / 0 error*

#### exp_4_1  ✓ SUPPORTED
**Hypothesis:** A dedicated query formulation module outperforms direct use of the current hidden state as a retrieval query.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 6s

**Metrics (mean ± std across seeds):**

- `direct_acc` = **0.0469**  *(stable across seeds)*
- `direct_loss` = **4.1343**  *(stable across seeds)*
- `gap_learned_minus_direct` = **0.0241**  *(stable across seeds)*
- `learned_acc` = **0.0709**  *(stable across seeds)*
- `learned_loss` = **4.1128**  *(stable across seeds)*

**Notes:** Learned acc=0.0709 vs Direct acc=0.0469, gap=+0.0241 (threshold ±0.02).

---
#### exp_4_2  ✗ REFUTED
**Hypothesis:** Multi-vector retrieval captures more relevant content than single-vector retrieval for queries with multi-faceted information needs.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 54s

**Metrics (mean ± std across seeds):**

- `gap_multi_minus_single` = **-0.0199**  *(stable across seeds)*
- `multi_acc` = **0.9545**  *(stable across seeds)*
- `multi_loss` = **0.5983**  *(stable across seeds)*
- `single_acc` = **0.9744**  *(stable across seeds)*
- `single_loss` = **0.4819**  *(stable across seeds)*

**Notes:** Multi acc=0.9545 vs Single acc=0.9744, gap=-0.0199 (threshold +0.03 for SUPPORTED).

---
#### exp_4_3  ~ INCONCLUSIVE
**Hypothesis:** There exists an optimal retrieval depth (top-k) beyond which additional retrieved entries introduce more noise than signal.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 22s

**Metrics (mean ± std across seeds):**

- `acc_at_k_1` = **0.0161**  *(stable across seeds)*
- `acc_at_k_12` = **0.0175**  *(stable across seeds)*
- `acc_at_k_16` = **0.0158**  *(stable across seeds)*
- `acc_at_k_2` = **0.0149**  *(stable across seeds)*
- `acc_at_k_4` = **0.0150**  *(stable across seeds)*
- `acc_at_k_8` = **0.0171**  *(stable across seeds)*
- `acc_at_max_k` = **0.0158**  *(stable across seeds)*
- `optimal_k` = **12.0000**  *(stable across seeds)*
- `peak_acc` = **0.0175**  *(stable across seeds)*

**Notes:** Accuracy is flat across k values (range=0.0026).

---
#### exp_4_4  ✓ SUPPORTED
**Hypothesis:** Soft retrieval (weighted average) produces more stable training than hard retrieval (discrete selection), though hard retrieval may achieve higher peak task accuracy.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 7s

**Metrics (mean ± std across seeds):**

- `hard_final_acc` = **0.0166**  *(stable across seeds)*
- `hard_is_more_accurate` = [True, True, True]
- `hard_loss_variance` = **0.0001**  *(stable across seeds)*
- `soft_final_acc` = **0.0164**  *(stable across seeds)*
- `soft_is_more_stable` = [True, True, True]
- `soft_loss_variance` = **0.0000**  *(stable across seeds)*

**Notes:** Soft var=0.000033 vs Hard var=0.000076; Soft acc=0.0164 vs Hard acc=0.0166.

---
#### exp_4_5  ~ INCONCLUSIVE
**Hypothesis:** Simultaneous cross-tier retrieval achieves better recall than sequential cascading retrieval.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 16s

**Metrics (mean ± std across seeds):**

- `avg_tiers_queried_sequential` = **3.0000**  *(stable across seeds)*
- `gap_sim_minus_seq` = **0.0006**  *(stable across seeds)*
- `sequential_acc` = **0.0156**  *(stable across seeds)*
- `simultaneous_acc` = **0.0163**  *(stable across seeds)*

**Notes:** Simultaneous acc=0.0163 vs Sequential acc=0.0156, gap=+0.0006 (threshold +0.02 for SUPPORTED). Sequential avg tiers queried=3.00.

---
#### exp_4_6  ~ INCONCLUSIVE
**Hypothesis:** For exact recall tasks similarity-based retrieval wins; for inferential completion tasks reconstruction-based retrieval wins.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 14s

**Metrics (mean ± std across seeds):**

- `rec_wins_inferential` = [False, False, False]
- `recon_exact_acc` = **0.0145**  *(stable across seeds)*
- `recon_inferential_acc` = **0.0340**  *(stable across seeds)*
- `sim_wins_exact` = [False, False, False]
- `similarity_exact_acc` = **0.0163**  *(stable across seeds)*
- `similarity_inferential_acc` = **0.0289**  *(stable across seeds)*

**Notes:** Exact(Sim=0.0163, Rec=0.0145), Inferential(Sim=0.0289, Rec=0.0340). Specialisation A=False, B=False.

---
#### exp_4_7  ✓ SUPPORTED
**Hypothesis:** A learned read gate can be trained to return null on tasks where most queries have no relevant memory content, without explicit null supervision.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 10s

**Metrics (mean ± std across seeds):**

- `null_f1` = **0.8886**  *(stable across seeds)*
- `null_precision` = **1.0000**  *(stable across seeds)*
- `null_recall` = **0.7996**  *(stable across seeds)*
- `p_relevant` = **0.2000**  *(stable across seeds)*
- `retrieval_rate_on_matches` = **0.0000**  *(stable across seeds)*

**Notes:** Null precision 1.000 vs threshold 0.7.

---
#### exp_4_8  ~ INCONCLUSIVE
**Hypothesis:** Retrieval quality degrades non-linearly as the number of near-duplicate entries in memory increases, with a specific saturation point.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 23s

**Metrics (mean ± std across seeds):**

- `acc_at_N_0` = **0.0172**  *(stable across seeds)*
- `acc_at_N_12` = **0.0148**  *(stable across seeds)*
- `acc_at_N_16` = **0.0152**  *(stable across seeds)*
- `acc_at_N_2` = **0.0167**  *(stable across seeds)*
- `acc_at_N_4` = **0.0160**  *(stable across seeds)*
- `acc_at_N_8` = **0.0144**  *(stable across seeds)*
- `degradation_is_nonlinear` = [True, True, True]
- `saturation_point_N` = **4.0000**  *(stable across seeds)*

**Notes:** Accuracy is flat across N values; no interference detected.

---
#### exp_4_9  ✓ SUPPORTED
**Hypothesis:** A learned retrieval mechanism can be trained to retrieve two separate memory entries and compose them to answer questions neither entry alone can answer.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 11s

**Metrics (mean ± std across seeds):**

- `compositional_gap` = **-0.0316**  *(stable across seeds)*
- `random_baseline` = **0.0625**  *(stable across seeds)*
- `single_hop_accuracy` = **0.9684**  *(stable across seeds)*
- `two_hop_accuracy` = **1.0000**  *(stable across seeds)*

**Notes:** Single-hop=0.968, Two-hop=1.000, Gap=-0.032, Random=0.062.

---

### Category 5 — When To Read
*2 supported / 4 refuted / 1 inconclusive / 0 error*

#### exp_5_1  ✗ REFUTED
**Hypothesis:** A learned read gate trained without explicit anti-collapse objectives will learn a degenerate policy (always read or never read) within N training steps.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 10s

**Metrics (mean ± std across seeds):**

- `A_task_only.collapsed` = [False, False, False]
- `A_task_only.final_read_rate` = **0.1562**  *(stable across seeds)*
- `A_task_only.mode` = ['stable', 'stable', 'stable']
- `B_sparsity.collapsed` = [True, True, True]
- `B_sparsity.final_read_rate` = **0.0938**  *(stable across seeds)*
- `B_sparsity.mode` = ['NEVER', 'NEVER', 'NEVER']
- `C_coverage.collapsed` = [False, False, False]
- `C_coverage.final_read_rate` = **0.1562**  *(stable across seeds)*
- `C_coverage.mode` = ['stable', 'stable', 'stable']
- `D_confidence.collapsed` = [False, False, False]
- `D_confidence.final_read_rate` = **0.6562**  *(stable across seeds)*
- `D_confidence.mode` = ['stable', 'stable', 'stable']

**Notes:** Regime A collapsed: False (stable). Read rates: A_task_only=0.16, B_sparsity=0.09, C_coverage=0.16, D_confidence=0.66.

---
#### exp_5_2  ✓ SUPPORTED
**Hypothesis:** Optimal read frequency is task-dependent and cannot be determined by a single fixed schedule across task types.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 464s

**Metrics (mean ± std across seeds):**

- `factual_qa_freq1` = **0.9956**  *(stable across seeds)*
- `factual_qa_freq2` = **0.9969**  *(stable across seeds)*
- `factual_qa_freq4` = **0.9975**  *(stable across seeds)*
- `factual_qa_freq8` = **0.9944**  *(stable across seeds)*
- `frequencies_differ` = [True, True, True]
- `optimal_freq_task1` = **4.0000**  *(stable across seeds)*
- `optimal_freq_task2` = **2.0000**  *(stable across seeds)*
- `optimal_freq_task3` = **4.0000**  *(stable across seeds)*
- `pattern_matching_freq1` = **0.9513**  *(stable across seeds)*
- `pattern_matching_freq2` = **0.9413**  *(stable across seeds)*
- `pattern_matching_freq4` = **0.9806**  *(stable across seeds)*
- `pattern_matching_freq8` = **0.9456**  *(stable across seeds)*
- `seq_completion_freq1` = **0.9900**  *(stable across seeds)*
- `seq_completion_freq2` = **0.9981**  *(stable across seeds)*
- `seq_completion_freq4` = **0.9931**  *(stable across seeds)*
- `seq_completion_freq8` = **0.9950**  *(stable across seeds)*

**Notes:** Optimal frequencies — task1(factual_qa):4, task2(seq_completion):2, task3(pattern_matching):4. Frequencies differ: True. Meaningful accuracy gap between frequencies: True.

---
#### exp_5_3  ~ INCONCLUSIVE
**Hypothesis:** Anticipatory retrieval (predicting need before it arises) improves latency without measurably hurting task quality.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 104s

**Metrics (mean ± std across seeds):**

- `acc_gap` = **0.0062**  *(stable across seeds)*
- `predictive_acc` = **0.9838**  *(stable across seeds)*
- `predictive_read_rate` = **0.4050**  *(stable across seeds)*
- `reactive_acc` = **0.9900**  *(stable across seeds)*
- `reactive_read_rate` = **0.1250**  *(stable across seeds)*

**Notes:** Reactive: acc=0.990, read_rate=0.125. Predictive: acc=0.984, read_rate=0.405. Acc gap: 0.0062 (positive = reactive better). Read rate reduction: -0.2800.

---
#### exp_5_4  ✗ REFUTED
**Hypothesis:** A controller can learn to prefer recomputation for cheap information and retrieval for expensive information.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 25s

**Metrics (mean ± std across seeds):**

- `rate_diff` = **0.0000**  *(stable across seeds)*
- `retrieval_rate_typeA` = **0.0003**  *(stable across seeds)*
- `retrieval_rate_typeB` = **0.0003**  *(stable across seeds)*
- `typeA_acc` = **1.0000**  *(stable across seeds)*
- `typeB_acc` = **0.0163**  *(stable across seeds)*

**Notes:** Type A (cheap/recompute): retrieval_rate=0.000, acc=1.000. Type B (expensive/retrieve): retrieval_rate=0.000, acc=0.016. Rate difference (B-A)=-0.000. Threshold: A<0.3 AND B>0.6 for SUPPORTED.

---
#### exp_5_5  ✗ REFUTED
**Hypothesis:** Confidence-gated cascading retrieval matches full-depth retrieval quality at significantly lower average compute cost.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 46s

**Metrics (mean ± std across seeds):**

- `cascading_acc` = **0.3962**  *(stable across seeds)*
- `cascading_tiers_avg` = **1.1690**  *(stable across seeds)*
- `compute_savings_pct` = **61.0200**  *(stable across seeds)*
- `full_acc` = **0.9613**  *(stable across seeds)*
- `full_tiers_avg` = **3.0000**  *(stable across seeds)*

**Notes:** Full-depth: acc=0.961, tiers=3.00. Cascading: acc=0.396, tiers=1.17. Acc drop=0.5650. Compute savings=61.0%. Threshold: acc_drop<=0.02 AND casc_tiers<2.0 for SUPPORTED.

---
#### exp_5_6  ✓ SUPPORTED
**Hypothesis:** Suppressing memory reads when next-token prediction confidence exceeds a threshold costs less than 1% task quality.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 62s

**Metrics (mean ± std across seeds):**

- `acc_at_T_50` = **0.9719**  *(stable across seeds)*
- `acc_at_T_60` = **0.9862**  *(stable across seeds)*
- `acc_at_T_70` = **0.9894**  *(stable across seeds)*
- `acc_at_T_80` = **0.9919**  *(stable across seeds)*
- `acc_at_T_90` = **0.9938**  *(stable across seeds)*
- `baseline_acc` = **0.9969**  *(stable across seeds)*
- `optimal_T` = **0.8000**  *(stable across seeds)*
- `quality_cost_at_optimal_T` = **0.0050**  *(stable across seeds)*
- `suppression_rate_at_T_50` = **0.5633**  *(stable across seeds)*
- `suppression_rate_at_T_60` = **0.4933**  *(stable across seeds)*
- `suppression_rate_at_T_70` = **0.3950**  *(stable across seeds)*
- `suppression_rate_at_T_80` = **0.3092**  *(stable across seeds)*
- `suppression_rate_at_T_90` = **0.0983**  *(stable across seeds)*

**Notes:** Baseline acc (no suppression): 0.997. Optimal T=0.8, quality_cost=0.0050, suppression_rate=0.309. SUPPORTED criterion: quality_cost<0.01 AND sup_rate>0.3 met at any T: True.

---
#### exp_5_7  ✗ REFUTED
**Hypothesis:** When local attention and external memory produce conflicting predictions, a learned arbitration policy outperforms both fixed-priority policies.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 49s

**Metrics (mean ± std across seeds):**

- `arbitrated_acc` = **0.4825**  *(stable across seeds)*
- `arbitration_advantage` = **-0.0281**  *(stable across seeds)*
- `attn_only_acc` = **0.5106**  *(stable across seeds)*
- `best_fixed_policy_acc` = **0.5106**  *(stable across seeds)*
- `mem_only_acc` = **0.5006**  *(stable across seeds)*

**Notes:** attn_only_acc=0.511, mem_only_acc=0.501, arbitrated_acc=0.482. Arbitration advantage over best fixed: -0.0281. SUPPORTED requires advantage > 0.05 over BOTH fixed policies.

---

### Category 6 — How To Forget
*2 supported / 1 refuted / 5 inconclusive / 0 error*

#### exp_6_1  ✓ SUPPORTED
**Hypothesis:** A learned importance-scored eviction policy significantly outperforms LRU on tasks requiring retention of low-frequency but high-importance information.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 163s

**Metrics (mean ± std across seeds):**

- `eviction_policy_ranking` = ['lfu(0.059) > learned(0.059) > random(0.022) > lru(0.020)', 'lfu(0.059) > learned(0.059) > random(0.022) > lru(0.020)', 'lfu(0.059) > learned(0.059) > random(0.022) > lru(0.020)']
- `learned_acc` = **0.0591**  *(stable across seeds)*
- `learned_vs_lru_gap` = **0.0395**  *(stable across seeds)*
- `lfu_acc` = **0.0594**  *(stable across seeds)*
- `lru_acc` = **0.0195**  *(stable across seeds)*
- `random_acc` = **0.0220**  *(stable across seeds)*

**Notes:** Learned vs LRU gap: 0.040. Threshold for SUPPORTED: >0.03. Ranking: lfu(0.059) > learned(0.059) > random(0.022) > lru(0.020).

---
#### exp_6_2  ~ INCONCLUSIVE
**Hypothesis:** Graceful degradation via iterative compression outperforms hard eviction for long-context tasks where storage budget is the binding constraint.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 150s

**Metrics (mean ± std across seeds):**

- `avg_compression_level_at_query_time` = **1.9333**  *(stable across seeds)*
- `compression_acc` = **0.0289**  *(stable across seeds)*
- `gap_compression_minus_lru` = **-0.0044**  *(stable across seeds)*
- `lru_acc` = **0.0333**  *(stable across seeds)*

**Notes:** Compression vs LRU gap: -0.004. Average compression level (0=full, 1=half, 2=quarter): 1.933.

---
#### exp_6_3  ✓ SUPPORTED
**Hypothesis:** A controller can learn to evict domain-mismatched memories when input distribution shifts, without explicit domain labels.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 403s

**Metrics (mean ± std across seeds):**

- `gap_selective_minus_lru` = **0.0764**  *(stable across seeds)*
- `lru_phase2_acc` = **0.0364**  *(stable across seeds)*
- `selective_eviction_rate_for_phase1_entries` = **0.7012**  *(stable across seeds)*
- `selective_phase2_acc` = **0.1128**  *(stable across seeds)*

**Notes:** Selective vs LRU gap on phase-2 queries: 0.076. Eviction rate for phase-1 entries: 0.701.

---
#### exp_6_4  ~ INCONCLUSIVE
**Hypothesis:** A controller can learn which memories deserve protection (never evict) without explicit supervision, and performance degrades predictably outside an optimal protected-set size.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 569s

**Metrics (mean ± std across seeds):**

- `acc_at_K_0` = **0.0183**  *(stable across seeds)*
- `acc_at_K_1` = **0.0208**  *(stable across seeds)*
- `acc_at_K_2` = **0.0338**  *(stable across seeds)*
- `acc_at_K_3` = **0.0427**  *(stable across seeds)*
- `acc_at_K_4` = **0.0530**  *(stable across seeds)*
- `acc_at_K_5` = **0.0630**  *(stable across seeds)*
- `optimal_K` = **5.0000**  *(stable across seeds)*
- `optimal_K_acc` = **0.0630**  *(stable across seeds)*
- `protection_recall_at_K_0` = **0.0000**  *(stable across seeds)*
- `protection_recall_at_K_1` = **0.1009**  *(stable across seeds)*
- `protection_recall_at_K_2` = **0.0931**  *(stable across seeds)*
- `protection_recall_at_K_3` = **0.0413**  *(stable across seeds)*
- `protection_recall_at_K_4` = **0.1758**  *(stable across seeds)*
- `protection_recall_at_K_5` = **0.3758**  *(stable across seeds)*
- `protection_recall_at_optimal_K` = **0.3758**  *(stable across seeds)*

**Notes:** Optimal K=5 with acc=0.063. K=0 acc=0.018, K=5 acc=0.063. Interior peak detected: False.

---
#### exp_6_5  ~ INCONCLUSIVE
**Hypothesis:** A biologically-inspired memory decay function (Ebbinghaus-style) improves long-horizon task performance compared to instant eviction.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 521s

**Metrics (mean ± std across seeds):**

- `ebbinghaus_acc` = **0.0241**  *(stable across seeds)*
- `gap_ebbinghaus_minus_lru` = **0.0036**  *(stable across seeds)*
- `learned_stability_S` = **11.3669**  *(stable across seeds)*
- `lru_acc` = **0.0205**  *(stable across seeds)*
- `mean_retention_at_query_time` = **0.7500**  *(stable across seeds)*

**Notes:** Ebbinghaus vs LRU gap: 0.004. Learned S=11.367 steps. Mean retention at query: 0.750.

---
#### exp_6_6  ~ INCONCLUSIVE
**Hypothesis:** The memory controller suffers measurable catastrophic forgetting of its learned policies when fine-tuned on a new domain, absent explicit anti-forgetting mechanisms.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 21s

**Metrics (mean ± std across seeds):**

- `acc_a_after_ewc` = **0.0228**  *(stable across seeds)*
- `acc_a_after_std` = **0.0191**  *(stable across seeds)*
- `acc_a_before` = **0.2267**  *(stable across seeds)*
- `acc_b_ewc` = **0.1250**  *(stable across seeds)*
- `acc_b_std` = **0.1061**  *(stable across seeds)*
- `forgetting_ewc` = **0.2039**  *(stable across seeds)*
- `forgetting_reduction_pct` = **1.8059**  *(stable across seeds)*
- `forgetting_std` = **0.2077**  *(stable across seeds)*

**Notes:** Standard forgetting: 0.208. EWC forgetting: 0.204. Significant: True.

---
#### exp_6_7  ✗ REFUTED
**Hypothesis:** Joint optimization of write and evict decisions outperforms treating them as independent operations when storage pressure is constant.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 811s

**Metrics (mean ± std across seeds):**

- `gap_joint_minus_independent` = **-0.0348**  *(stable across seeds)*
- `independent_acc` = **0.0642**  *(stable across seeds)*
- `joint_acc` = **0.0294**  *(stable across seeds)*
- `write_evict_correlation_independent` = **-0.1911**  *(stable across seeds)*
- `write_evict_correlation_joint` = **0.9904**  *(stable across seeds)*

**Notes:** Joint vs independent gap: -0.035. Write-evict correlation — independent: -0.191, joint: 0.990.

---
#### exp_6_8  ~ INCONCLUSIVE
**Hypothesis:** Periodic offline consolidation (merging memory entries into higher-level representations) improves long-horizon recall without active context.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 326s

**Metrics (mean ± std across seeds):**

- `consolidation_acc` = **0.0213**  *(stable across seeds)*
- `consolidation_compression_ratio` = **2.0000**  *(stable across seeds)*
- `gap_consolidation_minus_none` = **0.0033**  *(stable across seeds)*
- `mean_entries_at_query_time_consolidation` = **8.0000**  *(stable across seeds)*
- `mean_entries_at_query_time_no_consolidation` = **8.0000**  *(stable across seeds)*
- `no_consolidation_acc` = **0.0180**  *(stable across seeds)*

**Notes:** Consolidation vs no-consolidation gap: 0.003. Consolidation compresses 8→4 entries (2.00x ratio). Mean entries at query: no-cons=8.0, cons=8.0.

---

### Category 7 — Cross-Cutting
*3 supported / 3 refuted / 3 inconclusive / 0 error*

#### exp_7_1  ✓ SUPPORTED
**Hypothesis:** Gumbel-softmax relaxation produces more stable training than straight-through estimators, and both outperform REINFORCE for discrete memory selection.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 34s

**Metrics (mean ± std across seeds):**

- `Gumbel.accuracy` = **0.1284**  *(stable across seeds)*
- `Gumbel.loss_variance` = **0.0071**  *(stable across seeds)*
- `Gumbel.mean_grad_norm` = **1.0596**  *(stable across seeds)*
- `REINFORCE.accuracy` = **0.1025**  *(stable across seeds)*
- `REINFORCE.loss_variance` = **2.6959**  *(stable across seeds)*
- `REINFORCE.mean_grad_norm` = **0.0000**  *(stable across seeds)*
- `STE.accuracy` = **0.1255**  *(stable across seeds)*
- `STE.loss_variance` = **0.0094**  *(stable across seeds)*
- `STE.mean_grad_norm` = **1.4468**  *(stable across seeds)*
- `ranking_by_accuracy` = [['Gumbel', 'STE', 'REINFORCE'], ['Gumbel', 'STE', 'REINFORCE'], ['Gumbel', 'STE', 'REINFORCE']]
- `ranking_by_stability` = [['Gumbel', 'STE', 'REINFORCE'], ['Gumbel', 'STE', 'REINFORCE'], ['Gumbel', 'STE', 'REINFORCE']]

**Notes:** Gumbel most stable: True. Both beat REINFORCE: True. Acc: STE=0.126, Gumbel=0.128, REINFORCE=0.102

---
#### exp_7_2  ✓ SUPPORTED
**Hypothesis:** There exists a maximum controller complexity (measured in parameter count) beyond which the controller's overhead exceeds its efficiency contribution.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 59s

**Metrics (mean ± std across seeds):**

- `acc_per_complexity.Large` = **0.1083**  *(stable across seeds)*
- `acc_per_complexity.Medium` = **0.1227**  *(stable across seeds)*
- `acc_per_complexity.Small` = **0.0939**  *(stable across seeds)*
- `acc_per_complexity.Tiny` = **0.1220**  *(stable across seeds)*
- `acc_per_complexity.XL` = **0.1044**  *(stable across seeds)*
- `declines_at_large_xl` = [True, True, True]
- `efficiency_ratio_per_complexity.Large` = **-0.0018**  *(stable across seeds)*
- `efficiency_ratio_per_complexity.Medium` = **0.0001**  *(stable across seeds)*
- `efficiency_ratio_per_complexity.Small` = **-0.0067**  *(stable across seeds)*
- `efficiency_ratio_per_complexity.Tiny` = **0.0000**  *(stable across seeds)*
- `efficiency_ratio_per_complexity.XL` = **-0.0020**  *(stable across seeds)*
- `flops_per_complexity.Large` = **164096.0000**  *(stable across seeds)*
- `flops_per_complexity.Medium` = **16640.0000**  *(stable across seeds)*
- `flops_per_complexity.Small` = **4160.0000**  *(stable across seeds)*
- `flops_per_complexity.Tiny` = **64.0000**  *(stable across seeds)*
- `flops_per_complexity.XL` = **426240.0000**  *(stable across seeds)*
- `largest_is_best` = [False, False, False]
- `optimal_complexity_level` = ['Medium', 'Medium', 'Medium']
- `params_per_complexity.Large` = **164865.0000**  *(stable across seeds)*
- `params_per_complexity.Medium` = **16897.0000**  *(stable across seeds)*
- `params_per_complexity.Small` = **4225.0000**  *(stable across seeds)*
- `params_per_complexity.Tiny` = **65.0000**  *(stable across seeds)*
- `params_per_complexity.XL` = **427521.0000**  *(stable across seeds)*
- `peaks_early` = [True, True, True]

**Notes:** Peak efficiency at: Medium. Efficiency ratios: {'Tiny': 0.0, 'Small': -0.006737515755360065, 'Medium': 0.00011239988114518264, 'Large': -0.0017541632748579459, 'XL': -0.0020083612849882103}. Acc: {'Tiny': 0.12203125, 'Small': 0.09390625, 'Medium': 0.12265625, 'Large': 0.10828125, 'XL': 0.104375}.

---
#### exp_7_3  ~ INCONCLUSIVE
**Hypothesis:** A controller trained on factual QA generalizes its memory management policies to reasoning tasks but not to generation tasks.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 9s

**Metrics (mean ± std across seeds):**

- `factual_acc` = **0.0939**  *(stable across seeds)*
- `factual_to_generation_gap` = **0.0923**  *(stable across seeds)*
- `factual_to_reasoning_gap` = **0.0395**  *(stable across seeds)*
- `generation_acc` = **0.0016**  *(stable across seeds)*
- `generation_fails` = [False, False, False]
- `reasoning_acc` = **0.0544**  *(stable across seeds)*
- `reasoning_transfers` = [True, True, True]

**Notes:** Reasoning gap=0.040 (threshold <0.15: True). Generation gap=0.092 (threshold >0.20: False).

---
#### exp_7_4  ~ INCONCLUSIVE
**Hypothesis:** Meaningful memory management behavior requires at minimum two layers of non-linearity in the controller network.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 33s

**Metrics (mean ± std across seeds):**

- `acc_per_depth.0` = **0.1220**  *(stable across seeds)*
- `acc_per_depth.1` = **0.1158**  *(stable across seeds)*
- `acc_per_depth.2` = **0.1212**  *(stable across seeds)*
- `acc_per_depth.3` = **0.1119**  *(stable across seeds)*
- `meaningful_threshold_per_depth.0` = [False, False, False]
- `meaningful_threshold_per_depth.1` = [False, False, False]
- `meaningful_threshold_per_depth.2` = [False, False, False]
- `meaningful_threshold_per_depth.3` = [False, False, False]
- `write_rate_per_depth.0` = **0.5188**  *(stable across seeds)*
- `write_rate_per_depth.1` = **0.3554**  *(stable across seeds)*
- `write_rate_per_depth.2` = **0.4763**  *(stable across seeds)*
- `write_rate_per_depth.3` = **0.5391**  *(stable across seeds)*
- `write_std_per_depth.0` = **0.0014**  *(stable across seeds)*
- `write_std_per_depth.1` = **0.0005**  *(stable across seeds)*
- `write_std_per_depth.2` = **0.0002**  *(stable across seeds)*
- `write_std_per_depth.3` = **0.0000**  *(stable across seeds)*

**Notes:** Min depth for meaningful behavior: None. depth=0 meaningful: False. depth=1 meaningful: False. depth=2 meaningful: False.

---
#### exp_7_5  ✗ REFUTED
**Hypothesis:** A controller's learned policy trained at small scale does not transfer directly to a larger model without additional fine-tuning.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 35s

**Metrics (mean ± std across seeds):**

- `finetuned_transfer_acc` = **0.1013**  *(stable across seeds)*
- `fresh_large_acc` = **0.1211**  *(stable across seeds)*
- `small_acc` = **0.0916**  *(stable across seeds)*
- `transfer_gap` = **0.0281**  *(stable across seeds)*
- `zero_shot_transfer_acc` = **0.0930**  *(stable across seeds)*

**Notes:** Transfer gap (fresh - zero_shot): 0.028. Supported threshold: gap > 0.10. Refuted threshold: gap <= 0.05.

---
#### exp_7_6  ✗ REFUTED
**Hypothesis:** The memory controller is measurably vulnerable to inputs designed to maximize write activity, and this vulnerability does not self-correct during training.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 11s

**Metrics (mean ± std across seeds):**

- `adversarial_ratio` = **1.0146**  *(stable across seeds)*
- `adversarial_write_rate` = **0.5819**  *(stable across seeds)*
- `normal_write_rate` = **0.5735**  *(stable across seeds)*
- `self_corrects` = [False, False, False]
- `write_rate_at_1000` = **0.5887**  *(stable across seeds)*
- `write_rate_at_1500` = **0.5819**  *(stable across seeds)*
- `write_rate_at_500` = **0.5882**  *(stable across seeds)*

**Notes:** Adversarial ratio: 1.015 (threshold >1.5: False). Self-corrects: False. Checkpoint rates: [0.5881532311439515, 0.5887086486816406, 0.5818839371204376].

---
#### exp_7_7  ✗ REFUTED
**Hypothesis:** Write quality (not read quality, compression ratio, or eviction policy) is the first performance bottleneck encountered during controller training.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 12s

**Metrics (mean ± std across seeds):**

- `bottleneck_at_20pct_training` = ['read_accuracy', 'read_accuracy', 'read_accuracy']
- `bottleneck_at_50pct_training` = ['read_accuracy', 'read_accuracy', 'read_accuracy']
- `bottleneck_order` = [['read_accuracy', 'eviction_correctness', 'write_quality', 'compression_fidelity'], ['read_accuracy', 'eviction_correctness', 'write_quality', 'compression_fidelity'], ['read_accuracy', 'eviction_correctness', 'write_quality', 'compression_fidelity']]
- `early_averages.compression_fidelity` = **0.5196**  *(stable across seeds)*
- `early_averages.eviction_correctness` = **0.2120**  *(stable across seeds)*
- `early_averages.read_accuracy` = **0.0309**  *(stable across seeds)*
- `early_averages.write_quality` = **0.5196**  *(stable across seeds)*
- `final_compression_fidelity` = **0.8068**  *(stable across seeds)*
- `final_eviction_correctness` = **0.3703**  *(stable across seeds)*
- `final_read_accuracy` = **0.0594**  *(stable across seeds)*
- `final_write_quality` = **0.8068**  *(stable across seeds)*

**Notes:** Bottleneck at 20%: read_accuracy. Bottleneck at 50%: read_accuracy. Order (lowest first): ['read_accuracy', 'eviction_correctness', 'write_quality', 'compression_fidelity'].

---
#### exp_7_8  ~ INCONCLUSIVE
**Hypothesis:** Curriculum training (one controller component at a time) produces more stable controller behavior than joint training from the start.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 20s

**Metrics (mean ± std across seeds):**

- `curriculum_acc` = **0.0750**  *(stable across seeds)*
- `curriculum_gate_collapse` = [False, False, False]
- `curriculum_loss_variance` = **0.0014**  *(stable across seeds)*
- `curriculum_read_collapsed` = [False, False, False]
- `curriculum_write_collapsed` = [False, False, False]
- `joint_acc` = **0.1095**  *(stable across seeds)*
- `joint_gate_collapse` = [False, False, False]
- `joint_loss_variance` = **0.0058**  *(stable across seeds)*
- `joint_read_collapsed` = [False, False, False]
- `joint_write_collapsed` = [False, False, False]

**Notes:** Curriculum acc >= joint acc: False. Curriculum loss_var < joint loss_var: True. Joint acc=0.110, Curriculum acc=0.075. Joint var=0.00581, Curriculum var=0.00137.

---
#### exp_7_9  ✓ SUPPORTED
**Hypothesis:** The controller's write and read decisions are interpretable (non-random, correlating with human-meaningful features) in their simplest form before any task-specific training.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 13s

**Metrics (mean ± std across seeds):**

- `heatmaps_saved_to` = ['/Users/wscholl/drex/research/results/interpretability', '/Users/wscholl/drex/research/results/interpretability', '/Users/wscholl/drex/research/results/interpretability']
- `interpretability_score_trained` = **0.0595**  *(stable across seeds)*
- `interpretability_score_untrained` = **0.0549**  *(stable across seeds)*
- `trained.corr_numeric_vs_gate` = **0.0757**  *(stable across seeds)*
- `trained.corr_position_vs_gate` = **-0.0039**  *(stable across seeds)*
- `trained.corr_punct_vs_gate` = **-0.0474**  *(stable across seeds)*
- `trained.corr_rare_vs_gate` = **0.0553**  *(stable across seeds)*
- `trained.gate_mean` = **0.5085**  *(stable across seeds)*
- `trained.gate_nonrandom` = [True, True, True]
- `trained.gate_std` = **0.1505**  *(stable across seeds)*
- `untrained.corr_numeric_vs_gate` = **0.0622**  *(stable across seeds)*
- `untrained.corr_position_vs_gate` = **-0.0077**  *(stable across seeds)*
- `untrained.corr_punct_vs_gate` = **-0.0429**  *(stable across seeds)*
- `untrained.corr_rare_vs_gate` = **0.0596**  *(stable across seeds)*
- `untrained.gate_mean` = **0.5099**  *(stable across seeds)*
- `untrained.gate_nonrandom` = [True, True, True]
- `untrained.gate_std` = **0.1457**  *(stable across seeds)*

**Notes:** Interp score: trained=0.059 untrained=0.055. Gate is non-random. Heatmaps saved to results/interpretability/.

---

## Cross-Cutting Observations

**All SUPPORTED experiments:** exp_1_5, exp_2_6, exp_3_1, exp_3_5, exp_3_6, exp_4_1, exp_4_4, exp_4_7, exp_4_9, exp_5_2, exp_5_6, exp_6_1, exp_6_3, exp_7_1, exp_7_2, exp_7_9

**All REFUTED experiments:** exp_1_1, exp_1_2, exp_1_8, exp_2_2, exp_2_4, exp_2_5, exp_2_9, exp_3_2, exp_3_3, exp_3_4, exp_3_7, exp_4_2, exp_5_1, exp_5_4, exp_5_5, exp_5_7, exp_6_7, exp_7_5, exp_7_6, exp_7_7

**Inconsistent across seeds (need more investigation):** none — all experiments were seed-stable

---
*Report generated by research/aggregate.py*