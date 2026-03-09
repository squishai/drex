# drex Research — Master Results Report

**Generated:** 2026-03-09 01:21 UTC
**Experiments:** 105  |  **Seeds per experiment:** 42, 123, 777
**Total runs evaluated:** 488

## Overall Scoreboard

| Outcome | Count | % |
|---------|-------|---|
| ✓ SUPPORTED    | 24    | 23% |
| ~ INCONCLUSIVE | 35 | 33% |
| ✗ REFUTED      | 46      | 44% |
| ! ERROR        | 0        | 0% |

**Seed consistency:** 92/105 experiments gave the same verdict across all seeds. 13 inconsistent.

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
| exp_8_1 | ✗ REFUTED | ⚠ ['INCONCLUSIVE', 'REFUTED', 'REFUTED'] | pearson_r_entropic=-0.077±0.446 | Softmax r=0.2286, raw dot-product r=-0.0775, entropy-normali… |
| exp_8_2 | ~ INCONCLUSIVE | ⚠ ['INCONCLUSIVE', 'SUPPORTED', 'INCONCLUSIVE'] | pearson_r_difficulty_vs_rate=0.398±0.405 | KV levels [1, 2, 4, 6]: write rates [0.2829, 0.6731, 0.5961,… |
| exp_8_3 | ✗ REFUTED | ✓ | acc_A=0.212±0.051 | Condition A (joint): corr=0.1756, acc=0.272. Condition B (or… |
| exp_8_4 | ~ INCONCLUSIVE | ⚠ ['INCONCLUSIVE', 'INCONCLUSIVE', 'REFUTED'] | acc_A=0.150±0.110 | A (full): acc=0.216, pos_r=-0.1920. B (pos-blind): acc=0.036… |
| exp_9_1 | ✗ REFUTED | ✓ | acc_at1_autoencoder=1.000±0.000 | AE Acc@1=1.0000, CL Acc@1=1.0000, gap=0.0000. Recon cosim: A… |
| exp_9_2 | ✓ SUPPORTED | ✓ | null_f1_A=0.812±0.011 | Cond A (learned, p=0.5): null_f1=0.800, retr_recall=0.825, d… |
| exp_9_3 | ~ INCONCLUSIVE | ✓ | acc_a_after_ewc=0.087±0.066 | Precondition failed: acc_A_before=0.031 < 0.70.… |
| exp_9_4 | ✓ SUPPORTED | ⚠ ['INCONCLUSIVE', 'SUPPORTED', 'SUPPORTED'] | acc_range=0.149±0.003 | K values [0, 2, 4, 6, 8, 10], accs [0.152, 0.144, 0.022, 0.0… |
| exp_9_5 | ✓ SUPPORTED | ⚠ ['SUPPORTED', 'SUPPORTED', 'REFUTED'] | acc_delta=0.018±0.139 | Uniform acc=0.159, adaptive acc=0.227, delta=0.068. Block-1 … |
| exp_10_1 | ~ INCONCLUSIVE | ⚠ ['INCONCLUSIVE', 'REFUTED', 'INCONCLUSIVE', 'REFUTED', 'INCONCLUSIVE'] | acc_w0=0.259±0.007 | gap@24=-0.059, gap@4=-0.100 — pattern inconclusive.… |
| exp_10_2 | ✗ REFUTED | ✓ | acc_A=0.230±0.000 | Overwrite gain (0.002) exceeds new-write gain (-0.031) by >0… |
| exp_10_3 | ~ INCONCLUSIVE | ✓ | acc_fwd_24=0.188±0.000 | Pearson r=0.773 — scaling relationship inconclusive.… |
| exp_11_1 | ✗ REFUTED | ✓ | normal_acc_A=0.211±0.000 | oracle_read_acc_B=0.175 not > oracle_read_acc_A=0.292 by 0.0… |
| exp_11_2 | ✗ REFUTED | ✓ | f1_A=0.343±0.000 | F1_B=0.309 not > F1_A=0.343 by 0.01.… |
| exp_11_3 | ✓ SUPPORTED | ✓ | acc_copy_T0.2=0.061±0.000 | Max pairwise T difference=0.750>0.15. Optimal T: factual=0.2… |
| exp_12_1 | ✗ REFUTED | ✓ | acc1_A=1.000±0.000 | Both models >80% Acc@1 or gap <5%.… |
| exp_12_2 | ~ INCONCLUSIVE | ✓ | a_fails_at_2x_4x=0.000±0.000 | A does not fail at 2x-4x; hypothesis conditions not triggere… |
| exp_13_1 | ✓ SUPPORTED | ✓ | interference_gap=0.120±0.000 | Two-hop acc within 5% of single-hop on interference subset (… |
| exp_13_2 | ✓ SUPPORTED | ✓ | acc_single=0.042±0.018 | Three-hop retains 4.00 of two-hop accuracy (>0.50).… |
| exp_14_1 | ✗ REFUTED | ✓ | acc_A=0.211±0.000 | gap_D=-0.138 < max(gap_B,gap_C)+0.005=-0.001.… |
| exp_14_2 | ✗ REFUTED | ✓ | acc_A=0.209±0.000 | Joint training A >= curriculum B - 0.01: acc_A=0.209, acc_B=… |
| exp_14_3 | ✗ REFUTED | ✓ | acc_A=0.191±0.000 | Constant A >= cosine C - 0.01: acc_A=0.191, acc_C=0.153.… |
| exp_15_1 | ✗ REFUTED | ⚠ ['REFUTED', 'REFUTED', 'INCONCLUSIVE'] | acc_delta=0.106±0.017 | Slot memory acc=0.105 >= delta acc=0.109 - 0.02.… |
| exp_15_2 | ✗ REFUTED | ⚠ ['REFUTED', 'REFUTED', 'INCONCLUSIVE'] | delta_acc_clean=0.110±0.009 | Hebbian acc_int=0.096 within 0.03 of delta acc_int=0.106.… |
| exp_15_3 | ✓ SUPPORTED | ✓ | acc_A_continuous=0.131±0.029 | Energy gate: acc_ratio=0.919>0.90, write_rate=0.519<0.70. Hy… |
| exp_15_4 | ~ INCONCLUSIVE | ⚠ ['REFUTED', 'INCONCLUSIVE', 'INCONCLUSIVE'] | delta_acc_normal=0.166±0.016 | Larimar acc_update=0.177 >= delta acc_update=0.169 - 0.02.… |
| exp_16_1 | ✗ REFUTED | ✓ | acc_gap=-0.012±0.011 | Slot acc=0.045 >= parametric acc=0.045 - 0.02.… |
| exp_16_2 | ~ INCONCLUSIVE | ✓ | baseline_acc=0.042±0.004 | Smooth degradation curve; no clear pareto knee found.… |
| exp_16_3 | ✓ SUPPORTED | ✓ | acc_parametric.24=0.035±0.011 | Parametric retention=1.000 vs slot retention=0.560; diff=0.4… |
| exp_17_1 | ✗ REFUTED | ⚠ ['REFUTED', 'SUPPORTED', 'REFUTED'] | acc_A=0.142±0.082 | Context-only gate matches or beats query-conditioned (gap=-0… |
| exp_17_2 | ✗ REFUTED | ⚠ ['REFUTED', 'SUPPORTED', 'REFUTED'] | acc_K0=0.130±0.068 | All lookahead K within 0.02 of K=0 (best gap=-0.014)… |
| exp_17_3 | ~ INCONCLUSIVE | ✓ | acc_A=0.147±0.073 | gap_B=-0.133 or gap_C=-0.127 too small for valid comparison… |
| exp_17_4 | ✗ REFUTED | ⚠ ['REFUTED', 'INCONCLUSIVE', 'REFUTED'] | acc_A_p000=0.148±0.076 | Gap is effectively constant (variance=0.000077)… |
| exp_18_1 | ✗ REFUTED | ✓ | acc_A=0.045±0.012 | Flat memory matches tiered (gap=-0.003)… |
| exp_18_2 | ~ INCONCLUSIVE | ✓ | corr_access=0.212±0.024 | corr_access=0.237 > 0.15 but corr_recency=0.175 > -0.10… |
| exp_18_3 | ~ INCONCLUSIVE | ⚠ ['INCONCLUSIVE', 'SUPPORTED', 'INCONCLUSIVE'] | acc_flat_128=0.047±0.024 | Crossover at 8 outside expected range 16-64… |
| exp_18_4 | ~ INCONCLUSIVE | ✓ | acc_A=0.032±0.001 | Simultaneous best but gap=-0.066 < 0.03… |
| exp_19_1 | ✗ REFUTED | ✓ | acc_A=0.023±0.000 | Soft attention matches or beats sparse on interference tasks… |
| exp_19_2 | ✗ REFUTED | ✓ | acc_A=0.129±0.000 | Energy writes too frequent (0.8214) or acc dropped vs learne… |
| exp_19_3 | ✗ REFUTED | ✓ | acc_soft_12=0.020±0.000 | Capacity cliff differs by only 0 (< 1).… |
| exp_20_1 | ~ INCONCLUSIVE | ✓ | acc_A=0.167±0.000 | Auxiliary adjusts write rate but accuracy gain (0.0219) belo… |
| exp_20_2 | ✗ REFUTED | ✓ | acc_A=0.167±0.000 | Task acc unchanged (gap=0.0019 <= 0.005).… |
| exp_20_3 | ✗ REFUTED | ✓ | acc_A=0.235±0.000 | No-auxiliary baseline (acc_A=0.2353) nearly matches full sys… |
| exp_20_4 | ~ INCONCLUSIVE | ✓ | acc_lam_0_0=0.166±0.000 | Peak at lambda=0.01 (in_range=True); boundary conditions not… |
| exp_21_1 | ✗ REFUTED | ✓ | acc_A=0.031±0.004 | LSTM: acc=0.0269 util=0.1542. FF: acc=0.0384 util=0.0175. ac… |
| exp_21_2 | ✗ REFUTED | ✓ | acc_A=0.032±0.004 | Task-only: acc=0.0275. Oracle-augmented: acc=0.0331. gap=0.0… |
| exp_21_3 | ✗ REFUTED | ✓ | acc_A=0.032±0.004 | A(e2e): acc=0.0275 gq=-0.0108. B(distilled): acc=0.0244 gq=0… |
| exp_21_4 | ✗ REFUTED | ✓ | acc_A=0.035±0.005 | A(LSTM+task)=0.0325  B(FF+task)=0.0394  C(FF+oracle)=0.0319 … |

---

## Detailed Results by Category

### Category 1 — What To Write
*1 supported / 3 refuted / 4 inconclusive / 0 error*

#### exp_1_1  ✗ REFUTED
**Hypothesis:** Attention weight correlates positively with memory importance and attention-based memory outperforms random memory on retrieval tasks.

**Runs:** 6 (seeds: [123, 123, 42, 42, 777, 777])  |  **Avg duration:** 115s

**Metrics (mean ± std across seeds):**

- `attention_correlation` = **-0.5026**  *(stable across seeds)*
- `attention_memory_acc` = **0.0877**  *(stable across seeds)*
- `oracle_memory_acc` = **0.0869**  *(stable across seeds)*
- `random_memory_acc` = **0.0874**  *(stable across seeds)*

**Notes:** Pearson r=-0.5026. Attention acc=0.088 vs random=0.087 vs oracle=0.087.

---
#### exp_1_2  ✗ REFUTED
**Hypothesis:** A memory built from high-surprise (high-perplexity) tokens supports better retrieval than attention-based memory.

**Runs:** 6 (seeds: [123, 123, 42, 42, 777, 777])  |  **Avg duration:** 54s

**Metrics (mean ± std across seeds):**

- `attention_acc` = **0.1442**  *(stable across seeds)*
- `baseline_acc` = **0.0306**  *(stable across seeds)*
- `gap_surprise_minus_attention` = **-0.0048**  *(stable across seeds)*
- `surprise_acc` = **0.1394**  *(stable across seeds)*

**Notes:** Surprise acc=0.139, attention acc=0.144, no-memory baseline=0.031. Gap=-0.0048.

---
#### exp_1_3  ~ INCONCLUSIVE
**Hypothesis:** Storing tokens where gradient magnitude is highest produces memories that generalize better than attention-selected memories.

**Runs:** 6 (seeds: [123, 123, 42, 42, 777, 777])  |  **Avg duration:** 51s

**Metrics (mean ± std across seeds):**

- `attention_acc` = **0.0900**  *(stable across seeds)*
- `gap_gradient_minus_attention` = **0.0168**  *(stable across seeds)*
- `gradient_acc` = **0.1068**  *(stable across seeds)*
- `random_acc` = **0.1086**  *(stable across seeds)*

**Notes:** Gradient acc=0.107, attention=0.090, random=0.109. Gap=+0.0168.

---
#### exp_1_4  ~ INCONCLUSIVE
**Hypothesis:** Diversity-driven storage (maximally dissimilar entries) outperforms importance-driven storage on recall tasks.

**Runs:** 6 (seeds: [123, 123, 42, 42, 777, 777])  |  **Avg duration:** 41s

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

**Runs:** 6 (seeds: [123, 123, 42, 42, 777, 777])  |  **Avg duration:** 136s

**Metrics (mean ± std across seeds):**

- `acc_attention` = **0.1210**  *(stable across seeds)*
- `acc_learned` = **0.1242**  *(stable across seeds)*
- `acc_random` = **0.1201**  *(stable across seeds)*
- `acc_surprise` = **0.1176**  *(stable across seeds)*
- `loss_attention` = **3.1545**  *(stable across seeds)*
- `loss_learned` = **3.0535**  *(stable across seeds)*
- `loss_random` = **3.1498**  *(stable across seeds)*
- `loss_surprise` = **3.1317**  *(stable across seeds)*
- `ranking` = [['learned', 'attention', 'random', 'surprise'], ['learned', 'attention', 'random', 'surprise'], ['learned', 'attention', 'random', 'surprise'], ['learned', 'attention', 'random', 'surprise'], ['learned', 'attention', 'random', 'surprise'], ['learned', 'attention', 'random', 'surprise']]

**Notes:** Learned gate delta over best baseline: +0.003. Ranking: ['learned', 'attention', 'random', 'surprise'].

---
#### exp_1_6  ~ INCONCLUSIVE
**Hypothesis:** Cosine-similarity deduplication at write time improves retrieval precision without dangerous information loss.

**Runs:** 6 (seeds: [123, 123, 42, 42, 777, 777])  |  **Avg duration:** 70s

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

**Runs:** 6 (seeds: [123, 123, 42, 42, 777, 777])  |  **Avg duration:** 50s

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

**Runs:** 6 (seeds: [123, 123, 42, 42, 777, 777])  |  **Avg duration:** 44s

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

**Runs:** 6 (seeds: [123, 123, 42, 42, 777, 777])  |  **Avg duration:** 1593s

**Metrics (mean ± std across seeds):**

- `catastrophic_cliff_detected` = [False, False, False, False, False, False]
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

**Runs:** 6 (seeds: [123, 123, 42, 42, 777, 777])  |  **Avg duration:** 18s

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

**Runs:** 6 (seeds: [123, 123, 42, 42, 777, 777])  |  **Avg duration:** 3s

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

**Runs:** 6 (seeds: [123, 123, 42, 42, 777, 777])  |  **Avg duration:** 272s

**Metrics (mean ± std across seeds):**

- `clear_peak_detected` = [False, False, False, False, False, False]
- `cosine_sim.16` = **0.3402**  *(stable across seeds)*
- `cosine_sim.32` = **0.2274**  *(stable across seeds)*
- `cosine_sim.4` = **0.6425**  *(stable across seeds)*
- `cosine_sim.64` = **0.1531**  *(stable across seeds)*
- `cosine_sim.8` = **0.4728**  *(stable across seeds)*
- `is_flat` = [False, False, False, False, False, False]
- `is_monotone_decrease` = [True, True, True, True, True, True]
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

**Runs:** 6 (seeds: [123, 123, 42, 42, 777, 777])  |  **Avg duration:** 14s

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

**Runs:** 6 (seeds: [123, 123, 42, 42, 777, 777])  |  **Avg duration:** 10s

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

**Runs:** 6 (seeds: [123, 123, 42, 42, 777, 777])  |  **Avg duration:** 9s

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

**Runs:** 6 (seeds: [123, 123, 42, 42, 777, 777])  |  **Avg duration:** 7s

**Metrics (mean ± std across seeds):**

- `catastrophic` = [False, False, False, False, False, False]
- `quality_a_in_mixed_seq` = **0.6087**  *(stable across seeds)*
- `quality_domain_a` = **0.6050**  *(stable across seeds)*
- `quality_domain_b_shifted` = **0.2901**  *(stable across seeds)*
- `quality_drop` = **0.3149**  *(stable across seeds)*

**Notes:** Domain A (in-distribution): 0.605. Domain B (shifted): 0.290. Drop: 0.315 vs catastrophic threshold 0.2. Graceful degradation (drop < threshold): False. Catastrophic failure: False.

---
#### exp_2_9  ✗ REFUTED
**Hypothesis:** Minimizing reconstruction loss and maximizing downstream retrieval accuracy are fundamentally different objectives and produce measurably different compressed representations.

**Runs:** 6 (seeds: [123, 123, 42, 42, 777, 777])  |  **Avg duration:** 207s

**Metrics (mean ± std across seeds):**

- `objectives_diverge` = [False, False, False, False, False, False]
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

**Runs:** 6 (seeds: [123, 123, 42, 42, 777, 777])  |  **Avg duration:** 35s

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

**Runs:** 6 (seeds: [123, 123, 42, 42, 777, 777])  |  **Avg duration:** 22s

**Metrics (mean ± std across seeds):**

- `A_no_signal.collapsed_always` = [False, False, False, False, False, False]
- `A_no_signal.collapsed_zero` = [False, False, False, False, False, False]
- `A_no_signal.final_write_rate` = **0.1927**  *(stable across seeds)*
- `B_entropy.collapsed_always` = [False, False, False, False, False, False]
- `B_entropy.collapsed_zero` = [False, False, False, False, False, False]
- `B_entropy.final_write_rate` = **0.7448**  *(stable across seeds)*
- `C_reconstruction.collapsed_always` = [False, False, False, False, False, False]
- `C_reconstruction.collapsed_zero` = [False, False, False, False, False, False]
- `C_reconstruction.final_write_rate` = **0.2786**  *(stable across seeds)*
- `D_penalty.collapsed_always` = [True, True, True, True, True, True]
- `D_penalty.collapsed_zero` = [False, False, False, False, False, False]
- `D_penalty.final_write_rate` = **0.9505**  *(stable across seeds)*

**Notes:** Regime A collapsed: False. Other regimes collapsed: True. Write rates: A_no_signal=0.19, B_entropy=0.74, C_reconstruction=0.28, D_penalty=0.95.

---
#### exp_3_3  ✗ REFUTED
**Hypothesis:** Writing later in context (more processed representations) outperforms early writing (raw representations) for inferential tasks.

**Runs:** 6 (seeds: [123, 123, 42, 42, 777, 777])  |  **Avg duration:** 83s

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

**Runs:** 6 (seeds: [123, 123, 42, 42, 777, 777])  |  **Avg duration:** 38s

**Metrics (mean ± std across seeds):**

- `acc_gap` = **-0.1729**  *(stable across seeds)*
- `boundary_acc` = **0.0359**  *(stable across seeds)*
- `boundary_per_segment_acc` = [[0.0391, 0.0406, 0.0281], [0.0391, 0.0406, 0.0281], [0.0391, 0.0406, 0.0281], [0.0391, 0.0406, 0.0281], [0.0391, 0.0406, 0.0281], [0.0391, 0.0406, 0.0281]]
- `fixed_acc` = **0.2089**  *(stable across seeds)*
- `fixed_per_segment_acc` = [[0.0234, 0.0141, 0.5891], [0.0234, 0.0141, 0.5891], [0.0234, 0.0141, 0.5891], [0.0234, 0.0141, 0.5891], [0.0234, 0.0141, 0.5891], [0.0234, 0.0141, 0.5891]]
- `segment_balance_score` = **0.9932**  *(stable across seeds)*

**Notes:** Boundary vs fixed accuracy gap: -0.173. Boundary per-segment: [0.039, 0.041, 0.028]. Fixed per-segment: [0.023, 0.014, 0.589]. Boundary balance score: 0.993.

---
#### exp_3_5  ✓ SUPPORTED
**Hypothesis:** Downstream retrieval quality degrades measurably when write latency exceeds a specific token distance threshold.

**Runs:** 6 (seeds: [123, 123, 42, 42, 777, 777])  |  **Avg duration:** 117s

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

**Runs:** 6 (seeds: [123, 123, 42, 42, 777, 777])  |  **Avg duration:** 37s

**Metrics (mean ± std across seeds):**

- `acc_gap` = **0.1328**  *(stable across seeds)*
- `forward_acc` = **0.0391**  *(stable across seeds)*
- `retroactive_write_rate` = **0.0833**  *(stable across seeds)*
- `two_pass_acc` = **0.1719**  *(stable across seeds)*

**Notes:** Two-pass vs forward-only accuracy gap: +0.133. Retroactive write rate (fraction of tokens upgraded): 0.083. Forward pass writes 4 slots; revision adds up to 2 more.

---
#### exp_3_7  ✗ REFUTED
**Hypothesis:** A controller given a fixed write budget learns to allocate it non-uniformly in a way that improves performance over uniform allocation.

**Runs:** 6 (seeds: [123, 123, 42, 42, 777, 777])  |  **Avg duration:** 55s

**Metrics (mean ± std across seeds):**

- `acc_delta` = **0.0266**  *(stable across seeds)*
- `adaptive_acc` = **0.2437**  *(stable across seeds)*
- `allocation_gini` = **0.1639**  *(stable across seeds)*
- `allocation_per_block` = [[1.9859, 1.0, 1.0234, 1.9906], [1.9859, 1.0, 1.0234, 1.9906], [1.9859, 1.0, 1.0234, 1.9906], [1.9859, 1.0, 1.0234, 1.9906], [1.9859, 1.0, 1.0234, 1.9906], [1.9859, 1.0, 1.0234, 1.9906]]
- `uniform_acc` = **0.2172**  *(stable across seeds)*
- `uniform_alloc_per_block` = [[1.0, 1.0, 2.0, 2.0], [1.0, 1.0, 2.0, 2.0], [1.0, 1.0, 2.0, 2.0], [1.0, 1.0, 2.0, 2.0], [1.0, 1.0, 2.0, 2.0], [1.0, 1.0, 2.0, 2.0]]

**Notes:** Adaptive vs uniform accuracy delta: +0.027. Allocation Gini coefficient: 0.164 (threshold 0.3). Adaptive per-block: [1.99, 1.0, 1.02, 1.99]. Uniform per-block: [1.0, 1.0, 2.0, 2.0] (fixed: [1, 1, 2, 2]).

---

### Category 4 — What To Read
*4 supported / 1 refuted / 4 inconclusive / 0 error*

#### exp_4_1  ✓ SUPPORTED
**Hypothesis:** A dedicated query formulation module outperforms direct use of the current hidden state as a retrieval query.

**Runs:** 6 (seeds: [123, 123, 42, 42, 777, 777])  |  **Avg duration:** 5s

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

**Runs:** 6 (seeds: [123, 123, 42, 42, 777, 777])  |  **Avg duration:** 45s

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

**Runs:** 6 (seeds: [123, 123, 42, 42, 777, 777])  |  **Avg duration:** 20s

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

**Runs:** 6 (seeds: [123, 123, 42, 42, 777, 777])  |  **Avg duration:** 6s

**Metrics (mean ± std across seeds):**

- `hard_final_acc` = **0.0166**  *(stable across seeds)*
- `hard_is_more_accurate` = [True, True, True, True, True, True]
- `hard_loss_variance` = **0.0001**  *(stable across seeds)*
- `soft_final_acc` = **0.0164**  *(stable across seeds)*
- `soft_is_more_stable` = [True, True, True, True, True, True]
- `soft_loss_variance` = **0.0000**  *(stable across seeds)*

**Notes:** Soft var=0.000033 vs Hard var=0.000076; Soft acc=0.0164 vs Hard acc=0.0166.

---
#### exp_4_5  ~ INCONCLUSIVE
**Hypothesis:** Simultaneous cross-tier retrieval achieves better recall than sequential cascading retrieval.

**Runs:** 6 (seeds: [123, 123, 42, 42, 777, 777])  |  **Avg duration:** 14s

**Metrics (mean ± std across seeds):**

- `avg_tiers_queried_sequential` = **3.0000**  *(stable across seeds)*
- `gap_sim_minus_seq` = **0.0006**  *(stable across seeds)*
- `sequential_acc` = **0.0156**  *(stable across seeds)*
- `simultaneous_acc` = **0.0163**  *(stable across seeds)*

**Notes:** Simultaneous acc=0.0163 vs Sequential acc=0.0156, gap=+0.0006 (threshold +0.02 for SUPPORTED). Sequential avg tiers queried=3.00.

---
#### exp_4_6  ~ INCONCLUSIVE
**Hypothesis:** For exact recall tasks similarity-based retrieval wins; for inferential completion tasks reconstruction-based retrieval wins.

**Runs:** 6 (seeds: [123, 123, 42, 42, 777, 777])  |  **Avg duration:** 13s

**Metrics (mean ± std across seeds):**

- `rec_wins_inferential` = [False, False, False, False, False, False]
- `recon_exact_acc` = **0.0145**  *(stable across seeds)*
- `recon_inferential_acc` = **0.0340**  *(stable across seeds)*
- `sim_wins_exact` = [False, False, False, False, False, False]
- `similarity_exact_acc` = **0.0163**  *(stable across seeds)*
- `similarity_inferential_acc` = **0.0289**  *(stable across seeds)*

**Notes:** Exact(Sim=0.0163, Rec=0.0145), Inferential(Sim=0.0289, Rec=0.0340). Specialisation A=False, B=False.

---
#### exp_4_7  ✓ SUPPORTED
**Hypothesis:** A learned read gate can be trained to return null on tasks where most queries have no relevant memory content, without explicit null supervision.

**Runs:** 6 (seeds: [123, 123, 42, 42, 777, 777])  |  **Avg duration:** 9s

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

**Runs:** 6 (seeds: [123, 123, 42, 42, 777, 777])  |  **Avg duration:** 20s

**Metrics (mean ± std across seeds):**

- `acc_at_N_0` = **0.0172**  *(stable across seeds)*
- `acc_at_N_12` = **0.0148**  *(stable across seeds)*
- `acc_at_N_16` = **0.0152**  *(stable across seeds)*
- `acc_at_N_2` = **0.0167**  *(stable across seeds)*
- `acc_at_N_4` = **0.0160**  *(stable across seeds)*
- `acc_at_N_8` = **0.0144**  *(stable across seeds)*
- `degradation_is_nonlinear` = [True, True, True, True, True, True]
- `saturation_point_N` = **4.0000**  *(stable across seeds)*

**Notes:** Accuracy is flat across N values; no interference detected.

---
#### exp_4_9  ✓ SUPPORTED
**Hypothesis:** A learned retrieval mechanism can be trained to retrieve two separate memory entries and compose them to answer questions neither entry alone can answer.

**Runs:** 6 (seeds: [123, 123, 42, 42, 777, 777])  |  **Avg duration:** 10s

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

**Runs:** 6 (seeds: [123, 123, 42, 42, 777, 777])  |  **Avg duration:** 8s

**Metrics (mean ± std across seeds):**

- `A_task_only.collapsed` = [False, False, False, False, False, False]
- `A_task_only.final_read_rate` = **0.1562**  *(stable across seeds)*
- `A_task_only.mode` = ['stable', 'stable', 'stable', 'stable', 'stable', 'stable']
- `B_sparsity.collapsed` = [True, True, True, True, True, True]
- `B_sparsity.final_read_rate` = **0.0938**  *(stable across seeds)*
- `B_sparsity.mode` = ['NEVER', 'NEVER', 'NEVER', 'NEVER', 'NEVER', 'NEVER']
- `C_coverage.collapsed` = [False, False, False, False, False, False]
- `C_coverage.final_read_rate` = **0.1562**  *(stable across seeds)*
- `C_coverage.mode` = ['stable', 'stable', 'stable', 'stable', 'stable', 'stable']
- `D_confidence.collapsed` = [False, False, False, False, False, False]
- `D_confidence.final_read_rate` = **0.6562**  *(stable across seeds)*
- `D_confidence.mode` = ['stable', 'stable', 'stable', 'stable', 'stable', 'stable']

**Notes:** Regime A collapsed: False (stable). Read rates: A_task_only=0.16, B_sparsity=0.09, C_coverage=0.16, D_confidence=0.66.

---
#### exp_5_2  ✓ SUPPORTED
**Hypothesis:** Optimal read frequency is task-dependent and cannot be determined by a single fixed schedule across task types.

**Runs:** 6 (seeds: [123, 123, 42, 42, 777, 777])  |  **Avg duration:** 412s

**Metrics (mean ± std across seeds):**

- `factual_qa_freq1` = **0.9956**  *(stable across seeds)*
- `factual_qa_freq2` = **0.9969**  *(stable across seeds)*
- `factual_qa_freq4` = **0.9975**  *(stable across seeds)*
- `factual_qa_freq8` = **0.9944**  *(stable across seeds)*
- `frequencies_differ` = [True, True, True, True, True, True]
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

**Runs:** 6 (seeds: [123, 123, 42, 42, 777, 777])  |  **Avg duration:** 93s

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

**Runs:** 6 (seeds: [123, 123, 42, 42, 777, 777])  |  **Avg duration:** 22s

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

**Runs:** 6 (seeds: [123, 123, 42, 42, 777, 777])  |  **Avg duration:** 42s

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

**Runs:** 6 (seeds: [123, 123, 42, 42, 777, 777])  |  **Avg duration:** 54s

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

**Runs:** 6 (seeds: [123, 123, 42, 42, 777, 777])  |  **Avg duration:** 44s

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

**Runs:** 6 (seeds: [123, 123, 42, 42, 777, 777])  |  **Avg duration:** 139s

**Metrics (mean ± std across seeds):**

- `eviction_policy_ranking` = ['lfu(0.059) > learned(0.059) > random(0.022) > lru(0.020)', 'lfu(0.059) > learned(0.059) > random(0.022) > lru(0.020)', 'lfu(0.059) > learned(0.059) > random(0.022) > lru(0.020)', 'lfu(0.059) > learned(0.059) > random(0.022) > lru(0.020)', 'lfu(0.059) > learned(0.059) > random(0.022) > lru(0.020)', 'lfu(0.059) > learned(0.059) > random(0.022) > lru(0.020)']
- `learned_acc` = **0.0591**  *(stable across seeds)*
- `learned_vs_lru_gap` = **0.0395**  *(stable across seeds)*
- `lfu_acc` = **0.0594**  *(stable across seeds)*
- `lru_acc` = **0.0195**  *(stable across seeds)*
- `random_acc` = **0.0220**  *(stable across seeds)*

**Notes:** Learned vs LRU gap: 0.040. Threshold for SUPPORTED: >0.03. Ranking: lfu(0.059) > learned(0.059) > random(0.022) > lru(0.020).

---
#### exp_6_2  ~ INCONCLUSIVE
**Hypothesis:** Graceful degradation via iterative compression outperforms hard eviction for long-context tasks where storage budget is the binding constraint.

**Runs:** 6 (seeds: [123, 123, 42, 42, 777, 777])  |  **Avg duration:** 131s

**Metrics (mean ± std across seeds):**

- `avg_compression_level_at_query_time` = **1.9333**  *(stable across seeds)*
- `compression_acc` = **0.0289**  *(stable across seeds)*
- `gap_compression_minus_lru` = **-0.0044**  *(stable across seeds)*
- `lru_acc` = **0.0333**  *(stable across seeds)*

**Notes:** Compression vs LRU gap: -0.004. Average compression level (0=full, 1=half, 2=quarter): 1.933.

---
#### exp_6_3  ✓ SUPPORTED
**Hypothesis:** A controller can learn to evict domain-mismatched memories when input distribution shifts, without explicit domain labels.

**Runs:** 6 (seeds: [123, 123, 42, 42, 777, 777])  |  **Avg duration:** 351s

**Metrics (mean ± std across seeds):**

- `gap_selective_minus_lru` = **0.0764**  *(stable across seeds)*
- `lru_phase2_acc` = **0.0364**  *(stable across seeds)*
- `selective_eviction_rate_for_phase1_entries` = **0.7012**  *(stable across seeds)*
- `selective_phase2_acc` = **0.1128**  *(stable across seeds)*

**Notes:** Selective vs LRU gap on phase-2 queries: 0.076. Eviction rate for phase-1 entries: 0.701.

---
#### exp_6_4  ~ INCONCLUSIVE
**Hypothesis:** A controller can learn which memories deserve protection (never evict) without explicit supervision, and performance degrades predictably outside an optimal protected-set size.

**Runs:** 6 (seeds: [123, 123, 42, 42, 777, 777])  |  **Avg duration:** 498s

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

**Runs:** 6 (seeds: [123, 123, 42, 42, 777, 777])  |  **Avg duration:** 459s

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

**Runs:** 6 (seeds: [123, 123, 42, 42, 777, 777])  |  **Avg duration:** 19s

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

**Runs:** 6 (seeds: [123, 123, 42, 42, 777, 777])  |  **Avg duration:** 700s

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

**Runs:** 6 (seeds: [123, 123, 42, 42, 777, 777])  |  **Avg duration:** 279s

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

**Runs:** 6 (seeds: [123, 123, 42, 42, 777, 777])  |  **Avg duration:** 30s

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
- `ranking_by_accuracy` = [['Gumbel', 'STE', 'REINFORCE'], ['Gumbel', 'STE', 'REINFORCE'], ['Gumbel', 'STE', 'REINFORCE'], ['Gumbel', 'STE', 'REINFORCE'], ['Gumbel', 'STE', 'REINFORCE'], ['Gumbel', 'STE', 'REINFORCE']]
- `ranking_by_stability` = [['Gumbel', 'STE', 'REINFORCE'], ['Gumbel', 'STE', 'REINFORCE'], ['Gumbel', 'STE', 'REINFORCE'], ['Gumbel', 'STE', 'REINFORCE'], ['Gumbel', 'STE', 'REINFORCE'], ['Gumbel', 'STE', 'REINFORCE']]

**Notes:** Gumbel most stable: True. Both beat REINFORCE: True. Acc: STE=0.126, Gumbel=0.128, REINFORCE=0.102

---
#### exp_7_2  ✓ SUPPORTED
**Hypothesis:** There exists a maximum controller complexity (measured in parameter count) beyond which the controller's overhead exceeds its efficiency contribution.

**Runs:** 6 (seeds: [123, 123, 42, 42, 777, 777])  |  **Avg duration:** 51s

**Metrics (mean ± std across seeds):**

- `acc_per_complexity.Large` = **0.1083**  *(stable across seeds)*
- `acc_per_complexity.Medium` = **0.1227**  *(stable across seeds)*
- `acc_per_complexity.Small` = **0.0939**  *(stable across seeds)*
- `acc_per_complexity.Tiny` = **0.1220**  *(stable across seeds)*
- `acc_per_complexity.XL` = **0.1044**  *(stable across seeds)*
- `declines_at_large_xl` = [True, True, True, True, True, True]
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
- `largest_is_best` = [False, False, False, False, False, False]
- `optimal_complexity_level` = ['Medium', 'Medium', 'Medium', 'Medium', 'Medium', 'Medium']
- `params_per_complexity.Large` = **164865.0000**  *(stable across seeds)*
- `params_per_complexity.Medium` = **16897.0000**  *(stable across seeds)*
- `params_per_complexity.Small` = **4225.0000**  *(stable across seeds)*
- `params_per_complexity.Tiny` = **65.0000**  *(stable across seeds)*
- `params_per_complexity.XL` = **427521.0000**  *(stable across seeds)*
- `peaks_early` = [True, True, True, True, True, True]

**Notes:** Peak efficiency at: Medium. Efficiency ratios: {'Tiny': 0.0, 'Small': -0.006737515755360065, 'Medium': 0.00011239988114518264, 'Large': -0.0017541632748579459, 'XL': -0.0020083612849882103}. Acc: {'Tiny': 0.12203125, 'Small': 0.09390625, 'Medium': 0.12265625, 'Large': 0.10828125, 'XL': 0.104375}.

---
#### exp_7_3  ~ INCONCLUSIVE
**Hypothesis:** A controller trained on factual QA generalizes its memory management policies to reasoning tasks but not to generation tasks.

**Runs:** 6 (seeds: [123, 123, 42, 42, 777, 777])  |  **Avg duration:** 9s

**Metrics (mean ± std across seeds):**

- `factual_acc` = **0.0939**  *(stable across seeds)*
- `factual_to_generation_gap` = **0.0923**  *(stable across seeds)*
- `factual_to_reasoning_gap` = **0.0395**  *(stable across seeds)*
- `generation_acc` = **0.0016**  *(stable across seeds)*
- `generation_fails` = [False, False, False, False, False, False]
- `reasoning_acc` = **0.0544**  *(stable across seeds)*
- `reasoning_transfers` = [True, True, True, True, True, True]

**Notes:** Reasoning gap=0.040 (threshold <0.15: True). Generation gap=0.092 (threshold >0.20: False).

---
#### exp_7_4  ~ INCONCLUSIVE
**Hypothesis:** Meaningful memory management behavior requires at minimum two layers of non-linearity in the controller network.

**Runs:** 6 (seeds: [123, 123, 42, 42, 777, 777])  |  **Avg duration:** 32s

**Metrics (mean ± std across seeds):**

- `acc_per_depth.0` = **0.1220**  *(stable across seeds)*
- `acc_per_depth.1` = **0.1158**  *(stable across seeds)*
- `acc_per_depth.2` = **0.1212**  *(stable across seeds)*
- `acc_per_depth.3` = **0.1119**  *(stable across seeds)*
- `meaningful_threshold_per_depth.0` = [False, False, False, False, False, False]
- `meaningful_threshold_per_depth.1` = [False, False, False, False, False, False]
- `meaningful_threshold_per_depth.2` = [False, False, False, False, False, False]
- `meaningful_threshold_per_depth.3` = [False, False, False, False, False, False]
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

**Runs:** 6 (seeds: [123, 123, 42, 42, 777, 777])  |  **Avg duration:** 34s

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

**Runs:** 6 (seeds: [123, 123, 42, 42, 777, 777])  |  **Avg duration:** 11s

**Metrics (mean ± std across seeds):**

- `adversarial_ratio` = **1.0146**  *(stable across seeds)*
- `adversarial_write_rate` = **0.5819**  *(stable across seeds)*
- `normal_write_rate` = **0.5735**  *(stable across seeds)*
- `self_corrects` = [False, False, False, False, False, False]
- `write_rate_at_1000` = **0.5887**  *(stable across seeds)*
- `write_rate_at_1500` = **0.5819**  *(stable across seeds)*
- `write_rate_at_500` = **0.5882**  *(stable across seeds)*

**Notes:** Adversarial ratio: 1.015 (threshold >1.5: False). Self-corrects: False. Checkpoint rates: [0.5881532311439515, 0.5887086486816406, 0.5818839371204376].

---
#### exp_7_7  ✗ REFUTED
**Hypothesis:** Write quality (not read quality, compression ratio, or eviction policy) is the first performance bottleneck encountered during controller training.

**Runs:** 6 (seeds: [123, 123, 42, 42, 777, 777])  |  **Avg duration:** 11s

**Metrics (mean ± std across seeds):**

- `bottleneck_at_20pct_training` = ['read_accuracy', 'read_accuracy', 'read_accuracy', 'read_accuracy', 'read_accuracy', 'read_accuracy']
- `bottleneck_at_50pct_training` = ['read_accuracy', 'read_accuracy', 'read_accuracy', 'read_accuracy', 'read_accuracy', 'read_accuracy']
- `bottleneck_order` = [['read_accuracy', 'eviction_correctness', 'write_quality', 'compression_fidelity'], ['read_accuracy', 'eviction_correctness', 'write_quality', 'compression_fidelity'], ['read_accuracy', 'eviction_correctness', 'write_quality', 'compression_fidelity'], ['read_accuracy', 'eviction_correctness', 'write_quality', 'compression_fidelity'], ['read_accuracy', 'eviction_correctness', 'write_quality', 'compression_fidelity'], ['read_accuracy', 'eviction_correctness', 'write_quality', 'compression_fidelity']]
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

**Runs:** 6 (seeds: [123, 123, 42, 42, 777, 777])  |  **Avg duration:** 18s

**Metrics (mean ± std across seeds):**

- `curriculum_acc` = **0.0750**  *(stable across seeds)*
- `curriculum_gate_collapse` = [False, False, False, False, False, False]
- `curriculum_loss_variance` = **0.0014**  *(stable across seeds)*
- `curriculum_read_collapsed` = [False, False, False, False, False, False]
- `curriculum_write_collapsed` = [False, False, False, False, False, False]
- `joint_acc` = **0.1095**  *(stable across seeds)*
- `joint_gate_collapse` = [False, False, False, False, False, False]
- `joint_loss_variance` = **0.0058**  *(stable across seeds)*
- `joint_read_collapsed` = [False, False, False, False, False, False]
- `joint_write_collapsed` = [False, False, False, False, False, False]

**Notes:** Curriculum acc >= joint acc: False. Curriculum loss_var < joint loss_var: True. Joint acc=0.110, Curriculum acc=0.075. Joint var=0.00581, Curriculum var=0.00137.

---
#### exp_7_9  ✓ SUPPORTED
**Hypothesis:** The controller's write and read decisions are interpretable (non-random, correlating with human-meaningful features) in their simplest form before any task-specific training.

**Runs:** 6 (seeds: [123, 123, 42, 42, 777, 777])  |  **Avg duration:** 11s

**Metrics (mean ± std across seeds):**

- `heatmaps_saved_to` = ['/Users/wscholl/drex/research/results/interpretability', '/Users/wscholl/drex/research/results/interpretability', '/Users/wscholl/drex/research/results/interpretability', '/Users/wscholl/drex/research/results/interpretability', '/Users/wscholl/drex/research/results/interpretability', '/Users/wscholl/drex/research/results/interpretability']
- `interpretability_score_trained` = **0.0595**  *(stable across seeds)*
- `interpretability_score_untrained` = **0.0549**  *(stable across seeds)*
- `trained.corr_numeric_vs_gate` = **0.0757**  *(stable across seeds)*
- `trained.corr_position_vs_gate` = **-0.0039**  *(stable across seeds)*
- `trained.corr_punct_vs_gate` = **-0.0474**  *(stable across seeds)*
- `trained.corr_rare_vs_gate` = **0.0553**  *(stable across seeds)*
- `trained.gate_mean` = **0.5085**  *(stable across seeds)*
- `trained.gate_nonrandom` = [True, True, True, True, True, True]
- `trained.gate_std` = **0.1505**  *(stable across seeds)*
- `untrained.corr_numeric_vs_gate` = **0.0622**  *(stable across seeds)*
- `untrained.corr_position_vs_gate` = **-0.0077**  *(stable across seeds)*
- `untrained.corr_punct_vs_gate` = **-0.0429**  *(stable across seeds)*
- `untrained.corr_rare_vs_gate` = **0.0596**  *(stable across seeds)*
- `untrained.gate_mean` = **0.5099**  *(stable across seeds)*
- `untrained.gate_nonrandom` = [True, True, True, True, True, True]
- `untrained.gate_std` = **0.1457**  *(stable across seeds)*

**Notes:** Interp score: trained=0.059 untrained=0.055. Gate is non-random. Heatmaps saved to results/interpretability/.

---

### Category 8 — Mechanistic Investigations (Phase 2)
*0 supported / 2 refuted / 2 inconclusive / 0 error*

#### exp_8_1  ✗ REFUTED ⚠ inconsistent across seeds ['INCONCLUSIVE', 'REFUTED', 'REFUTED']
**Hypothesis:** The attention-importance anti-correlation (r=-0.503 from exp_1_1) is caused by softmax normalization forcing zero-sum redistribution, not a semantic mismatch — removing normalization will produce positive or near-zero correlation.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 7s

**Metrics (mean ± std across seeds):**

- `pearson_r_entropic` = **-0.0772** ± 0.4460  *(runs: 0.220, 0.139, -0.590)*
- `pearson_r_raw` = **-0.2887** ± 0.3413  *(runs: -0.077, -0.106, -0.682)*
- `pearson_r_softmax` = **-0.0333** ± 0.4882  *(runs: 0.229, 0.268, -0.597)*

**Notes:** Softmax r=0.2286, raw dot-product r=-0.0775, entropy-normalized r=0.2198. Hypothesis: raw should be > 0.05 while softmax < -0.10.

---
#### exp_8_2  ~ INCONCLUSIVE ⚠ inconsistent across seeds ['INCONCLUSIVE', 'SUPPORTED', 'INCONCLUSIVE']
**Hypothesis:** The natural ~16-20% gate activity equilibrium is not fixed by architecture but scales with task memory demand — harder tasks requiring more KV pairs will drive equilibrium write rates upward.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 69s

**Metrics (mean ± std across seeds):**

- `accuracies` = [[0.75, 0.0938, 0.1191, 0.0469], [0.6582, 0.3398, 0.2227, 0.0801], [0.377, 0.4043, 0.2656, 0.0957]]
- `kv_levels` = [[1, 2, 4, 6], [1, 2, 4, 6], [1, 2, 4, 6]]
- `pearson_r_difficulty_vs_rate` = **0.3976** ± 0.4053  *(runs: 0.302, 0.842, 0.049)*
- `write_rate_variance` = **0.0225** ± 0.0086  *(runs: 0.030, 0.013, 0.025)*
- `write_rates` = [[0.2829, 0.6731, 0.5961, 0.4431], [0.3564, 0.4239, 0.3966, 0.6134], [0.4159, 0.2534, 0.6117, 0.3162]]

**Notes:** KV levels [1, 2, 4, 6]: write rates [0.2829, 0.6731, 0.5961, 0.4431]. Pearson r=0.3017, variance=0.029855.

---
#### exp_8_3  ✗ REFUTED
**Hypothesis:** The write-evict correlation collapse (r=0.990 in exp_6_7) is gradient aliasing — both gates receive identical gradient from shared loss. Oracle pre-training of each gate on independent labels breaks this, yielding write-evict correlation < 0.5.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 23s

**Metrics (mean ± std across seeds):**

- `acc_A` = **0.2125** ± 0.0515  *(runs: 0.272, 0.184, 0.181)*
- `acc_B` = **0.2239** ± 0.0144  *(runs: 0.241, 0.216, 0.216)*
- `acc_C` = **0.1198** ± 0.0407  *(runs: 0.078, 0.122, 0.159)*
- `write_evict_corr_A` = **0.2876** ± 0.2877  *(runs: 0.176, 0.073, 0.615)*
- `write_evict_corr_B` = **0.8816** ± 0.0618  *(runs: 0.814, 0.896, 0.935)*
- `write_evict_corr_C` = **-0.0930** ± 0.3026  *(runs: 0.250, -0.207, -0.322)*

**Notes:** Condition A (joint): corr=0.1756, acc=0.272. Condition B (oracle pretrain): corr=0.8137, acc=0.241. Condition C (grad isolation): corr=0.2500, acc=0.078.

---
#### exp_8_4  ~ INCONCLUSIVE ⚠ inconsistent across seeds ['INCONCLUSIVE', 'INCONCLUSIVE', 'REFUTED']
**Hypothesis:** The learned write gate's small advantage over random selection (exp_1_5 +0.003) comes from exploiting token position as proxy for importance, not from detecting semantic content.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 49s

**Metrics (mean ± std across seeds):**

- `acc_A` = **0.1498** ± 0.1103  *(runs: 0.216, 0.211, 0.022)*
- `acc_B` = **0.1031** ± 0.0581  *(runs: 0.036, 0.141, 0.133)*
- `acc_C` = **0.0577** ± 0.0426  *(runs: 0.033, 0.107, 0.033)*
- `content_corr_A` = **0.1418** ± 0.5253  *(runs: 0.384, 0.502, -0.461)*
- `content_corr_B` = **-0.2957** ± 0.0891  *(runs: -0.391, -0.214, -0.282)*
- `content_corr_C` = **0.0646** ± 0.1687  *(runs: -0.010, -0.054, 0.258)*
- `pos_corr_A` = **-0.0324** ± 0.3364  *(runs: -0.192, -0.259, 0.354)*
- `pos_corr_B` = **0.1271** ± 0.0788  *(runs: 0.195, 0.146, 0.041)*
- `pos_corr_C` = **0.0039** ± 0.1693  *(runs: -0.024, 0.185, -0.149)*

**Notes:** A (full): acc=0.216, pos_r=-0.1920. B (pos-blind): acc=0.036. C (pos-only): acc=0.033. acc_A - acc_B = 0.1800.

---

### Category 9 — Inconclusive Redesigns (Phase 2)
*3 supported / 1 refuted / 1 inconclusive / 0 error*

#### exp_9_1  ✗ REFUTED
**Hypothesis:** At 64x compression with 100-way gallery discrimination, retrieval-objective compressor achieves at least 15% higher Acc@1 than reconstruction-objective compressor, because 64x bottleneck forces genuine tradeoffs between fidelity and discriminability.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 62s

**Metrics (mean ± std across seeds):**

- `acc_at1_autoencoder` = **1.0000**  *(stable across seeds)*
- `acc_at1_contrastive` = **1.0000**  *(stable across seeds)*
- `recon_cosim_ae` = **0.1877** ± 0.0017  *(runs: 0.187, 0.187, 0.190)*
- `recon_cosim_cl` = **-0.0001** ± 0.0002  *(runs: -0.000, 0.000, -0.000)*
- `retrieval_gap` = **0.0000**  *(stable across seeds)*

**Notes:** AE Acc@1=1.0000, CL Acc@1=1.0000, gap=0.0000. Recon cosim: AE=0.1867, CL(dummy dec)=-0.0000.

---
#### exp_9_2  ✓ SUPPORTED
**Hypothesis:** With a 50/50 null-to-retrieval query distribution (fixing exp_4_7's degenerate 80% null), a learned read gate achieves null precision > 0.65 and retrieval recall > 0.65, outperforming always-null and always-retrieve baselines on harmonic F1.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 81s

**Metrics (mean ± std across seeds):**

- `is_degenerate_A` = [False, False, False]
- `is_degenerate_B` = [True, True, True]
- `is_degenerate_C` = [True, True, True]
- `is_degenerate_D` = [False, False, False]
- `null_f1_A` = **0.8120** ± 0.0113  *(runs: 0.800, 0.823, 0.813)*
- `null_f1_B` = **0.0000**  *(stable across seeds)*
- `null_f1_C` = **0.6640** ± 0.0046  *(runs: 0.664, 0.669, 0.659)*
- `null_f1_D` = **0.8579** ± 0.0037  *(runs: 0.857, 0.862, 0.855)*
- `null_precision_A` = **0.8106** ± 0.0189  *(runs: 0.830, 0.810, 0.792)*
- `null_precision_B` = **0.0000**  *(stable across seeds)*
- `null_precision_C` = **0.4970** ± 0.0052  *(runs: 0.497, 0.502, 0.492)*
- `null_precision_D` = **0.9386** ± 0.0054  *(runs: 0.933, 0.944, 0.939)*
- `null_recall_A` = **0.8146** ± 0.0361  *(runs: 0.773, 0.837, 0.834)*
- `null_recall_B` = **0.0000**  *(stable across seeds)*
- `null_recall_C` = **1.0000**  *(stable across seeds)*
- `null_recall_D` = **0.7900** ± 0.0043  *(runs: 0.792, 0.793, 0.785)*
- `retrieval_recall_A` = **0.8054** ± 0.0200  *(runs: 0.825, 0.807, 0.785)*
- `retrieval_recall_B` = **0.9437** ± 0.0016  *(runs: 0.945, 0.944, 0.942)*
- `retrieval_recall_C` = **0.0000**  *(stable across seeds)*
- `retrieval_recall_D` = **0.7824** ± 0.0121  *(runs: 0.771, 0.795, 0.781)*

**Notes:** Cond A (learned, p=0.5): null_f1=0.800, retr_recall=0.825, degenerate=False. Cond B (always-retrieve): null_f1=0.000. Cond D (p=0.2 control): degenerate=False.

---
#### exp_9_3  ~ INCONCLUSIVE
**Hypothesis:** When the memory controller achieves >70% domain A accuracy before domain B training, EWC with lambda=5.0 reduces catastrophic forgetting to <50% of standard fine-tuning's forgetting.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 14s

**Metrics (mean ± std across seeds):**

- `acc_a_after_ewc` = **0.0871** ± 0.0664  *(runs: 0.014, 0.104, 0.143)*
- `acc_a_after_std` = **0.0696** ± 0.0504  *(runs: 0.019, 0.071, 0.119)*
- `acc_a_before` = **0.1148** ± 0.0850  *(runs: 0.031, 0.113, 0.201)*
- `forgetting_ewc` = **0.0277** ± 0.0261  *(runs: 0.017, 0.009, 0.058)*
- `forgetting_ratio` = **0.7782** ± 0.6107  *(runs: 1.421, 0.206, 0.708)*
- `forgetting_std` = **0.0452** ± 0.0348  *(runs: 0.012, 0.043, 0.081)*
- `phase1_steps` = **1000.0000**  *(stable across seeds)*

**Notes:** Precondition failed: acc_A_before=0.031 < 0.70.

---
#### exp_9_4  ✓ SUPPORTED ⚠ inconsistent across seeds ['INCONCLUSIVE', 'SUPPORTED', 'SUPPORTED']
**Hypothesis:** With MEMORY_SLOTS=12 and K extended to 0-10, an interior optimum exists at K=3-6 — fewer protected slots are insufficient to cover all 3 critical items, more wastes capacity on non-critical entries, creating a U-shaped accuracy curve.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 27s

**Metrics (mean ± std across seeds):**

- `acc_range` = **0.1490** ± 0.0030  *(runs: 0.150, 0.146, 0.151)*
- `accuracies` = [[0.1519, 0.1437, 0.0225, 0.0469, 0.105, 0.1725], [0.0381, 0.1575, 0.1031, 0.1837, 0.1062, 0.1638], [0.0294, 0.0306, 0.0912, 0.0725, 0.1806, 0.1469]]
- `critical_accuracies` = [[0.0328, 0.2215, 0.02, 0.0583, 0.0959, 0.1686], [0.0279, 0.1462, 0.1161, 0.19, 0.1144, 0.1644], [0.0267, 0.036, 0.112, 0.0818, 0.1821, 0.1317]]
- `interior_peak_exists` = [False, True, True]
- `is_flat` = [False, False, False]
- `is_monotone` = [False, False, False]
- `k_opt` = **4.0000** ± 1.0000  *(runs: 5.000, 3.000, 4.000)*
- `max_acc` = **0.1790** ± 0.0058  *(runs: 0.172, 0.184, 0.181)*
- `min_acc` = **0.0300** ± 0.0078  *(runs: 0.022, 0.038, 0.029)*
- `noncritical_accuracies` = [[0.3345, 0.0189, 0.0239, 0.0314, 0.1157, 0.1692], [0.0571, 0.1701, 0.0844, 0.1779, 0.0963, 0.1652], [0.0325, 0.0237, 0.0591, 0.0562, 0.1758, 0.1691]]

**Notes:** K values [0, 2, 4, 6, 8, 10], accs [0.152, 0.144, 0.022, 0.047, 0.105, 0.172]. Optimal K=5, max_acc=0.172. Interior peak: False.

---
#### exp_9_5  ✓ SUPPORTED ⚠ inconsistent across seeds ['SUPPORTED', 'SUPPORTED', 'REFUTED']
**Hypothesis:** With all KV pairs concentrated in block 1 (positions 0-7 of 32), an adaptive write budget allocator learns non-uniform allocation (Gini > 0.5, block-1 fraction > 0.60), outperforming uniform allocation by > 5%.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 40s

**Metrics (mean ± std across seeds):**

- `acc_delta` = **0.0179** ± 0.1392  *(runs: 0.068, 0.125, -0.139)*
- `adaptive_acc` = **0.1725** ± 0.1240  *(runs: 0.227, 0.260, 0.031)*
- `block1_allocation_frac` = **0.6578** ± 0.5698  *(runs: 0.974, 1.000, 0.000)*
- `gini_coefficient` = **0.6617** ± 0.1410  *(runs: 0.736, 0.750, 0.499)*
- `mean_block_weights` = [[0.9738, 0.0004, 0.0004, 0.0255], [0.9997, 0.0, 0.0, 0.0003], [0.0, 0.4994, 0.4994, 0.0012]]
- `uniform_acc` = **0.1546** ± 0.0179  *(runs: 0.159, 0.135, 0.170)*

**Notes:** Uniform acc=0.159, adaptive acc=0.227, delta=0.068. Block-1 fraction=0.974, Gini=0.736. Block weights: [0.974, 0.0, 0.0, 0.026].

---

### Category 10 — Retroactive Writing Mechanism (Phase 2)
*0 supported / 1 refuted / 2 inconclusive / 0 error*

#### exp_10_1  ~ INCONCLUSIVE ⚠ inconsistent across seeds ['INCONCLUSIVE', 'REFUTED', 'INCONCLUSIVE', 'REFUTED', 'INCONCLUSIVE']
**Hypothesis:** The retroactive writing benefit decays to <5% accuracy gain when the revision gate's lookahead window is fewer than 6 tokens.

**Runs:** 5 (seeds: [123, 42, 42, 777, 777])  |  **Avg duration:** 213s

**Metrics (mean ± std across seeds):**

- `acc_w0` = **0.2591** ± 0.0068  *(runs: 0.264, 0.252, 0.264, 0.252, 0.264)*
- `acc_w2` = **0.1922** ± 0.0385  *(runs: 0.164, 0.234, 0.164, 0.234, 0.164)*
- `acc_w24` = **0.2172** ± 0.0171  *(runs: 0.205, 0.236, 0.205, 0.236, 0.205)*
- `acc_w4` = **0.2003** ± 0.0496  *(runs: 0.164, 0.255, 0.164, 0.255, 0.164)*
- `acc_w6` = **0.2185** ± 0.0188  *(runs: 0.205, 0.239, 0.205, 0.239, 0.205)*
- `acc_w8` = **0.2237** ± 0.0154  *(runs: 0.212, 0.241, 0.212, 0.241, 0.212)*
- `gap_at_24` = **-0.0419** ± 0.0240  *(runs: -0.059, -0.016, -0.059, -0.016, -0.059)*
- `gap_at_4` = **-0.0588** ± 0.0565  *(runs: -0.100, 0.003, -0.100, 0.003, -0.100)*
- `gap_w0` = **0.0000**  *(stable across seeds)*
- `gap_w2` = **-0.0669** ± 0.0454  *(runs: -0.100, -0.017, -0.100, -0.017, -0.100)*
- `gap_w24` = **-0.0419** ± 0.0240  *(runs: -0.059, -0.016, -0.059, -0.016, -0.059)*
- `gap_w4` = **-0.0588** ± 0.0565  *(runs: -0.100, 0.003, -0.100, 0.003, -0.100)*
- `gap_w6` = **-0.0406** ± 0.0257  *(runs: -0.059, -0.013, -0.059, -0.013, -0.059)*
- `gap_w8` = **-0.0353** ± 0.0223  *(runs: -0.052, -0.011, -0.052, -0.011, -0.052)*
- `max_gap_minus_min_gap` = **0.0681** ± 0.0437  *(runs: 0.100, 0.020, 0.100, 0.020, 0.100)*

**Notes:** gap@24=-0.059, gap@4=-0.100 — pattern inconclusive.

---
#### exp_10_2  ✗ REFUTED
**Hypothesis:** The retroactive writing benefit comes primarily (>80%) from adding new entries never written in the forward pass, not from re-encoding existing forward-pass entries with full context.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 62s

**Metrics (mean ± std across seeds):**

- `acc_A` = **0.2297**  *(stable across seeds)*
- `acc_B` = **0.1984**  *(stable across seeds)*
- `acc_C` = **0.2313**  *(stable across seeds)*
- `acc_D` = **0.2562**  *(stable across seeds)*
- `gain_B_over_A` = **-0.0312**  *(stable across seeds)*
- `gain_C_over_A` = **0.0016**  *(stable across seeds)*
- `gain_D_over_A` = **0.0266**  *(stable across seeds)*
- `new_write_fraction` = **-1.1765**  *(stable across seeds)*

**Notes:** Overwrite gain (0.002) exceeds new-write gain (-0.031) by >0.03.

---
#### exp_10_3  ~ INCONCLUSIVE
**Hypothesis:** The retroactive writing accuracy gain scales with sequence length (Pearson r > 0.8 across seq_len 24, 32, 48, 64).

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 86s

**Metrics (mean ± std across seeds):**

- `acc_fwd_24` = **0.1875**  *(stable across seeds)*
- `acc_fwd_32` = **0.1656**  *(stable across seeds)*
- `acc_fwd_48` = **0.0422**  *(stable across seeds)*
- `acc_fwd_64` = **0.0297**  *(stable across seeds)*
- `acc_retro_24` = **0.1344**  *(stable across seeds)*
- `acc_retro_32` = **0.0422**  *(stable across seeds)*
- `acc_retro_48` = **0.0328**  *(stable across seeds)*
- `acc_retro_64` = **0.0484**  *(stable across seeds)*
- `gap_24` = **-0.0531**  *(stable across seeds)*
- `gap_32` = **-0.1234**  *(stable across seeds)*
- `gap_48` = **-0.0094**  *(stable across seeds)*
- `gap_64` = **0.0188**  *(stable across seeds)*
- `pearson_r` = **0.7726**  *(stable across seeds)*

**Notes:** Pearson r=0.773 — scaling relationship inconclusive.

---

### Category 11 — Read Bottleneck Interventions (Phase 2)
*1 supported / 2 refuted / 0 inconclusive / 0 error*

#### exp_11_1  ✗ REFUTED
**Hypothesis:** A two-step query former (linear -> cross-attention over last 4 hidden states -> linear) shifts the bottleneck away from read accuracy (identified in exp_7_7).

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 22s

**Metrics (mean ± std across seeds):**

- `normal_acc_A` = **0.2109**  *(stable across seeds)*
- `normal_acc_B` = **0.1656**  *(stable across seeds)*
- `normal_acc_C` = **0.2000**  *(stable across seeds)*
- `oracle_acc_A_final` = **0.2922**  *(stable across seeds)*
- `oracle_acc_A_step0.2` = **0.1203**  *(stable across seeds)*
- `oracle_acc_A_step0.5` = **0.2516**  *(stable across seeds)*
- `oracle_acc_A_step1.0` = **0.2922**  *(stable across seeds)*
- `oracle_acc_B_final` = **0.1750**  *(stable across seeds)*
- `oracle_acc_B_minus_A` = **-0.1172**  *(stable across seeds)*
- `oracle_acc_B_step0.2` = **0.1062**  *(stable across seeds)*
- `oracle_acc_B_step0.5` = **0.1203**  *(stable across seeds)*
- `oracle_acc_B_step1.0` = **0.1750**  *(stable across seeds)*
- `oracle_acc_C_final` = **0.1953**  *(stable across seeds)*
- `oracle_acc_C_step0.2` = **0.1328**  *(stable across seeds)*
- `oracle_acc_C_step0.5` = **0.1625**  *(stable across seeds)*
- `oracle_acc_C_step1.0` = **0.1953**  *(stable across seeds)*

**Notes:** oracle_read_acc_B=0.175 not > oracle_read_acc_A=0.292 by 0.02.

---
#### exp_11_2  ✗ REFUTED
**Hypothesis:** Read-before-write duplicate suppression (skip write if cosine similarity to any existing memory slot > 0.8) improves retrieval F1 by >3% without reducing recall by >5%.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 52s

**Metrics (mean ± std across seeds):**

- `f1_A` = **0.3432**  *(stable across seeds)*
- `f1_B` = **0.3088**  *(stable across seeds)*
- `f1_gain` = **-0.0343**  *(stable across seeds)*
- `precision_A` = **0.2256**  *(stable across seeds)*
- `precision_B` = **0.1856**  *(stable across seeds)*
- `recall_A` = **0.7163**  *(stable across seeds)*
- `recall_B` = **0.9187**  *(stable across seeds)*
- `recall_drop` = **-0.2025**  *(stable across seeds)*
- `write_rate_A` = **0.3810**  *(stable across seeds)*
- `write_rate_B` = **0.3714**  *(stable across seeds)*

**Notes:** F1_B=0.309 not > F1_A=0.343 by 0.01.

---
#### exp_11_3  ✓ SUPPORTED
**Hypothesis:** The optimal read suppression threshold T varies systematically by task type — factual QA, pattern matching, and sequence completion each have different optimal thresholds (differ by >0.15).

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 55s

**Metrics (mean ± std across seeds):**

- `acc_copy_T0.2` = **0.0609**  *(stable across seeds)*
- `acc_copy_T0.4` = **0.0563**  *(stable across seeds)*
- `acc_copy_T0.6` = **0.0266**  *(stable across seeds)*
- `acc_copy_T0.8` = **0.0266**  *(stable across seeds)*
- `acc_copy_T0.95` = **0.0594**  *(stable across seeds)*
- `acc_factual_T0.2` = **0.2313**  *(stable across seeds)*
- `acc_factual_T0.4` = **0.2016**  *(stable across seeds)*
- `acc_factual_T0.6` = **0.1437**  *(stable across seeds)*
- `acc_factual_T0.8` = **0.1375**  *(stable across seeds)*
- `acc_factual_T0.95` = **0.1609**  *(stable across seeds)*
- `acc_pattern_T0.2` = **0.3781**  *(stable across seeds)*
- `acc_pattern_T0.4` = **0.7781**  *(stable across seeds)*
- `acc_pattern_T0.6` = **0.6109**  *(stable across seeds)*
- `acc_pattern_T0.8` = **0.5344**  *(stable across seeds)*
- `acc_pattern_T0.95` = **0.7969**  *(stable across seeds)*
- `max_pairwise_optimal_T_diff` = **0.7500**  *(stable across seeds)*
- `optimal_T_copy` = **0.2000**  *(stable across seeds)*
- `optimal_T_factual` = **0.2000**  *(stable across seeds)*
- `optimal_T_pattern` = **0.9500**  *(stable across seeds)*
- `supp_copy_T0.2` = **0.0000**  *(stable across seeds)*
- `supp_copy_T0.4` = **0.0000**  *(stable across seeds)*
- `supp_copy_T0.6` = **0.0000**  *(stable across seeds)*
- `supp_copy_T0.8` = **0.0000**  *(stable across seeds)*
- `supp_copy_T0.95` = **0.0000**  *(stable across seeds)*
- `supp_factual_T0.2` = **0.0000**  *(stable across seeds)*
- `supp_factual_T0.4` = **0.0000**  *(stable across seeds)*
- `supp_factual_T0.6` = **0.0000**  *(stable across seeds)*
- `supp_factual_T0.8` = **0.0000**  *(stable across seeds)*
- `supp_factual_T0.95` = **0.0000**  *(stable across seeds)*
- `supp_pattern_T0.2` = **0.0000**  *(stable across seeds)*
- `supp_pattern_T0.4` = **0.0000**  *(stable across seeds)*
- `supp_pattern_T0.6` = **0.0000**  *(stable across seeds)*
- `supp_pattern_T0.8` = **0.0000**  *(stable across seeds)*
- `supp_pattern_T0.95` = **0.0000**  *(stable across seeds)*

**Notes:** Max pairwise T difference=0.750>0.15. Optimal T: factual=0.2, pattern=0.95, copy=0.2.

---

### Category 12 — Compression Hard Regimes (Phase 2)
*0 supported / 1 refuted / 1 inconclusive / 0 error*

#### exp_12_1  ✗ REFUTED
**Hypothesis:** At 64x compression with 100-way gallery discrimination, retrieval-objective compressor achieves >=15% higher Acc@1 than reconstruction-objective compressor.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 15s

**Metrics (mean ± std across seeds):**

- `acc1_A` = **1.0000**  *(stable across seeds)*
- `acc1_B` = **1.0000**  *(stable across seeds)*
- `recon_cosim_A` = **0.1205** ± 0.0046  *(runs: 0.124, 0.122, 0.115)*
- `recon_cosim_B` = **0.0044** ± 0.0009  *(runs: 0.004, 0.004, 0.005)*
- `retrieval_gap` = **0.0000**  *(stable across seeds)*

**Notes:** Both models >80% Acc@1 or gap <5%.

---
#### exp_12_2  ~ INCONCLUSIVE
**Hypothesis:** The 2x-8x compression training failure is gradient starvation from wide bottleneck, not structural — LR warmup restores training.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 3s

**Metrics (mean ± std across seeds):**

- `a_fails_at_2x_4x` = **0.0000**  *(stable across seeds)*
- `b_still_bad_count` = **0.0000**  *(stable across seeds)*
- `ratio16_schedA_cosim_100` = **0.2010** ± 0.0017  *(runs: 0.201, 0.203, 0.200)*
- `ratio16_schedA_cosim_final` = **0.2235** ± 0.0017  *(runs: 0.223, 0.226, 0.222)*
- `ratio16_schedB_cosim_100` = **0.0170** ± 0.0062  *(runs: 0.011, 0.024, 0.016)*
- `ratio16_schedB_cosim_final` = **0.2033** ± 0.0055  *(runs: 0.209, 0.198, 0.203)*
- `ratio16_schedC_cosim_100` = **0.0033** ± 0.0016  *(runs: 0.005, 0.003, 0.002)*
- `ratio16_schedC_cosim_final` = **0.0229** ± 0.0120  *(runs: 0.017, 0.015, 0.037)*
- `ratio2_schedA_cosim_100` = **0.3955** ± 0.0100  *(runs: 0.391, 0.407, 0.389)*
- `ratio2_schedA_cosim_final` = **0.5940** ± 0.0126  *(runs: 0.599, 0.603, 0.580)*
- `ratio2_schedB_cosim_100` = **0.0267** ± 0.0042  *(runs: 0.023, 0.031, 0.026)*
- `ratio2_schedB_cosim_final` = **0.3891** ± 0.0059  *(runs: 0.388, 0.384, 0.396)*
- `ratio2_schedC_cosim_100` = **0.0054** ± 0.0153  *(runs: 0.011, 0.017, -0.012)*
- `ratio2_schedC_cosim_final` = **0.0193** ± 0.0148  *(runs: 0.036, 0.011, 0.010)*
- `ratio4_schedA_cosim_100` = **0.3354** ± 0.0041  *(runs: 0.338, 0.338, 0.331)*
- `ratio4_schedA_cosim_final` = **0.4580** ± 0.0050  *(runs: 0.463, 0.454, 0.457)*
- `ratio4_schedB_cosim_100` = **0.0304** ± 0.0047  *(runs: 0.027, 0.028, 0.036)*
- `ratio4_schedB_cosim_final` = **0.3181** ± 0.0128  *(runs: 0.304, 0.329, 0.322)*
- `ratio4_schedC_cosim_100` = **0.0114** ± 0.0135  *(runs: 0.011, 0.025, -0.002)*
- `ratio4_schedC_cosim_final` = **0.0285** ± 0.0128  *(runs: 0.017, 0.026, 0.042)*
- `ratio8_schedA_cosim_100` = **0.2569** ± 0.0049  *(runs: 0.257, 0.262, 0.252)*
- `ratio8_schedA_cosim_final` = **0.3317** ± 0.0028  *(runs: 0.332, 0.334, 0.329)*
- `ratio8_schedB_cosim_100` = **0.0221** ± 0.0062  *(runs: 0.017, 0.020, 0.029)*
- `ratio8_schedB_cosim_final` = **0.2627** ± 0.0078  *(runs: 0.272, 0.258, 0.259)*
- `ratio8_schedC_cosim_100` = **0.0096** ± 0.0088  *(runs: 0.003, 0.020, 0.006)*
- `ratio8_schedC_cosim_final` = **0.0197** ± 0.0091  *(runs: 0.010, 0.028, 0.022)*
- `warmup_restores_at_2x_4x` = **0.0000**  *(stable across seeds)*

**Notes:** A does not fail at 2x-4x; hypothesis conditions not triggered.

---

### Category 13 — Compositional Retrieval at Scale (Phase 2)
*2 supported / 0 refuted / 0 inconclusive / 0 error*

#### exp_13_1  ✓ SUPPORTED
**Hypothesis:** The two-hop retrieval regularization effect (exp_4_9) persists at 64-entity KB with 40% near-duplicate interference.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 1s

**Metrics (mean ± std across seeds):**

- `interference_gap` = **0.1200**  *(stable across seeds)*
- `interference_sh_acc` = **0.0800**  *(stable across seeds)*
- `interference_th_acc` = **0.2000**  *(stable across seeds)*
- `single_hop_acc` = **0.0625**  *(stable across seeds)*
- `two_hop_acc` = **0.1250**  *(stable across seeds)*
- `two_hop_vs_single_gap` = **0.0625**  *(stable across seeds)*

**Notes:** Two-hop acc within 5% of single-hop on interference subset (gap=0.120).

---
#### exp_13_2  ✓ SUPPORTED
**Hypothesis:** Three-hop compositional retrieval retains >50% of two-hop accuracy (degradation_ratio > 0.50) at hidden_dim=64.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 1s

**Metrics (mean ± std across seeds):**

- `acc_single` = **0.0416** ± 0.0181  *(runs: 0.062, 0.031, 0.031)*
- `acc_three` = **0.1146** ± 0.0180  *(runs: 0.125, 0.094, 0.125)*
- `acc_two` = **0.0312**  *(stable across seeds)*
- `degradation_ratio` = **3.6667** ± 0.5774  *(runs: 4.000, 3.000, 4.000)*

**Notes:** Three-hop retains 4.00 of two-hop accuracy (>0.50).

---

### Category 14 — System Integration (Phase 2)
*0 supported / 3 refuted / 0 inconclusive / 0 error*

#### exp_14_1  ✗ REFUTED
**Hypothesis:** Combining retroactive write and read confidence suppression yields super-additive accuracy gains.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 22s

**Metrics (mean ± std across seeds):**

- `acc_A` = **0.2109**  *(stable across seeds)*
- `acc_B` = **0.2047**  *(stable across seeds)*
- `acc_C` = **0.1359**  *(stable across seeds)*
- `acc_D` = **0.0734**  *(stable across seeds)*
- `gap_B` = **-0.0063**  *(stable across seeds)*
- `gap_C` = **-0.0750**  *(stable across seeds)*
- `gap_D` = **-0.1375**  *(stable across seeds)*
- `super_additive` = [False, False, False]

**Notes:** gap_D=-0.138 < max(gap_B,gap_C)+0.005=-0.001.

---
#### exp_14_2  ✗ REFUTED
**Hypothesis:** Write-first curriculum (train write gate before enabling reads) outperforms joint training from step 0.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 7s

**Metrics (mean ± std across seeds):**

- `acc_A` = **0.2094**  *(stable across seeds)*
- `acc_B` = **0.1812**  *(stable across seeds)*
- `acc_diff_B_minus_A` = **-0.0281**  *(stable across seeds)*
- `wq_diff_B_minus_A` = **0.0000**  *(stable across seeds)*
- `write_quality_at_1000_A` = **1.0000**  *(stable across seeds)*
- `write_quality_at_1000_B` = **1.0000**  *(stable across seeds)*

**Notes:** Joint training A >= curriculum B - 0.01: acc_A=0.209, acc_B=0.181.

---
#### exp_14_3  ✗ REFUTED
**Hypothesis:** Cosine-annealed Gumbel temperature (1.0->0.1) produces higher final accuracy than constant temperature 0.5 by >2%.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 11s

**Metrics (mean ± std across seeds):**

- `acc_A` = **0.1906**  *(stable across seeds)*
- `acc_B` = **0.1750**  *(stable across seeds)*
- `acc_C` = **0.1531**  *(stable across seeds)*
- `diff_C_vs_A` = **-0.0375**  *(stable across seeds)*

**Notes:** Constant A >= cosine C - 0.01: acc_A=0.191, acc_C=0.153.

---

### Category 15 — Delta Rule / Associative Matrix Writes (Phase 3)
*1 supported / 2 refuted / 1 inconclusive / 0 error*

#### exp_15_1  ✗ REFUTED ⚠ inconsistent across seeds ['REFUTED', 'REFUTED', 'INCONCLUSIVE']
**Hypothesis:** A delta rule associative matrix write outperforms standard slot write by >5% due to built-in interference correction.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 11s

**Metrics (mean ± std across seeds):**

- `acc_delta` = **0.1056** ± 0.0174  *(runs: 0.109, 0.087, 0.121)*
- `acc_gap` = **0.0258** ± 0.0427  *(runs: 0.004, -0.001, 0.075)*
- `acc_slot` = **0.0798** ± 0.0303  *(runs: 0.105, 0.088, 0.046)*
- `interference_delta` = **0.9525** ± 0.0299  *(runs: 0.967, 0.918, 0.973)*
- `interference_gap` = **-0.4549** ± 0.0925  *(runs: -0.449, -0.550, -0.365)*
- `interference_slot` = **0.4976** ± 0.1208  *(runs: 0.518, 0.368, 0.607)*

**Notes:** Slot memory acc=0.105 >= delta acc=0.109 - 0.02.

---
#### exp_15_2  ✗ REFUTED ⚠ inconsistent across seeds ['REFUTED', 'REFUTED', 'INCONCLUSIVE']
**Hypothesis:** The correction term in the delta rule is essential — Hebbian M += v*k^T degrades by >10% on key-interference tasks.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 29s

**Metrics (mean ± std across seeds):**

- `delta_acc_clean` = **0.1096** ± 0.0091  *(runs: 0.099, 0.113, 0.117)*
- `delta_acc_interfered` = **0.1162** ± 0.0277  *(runs: 0.106, 0.095, 0.147)*
- `delta_vs_hebbian_gap_on_interference` = **0.0154** ± 0.0359  *(runs: 0.010, -0.018, 0.054)*
- `hebbian_acc_clean` = **0.1017** ± 0.0164  *(runs: 0.083, 0.114, 0.107)*
- `hebbian_acc_interfered` = **0.1009** ± 0.0102  *(runs: 0.096, 0.113, 0.094)*
- `norm_hebbian_acc_clean` = **0.1058** ± 0.0058  *(runs: 0.107, 0.111, 0.099)*
- `norm_hebbian_acc_interfered` = **0.1059** ± 0.0083  *(runs: 0.110, 0.111, 0.096)*

**Notes:** Hebbian acc_int=0.096 within 0.03 of delta acc_int=0.106.

---
#### exp_15_3  ✓ SUPPORTED
**Hypothesis:** Energy-gated delta rule (write only when delta_E < 0) achieves >90% accuracy of continuous write at <70% write rate.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 26s

**Metrics (mean ± std across seeds):**

- `acc_A_continuous` = **0.1310** ± 0.0293  *(runs: 0.138, 0.099, 0.156)*
- `acc_B_energy_gated` = **0.1477** ± 0.0219  *(runs: 0.127, 0.171, 0.146)*
- `acc_C_learned_gate` = **0.1727** ± 0.0185  *(runs: 0.154, 0.174, 0.191)*
- `acc_ratio_B` = **1.1928** ± 0.4634  *(runs: 0.919, 1.728, 0.932)*
- `acc_ratio_C` = **1.3642** ± 0.3465  *(runs: 1.113, 1.760, 1.220)*
- `write_rate_A` = **1.0000**  *(stable across seeds)*
- `write_rate_B` = **0.5159** ± 0.0029  *(runs: 0.519, 0.515, 0.514)*
- `write_rate_C` = **0.2650** ± 0.3384  *(runs: 0.070, 0.656, 0.070)*

**Notes:** Energy gate: acc_ratio=0.919>0.90, write_rate=0.519<0.70. Hypothesis confirmed.

---
#### exp_15_4  ~ INCONCLUSIVE ⚠ inconsistent across seeds ['REFUTED', 'INCONCLUSIVE', 'INCONCLUSIVE']
**Hypothesis:** Delta rule outperforms Larimar outer-product write on overwrite tasks (same key, updated value).

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 13s

**Metrics (mean ± std across seeds):**

- `delta_acc_normal` = **0.1663** ± 0.0159  *(runs: 0.166, 0.151, 0.182)*
- `delta_acc_overall` = **0.1700** ± 0.0196  *(runs: 0.168, 0.152, 0.191)*
- `delta_acc_update` = **0.1735** ± 0.0234  *(runs: 0.169, 0.153, 0.199)*
- `larimar_acc_normal` = **0.1319** ± 0.0432  *(runs: 0.182, 0.106, 0.107)*
- `larimar_acc_overall` = **0.1338** ± 0.0400  *(runs: 0.180, 0.106, 0.116)*
- `larimar_acc_update` = **0.1357** ± 0.0373  *(runs: 0.177, 0.106, 0.124)*
- `update_gap_delta_minus_larimar` = **0.0379** ± 0.0424  *(runs: -0.008, 0.047, 0.075)*

**Notes:** Larimar acc_update=0.177 >= delta acc_update=0.169 - 0.02.

---

### Category 16 — Online Gradient Descent Memory / Titans-Style (Phase 3)
*1 supported / 1 refuted / 1 inconclusive / 0 error*

#### exp_16_1  ✗ REFUTED
**Hypothesis:** Parametric MLP memory (1 gradient step per token) outperforms fixed-slot memory at matched parameter count.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 6s

**Metrics (mean ± std across seeds):**

- `acc_gap` = **-0.0117** ± 0.0113  *(runs: 0.000, -0.013, -0.022)*
- `acc_parametric` = **0.0342** ± 0.0128  *(runs: 0.045, 0.037, 0.020)*
- `acc_slot` = **0.0458** ± 0.0038  *(runs: 0.045, 0.050, 0.043)*
- `param_count_mlp` = **552.0000**  *(stable across seeds)*
- `param_count_slot_storage` = **256.0000**  *(stable across seeds)*

**Notes:** Slot acc=0.045 >= parametric acc=0.045 - 0.02.

---
#### exp_16_2  ~ INCONCLUSIVE
**Hypothesis:** Skipping MLP memory updates for low-surprise tokens achieves same accuracy with >40% fewer update steps.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 18s

**Metrics (mean ± std across seeds):**

- `baseline_acc` = **0.0417** ± 0.0038  *(runs: 0.043, 0.037, 0.045)*
- `baseline_write_rate` = **1.0000**  *(stable across seeds)*
- `pareto_found` = [False, False, False]
- `per_threshold.0.0.acc` = **0.0417** ± 0.0038  *(runs: 0.043, 0.037, 0.045)*
- `per_threshold.0.0.write_rate` = **1.0000**  *(stable across seeds)*
- `per_threshold.0.3.acc` = **0.0267** ± 0.0038  *(runs: 0.022, 0.028, 0.030)*
- `per_threshold.0.3.write_rate` = **1.0000**  *(stable across seeds)*
- `per_threshold.0.6.acc` = **0.0325** ± 0.0198  *(runs: 0.018, 0.025, 0.055)*
- `per_threshold.0.6.write_rate` = **1.0000**  *(stable across seeds)*
- `per_threshold.1.0.acc` = **0.0308** ± 0.0095  *(runs: 0.020, 0.035, 0.037)*
- `per_threshold.1.0.write_rate` = **1.0000**  *(stable across seeds)*
- `per_threshold.1.5.acc` = **0.0333** ± 0.0063  *(runs: 0.040, 0.028, 0.033)*
- `per_threshold.1.5.write_rate` = **1.0000**  *(stable across seeds)*

**Notes:** Smooth degradation curve; no clear pareto knee found.

---
#### exp_16_3  ✓ SUPPORTED
**Hypothesis:** Parametric memory scales more gracefully with seq_len than slot memory (higher accuracy retention at 4x length).

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 14s

**Metrics (mean ± std across seeds):**

- `acc_parametric.24` = **0.0354** ± 0.0110  *(runs: 0.025, 0.047, 0.034)*
- `acc_parametric.48` = **0.0291** ± 0.0130  *(runs: 0.025, 0.019, 0.044)*
- `acc_slot.24` = **0.0844** ± 0.0109  *(runs: 0.078, 0.078, 0.097)*
- `acc_slot.48` = **0.0323** ± 0.0148  *(runs: 0.044, 0.016, 0.037)*
- `retention_diff` = **0.5076** ± 0.3471  *(runs: 0.441, 0.199, 0.883)*
- `retention_parametric` = **0.8897** ± 0.4462  *(runs: 1.000, 0.399, 1.270)*
- `retention_slot` = **0.3821** ± 0.1800  *(runs: 0.559, 0.200, 0.387)*

**Notes:** Parametric retention=1.000 vs slot retention=0.560; diff=0.440 > 0.15. Scales more gracefully.

---

### Category 17 — Prospective / Query-Conditioned Writing (Phase 3)
*0 supported / 3 refuted / 1 inconclusive / 0 error*

#### exp_17_1  ✗ REFUTED ⚠ inconsistent across seeds ['REFUTED', 'SUPPORTED', 'REFUTED']
**Hypothesis:** A write gate conditioned on predicted future query type outperforms context-only gate by >5% on tasks with 4 different query types.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 11s

**Metrics (mean ± std across seeds):**

- `acc_A` = **0.1420** ± 0.0818  *(runs: 0.222, 0.059, 0.145)*
- `acc_B` = **0.0823** ± 0.0406  *(runs: 0.057, 0.129, 0.060)*
- `gap` = **-0.0597** ± 0.1196  *(runs: -0.165, 0.070, -0.084)*
- `query_pred_acc` = **0.7168** ± 0.1323  *(runs: 0.583, 0.720, 0.847)*

**Notes:** Context-only gate matches or beats query-conditioned (gap=-0.165)

---
#### exp_17_2  ✗ REFUTED ⚠ inconsistent across seeds ['REFUTED', 'SUPPORTED', 'REFUTED']
**Hypothesis:** K-token lookahead async write gate outperforms same-time write by >3% at some K in {2, 4, 6}.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 24s

**Metrics (mean ± std across seeds):**

- `acc_K0` = **0.1303** ± 0.0680  *(runs: 0.208, 0.080, 0.103)*
- `acc_K2` = **0.0813** ± 0.0851  *(runs: 0.032, 0.179, 0.032)*
- `acc_K4` = **0.1263** ± 0.0586  *(runs: 0.194, 0.092, 0.093)*
- `acc_K6` = **0.0376** ± 0.0088  *(runs: 0.032, 0.034, 0.048)*
- `best_gap` = **0.0252** ± 0.0641  *(runs: -0.014, 0.099, -0.010)*
- `best_k` = **3.3333** ± 1.1547  *(runs: 4.000, 2.000, 4.000)*

**Notes:** All lookahead K within 0.02 of K=0 (best gap=-0.014)

---
#### exp_17_3  ~ INCONCLUSIVE
**Hypothesis:** Prospective and retroactive writing are redundant — their combination yields <1.5x the gain of either alone.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 22s

**Metrics (mean ± std across seeds):**

- `acc_A` = **0.1471** ± 0.0726  *(runs: 0.219, 0.074, 0.149)*
- `acc_B` = **0.1428** ± 0.0529  *(runs: 0.086, 0.191, 0.151)*
- `acc_C` = **0.0719** ± 0.0190  *(runs: 0.092, 0.069, 0.054)*
- `acc_D` = **0.1581** ± 0.0254  *(runs: 0.185, 0.135, 0.154)*
- `gap_B` = **-0.0044** ± 0.1251  *(runs: -0.133, 0.117, 0.002)*
- `gap_C` = **-0.0752** ± 0.0633  *(runs: -0.127, -0.004, -0.095)*
- `gap_D` = **0.0109** ± 0.0477  *(runs: -0.034, 0.061, 0.005)*
- `multiplier` = **-10.2538** ± 20.2328  *(runs: -33.594, 0.522, 2.310)*

**Notes:** gap_B=-0.133 or gap_C=-0.127 too small for valid comparison

---
#### exp_17_4  ✗ REFUTED ⚠ inconsistent across seeds ['REFUTED', 'INCONCLUSIVE', 'REFUTED']
**Hypothesis:** Query-conditioned write gain scales linearly with query predictability (Pearson r > 0.85).

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 38s

**Metrics (mean ± std across seeds):**

- `acc_A_p000` = **0.1480** ± 0.0764  *(runs: 0.221, 0.069, 0.154)*
- `acc_A_p033` = **0.1025** ± 0.0690  *(runs: 0.110, 0.030, 0.168)*
- `acc_A_p066` = **0.0601** ± 0.0431  *(runs: 0.040, 0.110, 0.031)*
- `acc_A_p100` = **0.1468** ± 0.0597  *(runs: 0.082, 0.199, 0.160)*
- `acc_B_p000` = **0.1544** ± 0.0444  *(runs: 0.195, 0.162, 0.107)*
- `acc_B_p033` = **0.1081** ± 0.0793  *(runs: 0.104, 0.189, 0.031)*
- `acc_B_p066` = **0.0462** ± 0.0259  *(runs: 0.031, 0.076, 0.031)*
- `acc_B_p100` = **0.0427** ± 0.0151  *(runs: 0.060, 0.034, 0.034)*
- `gap_p000` = **0.0064** ± 0.0757  *(runs: -0.026, 0.093, -0.048)*
- `gap_p033` = **0.0057** ± 0.1482  *(runs: -0.005, 0.159, -0.137)*
- `gap_p066` = **-0.0139** ± 0.0174  *(runs: -0.008, -0.034, 0.000)*
- `gap_p100` = **-0.1041** ± 0.0741  *(runs: -0.021, -0.165, -0.126)*
- `gap_variance` = **0.0062** ± 0.0081  *(runs: 0.000, 0.015, 0.003)*
- `pearson_r` = **-0.3135** ± 0.5119  *(runs: 0.132, -0.873, -0.200)*

**Notes:** Gap is effectively constant (variance=0.000077)

---

### Category 18 — Tiered Memory Architecture (Phase 3)
*0 supported / 1 refuted / 3 inconclusive / 0 error*

#### exp_18_1  ✗ REFUTED
**Hypothesis:** Two-tier memory (16-slot fast + 64-slot slow with learned demotion) outperforms flat 64-slot memory by >5% on long-context tasks.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 192s

**Metrics (mean ± std across seeds):**

- `acc_A` = **0.0449** ± 0.0117  *(runs: 0.032, 0.047, 0.055)*
- `acc_B` = **0.0408** ± 0.0190  *(runs: 0.029, 0.063, 0.030)*
- `acc_C` = **0.0328** ± 0.0004  *(runs: 0.032, 0.033, 0.033)*
- `gap_B` = **-0.0041** ± 0.0206  *(runs: -0.003, 0.016, -0.025)*
- `gap_C` = **-0.0120** ± 0.0113  *(runs: 0.000, -0.014, -0.022)*
- `slow_coverage_B` = **0.3631** ± 0.0476  *(runs: 0.409, 0.367, 0.314)*
- `slow_coverage_C` = **0.3951** ± 0.0295  *(runs: 0.425, 0.366, 0.394)*

**Notes:** Flat memory matches tiered (gap=-0.003)

---
#### exp_18_2  ~ INCONCLUSIVE
**Hypothesis:** Learned demotion controller discovers frequency-not-recency policy (corr_access > 0.15, corr_recency < -0.10).

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 48s

**Metrics (mean ± std across seeds):**

- `corr_access` = **0.2119** ± 0.0239  *(runs: 0.237, 0.189, 0.210)*
- `corr_content_norm` = **0.2629** ± 0.1190  *(runs: 0.399, 0.211, 0.179)*
- `corr_recency` = **0.2354** ± 0.0684  *(runs: 0.175, 0.309, 0.222)*
- `n_demotion_events` = **3263.6667** ± 807.1557  *(runs: 2488.000, 4099.000, 3204.000)*

**Notes:** corr_access=0.237 > 0.15 but corr_recency=0.175 > -0.10

---
#### exp_18_3  ~ INCONCLUSIVE ⚠ inconsistent across seeds ['INCONCLUSIVE', 'SUPPORTED', 'INCONCLUSIVE']
**Hypothesis:** Tiered memory has a capacity crossover point — flat is better below it, tiered above it.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 390s

**Metrics (mean ± std across seeds):**

- `acc_flat_128` = **0.0471** ± 0.0244  *(runs: 0.032, 0.075, 0.035)*
- `acc_flat_16` = **0.0549** ± 0.0029  *(runs: 0.052, 0.057, 0.056)*
- `acc_flat_32` = **0.0479** ± 0.0154  *(runs: 0.030, 0.060, 0.053)*
- `acc_flat_64` = **0.0550** ± 0.0228  *(runs: 0.077, 0.032, 0.056)*
- `acc_flat_8` = **0.0396** ± 0.0133  *(runs: 0.033, 0.055, 0.030)*
- `acc_tiered_128` = **0.0399** ± 0.0123  *(runs: 0.054, 0.033, 0.033)*
- `acc_tiered_16` = **0.0320** ± 0.0022  *(runs: 0.030, 0.035, 0.031)*
- `acc_tiered_32` = **0.0306** ± 0.0023  *(runs: 0.030, 0.033, 0.029)*
- `acc_tiered_64` = **0.0328** ± 0.0015  *(runs: 0.034, 0.033, 0.031)*
- `acc_tiered_8` = **0.0543** ± 0.0038  *(runs: 0.055, 0.050, 0.058)*
- `crossover_capacity` = **26.6667** ± 32.3316  *(runs: 8.000, 64.000, 8.000)*

**Notes:** Crossover at 8 outside expected range 16-64

---
#### exp_18_4  ~ INCONCLUSIVE
**Hypothesis:** Simultaneous cross-tier retrieval outperforms cascaded sequential retrieval by >3%.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 200s

**Metrics (mean ± std across seeds):**

- `acc_A` = **0.0322** ± 0.0006  *(runs: 0.032, 0.033, 0.032)*
- `acc_B` = **0.0652** ± 0.0102  *(runs: 0.058, 0.061, 0.077)*
- `acc_C` = **0.0717** ± 0.0226  *(runs: 0.098, 0.056, 0.062)*
- `gap_A_vs_B` = **-0.0330** ± 0.0105  *(runs: -0.026, -0.028, -0.045)*
- `gap_A_vs_C` = **-0.0395** ± 0.0230  *(runs: -0.066, -0.023, -0.030)*
- `gap_A_vs_best_seq` = **-0.0462** ± 0.0190  *(runs: -0.066, -0.028, -0.045)*
- `slow_access_rate_B` = **0.3806** ± 0.0565  *(runs: 0.339, 0.358, 0.445)*

**Notes:** Simultaneous best but gap=-0.066 < 0.03

---

### Category 19 — Sparse Hopfield Addressing (Phase 3)
*0 supported / 3 refuted / 0 inconclusive / 0 error*

#### exp_19_1  ✗ REFUTED
**Hypothesis:** Sparse Hopfield retrieval (sparsemax top-k=2) outperforms standard softmax attention by >5% precision@1 on 40% interference tasks.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 15s

**Metrics (mean ± std across seeds):**

- `acc_A` = **0.0228**  *(stable across seeds)*
- `acc_B` = **0.0175**  *(stable across seeds)*
- `acc_C` = **0.0316**  *(stable across seeds)*
- `acc_gap_B_minus_A` = **-0.0053**  *(stable across seeds)*
- `precision_A` = **0.0000**  *(stable across seeds)*
- `precision_B` = **0.0000**  *(stable across seeds)*
- `precision_C` = **0.0316**  *(stable across seeds)*
- `precision_gap_B_minus_A` = **0.0000**  *(stable across seeds)*

**Notes:** Soft attention matches or beats sparse on interference tasks.

---
#### exp_19_2  ✗ REFUTED
**Hypothesis:** Hopfield energy write criterion (write only if ΔE < 0) produces <35% write rate with >3% accuracy improvement.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 46s

**Metrics (mean ± std across seeds):**

- `acc_A` = **0.1291**  *(stable across seeds)*
- `acc_B` = **0.0297**  *(stable across seeds)*
- `acc_C` = **0.1228**  *(stable across seeds)*
- `acc_gap_B_minus_A` = **-0.0994**  *(stable across seeds)*
- `write_rate_B` = **0.8214**  *(stable across seeds)*
- `write_rate_C` = **0.1367**  *(stable across seeds)*

**Notes:** Energy writes too frequent (0.8214) or acc dropped vs learned gate.

---
#### exp_19_3  ✗ REFUTED
**Hypothesis:** Sparse Hopfield sustains accuracy 2+ patterns longer than dense before capacity cliff.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 65s

**Metrics (mean ± std across seeds):**

- `acc_soft_12` = **0.0200**  *(stable across seeds)*
- `acc_soft_16` = **0.0206**  *(stable across seeds)*
- `acc_soft_20` = **0.0147**  *(stable across seeds)*
- `acc_soft_24` = **0.0144**  *(stable across seeds)*
- `acc_soft_4` = **0.1212**  *(stable across seeds)*
- `acc_soft_8` = **0.0256**  *(stable across seeds)*
- `acc_sparse_12` = **0.0244**  *(stable across seeds)*
- `acc_sparse_16` = **0.0194**  *(stable across seeds)*
- `acc_sparse_20` = **0.0147**  *(stable across seeds)*
- `acc_sparse_24` = **0.0144**  *(stable across seeds)*
- `acc_sparse_4` = **0.1253**  *(stable across seeds)*
- `acc_sparse_8` = **0.0344**  *(stable across seeds)*
- `capacity_cliff_soft` = **28.0000**  *(stable across seeds)*
- `capacity_cliff_sparse` = **28.0000**  *(stable across seeds)*
- `cliff_diff_sparse_minus_soft` = **0.0000**  *(stable across seeds)*

**Notes:** Capacity cliff differs by only 0 (< 1).

---

### Category 20 — Three-Gate Coordinated Controller (Phase 3)
*0 supported / 2 refuted / 2 inconclusive / 0 error*

#### exp_20_1  ~ INCONCLUSIVE
**Hypothesis:** L1 auxiliary loss on write gate (targeting ~15% activity) improves accuracy and avoids degenerate modes.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 11s

**Metrics (mean ± std across seeds):**

- `acc_A` = **0.1672**  *(stable across seeds)*
- `acc_B` = **0.1891**  *(stable across seeds)*
- `acc_C` = **0.1806**  *(stable across seeds)*
- `acc_gap_B_minus_A` = **0.0219**  *(stable across seeds)*
- `is_collapsed_A` = [False, False, False]
- `is_collapsed_B` = [False, False, False]
- `is_collapsed_C` = [True, True, True]
- `write_rate_A` = **0.4111**  *(stable across seeds)*
- `write_rate_B` = **0.0357**  *(stable across seeds)*
- `write_rate_C` = **0.0159**  *(stable across seeds)*

**Notes:** Auxiliary adjusts write rate but accuracy gain (0.0219) below threshold or write_rate out of range.

---
#### exp_20_2  ✗ REFUTED
**Hypothesis:** Explicit read accuracy auxiliary loss reduces the read bottleneck more effectively than implicit task loss.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 11s

**Metrics (mean ± std across seeds):**

- `acc_A` = **0.1672**  *(stable across seeds)*
- `acc_B` = **0.1691**  *(stable across seeds)*
- `acc_C` = **0.1762**  *(stable across seeds)*
- `acc_gap_B_minus_A` = **0.0019**  *(stable across seeds)*
- `oracle_gap_B_minus_A` = **0.0050**  *(stable across seeds)*
- `oracle_read_acc_A` = **0.1578**  *(stable across seeds)*
- `oracle_read_acc_B` = **0.1628**  *(stable across seeds)*
- `oracle_read_acc_C` = **0.1744**  *(stable across seeds)*

**Notes:** Task acc unchanged (gap=0.0019 <= 0.005).

---
#### exp_20_3  ✗ REFUTED
**Hypothesis:** Three-gate controller with all auxiliary losses combined outperforms any single-auxiliary system.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 22s

**Metrics (mean ± std across seeds):**

- `acc_A` = **0.2353**  *(stable across seeds)*
- `acc_B` = **0.2506**  *(stable across seeds)*
- `acc_C` = **0.2425**  *(stable across seeds)*
- `acc_D` = **0.2334**  *(stable across seeds)*
- `acc_E` = **0.2431**  *(stable across seeds)*
- `best_single_aux` = **0.2506**  *(stable across seeds)*
- `gap_E_minus_A` = **0.0078**  *(stable across seeds)*
- `gap_E_minus_best_single` = **-0.0075**  *(stable across seeds)*

**Notes:** No-auxiliary baseline (acc_A=0.2353) nearly matches full system (acc_E=0.2431).

---
#### exp_20_4  ~ INCONCLUSIVE
**Hypothesis:** Optimal write sparsity auxiliary weight is in [0.01, 0.1] — outside this range gate collapses or task signal drowns.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 18s

**Metrics (mean ± std across seeds):**

- `acc_lam_0_0` = **0.1663**  *(stable across seeds)*
- `acc_lam_0_01` = **0.2018**  *(stable across seeds)*
- `acc_lam_0_1` = **0.1985**  *(stable across seeds)*
- `acc_lam_1_0` = **0.2062**  *(stable across seeds)*
- `below_left_of_range` = [False, False, False]
- `below_right_of_range` = [False, False, False]
- `collapsed_lam_0_0` = [False, False, False]
- `collapsed_lam_0_01` = [False, False, False]
- `collapsed_lam_0_1` = [False, False, False]
- `collapsed_lam_1_0` = [True, True, True]
- `monotone_decay` = [False, False, False]
- `peak_acc` = **0.2018**  *(stable across seeds)*
- `peak_in_range` = [True, True, True]
- `peak_lambda` = **0.0100**  *(stable across seeds)*
- `wr_lam_0_0` = **0.4075**  *(stable across seeds)*
- `wr_lam_0_01` = **0.0350**  *(stable across seeds)*
- `wr_lam_0_1` = **0.0247**  *(stable across seeds)*
- `wr_lam_1_0` = **0.0078**  *(stable across seeds)*

**Notes:** Peak at lambda=0.01 (in_range=True); boundary conditions not fully met.

---

### Category 21 — Feedforward Controller + Hindsight Distillation (Phase 3)
*0 supported / 4 refuted / 0 inconclusive / 0 error*

#### exp_21_1  ✗ REFUTED
**Hypothesis:** A feedforward-only memory controller achieves higher external memory utilization than an LSTM controller at equal parameter count.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 8s

**Metrics (mean ± std across seeds):**

- `acc_A` = **0.0313** ± 0.0040  *(runs: 0.027, 0.035, 0.032)*
- `acc_B` = **0.0338** ± 0.0041  *(runs: 0.038, 0.031, 0.033)*
- `acc_gap` = **0.0026** ± 0.0081  *(runs: 0.012, -0.004, 0.000)*
- `params_A` = **10545.0000**  *(stable across seeds)*
- `params_B` = **9473.0000**  *(stable across seeds)*
- `util_A` = **0.0840** ± 0.0622  *(runs: 0.154, 0.036, 0.062)*
- `util_B` = **0.0209** ± 0.0044  *(runs: 0.018, 0.026, 0.019)*
- `util_gap` = **-0.0631** ± 0.0658  *(runs: -0.137, -0.010, -0.042)*

**Notes:** LSTM: acc=0.0269 util=0.1542. FF: acc=0.0384 util=0.0175. acc_gap=0.0116 util_gap=-0.1368.

---
#### exp_21_2  ✗ REFUTED
**Hypothesis:** Hindsight oracle labels (which writes were causally relevant) provide a stronger training signal than task loss alone (+3% accuracy).

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 13s

**Metrics (mean ± std across seeds):**

- `acc_A` = **0.0321** ± 0.0040  *(runs: 0.028, 0.034, 0.034)*
- `acc_B` = **0.0316** ± 0.0013  *(runs: 0.033, 0.031, 0.031)*
- `acc_gap` = **-0.0004** ± 0.0052  *(runs: 0.006, -0.003, -0.004)*
- `gate_quality_r` = **0.0781** ± 0.1465  *(runs: 0.077, -0.068, 0.225)*

**Notes:** Task-only: acc=0.0275. Oracle-augmented: acc=0.0331. gap=0.0056, gate_quality_r=0.0773.

---
#### exp_21_3  ✗ REFUTED
**Hypothesis:** Write gate distilled from oracle labels (trained primarily on oracle supervision) achieves higher accuracy than end-to-end learned gate.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 24s

**Metrics (mean ± std across seeds):**

- `acc_A` = **0.0321** ± 0.0040  *(runs: 0.028, 0.034, 0.034)*
- `acc_B` = **0.0294** ± 0.0053  *(runs: 0.024, 0.035, 0.029)*
- `acc_C` = **0.0367** ± 0.0058  *(runs: 0.041, 0.039, 0.030)*
- `acc_gap_BA` = **-0.0027** ± 0.0031  *(runs: -0.003, 0.001, -0.006)*
- `gate_quality_A` = **0.0019** ± 0.0181  *(runs: -0.011, -0.006, 0.023)*
- `gate_quality_B` = **0.0270** ± 0.0294  *(runs: 0.059, 0.019, 0.002)*
- `gate_quality_C` = **0.1177** ± 0.0567  *(runs: 0.179, 0.108, 0.067)*
- `quality_gap_BA` = **0.0251** ± 0.0453  *(runs: 0.070, 0.025, -0.020)*

**Notes:** A(e2e): acc=0.0275 gq=-0.0108. B(distilled): acc=0.0244 gq=0.0595. C(mixed): acc=0.0406 gq=0.1786. acc_gap(B-A)=-0.0031 quality_gap=0.0703.

---
#### exp_21_4  ✗ REFUTED
**Hypothesis:** Feedforward controller + hindsight distillation is the strongest write policy (outperforms either alone and LSTM baseline).

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 43s

**Metrics (mean ± std across seeds):**

- `acc_A` = **0.0348** ± 0.0051  *(runs: 0.033, 0.041, 0.031)*
- `acc_B` = **0.0338** ± 0.0050  *(runs: 0.039, 0.032, 0.030)*
- `acc_C` = **0.0325** ± 0.0017  *(runs: 0.032, 0.031, 0.034)*
- `acc_D` = **0.0315** ± 0.0069  *(runs: 0.024, 0.038, 0.032)*
- `acc_E` = **0.0298** ± 0.0019  *(runs: 0.029, 0.028, 0.032)*
- `combination_is_best` = [False, False, False]
- `gap_over_best_other` = **-0.0083** ± 0.0052  *(runs: -0.010, -0.013, -0.003)*
- `gap_over_lstm_baseline` = **-0.0050** ± 0.0068  *(runs: -0.003, -0.013, 0.001)*

**Notes:** A(LSTM+task)=0.0325  B(FF+task)=0.0394  C(FF+oracle)=0.0319  D(LSTM+oracle)=0.0244  E(FF+hindsight)=0.0294. gap_over_best=-0.0100  combination_is_best=False.

---

## Cross-Cutting Observations

**All SUPPORTED experiments:** exp_1_5, exp_2_6, exp_3_1, exp_3_5, exp_3_6, exp_4_1, exp_4_4, exp_4_7, exp_4_9, exp_5_2, exp_5_6, exp_6_1, exp_6_3, exp_7_1, exp_7_2, exp_7_9, exp_9_2, exp_9_4, exp_9_5, exp_11_3, exp_13_1, exp_13_2, exp_15_3, exp_16_3

**All REFUTED experiments:** exp_1_1, exp_1_2, exp_1_8, exp_2_2, exp_2_4, exp_2_5, exp_2_9, exp_3_2, exp_3_3, exp_3_4, exp_3_7, exp_4_2, exp_5_1, exp_5_4, exp_5_5, exp_5_7, exp_6_7, exp_7_5, exp_7_6, exp_7_7, exp_8_1, exp_8_3, exp_9_1, exp_10_2, exp_11_1, exp_11_2, exp_12_1, exp_14_1, exp_14_2, exp_14_3, exp_15_1, exp_15_2, exp_16_1, exp_17_1, exp_17_2, exp_17_4, exp_18_1, exp_19_1, exp_19_2, exp_19_3, exp_20_2, exp_20_3, exp_21_1, exp_21_2, exp_21_3, exp_21_4

**Inconsistent across seeds (need more investigation):** exp_8_1, exp_8_2, exp_8_4, exp_9_4, exp_9_5, exp_10_1, exp_15_1, exp_15_2, exp_15_4, exp_17_1, exp_17_2, exp_17_4, exp_18_3

**High-variance metrics (std > 0.05 — seed-sensitive, interpret carefully):**

- exp_8_1.pearson_r_entropic (std=0.446)
- exp_8_1.pearson_r_raw (std=0.341)
- exp_8_1.pearson_r_softmax (std=0.488)
- exp_8_2.pearson_r_difficulty_vs_rate (std=0.405)
- exp_8_3.acc_A (std=0.051)
- exp_8_3.write_evict_corr_A (std=0.288)
- exp_8_3.write_evict_corr_B (std=0.062)
- exp_8_3.write_evict_corr_C (std=0.303)
- exp_8_4.acc_A (std=0.110)
- exp_8_4.acc_B (std=0.058)
- exp_8_4.content_corr_A (std=0.525)
- exp_8_4.content_corr_B (std=0.089)
- exp_8_4.content_corr_C (std=0.169)
- exp_8_4.pos_corr_A (std=0.336)
- exp_8_4.pos_corr_B (std=0.079)
- exp_8_4.pos_corr_C (std=0.169)
- exp_9_3.acc_a_after_ewc (std=0.066)
- exp_9_3.acc_a_after_std (std=0.050)
- exp_9_3.acc_a_before (std=0.085)
- exp_9_3.forgetting_ratio (std=0.611)

---
*Report generated by research/aggregate.py*