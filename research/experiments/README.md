# Experiment Catalog

All 44 experiments organized by category. Priority 10 are marked **P**.

## Category 1: What To Write (8 experiments)

| ID | File | Status | Priority |
|----|------|--------|----------|
| 1.1 | cat1_what_to_write/exp_1_1_relevance_signal_baseline.py | NOT BUILT | |
| 1.2 | cat1_what_to_write/exp_1_2_surprise_write_signal.py | NOT BUILT | |
| 1.3 | cat1_what_to_write/exp_1_3_gradient_magnitude_write.py | NOT BUILT | |
| 1.4 | cat1_what_to_write/exp_1_4_contrastive_write_selection.py | NOT BUILT | |
| **1.5** | **cat1_what_to_write/exp_1_5_write_signal_ablation.py** | **READY** | **P1** |
| 1.6 | cat1_what_to_write/exp_1_6_semantic_deduplication.py | NOT BUILT | |
| 1.7 | cat1_what_to_write/exp_1_7_write_frequency_vs_quality.py | NOT BUILT | |
| 1.8 | cat1_what_to_write/exp_1_8_hierarchical_write_decisions.py | NOT BUILT | |

## Category 2: How To Write / Compression (9 experiments)

| ID | File | Status | Priority |
|----|------|--------|----------|
| **2.1** | **cat2_how_to_write/exp_2_1_compression_ratio_curve.py** | **READY** | **P2** |
| 2.2 | cat2_how_to_write/exp_2_2_autoencoder_vs_attention_compression.py | NOT BUILT | |
| 2.3 | cat2_how_to_write/exp_2_3_lossy_vs_lossless.py | NOT BUILT | |
| 2.4 | cat2_how_to_write/exp_2_4_chunk_size_sensitivity.py | NOT BUILT | |
| 2.5 | cat2_how_to_write/exp_2_5_structured_vs_unstructured_compression.py | NOT BUILT | |
| 2.6 | cat2_how_to_write/exp_2_6_compression_generalization.py | NOT BUILT | |
| 2.7 | cat2_how_to_write/exp_2_7_iterative_compression.py | NOT BUILT | |
| 2.8 | cat2_how_to_write/exp_2_8_compression_under_distribution_shift.py | NOT BUILT | |
| **2.9** | **cat2_how_to_write/exp_2_9_compression_objectives.py** | **READY** | **P7** |

## Category 3: When To Write (7 experiments)

| ID | File | Status | Priority |
|----|------|--------|----------|
| 3.1 | cat3_when_to_write/exp_3_1_continuous_vs_event_driven.py | NOT BUILT | |
| **3.2** | **cat3_when_to_write/exp_3_2_write_gate_collapse.py** | **READY** | **P3** |
| 3.3 | cat3_when_to_write/exp_3_3_write_timing_vs_content.py | NOT BUILT | |
| 3.4 | cat3_when_to_write/exp_3_4_boundary_detection_write_trigger.py | NOT BUILT | |
| 3.5 | cat3_when_to_write/exp_3_5_write_latency_sensitivity.py | NOT BUILT | |
| 3.6 | cat3_when_to_write/exp_3_6_retroactive_writing.py | NOT BUILT | |
| 3.7 | cat3_when_to_write/exp_3_7_write_budget_allocation.py | NOT BUILT | |

## Category 4: What To Read (9 experiments)

| ID | File | Status | Priority |
|----|------|--------|----------|
| 4.1 | cat4_what_to_read/exp_4_1_query_formulation_quality.py | NOT BUILT | |
| 4.2 | cat4_what_to_read/exp_4_2_single_vs_multi_vector_retrieval.py | NOT BUILT | |
| 4.3 | cat4_what_to_read/exp_4_3_retrieval_depth_sensitivity.py | NOT BUILT | |
| 4.4 | cat4_what_to_read/exp_4_4_soft_vs_hard_retrieval.py | NOT BUILT | |
| 4.5 | cat4_what_to_read/exp_4_5_cross_level_retrieval.py | NOT BUILT | |
| 4.6 | cat4_what_to_read/exp_4_6_retrieval_by_reconstruction_vs_similarity.py | NOT BUILT | |
| **4.7** | **cat4_what_to_read/exp_4_7_null_retrieval.py** | **READY** | **P4** |
| 4.8 | cat4_what_to_read/exp_4_8_retrieval_interference.py | NOT BUILT | |
| **4.9** | **cat4_what_to_read/exp_4_9_compositional_retrieval.py** | **READY** | **P8** |

## Category 5: When To Read (7 experiments)

| ID | File | Status | Priority |
|----|------|--------|----------|
| **5.1** | **cat5_when_to_read/exp_5_1_read_gate_collapse.py** | **READY** | **P5** |
| 5.2 | cat5_when_to_read/exp_5_2_read_frequency_vs_performance.py | NOT BUILT | |
| 5.3 | cat5_when_to_read/exp_5_3_predictive_read_triggering.py | NOT BUILT | |
| 5.4 | cat5_when_to_read/exp_5_4_read_vs_recompute.py | NOT BUILT | |
| 5.5 | cat5_when_to_read/exp_5_5_cascading_read_depth.py | NOT BUILT | |
| 5.6 | cat5_when_to_read/exp_5_6_read_suppression_high_confidence.py | NOT BUILT | |
| 5.7 | cat5_when_to_read/exp_5_7_attention_memory_arbitration.py | NOT BUILT | |

## Category 6: How To Forget (8 experiments)

| ID | File | Status | Priority |
|----|------|--------|----------|
| 6.1 | cat6_how_to_forget/exp_6_1_eviction_policy_comparison.py | NOT BUILT | |
| 6.2 | cat6_how_to_forget/exp_6_2_forgetting_as_compression.py | NOT BUILT | |
| 6.3 | cat6_how_to_forget/exp_6_3_selective_forgetting_distribution_shift.py | NOT BUILT | |
| 6.4 | cat6_how_to_forget/exp_6_4_protected_memory_slots.py | NOT BUILT | |
| 6.5 | cat6_how_to_forget/exp_6_5_forgetting_curve_mimicry.py | NOT BUILT | |
| **6.6** | **cat6_how_to_forget/exp_6_6_controller_catastrophic_forgetting.py** | **READY** | **P9** |
| 6.7 | cat6_how_to_forget/exp_6_7_write_evict_coupling.py | NOT BUILT | |
| 6.8 | cat6_how_to_forget/exp_6_8_memory_consolidation.py | NOT BUILT | |

## Category 7: Cross-Cutting (9 experiments)

| ID | File | Status | Priority |
|----|------|--------|----------|
| **7.1** | **cat7_cross_cutting/exp_7_1_differentiability.py** | **READY** | **P6** |
| 7.2 | cat7_cross_cutting/exp_7_2_controller_overhead_budget.py | NOT BUILT | |
| 7.3 | cat7_cross_cutting/exp_7_3_controller_generalization.py | NOT BUILT | |
| 7.4 | cat7_cross_cutting/exp_7_4_minimal_controller_architecture.py | NOT BUILT | |
| 7.5 | cat7_cross_cutting/exp_7_5_controller_stability_under_scale.py | NOT BUILT | |
| 7.6 | cat7_cross_cutting/exp_7_6_adversarial_memory_probing.py | NOT BUILT | |
| 7.7 | cat7_cross_cutting/exp_7_7_controller_bottleneck_identification.py | NOT BUILT | |
| 7.8 | cat7_cross_cutting/exp_7_8_joint_vs_sequential_training.py | NOT BUILT | |
| **7.9** | **cat7_cross_cutting/exp_7_9_interpretability_baseline.py** | **READY** | **P10** |

---

## Running the Priority Queue

```bash
cd research/

# P1 — Write signal ablation
python experiments/cat1_what_to_write/exp_1_5_write_signal_ablation.py

# P2 — Compression ratio curve
python experiments/cat2_how_to_write/exp_2_1_compression_ratio_curve.py

# P3 — Write gate collapse
python experiments/cat3_when_to_write/exp_3_2_write_gate_collapse.py

# P4 — Null retrieval learning
python experiments/cat4_what_to_read/exp_4_7_null_retrieval.py

# P5 — Read gate collapse
python experiments/cat5_when_to_read/exp_5_1_read_gate_collapse.py

# P6 — Differentiability
python experiments/cat7_cross_cutting/exp_7_1_differentiability.py

# P7 — Compression objectives
python experiments/cat2_how_to_write/exp_2_9_compression_objectives.py

# P8 — Compositional retrieval
python experiments/cat4_what_to_read/exp_4_9_compositional_retrieval.py

# P9 — Controller catastrophic forgetting
python experiments/cat6_how_to_forget/exp_6_6_controller_catastrophic_forgetting.py

# P10 — Interpretability baseline (run alongside all others)
python experiments/cat7_cross_cutting/exp_7_9_interpretability_baseline.py
```

Results land in `research/results/`. Record findings in `research/log/research_log.md`.
