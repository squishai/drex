# drex / research

This directory is the primary workspace for hypothesis-driven spiral research into
the memory controller problem. No production code lives here. No architecture is
assumed. The goal of this phase is to earn the right to know what to build.

## The Premise

The best architecture for a learned memory controller has not been invented yet.
That belief shapes everything about how work in this directory is structured.

Racing five known architectures optimizes within a known solution space. Open-ended
research without implementation never converges. The only path that works is:

```
Identify the core insight nobody has acted on yet
          ↓
Build the smallest possible thing that tests that insight
          ↓
Let what you learn reshape the next question
          ↓
Repeat until the architecture reveals itself
          ↓
Then commit and build seriously
```

## The Six Hard Problems

Every experiment maps to one or more of these. No existing system solves all six.

| # | Problem | Core Question |
|---|---------|---------------|
| 1 | **What to write** | What information is worth storing? |
| 2 | **How to write** | How do you compress it faithfully? |
| 3 | **When to write** | At what point during processing should you store? |
| 4 | **What to read** | What do you retrieve given a current query? |
| 5 | **When to read** | When is retrieval needed vs. working memory sufficient? |
| 6 | **How to forget** | What do you evict when storage is full? |

## Directory Structure

```
research/
├── README.md                     # this file
├── hypotheses.md                 # falsifiable hypotheses, one per experiment
├── requirements.txt              # python dependencies
├── experiments/
│   ├── base.py                   # base experiment class, logging, result schema
│   ├── cat1_what_to_write/       # experiments 1.1–1.8
│   ├── cat2_how_to_write/        # experiments 2.1–2.9
│   ├── cat3_when_to_write/       # experiments 3.1–3.7
│   ├── cat4_what_to_read/        # experiments 4.1–4.9
│   ├── cat5_when_to_read/        # experiments 5.1–5.7
│   ├── cat6_how_to_forget/       # experiments 6.1–6.8
│   └── cat7_cross_cutting/       # experiments 7.1–7.9
├── results/                      # JSON result files written by experiments
└── log/
    └── research_log.md           # living document — findings, surprises, pivots
```

## How To Run An Experiment

Each experiment is a standalone Python script. From the `research/` directory:

```bash
python experiments/cat1_what_to_write/exp_1_5_write_signal_ablation.py
```

Results are written to `results/<experiment_id>_<timestamp>.json` automatically.
Every result includes: hypothesis, outcome (SUPPORTED / REFUTED / INCONCLUSIVE),
key metrics, and a notes field for observations.

## The Priority Queue

Run these ten first, in order. They establish the most load-bearing knowledge
before anything else.

1. `exp_1_5` — Write signal ablation (baseline before everything)
2. `exp_2_1` — Compression ratio curve (hard physical limits)
3. `exp_3_2` — Write gate collapse detection (most likely early failure mode)
4. `exp_4_7` — Null retrieval learning (efficiency lives or dies here)
5. `exp_5_1` — Read gate collapse detection (mirror of write gate collapse)
6. `exp_7_1` — Differentiability problem (load-bearing for whole training approach)
7. `exp_2_9` — Retrieval vs storage compression objectives (may reveal architectural split)
8. `exp_4_9` — Compositional retrieval (tests what transformers genuinely cannot do)
9. `exp_6_6` — Controller catastrophic forgetting (meta-level problem nobody talks about)
10. `exp_7_9` — Interpretability baseline (run alongside everything else from day one)

## What To Do With Surprising Results

When something breaks in a way that doesn't make sense, or when a simple approach
dramatically outperforms a complex one, or when two objectives that should align
turn out to be fundamentally at odds — stop everything and understand exactly why.

That moment is where the architecture nobody has built yet is hiding.

Record it in `log/research_log.md` immediately. The failure especially.

## Research Phases

**Phase 1 — Mapping the Unknown**
Run the experiments. Build a map of the problem space. No full architectures yet.

**Phase 2 — Insight Crystallization**
Something unexpected will happen in Phase 1. Follow that signal.

**Phase 3 — Core Architecture POC**
One architecture. The one Phase 2 pointed at. Not five. One.

**Phase 4 — Stress Testing**
Break it deliberately. Find the failure modes.

**Phase 5 — Scale Decision**
Only now decide whether to scale. You'll have real confidence, not hope.
