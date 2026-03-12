# Drex

A novel language model architecture combining sliding-window attention, Infini-Attention
matrix memory, Titans-style disk cache, and a validated episodic/semantic associative
memory module. Built on 12 phases of hypothesis-driven research (247+ experiments).

## Architecture

Drex uses a four-tier memory hierarchy:

| Layer | Mechanism | Scope |
|-------|-----------|-------|
| L1 | Sliding-window causal attention | In-context (short range) |
| L2 | Infini-Attention delta-rule matrix | Cross-segment (medium range) |
| L3 | Titans-style MLP weight snapshots | Disk (long range, async) |
| L4 | Episodic/semantic split delta-rule | Per-segment associative recall |

### L4 MemoryModule (Phase 13, validated)

The episodic/semantic memory layer is the primary research contribution. Key properties:

- **Two H/2 associative matrices**: `M_sem` (semantic, uniform weight) and `M_epi`
  (episodic, recency-weighted writes)
- **Delta-rule update**: `Δ = (k − Mk̂) ⊗ k̂`, written via EMA with `(1−α)` smoothing
- **Length-adaptive EMA**: `α(L) = 0.95^(96/L)` — keeps τ/L ≈ 0.21 constant
  across L=32–128, solving the EMA bootstrap failure at short sequences
- **OR relative-norm write gate**: fires when `‖k − vp‖ ≥ thresh·‖k‖` on either branch;
  thresh\*=0.70 (confirmed exp_48_1, Phase 12)
- **Null retrieval gate**: learned scalar `g = σ(w·q)` suppresses empty-memory reads
- **Soft concatenated retrieval**: `concat(r_sem, r_epi)` — no learned combination gate
  (exp_38_3 ruled this out)

Validated write rates at thresh=0.70:
- L=32: wr=0.581 (target: 0.20–0.70)
- L=96: wr=0.421 (target: 0.15–0.50)

See [ARCHITECTURE_FINDINGS.md](ARCHITECTURE_FINDINGS.md) for the full specification and
the complete list of research dead ends.

## Installation

### Prerequisites

- Python ≥ 3.11
- Rust toolchain (for the `drex._sys` extension — SnapshotStore, PrefetchEngine)
- PyTorch ≥ 2.3.0

### Build

```bash
git clone https://github.com/yourusername/drex.git
cd drex

# Install maturin (Rust-Python build tool)
pip install maturin

# Build and install the Rust extension + Python package
maturin develop --release
```

### Development install

```bash
pip install -e ".[dev]"
```

## Usage

### Training

```python
from drex.models.transformer import DrexConfig, DrexTransformer
from drex.training.trainer import DrexTrainer
import torch

config = DrexConfig(
    d_model=512,
    n_heads=8,
    n_layers=8,
    ff_mult=4,
    vocab_size=32000,
    window_size=2048,
    max_seq_len=8192,
    use_episodic_memory=True,   # Phase 13 validated architecture
    episodic_gate_thresh=0.70,  # thresh* from exp_48_1
)

model = DrexTransformer(config).to("cuda")
trainer = DrexTrainer(model, config, lr=3e-4, n_segments_per_step=4, segment_len=2048)

# Training loop
for batch_ids in dataloader:
    loss = trainer.train_step(batch_ids)
```

### Passkey recall evaluation

```bash
# Random-init baseline (expected ~0% accuracy)
python scripts/eval_passkey.py --d-model 256 --n-heads 4 --n-layers 4

# With episodic memory and write-rate reporting
python scripts/eval_passkey.py \
    --checkpoint checkpoints/step_0100000.safetensors \
    --use-episodic-memory \
    --report-write-rate \
    --lengths 2048 4096 8192 16384 32768
```

### Write-rate monitoring during training

```python
from drex.models.memory import MemoryModule

for module in model.modules():
    if isinstance(module, MemoryModule):
        wr = module.last_write_rate()
        module.assert_write_rate_valid()  # raises if outside [0.10, 0.85]
```

## Testing

```bash
# Run full test suite with 100% branch coverage requirement
pytest tests/python/

# Run a specific test class
pytest tests/python/test_memory.py::TestMemoryModule -v
```

## Project Structure

```
drex/
├── python/drex/
│   ├── models/
│   │   ├── memory.py          # MemoryState, MemoryModule, TitanMemory, L3MemoryBridge
│   │   ├── attention.py       # SlidingWindowAttention, InfiniAttention, HybridAttention
│   │   └── transformer.py     # DrexConfig, DrexLayer, DrexTransformer
│   ├── training/
│   │   └── trainer.py         # DrexTrainer (TBPTT, grad clip, segment loop)
│   ├── eval/
│   │   └── passkey.py         # PasskeyBenchmark
│   └── utils/
│       └── config.py          # save_checkpoint, load_checkpoint
├── src/                       # Rust source (_sys extension)
├── scripts/
│   ├── train.py
│   └── eval_passkey.py
├── tests/python/
├── research/experiments/      # All 247+ research experiments (cat1–cat48)
├── PLAN.md                    # Implementation roadmap
├── ARCHITECTURE_FINDINGS.md   # Full spec + dead ends
└── CLAUDE.md                  # Project conventions for AI collaboration
```

## Research Summary

12 phases of hypothesis-driven experimentation established the architecture:

- **Phases 1–4**: Established delta-rule update, ELU+1 feature map, L2/L3 baseline
- **Phases 5–6**: Ruled out offline consolidation, hierarchical routing
- **Phases 7–8**: Confirmed outer-product write, eliminated bidirectional rule
- **Phases 9–10**: Confirmed relative-norm gate at thresh=0.40; ruled out
  regularisation and two-phase training
- **Phase 11 (exp_47)**: Discovered EMA bootstrap failure at L≤32; resolved with
  α(L)=0.95^(96/L) length-adaptive formula
- **Phase 12 (exp_48)**: Confirmed thresh\*=0.70 for OR-gate full system

## License

MIT
