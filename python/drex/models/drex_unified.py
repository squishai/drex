"""
drex.models.drex_unified — Full DREX-UNIFIED Pipeline (Phase 30).

Wires all DREX components into a single end-to-end differentiable (where
intended) pipeline:

    DATA FLOW (DREX_UNIFIED_SPEC.md Integration Spec):

        tokens  (B, S) int32
             ↓
        HDC Encoder      → (B, S, d_model)  float32   (frozen, zero training)
             ↓
        Mamba Backbone   → (B, S, d_model)  bfloat16  (trained via backprop)
             ↓
        DREX Controller  → write_decisions, read_weights, sparse_gates, log_probs
             ↓
        ┌─── L1: EchoStateMemory    (B, S, d_model) float32 — zero training
        ├─── L2: MemoryModule       (B, S, d_model) float32 — near-zero training
        └─── L3: SemanticMemory ────(B, d_model)    float32 — NoProp training
             ↓
        Sparse Router    → (B, d_model)  float32    — top-k mixture
             ↓
        KAN Readout      → (B, d_out)   float32    — spline output projection
             ↓
        RewardLoop       → reward float, esn_feedback (B, d_model)
             ↓ feedback
        ← L1 / L2 →  (feedback injected into ESN state updates)

    GRADIENT POLICY:
        - Mamba backbone: full backprop via cross-entropy task loss.
        - Controller: REINFORCE on detached inputs (no grad to backbone).
        - L1, L2: zero gradient from task loss — buffers/frozen.
        - L3: NoProp local block losses; may update at inference.
        - KAN Readout: gradients from task loss (fit_method="gradient").
        - Sparse Router gate_proj: gradients from task loss.

    DTYPE CONTRACTS (DREX_UNIFIED_SPEC.md Global Conventions):
        HDC Encoder output:        float32
        Mamba input projection:    float32 → bfloat16  [explicit cast here only]
        Mamba output:              bfloat16
        Controller input:          bfloat16   (from Mamba; detached)
        Controller output dtypes:  write_decisions int32, read_weights float32,
                                   sparse_gates bool, log_probs float32
        L1/L2 output:              float32
        L3 output:                 float32
        Sparse Router output:      float32
        KAN Readout output:        float32
        Reward signal:             float32

    All implicit casts between components are bugs.  Explicit casts are
    documented with inline comments.

References:
    DREX_UNIFIED_SPEC.md §Integration Spec
    DREX_UNIFIED_PLAN.md Phase 30
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from drex.models.controller import DREXController, ControllerOutput
from drex.models.hdc_encoder import HDCEncoder
from drex.models.kan_readout import KANReadout
from drex.models.memory import MemoryModule, MemoryState
from drex.models.memory_esn import EchoStateMemory
from drex.models.mamba import MambaLayer
from drex.models.reward import RewardLoop
from drex.models.router import SparseRouter
from drex.models.semantic import SemanticMemory

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class DREXUnifiedConfig:
    """Full configuration for the DREX-UNIFIED architecture.

    All defaults are sized for the "small version" target:
        d_model=256, 4 Mamba layers, char-level vocab=256
    suitable for TinyStories / WikiText-2 experiments on M3 16GB.
    """

    # Core dimensions
    d_model: int = 256
    vocab_size: int = 256           # char-level default; set 32768 for BPE
    n_mamba_layers: int = 4
    n_heads: int = 4                # for Mamba+L2 hybrid mode

    # HDC Encoder
    hdc_dim: int = 4096

    # L1 — EchoStateMemory
    l1_reservoir_mult: int = 4
    l1_spectral_radius: float = 0.95
    l1_connectivity: float = 0.01

    # L2 — MemoryModule
    l2_gate_thresh: float = 0.70
    l2_use_null_gate: bool = True
    l2_use_recency_weight: bool = True

    # L3 — SemanticMemory
    l3_n_blocks: int = 4
    l3_noise_std: float = 0.1
    l3_inference_lr: float = 1e-5
    l3_update_at_inference: bool = True

    # Controller
    ctrl_d_hidden: int = 128
    ctrl_lr: float = 3e-4
    ctrl_gamma: float = 0.99

    # Sparse Router
    router_top_k: int = 2
    router_lb_coeff: float = 0.01

    # KAN Readout
    kan_n_grid: int = 5
    kan_spline_order: int = 3
    kan_n_layers: int = 2
    kan_fit_method: str = "gradient"   # "gradient" for end-to-end; "closed_form" for one-shot

    # Reward
    reward_lambda_quality: float = 1.0
    reward_lambda_sparse: float = 0.01
    reward_lambda_balance: float = 0.001

    # Ablation flags (for experiments — each can be set False to ablate)
    components: dict = field(default_factory=lambda: {
        "hdc_encoder": True,
        "mamba_backbone": True,
        "esn_working_memory": True,
        "episodic_memory": True,
        "semantic_memory_noprop": True,
        "sparse_router": True,
        "kan_readout": True,
        "controller_rl": True,
        "reward_feedback": True,
    })


# ---------------------------------------------------------------------------
# DREX Unified output
# ---------------------------------------------------------------------------

@dataclass
class DREXUnifiedOutput:
    """Named output bundle from a DREXUnified forward pass."""

    logits: torch.Tensor             # (B, vocab_size)  float32
    loss: Optional[torch.Tensor]     # scalar if targets provided
    reward: float                    # scalar reward for controller update
    router_lb_loss: Optional[torch.Tensor]  # scalar load-balance penalty
    ctrl_output: Optional[ControllerOutput]
    merged: torch.Tensor             # (B, d_model) from sparse router
    l1_out: torch.Tensor             # (B, d_model)
    l2_out: torch.Tensor             # (B, d_model)
    l3_out: torch.Tensor             # (B, d_model)
    mamba_hidden: torch.Tensor       # (B, d_model) — last Mamba position


# ---------------------------------------------------------------------------
# DREXUnified model
# ---------------------------------------------------------------------------

class DREXUnified(nn.Module):
    """Full DREX-UNIFIED model.

    Usage (training loop):

        model = DREXUnified(cfg)

        # Forward pass
        out = model(tokens, targets=targets)

        # Task loss + load-balance penalty
        total_loss = out.loss + (out.router_lb_loss or 0.0)
        total_loss.backward()
        optimizer.step()

        # Controller REINFORCE update (separate)
        if out.ctrl_output is not None:
            model.controller.store(out.ctrl_output.log_probs)
            model.controller.update([out.reward])

        # Semantic memory NoProp update (separate; called by trainer on write decision)
        # semantic.train_step(write_signal)  — see SemanticMemory.train_step()

    Args:
        cfg: DREXUnifiedConfig with all component settings.
    """

    def __init__(self, cfg: DREXUnifiedConfig) -> None:
        super().__init__()
        self.cfg = cfg
        dm = cfg.d_model

        # ── HDC Encoder ──────────────────────────────────────────────────────
        if cfg.components.get("hdc_encoder", True):
            self.hdc = HDCEncoder(d_model=dm, hdc_dim=cfg.hdc_dim)
        else:
            self.hdc = None
            log.info("[ablation] hdc_encoder disabled")

        # ── Mamba Backbone ───────────────────────────────────────────────────
        if cfg.components.get("mamba_backbone", True):
            self.mamba_layers = nn.ModuleList([
                MambaLayer(dm, cfg.n_heads, use_l2=cfg.components.get("episodic_memory", True))
                for _ in range(cfg.n_mamba_layers)
            ])
            # DREX dtype contract: Mamba backbone parameters are bfloat16.
            # The explicit float32 → bfloat16 cast happens at input projection.
            self.mamba_layers = self.mamba_layers.to(torch.bfloat16)
        else:
            self.mamba_layers = None
            # Fallback: simple feed-forward
            self._ff_fallback = nn.Sequential(
                nn.Linear(dm, dm * 4), nn.GELU(), nn.Linear(dm * 4, dm)
            )
            log.info("[ablation] mamba_backbone disabled")

        # ── Controller ───────────────────────────────────────────────────────
        if cfg.components.get("controller_rl", True):
            self.controller = DREXController(
                d_input=dm,
                n_tiers=3,
                n_modules=cfg.l3_n_blocks,
                hidden_dim=cfg.ctrl_d_hidden,
                lr=cfg.ctrl_lr,
                gamma=cfg.ctrl_gamma,
            )
        else:
            self.controller = None
            log.info("[ablation] controller_rl disabled")

        # ── L1 — ESN Working Memory ──────────────────────────────────────────
        if cfg.components.get("esn_working_memory", True):
            self.l1 = EchoStateMemory(
                d_model=dm,
                reservoir_mult=cfg.l1_reservoir_mult,
                spectral_radius=cfg.l1_spectral_radius,
                connectivity=cfg.l1_connectivity,
            )
        else:
            self.l1 = None
            self._l1_fallback = nn.Linear(dm, dm, bias=False)
            log.info("[ablation] esn_working_memory disabled")

        # ── L2 — Episodic Memory ─────────────────────────────────────────────
        # MemoryModule is the L2 component (not the MambaLayer's L2).
        if cfg.components.get("episodic_memory", True):
            from drex.models.memory import MemoryModule
            self.l2 = MemoryModule(
                d_model=dm,
                gate_thresh=cfg.l2_gate_thresh,
                use_null_gate=cfg.l2_use_null_gate,
                use_recency_weight=cfg.l2_use_recency_weight,
            )
        else:
            self.l2 = None
            self._l2_fallback = nn.Linear(dm, dm, bias=False)
            log.info("[ablation] episodic_memory disabled")

        # ── L3 — Semantic Memory ─────────────────────────────────────────────
        if cfg.components.get("semantic_memory_noprop", True):
            self.l3 = SemanticMemory(
                d_model=dm,
                n_blocks=cfg.l3_n_blocks,
                noise_std=cfg.l3_noise_std,
                inference_lr=cfg.l3_inference_lr,
                update_at_inference=cfg.l3_update_at_inference,
            )
        else:
            self.l3 = None
            self._l3_fallback = nn.Linear(dm, dm, bias=False)
            log.info("[ablation] semantic_memory_noprop disabled")

        # ── Sparse Router ────────────────────────────────────────────────────
        if cfg.components.get("sparse_router", True):
            self.router = SparseRouter(
                d_model=dm,
                n_tiers=3,
                top_k=cfg.router_top_k,
                load_balance_coeff=cfg.router_lb_coeff,
            )
        else:
            self.router = None
            # Ablated: simple mean over tier outputs.
            log.info("[ablation] sparse_router disabled")

        # ── KAN Readout ──────────────────────────────────────────────────────
        if cfg.components.get("kan_readout", True):
            self.readout = KANReadout(
                d_in=dm,
                d_out=cfg.vocab_size,
                n_grid=cfg.kan_n_grid,
                spline_order=cfg.kan_spline_order,
                n_kan_layers=cfg.kan_n_layers,
                fit_method=cfg.kan_fit_method,
            )
        else:
            # Ablated: standard linear.
            self.readout = nn.Linear(dm, cfg.vocab_size, bias=True)
            log.info("[ablation] kan_readout disabled — using Linear")

        # ── Reward Loop ──────────────────────────────────────────────────────
        if cfg.components.get("reward_feedback", True):
            self.reward_loop = RewardLoop(
                d_model=dm,
                lambda_quality=cfg.reward_lambda_quality,
                lambda_sparse=cfg.reward_lambda_sparse,
                lambda_balance=cfg.reward_lambda_balance,
            )
        else:
            self.reward_loop = None
            log.info("[ablation] reward_feedback disabled")

        # ── Token embedding for backbone input ──────────────────────────────
        self.embedding = nn.Embedding(cfg.vocab_size, dm)
        nn.init.normal_(self.embedding.weight, std=0.02)

        # ── Internal state ───────────────────────────────────────────────────
        self._prev_loss: float = 0.0     # for delta-loss reward computation

        log.info(
            "DREXUnified: d_model=%d, vocab=%d, n_mamba=%d, total_params=%s",
            dm, cfg.vocab_size, cfg.n_mamba_layers, f"{self.n_params():,}",
        )

    # ── helpers ────────────────────────────────────────────────────────────────

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def __repr__(self) -> str:
        return (
            f"DREXUnified("
            f"d_model={self.cfg.d_model}, vocab={self.cfg.vocab_size}, "
            f"n_mamba={self.cfg.n_mamba_layers}, n_params={self.n_params():,})"
        )

    def _make_mamba_states(self, B: int, device: torch.device) -> list[MemoryState]:
        """Initialise MemoryState for each Mamba+L2 layer."""
        n_heads = self.cfg.n_heads
        dm = self.cfg.d_model
        d_k = dm // n_heads
        d_v = dm // n_heads
        return [MemoryState.zeros(B, n_heads, d_k, d_v, device) for _ in range(self.cfg.n_mamba_layers)]

    # ── forward ────────────────────────────────────────────────────────────────

    def forward(
        self,
        tokens: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        mamba_states: Optional[list[MemoryState]] = None,
    ) -> DREXUnifiedOutput:
        """End-to-end DREX-UNIFIED forward pass.

        Args:
            tokens:        (B, S) int32 — input token ids.
            targets:       (B, S) int32 — next-token targets; if provided,
                           cross-entropy loss is computed.
            mamba_states:  Pre-existing Mamba layer states for TBPTT.
                           If None, fresh zero states are created.

        Returns:
            DREXUnifiedOutput — see dataclass definition above.
        """
        B, S = tokens.shape
        device = tokens.device
        dm = self.cfg.d_model

        # ── 1. Token embedding ────────────────────────────────────────────────
        x = self.embedding(tokens).float()  # (B, S, d_model) float32

        # ── 2. HDC Encoder ────────────────────────────────────────────────────
        # HDC output: float32
        if self.hdc is not None:
            x = self.hdc(x)  # (B, S, d_model) float32

        assert x.dtype == torch.float32, f"HDC output dtype mismatch: {x.dtype}"
        assert not (torch.isnan(x).any() or torch.isinf(x).any()), "NaN/Inf after HDC"

        # ── 3. Mamba Backbone ─────────────────────────────────────────────────
        # Explicit dtype cast: float32 → bfloat16 at Mamba input projection boundary.
        # This is the ONLY explicit cast in the pipeline (DREX dtype contract).
        if self.mamba_layers is not None:
            if mamba_states is None:
                mamba_states = self._make_mamba_states(B, device)

            h = x.to(torch.bfloat16)  # (B, S, d_model) bfloat16 — explicit cast
            for layer, state in zip(self.mamba_layers, mamba_states):
                h, state = layer(h, state)
            # Mamba output: bfloat16
        else:
            h = self._ff_fallback(x).to(torch.bfloat16)
            mamba_states = []

        assert h.dtype == torch.bfloat16, f"Mamba output dtype mismatch: {h.dtype}"
        assert not (torch.isnan(h).any() or torch.isinf(h).any()), "NaN/Inf after Mamba"

        # Last-position hidden state used as context for controller and memory tiers.
        # Shape: (B, d_model) bfloat16
        mamba_hidden = h[:, -1, :]

        # ── 4. DREX Controller ────────────────────────────────────────────────
        # Controller input: bfloat16 (passed as-is; controller internally casts to float32)
        ctrl_output: Optional[ControllerOutput] = None
        if self.controller is not None:
            ctrl_output = self.controller(mamba_hidden)  # detaches internally
            write_decisions = ctrl_output.write_decisions  # (B, 3) int32
            sparse_gates    = ctrl_output.sparse_gates     # (B, n_modules) bool
        else:
            # Ablated: always write to all tiers.
            write_decisions = torch.ones(B, 3, dtype=torch.int32, device=device)
            sparse_gates    = torch.ones(B, self.cfg.l3_n_blocks, dtype=torch.bool, device=device)

        # ── 5. Memory Tiers ──────────────────────────────────────────────────
        # All tier inputs and outputs are float32 per dtype contract.
        # mamba_hidden is bfloat16; cast to float32 before memory tiers.
        ctx_f32 = mamba_hidden.float()  # (B, d_model) float32 — explicit cast

        # L1 — ESN Working Memory: takes (B, S, d_model), returns (B, d_model)
        if self.l1 is not None:
            l1_out = self.l1(h.float())
        else:
            l1_out = self._l1_fallback(ctx_f32)
        assert l1_out.dtype == torch.float32, f"L1 output dtype: {l1_out.dtype}"

        # L2 — Episodic Memory: (B, S, d_model) → (B, d_model)
        if self.l2 is not None:
            l2_out = self.l2(h.float())  # internally takes (B, S, d_model), returns (B, d_model)
        else:
            l2_out = self._l2_fallback(ctx_f32)
        assert l2_out.dtype == torch.float32, f"L2 output dtype: {l2_out.dtype}"

        # L3 — Semantic Memory: query with last hidden; update at inference if write.
        if self.l3 is not None:
            l3_out = self.l3.query(ctx_f32)  # (B, d_model) float32
            # Inference-time update — only when training=False + controller wrote.
            if not self.training and self.cfg.l3_update_at_inference:
                self.l3.inference_update(ctx_f32, write_decisions)
        else:
            l3_out = self._l3_fallback(ctx_f32)
        assert l3_out.dtype == torch.float32, f"L3 output dtype: {l3_out.dtype}"

        # ── 6. Sparse Router ─────────────────────────────────────────────────
        # Router input: query = ctx_f32, tier_outputs = [l1_out, l2_out, l3_out]
        router_lb_loss: Optional[torch.Tensor] = None
        if self.router is not None:
            merged, gate_weights, router_logits = self.router(
                tier_outputs=[l1_out, l2_out, l3_out],
                query=ctx_f32,
                sparse_gates=sparse_gates,
            )
            router_lb_loss = self.router.load_balance_loss()
        else:
            # Ablated: simple mean.
            merged = (l1_out + l2_out + l3_out) / 3.0
            gate_weights = None

        assert merged.dtype == torch.float32, f"Router output dtype: {merged.dtype}"
        assert not (torch.isnan(merged).any() or torch.isinf(merged).any()), "NaN/Inf in merged"

        # ── 7. KAN Readout ────────────────────────────────────────────────────
        logits = self.readout(merged)  # (B, d_out=vocab_size) float32
        assert logits.dtype in (torch.float32, torch.bfloat16), f"Logits dtype: {logits.dtype}"
        logits = logits.float()  # normalise to float32 for loss computation

        assert not (torch.isnan(logits).any() or torch.isinf(logits).any()), "NaN/Inf in logits"

        # ── 8. Loss ───────────────────────────────────────────────────────────
        loss: Optional[torch.Tensor] = None
        if targets is not None:
            # targets: (B, S); logits: (B, vocab_size) — use last-position pred.
            # For LM, targets are shifted by caller.
            last_targets = targets[:, -1].long()  # (B,)
            loss = F.cross_entropy(logits, last_targets)

        # ── 9. Reward Loop ────────────────────────────────────────────────────
        reward = 0.0
        if self.reward_loop is not None and loss is not None:
            reward, _ = self.reward_loop.compute(
                loss_t=loss.item(),
                loss_prev=self._prev_loss,
                write_decisions=write_decisions,
            )
            # ESN feedback: inject into memory tiers' next step.
            # (Injected on the next cycle; stored as module state.)
            esn_fb = self.reward_loop.esn_feedback(ctx_f32)  # (B, d_model) float32
            # Feedback is available for the caller to inject; not automatic here
            # to keep component boundaries clean.
            self._last_esn_feedback = esn_fb
            self._prev_loss = loss.item()

        # Controller collapse load-balance nudge: inject into reward signal.
        if self.controller is not None and getattr(self.controller, "collapse_penalty", False):
            reward -= 0.1  # small penalty per collapse step

        return DREXUnifiedOutput(
            logits=logits,
            loss=loss,
            reward=reward,
            router_lb_loss=router_lb_loss,
            ctrl_output=ctrl_output,
            merged=merged,
            l1_out=l1_out,
            l2_out=l2_out,
            l3_out=l3_out,
            mamba_hidden=mamba_hidden.float(),
        )

    # ── convenience helpers ────────────────────────────────────────────────────

    def step(
        self,
        tokens: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        mamba_states: Optional[list[MemoryState]] = None,
    ) -> tuple[DREXUnifiedOutput, list[MemoryState]]:
        """Single training step: forward + controller store.

        Returns the output and updated mamba_states so the caller can
        implement TBPTT correctly.

        The caller is responsible for:
          1. Accumulating rewards and calling controller.update(rewards).
          2. Calling l3.train_step(write_signal) for NoProp.
          3. Adding router_lb_loss to the total loss before backward().
        """
        mamba_states = mamba_states or self._make_mamba_states(tokens.shape[0], tokens.device)
        out = self.forward(tokens, targets=targets, mamba_states=mamba_states)
        if self.controller is not None and out.ctrl_output is not None:
            self.controller.store(out.ctrl_output.log_probs)
        return out, mamba_states

    def ablation_config(self) -> dict:
        """Return the current ablation state as the canonical log format.

        Format required by DREX_UNIFIED_SPEC.md Ablation Discipline:
            {"hdc_encoder": bool, "mamba_backbone": bool, ...}
        """
        return dict(self.cfg.components)
