"""DREX-UNIFIED: Full integration pipeline.

Wires all 9 components into a single nn.Module.

Dtype boundary contract (enforced here):
  DrexTokenizer.encode     → (B, S) int32
  HDCTokenEncoder.forward  → (B, S, d_hdc)   float32
  input_proj               → (B, S, d_model)  float32
  PCNMambaBackbone.forward → (B, S, d_model)  bfloat16  (cast happens inside backbone)
  DREXController.forward   → asserts bfloat16 input
  EchoStateNetwork.forward → (B, S, N)        float32   (detached for L1 isolation)
  EpisodicMemory.forward   → (B, d_model)     float32   (detached for L2 isolation)
  NoPropSemanticMemory     → (B, S, d_model)  bfloat16  (detached; trains via local loss)
  SparseRouter.forward     → (B, d_model)     float32
  KANReadout.forward       → (B, vocab_size)  float32

Ablation log format (JSON-serialisable dict) is returned on every forward pass.
A run without this field is invalid per DREX_UNIFIED_SPEC.md § ABLATION DISCIPLINE.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from backbone.mamba import PCNMambaBackbone
from controller.policy import DREXController
from controller.reward import RewardSignal
from hdc.encoder import HDCTokenEncoder
from input.tokenizer import DrexTokenizer
from memory.episodic import EpisodicMemory
from memory.reservoir import EchoStateNetwork
from memory.semantic import NoPropSemanticMemory
from readout.kan import KANReadout
from router.sparse import SparseRouter


class DREXUnified(nn.Module):
    """Integrated DREX-UNIFIED inference and training module.

    Connects all nine DREX components in the canonical order defined in
    DREX_UNIFIED_SPEC.md § INTEGRATION SPEC.

    Args:
        vocab_size:        Tokenizer / readout vocabulary size.  Default 256.
        d_hdc:             HDC hypervector dimension.  Default 10_000.
        d_model:           Core model dimension (Mamba, controller, memories).
                           Default 256.
        n_reservoir:       ESN reservoir size.  Default 2000.
        n_mamba_layers:    Number of stacked Mamba layers.  Default 4.
        n_semantic_blocks: Number of NoProp semantic memory blocks.  Default 4.
        n_tiers:           Number of memory routing tiers.  Default 3.
        top_k:             Router top-k selection.  Default 2.
        seed:              Random seed for HDCTokenEncoder and EchoStateNetwork.
                           Default 42.
    """

    def __init__(
        self,
        vocab_size: int = 256,
        d_hdc: int = 10_000,
        d_model: int = 256,
        n_reservoir: int = 2000,
        n_mamba_layers: int = 4,
        n_semantic_blocks: int = 4,
        n_tiers: int = 3,
        top_k: int = 2,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.tokenizer = DrexTokenizer(mode="byte", vocab_size=vocab_size)
        self.hdc_encoder = HDCTokenEncoder(
            d_hdc=d_hdc, vocab_size=vocab_size, seed=seed
        )
        # Projects HDC float32 → d_model float32 before Mamba.
        # The float32 → bfloat16 cast happens inside PCNMambaBackbone.
        self.input_proj = nn.Linear(d_hdc, d_model)
        self.backbone = PCNMambaBackbone(d_model=d_model, n_layers=n_mamba_layers)
        self.controller = DREXController(d_model=d_model, n_tiers=n_tiers)
        self.working_memory = EchoStateNetwork(
            d_model=d_model,
            n_reservoir=n_reservoir,
            d_read=d_model,
            seed=seed,
        )
        # Projects ESN states (n_reservoir) → d_model for the router.
        self.esn_readout = nn.Linear(n_reservoir, d_model, bias=False)
        self.episodic_memory = EpisodicMemory(d_model=d_model)
        self.semantic_memory = NoPropSemanticMemory(
            d_model=d_model, n_blocks=n_semantic_blocks
        )
        self.router = SparseRouter(
            d_model=d_model, n_tiers=n_tiers, top_k=top_k
        )
        self.readout = KANReadout(d_in=d_model, d_out=vocab_size)

    def forward(
        self,
        texts: list[str],
        targets: torch.Tensor | None = None,
        episodic_state: torch.Tensor | None = None,
    ) -> dict:
        """Run the full DREX-UNIFIED pipeline.

        Args:
            texts:          Batch of raw text strings.
            targets:        (B,) int32/int64 next-token targets.  When provided,
                            computes task loss and updates the controller via
                            REINFORCE.  Optional.
            episodic_state: (B, d_model) float32 episodic state carried across
                            calls.  Zeros if None.

        Returns:
            dict with keys:
                logits             – (B, vocab_size) float32
                task_loss          – scalar float32 or None
                new_episodic_state – (B, d_model) float32
                routing_weights    – (B, n_tiers) float32
                ablation_log       – dict  (MUST be present on every forward pass)
        """
        # ------------------------------------------------------------------
        # Step 1: Tokenize
        # ------------------------------------------------------------------
        token_ids = self.tokenizer.encode(texts)  # (B, S) int32

        # ------------------------------------------------------------------
        # Step 2: HDC encode
        # ------------------------------------------------------------------
        hdc_out = self.hdc_encoder(token_ids)  # (B, S, d_hdc) float32

        # ------------------------------------------------------------------
        # Step 3: Project to d_model — float32 → float32
        # The only float32 → bfloat16 boundary is INSIDE PCNMambaBackbone.
        # ------------------------------------------------------------------
        hdc_proj = self.input_proj(hdc_out)  # (B, S, d_model) float32

        # ------------------------------------------------------------------
        # Step 4: Mamba backbone
        # ------------------------------------------------------------------
        hidden, _ = self.backbone(hdc_proj)  # hidden: (B, S, d_model) bfloat16

        # ------------------------------------------------------------------
        # Step 5: Controller routing — DREXController.forward asserts bfloat16
        # Detach hidden before the controller so that controller.update()'s
        # REINFORCE backward() does not free the backbone graph that
        # task_loss.backward() must traverse later.  The controller is trained
        # exclusively via the REINFORCE reward signal — it must never receive
        # gradients from the task loss.
        # ------------------------------------------------------------------
        write_decisions, read_weights, _ = self.controller(hidden.detach())
        # write_decisions: (B, S, n_tiers) int32
        # read_weights:    (B, S, n_tiers) float32

        # ------------------------------------------------------------------
        # Step 6: L1 — Working memory (ESN has no nn.Parameters — gradient
        # isolation is structural; .detach() is an explicit guarantee)
        # ------------------------------------------------------------------
        esn_states = self.working_memory(hdc_proj)  # (B, S, n_reservoir) float32
        l1_out = self.esn_readout(esn_states[:, -1, :]).detach()  # (B, d_model)

        # ------------------------------------------------------------------
        # Step 7: L2 — Episodic memory
        # mamba_last is the last-step bfloat16 Mamba output cast to float32.
        # ------------------------------------------------------------------
        mamba_last = hidden[:, -1, :].float()  # (B, d_model) float32
        new_episodic_state, l2_out = self.episodic_memory(
            mamba_last, state=episodic_state
        )  # both (B, d_model) float32
        l2_out = l2_out.detach()  # explicit L2 gradient isolation

        # ------------------------------------------------------------------
        # Step 8: L3 — Semantic memory (NoProp trains via local block losses;
        # task-loss gradient must not reach semantic parameters)
        # ------------------------------------------------------------------
        sem_out, _ = self.semantic_memory(hdc_proj)  # (B, S, d_model) bfloat16
        l3_out = sem_out[:, -1, :].float().detach()  # (B, d_model) float32

        # ------------------------------------------------------------------
        # Step 9: Sparse router
        # ------------------------------------------------------------------
        merged, routing_weights, _ = self.router(
            [l1_out, l2_out, l3_out], mamba_last
        )  # merged: (B, d_model) float32

        # ------------------------------------------------------------------
        # Step 10: KAN readout
        # ------------------------------------------------------------------
        logits = self.readout(merged)  # (B, vocab_size) float32

        # ------------------------------------------------------------------
        # Step 11: Task loss + REINFORCE update (training-time only)
        # ------------------------------------------------------------------
        task_loss: torch.Tensor | None = None
        if targets is not None:
            task_loss = F.cross_entropy(logits, targets.long())
            reward = RewardSignal.compute(logits.detach(), targets)
            self.controller.update(reward.item())

        # ------------------------------------------------------------------
        # Step 12: Ablation log — required on EVERY forward pass
        # (DREX_UNIFIED_SPEC.md § ABLATION DISCIPLINE)
        # ------------------------------------------------------------------
        ablation_log = {
            "components": {
                "hdc_encoder": True,
                "mamba_backbone": True,
                "esn_working_memory": True,
                "episodic_memory": True,
                "semantic_memory_noprop": True,
                "sparse_router": True,
                "kan_readout": True,
                "controller_rl": True,
                "reward_feedback": targets is not None,
            }
        }

        return {
            "logits": logits,
            "task_loss": task_loss,
            "new_episodic_state": new_episodic_state,
            "routing_weights": routing_weights,
            "ablation_log": ablation_log,
        }
