"""Integration tests for DREXUnified — covers spec validation criteria 1–3.

Criteria 4 (BoW perplexity baseline) and 5 (transformer comparison) require
WikiText-2 benchmarks and are evaluated via separate scripts (Phase 4 gate).

Small config used throughout — fast on CPU, no GPU needed.
"""
from __future__ import annotations

import unittest

import torch

from drex_unified import DREXUnified

# Shared small config — keeps construction fast and forward cheap.
_SMALL = dict(
    vocab_size=256,
    d_hdc=512,
    d_model=64,
    n_reservoir=64,
    n_mamba_layers=2,
    n_semantic_blocks=2,
    n_tiers=3,
    top_k=2,
)


def _make_model() -> DREXUnified:
    return DREXUnified(**_SMALL)


# ---------------------------------------------------------------------------
# Criterion 1: Shape propagation
# ---------------------------------------------------------------------------


class TestShapePropagation(unittest.TestCase):
    """All intermediate and output tensors have correct shapes end-to-end."""

    def setUp(self) -> None:
        self.model = _make_model()
        self.model.eval()

    def test_single_input_shapes(self) -> None:
        with torch.no_grad():
            out = self.model.forward(["hello"])

        self.assertEqual(out["logits"].shape, (1, _SMALL["vocab_size"]))
        self.assertEqual(out["routing_weights"].shape, (1, _SMALL["n_tiers"]))
        self.assertEqual(
            out["new_episodic_state"].shape, (1, _SMALL["d_model"])
        )
        self.assertIsNone(out["task_loss"])

    def test_batch_input_shapes(self) -> None:
        with torch.no_grad():
            out = self.model.forward(["hello", "world", "test"])

        self.assertEqual(out["logits"].shape, (3, _SMALL["vocab_size"]))
        self.assertEqual(out["routing_weights"].shape, (3, _SMALL["n_tiers"]))
        self.assertEqual(
            out["new_episodic_state"].shape, (3, _SMALL["d_model"])
        )

    def test_logits_are_finite(self) -> None:
        with torch.no_grad():
            out = self.model.forward(["hello world"])

        self.assertTrue(
            torch.isfinite(out["logits"]).all().item(),
            "logits contain NaN or Inf",
        )


# ---------------------------------------------------------------------------
# Criterion 2: Gradient isolation
# ---------------------------------------------------------------------------


class TestGradientIsolation(unittest.TestCase):
    """L1 and L2 receive zero gradient from the task loss.

    NoProp semantic (L3) trains via local block losses, not via task loss,
    so its parameters must also be unaffected by task_loss.backward().
    """

    def test_semantic_blocks_no_gradient_from_task_loss(self) -> None:
        model = _make_model()
        model.train()

        # Single next-token target: ASCII 'w' (119)
        targets = torch.tensor([119], dtype=torch.int32)
        out = model.forward(["hello"], targets=targets)

        # DREXController.update() is called inside forward() which zeroes the
        # policy gradient internally — controller params have no grad here.
        out["task_loss"].backward()

        # All NoProp semantic block parameters must have grad is None.
        for i, block in enumerate(model.semantic_memory.blocks):
            for name, param in block.named_parameters():
                self.assertIsNone(
                    param.grad,
                    f"semantic_memory.blocks[{i}].{name} received gradient "
                    f"from task loss (NoProp contract violated)",
                )

        # ESN has no trainable parameters → structural isolation.
        esn_params = list(model.working_memory.parameters())
        self.assertEqual(
            len(esn_params),
            0,
            "EchoStateNetwork must have no nn.Parameters (structural L1 isolation)",
        )


# ---------------------------------------------------------------------------
# Criterion 3: Memory tier independence (ablation without crash)
# ---------------------------------------------------------------------------


class TestTierAblation(unittest.TestCase):
    """Each memory tier can be ablated (zeroed) without crashing the pipeline."""

    def setUp(self) -> None:
        self.model = _make_model()
        self.model.eval()

    def test_esn_ablation(self) -> None:
        """Zero L1 (working memory / ESN) — pipeline must not crash."""
        n_reservoir = _SMALL["n_reservoir"]
        original_fwd = self.model.working_memory.forward

        def zero_esn(
            x_seq: torch.Tensor,
            feedback_seq: object = None,
            initial_state: object = None,
        ) -> torch.Tensor:
            B, S, _ = x_seq.shape
            return torch.zeros(B, S, n_reservoir, dtype=torch.float32)

        self.model.working_memory.forward = zero_esn  # type: ignore[method-assign]
        try:
            with torch.no_grad():
                out = self.model.forward(["hi"])
            self.assertIn("logits", out)
            self.assertEqual(out["logits"].shape, (1, _SMALL["vocab_size"]))
        finally:
            self.model.working_memory.forward = original_fwd

    def test_episodic_ablation(self) -> None:
        """Zero L2 (episodic memory) — pipeline must not crash."""
        d_model = _SMALL["d_model"]
        original_fwd = self.model.episodic_memory.forward

        def zero_episodic(
            write_signal: torch.Tensor,
            state: object = None,
            force_overwrite: bool = False,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            B = write_signal.shape[0]
            zeros = torch.zeros(B, d_model, dtype=torch.float32)
            return zeros, zeros

        self.model.episodic_memory.forward = zero_episodic  # type: ignore[method-assign]
        try:
            with torch.no_grad():
                out = self.model.forward(["hi"])
            self.assertIn("logits", out)
            self.assertEqual(out["logits"].shape, (1, _SMALL["vocab_size"]))
        finally:
            self.model.episodic_memory.forward = original_fwd

    def test_semantic_ablation(self) -> None:
        """Zero L3 (semantic memory) — pipeline must not crash."""
        d_model = _SMALL["d_model"]
        original_fwd = self.model.semantic_memory.forward

        def zero_semantic(
            x: torch.Tensor,
        ) -> tuple[torch.Tensor, list]:
            B, S, _ = x.shape
            return torch.zeros(B, S, d_model, dtype=torch.float32), []

        self.model.semantic_memory.forward = zero_semantic  # type: ignore[method-assign]
        try:
            with torch.no_grad():
                out = self.model.forward(["hi"])
            self.assertIn("logits", out)
            self.assertEqual(out["logits"].shape, (1, _SMALL["vocab_size"]))
        finally:
            self.model.semantic_memory.forward = original_fwd


# ---------------------------------------------------------------------------
# Criterion: Ablation log integrity
# ---------------------------------------------------------------------------

_REQUIRED_COMPONENTS = {
    "hdc_encoder",
    "mamba_backbone",
    "esn_working_memory",
    "episodic_memory",
    "semantic_memory_noprop",
    "sparse_router",
    "kan_readout",
    "controller_rl",
    "reward_feedback",
}


class TestAblationLog(unittest.TestCase):
    """ablation_log is present and schema-correct on every forward pass."""

    def setUp(self) -> None:
        self.model = _make_model()
        self.model.eval()

    def test_ablation_log_schema(self) -> None:
        with torch.no_grad():
            out = self.model.forward(["hi"])

        log = out["ablation_log"]
        self.assertIn("components", log)
        components = log["components"]
        self.assertEqual(
            set(components.keys()),
            _REQUIRED_COMPONENTS,
            f"ablation_log['components'] keys mismatch: "
            f"got {set(components.keys())}",
        )
        for key, val in components.items():
            self.assertIsInstance(
                val, bool, f"component '{key}' must be bool, got {type(val)}"
            )

    def test_reward_feedback_flag(self) -> None:
        with torch.no_grad():
            out_no_target = self.model.forward(["hello"])
        self.assertFalse(
            out_no_target["ablation_log"]["components"]["reward_feedback"],
            "reward_feedback must be False when no targets provided",
        )

        targets = torch.tensor([104], dtype=torch.int32)  # ASCII 'h'
        with torch.no_grad():
            out_with_target = self.model.forward(["hello"], targets=targets)
        self.assertTrue(
            out_with_target["ablation_log"]["components"]["reward_feedback"],
            "reward_feedback must be True when targets provided",
        )


# ---------------------------------------------------------------------------
# Criterion: Dtype contracts
# ---------------------------------------------------------------------------


class TestDtypeContracts(unittest.TestCase):
    """All output tensors carry the correct dtypes."""

    def test_output_dtypes(self) -> None:
        model = _make_model()
        model.eval()
        with torch.no_grad():
            out = model.forward(["hello"])

        self.assertEqual(
            out["logits"].dtype,
            torch.float32,
            f"logits dtype: expected float32, got {out['logits'].dtype}",
        )
        self.assertEqual(
            out["new_episodic_state"].dtype,
            torch.float32,
            f"new_episodic_state dtype: expected float32, "
            f"got {out['new_episodic_state'].dtype}",
        )
        self.assertEqual(
            out["routing_weights"].dtype,
            torch.float32,
            f"routing_weights dtype: expected float32, "
            f"got {out['routing_weights'].dtype}",
        )


if __name__ == "__main__":
    unittest.main()
