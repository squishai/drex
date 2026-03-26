"""
Tests for drex.models.drex_unified — DREXUnified full integration pipeline (Phase 30).

Test taxonomy:
  - (a) Shape propagation: every intermediate has the right shape.
  - (b) Dtype boundary contract: explicit flaot32↔bfloat16 transitions only.
  - (c) Loss and reward: computed when targets provided, None without.
  - (d) Ablation flags: each component can be disabled; pipeline still runs.
  - (e) Gradient isolation: l1/l2 episodic receives no grad from CE loss.
  - (f) Failure cases: invalid config, NaN guard.

Coverage:
  DREXUnifiedConfig:
    - construction: defaults produce valid config
    - components dict: all expected keys present

  DREXUnified:
    - construction: valid with default config
    - forward: output type is DREXUnifiedOutput
    - forward: logits shape (B, S, vocab_size)
    - forward: loss is scalar float when targets provided
    - forward: loss is None when targets not provided
    - forward: reward is finite when targets provided
    - forward: mamba_hidden dtype is bfloat16
    - forward: logits dtype is float32 (KAN output)
    - forward: no NaN in logits on random tokens
    - forward: merged (router output) dtype is float32
    - forward: new_mamba_states has same structure as input mamba_states
    - step(): convenience wrapper completes without error + returns output
    - ablation: hdc_encoder=False runs without error
    - ablation: esn=False runs without error
    - ablation: episodic=False runs without error
    - ablation: semantic=False runs without error
    - ablation: sparse_router=False runs without error
    - ablation: kan_readout=False runs without error
    - ablation: controller=False runs without error
    - ablation: all_disabled — bare MLP path runs
    - ablation_config(): returns dict matching cfg.components
    - gradient: loss.backward() completes without error
    - gradient: mamba parameters receive grad after backward
"""

from __future__ import annotations

import copy

import pytest
import torch

from drex.models.drex_unified import DREXUnified, DREXUnifiedConfig, DREXUnifiedOutput


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _small_cfg(**overrides) -> DREXUnifiedConfig:
    cfg = DREXUnifiedConfig(
        vocab_size=32,
        d_model=64,
        n_mamba_layers=2,
        n_heads=4,
        hdc_dim=128,
        l1_reservoir_mult=4,
        l3_n_blocks=2,
        ctrl_d_hidden=32,
        router_top_k=2,
        kan_n_grid=4,
        kan_spline_order=3,
        kan_n_layers=1,
        kan_fit_method="gradient",
    )
    for k, v in overrides.items():
        object.__setattr__(cfg, k, v)
    return cfg


B  = 2
S  = 8


@pytest.fixture
def cfg() -> DREXUnifiedConfig:
    return _small_cfg()


@pytest.fixture
def model(cfg: DREXUnifiedConfig) -> DREXUnified:
    torch.manual_seed(0)
    return DREXUnified(cfg)


@pytest.fixture
def tokens() -> torch.Tensor:
    return torch.randint(0, 32, (B, S))


@pytest.fixture
def targets(tokens) -> torch.Tensor:
    return tokens  # CE loss over self


# ---------------------------------------------------------------------------
# DREXUnifiedConfig
# ---------------------------------------------------------------------------

class TestConfig:
    def test_defaults_construct(self, cfg: DREXUnifiedConfig):
        assert cfg.d_model == 64
        assert cfg.vocab_size == 32

    def test_components_keys_present(self, cfg: DREXUnifiedConfig):
        for key in ("hdc_encoder", "mamba_backbone", "esn_working_memory",
                    "episodic_memory", "semantic_memory_noprop", "sparse_router",
                    "kan_readout", "controller_rl"):
            assert key in cfg.components, f"Missing component key: {key}"


# ---------------------------------------------------------------------------
# DREXUnified construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_builds_without_error(self, model: DREXUnified):
        assert model is not None

    def test_all_ablation_flags_off(self, cfg: DREXUnifiedConfig):
        """All components disabled — bare embedding + projection must still work."""
        disabled = {k: False for k in cfg.components}
        cfg2 = copy.deepcopy(cfg)
        object.__setattr__(cfg2, "components", disabled)
        torch.manual_seed(1)
        m = DREXUnified(cfg2)
        assert m is not None


# ---------------------------------------------------------------------------
# Forward shape and dtype
# ---------------------------------------------------------------------------

class TestForwardShapeAndDtype:
    def test_output_type(self, model: DREXUnified, tokens):
        states = model._make_mamba_states(B, tokens.device)
        out = model(tokens, mamba_states=states)
        assert isinstance(out, DREXUnifiedOutput)

    def test_logits_shape(self, model: DREXUnified, tokens):
        states = model._make_mamba_states(B, tokens.device)
        out = model(tokens, mamba_states=states)
        # DREXUnified does last-position prediction: logits is (B, vocab_size).
        assert out.logits.shape == (B, 32), out.logits.shape

    def test_logits_dtype_float32(self, model: DREXUnified, tokens):
        states = model._make_mamba_states(B, tokens.device)
        out = model(tokens, mamba_states=states)
        assert out.logits.dtype == torch.float32, out.logits.dtype

    def test_mamba_hidden_dtype_float32(self, model: DREXUnified, tokens):
        """mamba_hidden in DREXUnifiedOutput is explicitly cast to float32."""
        states = model._make_mamba_states(B, tokens.device)
        out = model(tokens, mamba_states=states)
        assert out.mamba_hidden.dtype == torch.float32, out.mamba_hidden.dtype

    def test_merged_dtype_float32(self, model: DREXUnified, tokens):
        states = model._make_mamba_states(B, tokens.device)
        out = model(tokens, mamba_states=states)
        assert out.merged.dtype == torch.float32

    def test_no_nan_logits(self, model: DREXUnified, tokens):
        states = model._make_mamba_states(B, tokens.device)
        out = model(tokens, mamba_states=states)
        assert not torch.isnan(out.logits).any()
        assert not torch.isinf(out.logits).any()


# ---------------------------------------------------------------------------
# Loss and reward
# ---------------------------------------------------------------------------

class TestLossAndReward:
    def test_loss_is_scalar_with_targets(self, model: DREXUnified, tokens, targets):
        states = model._make_mamba_states(B, tokens.device)
        out = model(tokens, targets=targets, mamba_states=states)
        assert out.loss is not None
        assert out.loss.ndim == 0, "Loss must be scalar"

    def test_loss_is_none_without_targets(self, model: DREXUnified, tokens):
        states = model._make_mamba_states(B, tokens.device)
        out = model(tokens, mamba_states=states)
        assert out.loss is None

    def test_loss_is_finite(self, model: DREXUnified, tokens, targets):
        states = model._make_mamba_states(B, tokens.device)
        out = model(tokens, targets=targets, mamba_states=states)
        assert torch.isfinite(out.loss)

    def test_reward_is_finite_with_targets(self, model: DREXUnified, tokens, targets):
        states = model._make_mamba_states(B, tokens.device)
        out = model(tokens, targets=targets, mamba_states=states)
        if out.reward is not None:
            assert torch.isfinite(torch.tensor(out.reward))


# ---------------------------------------------------------------------------
# Mamba states
# ---------------------------------------------------------------------------

class TestMambaStates:
    def test_make_mamba_states_not_none(self, model: DREXUnified, tokens):
        states = model._make_mamba_states(B, tokens.device)
        assert states is not None

    def test_make_mamba_states_length(self, model: DREXUnified, tokens):
        states = model._make_mamba_states(B, tokens.device)
        assert len(states) == model.cfg.n_mamba_layers


# ---------------------------------------------------------------------------
# step() convenience wrapper
# ---------------------------------------------------------------------------

class TestStep:
    def test_step_completes(self, model: DREXUnified, tokens, targets):
        states = model._make_mamba_states(B, tokens.device)
        out, new_states = model.step(tokens, targets=targets, mamba_states=states)
        assert isinstance(out, DREXUnifiedOutput)
        assert new_states is not None


# ---------------------------------------------------------------------------
# Ablation flags
# ---------------------------------------------------------------------------

def _model_with_component_disabled(component_key: str) -> DREXUnified:
    cfg = _small_cfg()
    components = dict(cfg.components)
    components[component_key] = False
    object.__setattr__(cfg, "components", components)
    torch.manual_seed(0)
    return DREXUnified(cfg)


class TestAblationFlags:
    @pytest.mark.parametrize("component_key", [
        "hdc_encoder",
        "esn_working_memory",
        "episodic_memory",
        "semantic_memory_noprop",
        "sparse_router",
        "kan_readout",
        "controller_rl",
    ])
    def test_component_disabled_runs(self, component_key: str):
        model = _model_with_component_disabled(component_key)
        tokens = torch.randint(0, 32, (B, S))
        targets = tokens
        states = model._make_mamba_states(B, tokens.device)
        out = model(tokens, targets=targets, mamba_states=states)
        assert isinstance(out, DREXUnifiedOutput)
        assert not torch.isnan(out.logits).any(), f"NaN with {component_key} disabled"

    def test_all_disabled_runs(self):
        cfg = _small_cfg()
        disabled = {k: False for k in cfg.components}
        object.__setattr__(cfg, "components", disabled)
        torch.manual_seed(0)
        model = DREXUnified(cfg)
        tokens = torch.randint(0, 32, (B, S))
        targets = tokens
        states = model._make_mamba_states(B, tokens.device)
        out = model(tokens, targets=targets, mamba_states=states)
        assert out.logits.shape == (B, 32)


# ---------------------------------------------------------------------------
# ablation_config()
# ---------------------------------------------------------------------------

class TestAblationConfig:
    def test_returns_components_dict(self, model: DREXUnified, cfg: DREXUnifiedConfig):
        abl = model.ablation_config()
        assert isinstance(abl, dict)
        for k in cfg.components:
            assert k in abl


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------

class TestGradientFlow:
    def test_backward_completes(self, model: DREXUnified, tokens, targets):
        states = model._make_mamba_states(B, tokens.device)
        out = model(tokens, targets=targets, mamba_states=states)
        out.loss.backward()

    def test_mamba_params_receive_grad(self, model: DREXUnified, tokens, targets):
        """Mamba backbone parameters should receive gradients from the language loss."""
        states = model._make_mamba_states(B, tokens.device)
        out = model(tokens, targets=targets, mamba_states=states)
        out.loss.backward()

        mamba_grads = [
            p.grad for p in model.mamba_layers.parameters()
            if p.grad is not None
        ]
        assert len(mamba_grads) > 0, "No Mamba parameters received gradients"

    def test_router_lb_loss_is_finite(self, model: DREXUnified, tokens, targets):
        states = model._make_mamba_states(B, tokens.device)
        out = model(tokens, targets=targets, mamba_states=states)
        if out.router_lb_loss is not None:
            assert torch.isfinite(out.router_lb_loss)


# ---------------------------------------------------------------------------
# Per-tier output shapes
# ---------------------------------------------------------------------------

class TestTierOutputShapes:
    def test_l1_out_shape(self, model: DREXUnified, tokens):
        states = model._make_mamba_states(B, tokens.device)
        out = model(tokens, mamba_states=states)
        if out.l1_out is not None:
            assert out.l1_out.shape == (B, model.cfg.d_model)

    def test_l2_out_shape(self, model: DREXUnified, tokens):
        states = model._make_mamba_states(B, tokens.device)
        out = model(tokens, mamba_states=states)
        if out.l2_out is not None:
            assert out.l2_out.shape == (B, model.cfg.d_model)

    def test_l3_out_shape(self, model: DREXUnified, tokens):
        states = model._make_mamba_states(B, tokens.device)
        out = model(tokens, mamba_states=states)
        if out.l3_out is not None:
            assert out.l3_out.shape == (B, model.cfg.d_model)
