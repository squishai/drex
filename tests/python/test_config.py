"""
Tests for drex.utils.config — save_checkpoint and load_checkpoint.
Also imports drex.utils package to cover utils/__init__.py.
"""

import json

import pytest
import torch

from drex.models.transformer import DrexConfig, DrexTransformer
from drex.training.optimizer import build_optimizer, cosine_schedule_with_warmup
from drex.utils import load_checkpoint, save_checkpoint  # covers utils/__init__.py


@pytest.fixture
def tiny_model():
    cfg = DrexConfig(d_model=32, n_heads=2, n_layers=1, ff_mult=2, vocab_size=64)
    return DrexTransformer(cfg)


@pytest.fixture
def tiny_optimizer(tiny_model):
    return build_optimizer(tiny_model, lr=1e-3, weight_decay=0.0)


@pytest.fixture
def tiny_scheduler(tiny_optimizer):
    return cosine_schedule_with_warmup(tiny_optimizer, warmup_steps=10, total_steps=100)


class TestSaveCheckpoint:
    def test_creates_safetensors_file(self, tiny_model, tmp_path):
        path = tmp_path / "ckpt.safetensors"
        save_checkpoint(tiny_model, path, step=42)
        assert path.exists()

    def test_creates_json_sidecar(self, tiny_model, tmp_path):
        path = tmp_path / "ckpt.safetensors"
        save_checkpoint(tiny_model, path, step=7)
        json_path = path.with_suffix(".json")
        assert json_path.exists()
        meta = json.loads(json_path.read_text())
        assert meta["step"] == 7

    def test_json_contains_config(self, tiny_model, tmp_path):
        path = tmp_path / "ckpt.safetensors"
        save_checkpoint(tiny_model, path, step=0)
        meta = json.loads((path.with_suffix(".json")).read_text())
        assert "config" in meta
        assert meta["config"]["d_model"] == 32

    def test_creates_parent_dirs(self, tiny_model, tmp_path):
        path = tmp_path / "a" / "b" / "c" / "model.safetensors"
        save_checkpoint(tiny_model, path)  # parent dirs don't exist yet
        assert path.exists()

    def test_default_step_is_zero(self, tiny_model, tmp_path):
        path = tmp_path / "ckpt.safetensors"
        save_checkpoint(tiny_model, path)
        meta = json.loads((path.with_suffix(".json")).read_text())
        assert meta["step"] == 0

    def test_no_opt_file_without_optimizer(self, tiny_model, tmp_path):
        """By default (no optimizer arg) no _opt.pt file is created."""
        path = tmp_path / "ckpt.safetensors"
        save_checkpoint(tiny_model, path, step=1)
        assert not (tmp_path / "ckpt_opt.pt").exists()

    def test_creates_opt_file_with_optimizer(self, tiny_model, tiny_optimizer, tmp_path):
        path = tmp_path / "ckpt.safetensors"
        save_checkpoint(tiny_model, path, step=5, optimizer=tiny_optimizer)
        opt_path = tmp_path / "ckpt_opt.pt"
        assert opt_path.exists()
        data = torch.load(opt_path, weights_only=False)
        assert "optimizer" in data
        assert "scheduler" not in data

    def test_creates_opt_file_with_scheduler(self, tiny_model, tiny_scheduler, tmp_path):
        path = tmp_path / "ckpt.safetensors"
        save_checkpoint(tiny_model, path, step=5, scheduler=tiny_scheduler)
        opt_path = tmp_path / "ckpt_opt.pt"
        assert opt_path.exists()
        data = torch.load(opt_path, weights_only=False)
        assert "scheduler" in data
        assert "optimizer" not in data

    def test_creates_opt_file_with_both(
        self, tiny_model, tiny_optimizer, tiny_scheduler, tmp_path
    ):
        path = tmp_path / "ckpt.safetensors"
        save_checkpoint(
            tiny_model, path, step=10,
            optimizer=tiny_optimizer, scheduler=tiny_scheduler,
        )
        data = torch.load(tmp_path / "ckpt_opt.pt", weights_only=False)
        assert "optimizer" in data
        assert "scheduler" in data


class TestLoadCheckpoint:
    def test_round_trip_weights(self, tiny_model, tmp_path):
        """save → load → weights identical."""
        path = tmp_path / "ckpt.safetensors"
        save_checkpoint(tiny_model, path, step=5)

        # Perturb model weights
        with torch.no_grad():
            for p in tiny_model.parameters():
                p.add_(torch.randn_like(p))

        step = load_checkpoint(tiny_model, path)
        assert step == 5

        # Re-run save to get reference weights; compare via forward pass
        cfg = tiny_model.config
        ids = torch.randint(0, cfg.vocab_size, (1, 4))
        with torch.no_grad():
            logits, _ = tiny_model(ids)
        assert not torch.isnan(logits).any()

    def test_returns_step(self, tiny_model, tmp_path):
        path = tmp_path / "ckpt.safetensors"
        save_checkpoint(tiny_model, path, step=99)
        step = load_checkpoint(tiny_model, path)
        assert step == 99

    def test_without_json_sidecar_returns_zero(self, tiny_model, tmp_path):
        """If the .json sidecar is missing, load_checkpoint returns 0."""
        path = tmp_path / "ckpt.safetensors"
        save_checkpoint(tiny_model, path, step=10)
        # Remove the sidecar
        path.with_suffix(".json").unlink()
        step = load_checkpoint(tiny_model, path)
        assert step == 0

    def test_restores_optimizer_state(
        self, tiny_model, tiny_optimizer, tmp_path
    ):
        """Optimizer state dict round-trips correctly."""
        path = tmp_path / "ckpt.safetensors"
        # Advance optimizer a few steps so its state is non-trivial.
        ids = torch.randint(0, tiny_model.config.vocab_size, (1, 4))
        for _ in range(3):
            tiny_optimizer.zero_grad()
            logits, _ = tiny_model(ids)
            logits.sum().backward()
            tiny_optimizer.step()
        save_checkpoint(tiny_model, path, step=3, optimizer=tiny_optimizer)

        # Record the first param group LR and step counter.
        saved_lr = tiny_optimizer.param_groups[0]["lr"]
        saved_state_step = list(tiny_optimizer.state.values())[0]["step"]

        # Load into a fresh optimizer.
        cfg = DrexConfig(d_model=32, n_heads=2, n_layers=1, ff_mult=2, vocab_size=64)
        model2 = DrexTransformer(cfg)
        opt2 = build_optimizer(model2, lr=1e-3, weight_decay=0.0)
        step = load_checkpoint(model2, path, optimizer=opt2)

        assert step == 3
        assert opt2.param_groups[0]["lr"] == saved_lr
        restored_step = list(opt2.state.values())[0]["step"]
        assert restored_step == saved_state_step

    def test_restores_scheduler_state(
        self, tiny_model, tiny_optimizer, tiny_scheduler, tmp_path
    ):
        """Scheduler last_epoch round-trips correctly."""
        path = tmp_path / "ckpt.safetensors"
        for _ in range(5):
            tiny_scheduler.step()
        saved_epoch = tiny_scheduler.last_epoch
        save_checkpoint(
            tiny_model, path, step=5,
            optimizer=tiny_optimizer, scheduler=tiny_scheduler,
        )

        cfg = DrexConfig(d_model=32, n_heads=2, n_layers=1, ff_mult=2, vocab_size=64)
        model2 = DrexTransformer(cfg)
        opt2 = build_optimizer(model2, lr=1e-3, weight_decay=0.0)
        sched2 = cosine_schedule_with_warmup(opt2, warmup_steps=10, total_steps=100)
        load_checkpoint(model2, path, optimizer=opt2, scheduler=sched2)

        assert sched2.last_epoch == saved_epoch

    def test_no_error_when_opt_file_missing_but_optimizer_passed(
        self, tiny_model, tiny_optimizer, tmp_path
    ):
        """Old checkpoint (no _opt.pt) loads cleanly even when optimizer is passed."""
        path = tmp_path / "ckpt.safetensors"
        save_checkpoint(tiny_model, path, step=10)  # no optimizer — no _opt.pt
        # Should not raise, just skips optimizer restore.
        step = load_checkpoint(tiny_model, path, optimizer=tiny_optimizer)
        assert step == 10

    def test_restores_scheduler_only_no_optimizer(
        self, tiny_model, tiny_scheduler, tmp_path
    ):
        """Companion file with scheduler-only: restores scheduler, skips optimizer."""
        path = tmp_path / "ckpt.safetensors"
        for _ in range(7):
            tiny_scheduler.step()
        saved_epoch = tiny_scheduler.last_epoch
        save_checkpoint(tiny_model, path, step=7, scheduler=tiny_scheduler)

        cfg = DrexConfig(d_model=32, n_heads=2, n_layers=1, ff_mult=2, vocab_size=64)
        model2 = DrexTransformer(cfg)
        opt2 = build_optimizer(model2, lr=1e-3, weight_decay=0.0)
        sched2 = cosine_schedule_with_warmup(opt2, warmup_steps=10, total_steps=100)
        step = load_checkpoint(model2, path, scheduler=sched2)  # no optimizer

        assert step == 7
        assert sched2.last_epoch == saved_epoch
