"""
drex.utils.config — safetensors checkpoint save / load utilities.

Saves model weights to a .safetensors file and stores DrexConfig +
global step in a sidecar .json file with the same stem.

Optionally persists optimizer and LR-scheduler state alongside the model
weights in a companion ``<stem>_opt.pt`` file, enabling faithful resume
(correct LR and Adam moment vectors) from any checkpoint.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

import torch

from drex.models.transformer import DrexConfig, DrexTransformer


def _opt_path(model_path: Path) -> Path:
    """Return the companion optimizer-state path for a given model checkpoint."""
    return model_path.with_name(model_path.stem + "_opt.pt")


def save_checkpoint(
    model: DrexTransformer,
    path: str | Path,
    step: int = 0,
    optimizer: Optional[Any] = None,
    scheduler: Optional[Any] = None,
) -> None:
    """
    Persist model weights, metadata, and optionally optimizer/scheduler state.

    Always creates:
        ``path``               — safetensors weight file
        ``path.stem + .json``  — config + training step

    When *optimizer* or *scheduler* is provided, also creates:
        ``path.stem + _opt.pt`` — optimizer and scheduler state dicts

    Saving optimizer state enables faithful resume: the Adam moment vectors
    and the LR-scheduler position are restored exactly, so training continues
    as if it were never interrupted.

    The parent directory is created automatically if it does not exist.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    from safetensors.torch import save_file

    state = {k: v.cpu().contiguous().clone() for k, v in model.state_dict().items()}
    save_file(state, path)

    meta = {"step": step, "config": asdict(model.config)}
    with open(path.with_suffix(".json"), "w") as fh:
        json.dump(meta, fh, indent=2)

    if optimizer is not None or scheduler is not None:
        opt_state: dict[str, Any] = {}
        if optimizer is not None:
            opt_state["optimizer"] = optimizer.state_dict()
        if scheduler is not None:
            opt_state["scheduler"] = scheduler.state_dict()
        torch.save(opt_state, _opt_path(path))


def load_checkpoint(
    model: DrexTransformer,
    path: str | Path,
    optimizer: Optional[Any] = None,
    scheduler: Optional[Any] = None,
) -> int:
    """
    Load weights from a safetensors checkpoint into *model* in-place.

    When *optimizer* and/or *scheduler* are passed and a companion
    ``<stem>_opt.pt`` file exists, their state dicts are also restored.
    If the companion file is absent (e.g. old checkpoints without saved
    optimizer state), the model is loaded normally and the caller is
    responsible for re-synchronising the scheduler (e.g. by fast-forwarding).

    Returns the global training step stored in the sidecar JSON, or 0 if the
    sidecar does not exist.
    """
    from safetensors.torch import load_file

    path = Path(path)
    state = load_file(path)
    model.load_state_dict(state, strict=True)

    meta_path = path.with_suffix(".json")
    step = 0
    if meta_path.exists():
        with open(meta_path) as fh:
            meta = json.load(fh)
        step = int(meta.get("step", 0))

    companion = _opt_path(path)
    if companion.exists() and (optimizer is not None or scheduler is not None):
        # weights_only=False is required because optimizer state dicts contain
        # plain Python scalars (step counters) alongside tensors.  This file is
        # written exclusively by save_checkpoint above and is trusted.
        opt_state = torch.load(companion, weights_only=False)  # noqa: S614
        if optimizer is not None and "optimizer" in opt_state:
            optimizer.load_state_dict(opt_state["optimizer"])
        if scheduler is not None and "scheduler" in opt_state:
            scheduler.load_state_dict(opt_state["scheduler"])

    return step
