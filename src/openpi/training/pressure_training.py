"""PyTorch-only VTLA initialization, freezing, and checkpoint contracts.

This module intentionally does not import the JAX training stack so its
contracts can be tested with small CPU models.
"""

import dataclasses
import hashlib
import json
import logging
from pathlib import Path
import shutil

from safetensors import safe_open
import safetensors.torch
import torch


def load_base_weights(model: torch.nn.Module, path: str | Path, *, allow_pressure_init: bool = False) -> None:
    """Load all original parameters, allowing only a whole new pressure branch to be absent.

    Full VTLA resume and inference should continue to use strict loading. A
    shape mismatch always raises in safetensors, including new pressure keys
    if they are present in the source checkpoint. Any checkpoint containing a
    pressure key is a VTLA checkpoint and must load the complete model strictly.
    """
    if not allow_pressure_init:
        safetensors.torch.load_model(model, str(path))
        return
    if getattr(model, "pressure_encoder", None) is None:
        raise ValueError("Pressure initialization requires an enabled pressure_encoder.")
    with safe_open(str(path), framework="pt", device="cpu") as checkpoint:
        keys = checkpoint.keys()  # safe_open is not a mapping and exposes its tensor names via keys().
        has_pressure_weights = any(name.startswith("pressure_encoder.") for name in keys)
    if has_pressure_weights:
        # Reject a partially saved/truncated VTLA branch rather than combining
        # trained pressure parameters with unnoticed random initializations.
        safetensors.torch.load_model(model, str(path))
        return
    missing, unexpected = safetensors.torch.load_model(model, str(path), strict=False)
    invalid_missing = sorted(name for name in missing if not name.startswith("pressure_encoder."))
    if invalid_missing or unexpected:
        raise RuntimeError(
            "Base checkpoint does not match the original model: "
            f"missing={invalid_missing}, unexpected={sorted(unexpected)}. "
            "Only pressure_encoder.* may be newly initialized."
        )
    logging.info("Loaded base weights; newly initialized pressure parameters: %s", sorted(missing))


def configure_trainable_parameters(
    model: torch.nn.Module, *, freeze_backbone: bool = False
) -> list[torch.nn.Parameter]:
    """Freeze original PaliGemma parameters without cutting input gradients."""
    if freeze_backbone:
        if getattr(model, "pressure_encoder", None) is None:
            raise ValueError("pressure_freeze_backbone requires an enabled pressure encoder.")
        frozen = 0
        for name, parameter in model.named_parameters():
            if name.startswith("paligemma_with_expert.paligemma."):
                parameter.requires_grad_(requires_grad=False)
                frozen += parameter.numel()
        if frozen == 0:
            raise ValueError("No original PaliGemma parameters matched the expected backbone path.")
        logging.info("Froze %s PaliGemma parameters; kept the prefix computation graph.", frozen)

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise ValueError("No trainable parameters remain.")
    logging.info("Trainable parameters: %s", sum(parameter.numel() for parameter in trainable))
    return trainable


def _pressure_config(model: torch.nn.Module) -> dict | None:
    config = getattr(getattr(model, "config", None), "pressure", None)
    return None if config is None else dataclasses.asdict(config)


def _file_hash(path: str | Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save_pressure_metadata(
    model: torch.nn.Module,
    checkpoint_dir: str | Path,
    *,
    freeze_backbone: bool,
    stats_path: str | Path | None = None,
) -> None:
    """Save the branch contract and an immutable copy of its normalization asset."""
    config = _pressure_config(model)
    if config is None:
        return
    checkpoint_dir = Path(checkpoint_dir)
    stats_hash = None
    if stats_path is not None:
        target = checkpoint_dir / "assets" / "pressure_stats.npz"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(stats_path, target)
        stats_hash = _file_hash(target)
    metadata = {
        "schema_version": 1,
        "pressure": config,
        "freeze_backbone": freeze_backbone,
        "stats_sha256": stats_hash,
    }
    (checkpoint_dir / "pressure_config.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def validate_pressure_metadata(
    model: torch.nn.Module,
    checkpoint_dir: str | Path,
    *,
    freeze_backbone: bool,
    stats_path: str | Path | None = None,
) -> None:
    """Reject resume with different hardware/token, freeze, or normalization contracts."""
    checkpoint_dir = Path(checkpoint_dir)
    config = _pressure_config(model)
    metadata_path = checkpoint_dir / "pressure_config.json"
    if not metadata_path.exists():
        if config is not None:
            raise ValueError(
                "A pressure-enabled resume requires pressure_config.json; use base initialization instead."
            )
        return
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("schema_version") != 1:
        raise ValueError("Unsupported pressure checkpoint metadata version.")
    if metadata.get("pressure") != config:
        raise ValueError("Pressure configuration differs from the checkpoint.")
    if metadata.get("freeze_backbone") != freeze_backbone:
        raise ValueError("Pressure backbone freeze setting differs from the checkpoint optimizer contract.")
    expected_hash = metadata.get("stats_sha256")
    actual_hash = None if stats_path is None else _file_hash(stats_path)
    if expected_hash != actual_hash:
        raise ValueError("Pressure normalization statistics differ from the checkpoint.")
    if expected_hash is not None and _file_hash(checkpoint_dir / "assets" / "pressure_stats.npz") != expected_hash:
        raise ValueError("The pressure normalization asset in the checkpoint was modified.")
