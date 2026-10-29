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


def _norm_stats_hash(content: str) -> str:
    """Hash the JSON data independently of whitespace and platform line endings."""
    canonical = json.dumps(json.loads(content), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def pressure_policy_contract(model_config, data_config) -> dict | None:
    """Capture the action and normalization settings needed to reproduce a real policy."""
    robot_action_dim = getattr(data_config, "robot_action_dim", None)
    if robot_action_dim is None:
        return None  # Fake-data diagnostics have no real robot or state/action statistics.
    asset_id = data_config.asset_id
    if not asset_id or Path(asset_id).is_absolute() or ".." in Path(asset_id).parts:
        raise ValueError("A real pressure policy requires a valid relative state/action asset_id.")
    return {
        "robot_action_dim": robot_action_dim,
        "model_action_dim": model_config.action_dim,
        "action_horizon": model_config.action_horizon,
        "max_token_len": model_config.max_token_len,
        "discrete_state_input": model_config.discrete_state_input,
        "use_quantile_norm": data_config.use_quantile_norm,
        "asset_id": asset_id,
        "paligemma_variant": getattr(model_config, "paligemma_variant", None),
        "action_expert_variant": getattr(model_config, "action_expert_variant", None),
        "pytorch_debug_tiny": getattr(model_config, "pytorch_debug_tiny", False),
    }


def should_save_checkpoint(global_step: int, num_train_steps: int, save_interval: int) -> bool:
    """Steps are counted after optimizer updates; always persist the final update."""
    if save_interval <= 0:
        raise ValueError("save_interval must be positive")
    return global_step > 0 and (global_step % save_interval == 0 or global_step == num_train_steps)


def save_pressure_metadata(
    model: torch.nn.Module,
    checkpoint_dir: str | Path,
    *,
    freeze_backbone: bool,
    stats_path: str | Path | None = None,
    policy_contract: dict | None = None,
    norm_stats_path: str | Path | None = None,
) -> None:
    """Save the branch contract and an immutable copy of its normalization asset."""
    config = _pressure_config(model)
    if config is None:
        return
    checkpoint_dir = Path(checkpoint_dir)
    if policy_contract is not None and stats_path is None:
        raise ValueError("Real pressure checkpoints require pressure normalization statistics")
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
    if policy_contract is not None:
        if norm_stats_path is None:
            raise ValueError("Real pressure checkpoints require saved state/action normalization statistics")
        expected_norm_path = checkpoint_dir / "assets" / policy_contract["asset_id"] / "norm_stats.json"
        if Path(norm_stats_path).resolve() != expected_norm_path.resolve():
            raise ValueError("State/action normalization must be saved in the checkpoint assets directory")
        metadata["schema_version"] = 2
        metadata["policy_contract"] = policy_contract
        metadata["norm_stats_sha256"] = _norm_stats_hash(Path(norm_stats_path).read_text(encoding="utf-8"))
    (checkpoint_dir / "pressure_config.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def validate_pressure_metadata(
    model: torch.nn.Module,
    checkpoint_dir: str | Path,
    *,
    freeze_backbone: bool,
    stats_path: str | Path | None = None,
    policy_contract: dict | None = None,
    current_norm_stats_json: str | None = None,
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
    schema = metadata.get("schema_version")
    if schema not in (1, 2):
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
    if schema == 1 and policy_contract is not None:
        logging.warning("Legacy pressure checkpoint has no action or state/action normalization contract to verify.")
    if schema == 2:
        if policy_contract is None or metadata.get("policy_contract") != policy_contract:
            raise ValueError("Pressure policy action, tokenization, or normalization contract differs from checkpoint.")
        asset_id = policy_contract.get("asset_id")
        if not asset_id or Path(asset_id).is_absolute() or ".." in Path(asset_id).parts:
            raise ValueError("Pressure checkpoint requires a valid relative state/action asset_id.")
        norm_path = checkpoint_dir / "assets" / asset_id / "norm_stats.json"
        expected_norm_hash = metadata.get("norm_stats_sha256")
        if not expected_norm_hash or _norm_stats_hash(norm_path.read_text(encoding="utf-8")) != expected_norm_hash:
            raise ValueError("Checkpoint state/action normalization statistics differ from the saved contract.")
        if current_norm_stats_json is not None and _norm_stats_hash(current_norm_stats_json) != expected_norm_hash:
            raise ValueError("Current state/action normalization statistics differ from the checkpoint.")
