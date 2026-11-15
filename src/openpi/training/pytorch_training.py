"""Small PyTorch training utilities, independent of the JAX/data stack."""

import logging
import math
from pathlib import Path
import shutil

import torch


def build_optimizer_param_groups(
    model: torch.nn.Module, *, lr: float, lingbot_pretrained_lr_scale: float = 1.0
) -> list[dict]:
    """Keep the default single group; optionally lower the pretrained encoder LR."""
    scale = lingbot_pretrained_lr_scale
    if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale <= 0:
        raise ValueError("lingbot_pretrained_lr_scale must be finite and positive")
    prefix = "pressure_encoder.pretrained."
    named = list(model.named_parameters())
    if scale != 1.0 and not any(name.startswith(prefix) for name, _ in named):
        raise ValueError("lingbot_pretrained_lr_scale requires a LingBot pressure encoder")
    trainable = [(name, parameter) for name, parameter in named if parameter.requires_grad]
    if not trainable:
        raise ValueError("No trainable parameters remain")
    partitions = [("trainable", 1.0, trainable)]
    if scale != 1.0:
        partitions = [
            ("adapters_and_policy", 1.0, [(n, p) for n, p in trainable if not n.startswith(prefix)]),
            ("lingbot_pretrained", scale, [(n, p) for n, p in trainable if n.startswith(prefix)]),
        ]
    groups = []
    for name, multiplier, entries in partitions:
        if not entries:
            continue
        groups.append(
            {
                "params": [parameter for _, parameter in entries],
                "param_names": [name for name, _ in entries],
                "name": name,
                "lr_scale": multiplier,
                "lr": lr * multiplier,
            }
        )
        logging.info(
            "Optimizer group %s: %s parameters, LR scale %s", name, sum(p.numel() for _, p in entries), multiplier
        )
    return groups


def set_learning_rate(optimizer, lr: float) -> None:
    """Apply the same schedule while preserving group-specific multipliers."""
    for group in optimizer.param_groups:
        # NumPy-based schedules must not leak NumPy scalars into optimizer files.
        group["lr"] = float(lr) * group.get("lr_scale", 1.0)


def load_optimizer_state(optimizer, state: dict) -> None:
    """Reject changed parameter membership or settings before restoring moments."""
    saved_groups = state.get("param_groups", [])
    current_groups = optimizer.param_groups
    if len(saved_groups) != len(current_groups):
        raise ValueError("Optimizer parameter groups changed; start a new experiment for a training-stage change")
    contracts = []
    for current, saved in zip(current_groups, saved_groups, strict=True):
        if len(current["params"]) != len(saved.get("params", [])):
            raise ValueError("Optimizer parameter membership changed; start a new experiment")
        if saved.get("lr_scale", 1.0) != current.get("lr_scale", 1.0):
            raise ValueError("Optimizer LR scale differs from checkpoint; start a new experiment")
        if "param_names" in saved:
            if saved["param_names"] != current["param_names"]:
                raise ValueError("Optimizer parameter names or order differ from checkpoint")
        else:
            logging.warning("Legacy optimizer checkpoint has no parameter-name contract; checking group sizes only")
        for field in ("betas", "eps", "weight_decay"):
            if saved.get(field) != current.get(field):
                raise ValueError(f"Optimizer {field} differs from checkpoint; start a new experiment")
        contracts.append({key: current[key] for key in ("name", "param_names", "lr_scale")})
    optimizer.load_state_dict(state)
    # Old checkpoints do not carry these fields; retain the checked current contract.
    for group, contract in zip(optimizer.param_groups, contracts, strict=True):
        group.update(contract)


def prepare_checkpoint_directory(path: str | Path, *, resume: bool, overwrite: bool) -> None:
    """Called by rank zero only; never silently reuse an existing experiment."""
    path = Path(path).resolve()
    if resume and overwrite:
        raise ValueError("Cannot resume and overwrite at the same time")
    if resume:
        if not path.is_dir():
            raise FileNotFoundError(f"Experiment checkpoint directory {path} does not exist for resume")
        return
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Experiment {path} already exists; choose a new exp-name, --resume, or --overwrite")
        if path.parent == path or not path.is_dir():
            raise ValueError(f"Refusing to overwrite invalid experiment directory {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True)
