import dataclasses
import json
from types import SimpleNamespace

import pytest
import safetensors.torch
import torch
from torch import nn

from openpi.models.pressure_config import PressureConfig
from openpi.training.pressure_training import configure_trainable_parameters
from openpi.training.pressure_training import load_base_weights
from openpi.training.pressure_training import pressure_policy_contract
from openpi.training.pressure_training import save_pressure_metadata
from openpi.training.pressure_training import should_save_checkpoint
from openpi.training.pressure_training import validate_pressure_metadata


class TinyPolicy(nn.Module):
    def __init__(self, *, pressure=True):
        super().__init__()
        self.paligemma_with_expert = nn.Module()
        self.paligemma_with_expert.paligemma = nn.Linear(3, 3)
        self.paligemma_with_expert.gemma_expert = nn.Linear(3, 2)
        self.action_out_proj = nn.Linear(2, 1)
        self.pressure_encoder = nn.Linear(3, 3) if pressure else None
        self.config = SimpleNamespace(pressure=PressureConfig(num_sensors=1, rows=4, cols=4) if pressure else None)

    def forward(self, values):
        if self.pressure_encoder is not None:
            values = self.pressure_encoder(values)
        values = self.paligemma_with_expert.paligemma(values)
        return self.action_out_proj(self.paligemma_with_expert.gemma_expert(values))


def test_base_migration_only_initializes_new_pressure_parameters(tmp_path):
    base = TinyPolicy(pressure=False)
    path = tmp_path / "base.safetensors"
    safetensors.torch.save_model(base, path)
    target = TinyPolicy()
    initial_pressure = target.pressure_encoder.weight.detach().clone()
    load_base_weights(target, path, allow_pressure_init=True)
    for name, parameter in base.state_dict().items():
        torch.testing.assert_close(target.state_dict()[name], parameter)
    torch.testing.assert_close(target.pressure_encoder.weight, initial_pressure)


def test_migration_rejects_missing_original_parameters(tmp_path):
    source = TinyPolicy(pressure=False).state_dict()
    source.pop("action_out_proj.weight")
    path = tmp_path / "bad.safetensors"
    safetensors.torch.save_file(source, path)
    with pytest.raises(RuntimeError, match=r"action_out_proj\.weight"):
        load_base_weights(TinyPolicy(), path, allow_pressure_init=True)


def test_migration_rejects_unexpected_parameters(tmp_path):
    source = TinyPolicy(pressure=False).state_dict()
    source["unrelated.weight"] = torch.ones(1)
    path = tmp_path / "unexpected.safetensors"
    safetensors.torch.save_file(source, path)
    with pytest.raises(RuntimeError, match=r"unrelated\.weight"):
        load_base_weights(TinyPolicy(), path, allow_pressure_init=True)


def test_shape_mismatch_is_not_allowed(tmp_path):
    source = TinyPolicy(pressure=False).state_dict()
    source["action_out_proj.weight"] = torch.ones(1, 3)
    path = tmp_path / "shape.safetensors"
    safetensors.torch.save_file(source, path)
    with pytest.raises(RuntimeError, match="size mismatch"):
        load_base_weights(TinyPolicy(), path, allow_pressure_init=True)


def test_strict_load_rejects_incomplete_vtla_checkpoint(tmp_path):
    path = tmp_path / "base.safetensors"
    safetensors.torch.save_model(TinyPolicy(pressure=False), path)
    with pytest.raises(RuntimeError):
        load_base_weights(TinyPolicy(), path)
    with pytest.raises(ValueError, match="enabled pressure_encoder"):
        load_base_weights(TinyPolicy(pressure=False), path, allow_pressure_init=True)


def test_base_initialization_rejects_partially_saved_pressure_branch(tmp_path):
    source = TinyPolicy().state_dict()
    source.pop("pressure_encoder.bias")
    path = tmp_path / "partial_pressure.safetensors"
    safetensors.torch.save_file(source, path)
    with pytest.raises(RuntimeError, match=r"pressure_encoder\.bias"):
        load_base_weights(TinyPolicy(), path, allow_pressure_init=True)


def test_complete_vtla_checkpoint_still_loads_under_initialization_entrypoint(tmp_path):
    original = TinyPolicy()
    path = tmp_path / "complete_pressure.safetensors"
    safetensors.torch.save_model(original, path)
    target = TinyPolicy()
    load_base_weights(target, path, allow_pressure_init=True)
    for name, parameter in original.state_dict().items():
        torch.testing.assert_close(target.state_dict()[name], parameter)


def test_freezing_preserves_pressure_gradients_and_optimizer_excludes_backbone():
    torch.manual_seed(7)
    model = TinyPolicy()
    parameters = configure_trainable_parameters(model, freeze_backbone=True)
    optimizer = torch.optim.AdamW(parameters, lr=0.1)
    before = model.pressure_encoder.weight.detach().clone()
    backbone_before = model.paligemma_with_expert.paligemma.weight.detach().clone()
    model(torch.randn(5, 3)).square().mean().backward()
    assert model.pressure_encoder.weight.grad is not None
    assert torch.isfinite(model.pressure_encoder.weight.grad).all()
    assert model.pressure_encoder.weight.grad.abs().sum() > 0
    assert model.paligemma_with_expert.paligemma.weight.grad is None
    assert model.paligemma_with_expert.gemma_expert.weight.grad is not None
    parameter_ids = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    assert id(model.paligemma_with_expert.paligemma.weight) not in parameter_ids
    optimizer.step()
    assert not torch.equal(model.pressure_encoder.weight, before)
    torch.testing.assert_close(model.paligemma_with_expert.paligemma.weight, backbone_before)


def test_freezing_requires_pressure_branch():
    with pytest.raises(ValueError, match="enabled pressure encoder"):
        configure_trainable_parameters(TinyPolicy(pressure=False), freeze_backbone=True)


def test_metadata_stats_roundtrip_and_contract_rejection(tmp_path):
    model = TinyPolicy()
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    stats = tmp_path / "source.npz"
    stats.write_bytes(b"normalization asset bytes")
    save_pressure_metadata(model, checkpoint, freeze_backbone=True, stats_path=stats)
    assert (checkpoint / "assets" / "pressure_stats.npz").read_bytes() == stats.read_bytes()
    validate_pressure_metadata(model, checkpoint, freeze_backbone=True, stats_path=stats)
    with pytest.raises(ValueError, match="freeze setting"):
        validate_pressure_metadata(model, checkpoint, freeze_backbone=False, stats_path=stats)
    original_config = model.config.pressure
    model.config.pressure = dataclasses.replace(original_config, history=2)
    with pytest.raises(ValueError, match="configuration differs"):
        validate_pressure_metadata(model, checkpoint, freeze_backbone=True, stats_path=stats)
    model.config.pressure = original_config
    stats.write_bytes(b"different normalization")
    with pytest.raises(ValueError, match="statistics differ"):
        validate_pressure_metadata(model, checkpoint, freeze_backbone=True, stats_path=stats)


def test_checkpoint_asset_integrity_is_checked(tmp_path):
    model = TinyPolicy()
    stats = tmp_path / "stats.npz"
    stats.write_bytes(b"fixed asset")
    save_pressure_metadata(model, tmp_path, freeze_backbone=False, stats_path=stats)
    (tmp_path / "assets" / "pressure_stats.npz").write_bytes(b"changed asset")
    with pytest.raises(ValueError, match="was modified"):
        validate_pressure_metadata(model, tmp_path, freeze_backbone=False, stats_path=stats)


def test_legacy_checkpoints_allowed_only_without_pressure(tmp_path):
    validate_pressure_metadata(TinyPolicy(pressure=False), tmp_path, freeze_backbone=False)
    with pytest.raises(ValueError, match=r"requires pressure_config\.json"):
        validate_pressure_metadata(TinyPolicy(), tmp_path, freeze_backbone=False)


def test_metadata_without_stats_and_full_weight_roundtrip(tmp_path):
    original = TinyPolicy()
    path = tmp_path / "complete.safetensors"
    safetensors.torch.save_model(original, path)
    target = TinyPolicy()
    load_base_weights(target, path)
    save_pressure_metadata(target, tmp_path, freeze_backbone=False)
    validate_pressure_metadata(target, tmp_path, freeze_backbone=False)
    sample = torch.randn(2, 3)
    torch.testing.assert_close(target(sample), original(sample))


@pytest.mark.parametrize(
    ("step", "total", "interval", "expected"),
    [(0, 1, 100, False), (1, 1, 100, True), (3, 4, 100, False), (4, 4, 100, True), (10, 25, 10, True)],
)
def test_checkpoint_schedule_includes_the_final_optimizer_update(step, total, interval, expected):
    assert should_save_checkpoint(step, total, interval) is expected


def test_checkpoint_schedule_rejects_nonpositive_interval():
    with pytest.raises(ValueError, match="positive"):
        should_save_checkpoint(1, 1, 0)


def _contract():
    model_config = SimpleNamespace(action_dim=32, action_horizon=10, max_token_len=200, discrete_state_input=True)
    data_config = SimpleNamespace(robot_action_dim=7, use_quantile_norm=True, asset_id="robot/data")
    return pressure_policy_contract(model_config, data_config)


def test_real_checkpoint_contract_rejects_action_and_normalization_drift(tmp_path):
    model = TinyPolicy()
    pressure_stats = tmp_path / "input-pressure.npz"
    pressure_stats.write_bytes(b"fixed pressure stats")
    asset = tmp_path / "checkpoint" / "assets" / "robot" / "data" / "norm_stats.json"
    asset.parent.mkdir(parents=True)
    content = {"norm_stats": {"state": {"mean": [0.0]}, "actions": {"mean": [1.0]}}}
    asset.write_text(json.dumps(content, indent=2), encoding="utf-8")
    contract = _contract()
    save_pressure_metadata(
        model,
        asset.parents[3],
        freeze_backbone=True,
        stats_path=pressure_stats,
        policy_contract=contract,
        norm_stats_path=asset,
    )
    checkpoint = asset.parents[3]
    validate_pressure_metadata(
        model,
        checkpoint,
        freeze_backbone=True,
        stats_path=pressure_stats,
        policy_contract=contract,
        current_norm_stats_json=json.dumps(content, separators=(",", ":")),
    )
    with pytest.raises(ValueError, match="contract differs"):
        validate_pressure_metadata(
            model,
            checkpoint,
            freeze_backbone=True,
            stats_path=pressure_stats,
            policy_contract={**contract, "robot_action_dim": 8},
        )
    with pytest.raises(ValueError, match="Current state/action"):
        validate_pressure_metadata(
            model,
            checkpoint,
            freeze_backbone=True,
            stats_path=pressure_stats,
            policy_contract=contract,
            current_norm_stats_json=json.dumps({"different": True}),
        )
    asset.write_text(json.dumps({"different": True}), encoding="utf-8")
    with pytest.raises(ValueError, match="Checkpoint state/action"):
        validate_pressure_metadata(
            model,
            checkpoint,
            freeze_backbone=True,
            stats_path=pressure_stats,
            policy_contract=contract,
        )


def test_legacy_checkpoint_loads_with_explicit_contract_warning(tmp_path, caplog):
    model = TinyPolicy()
    save_pressure_metadata(model, tmp_path, freeze_backbone=False)
    validate_pressure_metadata(model, tmp_path, freeze_backbone=False, policy_contract=_contract())
    assert "no action or state/action normalization contract" in caplog.text


def test_real_contract_requires_pressure_and_state_action_assets(tmp_path):
    model = TinyPolicy()
    with pytest.raises(ValueError, match="pressure normalization"):
        save_pressure_metadata(model, tmp_path, freeze_backbone=False, policy_contract=_contract())
    pressure_stats = tmp_path / "pressure.npz"
    pressure_stats.write_bytes(b"pressure")
    with pytest.raises(ValueError, match="state/action normalization"):
        save_pressure_metadata(
            model,
            tmp_path,
            freeze_backbone=False,
            policy_contract=_contract(),
            stats_path=pressure_stats,
        )
