"""CPU contracts using small random upstream bundles, not trained-policy results."""

import dataclasses
import json
from types import SimpleNamespace

import numpy as np
import pytest
import safetensors.torch
import torch
from torch import nn

from openpi.models.pressure_config import LingBotTouchConfig
from openpi.models.pressure_config import PressureConfig
from openpi.models_pytorch.lingbot_forearm import LingBotForearmEncoder
from openpi.models_pytorch.lingbot_forearm import initialize_lingbot_forearm
from openpi.models_pytorch.lingbot_touch import EncoderConfig
from openpi.models_pytorch.lingbot_touch import TargetEncoder
from openpi.training.pressure_training import configure_trainable_parameters
from openpi.training.pressure_training import save_pressure_metadata
from openpi.training.pressure_training import validate_finetune_checkpoint
from openpi.training.pressure_training import validate_pressure_metadata


@pytest.fixture
def setup(tmp_path):
    torch.manual_seed(71)
    architecture = LingBotTouchConfig(sensor_zero=(0,), sensor_span=(1,), time_scale_seconds=0.01).encoder_config
    architecture.update(embed_dim=40, depth=2, num_heads=2, output_layers=[0, 1], mlp_ratio=2)
    config = PressureConfig(
        num_sensors=1,
        rows=1,
        cols=3,
        history=4,
        patch_rows=1,
        patch_cols=1,
        taxel_positions_mm=(((0, 0, 0), (10, 0, 0), (40, 5, 0)),),
        max_age_seconds=0.1,
        use_temporal_differences=True,
        lingbot_touch=LingBotTouchConfig(
            sensor_zero=(0,), sensor_span=(1,), time_scale_seconds=0.01, encoder_config=architecture, readout_dim=12
        ),
    )
    reference = TargetEncoder(EncoderConfig.from_dict(architecture))
    (tmp_path / "config.json").write_text(json.dumps(architecture))
    torch.save(reference.state_dict(), tmp_path / "encoder.pt")
    stats_path = tmp_path / "stats.npz"
    np.savez(stats_path, mean=np.array(0), scale=np.array(1))
    encoder = LingBotForearmEncoder(config, output_dim=16, state_dim=3)
    encoder.initialize_from_bundle(tmp_path)
    encoder.bind_normalization(stats_path)
    pressure = {
        "values": torch.rand(2, 1, 4, 1, 3),
        "frame_valid": torch.ones(2, 1, 4, dtype=torch.bool),
        "taxel_valid": torch.ones(2, 1, 4, 1, 3, dtype=torch.bool),
        "saturated": torch.zeros(2, 1, 4, 1, 3, dtype=torch.bool),
        "age": torch.tensor([0.03, 0.02, 0.01, 0.0])[None, None].expand(2, 1, 4).clone(),
    }
    return SimpleNamespace(
        encoder=encoder,
        config=config,
        reference=reference,
        pressure=pressure,
        directory=tmp_path,
        stats_path=stats_path,
    )


def test_pretrained_reference_parity_without_forearm_offsets(setup):
    encoder, reference, pressure = setup.encoder, setup.reference, setup.pressure
    nn.init.zeros_(encoder.sensor_embedding.weight)
    signal = torch.zeros(2, 4, 5, 3, 3)
    signal[:, :, 0, :, 2] = pressure["values"][:, 0, :, 0]
    xyz = torch.zeros(5, 3, 3)
    xyz[0] = encoder.positions[0]
    coverage = torch.zeros(5, 3, dtype=torch.bool)
    coverage[0] = True
    expected = encoder.output_projection(reference(signal, xyz, coverage, modality="piezoresistive").latent)
    actual, valid = encoder(pressure)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    assert valid.all()
    assert actual.shape == (2, setup.config.num_tokens, 16)


def test_location_swap_is_observable_after_attention_readout(setup):
    encoder, pressure = setup.encoder, setup.pressure
    state = torch.zeros(2, 3)
    before, valid = encoder(pressure)
    pressure["values"] = pressure["values"].flip(-1)
    after, after_valid = encoder(pressure)
    assert not torch.allclose(encoder.pool(before, valid, state), encoder.pool(after, after_valid, state))


def test_dynamic_mask_prevents_invalid_values_from_influencing_other_tokens(setup):
    encoder, pressure = setup.encoder, setup.pressure
    pressure["taxel_valid"][0, 0, 0, 0, 1] = False
    baseline, valid = encoder(pressure)
    pressure["values"][0, 0, 0, 0, 1] = float("nan")
    pressure["values"][0, 0, 1, 0, 1] = 1e30  # entire partial tubelet is excluded
    pressure["saturated"][0, 0, :2, 0, 1] = True
    actual, actual_valid = encoder(pressure)
    torch.testing.assert_close(actual, baseline)
    torch.testing.assert_close(actual_valid, valid)
    assert not valid[0, 1]


@pytest.mark.parametrize("age", [-0.1, float("nan"), float("inf"), 0.2])
def test_stale_future_and_nonfinite_frames_excluded(setup, age):
    setup.pressure["age"][0, 0, 0] = age
    tokens, valid = setup.encoder(setup.pressure)
    assert not valid[0, :3].any()
    assert valid[0, 3:].all()
    assert torch.isfinite(tokens).all()


def test_no_contact_valid_all_missing_zero_and_backward_finite(setup):
    encoder, pressure = setup.encoder, setup.pressure
    pressure["values"].zero_()
    tokens, valid = encoder(pressure)
    assert valid.all()
    pressure["frame_valid"].zero_()
    tokens, valid = encoder(pressure)
    pooled = encoder.pool(tokens, valid, torch.zeros(2, 3))
    assert not valid.any()
    assert torch.count_nonzero(tokens) == torch.count_nonzero(pooled) == 0
    pooled.sum().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in encoder.parameters())


def test_frozen_weights_preserve_adapter_gradient_and_state_conditioning(setup):
    encoder = setup.encoder
    state = torch.randn(2, 3, requires_grad=True)
    tokens, valid = encoder(setup.pressure)
    encoder.pool(tokens, valid, state).square().mean().backward()
    assert all(p.grad is None for p in encoder.pretrained.parameters())
    assert encoder.sensor_embedding.weight.grad.abs().sum() > 0
    assert encoder.quality_adapter[-1].weight.grad.abs().sum() > 0
    assert encoder.state_query.weight.grad.abs().sum() > 0
    assert state.grad.abs().sum() > 0


def test_normalization_roundtrip_and_signed_resistance_calibration(setup):
    raw = setup.pressure["values"].clone()
    baseline, _ = setup.encoder(setup.pressure)
    np.savez(setup.stats_path, mean=np.array(0.4), scale=np.array(0.2))
    setup.encoder.bind_normalization(setup.stats_path)
    setup.pressure["values"] = (raw - 0.4) / 0.2
    actual, _ = setup.encoder(setup.pressure)
    torch.testing.assert_close(actual, baseline)
    reverse = dataclasses.replace(
        setup.config, lingbot_touch=dataclasses.replace(setup.config.lingbot_touch, sensor_zero=(1,), sensor_span=(-1,))
    )
    encoder = LingBotForearmEncoder(reverse, 16, 3)
    encoder.load_state_dict(setup.encoder.state_dict())
    setup.pressure["values"] = ((1 - raw) - 0.4) / 0.2
    actual, _ = encoder(setup.pressure)
    torch.testing.assert_close(actual, baseline)


def test_checkpoint_is_self_contained_and_contract_changes_rejected(setup):
    model = nn.Module()
    model.pressure_encoder = setup.encoder
    model.config = SimpleNamespace(pressure=setup.config)
    path = setup.directory / "model.safetensors"
    safetensors.torch.save_model(model, path)
    save_pressure_metadata(model, setup.directory, freeze_backbone=False, stats_path=setup.stats_path)
    restored = nn.Module()
    restored.pressure_encoder = LingBotForearmEncoder(setup.config, 16, 3)
    restored.config = model.config
    safetensors.torch.load_model(restored, path)
    initialize_lingbot_forearm(restored)
    torch.testing.assert_close(restored.pressure_encoder(setup.pressure)[0], setup.encoder(setup.pressure)[0])
    validate_pressure_metadata(restored, setup.directory, freeze_backbone=False, stats_path=setup.stats_path)
    restored.config = SimpleNamespace(
        pressure=dataclasses.replace(
            setup.config, lingbot_touch=dataclasses.replace(setup.config.lingbot_touch, sensor_span=(2,))
        )
    )
    with pytest.raises(ValueError, match="configuration differs"):
        validate_pressure_metadata(restored, setup.directory, freeze_backbone=False, stats_path=setup.stats_path)


def test_uninitialized_and_mismatched_bundles_fail_closed(setup):
    fresh = LingBotForearmEncoder(setup.config, 16, 3)
    with pytest.raises(RuntimeError, match="official bundle"):
        fresh(setup.pressure)
    state = torch.load(setup.directory / "encoder.pt", weights_only=True)
    state.pop(next(iter(state)))
    torch.save(state, setup.directory / "encoder.pt")
    with pytest.raises(ValueError, match="exactly"):
        fresh.initialize_from_bundle(setup.directory)


def test_config_roundtrip_and_geometry_requirement(setup):
    assert PressureConfig.from_dict(setup.config.to_dict()) == setup.config
    with pytest.raises(ValueError, match="real taxel_positions"):
        dataclasses.replace(setup.config, taxel_positions_mm=())
    with pytest.raises(ValueError, match="divisible"):
        dataclasses.replace(setup.config, history=3)


def test_base_freeze_keeps_pretrained_freeze_and_readout_trainable(setup):
    model = nn.Module()
    model.pressure_encoder = setup.encoder
    model.tactile_residual_adapter = nn.Linear(16, 3)
    model.base = nn.Linear(3, 3)
    configure_trainable_parameters(model, freeze_base_policy=True)
    assert not model.base.weight.requires_grad
    assert not any(p.requires_grad for p in setup.encoder.pretrained.parameters())
    assert setup.encoder.readout_queries.requires_grad


def test_stage_transition_changes_freezing_but_preserves_sensor_contract(setup):
    model = nn.Module()
    model.pressure_encoder = setup.encoder
    model.config = SimpleNamespace(pressure=setup.config)
    safetensors.torch.save_model(model, setup.directory / "model.safetensors")
    save_pressure_metadata(model, setup.directory, freeze_backbone=True, stats_path=setup.stats_path)
    stage2 = dataclasses.replace(
        setup.config, lingbot_touch=dataclasses.replace(setup.config.lingbot_touch, freeze_pretrained=False)
    )
    model.config = SimpleNamespace(pressure=stage2)
    contract = {"freeze_backbone": False, "stats_path": setup.stats_path}
    validate_finetune_checkpoint(model, setup.directory, **contract)
    assert not model.config.pressure.lingbot_touch.freeze_pretrained
    with pytest.raises(ValueError, match="configuration differs"):
        validate_pressure_metadata(model, setup.directory, **contract)
    with pytest.raises(ValueError, match="cannot be combined"):
        validate_finetune_checkpoint(model, setup.directory, bundle_path=setup.directory, **contract)
    model.config = SimpleNamespace(pressure=dataclasses.replace(stage2, coordinate_scale_mm=50))
    with pytest.raises(ValueError, match="configuration differs"):
        validate_finetune_checkpoint(model, setup.directory, **contract)
    model.config = SimpleNamespace(pressure=stage2)
    np.savez(setup.stats_path, mean=np.array(5), scale=np.array(1))
    with pytest.raises(ValueError, match="statistics differ"):
        validate_finetune_checkpoint(model, setup.directory, **contract)


def test_loaded_encoder_cannot_be_reinitialized_from_bundle(setup):
    model = nn.Module()
    model.pressure_encoder = setup.encoder
    with pytest.raises(ValueError, match="already initialized"):
        initialize_lingbot_forearm(model, bundle_path=setup.directory)
