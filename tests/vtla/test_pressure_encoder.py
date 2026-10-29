"""CPU-only pressure tests; importing these modules does not require JAX or Flax."""

import importlib.util
from pathlib import Path
import sys

import pytest
import torch


def _load_source_module(name, relative_path):
    path = Path(__file__).resolve().parents[2] / "src" / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


PressureConfig = _load_source_module("_vtla_pressure_config", "openpi/models/pressure_config.py").PressureConfig
PressureEncoder = _load_source_module(
    "_vtla_pressure_encoder", "openpi/models_pytorch/pressure_pytorch.py"
).PressureEncoder


@pytest.fixture
def encoder():
    torch.manual_seed(11)
    config = PressureConfig(
        num_sensors=2,
        rows=4,
        cols=6,
        history=3,
        patch_rows=2,
        patch_cols=3,
        hidden_dim=12,
        max_age_seconds=0.5,
    )
    return PressureEncoder(config, output_dim=16)


@pytest.fixture
def pressure():
    generator = torch.Generator().manual_seed(37)
    return {
        "values": torch.rand(2, 2, 3, 4, 6, generator=generator),
        "frame_valid": torch.ones(2, 2, 3, dtype=torch.bool),
        "taxel_valid": torch.ones(2, 2, 3, 4, 6, dtype=torch.bool),
        "saturated": torch.zeros(2, 2, 3, 4, 6, dtype=torch.bool),
        "age": torch.full((2, 2, 3), 0.1),
    }


def test_shape_dtype_and_zero_contact_are_valid(encoder, pressure):
    pressure["values"] = torch.zeros_like(pressure["values"], dtype=torch.float64)
    tokens, valid = encoder(pressure)
    assert tokens.shape == (2, 24, 16)
    assert tokens.dtype == torch.float32
    assert valid.shape == (2, 24)
    assert valid.dtype == torch.bool
    assert valid.all()
    assert torch.isfinite(tokens).all()


def test_frame_and_patch_missingness_are_distinct(encoder, pressure):
    pressure["frame_valid"][0, 1, 1] = False
    pressure["taxel_valid"][0, 0, 0, :2, :3] = False
    # A partial patch is still measured; a zero-valued valid taxel is not missing.
    pressure["taxel_valid"][0, 0, 0, 0, 3] = False
    pressure["values"][0, 0, 0, 0, 4] = 0.0
    tokens, valid = encoder(pressure)
    expected = torch.ones(2, 24, dtype=torch.bool)
    expected[0, 0] = False
    expected[0, 16:20] = False
    torch.testing.assert_close(valid, expected)
    assert torch.count_nonzero(tokens[~valid]) == 0


def test_saturation_remains_valid_and_reaches_tokens(encoder, pressure):
    original_tokens, original_valid = encoder(pressure)
    pressure["saturated"][0, 0, 0, 0, 0] = True
    tokens, valid = encoder(pressure)
    torch.testing.assert_close(valid, original_valid)
    assert not torch.allclose(tokens[0, 0], original_tokens[0, 0])
    torch.testing.assert_close(tokens[0, 1:], original_tokens[0, 1:])


def test_invalid_taxel_values_cannot_affect_output(encoder, pressure):
    pressure["taxel_valid"][0, 0, 0, 0, :3] = False
    original_tokens, original_valid = encoder(pressure)
    pressure["values"][0, 0, 0, 0, :3] = torch.tensor([float("nan"), float("inf"), -float("inf")])
    pressure["saturated"][0, 0, 0, 0, :3] = True
    tokens, valid = encoder(pressure)
    torch.testing.assert_close(tokens, original_tokens)
    torch.testing.assert_close(valid, original_valid)
    assert torch.isfinite(tokens).all()


def test_nonfinite_measured_taxels_become_unavailable(encoder, pressure):
    pressure["values"][0, 0, 0, :2, :3] = float("nan")
    pressure["values"][0, 0, 0, 0, 3] = float("inf")
    tokens, valid = encoder(pressure)
    assert not valid[0, 0]
    assert valid[0, 1]
    assert torch.isfinite(tokens).all()
    assert torch.count_nonzero(tokens[0, 0]) == 0


@pytest.mark.parametrize("invalid_age", [-0.1, float("nan"), float("inf"), 0.5001])
def test_invalid_or_stale_age_masks_entire_frame(encoder, pressure, invalid_age):
    pressure["age"][0, 0, 1] = invalid_age
    tokens, valid = encoder(pressure)
    assert not valid[0, 4:8].any()
    assert valid[0, :4].all()
    assert torch.isfinite(tokens).all()
    assert torch.count_nonzero(tokens[0, 4:8]) == 0


def test_age_at_limit_is_valid_and_actual_age_changes_embedding(encoder, pressure):
    original_tokens, _ = encoder(pressure)
    pressure["age"][0, 0, 0] = 0.5
    tokens, valid = encoder(pressure)
    assert valid.all()
    assert not torch.allclose(tokens[0, :4], original_tokens[0, :4])
    torch.testing.assert_close(tokens[0, 4:], original_tokens[0, 4:])


def test_pressure_change_stays_in_its_spatial_patch(encoder, pressure):
    original_tokens, _ = encoder(pressure)
    pressure["values"][0, 0, 0, 2, 3] += 0.25
    tokens, _ = encoder(pressure)
    changed = (tokens - original_tokens).abs().amax(dim=-1) > 1e-6
    expected = torch.zeros_like(changed)
    expected[0, 3] = True
    torch.testing.assert_close(changed, expected)


def test_parameters_and_valid_pressure_receive_finite_gradients(encoder, pressure):
    pressure["values"].requires_grad_()
    pressure["taxel_valid"][0, 0, 0, 0, 0] = False
    tokens, valid = encoder(pressure)
    tokens[valid].square().mean().backward()
    for name, parameter in encoder.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert torch.count_nonzero(parameter.grad) > 0, name
    value_grad = pressure["values"].grad
    assert torch.isfinite(value_grad).all()
    assert value_grad[0, 0, 0, 0, 0] == 0
    assert torch.count_nonzero(value_grad) > 0


def test_batch_samples_do_not_share_normalization_statistics(encoder, pressure):
    tokens, valid = encoder(pressure)
    for index in range(2):
        sample = {key: value[index : index + 1] for key, value in pressure.items()}
        sample_tokens, sample_valid = encoder(sample)
        torch.testing.assert_close(sample_tokens[0], tokens[index])
        torch.testing.assert_close(sample_valid[0], valid[index])


def test_float32_output_under_autocast(encoder, pressure):
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        tokens, _ = encoder(pressure)
    assert tokens.dtype == torch.float32
    assert torch.isfinite(tokens).all()


@pytest.mark.parametrize("field", ["values", "frame_valid", "taxel_valid", "saturated", "age"])
def test_incorrect_shapes_are_rejected(encoder, pressure, field):
    pressure[field] = pressure[field][..., 0]
    with pytest.raises(ValueError, match="shape"):
        encoder(pressure)


@pytest.mark.parametrize("field", ["frame_valid", "taxel_valid", "saturated"])
def test_numeric_validity_masks_are_rejected(encoder, pressure, field):
    pressure[field] = pressure[field].float()
    with pytest.raises(TypeError, match=r"torch\.bool"):
        encoder(pressure)


def test_missing_age_is_rejected(encoder, pressure):
    del pressure["age"]
    with pytest.raises(ValueError, match="missing fields"):
        encoder(pressure)
