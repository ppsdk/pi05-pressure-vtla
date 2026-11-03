"""Forearm numeric tokens retain contact strength, layout and causal changes."""

import dataclasses

import pytest
import torch
from torch.nn import functional as F  # noqa: N812

from openpi.models.pressure_config import PressureConfig
from openpi.models_pytorch.pressure_pytorch import PressureEncoder


def sequence(values, ages):
    values = torch.tensor(values, dtype=torch.float32).reshape(1, 1, len(ages), 1, -1)
    return {
        "values": values,
        "frame_valid": torch.ones(values.shape[:3], dtype=torch.bool),
        "taxel_valid": torch.ones_like(values, dtype=torch.bool),
        "saturated": torch.zeros_like(values, dtype=torch.bool),
        "age": torch.tensor(ages).reshape(1, 1, -1),
    }


def transparent_encoder(**kwargs):
    """Expose input channels using controlled weights in the real encoder."""
    cfg = PressureConfig(
        num_sensors=1,
        rows=1,
        cols=1,
        patch_rows=1,
        patch_cols=1,
        history=3,
        hidden_dim=5,
        use_temporal_differences=True,
        **kwargs,
    )
    model = PressureEncoder(cfg, 5)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.patch_in.weight.copy_(torch.eye(5))
        model.patch_out.weight.copy_(torch.eye(5))
    return model


def test_difference_preserves_absolute_contact_and_changes_only_later_tokens():
    model = transparent_encoder()
    pressure = sequence([2, 5, 5], [0.2, 0.1, 0.0])
    tokens, valid = model(pressure)
    expected = F.silu(torch.tensor([[2, 1, 0, 0, 0], [5, 1, 0, 3, 1], [5, 1, 0, 0, 1]]).float())
    torch.testing.assert_close(tokens[0], expected)
    assert valid.all()
    pressure["values"][0, 0, 2, 0, 0] = 9
    altered, _ = model(pressure)
    torch.testing.assert_close(tokens[:, :2], altered[:, :2])
    assert not torch.equal(tokens[:, 2], altered[:, 2])


@pytest.mark.parametrize(
    "reason", ["missing_frame", "bad_taxel", "saturated", "stale", "nan", "equal", "reversed", "gap"]
)
def test_unreliable_history_never_creates_a_contact_change(reason):
    model = transparent_encoder(max_age_seconds=0.3, max_difference_gap_seconds=0.15)
    pressure = sequence([2, 5, 9], [0.2, 0.1, 0.0])
    if reason == "missing_frame":
        pressure["frame_valid"][0, 0, 0] = False
    elif reason == "bad_taxel":
        pressure["taxel_valid"][0, 0, 0] = False
    elif reason == "saturated":
        pressure["saturated"][0, 0, 0] = True
    elif reason == "stale":
        pressure["age"][0, 0, 0] = 0.4
    elif reason == "nan":
        pressure["values"][0, 0, 0] = float("nan")
    elif reason == "equal":
        pressure["age"][0, 0, 0] = 0.1
    elif reason == "reversed":
        pressure["age"][0, 0, 0] = 0.05
    else:
        pressure["age"][0, 0, 0] = 0.29
    tokens, valid = model(pressure)
    assert tokens[0, 1, 3] == 0
    assert tokens[0, 1, 4] == 0
    assert tokens[0, 1, 0] > 0  # Current contact remains available.
    assert valid[0, 1]
    assert torch.isfinite(tokens).all()
    torch.testing.assert_close(tokens[0, 2, 3:], F.silu(torch.tensor([4.0, 1.0])))


def test_difference_does_not_use_invalid_values_and_has_finite_gradients():
    model = transparent_encoder()
    pressure = sequence([float("nan"), 5, 9], [0.2, 0.1, 0.0])
    pressure["values"].requires_grad_()
    tokens, valid = model(pressure)
    tokens[valid].sum().backward()
    grad = pressure["values"].grad
    assert torch.isfinite(grad).all()
    assert grad[0, 0, 0, 0, 0] == 0
    assert grad[0, 0, 1:].abs().sum() > 0


def test_forearm_coordinates_affect_only_corresponding_patch_and_survive_reload():
    torch.manual_seed(7)
    cfg = PressureConfig(
        num_sensors=1,
        rows=1,
        cols=2,
        patch_rows=1,
        patch_cols=1,
        taxel_positions_mm=(((0, 0, 0), (20, 0, 0)),),
    )
    model = PressureEncoder(cfg, 8)
    pressure = sequence([0, 0], [0.0])
    original, valid = model(pressure)
    changed_cfg = dataclasses.replace(cfg, taxel_positions_mm=(((0, 0, 0), (20, 0, 15)),))
    changed = PressureEncoder(changed_cfg, 8)
    changed.load_state_dict(model.state_dict())
    tokens, _ = changed(pressure)
    torch.testing.assert_close(tokens[:, 0], original[:, 0])
    assert not torch.allclose(tokens[:, 1], original[:, 1])
    restored = PressureEncoder(PressureConfig.from_dict(cfg.to_dict()), 8)
    restored.load_state_dict(model.state_dict())
    torch.testing.assert_close(restored(pressure)[0], original)
    assert valid.all()  # No contact is still an observed state.
    original.sum().backward()
    assert model.coordinate_in.weight.grad.abs().sum() > 0


def test_coordinate_scale_preserves_equivalent_physical_layout():
    cfg = PressureConfig(
        num_sensors=1,
        rows=1,
        cols=2,
        patch_rows=1,
        patch_cols=1,
        taxel_positions_mm=(((0, 0, 0), (20, 0, 10)),),
        coordinate_scale_mm=100,
    )
    model = PressureEncoder(cfg, 8)
    other = PressureEncoder(
        dataclasses.replace(
            cfg,
            taxel_positions_mm=(((0, 0, 0), (40, 0, 20)),),
            coordinate_scale_mm=200,
        ),
        8,
    )
    other.load_state_dict(model.state_dict())
    pressure = sequence([1, 2], [0.0])
    torch.testing.assert_close(other(pressure)[0], model(pressure)[0])


def test_multiple_sensors_keep_all_patch_coordinates_in_row_major_order():
    cfg = PressureConfig(
        num_sensors=2,
        rows=2,
        cols=2,
        patch_rows=2,
        patch_cols=2,
        history=1,
        taxel_positions_mm=(
            ((0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)),
            ((10, 0, 0), (11, 0, 0), (10, 1, 0), (11, 1, 0)),
        ),
        coordinate_scale_mm=1,
    )
    model = PressureEncoder(cfg, 8)
    assert model.patch_coordinates.shape == (1, 2, 1, 1, 12)
    torch.testing.assert_close(
        model.patch_coordinates[0, 1, 0, 0],
        torch.tensor([10, 0, 0, 11, 0, 0, 10, 1, 0, 11, 1, 0], dtype=torch.float32),
    )
