"""Model-side forearm geometry and JSON/checkpoint contracts."""

import dataclasses
import json
from types import SimpleNamespace

import pytest

from openpi.models.pressure_config import PressureConfig
from openpi.models.pressure_config import resolve_pressure_config
from openpi.training.pressure_training import save_pressure_metadata
from openpi.training.pressure_training import validate_pressure_metadata


def test_json_layout_roundtrip_and_token_budget(tmp_path):
    content = {
        "num_sensors": 1,
        "rows": 1,
        "cols": 2,
        "patch_rows": 1,
        "patch_cols": 1,
        "history": 3,
        "sensor_names": ["forearm_skin"],
        "sensor_locations": ["forearm"],
        "signal_type": "capacitance",
        "signal_unit": "pF",
        "sample_period_seconds": 0.01,
        "use_temporal_differences": True,
        "max_difference_gap_seconds": 0.05,
        "taxel_positions_mm": [[[0, 0, 0], [10, 0, 2]]],
    }
    path = tmp_path / "forearm.json"
    path.write_text(json.dumps(content))
    config = PressureConfig.from_json(path)
    assert config.num_tokens == 6
    assert config.taxel_positions_mm == (((0.0, 0.0, 0.0), (10.0, 0.0, 2.0)),)
    assert PressureConfig.from_dict(config.to_dict()) == config
    assert config.to_dict()["taxel_positions_mm"] == [[[0.0, 0.0, 0.0], [10.0, 0.0, 2.0]]]


@pytest.mark.parametrize(
    "updates",
    [
        {"taxel_positions_mm": [[[0, 0, 0]]]},
        {"taxel_positions_mm": [[[0, 0], [1, 0]]]},
        {"taxel_positions_mm": [[[0, 0, 0], [1, float("nan"), 0]]]},
        {"taxel_positions_mm": [[[0, 0, 0], [1, float("inf"), 0]]]},
        {"coordinate_scale_mm": 0},
        {"coordinate_scale_mm": None},
        {"coordinate_scale_mm": float("inf")},
        {"sample_period_seconds": -0.01},
        {"sample_period_seconds": True},
        {"signal_type": "unknown"},
        {"signal_unit": " "},
        {"use_temporal_differences": "false"},
        {"use_temporal_differences": True, "history": 1},
        {"max_difference_gap_seconds": 0},
        {"max_difference_gap_seconds": float("nan")},
    ],
)
def test_invalid_model_contract_rejected(updates):
    with pytest.raises(
        ValueError, match=r"taxel_positions|coordinate_scale|sample_period|signal_|differences|difference_gap"
    ):
        PressureConfig(num_sensors=1, rows=1, cols=2, patch_rows=1, patch_cols=1, **updates)


def test_json_rejects_unknown_fields_and_non_object(tmp_path):
    path = tmp_path / "bad.json"
    for value in ({"num_sensors": 1, "rows": 4, "cols": 4, "historry": 3}, []):
        path.write_text(json.dumps(value))
        with pytest.raises(ValueError, match=r"Unknown pressure|JSON object"):
            PressureConfig.from_json(path)


def test_nested_geometry_survives_checkpoint_and_semantic_change_is_rejected(tmp_path):
    config = PressureConfig(
        num_sensors=1,
        rows=1,
        cols=2,
        patch_rows=1,
        patch_cols=1,
        taxel_positions_mm=(((0, 0, 0), (10, 0, 0)),),
        signal_type="capacitance",
        signal_unit="pF",
    )
    model = SimpleNamespace(config=SimpleNamespace(pressure=config))
    save_pressure_metadata(model, tmp_path, freeze_backbone=False)
    validate_pressure_metadata(model, tmp_path, freeze_backbone=False)
    model.config.pressure = dataclasses.replace(config, signal_unit="nF")
    with pytest.raises(ValueError, match="configuration differs"):
        validate_pressure_metadata(model, tmp_path, freeze_backbone=False)


def test_old_checkpoint_defaults_remain_compatible(tmp_path):
    model = SimpleNamespace(config=SimpleNamespace(pressure=PressureConfig(num_sensors=1, rows=4, cols=4)))
    old_fields = {
        "num_sensors",
        "rows",
        "cols",
        "sensor_names",
        "sensor_locations",
        "history",
        "patch_rows",
        "patch_cols",
        "hidden_dim",
        "max_age_seconds",
    }
    legacy = {k: v for k, v in dataclasses.asdict(model.config.pressure).items() if k in old_fields}
    (tmp_path / "pressure_config.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "pressure": legacy,
                "freeze_backbone": False,
                "stats_sha256": None,
            }
        )
    )
    validate_pressure_metadata(model, tmp_path, freeze_backbone=False)


@dataclasses.dataclass(frozen=True)
class ModelSettings:
    pressure: PressureConfig | None = None


def test_external_layout_replaces_only_pressure_and_can_be_deleted(tmp_path):
    path = tmp_path / "layout.json"
    path.write_text(
        json.dumps(
            {
                "num_sensors": 1,
                "rows": 4,
                "cols": 4,
                "sensor_names": ["forearm_skin"],
                "sensor_locations": ["forearm"],
            }
        )
    )
    original = ModelSettings(PressureConfig(num_sensors=2, rows=8, cols=8))
    resolved = resolve_pressure_config(original, path)
    path.unlink()
    assert resolved.pressure.num_sensors == 1
    assert resolved.pressure.sensor_locations == ("forearm",)
    assert original.pressure.num_sensors == 2
    assert resolve_pressure_config(resolved, None) is resolved


def test_external_layout_rejects_models_without_a_pressure_branch(tmp_path):
    path = tmp_path / "layout.json"
    path.write_text('{"num_sensors":1,"rows":4,"cols":4}')
    with pytest.raises(ValueError, match="pressure-enabled"):
        resolve_pressure_config(ModelSettings(), path)


def test_geometry_rejects_scale_that_would_overflow_float32():
    with pytest.raises(ValueError, match="taxel_positions"):
        PressureConfig(
            num_sensors=1,
            rows=1,
            cols=1,
            patch_rows=1,
            patch_cols=1,
            taxel_positions_mm=(((1, 0, 0),),),
            coordinate_scale_mm=1e-300,
        )


@pytest.mark.parametrize("name", ["sensor_names", "sensor_locations"])
def test_json_does_not_turn_a_single_name_string_into_sensor_slots(name):
    with pytest.raises(ValueError, match=name):
        PressureConfig.from_dict({"num_sensors": 2, "rows": 4, "cols": 4, name: "ab"})
