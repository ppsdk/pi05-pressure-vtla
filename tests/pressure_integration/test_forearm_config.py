"""Resolve external forearm layouts through real training and CLI configuration."""

import dataclasses
import json
import sys

import pytest

pytest.importorskip("jax")
pytest.importorskip("flax")

from openpi.training import config as training_config


@pytest.fixture
def layout(tmp_path):
    path = tmp_path / "layout.json"
    path.write_text(
        json.dumps(
            {
                "num_sensors": 1,
                "rows": 2,
                "cols": 4,
                "patch_rows": 1,
                "patch_cols": 2,
                "history": 3,
                "sensor_names": ["forearm_skin"],
                "sensor_locations": ["forearm"],
                "signal_type": "capacitance",
                "signal_unit": "pF",
                "use_temporal_differences": True,
                "taxel_positions_mm": [[[i * 10, 0, 0] for i in range(8)]],
            }
        )
    )
    return path


def test_forearm_json_resolved_in_get_config_and_survives_deleting_source(layout):
    config = training_config.get_config("pi05_pressure_forearm", pressure_config_path=str(layout))
    assert config.pressure_config_path is None
    assert config.model.pressure.num_tokens == 12
    assert config.model.pressure.sensor_locations == ("forearm",)
    assert config.model.pressure.signal_unit == "pF"
    layout.unlink()
    resumed = dataclasses.replace(config, exp_name="resume", resume=True)
    assert resumed.model.pressure == config.model.pressure
    # Registry defaults have not been changed by external configuration.
    assert training_config.get_config("pi05_pressure_forearm").model.pressure.rows == 4


def test_forearm_json_resolved_through_training_cli(layout, monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_pytorch.py",
            "pi05_pressure_forearm",
            "--exp-name",
            "forearm_smoke",
            "--pressure-config-path",
            str(layout),
        ],
    )
    config = training_config.cli()
    assert config.exp_name == "forearm_smoke"
    assert config.pressure_config_path is None
    assert config.model.pressure.cols == 4
    assert config.model.pressure.rows == 2
    assert config.model.pressure.num_tokens == 12


def test_json_rejects_non_pressure_training_config(layout):
    with pytest.raises(ValueError, match="pressure-enabled"):
        training_config.get_config("debug_pi05", pressure_config_path=str(layout))
