"""Real-file contract tests; require NumPy and pytest, without a GPU/model stack."""

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from openpi.pressure_data import NormalizePressure
from openpi.training.pressure_dataset import PressureNpzDataset
from openpi.training.pressure_dataset import fit_pressure_stats


def write_episode(folder, name="episode", *, count=4, offset=0, robot_updates=None, pressure_updates=None):
    robot = {
        "timestamps": np.arange(count, dtype=np.float64) + offset,
        "state": np.arange(count * 3).reshape(count, 3).astype(np.float32),
        "actions": (np.arange(count * 2).reshape(count, 2) + offset).astype(np.float32),
        "base_0_rgb": np.full((count, 3, 4, 3), 17, dtype=np.uint8),
    }
    if robot_updates:
        robot.update(robot_updates)
    pressure = {
        "timestamps": np.arange(count, dtype=np.float64) + offset - 0.1,
        "values": np.arange(count * 4).reshape(count, 1, 2, 2).astype(np.float32),
    }
    if pressure_updates:
        pressure.update(pressure_updates)
    np.savez(folder / f"{name}_robot.npz", **robot)
    np.savez(folder / f"{name}_touch.npz", **pressure)
    return {"observations": f"{name}_robot.npz", "pressure": f"{name}_touch.npz", "prompt": "grasp object"}


def manifest(folder, episodes):
    path = folder / "train.json"
    path.write_text(json.dumps({"episodes": episodes}), encoding="utf-8")
    return path


def test_disk_sample_canonical_shapes_causal_pressure_and_missing_camera(tmp_path):
    path = manifest(tmp_path, [write_episode(tmp_path)])
    dataset = PressureNpzDataset(path, 3, 2)
    assert len(dataset) == 3
    assert dataset.dropped_tail_samples == 1
    assert dataset.episode_summaries[0]["dropped_tail"] == 1
    assert dataset.pressure_geometry == (1, 2, 2)
    assert dataset.robot_action_dim == 2
    assert dataset.robot_state_dim == 3
    sample = dataset[0]
    assert set(sample) == {"image", "image_mask", "state", "actions", "prompt", "pressure"}
    assert sample["pressure"]["values"].shape == (1, 3, 2, 2)
    np.testing.assert_array_equal(sample["pressure"]["frame_valid"], [[False, False, True]])
    np.testing.assert_allclose(sample["pressure"]["age"], [[0, 0, 0.1]])
    np.testing.assert_array_equal(sample["actions"], [[0, 1], [2, 3]])
    assert sample["prompt"].shape == ()
    assert sample["prompt"].item() == "grasp object"
    assert sample["image_mask"]["base_0_rgb"]
    assert not sample["image_mask"]["left_wrist_0_rgb"]
    assert sample["image"]["left_wrist_0_rgb"].dtype == np.uint8
    assert not sample["image"]["left_wrist_0_rgb"].any()


def test_episode_tail_does_not_cross_labels_or_history_boundary(tmp_path):
    first = write_episode(tmp_path, "first", count=4)
    second = write_episode(tmp_path, "second", count=4, offset=100)
    dataset = PressureNpzDataset(manifest(tmp_path, [first, second]), 2, 3)
    assert len(dataset) == 4
    assert dataset.dropped_tail_samples == 4
    np.testing.assert_array_equal(dataset[1]["actions"], [[2, 3], [4, 5], [6, 7]])
    np.testing.assert_array_equal(dataset[2]["actions"], [[100, 101], [102, 103], [104, 105]])
    np.testing.assert_array_equal(dataset[2]["pressure"]["frame_valid"], [[False, True]])
    np.testing.assert_array_equal(dataset[-1]["actions"], dataset[3]["actions"])
    with pytest.raises(IndexError):
        dataset[4]


def test_short_episodes_emit_no_fabricated_action_chunks(tmp_path):
    dataset = PressureNpzDataset(manifest(tmp_path, [write_episode(tmp_path, count=2)]), 2, 3)
    assert len(dataset) == 0
    assert dataset.dropped_tail_samples == 2
    with pytest.raises(IndexError):
        dataset[0]


def test_cache_returns_independent_samples_and_preserves_camera_validity(tmp_path):
    valid = np.array([False, True, True, True])
    entry = write_episode(
        tmp_path,
        robot_updates={
            "base_0_rgb_valid": valid,
            "left_wrist_0_rgb": np.zeros((4, 2, 2, 3), dtype=np.float32),
        },
    )
    dataset = PressureNpzDataset(manifest(tmp_path, [entry]), 2, 2)
    first = dataset[0]
    assert not first["image_mask"]["base_0_rgb"]
    assert first["image_mask"]["left_wrist_0_rgb"]
    first["state"][:] = -100
    first["actions"][:] = -100
    first["image"]["base_0_rgb"][:] = 0
    first["pressure"]["values"][:] = -100
    second = dataset[0]
    np.testing.assert_array_equal(second["state"], [0, 1, 2])
    assert second["image"]["base_0_rgb"].min() == 17
    assert second["pressure"]["values"].max() == 3


@pytest.mark.parametrize(
    ("updates", "match"),
    [
        ({"timestamps": np.array([0, 1, 1, 3])}, "timestamps"),
        ({"state": np.zeros((3, 2))}, "state"),
        ({"actions": np.full((4, 2), np.nan)}, "actions"),
        ({"base_0_rgb": np.zeros((4, 2, 2), dtype=np.uint8)}, "base_0_rgb"),
        ({"base_0_rgb": np.full((4, 2, 2, 3), 2, dtype=np.float32)}, "float32"),
        ({"base_0_rgb_valid": np.ones(4)}, "bool"),
        ({"right_wrist_0_rgb_valid": np.ones(4, dtype=bool)}, "without its camera"),
    ],
)
def test_reject_bad_robot_data(tmp_path, updates, match):
    path = manifest(tmp_path, [write_episode(tmp_path, robot_updates=updates)])
    with pytest.raises(ValueError, match=match):
        PressureNpzDataset(path, 2, 2)


def test_required_base_camera_and_manifest_fields(tmp_path):
    entry = write_episode(tmp_path)
    np.savez(tmp_path / entry["observations"], timestamps=[0], state=[[0]], actions=[[0]])
    with pytest.raises(ValueError, match="base_0_rgb"):
        PressureNpzDataset(manifest(tmp_path, [entry]), 1, 1)
    with pytest.raises(ValueError, match="requires"):
        PressureNpzDataset(manifest(tmp_path, [{"observations": "x"}]), 1, 1)


def test_sensor_slot_metadata_supports_gripper_and_forearm(tmp_path):
    entry = write_episode(
        tmp_path,
        pressure_updates={"values": np.zeros((4, 2, 2, 2), dtype=np.float32)},
    )
    manifest_path = manifest(
        tmp_path,
        [
            {
                **entry,
                "sensor_names": ["left_gripper", "forearm_left"],
                "sensor_locations": ["gripper_fingertip", "left_forearm"],
            }
        ],
    )
    dataset = PressureNpzDataset(manifest_path, 1, 1)
    assert dataset.sensor_names == ("left_gripper", "forearm_left")
    assert dataset.sensor_locations == ("gripper_fingertip", "left_forearm")
    sample = dataset[0]
    assert set(sample) == {"image", "image_mask", "state", "actions", "prompt", "pressure"}


def test_sensor_slot_metadata_must_match_geometry(tmp_path):
    entry = write_episode(tmp_path)
    path = manifest(
        tmp_path,
        [{**entry, "sensor_names": ["only_one", "too_many"]}],
    )
    with pytest.raises(ValueError, match="sensor_names"):
        PressureNpzDataset(path, 1, 1)


def test_stats_exclude_missing_saturation_and_nonfinite_taxels(tmp_path):
    values = np.array([[[[0, 10, np.nan, 9]]], [[[2, 30, np.inf, 9]]], [[[4, 50, 7, 9]]]], dtype=np.float32)
    saturated = np.zeros_like(values, dtype=bool)
    saturated[1, 0, 0, 1] = True
    valid = np.ones_like(values, dtype=bool)
    valid[:, 0, 0, 3] = False
    entry = write_episode(
        tmp_path,
        count=3,
        pressure_updates={"values": values, "saturated": saturated, "taxel_valid": valid},
    )
    stats = fit_pressure_stats(manifest(tmp_path, [entry]))
    np.testing.assert_array_equal(stats["count"], [[[3, 2, 1, 0]]])
    np.testing.assert_allclose(stats["mean"], [[[2, 30, 7, 0]]])
    np.testing.assert_allclose(stats["scale"], [[[np.sqrt(8 / 3), 20, 1e-6, 1]]], rtol=1e-6)


def test_stats_merge_episodes_and_skip_invalid_frames(tmp_path):
    first = write_episode(
        tmp_path,
        "first",
        count=2,
        pressure_updates={"values": np.array([[[[2.0]]], [[[1000.0]]]]), "frame_valid": np.array([[True], [False]])},
    )
    second = write_episode(tmp_path, "second", count=2, pressure_updates={"values": np.array([[[[4.0]]], [[[6.0]]]])})
    stats = fit_pressure_stats(manifest(tmp_path, [first, second]))
    np.testing.assert_array_equal(stats["count"], [[[3]]])
    np.testing.assert_allclose(stats["mean"], [[[4]]])
    np.testing.assert_allclose(stats["scale"], [[[np.sqrt(8 / 3)]]])


def test_stats_geometry_mismatch_is_explicit(tmp_path):
    first = write_episode(tmp_path, "first", count=2)
    second = write_episode(tmp_path, "second", count=2, pressure_updates={"values": np.zeros((2, 2, 2, 2))})
    with pytest.raises(ValueError, match="geometry"):
        fit_pressure_stats(manifest(tmp_path, [first, second]))


@pytest.mark.parametrize(
    ("robot_updates", "pressure_updates", "message"),
    [
        ({}, {"values": np.zeros((4, 2, 2, 2))}, "pressure geometry"),
        ({"actions": np.zeros((4, 7))}, {}, "robot action dimension"),
        ({"state": np.zeros((4, 9))}, {}, "robot state dimension"),
    ],
)
def test_all_episode_geometry_and_robot_dimensions_validated_before_training(
    tmp_path, robot_updates, pressure_updates, message
):
    first = write_episode(tmp_path, "first")
    second = write_episode(tmp_path, "second", robot_updates=robot_updates, pressure_updates=pressure_updates)
    with pytest.raises(ValueError, match=message):
        PressureNpzDataset(manifest(tmp_path, [first, second]), 2, 2)


def test_stats_cli_produces_loadable_independent_asset(tmp_path):
    path = manifest(tmp_path, [write_episode(tmp_path)])
    output = tmp_path / "assets" / "pressure_stats.npz"
    script = Path(__file__).resolve().parents[2] / "scripts" / "compute_pressure_stats.py"
    result = subprocess.run(
        [sys.executable, str(script), "--manifest", str(path), "--output", str(output)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "signal normalization" in result.stdout
    normalizer = NormalizePressure.from_npz(output)
    sample = PressureNpzDataset(path, 2, 2)[0]
    assert np.isfinite(normalizer(sample)["pressure"]["values"]).all()
    with np.load(output, allow_pickle=False) as asset:
        assert asset["mean"].shape == (1, 2, 2)
        assert asset["count"].min() == 4
        assert asset["signal_unit"].item() == "signal"
