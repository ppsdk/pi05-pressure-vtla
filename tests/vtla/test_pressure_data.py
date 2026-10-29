"""These tests require only NumPy and pytest; no model/GPU stack is imported."""

import numpy as np
import pytest

from openpi.pressure_data import NormalizePressure
from openpi.pressure_data import PressureEpisode
from openpi.pressure_data import PressureHistoryTransform
from openpi.pressure_data import TrimPressureActions
from openpi.pressure_data import validate_pressure


def episode(**kwargs):
    return PressureEpisode(timestamps=np.array([1.0, 2.0, 3.0]), values=np.arange(12).reshape(3, 2, 1, 2), **kwargs)


def test_history_is_causal_unique_and_left_padded():
    pressure = episode().history(2.5, 4)
    np.testing.assert_array_equal(pressure["frame_valid"], [[False, False, True, True]] * 2)
    np.testing.assert_allclose(pressure["age"], [[0, 0, 1.5, 0.5]] * 2)
    np.testing.assert_array_equal(pressure["values"][0, :, 0, 0], [0, 0, 0, 4])
    assert not pressure["taxel_valid"][:, :2].any()
    assert pressure["taxel_valid"][0, 2, 0, 0]  # A genuine zero signal is valid.


def test_arrival_time_blocks_unavailable_observations_even_with_older_timestamp():
    pressure = episode(arrival_times=np.array([1.1, 4.0, 3.0])).history(3.1, 3)
    np.testing.assert_array_equal(pressure["frame_valid"], [[False, True, True]] * 2)
    np.testing.assert_array_equal(pressure["values"][0, :, 0, 0], [0, 0, 8])
    np.testing.assert_allclose(pressure["age"][:, 1:], [[2.1, 0.1]] * 2, atol=1e-6)


def test_before_episode_has_no_fabricated_history():
    pressure = episode().history(0, 2)
    assert not pressure["frame_valid"].any()
    assert not pressure["taxel_valid"].any()
    assert not pressure["values"].any()


def test_stale_frames_are_invalid_but_retain_real_age():
    pressure = episode().history(3.5, 3, max_age=0.6)
    np.testing.assert_array_equal(pressure["frame_valid"], [[False, False, True]] * 2)
    np.testing.assert_allclose(pressure["age"], [[2.5, 1.5, 0.5]] * 2)
    assert not pressure["values"][:, :2].any()


def test_missing_taxels_nonfinite_and_saturated_are_separate():
    values = np.array([[[[0.0, np.nan, np.inf, 100.0]]]])
    saturated = np.array([[[[False, False, False, True]]]])
    pressure = PressureEpisode([1], values, saturated=saturated).history(1, 1)
    np.testing.assert_array_equal(pressure["taxel_valid"], [[[[True, False, False, True]]]])
    np.testing.assert_array_equal(pressure["values"], [[[[0, 0, 0, 100]]]])
    assert pressure["frame_valid"].all()
    assert pressure["saturated"][0, 0, 0, 3]


@pytest.mark.parametrize("timestamps", [[1, 1, 2], [1, 0, 2], [1, np.nan, 3]])
def test_reject_nonmonotonic_and_nonfinite_episode_times(timestamps):
    with pytest.raises(ValueError, match="timestamps"):
        PressureEpisode(timestamps, np.zeros((3, 1, 1, 1)))


def test_reject_impossible_arrival_time():
    with pytest.raises(ValueError, match="arrival_times"):
        episode(arrival_times=np.array([0, 2, 3]))


@pytest.mark.parametrize("count", [0, -1, 1.5, True])
def test_reject_invalid_history_size(count):
    with pytest.raises(ValueError, match="history_frames"):
        episode().history(2, count)


def test_npz_episode_roundtrip_and_pickle_rejection(tmp_path):
    path = tmp_path / "one_episode.npz"
    np.savez(path, timestamps=[1, 2], values=np.ones((2, 1, 2, 2)))
    loaded = PressureEpisode.from_npz(path)
    assert loaded.history(2, 2)["frame_valid"].all()
    np.savez(path, timestamps=[1], values=np.array([{"bad": "object"}], dtype=object))
    with pytest.raises(ValueError, match="Object arrays"):
        PressureEpisode.from_npz(path)


def test_history_transform_preserves_state_and_actions():
    state = np.array([42.0])
    actions = np.array([[17.0]])
    data = PressureHistoryTransform(episode(), 2)({"timestamp": np.array([2.5]), "state": state, "actions": actions})
    assert data["state"] is state
    assert data["actions"] is actions
    assert data["pressure"]["values"].shape == (2, 2, 1, 2)


def test_fixed_sensor_normalization_keeps_contact_amplitude_and_metadata():
    pressure = episode().history(3, 2)
    normalized = NormalizePressure(np.array([0, 2]), np.array([2, 4])).normalize(pressure)
    np.testing.assert_allclose(normalized["values"][0], pressure["values"][0] / 2)
    np.testing.assert_allclose(normalized["values"][1], (pressure["values"][1] - 2) / 4)
    for key in ("frame_valid", "taxel_valid", "saturated", "age"):
        np.testing.assert_array_equal(normalized[key], pressure[key])


def test_normalization_zeros_invalid_values_after_subtraction():
    pressure = episode().history(1, 3)
    normalized = NormalizePressure(np.array(10), np.array(2)).normalize(pressure)
    assert not normalized["values"][:, :2].any()
    np.testing.assert_allclose(normalized["values"][:, -1], (pressure["values"][:, -1] - 10) / 2)


def test_batch_and_taxel_statistics_broadcast_without_history_aliasing():
    pressure = episode().history(3, 3)
    batched = {key: np.stack([value, value]) for key, value in pressure.items()}
    mean = np.arange(4).reshape(2, 1, 2)
    normalized = NormalizePressure(mean, np.ones_like(mean)).normalize(batched)
    assert normalized["values"].shape == (2, 2, 3, 1, 2)
    np.testing.assert_allclose(normalized["values"], batched["values"] - mean[None, :, None])


def test_reject_shape_mismatch_instead_of_implicit_broadcast():
    pressure = episode().history(3, 2)
    with pytest.raises(ValueError, match="statistics shape"):
        NormalizePressure(np.zeros((2, 2, 1, 2)), np.array(1)).normalize(pressure)
    with pytest.raises(ValueError, match="frame_valid"):
        validate_pressure({**pressure, "frame_valid": np.ones((2, 2))})


@pytest.mark.parametrize("scale", [0, -1, np.nan, np.inf])
def test_reject_invalid_scale(scale):
    with pytest.raises(ValueError, match="scale"):
        NormalizePressure(np.array(0), np.array(scale))


def test_npz_fixed_statistics_and_clipping_are_pressure_only(tmp_path):
    path = tmp_path / "pressure_stats.npz"
    np.savez(path, mean=0.0, scale=2.0)
    transform = NormalizePressure.from_npz(path, clip=1.5)
    data = {"pressure": episode().history(3, 1), "state": np.array([10]), "actions": np.array([20])}
    normalized = transform(data)
    assert np.max(np.abs(normalized["pressure"]["values"])) <= 1.5
    assert normalized["state"] is data["state"]
    assert normalized["actions"] is data["actions"]


def test_trim_model_padding_preserves_robot_coordinates_and_other_keys():
    data = {"actions": np.arange(24).reshape(2, 3, 4), "state": np.array([1]), "custom": "keep"}
    output = TrimPressureActions(2)(data)
    assert output["actions"].shape == (2, 3, 2)
    np.testing.assert_array_equal(output["actions"], data["actions"][..., :2])
    assert output["state"] is data["state"]
    assert output["custom"] == "keep"
    assert data["actions"].shape == (2, 3, 4)


@pytest.mark.parametrize("dimension", [0, -1, 2.5, True])
def test_trim_rejects_invalid_robot_action_dimension(dimension):
    with pytest.raises(ValueError, match="action_dim"):
        TrimPressureActions(dimension)


def test_trim_rejects_missing_or_too_short_actions():
    trim = TrimPressureActions(3)
    with pytest.raises(ValueError, match="requires actions"):
        trim({"state": np.array([0])})
    with pytest.raises(ValueError, match="fewer"):
        trim({"actions": np.zeros((4, 2))})
    with pytest.raises(ValueError, match="fewer"):
        trim({"actions": np.asarray(1)})
