"""NumPy-only resistive-array loading and causal history preprocessing.

One NPZ file represents one episode in one shared clock (seconds). ``values``
are sensor signals: this module does not turn ADC readings into physical pressure.
Calibrate explicitly upstream, and keep the calibration and normalization assets.
History has shape [sensors, history, rows, cols], oldest frame first. Batching
adds one leading dimension. Zero signal, missing data, and saturation are distinct.
"""

from collections.abc import Mapping
import dataclasses
from pathlib import Path

import numpy as np

PRESSURE_KEYS = frozenset({"values", "frame_valid", "taxel_valid", "saturated", "age"})


def _boolean(value, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != shape or array.dtype != np.bool_:
        raise ValueError(f"{name} must be bool with shape {shape}, got {array.shape}/{array.dtype}")
    return array


def validate_pressure(pressure: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Validate the model contract for one sample or one batched collection.

    Non-finite values are treated as unavailable taxels. Invalid values are
    sanitized before arithmetic so that NaN/Inf cannot poison attention inputs.
    Saturated taxels remain valid if a reading exists; saturation is metadata.
    """
    if set(pressure) != PRESSURE_KEYS:
        raise ValueError(f"pressure requires exactly {sorted(PRESSURE_KEYS)}, got {sorted(pressure)}")
    values = np.asarray(pressure["values"])
    if values.ndim not in (4, 5) or any(dim < 1 for dim in values.shape):
        raise ValueError("pressure values must have shape [S,K,R,C] or [B,S,K,R,C], with positive dimensions")
    if not np.issubdtype(values.dtype, np.number) or np.issubdtype(values.dtype, np.complexfloating):
        raise ValueError("pressure values must be real numeric sensor signals")
    frame_shape = values.shape[:-2]
    frame_valid = _boolean(pressure["frame_valid"], frame_shape, "frame_valid")
    taxel_valid = _boolean(pressure["taxel_valid"], values.shape, "taxel_valid")
    saturated = _boolean(pressure["saturated"], values.shape, "saturated")
    age = np.asarray(pressure["age"], dtype=np.float32)
    if age.shape != frame_shape or not np.all(np.isfinite(age)) or np.any(age < 0):
        raise ValueError("pressure age must match frame_valid and contain finite nonnegative seconds")
    values = values.astype(np.float32)
    valid = frame_valid[..., None, None] & taxel_valid & np.isfinite(values)
    return {
        "values": np.where(valid, values, 0).astype(np.float32),
        "frame_valid": frame_valid.copy(),
        "taxel_valid": valid,
        "saturated": saturated.copy(),
        "age": age.copy(),
    }


@dataclasses.dataclass(frozen=True)
class PressureEpisode:
    """A single episode of synchronized sensor frames, with optional arrival times.

    Required NPZ entries: timestamps[T], values[T,S,R,C]. Optional entries:
    frame_valid[T,S], taxel_valid/saturated[T,S,R,C], arrival_times[T].
    Timestamps must be strictly increasing. Arrival times need not be ordered,
    but cannot precede acquisition. Without arrival_times, zero transport latency
    is an explicit dataset assumption. Do not combine episodes in one file.
    """

    timestamps: np.ndarray
    values: np.ndarray
    frame_valid: np.ndarray | None = None
    taxel_valid: np.ndarray | None = None
    saturated: np.ndarray | None = None
    arrival_times: np.ndarray | None = None

    def __post_init__(self):
        timestamps = np.asarray(self.timestamps, dtype=np.float64)
        values = np.asarray(self.values)
        if timestamps.ndim != 1 or timestamps.size < 1 or not np.all(np.isfinite(timestamps)):
            raise ValueError("timestamps must be a nonempty finite vector")
        if np.any(np.diff(timestamps) <= 0):
            raise ValueError("timestamps must be strictly increasing within one episode")
        if values.ndim != 4 or values.shape[0] != timestamps.size or any(d < 1 for d in values.shape):
            raise ValueError("values must have shape [T,S,R,C], aligned with timestamps")
        if not np.issubdtype(values.dtype, np.number) or np.issubdtype(values.dtype, np.complexfloating):
            raise ValueError("values must contain real numeric signals")
        frame_shape = values.shape[:2]
        frame = np.ones(frame_shape, dtype=bool) if self.frame_valid is None else self.frame_valid
        taxels = np.ones(values.shape, dtype=bool) if self.taxel_valid is None else self.taxel_valid
        saturated = np.zeros(values.shape, dtype=bool) if self.saturated is None else self.saturated
        arrival = timestamps if self.arrival_times is None else np.asarray(self.arrival_times, dtype=np.float64)
        if arrival.shape != timestamps.shape or not np.all(np.isfinite(arrival)) or np.any(arrival < timestamps):
            raise ValueError("arrival_times must match timestamps, be finite, and not precede acquisition")
        object.__setattr__(self, "timestamps", timestamps.copy())
        object.__setattr__(self, "values", values.astype(np.float32))
        object.__setattr__(self, "frame_valid", _boolean(frame, frame_shape, "frame_valid").copy())
        object.__setattr__(self, "taxel_valid", _boolean(taxels, values.shape, "taxel_valid").copy())
        object.__setattr__(self, "saturated", _boolean(saturated, values.shape, "saturated").copy())
        object.__setattr__(self, "arrival_times", arrival.copy())

    @classmethod
    def from_npz(cls, path: str | Path) -> "PressureEpisode":
        """Read numeric arrays without allowing pickle/object deserialization."""
        with np.load(path, allow_pickle=False) as asset:
            if not {"timestamps", "values"}.issubset(asset.files):
                raise ValueError("pressure episode NPZ requires timestamps and values")
            return cls(
                timestamps=asset["timestamps"],
                values=asset["values"],
                **{
                    key: asset[key]
                    for key in ("frame_valid", "taxel_valid", "saturated", "arrival_times")
                    if key in asset
                },
            )

    def history(
        self, observation_time: float, history_frames: int, *, max_age: float | None = None
    ) -> dict[str, np.ndarray]:
        """Select the latest K distinct acquired AND arrived frames, causally.

        Missing history is left-padded with invalid frames, never repeated data.
        Frames older than max_age keep their real age but are marked unavailable.
        Valid zero readings and saturated readings are not automatically missing.
        """
        if not np.isscalar(observation_time) or not np.isfinite(observation_time):
            raise ValueError("observation_time must be a finite scalar in the episode clock")
        if isinstance(history_frames, bool) or not isinstance(history_frames, (int, np.integer)) or history_frames < 1:
            raise ValueError("history_frames must be a positive integer")
        if max_age is not None and (not np.isfinite(max_age) or max_age < 0):
            raise ValueError("max_age must be finite and nonnegative")
        indices = np.flatnonzero((self.timestamps <= observation_time) & (self.arrival_times <= observation_time))
        indices = indices[-history_frames:]
        _, sensors, rows, cols = self.values.shape
        pressure = {
            "values": np.zeros((sensors, history_frames, rows, cols), dtype=np.float32),
            "frame_valid": np.zeros((sensors, history_frames), dtype=bool),
            "taxel_valid": np.zeros((sensors, history_frames, rows, cols), dtype=bool),
            "saturated": np.zeros((sensors, history_frames, rows, cols), dtype=bool),
            "age": np.zeros((sensors, history_frames), dtype=np.float32),
        }
        if indices.size:
            tail = slice(history_frames - indices.size, None)
            pressure["values"][:, tail] = self.values[indices].transpose(1, 0, 2, 3)
            pressure["frame_valid"][:, tail] = self.frame_valid[indices].T
            pressure["taxel_valid"][:, tail] = self.taxel_valid[indices].transpose(1, 0, 2, 3)
            pressure["saturated"][:, tail] = self.saturated[indices].transpose(1, 0, 2, 3)
            ages = observation_time - self.timestamps[indices]
            pressure["age"][:, tail] = ages[None, :]
            if max_age is not None:
                pressure["frame_valid"][:, tail] &= ages[None, :] <= max_age
        return validate_pressure(pressure)


@dataclasses.dataclass(frozen=True)
class PressureHistoryTransform:
    """Add episode history to one unbatched dataset/online observation dictionary.

    Construct one transform per episode (or select the appropriate instance in a
    dataset wrapper). The timestamp key must already use the episode clock.
    """

    episode: PressureEpisode
    history_frames: int
    timestamp_key: str = "timestamp"
    max_age: float | None = None

    def __call__(self, data: dict) -> dict:
        if self.timestamp_key not in data:
            raise ValueError(f"observation is missing decision timestamp {self.timestamp_key!r}")
        timestamp = np.asarray(data[self.timestamp_key])
        if timestamp.size != 1:
            raise ValueError("PressureHistoryTransform expects one unbatched observation timestamp")
        return {
            **data,
            "pressure": self.episode.history(float(timestamp.reshape(())), self.history_frames, max_age=self.max_age),
        }


@dataclasses.dataclass(frozen=True)
class NormalizePressure:
    """Apply fixed training-set sensor statistics, preserving quality metadata.

    mean/scale can be scalar, [S], [S,R,C], or [S,1,R,C]. No frame-dependent
    statistics are accepted. scale must be positive and finite. These assets
    are independent of state/action normalization and state text tokenization.
    Optional symmetric clipping acts in normalized signal units.
    """

    mean: np.ndarray
    scale: np.ndarray
    clip: float | None = None

    def __post_init__(self):
        for name in ("mean", "scale"):
            value = np.asarray(getattr(self, name), dtype=np.float32)
            if value.ndim not in (0, 1, 3, 4) or not np.all(np.isfinite(value)):
                raise ValueError(f"pressure {name} must be finite scalar, [S], [S,R,C], or [S,1,R,C]")
            if name == "scale" and np.any(value <= 0):
                raise ValueError("pressure scale must be strictly positive")
            object.__setattr__(self, name, value.copy())
        if self.clip is not None and (not np.isfinite(self.clip) or self.clip <= 0):
            raise ValueError("clip must be finite and positive")

    @classmethod
    def from_npz(cls, path: str | Path, *, clip: float | None = None) -> "NormalizePressure":
        """Load explicit fixed mean/scale assets; never fit on the current frame."""
        with np.load(path, allow_pickle=False) as asset:
            if not {"mean", "scale"}.issubset(asset.files):
                raise ValueError("pressure normalization NPZ requires mean and scale")
            return cls(asset["mean"], asset["scale"], clip=clip)

    @staticmethod
    def _expand(stats: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
        sensors, _, rows, cols = shape[-4:]
        if stats.ndim == 0:
            return stats
        if stats.ndim == 1 and stats.shape == (sensors,):
            return stats[:, None, None, None]
        if stats.ndim == 3 and stats.shape == (sensors, rows, cols):
            return stats[:, None, :, :]
        if stats.ndim == 4 and stats.shape == (sensors, 1, rows, cols):
            return stats
        raise ValueError(f"statistics shape {stats.shape} is incompatible with pressure values {shape}")

    def normalize(self, pressure: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        result = validate_pressure(pressure)
        values = result["values"]
        mean = self._expand(self.mean, values.shape)
        scale = self._expand(self.scale, values.shape)
        with np.errstate(over="raise", divide="raise", invalid="raise"):
            normalized = (values - mean) / scale
        if self.clip is not None:
            normalized = np.clip(normalized, -self.clip, self.clip)
        result["values"] = np.where(result["taxel_valid"], normalized, 0).astype(np.float32)
        return result

    def __call__(self, data: dict) -> dict:
        if data.get("pressure") is None:
            raise ValueError("NormalizePressure requires the independent pressure observation field")
        return {**data, "pressure": self.normalize(data["pressure"])}


@dataclasses.dataclass(frozen=True)
class TrimPressureActions:
    """Remove model action padding before passing actions to the robot adapter.

    The explicit action dimension must match the training robot dataset. This
    slices output coordinates only; it does not change coordinate conventions.
    """

    action_dim: int

    def __post_init__(self):
        if (
            isinstance(self.action_dim, bool)
            or not isinstance(self.action_dim, (int, np.integer))
            or self.action_dim < 1
        ):
            raise ValueError("robot action_dim must be a positive integer")

    def __call__(self, data: dict) -> dict:
        if "actions" not in data:
            raise ValueError("TrimPressureActions requires actions")
        actions = np.asarray(data["actions"])
        if actions.ndim < 1 or actions.shape[-1] < self.action_dim:
            raise ValueError(f"actions shape {actions.shape} has fewer than {self.action_dim} robot action dimensions")
        return {**data, "actions": actions[..., : self.action_dim]}
