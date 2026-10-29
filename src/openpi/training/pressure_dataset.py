"""NumPy-only manifest dataset for synchronized robot and resistive-array signals.

Each manifest entry is one episode: observations NPZ and pressure NPZ paths are
resolved relative to the JSON manifest. Actions are supplied in the intended
robot action convention; no synthetic labels, cross-episode padding, calibration,
or state/action normalization is performed here.
"""

import bisect
import dataclasses
import json
from pathlib import Path

import numpy as np

from openpi.pressure_data import PressureEpisode

CAMERA_KEYS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")


@dataclasses.dataclass(frozen=True)
class _EpisodeSpec:
    observations: Path
    pressure: Path
    prompt: str


def _read_manifest(manifest_path: str | Path) -> tuple[Path, list[_EpisodeSpec]]:
    path = Path(manifest_path).resolve()
    with path.open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    if not isinstance(manifest, dict) or not isinstance(manifest.get("episodes"), list) or not manifest["episodes"]:
        raise ValueError("manifest must contain a nonempty episodes list")
    episodes = []
    for index, entry in enumerate(manifest["episodes"]):
        if not isinstance(entry, dict) or any(key not in entry for key in ("observations", "pressure", "prompt")):
            raise ValueError(f"manifest episode {index} requires observations, pressure, and prompt")
        if not isinstance(entry["prompt"], str):
            raise ValueError(f"manifest episode {index} prompt must be a string")
        for key in ("observations", "pressure"):
            if not isinstance(entry[key], str) or not entry[key]:
                raise ValueError(f"manifest episode {index} {key} must be a nonempty path string")
        episodes.append(
            _EpisodeSpec(
                observations=(path.parent / entry["observations"]).resolve(),
                pressure=(path.parent / entry["pressure"]).resolve(),
                prompt=entry["prompt"],
            )
        )
    return path, episodes


def _read_robot(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as asset:
        required = {"timestamps", "state", "actions", "base_0_rgb"}
        if not required.issubset(asset.files):
            raise ValueError(f"robot NPZ {path} is missing {sorted(required - set(asset.files))}")
        selected = required | {key for camera in CAMERA_KEYS for key in (camera, f"{camera}_valid") if key in asset}
        data = {key: np.array(asset[key], copy=True) for key in selected}
    timestamps = np.asarray(data["timestamps"], dtype=np.float64)
    if (
        timestamps.ndim != 1
        or timestamps.size < 1
        or not np.all(np.isfinite(timestamps))
        or np.any(np.diff(timestamps) <= 0)
    ):
        raise ValueError(f"robot timestamps in {path} must be nonempty, finite, and strictly increasing")
    count = timestamps.size
    data["timestamps"] = timestamps
    for key in ("state", "actions"):
        values = data[key]
        if (
            values.ndim != 2
            or values.shape[0] != count
            or values.shape[1] < 1
            or not np.issubdtype(values.dtype, np.number)
            or np.issubdtype(values.dtype, np.complexfloating)
            or not np.all(np.isfinite(values))
        ):
            raise ValueError(f"robot {key} in {path} must be finite real [N,D] aligned with timestamps")
        data[key] = values.astype(np.float32)
    for key in CAMERA_KEYS:
        if key not in data:
            if f"{key}_valid" in data:
                raise ValueError(f"{key}_valid is present without its camera in {path}")
            continue
        image = data[key]
        if image.ndim != 4 or image.shape[0] != count or image.shape[-1] != 3 or min(image.shape[1:3]) < 1:
            raise ValueError(f"{key} in {path} must have shape [N,H,W,3]")
        if image.dtype != np.uint8 and (
            image.dtype != np.float32 or not np.all(np.isfinite(image)) or np.any((image < -1) | (image > 1))
        ):
            raise ValueError(f"{key} in {path} must be uint8 or finite float32 in [-1,1]")
        mask_key = f"{key}_valid"
        if mask_key in data and (data[mask_key].shape != (count,) or data[mask_key].dtype != np.bool_):
            raise ValueError(f"{mask_key} in {path} must be bool [N]")
    return data


class PressureNpzDataset:
    """Canonical unbatched openpi samples with causal resistive-array history.

    Only N-H+1 complete action chunks are emitted in each episode. Short episodes
    emit no samples. ``episode_summaries`` and ``dropped_tail_samples`` explicitly
    record observations excluded for lacking a complete action horizon. A single
    episode is cached per dataset/worker; arrays returned by __getitem__ are copies.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        history_frames: int,
        action_horizon: int,
        max_age: float | None = None,
    ):
        for name, value in (("history_frames", history_frames), ("action_horizon", action_horizon)):
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if max_age is not None and (not np.isfinite(max_age) or max_age < 0):
            raise ValueError("max_age must be finite and nonnegative")
        self.manifest_path, self.episodes = _read_manifest(manifest_path)
        self.history_frames = int(history_frames)
        self.action_horizon = int(action_horizon)
        self.max_age = max_age
        self.episode_summaries = []
        self.dropped_tail_samples = 0
        self._ends = []
        self._cached_index = None
        self._cached_data = None
        self.pressure_geometry = None
        self.robot_action_dim = None
        self.robot_state_dim = None
        total = 0
        for index, spec in enumerate(self.episodes):
            robot, pressure = self._load_episode(index)
            geometry = pressure.values.shape[1:]
            action_dim = robot["actions"].shape[-1]
            state_dim = robot["state"].shape[-1]
            if self.pressure_geometry is None:
                self.pressure_geometry = geometry
                self.robot_action_dim = action_dim
                self.robot_state_dim = state_dim
            elif geometry != self.pressure_geometry:
                raise ValueError(
                    f"pressure geometry in episode {index} is {geometry}; expected {self.pressure_geometry} [S,R,C]"
                )
            elif action_dim != self.robot_action_dim:
                raise ValueError(
                    f"robot action dimension in episode {index} is {action_dim}; expected {self.robot_action_dim}"
                )
            elif state_dim != self.robot_state_dim:
                raise ValueError(
                    f"robot state dimension in episode {index} is {state_dim}; expected {self.robot_state_dim}"
                )
            frames = len(robot["timestamps"])
            samples = max(0, frames - self.action_horizon + 1)
            dropped = frames - samples
            self.dropped_tail_samples += dropped
            total += samples
            self._ends.append(total)
            self.episode_summaries.append(
                {"observations": str(spec.observations), "frames": frames, "samples": samples, "dropped_tail": dropped}
            )

    def _load_episode(self, index: int) -> tuple[dict[str, np.ndarray], PressureEpisode]:
        if self._cached_index != index:
            spec = self.episodes[index]
            self._cached_data = (_read_robot(spec.observations), PressureEpisode.from_npz(spec.pressure))
            self._cached_index = index
        return self._cached_data

    def __len__(self) -> int:
        return self._ends[-1]

    def __getitem__(self, index: int) -> dict:
        if isinstance(index, bool) or not isinstance(index, (int, np.integer)):
            raise TypeError("PressureNpzDataset indices must be integers")
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError("PressureNpzDataset index out of range")
        episode_index = bisect.bisect_right(self._ends, index)
        start = 0 if episode_index == 0 else self._ends[episode_index - 1]
        frame = index - start
        robot, pressure = self._load_episode(episode_index)
        images, masks = {}, {}
        for key in CAMERA_KEYS:
            if key in robot:
                images[key] = robot[key][frame].copy()
                mask_key = f"{key}_valid"
                masks[key] = np.asarray(robot[mask_key][frame] if mask_key in robot else True)
            else:
                images[key] = np.zeros(robot["base_0_rgb"].shape[1:], dtype=np.uint8)
                masks[key] = np.zeros((), dtype=bool)
        return {
            "image": images,
            "image_mask": masks,
            "state": robot["state"][frame].copy(),
            "actions": robot["actions"][frame : frame + self.action_horizon].copy(),
            "prompt": np.asarray(self.episodes[episode_index].prompt),
            "pressure": pressure.history(float(robot["timestamps"][frame]), self.history_frames, max_age=self.max_age),
        }


def fit_pressure_stats(manifest_path: str | Path, *, min_scale: float = 1e-6) -> dict[str, np.ndarray]:
    """Fit fixed per-taxel signal statistics from an explicitly supplied TRAIN manifest.

    Every pressure frame in each listed episode is visited once; history windows
    do not duplicate frames. Missing frames/taxels, NaN/Inf, and saturated readings
    are excluded. This normalizes raw/calibrated signals without claiming a unit
    conversion. Caller must ensure this manifest excludes validation/test episodes.
    """
    if not np.isfinite(min_scale) or min_scale <= 0:
        raise ValueError("min_scale must be finite and positive")
    _, specs = _read_manifest(manifest_path)
    count = mean = moment = None
    for spec in specs:
        episode = PressureEpisode.from_npz(spec.pressure)
        values = episode.values.astype(np.float64)
        valid = episode.frame_valid[..., None, None] & episode.taxel_valid & ~episode.saturated & np.isfinite(values)
        local_count = valid.sum(axis=0, dtype=np.int64)
        clean = np.where(valid, values, 0)
        local_mean = clean.sum(axis=0) / np.maximum(local_count, 1)
        residual = np.where(valid, values - local_mean, 0)
        local_moment = np.square(residual).sum(axis=0)
        if count is None:
            count, mean, moment = local_count, local_mean, local_moment
            continue
        if local_count.shape != count.shape:
            raise ValueError("all training pressure episodes must have the same [S,R,C] geometry")
        combined_count = count + local_count
        delta = local_mean - mean
        safe_count = np.maximum(combined_count, 1)
        mean = mean + delta * local_count / safe_count
        moment = moment + local_moment + np.square(delta) * count * local_count / safe_count
        count = combined_count
    scale = np.sqrt(moment / np.maximum(count, 1))
    scale = np.where(count > 0, np.maximum(scale, min_scale), 1)
    mean = np.where(count > 0, mean, 0)
    return {"mean": mean.astype(np.float32), "scale": scale.astype(np.float32), "count": count}
