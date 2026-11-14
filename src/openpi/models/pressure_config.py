"""Hardware-explicit configuration for the optional PyTorch pressure prefix."""

import dataclasses
import json
import math
from pathlib import Path
from typing import Literal


@dataclasses.dataclass(frozen=True)
class LingBotTouchConfig:
    """Forearm adaptation contract; ranges are in the original dataset signal units."""

    sensor_zero: tuple[float, ...]
    # Signed full-contact excursion: negative for resistance decreasing with load.
    sensor_span: tuple[float, ...]
    time_scale_seconds: float
    freeze_pretrained: bool = True
    readout_queries: int = 4
    readout_dim: int = 128
    # Official Tiny inference architecture, not randomly selected hyperparameters.
    encoder_config: dict = dataclasses.field(
        default_factory=lambda: {
            "embed_dim": 192,
            "depth": 12,
            "num_heads": 3,
            "tubelet_size": 2,
            "mlp_ratio": 4.0,
            "output_layers": [2, 5, 8, 11],
            "group_by_source": {name: name for name in ("magnetic", "piezoresistive", "visuotactile")},
            "address_rope": {
                "mode": "axial_time_finger_xyz_qk_v1",
                "time_theta": 10000.0,
                "finger_theta": 10000.0,
                "spatial_theta": 100.0,
                "spatial_coordinate_scale": math.pi / 2,
            },
            "use_shared_projector": False,
        }
    )

    def __post_init__(self):
        for name in ("sensor_zero", "sensor_span"):
            values = tuple(getattr(self, name))
            if not values or any(
                isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or abs(v) > 1e30
                for v in values
            ):
                raise ValueError(f"{name} requires finite sensor calibration values")
            if name == "sensor_span" and any(abs(v) < 1e-30 for v in values):
                raise ValueError("sensor_span must be nonzero")
            object.__setattr__(self, name, values)
        if (
            isinstance(self.time_scale_seconds, bool)
            or not isinstance(self.time_scale_seconds, (int, float))
            or not math.isfinite(self.time_scale_seconds)
            or self.time_scale_seconds < 1e-9
        ):
            raise ValueError("time_scale_seconds must be finite and >= 1e-9")
        if not isinstance(self.freeze_pretrained, bool):
            raise ValueError("freeze_pretrained must be bool")
        for name in ("readout_queries", "readout_dim"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        # Copy caller-owned dictionaries; full architecture validation is done by EncoderConfig.
        object.__setattr__(self, "encoder_config", json.loads(json.dumps(self.encoder_config, allow_nan=False)))


@dataclasses.dataclass(frozen=True)
class PressureConfig:
    # No hardware defaults: these must match the actual sensor layout.
    num_sensors: int
    rows: int
    cols: int
    # Stable slot metadata for mixed gripper + forearm deployments.
    sensor_names: tuple[str, ...] = ()
    sensor_locations: tuple[str, ...] = ()
    history: int = 1
    patch_rows: int = 2
    patch_cols: int = 2
    hidden_dim: int = 128
    max_age_seconds: float | None = None
    # Metadata describes raw inputs before fixed training-set normalization.
    signal_type: Literal["raw", "capacitance", "resistance", "pressure", "force"] = "raw"
    signal_unit: str = "raw"
    sample_period_seconds: float | None = None
    # Optional forearm-local XYZ, millimetres, [S,R*C,3] in row-major order.
    taxel_positions_mm: tuple[tuple[tuple[float, float, float], ...], ...] = ()
    coordinate_scale_mm: float = 100.0
    use_temporal_differences: bool = False
    max_difference_gap_seconds: float | None = None
    lingbot_touch: LingBotTouchConfig | None = None

    def __post_init__(self):
        for name in ("num_sensors", "rows", "cols", "history", "patch_rows", "patch_cols", "hidden_dim"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.rows % self.patch_rows or self.cols % self.patch_cols:
            raise ValueError("Pressure rows/cols must be divisible by patch_rows/patch_cols")
        for name, entries in (("sensor_names", self.sensor_names), ("sensor_locations", self.sensor_locations)):
            if not isinstance(entries, (list, tuple)):
                raise ValueError(f"{name} must be a list or tuple of strings")
            values = tuple(entries)
            object.__setattr__(self, name, values)
            if values and len(values) != self.num_sensors:
                raise ValueError(f"{name} must be empty or have num_sensors entries")
            if any(not isinstance(value, str) or not value.strip() for value in values):
                raise ValueError(f"{name} entries must be nonempty strings")
        if self.sensor_names and len(set(self.sensor_names)) != len(self.sensor_names):
            raise ValueError("sensor_names must be unique")
        if self.max_age_seconds is not None and (
            isinstance(self.max_age_seconds, bool)
            or not isinstance(self.max_age_seconds, (int, float))
            or not math.isfinite(self.max_age_seconds)
            or self.max_age_seconds < 0
        ):
            raise ValueError("max_age_seconds must be finite and nonnegative")
        if self.signal_type not in ("raw", "capacitance", "resistance", "pressure", "force"):
            raise ValueError("Unsupported signal_type")
        if not isinstance(self.signal_unit, str) or not self.signal_unit.strip():
            raise ValueError("signal_unit must be a nonempty string")
        for name in ("coordinate_scale_mm", "sample_period_seconds", "max_difference_gap_seconds"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0
            ):
                raise ValueError(f"{name} must be finite and positive")
        if self.coordinate_scale_mm is None or not 1e-6 <= self.coordinate_scale_mm <= 1e9:
            raise ValueError("coordinate_scale_mm must keep taxel_positions_mm normalized and finite")
        if not isinstance(self.use_temporal_differences, bool):
            raise ValueError("use_temporal_differences must be bool")
        if self.use_temporal_differences and self.history < 2:
            raise ValueError("Temporal differences require history >= 2")
        try:
            positions = tuple(tuple(tuple(point) for point in sensor) for sensor in self.taxel_positions_mm)
            if positions and (
                len(positions) != self.num_sensors
                or any(len(sensor) != self.rows * self.cols for sensor in positions)
                or any(
                    len(point) != 3
                    or any(
                        isinstance(value, bool)
                        or not isinstance(value, (int, float))
                        or not math.isfinite(value)
                        or abs(value) > 1e9
                        for value in point
                    )
                    for sensor in positions
                    for point in sensor
                )
            ):
                raise ValueError
        except (TypeError, ValueError) as error:
            raise ValueError("taxel_positions_mm must be finite [S,R*C,3] coordinates in millimetres") from error
        object.__setattr__(self, "taxel_positions_mm", positions)
        if isinstance(self.lingbot_touch, dict):
            object.__setattr__(self, "lingbot_touch", LingBotTouchConfig(**self.lingbot_touch))
        if self.lingbot_touch is not None:
            cfg = self.lingbot_touch
            if not isinstance(cfg, LingBotTouchConfig):
                raise ValueError("lingbot_touch must be a LingBotTouchConfig or object")
            if not positions:
                raise ValueError("LingBot forearm encoding requires real taxel_positions_mm")
            if self.patch_rows != 1 or self.patch_cols != 1:
                raise ValueError("LingBot forearm encoding requires 1x1 patches")
            if len(cfg.sensor_zero) != self.num_sensors or len(cfg.sensor_span) != self.num_sensors:
                raise ValueError("LingBot calibration must have num_sensors entries")
            tubelet = cfg.encoder_config.get("tubelet_size")
            if isinstance(tubelet, bool) or not isinstance(tubelet, int) or tubelet < 1 or self.history % tubelet:
                raise ValueError("history must be divisible by LingBot tubelet_size")

    @classmethod
    def from_dict(cls, content: dict) -> "PressureConfig":
        """Load strict model settings, also accepting nested lists from JSON."""
        if not isinstance(content, dict):
            raise ValueError("Pressure config must be a JSON object")
        unknown = set(content) - {field.name for field in dataclasses.fields(cls)}
        if unknown:
            raise ValueError(f"Unknown pressure config fields: {sorted(unknown)}")
        try:
            return cls(**content)
        except TypeError as error:
            raise ValueError(f"Invalid pressure config: {error}") from error

    @classmethod
    def from_json(cls, path: str | Path) -> "PressureConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def to_dict(self) -> dict:
        """Canonical JSON-compatible contract, including nested geometry."""
        return json.loads(json.dumps(dataclasses.asdict(self), allow_nan=False))

    @property
    def tokens_per_frame(self) -> int:
        return (self.rows // self.patch_rows) * (self.cols // self.patch_cols)

    @property
    def num_tokens(self) -> int:
        tubelet = self.lingbot_touch.encoder_config["tubelet_size"] if self.lingbot_touch else 1
        return self.num_sensors * (self.history // tubelet) * self.tokens_per_frame


def resolve_pressure_config(model_config, path: str | Path | None):
    """Resolve optional JSON before constructing data transforms, models or checkpoints."""
    if path is None:
        return model_config
    if getattr(model_config, "pressure", None) is None:
        raise ValueError("pressure_config_path requires a pressure-enabled model")
    return dataclasses.replace(model_config, pressure=PressureConfig.from_json(path))
