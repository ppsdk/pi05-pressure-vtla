"""Hardware-explicit configuration for the optional PyTorch pressure prefix."""

import dataclasses
import math


@dataclasses.dataclass(frozen=True)
class PressureConfig:
    # No hardware defaults: these must match the actual sensor layout.
    num_sensors: int
    rows: int
    cols: int
    history: int = 1
    patch_rows: int = 2
    patch_cols: int = 2
    hidden_dim: int = 128
    max_age_seconds: float | None = None

    def __post_init__(self):
        for name in ("num_sensors", "rows", "cols", "history", "patch_rows", "patch_cols", "hidden_dim"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.rows % self.patch_rows or self.cols % self.patch_cols:
            raise ValueError("Pressure rows/cols must be divisible by patch_rows/patch_cols")
        if self.max_age_seconds is not None and (not math.isfinite(self.max_age_seconds) or self.max_age_seconds < 0):
            raise ValueError("max_age_seconds must be finite and nonnegative")

    @property
    def tokens_per_frame(self) -> int:
        return (self.rows // self.patch_rows) * (self.cols // self.patch_cols)

    @property
    def num_tokens(self) -> int:
        return self.num_sensors * self.history * self.tokens_per_frame
