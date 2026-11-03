"""Pressure-array tokens for the observation prefix, independent of vision encoders."""

from typing import TYPE_CHECKING

import torch
from torch import nn
import torch.nn.functional as F  # noqa: N812

if TYPE_CHECKING:
    from openpi.models.pressure_config import PressureConfig


class PressureEncoder(nn.Module):
    """Encode fixed-size spatial patches without mixing sensors or history frames.

    Values have already been calibrated and scaled with fixed training-data statistics.
    Zero pressure is a valid no-contact measurement. Unavailable, nonfinite, future,
    or stale samples are masked; a saturated but available taxel remains valid.
    Token order is sensor, history, patch row, patch column (row-major).
    """

    def __init__(self, config: "PressureConfig", output_dim: int):
        super().__init__()
        self.config = config
        if output_dim <= 0:
            raise ValueError("output_dim must be positive")
        if config.rows % config.patch_rows or config.cols % config.patch_cols:
            raise ValueError("Pressure rows and cols must be divisible by patch_rows and patch_cols")

        self.patch_grid_rows = config.rows // config.patch_rows
        self.patch_grid_cols = config.cols // config.patch_cols
        self.num_patches = self.patch_grid_rows * self.patch_grid_cols
        self.output_dim = output_dim
        patch_area = config.patch_rows * config.patch_cols

        # Preserve pressure magnitude, availability, and saturation as separate features.
        feature_channels = 5 if config.use_temporal_differences else 3
        self.patch_in = nn.Linear(feature_channels * patch_area, config.hidden_dim)
        self.patch_out = nn.Linear(config.hidden_dim, output_dim)
        self.sensor_embedding = nn.Embedding(config.num_sensors, output_dim)
        self.spatial_embedding = nn.Embedding(self.num_patches, output_dim)
        self.age_in = nn.Linear(1, config.hidden_dim)
        self.age_out = nn.Linear(config.hidden_dim, output_dim)
        nn.init.normal_(self.sensor_embedding.weight, std=0.02)
        nn.init.normal_(self.spatial_embedding.weight, std=0.02)
        if config.taxel_positions_mm:
            # Geometry is reconstructed from the checked configuration, not a mutable checkpoint tensor.
            coordinates = (
                torch.tensor(config.taxel_positions_mm, dtype=torch.float32).reshape(
                    1, config.num_sensors, 1, config.rows, config.cols, 3
                )
                / config.coordinate_scale_mm
            )
            patch_coordinates = torch.stack(
                [self._patchify(coordinates[..., axis]) for axis in range(3)], dim=-1
            ).flatten(start_dim=-2)
            self.register_buffer("patch_coordinates", patch_coordinates, persistent=False)
            self.coordinate_in = nn.Linear(3 * patch_area, config.hidden_dim)
            self.coordinate_out = nn.Linear(config.hidden_dim, output_dim)

    @staticmethod
    def _linear_float32(layer: nn.Linear, inputs: torch.Tensor) -> torch.Tensor:
        # Retain float32 arithmetic even under AMP or an enclosing model dtype cast.
        # Casting a parameter here preserves its autograd connection.
        bias = None if layer.bias is None else layer.bias.float()
        return F.linear(inputs.float(), layer.weight.float(), bias)

    def _patchify(self, values: torch.Tensor) -> torch.Tensor:
        batch_size, sensors, history = values.shape[:3]
        values = values.reshape(
            batch_size,
            sensors,
            history,
            self.patch_grid_rows,
            self.config.patch_rows,
            self.patch_grid_cols,
            self.config.patch_cols,
        )
        return values.permute(0, 1, 2, 3, 5, 4, 6).reshape(batch_size, sensors, history, self.num_patches, -1)

    def _validate(self, pressure: dict[str, torch.Tensor]) -> None:
        required = ("values", "frame_valid", "taxel_valid", "saturated", "age")
        missing = [key for key in required if key not in pressure]
        if missing:
            raise ValueError(f"Pressure observation is missing fields: {missing}")
        for key in required:
            if not isinstance(pressure[key], torch.Tensor):
                raise TypeError(f"pressure[{key!r}] must be a torch.Tensor")

        values = pressure["values"]
        if values.ndim != 5:
            raise ValueError("pressure['values'] must have shape [B, S, K, R, C]")
        expected = (
            values.shape[0],
            self.config.num_sensors,
            self.config.history,
            self.config.rows,
            self.config.cols,
        )
        if tuple(values.shape) != expected:
            raise ValueError(f"pressure['values'] has shape {tuple(values.shape)}, expected {expected}")
        frame_shape = expected[:3]
        for key in ("frame_valid", "age"):
            if tuple(pressure[key].shape) != frame_shape:
                raise ValueError(f"pressure[{key!r}] must have shape {frame_shape}")
        for key in ("taxel_valid", "saturated"):
            if tuple(pressure[key].shape) != expected:
                raise ValueError(f"pressure[{key!r}] must have shape {expected}")
        for key in ("frame_valid", "taxel_valid", "saturated"):
            if pressure[key].dtype != torch.bool:
                raise TypeError(f"pressure[{key!r}] must have dtype torch.bool")
        for key in ("values", "age"):
            if not pressure[key].is_floating_point():
                raise TypeError(f"pressure[{key!r}] must have a floating-point dtype")
        device = self.sensor_embedding.weight.device
        for key in required:
            if pressure[key].device != device:
                raise ValueError(f"pressure[{key!r}] must be on encoder device {device}")

    def forward(self, pressure: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Return float32 tokens [B, S*K*Np, D] and boolean validity [B, S*K*Np]."""
        self._validate(pressure)
        with torch.autocast(device_type=pressure["values"].device.type, enabled=False):
            values = pressure["values"].float()
            age = pressure["age"].float()
            age_valid = torch.isfinite(age) & (age >= 0)
            if self.config.max_age_seconds is not None:
                age_valid = age_valid & (age <= self.config.max_age_seconds)
            frame_valid = pressure["frame_valid"] & age_valid
            taxel_valid = pressure["taxel_valid"] & torch.isfinite(values) & frame_valid[..., None, None]

            # Mask before any learned operation: NaNs multiplied by zero remain NaN.
            values = torch.where(taxel_valid, values, 0.0)
            saturated = pressure["saturated"] & taxel_valid
            patch_values = self._patchify(values)
            patch_valid = self._patchify(taxel_valid)
            patch_saturated = self._patchify(saturated)
            token_valid = patch_valid.any(dim=-1)
            features = [patch_values, patch_valid.float(), patch_saturated.float()]
            if self.config.use_temporal_differences:
                # Values are oldest -> newest; actual age must strictly decrease for a usable pair.
                safe_pair_age = torch.where(frame_valid, age, 0.0)
                gap = safe_pair_age[..., :-1] - safe_pair_age[..., 1:]
                pair_time_valid = torch.isfinite(gap) & (gap > 0)
                if self.config.max_difference_gap_seconds is not None:
                    pair_time_valid &= gap <= self.config.max_difference_gap_seconds
                pair_valid = (
                    taxel_valid[:, :, 1:]
                    & taxel_valid[:, :, :-1]
                    & ~saturated[:, :, 1:]
                    & ~saturated[:, :, :-1]
                    & pair_time_valid[..., None, None]
                )
                # Sanitize both endpoints before subtraction; masked overflow/NaN cannot enter the MLP.
                current = torch.where(pair_valid, values[:, :, 1:], 0.0)
                previous = torch.where(pair_valid, values[:, :, :-1], 0.0)
                delta = current - previous
                pair_valid = pair_valid & torch.isfinite(delta)
                differences = torch.cat(
                    (torch.zeros_like(values[:, :, :1]), torch.where(pair_valid, delta, 0.0)), dim=2
                )
                difference_valid = torch.cat((torch.zeros_like(taxel_valid[:, :, :1]), pair_valid), dim=2)
                features.extend((self._patchify(differences), self._patchify(difference_valid).float()))
            patch_features = torch.stack(features, dim=-1).flatten(start_dim=-2)
            tokens = self._linear_float32(self.patch_in, patch_features)
            tokens = self._linear_float32(self.patch_out, F.silu(tokens))
            if self.config.taxel_positions_mm:
                coordinates = self._linear_float32(self.coordinate_in, self.patch_coordinates)
                tokens = tokens + self._linear_float32(self.coordinate_out, F.silu(coordinates))

            # log1p preserves real age ordering and keeps very large finite ages numerically bounded.
            safe_age = torch.where(frame_valid, age, 0.0)
            age_features = torch.log1p(safe_age)[..., None]
            age_embedding = self._linear_float32(self.age_in, age_features)
            age_embedding = self._linear_float32(self.age_out, F.silu(age_embedding))
            tokens = (
                tokens
                + self.sensor_embedding.weight.float()[None, :, None, None, :]
                + self.spatial_embedding.weight.float()[None, None, None, :, :]
                + age_embedding[..., None, :]
            )
            tokens = torch.where(token_valid[..., None], tokens, 0.0)
            return tokens.flatten(start_dim=1, end_dim=3), token_valid.flatten(start_dim=1)
