"""LingBot-Touch weights with forearm coordinates, dynamic masks and trainable readout.

The upstream inference package is vendored unchanged in ``lingbot_touch``. We reuse
its exact parameter shapes, but replace the five-finger tokenizer layout and the
unmasked forward pass here. This is a domain adaptation, not a claimed zero-shot
equivalence to fingertip inference.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F  # noqa: N812

from openpi.models_pytorch.lingbot_touch import EncoderConfig
from openpi.models_pytorch.lingbot_touch import TargetEncoder
from openpi.models_pytorch.lingbot_touch.address import TokenAddress
from openpi.models_pytorch.pressure_pytorch import PressureEncoder
from openpi.pressure_data import NormalizePressure


class LingBotForearmEncoder(nn.Module):
    def __init__(self, config, output_dim: int, state_dim: int):
        super().__init__()
        self.config = config
        self.settings = config.lingbot_touch
        self.output_dim = output_dim
        self.state_dim = state_dim
        architecture = EncoderConfig.from_dict(self.settings.encoder_config)
        self.pretrained = TargetEncoder(architecture)
        # The unused modality tokenizers and unused final norm remain frozen.
        if not self.settings.freeze_pretrained:
            self.pretrained.tokenizer.tokenizers["piezoresistive"].requires_grad_(requires_grad=True)
            self.pretrained.transformer.blocks.requires_grad_(requires_grad=True)
            self.pretrained.transformer.norms_block.requires_grad_(requires_grad=True)
            if self.pretrained.shared_projector is not None:
                self.pretrained.shared_projector.requires_grad_(requires_grad=True)
        self.tubelet = architecture.tubelet_size
        dim = architecture.embed_dim
        self.sensor_embedding = nn.Embedding(config.num_sensors, dim)
        nn.init.normal_(self.sensor_embedding.weight, std=0.02)
        # Quality, real age, within-tubelet duration and optional temporal difference.
        self.quality_adapter = nn.Sequential(nn.Linear(6, dim), nn.SiLU(), nn.Linear(dim, dim))
        nn.init.zeros_(self.quality_adapter[-1].weight)
        nn.init.zeros_(self.quality_adapter[-1].bias)
        self.output_projection = nn.Linear(self.pretrained.target_dim, output_dim)
        readout_dim = self.settings.readout_dim
        self.readout_queries = nn.Parameter(torch.randn(self.settings.readout_queries, readout_dim) * 0.02)
        self.state_query = nn.Linear(state_dim, readout_dim)
        self.readout_key = nn.Linear(output_dim, readout_dim)
        self.readout_value = nn.Linear(output_dim, readout_dim)
        self.readout_out = nn.Linear(readout_dim, output_dim)
        shape = (1, config.num_sensors, 1, config.rows, config.cols)
        self.register_buffer("input_mean", torch.zeros(shape))
        self.register_buffer("input_scale", torch.ones(shape))
        self.register_buffer("normalization_ready", torch.zeros((), dtype=torch.bool))
        self.register_buffer("pretrained_ready", torch.zeros((), dtype=torch.bool))
        self.register_buffer("source_sha256", torch.zeros(32, dtype=torch.uint8))
        # Geometry is an immutable checkpoint configuration, not a learned buffer.
        positions = torch.tensor(config.taxel_positions_mm, dtype=torch.float32) / config.coordinate_scale_mm
        self.register_buffer("positions", positions, persistent=False)
        self.register_buffer(
            "signal_zero", torch.tensor(self.settings.sensor_zero).reshape(1, -1, 1, 1, 1), persistent=False
        )
        self.register_buffer(
            "signal_span", torch.tensor(self.settings.sensor_span).reshape(1, -1, 1, 1, 1), persistent=False
        )

    def initialize_from_bundle(self, bundle_dir: str | Path) -> None:
        """Only initialization reads external assets; full checkpoints are self-contained."""
        bundle_dir = Path(bundle_dir)
        architecture = EncoderConfig.from_dict(json.loads((bundle_dir / "config.json").read_text(encoding="utf-8")))
        if architecture.to_dict() != self.pretrained.config.to_dict():
            raise ValueError("LingBot bundle architecture differs from the resolved model configuration")
        weight_path = bundle_dir / "encoder.pt"
        state = torch.load(weight_path, map_location="cpu", weights_only=True)
        expected = self.pretrained.state_dict()
        if not isinstance(state, dict) or set(state) != set(expected):
            raise ValueError("LingBot encoder.pt must contain exactly the official encoder keys")
        for name, target in expected.items():
            value = state[name]
            if (
                not isinstance(value, torch.Tensor)
                or value.shape != target.shape
                or value.dtype != target.dtype
                or value.layout != torch.strided
                or not torch.isfinite(value).all()
            ):
                raise ValueError(f"Invalid LingBot pretrained tensor: {name}")
        self.pretrained.load_state_dict(state, strict=True)
        with weight_path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").digest()
        self.source_sha256.copy_(torch.tensor(list(digest), dtype=torch.uint8, device=self.source_sha256.device))
        self.pretrained_ready.fill_(1)

    def bind_normalization(self, stats_path: str | Path) -> None:
        """Undo existing training normalization before fixed sensor-range conversion."""
        stats = NormalizePressure.from_npz(stats_path)
        shape = tuple(self.input_mean.shape)
        for name in ("mean", "scale"):
            expanded = NormalizePressure._expand(getattr(stats, name), shape)  # noqa: SLF001
            values = np.broadcast_to(expanded, shape).copy()
            getattr(self, f"input_{name}").copy_(torch.as_tensor(values, device=self.input_mean.device))
        self.normalization_ready.fill_(1)

    def assert_ready(self) -> None:
        if not bool(self.pretrained_ready):
            raise RuntimeError("Initialize LingBot from an official bundle or load a complete trained checkpoint")
        if not bool(self.normalization_ready):
            raise RuntimeError("LingBot requires bound pressure normalization statistics")

    def _tokenize(self, pressure):
        # Reuse the existing five-field validation and device/shape contract.
        PressureEncoder._validate(self, pressure)  # noqa: SLF001
        values, age = pressure["values"].float(), pressure["age"].float()
        age_valid = torch.isfinite(age) & (age >= 0)
        if self.config.max_age_seconds is not None:
            age_valid &= age <= self.config.max_age_seconds
        frame_valid = pressure["frame_valid"] & age_valid
        valid = pressure["taxel_valid"] & torch.isfinite(values) & frame_valid[..., None, None]
        values = torch.where(valid, values, 0.0)
        # Training standardization is reversed exactly once. No per-window scaling.
        raw = values * self.input_scale.float() + self.input_mean.float()
        z = (raw - self.signal_zero.float()) / self.signal_span.float()
        valid &= torch.isfinite(z)
        z = torch.where(valid, z, 0.0).clamp(0, 1)
        safe_age = torch.where(frame_valid, age, 0.0)
        saturated = pressure["saturated"] & valid
        batch, sensors, frames, rows, cols = values.shape
        points, groups = rows * cols, frames // self.tubelet

        def tubelets(tensor):
            return tensor.reshape(batch, sensors, groups, self.tubelet, points).permute(0, 1, 2, 4, 3)

        z = tubelets(z)
        measured = tubelets(valid)
        sat = tubelets(saturated)
        ages = safe_age.reshape(batch, sensors, groups, self.tubelet)
        # Reject partial tubelets instead of pretending missing frames measured zero contact.
        token_valid = measured.all(-1)
        gaps = ages[..., :-1] - ages[..., 1:]
        ordered = (gaps > 0) & torch.isfinite(gaps)
        if self.config.max_difference_gap_seconds is not None:
            ordered &= gaps <= self.config.max_difference_gap_seconds
        token_valid &= ordered.all(-1)[..., None]
        signal = torch.stack((torch.zeros_like(z), torch.zeros_like(z), z), -1).flatten(-2)
        signal = torch.where(token_valid[..., None], signal, 0.0)
        tokenizer = self.pretrained.tokenizer.tokenizers["piezoresistive"]
        tokens = tokenizer.signal_mlp(signal)
        if self.pretrained.shared_projector is not None:
            tokens = self.pretrained.shared_projector(tokens)
        duration = ages[..., 0] - ages[..., -1]
        mean_age = (ages / self.tubelet).sum(-1)
        time_coordinate = -mean_age / (self.settings.time_scale_seconds * self.tubelet)
        token_valid &= torch.isfinite(time_coordinate)[..., None]
        delta_valid = token_valid & ~sat.any(-1) & (self.tubelet > 1) & self.config.use_temporal_differences
        delta = torch.where(delta_valid, z[..., -1] - z[..., 0], 0.0)
        metadata = torch.stack(
            (
                sat.float().mean(-1),
                measured.float().mean(-1),
                torch.log1p(mean_age)[..., None].expand_as(delta),
                torch.log1p(duration.clamp_min(0))[..., None].expand_as(delta),
                delta,
                delta_valid.float(),
            ),
            -1,
        )
        metadata = torch.where(token_valid[..., None], metadata, 0.0)
        tokens = tokens + self.quality_adapter(metadata)
        tokens = tokens + self.sensor_embedding.weight.float()[None, :, None, None, :]
        tokens = tokens.flatten(1, 3)
        token_valid = token_valid.flatten(1, 3)
        tokens = torch.where(token_valid[..., None], tokens, 0.0)
        xyz = self.positions.float()[None, :, None].expand(batch, sensors, groups, points, 3).flatten(1, 3)
        time = time_coordinate[..., None].expand(batch, sensors, groups, points).flatten(1, 3)
        time = torch.where(token_valid, time, 0.0)
        # No finger-index prior: skin identity is a learned embedding, geometry is forearm-local XYZ.
        address = TokenAddress(time_id=time, finger_id=torch.zeros_like(time), pos_xyz=xyz)
        return tokens, token_valid, address

    def forward(self, pressure):
        self.assert_ready()
        with torch.autocast(device_type=self.input_mean.device.type, enabled=False):
            x, valid, address = self._tokenize(pressure)
            outputs = []
            transformer = self.pretrained.transformer
            # Invalid observations cannot become keys/values in any pretrained block.
            attention_mask = valid[:, None, :, None] & valid[:, None, None, :]
            for index, block in enumerate(transformer.blocks):
                attention = block.attn
                batch, count, dim = x.shape
                qkv = (
                    attention.qkv(block.norm1(x))
                    .reshape(batch, count, 3, attention.num_heads, attention.head_dim)
                    .permute(2, 0, 3, 1, 4)
                )
                q, k, v = qkv.unbind(0)
                q, k = attention.address_rope(q, k, address)
                attended = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask, dropout_p=0.0)
                x = x + attention.proj(attended.transpose(1, 2).reshape(batch, count, dim))
                x = x + block.mlp(block.norm2(x))
                x = torch.where(valid[..., None], x, 0.0)
                if index in transformer.output_layers:
                    outputs.append(transformer.norms_block[transformer.output_layers.index(index)](x))
            tokens = self.output_projection(torch.cat(outputs, dim=-1))
            return torch.where(valid[..., None], tokens, 0.0), valid

    def pool(self, tokens, valid, state):
        """State-conditioned queries replace location-erasing mean pooling."""
        if state is None or state.shape != (tokens.shape[0], self.state_dim):
            raise ValueError("LingBot tactile readout requires the normalized robot state")
        with torch.autocast(device_type=tokens.device.type, enabled=False):
            q = self.readout_queries.float()[None] + self.state_query(state.float())[:, None]
            k, v = self.readout_key(tokens.float()), self.readout_value(tokens.float())
            out = F.scaled_dot_product_attention(
                q[:, None], k[:, None], v[:, None], attn_mask=valid[:, None, None, :], dropout_p=0.0
            )
            pooled = self.readout_out(out[:, 0].mean(1))
            return torch.where(valid.any(-1, keepdim=True), pooled, 0.0)


def initialize_lingbot_forearm(model, *, bundle_path=None, stats_path=None):
    """Called once by training; resume/inference use tensors from the full checkpoint."""
    encoder = getattr(model, "pressure_encoder", None)
    if not isinstance(encoder, LingBotForearmEncoder):
        if bundle_path is not None:
            raise ValueError("lingbot_touch_bundle_path requires a LingBot forearm encoder")
        return
    if bundle_path is not None:
        encoder.initialize_from_bundle(bundle_path)
    if stats_path is not None:
        encoder.bind_normalization(stats_path)
    encoder.assert_ready()
