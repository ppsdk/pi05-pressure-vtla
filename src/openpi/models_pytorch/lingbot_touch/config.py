from dataclasses import dataclass
from collections.abc import Mapping
import math
from numbers import Integral, Real

import torch


MODALITIES = ("magnetic", "piezoresistive", "visuotactile")
ROPE_MODE = "axial_time_finger_xyz_qk_v1"
ROPE_FIELDS = ("time_theta", "finger_theta", "spatial_theta", "spatial_coordinate_scale")


def positive_float(value, name):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be numeric")
    parsed = float(value)
    fp32 = torch.tensor(parsed, dtype=torch.float32)
    if not math.isfinite(parsed) or parsed <= 0 or not bool(torch.isfinite(fp32)) or not bool(fp32 > 0):
        raise ValueError(f"{name} must be positive finite FP32")
    return parsed


@dataclass(frozen=True)
class EncoderConfig:
    embed_dim: int
    depth: int
    num_heads: int
    tubelet_size: int
    mlp_ratio: float
    output_layers: tuple[int, ...]
    group_by_source: dict[str, str]
    address_rope: dict[str, float | str]
    use_shared_projector: bool

    def __post_init__(self):
        for name in ("embed_dim", "depth", "num_heads", "tubelet_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
            object.__setattr__(self, name, int(value))
        if self.embed_dim % self.num_heads or self.embed_dim // self.num_heads < 10:
            raise ValueError("embed_dim must divide num_heads with head dimension >= 10")
        ratio = positive_float(self.mlp_ratio, "mlp_ratio")
        if int(self.embed_dim * ratio) < 1:
            raise ValueError("mlp_ratio produces an empty hidden layer")
        object.__setattr__(self, "mlp_ratio", ratio)
        layers = self.output_layers
        if not isinstance(layers, (list, tuple)) or not layers or any(isinstance(i, bool) or not isinstance(i, Integral) or i < 0 or i >= self.depth for i in layers):
            raise ValueError("output_layers must contain valid zero-based block indices")
        if list(layers) != sorted(set(layers)):
            raise ValueError("output_layers must be unique and increasing")
        object.__setattr__(self, "output_layers", tuple(int(i) for i in layers))
        routes = self.group_by_source
        if not isinstance(routes, Mapping) or dict(routes) != {name: name for name in MODALITIES}:
            raise ValueError("group_by_source must map magnetic, piezoresistive, and visuotactile to their identically named tokenizer groups")
        object.__setattr__(self, "group_by_source", dict(routes))
        rope = self.address_rope
        if not isinstance(rope, Mapping) or set(rope) != {"mode", *ROPE_FIELDS} or rope["mode"] != ROPE_MODE:
            raise ValueError("address_rope must use the exact axial_time_finger_xyz_qk_v1 schema")
        parsed = {"mode": ROPE_MODE}
        parsed.update({name: positive_float(rope[name], f"address_rope.{name}") for name in ROPE_FIELDS})
        object.__setattr__(self, "address_rope", parsed)
        if not isinstance(self.use_shared_projector, bool):
            raise ValueError("use_shared_projector must be boolean")

    @classmethod
    def from_dict(cls, mapping):
        fields = set(cls.__dataclass_fields__)
        if not isinstance(mapping, Mapping) or set(mapping) != fields:
            raise ValueError(f"config requires exactly {sorted(fields)}")
        return cls(**dict(mapping))

    def to_dict(self):
        return {
            "embed_dim": self.embed_dim,
            "depth": self.depth,
            "num_heads": self.num_heads,
            "tubelet_size": self.tubelet_size,
            "mlp_ratio": self.mlp_ratio,
            "output_layers": list(self.output_layers),
            "group_by_source": dict(self.group_by_source),
            "address_rope": dict(self.address_rope),
            "use_shared_projector": self.use_shared_projector,
        }
