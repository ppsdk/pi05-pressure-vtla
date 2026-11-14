from collections.abc import Mapping
import json
from pathlib import Path

import torch

from .config import EncoderConfig
from .model import TargetEncoder


def load_encoder(bundle_dir, device="cpu"):
    bundle_dir = Path(bundle_dir)
    config = EncoderConfig.from_dict(json.loads((bundle_dir / "config.json").read_text()))
    encoder = TargetEncoder(config)
    state = torch.load(bundle_dir / "encoder.pt", map_location="cpu", weights_only=True)
    expected = encoder.state_dict()
    if not isinstance(state, Mapping) or set(state) != set(expected):
        raise ValueError("encoder.pt must contain exactly the target encoder state keys")
    for key, target in expected.items():
        value = state[key]
        if not isinstance(value, torch.Tensor) or value.layout != torch.strided or value.shape != target.shape or value.dtype != target.dtype:
            raise ValueError(f"invalid encoder state tensor shape or dtype: {key}")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"nonfinite encoder state tensor: {key}")
    encoder.load_state_dict(state, strict=True)
    return encoder.to(device=device).eval()
