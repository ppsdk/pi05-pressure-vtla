"""CPU smoke test with real LingBot weights and explicitly synthetic forearm inputs."""

import argparse
import dataclasses
import json
from pathlib import Path
import tempfile

import numpy as np
import safetensors.torch
import torch

from openpi.models.pressure_config import LingBotTouchConfig
from openpi.models.pressure_config import PressureConfig
from openpi.models_pytorch.lingbot_forearm import LingBotForearmEncoder
from openpi.models_pytorch.lingbot_touch import load_encoder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(17)
    reference = load_encoder(args.bundle)
    settings = LingBotTouchConfig(
        sensor_zero=(0,), sensor_span=(1,), time_scale_seconds=0.01, encoder_config=reference.config.to_dict()
    )
    config = PressureConfig(
        num_sensors=1,
        rows=1,
        cols=3,
        history=4,
        patch_rows=1,
        patch_cols=1,
        taxel_positions_mm=(((0, 0, 0), (10, 0, 0), (40, 5, 0)),),
        lingbot_touch=settings,
    )
    encoder = LingBotForearmEncoder(config, output_dim=2048, state_dim=32)
    encoder.initialize_from_bundle(args.bundle)
    with tempfile.TemporaryDirectory() as scratch:
        stats = Path(scratch) / "stats.npz"
        np.savez(stats, mean=np.array(0), scale=np.array(1))
        encoder.bind_normalization(stats)
        pressure = {
            "values": torch.rand(1, 1, 4, 1, 3),
            "frame_valid": torch.ones(1, 1, 4, dtype=torch.bool),
            "taxel_valid": torch.ones(1, 1, 4, 1, 3, dtype=torch.bool),
            "saturated": torch.zeros(1, 1, 4, 1, 3, dtype=torch.bool),
            "age": torch.tensor([[[0.03, 0.02, 0.01, 0.0]]]),
        }
        # Matching a single real finger in the reference tests weight/operation parity,
        # not zero-shot forearm performance. Remove only the added skin embedding.
        with torch.no_grad():
            encoder.sensor_embedding.weight.zero_()
        signal = torch.zeros(1, 4, 5, 3, 3)
        signal[:, :, 0, :, 2] = pressure["values"][:, 0, :, 0]
        positions = torch.zeros(5, 3, 3)
        positions[0] = encoder.positions[0]
        coverage = torch.zeros(5, 3, dtype=torch.bool)
        coverage[0] = True
        expected = encoder.output_projection(reference(signal, positions, coverage, modality="piezoresistive").latent)
        tokens, valid = encoder(pressure)
        parity_error = (expected - tokens).abs().max().item()
        torch.testing.assert_close(tokens, expected, atol=1e-5, rtol=1e-4)
        pooled = encoder.pool(tokens, valid, torch.zeros(1, 32))
        pooled.square().mean().backward()
        assert encoder.quality_adapter[-1].weight.grad.abs().sum() > 0
        assert all(p.grad is None for p in encoder.pretrained.parameters())
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in encoder.parameters())
        path = Path(scratch) / "adapted.safetensors"
        safetensors.torch.save_model(encoder, path)
        restored = LingBotForearmEncoder(PressureConfig.from_dict(config.to_dict()), 2048, 32)
        safetensors.torch.load_model(restored, path)
        torch.testing.assert_close(restored(pressure)[0], tokens)
        # Exercise the real fine-tuning path as well as the default frozen path.
        trainable_config = dataclasses.replace(
            config, lingbot_touch=dataclasses.replace(settings, freeze_pretrained=False)
        )
        trainable = LingBotForearmEncoder(trainable_config, 2048, 32)
        trainable.load_state_dict(encoder.state_dict())
        t, m = trainable(pressure)
        trainable.pool(t, m, torch.zeros(1, 32)).square().mean().backward()
        assert trainable.pretrained.transformer.blocks[0].attn.qkv.weight.grad.abs().sum() > 0
        report = {
            "input_kind": "synthetic forearm signals and geometry; not robot data",
            "torch": torch.__version__,
            "device": "cpu",
            "source_sha256": bytes(encoder.source_sha256.tolist()).hex(),
            "pretrained_parameters": sum(p.numel() for p in encoder.pretrained.parameters()),
            "token_shape": list(tokens.shape),
            "reference_max_abs_error": parity_error,
            "frozen_adapter_backward": "passed",
            "pretrained_finetune_backward": "passed",
            "self_contained_checkpoint": "passed",
        }
    text = json.dumps(report, indent=2)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
