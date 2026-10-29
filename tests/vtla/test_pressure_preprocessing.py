"""Exercise actual pressure-aware RGB preprocessing without importing the JAX stack."""

import importlib.util
from pathlib import Path
import sys
from types import ModuleType
from types import SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize("channels_first", [False, True])
def test_pressure_and_camera_layout_are_preserved(monkeypatch, channels_first):
    image_tools = ModuleType("openpi.shared.image_tools")
    monkeypatch.setitem(sys.modules, "openpi.shared.image_tools", image_tools)
    path = Path(__file__).resolve().parents[2] / "src/openpi/models_pytorch/preprocessing_pytorch.py"
    spec = importlib.util.spec_from_file_location("_pressure_preprocessing_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    images = torch.rand(2, 3, 224, 224) if channels_first else torch.rand(2, 224, 224, 3)
    pressure = {"values": torch.zeros(2, 1, 1, 4, 4)}
    observation = SimpleNamespace(
        images=dict.fromkeys(module.IMAGE_KEYS, images),
        image_masks={key: torch.ones(2, dtype=torch.bool) for key in module.IMAGE_KEYS},
        state=torch.zeros(2, 3),
        tokenized_prompt=None,
        tokenized_prompt_mask=None,
        token_ar_mask=None,
        token_loss_mask=None,
        pressure=pressure,
    )
    output = module.preprocess_observation_pytorch(observation, train=False)
    assert output.pressure is pressure
    expected = images if channels_first else images.permute(0, 3, 1, 2)
    for image in output.images.values():
        torch.testing.assert_close(image, expected)
