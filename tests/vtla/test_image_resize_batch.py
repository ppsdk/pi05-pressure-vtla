"""Test the real torch resize implementation without loading the optional JAX stack."""

import ast
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F  # noqa: N812


def load_torch_resize():
    source = Path(__file__).resolve().parents[2] / "src/openpi/shared/image_tools.py"
    module = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    function = next(
        node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "resize_with_pad_torch"
    )
    env = {"torch": torch, "F": F}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), env)
    return env["resize_with_pad_torch"]


@pytest.mark.parametrize("batch", [None, 1, 2])
@pytest.mark.parametrize("channels_last", [False, True])
def test_resize_preserves_original_batch_and_layout(batch, channels_last):
    spatial = (12, 16, 3) if channels_last else (3, 12, 16)
    image = torch.rand(spatial if batch is None else (batch, *spatial), dtype=torch.float32)
    output = load_torch_resize()(image, 6, 8)
    expected = (6, 8, 3) if channels_last else (3, 6, 8)
    assert output.shape == (expected if batch is None else (batch, *expected))
    assert torch.isfinite(output).all()


def test_resize_rejects_unsupported_rank():
    with pytest.raises(ValueError, match="unbatched or batched"):
        load_torch_resize()(torch.zeros(2, 3), 6, 8)
