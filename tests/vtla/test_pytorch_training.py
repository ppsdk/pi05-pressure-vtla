"""Optimizer updates and checkpoint guards on small CPU models."""

import copy

import numpy as np
import pytest
import torch
from torch import nn

from openpi.training.pytorch_training import build_optimizer_param_groups
from openpi.training.pytorch_training import load_optimizer_state
from openpi.training.pytorch_training import prepare_checkpoint_directory
from openpi.training.pytorch_training import set_learning_rate


def _model():
    model = nn.Module()
    model.pressure_encoder = nn.Module()
    model.pressure_encoder.pretrained = nn.Linear(2, 2, bias=False)
    model.adapter = nn.Linear(2, 2, bias=False)
    model.base = nn.Linear(2, 2, bias=False).requires_grad_(requires_grad=False)
    return model


def _optimizer(model, scale=0.1):
    return torch.optim.AdamW(build_optimizer_param_groups(model, lr=0.01, lingbot_pretrained_lr_scale=scale))


def _update(model, optimizer):
    optimizer.zero_grad(set_to_none=True)
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.grad = torch.ones_like(parameter)
    optimizer.step()


def test_learning_rate_ratio_survives_schedule_and_controls_actual_updates():
    model = _model()
    nn.init.zeros_(model.adapter.weight)
    nn.init.zeros_(model.pressure_encoder.pretrained.weight)
    base = model.base.weight.detach().clone()
    optimizer = _optimizer(model)
    set_learning_rate(optimizer, 0.02)
    _update(model, optimizer)
    torch.testing.assert_close(model.adapter.weight, model.pressure_encoder.pretrained.weight * 10)
    torch.testing.assert_close(model.base.weight, base)
    optimized = [id(p) for group in optimizer.param_groups for p in group["params"]]
    assert len(optimized) == len(set(optimized)) == 2


def test_optimizer_resume_matches_uninterrupted_next_update():
    model = _model()
    optimizer = _optimizer(model)
    _update(model, optimizer)
    checkpoint = copy.deepcopy(optimizer.state_dict())
    restored = copy.deepcopy(model)
    resumed = _optimizer(restored)
    load_optimizer_state(resumed, checkpoint)
    for instance, opt in ((model, optimizer), (restored, resumed)):
        set_learning_rate(opt, 0.003)
        _update(instance, opt)
    for actual, expected in zip(restored.parameters(), model.parameters(), strict=True):
        torch.testing.assert_close(actual, expected)
    assert resumed.param_groups[1]["lr"] == pytest.approx(0.0003)


def test_numpy_schedule_produces_portable_optimizer_checkpoint(tmp_path):
    model = _model()
    optimizer = _optimizer(model)
    set_learning_rate(optimizer, np.float64(0.02))
    _update(model, optimizer)
    checkpoint = tmp_path / "optimizer.pt"
    torch.save(optimizer.state_dict(), checkpoint)
    restored = _optimizer(model)
    load_optimizer_state(restored, torch.load(checkpoint, weights_only=True))
    assert restored.param_groups[1]["lr"] == pytest.approx(0.002)


@pytest.mark.parametrize("change", ["scale", "order", "betas", "freeze"])
def test_optimizer_resume_rejects_training_contract_changes(change):
    model = _model()
    optimizer = _optimizer(model)
    state = copy.deepcopy(optimizer.state_dict())
    if change == "scale":
        state["param_groups"][1]["lr_scale"] = 0.5
    elif change == "order":
        state["param_groups"][0]["param_names"] = ["unrelated.weight"]
    elif change == "betas":
        state["param_groups"][0]["betas"] = (0.8, 0.9)
    else:
        model.pressure_encoder.pretrained.requires_grad_(requires_grad=False)
        optimizer = _optimizer(model)
    with pytest.raises(ValueError, match="Optimizer"):
        load_optimizer_state(optimizer, state)


def test_legacy_optimizer_resume_retains_new_contract(caplog):
    model = _model()
    old = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.01)
    _update(model, old)
    current = _optimizer(model, scale=1.0)
    load_optimizer_state(current, old.state_dict())
    assert "no parameter-name contract" in caplog.text
    assert current.param_groups[0]["lr_scale"] == 1.0
    assert len(current.param_groups[0]["param_names"]) == 2


@pytest.mark.parametrize("scale", [0, -1, float("nan"), float("inf"), True])
def test_invalid_lr_multiplier_is_rejected(scale):
    with pytest.raises(ValueError, match="finite and positive"):
        _optimizer(_model(), scale=scale)


def test_lr_multiplier_requires_lingbot_and_respects_freezing():
    with pytest.raises(ValueError, match="requires a LingBot"):
        _optimizer(nn.Linear(2, 2))
    model = _model()
    model.pressure_encoder.pretrained.requires_grad_(requires_grad=False)
    optimizer = _optimizer(model)
    assert len(optimizer.param_groups) == 1
    assert optimizer.param_groups[0]["param_names"] == ["adapter.weight"]


def test_existing_experiment_requires_explicit_resume_or_overwrite(tmp_path):
    experiment = tmp_path / "experiment"
    prepare_checkpoint_directory(experiment, resume=False, overwrite=False)
    marker = experiment / "keep.txt"
    marker.write_text("existing run")
    with pytest.raises(FileExistsError, match="already exists"):
        prepare_checkpoint_directory(experiment, resume=False, overwrite=False)
    prepare_checkpoint_directory(experiment, resume=True, overwrite=False)
    assert marker.read_text() == "existing run"
    prepare_checkpoint_directory(experiment, resume=False, overwrite=True)
    assert list(experiment.iterdir()) == []


def test_resume_rejects_missing_or_conflicting_directory_options(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        prepare_checkpoint_directory(tmp_path / "missing", resume=True, overwrite=False)
    with pytest.raises(ValueError, match="Cannot resume and overwrite"):
        prepare_checkpoint_directory(tmp_path, resume=True, overwrite=True)
