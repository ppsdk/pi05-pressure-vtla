"""Exercise real PI0Pytorch integration methods with a small CPU attention harness.

The backbone below is a two-layer test double, not an official PaliGemma/Gemma
model. These tests check dataflow, gradients, masks, positions, cache parity, and
serialization without downloading weights or importing JAX/Flax/Transformers.
They do not establish numerical parity with official weights or task performance.
"""

import importlib.util
import math
from pathlib import Path
import sys
import types

import pytest
import torch
from torch import nn

_SOURCE_ROOT = Path(__file__).resolve().parents[2] / "src"


def _load_source(monkeypatch, module_name, relative_path):
    spec = importlib.util.spec_from_file_location(module_name, _SOURCE_ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


class _AttentionLayer(nn.Module):
    def __init__(self, width, attention_dim=8):
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.q_proj = nn.Linear(width, attention_dim)
        self.self_attn.k_proj = nn.Linear(width, attention_dim)
        self.self_attn.v_proj = nn.Linear(width, attention_dim)
        self.out_proj = nn.Linear(attention_dim, width)
        self.ffn = nn.Linear(width, width)
        self.condition = nn.Linear(width, width, bias=False)

    def project(self, hidden, condition):
        if condition is not None:
            hidden = hidden + self.condition(condition)[:, None, :]
        return (
            self.self_attn.q_proj(hidden),
            self.self_attn.k_proj(hidden),
            self.self_attn.v_proj(hidden),
        )

    def finish(self, hidden, attended):
        hidden = hidden + self.out_proj(attended)
        return hidden + 0.1 * self.ffn(torch.tanh(hidden))


class _AttentionBackbone(nn.Module):
    """Two expert streams with shared attention dimensions and immutable prefix caches."""

    def __init__(self, prefix_dim=8, expert_dim=6):
        super().__init__()
        self.prefix_layers = nn.ModuleList([_AttentionLayer(prefix_dim) for _ in range(2)])
        self.suffix_layers = nn.ModuleList([_AttentionLayer(expert_dim) for _ in range(2)])
        self.image_projection = nn.Linear(prefix_dim, prefix_dim)
        self.language_embedding = nn.Embedding(16, prefix_dim)
        # Match only the attributes accessed by the real PI0Pytorch methods.
        self.paligemma = types.SimpleNamespace(
            language_model=types.SimpleNamespace(layers=self.prefix_layers, config=types.SimpleNamespace())
        )
        self.gemma_expert = types.SimpleNamespace(
            model=types.SimpleNamespace(layers=self.suffix_layers, config=types.SimpleNamespace())
        )
        self.calls = []

    def embed_image(self, image):
        # Harness observations already carry small image-token feature vectors.
        return self.image_projection(image)

    def embed_language_tokens(self, tokens):
        return self.language_embedding(tokens)

    @staticmethod
    def _rotary(hidden, positions):
        half = hidden.shape[-1] // 2
        scale = torch.arange(1, half + 1, dtype=hidden.dtype, device=hidden.device)
        angle = positions[..., None].to(hidden.dtype) / scale
        left, right = hidden.split(half, dim=-1)
        return torch.cat((left * angle.cos() - right * angle.sin(), right * angle.cos() + left * angle.sin()), -1)

    @staticmethod
    def _attend(query, key, value, mask):
        assert mask.shape == (query.shape[0], 1, query.shape[1], key.shape[1])
        allowed = mask[:, 0] == 0
        logits = query @ key.transpose(-1, -2) / math.sqrt(query.shape[-1])
        probabilities = torch.softmax(logits.masked_fill(~allowed, -1e9), dim=-1) * allowed
        probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        return probabilities @ value

    def forward(
        self,
        attention_mask,
        position_ids,
        past_key_values,
        inputs_embeds,
        use_cache,
        adarms_cond=None,
    ):
        prefix, suffix = inputs_embeds
        conditions = [None, None] if adarms_cond is None else adarms_cond
        mode = "joint" if prefix is not None and suffix is not None else "prefix" if prefix is not None else "suffix"
        self.calls.append(
            {
                "mode": mode,
                "mask": attention_mask.detach().clone(),
                "positions": position_ids.detach().clone(),
                "cache_length": None if past_key_values is None else past_key_values[0][0].shape[1],
            }
        )
        caches = []
        for index, (prefix_layer, suffix_layer) in enumerate(zip(self.prefix_layers, self.suffix_layers, strict=True)):
            if mode == "joint":
                prefix_query, prefix_key, prefix_value = prefix_layer.project(prefix, conditions[0])
                suffix_query, suffix_key, suffix_value = suffix_layer.project(suffix, conditions[1])
                prefix_length = prefix.shape[1]
                query = self._rotary(torch.cat((prefix_query, suffix_query), dim=1), position_ids)
                key = self._rotary(torch.cat((prefix_key, suffix_key), dim=1), position_ids)
                value = torch.cat((prefix_value, suffix_value), dim=1)
                attended = self._attend(query, key, value, attention_mask)
                prefix = prefix_layer.finish(prefix, attended[:, :prefix_length])
                suffix = suffix_layer.finish(suffix, attended[:, prefix_length:])
            elif mode == "prefix":
                query, key, value = prefix_layer.project(prefix, conditions[0])
                query = self._rotary(query, position_ids)
                key = self._rotary(key, position_ids)
                caches.append((key, value))
                prefix = prefix_layer.finish(prefix, self._attend(query, key, value, attention_mask))
            else:
                assert past_key_values is not None
                query, key, value = suffix_layer.project(suffix, conditions[1])
                query = self._rotary(query, position_ids)
                key = self._rotary(key, position_ids)
                cached_key, cached_value = past_key_values[index]
                key = torch.cat((cached_key, key), dim=1)
                value = torch.cat((cached_value, value), dim=1)
                suffix = suffix_layer.finish(suffix, self._attend(query, key, value, attention_mask))
        return [prefix, suffix], caches if use_cache else None


@pytest.fixture
def modules(monkeypatch):
    # All replacement modules are scoped to this fixture and restored afterwards.
    packages = {}
    for name in ("openpi", "openpi.models", "openpi.models_pytorch"):
        package = types.ModuleType(name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, name, package)
        packages[name] = package
    packages["openpi"].models = packages["openpi.models"]
    packages["openpi"].models_pytorch = packages["openpi.models_pytorch"]
    gemma = types.ModuleType("openpi.models.gemma")
    monkeypatch.setitem(sys.modules, gemma.__name__, gemma)
    packages["openpi.models"].gemma = gemma
    expert = types.ModuleType("openpi.models_pytorch.gemma_pytorch")
    expert.PaliGemmaWithExpertModel = _AttentionBackbone
    monkeypatch.setitem(sys.modules, expert.__name__, expert)
    preprocessing = types.ModuleType("openpi.models_pytorch.preprocessing_pytorch")
    preprocessing.calls = []

    def preprocess(observation, *, train=True):
        preprocessing.calls.append(train)
        return observation

    preprocessing.preprocess_observation_pytorch = preprocess
    monkeypatch.setitem(sys.modules, preprocessing.__name__, preprocessing)
    packages["openpi.models_pytorch"].preprocessing_pytorch = preprocessing
    config_module = _load_source(monkeypatch, "openpi.models.pressure_config", "openpi/models/pressure_config.py")
    pressure_module = _load_source(
        monkeypatch, "openpi.models_pytorch.pressure_pytorch", "openpi/models_pytorch/pressure_pytorch.py"
    )
    pi_module = _load_source(monkeypatch, "_vtla_real_pi0_pytorch", "openpi/models_pytorch/pi0_pytorch.py")
    return types.SimpleNamespace(
        pi=pi_module,
        pressure=pressure_module,
        config=config_module,
        preprocessing=preprocessing,
    )


def _make_model(modules, *, with_pressure=True):
    # Bypass the official constructor: no pretrained model or replacement Transformers is installed here.
    model = modules.pi.PI0Pytorch.__new__(modules.pi.PI0Pytorch)
    nn.Module.__init__(model)
    pressure_config = (
        modules.config.PressureConfig(
            num_sensors=2, rows=2, cols=4, history=2, patch_rows=1, patch_cols=2, hidden_dim=12, max_age_seconds=0.5
        )
        if with_pressure
        else None
    )
    model.config = types.SimpleNamespace(action_dim=3, action_horizon=4, pressure=pressure_config)
    model.pi05 = True
    model.gradient_checkpointing_enabled = False
    model.paligemma_with_expert = _AttentionBackbone()
    model.action_in_proj = nn.Linear(3, 6)
    model.action_out_proj = nn.Linear(6, 3)
    model.time_mlp_in = nn.Linear(6, 6)
    model.time_mlp_out = nn.Linear(6, 6)
    model.pressure_encoder = modules.pressure.PressureEncoder(pressure_config, 8) if with_pressure else None
    return model


@pytest.fixture
def harness(modules):
    torch.manual_seed(29)
    model = _make_model(modules)
    pressure = {
        "values": torch.rand(2, 2, 2, 2, 4),
        "frame_valid": torch.ones(2, 2, 2, dtype=torch.bool),
        "taxel_valid": torch.ones(2, 2, 2, 2, 4, dtype=torch.bool),
        "saturated": torch.zeros(2, 2, 2, 2, 4, dtype=torch.bool),
        "age": torch.full((2, 2, 2), 0.1),
    }
    pressure["frame_valid"][1, 0, 1] = False
    pressure["taxel_valid"][0, 1, 0, 0, :2] = False
    observation = types.SimpleNamespace(
        images={"camera": torch.randn(2, 2, 8)},
        image_masks={"camera": torch.ones(2, dtype=torch.bool)},
        tokenized_prompt=torch.tensor([[1, 2, 0], [3, 0, 0]]),
        tokenized_prompt_mask=torch.tensor([[True, True, False], [True, False, False]]),
        state=torch.zeros(2, 3),
        pressure=pressure,
    )
    return types.SimpleNamespace(
        model=model,
        observation=observation,
        actions=torch.randn(2, 4, 3),
        noise=torch.randn(2, 4, 3),
        time=torch.tensor([0.2, 0.7]),
        modules=modules,
    )


def _prefix(harness):
    observation = harness.observation
    return harness.model.embed_prefix(
        list(observation.images.values()),
        list(observation.image_masks.values()),
        observation.tokenized_prompt,
        observation.tokenized_prompt_mask,
        observation.pressure,
    )


def test_real_prefix_appends_pressure_with_validity_and_same_attention_block(harness):
    tokens, valid, block = _prefix(harness)
    pressure_tokens, pressure_valid = harness.model.pressure_encoder(harness.observation.pressure)
    assert tokens.shape == (2, 21, 8)
    torch.testing.assert_close(tokens[:, 5:], pressure_tokens)
    torch.testing.assert_close(valid[:, 5:], pressure_valid)
    assert not block.any()
    suffix_valid = torch.ones(2, 4, dtype=torch.bool)
    suffix_block = torch.tensor([[True, False, False, False]]).expand(2, -1)
    mask = harness.modules.pi.make_att_2d_masks(
        torch.cat((valid, suffix_valid), dim=1), torch.cat((block, suffix_block), dim=1)
    )
    assert not mask[:, :21, 21:].any()
    torch.testing.assert_close(mask[:, 21:, :21], valid[:, None, :].expand(-1, 4, -1))
    assert mask[:, 21:, 21:].all()


def test_real_flow_loss_reaches_pressure_through_frozen_backbone(harness):
    harness.model.requires_grad_(requires_grad=False)
    harness.model.pressure_encoder.requires_grad_(requires_grad=True)
    loss = harness.model(harness.observation, harness.actions, noise=harness.noise, time=harness.time)
    assert loss.shape == harness.actions.shape
    assert torch.isfinite(loss).all()
    loss.mean().backward()
    total_gradient = 0.0
    for parameter in harness.model.pressure_encoder.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        total_gradient += parameter.grad.abs().sum().item()
    assert total_gradient > 0
    assert harness.modules.preprocessing.calls == [True]
    assert harness.model.paligemma_with_expert.calls[0]["mode"] == "joint"


def test_real_joint_forward_matches_cached_denoise_with_padding(harness):
    velocities = []
    handle = harness.model.action_out_proj.register_forward_hook(
        lambda _module, _args, output: velocities.append(output)
    )
    harness.model(harness.observation, harness.actions, noise=harness.noise, time=harness.time)
    handle.remove()
    joint_velocity = velocities[0]
    prefix, valid, block = _prefix(harness)
    mask = harness.modules.pi.make_att_2d_masks(valid, block)
    _, cache = harness.model.paligemma_with_expert(
        attention_mask=harness.model._prepare_attention_masks_4d(mask),  # noqa: SLF001
        position_ids=valid.cumsum(dim=1) - 1,
        past_key_values=None,
        inputs_embeds=[prefix, None],
        use_cache=True,
    )
    cache_snapshot = [(key.clone(), value.clone()) for key, value in cache]
    noisy_actions = harness.time[:, None, None] * harness.noise + (1 - harness.time[:, None, None]) * harness.actions
    cached_velocity = harness.model.denoise_step(harness.observation.state, valid, cache, noisy_actions, harness.time)
    torch.testing.assert_close(cached_velocity, joint_velocity, atol=1e-6, rtol=1e-5)
    for actual, expected in zip(cache, cache_snapshot, strict=True):
        torch.testing.assert_close(actual[0], expected[0])
        torch.testing.assert_close(actual[1], expected[1])
    calls = harness.model.paligemma_with_expert.calls
    assert [call["mode"] for call in calls] == ["joint", "prefix", "suffix"]
    assert calls[-1]["cache_length"] == 21
    torch.testing.assert_close(calls[0]["positions"][:, -4:], calls[-1]["positions"])


def test_unavailable_pressure_values_cannot_change_real_loss_or_sample(harness):
    observation = harness.observation
    original_loss = harness.model(observation, harness.actions, noise=harness.noise, time=harness.time)
    original_sample = harness.model.sample_actions(torch.device("cpu"), observation, noise=harness.noise, num_steps=3)
    pressure = observation.pressure
    invalid = ~(pressure["taxel_valid"] & pressure["frame_valid"][..., None, None])
    pressure["values"][invalid] = float("nan")
    pressure["saturated"][invalid] = True
    changed_loss = harness.model(observation, harness.actions, noise=harness.noise, time=harness.time)
    changed_sample = harness.model.sample_actions(torch.device("cpu"), observation, noise=harness.noise, num_steps=3)
    torch.testing.assert_close(changed_loss, original_loss)
    torch.testing.assert_close(changed_sample, original_sample)


def test_real_sampler_fixed_noise_and_saved_pressure_weights_restore(harness, tmp_path):
    harness.model.eval()
    sample = harness.model.sample_actions(torch.device("cpu"), harness.observation, noise=harness.noise, num_steps=3)
    assert sample.shape == (2, 4, 3)
    assert torch.isfinite(sample).all()
    assert [call["mode"] for call in harness.model.paligemma_with_expert.calls] == [
        "prefix",
        "suffix",
        "suffix",
        "suffix",
    ]
    assert harness.modules.preprocessing.calls == [False]
    state_path = tmp_path / "harness_state.pt"
    torch.save(harness.model.state_dict(), state_path)
    restored = _make_model(harness.modules)
    restored.load_state_dict(torch.load(state_path, weights_only=True), strict=True)
    restored.eval()
    restored_sample = restored.sample_actions(
        torch.device("cpu"), harness.observation, noise=harness.noise, num_steps=3
    )
    torch.testing.assert_close(restored_sample, sample)


def test_all_pressure_masked_matches_same_weights_without_pressure(harness):
    harness.observation.pressure["frame_valid"].zero_()
    baseline = _make_model(harness.modules, with_pressure=False)
    baseline.load_state_dict(
        {name: value for name, value in harness.model.state_dict().items() if not name.startswith("pressure_encoder.")},
        strict=True,
    )
    baseline_observation = types.SimpleNamespace(**vars(harness.observation))
    baseline_observation.pressure = None
    pressure_loss = harness.model(harness.observation, harness.actions, noise=harness.noise, time=harness.time)
    baseline_loss = baseline(baseline_observation, harness.actions, noise=harness.noise, time=harness.time)
    torch.testing.assert_close(pressure_loss, baseline_loss, atol=1e-6, rtol=1e-5)
    pressure_sample = harness.model.sample_actions(
        torch.device("cpu"), harness.observation, noise=harness.noise, num_steps=3
    )
    baseline_sample = baseline.sample_actions(
        torch.device("cpu"), baseline_observation, noise=harness.noise, num_steps=3
    )
    torch.testing.assert_close(pressure_sample, baseline_sample, atol=1e-6, rtol=1e-5)
    pressure_positions = harness.model.paligemma_with_expert.calls[0]["positions"]
    baseline_positions = baseline.paligemma_with_expert.calls[0]["positions"]
    torch.testing.assert_close(pressure_positions[:, :5], baseline_positions[:, :5])
    torch.testing.assert_close(pressure_positions[:, -4:], baseline_positions[:, -4:])
