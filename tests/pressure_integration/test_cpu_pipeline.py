"""Exercise the actual openpi pipeline with tiny random models and synthetic data.

This does not load official weights, use a real sensor, or measure task success.
It requires the optional CPU integration dependencies and upstream Transformers patch.
"""

import copy
import dataclasses
import gc
import json

import numpy as np
import pytest
import safetensors.torch

torch = pytest.importorskip("torch")
pytest.importorskip("jax")
pytest.importorskip("flax")
spm = pytest.importorskip("sentencepiece")
pytest.importorskip("transformers")


@pytest.mark.parametrize("encoder_kind", ["mlp", "lingbot"])
def test_synthetic_npz_train_final_checkpoint_resume_and_policy_inference(tmp_path, monkeypatch, encoder_kind):
    """Run real reader/transforms/trainer/model/checkpoint/policy on CPU with small configs."""
    monkeypatch.setenv("OPENPI_DATA_HOME", str(tmp_path / "openpi-cache"))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf-cache"))
    monkeypatch.setenv("WANDB_DIR", str(tmp_path / "wandb"))
    monkeypatch.setenv("WANDB_CACHE_DIR", str(tmp_path / "wandb-cache"))
    monkeypatch.setenv("WANDB_CONFIG_DIR", str(tmp_path / "wandb-config"))
    monkeypatch.setenv("WANDB_DATA_DIR", str(tmp_path / "wandb-data"))
    for name in ("wandb", "wandb-cache", "wandb-config", "wandb-data"):
        (tmp_path / name).mkdir()
    torch.set_num_threads(2)

    from openpi.models import pi0_config  # noqa: PLC0415
    from openpi.models import tokenizer as tokenizer_module  # noqa: PLC0415
    from openpi.models.pressure_config import LingBotTouchConfig  # noqa: PLC0415
    from openpi.models.pressure_config import PressureConfig  # noqa: PLC0415
    from openpi.models_pytorch.lingbot_touch import EncoderConfig  # noqa: PLC0415
    from openpi.models_pytorch.lingbot_touch import TargetEncoder  # noqa: PLC0415
    from openpi.policies import policy_config  # noqa: PLC0415
    from openpi.training import config as training_config  # noqa: PLC0415
    from openpi.training.pressure_dataset import PressureNpzDataset  # noqa: PLC0415
    from openpi.training.pressure_dataset import fit_pressure_stats  # noqa: PLC0415
    from scripts import compute_norm_stats  # noqa: PLC0415
    from scripts import train_pytorch  # noqa: PLC0415

    # A local test tokenizer exercises the real SentencePiece and transform path.
    # Its random test vocabulary is not compatible with an official checkpoint.
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(("Task: move object, State: 10 20 30 40; Action:\n" * 50), encoding="utf-8")
    spm.SentencePieceTrainer.train(
        input=str(corpus),
        model_prefix=str(tmp_path / "tokenizer"),
        vocab_size=64,
        model_type="bpe",
        hard_vocab_limit=False,
        minloglevel=2,
    )
    tokenizer_path = tmp_path / "tokenizer.model"
    original_download = tokenizer_module.download.maybe_download

    def local_tokenizer(url, **kwargs):
        if str(url) == "gs://big_vision/paligemma_tokenizer.model":
            return tokenizer_path
        return original_download(url, **kwargs)

    monkeypatch.setattr(tokenizer_module.download, "maybe_download", local_tokenizer)

    rng = np.random.default_rng(13)
    n = 8
    timestamps = np.arange(n, dtype=np.float64) * 0.1
    robot_path = tmp_path / "robot.npz"
    pressure_path = tmp_path / "pressure.npz"
    np.savez(
        robot_path,
        timestamps=timestamps,
        state=rng.normal(size=(n, 3)).astype(np.float32),
        actions=rng.normal(size=(n, 3)).astype(np.float32),
        base_0_rgb=rng.integers(0, 256, (n, 28, 28, 3), dtype=np.uint8),
    )
    np.savez(
        pressure_path,
        timestamps=timestamps,
        values=rng.normal(size=(n, 1, 2, 2)).astype(np.float32),
        arrival_times=timestamps,
    )
    manifest = tmp_path / "train.json"
    manifest.write_text(
        json.dumps({"episodes": [{"observations": "robot.npz", "pressure": "pressure.npz", "prompt": "move object"}]}),
        encoding="utf-8",
    )
    pressure_stats = tmp_path / "pressure_stats.npz"
    np.savez(pressure_stats, **fit_pressure_stats(manifest))
    cfg = training_config.TrainConfig(
        name="pressure_integration",
        exp_name="smoke",
        model=pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="dummy",
            action_expert_variant="dummy",
            pytorch_debug_tiny=True,
            pytorch_compile_mode=None,
            action_dim=4,
            action_horizon=2,
            max_token_len=64,
            dtype="float32",
            pressure=PressureConfig(
                num_sensors=1,
                rows=2,
                cols=2,
                patch_rows=1,
                patch_cols=1,
                hidden_dim=8,
                history=2,
                use_temporal_differences=True,
                sensor_names=("forearm_skin",),
                sensor_locations=("forearm",),
                taxel_positions_mm=(((0, 0, 0), (10, 0, 0), (0, 10, 0), (10, 10, 0)),),
            ),
        ),
        data=training_config.PressureNpzDataConfig(
            manifest_path=str(manifest),
            pressure_stats_path=str(pressure_stats),
            robot_action_dim=3,
        ),
        assets_base_dir=str(tmp_path / "assets"),
        checkpoint_base_dir=str(tmp_path / "checkpoints"),
        batch_size=1,
        num_workers=0,
        num_train_steps=2,
        save_interval=100,
        pytorch_training_precision="float32",
        pressure_freeze_backbone=True,
        wandb_enabled=False,
    )
    if encoder_kind == "lingbot":
        # Small random bundle, deliberately not an official pretrained checkpoint.
        settings = LingBotTouchConfig(sensor_zero=(-3,), sensor_span=(6,), time_scale_seconds=0.1, readout_dim=16)
        architecture = {**settings.encoder_config, "embed_dim": 40, "depth": 2, "num_heads": 2, "output_layers": [0, 1]}
        settings = dataclasses.replace(settings, encoder_config=architecture)
        bundle = tmp_path / "synthetic-lingbot"
        bundle.mkdir()
        (bundle / "config.json").write_text(json.dumps(architecture), encoding="utf-8")
        initial_encoder = TargetEncoder(EncoderConfig.from_dict(architecture)).state_dict()
        torch.save(initial_encoder, bundle / "encoder.pt")
        cfg = dataclasses.replace(
            cfg,
            model=dataclasses.replace(
                cfg.model,
                pressure=dataclasses.replace(cfg.model.pressure, lingbot_touch=settings),
                tactile_residual_enabled=True,
                tactile_residual_only=True,
                tactile_residual_freeze_base=True,
            ),
            lingbot_touch_bundle_path=str(bundle),
            lr_schedule=dataclasses.replace(cfg.lr_schedule, warmup_steps=0, peak_lr=1e-3, decay_steps=20),
        )
    monkeypatch.setattr(training_config, "get_config", lambda name: cfg)
    compute_norm_stats.main(cfg.name)
    train_pytorch.train_loop(cfg)
    first = cfg.checkpoint_dir / "2"
    assert (first / "model.safetensors").is_file()
    assert not (cfg.checkpoint_dir / "1").exists()
    metadata = json.loads((first / "pressure_config.json").read_text(encoding="utf-8"))
    assert metadata["schema_version"] == 2
    assert metadata["policy_contract"]["robot_action_dim"] == 3

    # Deployment must work without the original training manifest or assets directory.
    deployed = dataclasses.replace(
        cfg,
        data=dataclasses.replace(cfg.data, manifest_path="", pressure_stats_path=""),
        assets_base_dir=str(tmp_path / "not-present"),
    )
    policy = policy_config.create_trained_policy(deployed, first, pytorch_device="cpu", sample_kwargs={"num_steps": 2})
    sample = PressureNpzDataset(manifest, cfg.model.pressure.history, 2)[3]
    sample.pop("actions")
    result = policy.infer(sample, noise=np.zeros((2, 4), dtype=np.float32))
    assert result["actions"].shape == (2, 3)
    assert np.isfinite(result["actions"]).all()
    altered_sample = copy.deepcopy(sample)
    altered_sample["pressure"]["values"] += 1.0
    altered = policy.infer(altered_sample, noise=np.zeros((2, 4), dtype=np.float32))
    assert not np.array_equal(result["actions"], altered["actions"])
    del policy
    gc.collect()

    incompatible = dataclasses.replace(deployed, data=dataclasses.replace(deployed.data, robot_action_dim=2))
    with pytest.raises(ValueError, match="contract differs"):
        policy_config.create_trained_policy(incompatible, first, pytorch_device="cpu")
    gc.collect()

    # A resumed third update must use the saved optimizer/model and write step 3.
    train_pytorch.train_loop(dataclasses.replace(cfg, resume=True, num_train_steps=3))
    third = cfg.checkpoint_dir / "3" / "model.safetensors"
    assert third.is_file()
    first_weights = safetensors.torch.load_file(first / "model.safetensors")
    third_weights = safetensors.torch.load_file(third)
    learned_key = "pressure_encoder.patch_in.weight" if encoder_kind == "mlp" else "tactile_residual_adapter.2.weight"
    assert not torch.equal(first_weights[learned_key], third_weights[learned_key])

    # Reusing a name must fail before training can overwrite an existing checkpoint.
    with pytest.raises(FileExistsError, match="already exists"):
        train_pytorch.train_loop(cfg)

    if encoder_kind == "lingbot":
        for name, weight in initial_encoder.items():
            torch.testing.assert_close(first_weights[f"pressure_encoder.pretrained.{name}"], weight, rtol=0, atol=0)
        second_stage = dataclasses.replace(
            cfg,
            exp_name="encoder_ft",
            pytorch_weight_path=str(first),
            lingbot_touch_bundle_path=None,
            lingbot_pretrained_lr_scale=0.1,
            model=dataclasses.replace(
                cfg.model,
                pressure=dataclasses.replace(
                    cfg.model.pressure,
                    lingbot_touch=dataclasses.replace(cfg.model.pressure.lingbot_touch, freeze_pretrained=False),
                ),
            ),
        )
        train_pytorch.train_loop(second_stage)
        stage2_weights = safetensors.torch.load_file(second_stage.checkpoint_dir / "2" / "model.safetensors")
        pretrained_key = "pressure_encoder.pretrained.transformer.blocks.0.attn.qkv.weight"
        assert not torch.equal(stage2_weights[pretrained_key], first_weights[pretrained_key])
        base_key = next(k for k in first_weights if k.startswith("paligemma_with_expert.") and k.endswith("weight"))
        torch.testing.assert_close(stage2_weights[base_key], first_weights[base_key], rtol=0, atol=0)
        train_pytorch.train_loop(dataclasses.replace(second_stage, resume=True, num_train_steps=3))
        optimizer_state = torch.load(second_stage.checkpoint_dir / "3" / "optimizer.pt", weights_only=True)
        groups = optimizer_state["param_groups"]
        assert len(groups) == 2
        assert groups[1]["lr"] == pytest.approx(groups[0]["lr"] * 0.1)
