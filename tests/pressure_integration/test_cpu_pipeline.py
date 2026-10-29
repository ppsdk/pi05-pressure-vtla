"""Exercise the actual openpi pipeline with tiny random models and synthetic data.

This does not load official weights, use a real sensor, or measure task success.
It requires the optional CPU integration dependencies and upstream Transformers patch.
"""

import dataclasses
import gc
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("jax")
pytest.importorskip("flax")
spm = pytest.importorskip("sentencepiece")
pytest.importorskip("transformers")


def test_synthetic_npz_train_final_checkpoint_resume_and_policy_inference(tmp_path, monkeypatch):
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
    from openpi.models.pressure_config import PressureConfig  # noqa: PLC0415
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
            pressure=PressureConfig(num_sensors=1, rows=2, cols=2, patch_rows=1, patch_cols=1, hidden_dim=8),
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
    sample = PressureNpzDataset(manifest, 1, 2)[3]
    sample.pop("actions")
    result = policy.infer(sample, noise=np.zeros((2, 4), dtype=np.float32))
    assert result["actions"].shape == (2, 3)
    assert np.isfinite(result["actions"]).all()
    del policy
    gc.collect()

    # A resumed third update must use the saved optimizer/model and write step 3.
    train_pytorch.train_loop(dataclasses.replace(cfg, resume=True, num_train_steps=3))
    assert (cfg.checkpoint_dir / "3" / "model.safetensors").is_file()
