"""Config-engine tests — pure logic, no torch/datasets/GPU required."""
from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from geolip_anima_trainer import config as C


def _model() -> C.ModelConfig:
    return C.ModelConfig(transformer_path="t.safetensors",
                         vae_path="v.safetensors", llm_path="l.safetensors")


def test_single_concept_preset_defaults():
    cfg = C.single_concept_preset("data/concept", output_dir="runs/x", model=_model())
    assert cfg.adapter is not None and cfg.adapter.rank == 32
    assert cfg.model.llm_adapter_lr == 0.0
    assert len(cfg.dataset.directories) == 1
    assert cfg.dataset.shuffle_caption is False


def test_validate_rejects_high_llm_adapter_lr():
    cfg = C.single_concept_preset("data/c", output_dir="r", model=_model())
    cfg.model.llm_adapter_lr = 1e-5  # above the 5e-6 ceiling
    with pytest.raises(C.ConfigError):
        C.validate(cfg)


def test_validate_allows_phase2_ceiling():
    cfg = C.single_concept_preset("data/c", output_dir="r", model=_model())
    cfg.model.llm_adapter_lr = 5e-6
    assert C.validate(cfg) is cfg  # warns, does not raise


def test_validate_rejects_shuffle_without_keep_tokens():
    cfg = C.single_concept_preset("data/c", output_dir="r", model=_model())
    cfg.dataset.shuffle_caption = True
    with pytest.raises(C.ConfigError):
        C.validate(cfg)
    cfg.dataset.keep_tokens = 6
    assert C.validate(cfg) is cfg


def test_apply_overrides_is_immutable_and_validated():
    base = C.single_concept_preset("data/c", output_dir="r", model=_model())
    out = C.apply_overrides(base, rank=64, lr=1e-5, output_dir="r2")
    assert out.adapter.rank == 64 and out.optimizer.lr == 1e-5
    assert base.adapter.rank == 32  # base untouched
    with pytest.raises(C.ConfigError):
        C.apply_overrides(base, bogus=1)


def test_round_trip_render_then_load(tmp_path: Path):
    cfg = C.single_concept_preset("data/concept", output_dir="runs/x", model=_model())
    lora_path, ds_path = C.render_train_toml(cfg, tmp_path)
    # valid TOML
    tomllib.loads(lora_path.read_text(encoding="utf-8"))
    tomllib.loads(ds_path.read_text(encoding="utf-8"))
    loaded = C.load_train_config(lora_path, ds_path)
    assert loaded.adapter.rank == cfg.adapter.rank
    assert loaded.model.llm_adapter_lr == cfg.model.llm_adapter_lr
    assert loaded.optimizer.lr == cfg.optimizer.lr
    assert loaded.dataset.shuffle_caption is False
    assert len(loaded.dataset.directories) == 1


def test_compile_and_ckpt_round_trip(tmp_path: Path):
    # the throughput/VRAM knobs render and load back (compile is the one real speed lever)
    cfg = C.single_concept_preset("data/concept", output_dir="runs/x", model=_model())
    cfg.run.compile = True
    cfg.run.activation_checkpointing = True
    lora_path, ds_path = C.render_train_toml(cfg, tmp_path)
    text = lora_path.read_text(encoding="utf-8")
    assert "compile = true" in text and "activation_checkpointing = true" in text
    loaded = C.load_train_config(lora_path, ds_path)
    assert loaded.run.compile is True
    assert loaded.run.activation_checkpointing is True


def test_sweep_emits_grid(tmp_path: Path):
    base = C.single_concept_preset("data/concept", output_dir="runs/x", model=_model())
    out = list(C.sweep(base, ranks=[32, 64], lrs=[1e-5, 2e-5],
                       runs_root=str(tmp_path / "runs"),
                       configs_root=str(tmp_path / "cfg")))
    assert len(out) == 4
    tags = {t for t, _ in out}
    assert tags == {"r32_lr1e-05", "r32_lr2e-05", "r64_lr1e-05", "r64_lr2e-05"}
    for _, path in out:
        assert path.is_file()


def test_fft_variant_omits_adapter(tmp_path: Path):
    cfg = C.single_concept_preset("data/c", output_dir="r", model=_model())
    cfg.adapter = None
    lora_path, _ = C.render_train_toml(cfg, tmp_path)
    parsed = tomllib.loads(lora_path.read_text(encoding="utf-8"))
    assert "adapter" not in parsed
