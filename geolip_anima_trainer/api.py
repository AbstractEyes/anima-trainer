#!/usr/bin/env python3
"""
api.py — the public Python API for geolip_anima_trainer.

Thin wrappers over the bridge scripts (download / inspect / export / build), the
diffusion-pipe orchestrator (cache / train), the environment doctor, and the config
"elemental construction" engine. The wrappers IMPORT the existing functions rather
than duplicating them, so there is a single source of truth for every behavior.

    import geolip_anima_trainer as anima
    paths = anima.download_models("models/anima", base="base-v1.0")
    info  = anima.inspect_source("AbstractPhil/diffusion-pretrain-set-ft1", "qwen_90k")
    cfg   = anima.single_concept_preset("datasets/anima_qwen90k/qwen_90k",
                                        output_dir="runs/anima", model=anima.ModelConfig(**paths.__dict__))
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from . import build_multiconcept_dataset as _build
from . import download_anima as _dl
from . import hf_to_diffusion_pipe as _bridge
from . import launch as _launch
from .config import (  # re-exported "elemental construction" surface
    AdapterConfig, ConfigError, DatasetConfig, DirectoryConfig, ModelConfig,
    OptimizerConfig, RunConfig, TrainConfig,
    apply_overrides, load_dataset_config, load_train_config,
    multi_concept_preset, rebalance, render_dataset_toml, render_lora_toml,
    render_train_toml, single_concept_preset, sweep, validate, validate_bridge,
)
from .doctor import DoctorReport, doctor

# Bridge configs re-exported under disambiguated names.
ExportConfig = _bridge.BridgeConfig
DatasetTomlConfig = _build.BaseConfig

__all__ = [
    # config engine
    "ModelConfig", "AdapterConfig", "OptimizerConfig", "RunConfig",
    "DatasetConfig", "DirectoryConfig", "TrainConfig", "ConfigError",
    "load_train_config", "load_dataset_config", "render_train_toml",
    "render_lora_toml", "render_dataset_toml", "apply_overrides", "rebalance",
    "single_concept_preset", "multi_concept_preset", "sweep",
    "validate", "validate_bridge",
    # bridge configs
    "ExportConfig", "DatasetTomlConfig", "ModelPaths",
    # operations
    "download_models", "inspect_source", "export_dataset", "build_dataset_toml",
    "cache", "train", "doctor", "DoctorReport",
    "WindowsTrainingRefused", "DiffusionPipeNotFound",
]

WindowsTrainingRefused = _launch.WindowsTrainingRefused
DiffusionPipeNotFound = _launch.DiffusionPipeNotFound


@dataclass
class ModelPaths:
    """The resolved local paths for the [model] block."""
    transformer_path: str
    vae_path: str
    llm_path: str


# =============================================================================
# 1. download
# =============================================================================
def download_models(dest: str | Path, base: str = "base-v1.0") -> ModelPaths:
    """Fetch the three Anima files via huggingface_hub. Returns resolved paths."""
    dest = Path(dest).expanduser().resolve()
    dest.mkdir(parents=True, exist_ok=True)
    base_file = f"split_files/diffusion_models/{_dl.BASE_CHOICES[base]}"
    return ModelPaths(
        transformer_path=_dl.fetch(base_file, dest),
        vae_path=_dl.fetch(_dl.VAE, dest),
        llm_path=_dl.fetch(_dl.TEXT_ENCODER, dest),
    )


# =============================================================================
# 2. inspect (read-only probe)
# =============================================================================
def inspect_source(repo: str, config: str, *, split: str = "train",
                   n: int = 200, verbose: bool = False) -> dict:
    """Sample n rows from one HF config and return columns / caption fill rates /
    audit + age gate distributions. Wraps hf_to_diffusion_pipe.inspect."""
    cfg = ExportConfig(repo=repo, configs=[config], split=split)
    return _bridge.inspect(cfg, n, verbose=verbose)


# =============================================================================
# 3. export (stream parquet -> img + .txt dirs)
# =============================================================================
def export_dataset(cfg: "ExportConfig | None" = None, /, **overrides) -> dict:
    """Stream the configured HF configs into per-concept image+.txt folders.
    Accepts an ExportConfig and/or field overrides. Returns per-concept counts."""
    cfg = cfg or ExportConfig()
    if overrides:
        cfg = replace(cfg, **overrides)
    counts: dict = {}
    for ds_config in cfg.configs:
        _bridge.export_config(ds_config, cfg, counts)
    return counts


# =============================================================================
# 4. build dataset.toml
# =============================================================================
def build_dataset_toml(root: str | Path, out: str | Path,
                       cfg: "DatasetTomlConfig | None" = None, /, **overrides) -> Path:
    """Scan concept folders, balance num_repeats, write a dataset.toml (utf-8)."""
    cfg = cfg or DatasetTomlConfig()
    if overrides:
        cfg = replace(cfg, **overrides)
    root = Path(root).expanduser().resolve()
    concepts = _build.discover_concepts(root, cfg)
    if not concepts:
        raise FileNotFoundError(f"No concept subfolders with images under {root}")
    _build.compute_repeats(concepts, cfg)
    out = Path(out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_build.render_toml(concepts, cfg), encoding="utf-8")
    return out


# =============================================================================
# 5/6. orchestration: cache + train (diffusion-pipe / deepspeed)
# =============================================================================
def cache(config_toml: str | Path, *, repo_root: str | Path | None = None,
          num_gpus: int = 1, gpu_ids: list[int] | None = None,
          regenerate: bool = False, dry_run: bool = False):
    """Precache latents/text-embeds: deepspeed ... train.py --cache_only."""
    plan = _launch.build_plan(config_toml=config_toml, repo_root=repo_root,
                              num_gpus=num_gpus, gpu_ids=gpu_ids,
                              cache_only=True, regenerate_cache=regenerate)
    return _launch.launch(plan, dry_run=dry_run)


def train(config_toml: str | Path, *, repo_root: str | Path | None = None,
          num_gpus: int = 1, gpu_ids: list[int] | None = None,
          pipeline_stages: int | None = None,
          resume: bool | str = False, dry_run: bool = False):
    """Launch training: deepspeed --num_gpus=N train.py --deepspeed --config ..."""
    plan = _launch.build_plan(config_toml=config_toml, repo_root=repo_root,
                              num_gpus=num_gpus, gpu_ids=gpu_ids,
                              pipeline_stages=pipeline_stages,
                              resume_from_checkpoint=resume)
    return _launch.launch(plan, dry_run=dry_run)
