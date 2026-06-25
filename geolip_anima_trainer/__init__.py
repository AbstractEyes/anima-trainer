"""geolip_anima_trainer — bridge + orchestration to finetune CircleStone Anima
(2B DiT) with tdrussell/diffusion-pipe.

See PRELIM_PLAN.md / CLAUDE.md for the domain brief. Public API:

    import geolip_anima_trainer as anima
    anima.doctor()                  # environment diagnostics
    anima.inspect_source(...)       # probe an HF dataset config
    anima.export_dataset(...)       # parquet -> img + .txt dirs
    anima.build_dataset_toml(...)   # balanced dataset.toml
    anima.single_concept_preset(...)  # composable TrainConfig
    anima.train(..., dry_run=True)  # diffusion-pipe deepspeed launch
"""

from __future__ import annotations

from .api import (  # noqa: F401
    AdapterConfig, CaptionMode, ConfigError, DatasetConfig, DatasetTomlConfig,
    DirectoryConfig, DiffusionPipeNotFound, DoctorReport, ExportConfig, ModelConfig,
    ModelPaths, OptimizerConfig, RunConfig, SubjectBucketConfig, TrainConfig,
    WindowsTrainingRefused, apply_overrides, build_dataset_toml, build_mode_tomls,
    cache, cache_pull, cache_push, doctor, download_models, export_dataset,
    export_subject_buckets,
    inspect_source, load_dataset_config, load_train_config, multi_concept_preset,
    rebalance, render_dataset_toml, render_lora_toml, render_train_toml,
    single_concept_preset, sweep, train, train_before_after, validate, validate_bridge,
)

try:
    from importlib.metadata import version
    __version__ = version("geolip-anima-trainer")
except Exception:  # noqa: BLE001 — not installed (e.g. running from source)
    __version__ = "0.1.0"

__all__ = [
    "__version__",
    # config engine
    "ModelConfig", "AdapterConfig", "OptimizerConfig", "RunConfig",
    "DatasetConfig", "DirectoryConfig", "TrainConfig", "ConfigError", "ModelPaths",
    "ExportConfig", "DatasetTomlConfig", "SubjectBucketConfig", "CaptionMode",
    "load_train_config", "load_dataset_config", "render_train_toml",
    "render_lora_toml", "render_dataset_toml", "apply_overrides", "rebalance",
    "single_concept_preset", "multi_concept_preset", "sweep",
    "validate", "validate_bridge",
    # operations
    "download_models", "inspect_source", "export_dataset", "export_subject_buckets",
    "build_dataset_toml", "build_mode_tomls",
    "cache", "cache_push", "cache_pull", "train", "train_before_after", "doctor",
    "DoctorReport", "WindowsTrainingRefused", "DiffusionPipeNotFound",
]
