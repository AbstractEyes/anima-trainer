"""geolip_anima_trainer — bridge + orchestration to finetune CircleStone Anima
(2B DiT) with tdrussell/diffusion-pipe, and Sana (diffusers format) with the
AbstractEyes diffusion-pipe fork.

See PRELIM_PLAN.md / CLAUDE.md for the domain brief. Public API:

    import geolip_anima_trainer as anima
    anima.doctor()                  # environment diagnostics
    anima.inspect_source(...)       # probe an HF dataset config
    anima.export_dataset(...)       # parquet -> img + .txt dirs
    anima.build_dataset_toml(...)   # balanced dataset.toml
    anima.single_concept_preset(...)  # composable TrainConfig
    anima.train(..., dry_run=True)  # diffusion-pipe deepspeed launch
    anima.sana_model(anima.download_sana("models/sana"))  # a Sana [model] block
"""

from __future__ import annotations

from .api import (  # noqa: F401
    MODEL_TYPES, AdapterConfig, CaptionMode, ConfigError, DatasetConfig, DatasetTomlConfig,
    DirectoryConfig, DiffusionPipeNotFound, DoctorReport, ExportConfig, ModelConfig,
    ModelPaths, OptimizerConfig, RunConfig, SamplesConfig, SubjectBucketConfig, TrainConfig,
    WindowsTrainingRefused, apply_overrides, build_dataset_toml, build_mode_tomls,
    cache, cache_pull, cache_push, doctor, download_models, download_sana, export_dataset,
    export_subject_buckets, reconstruct_dataset, prune_source_cache, keepalive, gpu_keepalive,
    inspect_source, load_dataset_config, load_train_config, multi_concept_preset,
    preset_optimizer, rebalance, render_dataset_toml, render_lora_toml, render_train_toml,
    sana_model, single_concept_preset, sweep, train, train_before_after, validate, validate_bridge,
)
from .cache_factory import (  # noqa: F401 — the Colab cache-factory runner (light import)
    CacheFactory, FactoryConfig, find_scratch, get_hf_token,
)
from .trainer_runner import TrainerRunner, TrainerConfig  # noqa: F401 — the RTX 6000 trainer runner

try:
    from importlib.metadata import version
    __version__ = version("geolip-anima-trainer")
except Exception:  # noqa: BLE001 — not installed (e.g. running from source)
    __version__ = "0.1.0"

__all__ = [
    "__version__",
    # config engine
    "MODEL_TYPES", "ModelConfig", "AdapterConfig", "OptimizerConfig", "RunConfig", "SamplesConfig",
    "DatasetConfig", "DirectoryConfig", "TrainConfig", "ConfigError", "ModelPaths",
    "ExportConfig", "DatasetTomlConfig", "SubjectBucketConfig", "CaptionMode",
    "load_train_config", "load_dataset_config", "render_train_toml",
    "render_lora_toml", "render_dataset_toml", "apply_overrides", "rebalance",
    "single_concept_preset", "multi_concept_preset", "preset_optimizer", "sweep",
    "validate", "validate_bridge",
    # operations
    "download_models", "download_sana", "sana_model",
    "inspect_source", "export_dataset", "export_subject_buckets",
    "build_dataset_toml", "build_mode_tomls",
    "cache", "cache_push", "cache_pull", "reconstruct_dataset", "prune_source_cache",
    "keepalive", "gpu_keepalive", "train", "train_before_after",
    # colab cache-factory runner + RTX 6000 trainer runner
    "CacheFactory", "FactoryConfig", "find_scratch", "get_hf_token",
    "TrainerRunner", "TrainerConfig",
    "doctor", "DoctorReport", "WindowsTrainingRefused", "DiffusionPipeNotFound",
]
