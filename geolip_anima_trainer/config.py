#!/usr/bin/env python3
"""
config.py — the "elemental construction" layer for Anima finetuning.

This module models a diffusion-pipe training run as composable dataclasses
(model / adapter / optimizer / run / dataset / directory), so a run can be built,
overridden, swept, and validated PROGRAMMATICALLY instead of by hand-editing TOML.

Design rules (kept deliberately minimal — see PRELIM_PLAN.md / CLAUDE.md):
  * stdlib @dataclass only (no pydantic) — matches the existing scripts, zero deps.
  * tomllib (stdlib, 3.11+) for READS; TOML is hand-rendered for WRITES so the
    load-bearing domain comments (tag-order rule, two-phase llm_adapter_lr
    protocol, balance notes) survive — a generic serializer would erase them.
  * Reuse, never duplicate: balancing/discovery come from
    build_multiconcept_dataset; caption rendering/gates from hf_to_diffusion_pipe.

The Anima invariants that MUST NOT be lost are encoded in validate():
  * tag-order sensitive  -> shuffle_caption = false unless keep_tokens is set.
  * llm_adapter_lr = 0   -> adapter frozen; raising it is a phase-2 opt-in (<=5e-6).
  * bf16 throughout, no fp8, no flash-attn, no block-swap on the 96GB target.
"""

from __future__ import annotations

import logging
import tomllib
from copy import deepcopy
from dataclasses import dataclass, field, fields, replace
from itertools import product
from pathlib import Path
from typing import Any, Iterator

log = logging.getLogger("anima.config")


class ConfigError(ValueError):
    """Raised by validate() on an incoherent / illegal run configuration."""


# =============================================================================
# LORA-SIDE ELEMENTS  (tables of anima_lora.toml)
# =============================================================================
@dataclass
class ModelConfig:
    """[model] — the DiT + VAE + Qwen text encoder, and the frozen LLM adapter."""
    type: str = "anima"
    transformer_path: str = ""
    vae_path: str = ""
    llm_path: str = ""
    dtype: str = "bfloat16"          # bf16 everywhere; never fp8 on this lineage.
    # The LLM adapter carries an outsized share of the model's knowledge and
    # degrades easily. 0 = FROZEN (the non-negotiable default). Phase-2 opt-in
    # only, ceiling 5e-6, A/B'd against a frozen baseline. validate() enforces this.
    llm_adapter_lr: float = 0.0


@dataclass
class AdapterConfig:
    """[adapter] — LoRA. Set TrainConfig.adapter=None for a full fine-tune."""
    type: str = "lora"
    rank: int = 32                   # single-concept default; 64 for multi-concept.
    dtype: str = "bfloat16"
    init_from_existing: str | None = None   # continue a LoRA; omitted when None.


@dataclass
class OptimizerConfig:
    """[optimizer] — adamw_optimi auto-applies Kahan summation for correct bf16."""
    type: str = "adamw_optimi"
    lr: float = 2e-5                 # author's rank-32 start; Anima likes LOW lr.
    betas: tuple[float, float] = (0.9, 0.99)
    weight_decay: float = 0.01
    eps: float = 1e-8


@dataclass
class RunConfig:
    """Top-level keys of anima_lora.toml (everything outside the named tables)."""
    output_dir: str = ""             # a new subdir is made per run.
    epochs: int = 100
    micro_batch_size_per_gpu: int = 4
    pipeline_stages: int = 1         # 1 = data-parallel; >1 = pipeline parallel.
    gradient_accumulation_steps: int = 1
    gradient_clipping: float = 1.0
    warmup_steps: int = 50
    eval_every_n_epochs: int = 1
    eval_before_first_step: bool = True
    eval_micro_batch_size_per_gpu: int = 4
    eval_gradient_accumulation_steps: int = 1
    save_every_n_epochs: int = 5
    checkpoint_every_n_minutes: int = 30
    save_dtype: str = "bfloat16"
    # activation_checkpointing is a VRAM<->speed TRADE, not a free win. OFF keeps all 28 DiT
    # blocks' activations resident (the bulk of VRAM at 1024² — ~89 GB at micro_batch 4) but
    # is ~25-33% FASTER per step. ON recomputes each block in backward -> drops VRAM to ~25-30
    # GB but costs that recompute. On the 96 GB box leave it OFF for speed; turn it ON only to
    # FREE VRAM for a bigger micro_batch / higher resolution (the activation term is linear in both).
    activation_checkpointing: bool = False
    # torch.compile(dynamic=True) on the DiT — the ONE real throughput lever (~+10-25%). Costs a
    # one-time compile + one recompile per distinct AR-bucket shape, so it amortizes over a real
    # multi-epoch run but is net-negative for a tiny smoke test. Maskless SDPA already uses the
    # efficient (flash/mem-efficient) backend on sm_120/cu128 (PyTorch PR 145602) — NOT math — so
    # ~2.8 samples/s at 1024² is compute-bound-normal, not a fallback; compile is the lever left.
    compile: bool = False
    partition_method: str = "parameters"
    # Caching throughput (--cache_only is decode/plumbing-bound, NOT GPU-bound; the 2B DiT
    # never loads, so VRAM is low by design). map_num_proc = the image-DECODE worker pool =
    # the real bottleneck (diffusion-pipe caps it at min(8, cpu); raise to the box's core
    # count). None -> leave at default. caching_batch_size is NOT a throughput knob: it only
    # sets how many images a worker decodes+stacks into ONE VAE forward before the first shard
    # (and the VRAM that spikes) — keep small (8, <=16); big values delay the first shard / OOM.
    caching_batch_size: int = 8
    map_num_proc: int | None = None
    steps_per_print: int = 10
    blocks_to_swap: int = 0          # 0 = disabled; VRAM is abundant.


# =============================================================================
# DATASET-SIDE ELEMENTS  (anima_dataset.toml)
# =============================================================================
@dataclass
class DirectoryConfig:
    """One [[directory]] block — a concept folder of images + .txt sidecars."""
    path: str
    caption_extension: str = ".txt"
    num_repeats: int = 1
    # Carried for balancing/reporting; NOT serialized into the toml.
    name: str | None = None
    image_count: int | None = None
    caption_count: int | None = None


@dataclass
class DatasetConfig:
    """Top-level of anima_dataset.toml + the balancing policy."""
    resolutions: list[int] = field(default_factory=lambda: [1024])
    enable_ar_bucket: bool = True
    min_ar: float = 0.5
    max_ar: float = 2.0
    num_ar_buckets: int = 7
    frame_buckets: list[int] = field(default_factory=lambda: [1])
    shuffle_caption: bool = False    # Anima is tag-order sensitive.
    keep_tokens: int | None = None   # required iff shuffle_caption is True.
    directories: list[DirectoryConfig] = field(default_factory=list)
    # Balancing policy (folded in from the builder script's BaseConfig):
    #   repeats(c) = round((top/images(c)) ** (1-balance_alpha)), capped — a
    #   diminishing-returns weighting so sparse concepts don't overtrain
    #   (balance_alpha=0 -> legacy equalization; 0.5 -> sqrt default; 1 -> all 1x).
    target_effective: int | None = None
    balance_alpha: float = 0.5
    cap_mult: float = 1.25
    max_repeats: int = 8


# =============================================================================
# COMPOSITION ROOT
# =============================================================================
@dataclass
class TrainConfig:
    """A full run: the lora side (run/model/adapter/optimizer) + the dataset side."""
    run: RunConfig = field(default_factory=RunConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    adapter: AdapterConfig | None = field(default_factory=AdapterConfig)  # None => FFT
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    # Where the dataset toml will be written; becomes `dataset =` in the lora toml.
    dataset_toml_path: str = "configs/anima_dataset.toml"
    eval_dataset: DatasetConfig | None = None


# =============================================================================
# LOAD  (tomllib — stdlib, read-only)
# =============================================================================
def _take(d: dict, cls: type) -> dict:
    """Filter a dict to the fields of a dataclass; warn on (ignored) extra keys."""
    known = {f.name for f in fields(cls)}
    extra = set(d) - known
    if extra:
        log.warning("%s: ignoring unknown TOML keys %s", cls.__name__, sorted(extra))
    out = {k: v for k, v in d.items() if k in known}
    # tomllib yields lists; coerce the one tuple field (betas) so equality/round-trip hold.
    if cls is OptimizerConfig and isinstance(out.get("betas"), list):
        out["betas"] = tuple(out["betas"])
    return out


def load_dataset_config(path: str | Path) -> DatasetConfig:
    """Parse an anima_dataset.toml into a DatasetConfig (comments dropped)."""
    d = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    dirs = [DirectoryConfig(**_take(b, DirectoryConfig)) for b in d.get("directory", [])]
    top = {k: v for k, v in d.items() if k != "directory"}
    return DatasetConfig(directories=dirs, **_take(top, DatasetConfig))


def load_train_config(lora_toml: str | Path,
                      dataset_toml: str | Path | None = None) -> TrainConfig:
    """Parse anima_lora.toml (+ its referenced dataset toml) into a TrainConfig.

    Comments are NOT preserved (tomllib does not retain them); render_*_toml
    regenerates the canonical comments from code. Documented, intentional.
    """
    lora = tomllib.loads(Path(lora_toml).read_text(encoding="utf-8"))
    model = ModelConfig(**_take(lora.get("model", {}), ModelConfig))
    adapter = (AdapterConfig(**_take(lora["adapter"], AdapterConfig))
               if "adapter" in lora else None)
    optimizer = OptimizerConfig(**_take(lora.get("optimizer", {}), OptimizerConfig))
    run_keys = {k: v for k, v in lora.items()
                if k not in ("model", "adapter", "optimizer", "dataset")}
    run = RunConfig(**_take(run_keys, RunConfig))

    ds_path = dataset_toml or lora.get("dataset")
    dataset = load_dataset_config(ds_path) if ds_path and Path(ds_path).is_file() \
        else DatasetConfig()
    return TrainConfig(run=run, model=model, adapter=adapter, optimizer=optimizer,
                       dataset=dataset,
                       dataset_toml_path=str(ds_path) if ds_path else "")


# =============================================================================
# RENDER  (hand-rendered; the ONLY writers — comment-preserving)
# =============================================================================
def _toml_scalar(v: Any) -> str:
    """Render a Python value as a TOML scalar."""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, str):
        return f"'{v}'"
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_toml_scalar(x) for x in v) + "]"
    return repr(v)


def render_dataset_toml(cfg: DatasetConfig) -> str:
    """Render a DatasetConfig to anima_dataset.toml text, balance comments intact."""
    lines = [
        "# =============================================================================",
        "# anima_dataset.toml  —  MULTI-CONCEPT dataset config (generated)",
        "# Referenced by anima_lora.toml via `dataset = '...'`. Regenerate whenever",
        "# images change so num_repeats stays balanced across concepts.",
        "# Anima is TAG-ORDER sensitive: quality -> subject -> character -> series ->",
        "# @artist -> general. Do NOT shuffle_caption unless keep_tokens protects the",
        "# leading ordered block.",
        "# =============================================================================",
        f"resolutions = {_toml_scalar(cfg.resolutions)}",
        f"enable_ar_bucket = {_toml_scalar(cfg.enable_ar_bucket)}",
        f"min_ar = {cfg.min_ar}",
        f"max_ar = {cfg.max_ar}",
        f"num_ar_buckets = {cfg.num_ar_buckets}",
        f"frame_buckets = {_toml_scalar(cfg.frame_buckets)}",
        f"shuffle_caption = {_toml_scalar(cfg.shuffle_caption)}",
    ]
    if cfg.keep_tokens is not None:
        lines.append(f"keep_tokens = {cfg.keep_tokens}")
    lines.append("")
    for c in cfg.directories:
        head = f"[[directory]]"
        if c.image_count is not None:
            head += (f"   # {c.name or Path(c.path).name}: {c.image_count} imgs "
                     f"x{c.num_repeats} = {c.image_count * c.num_repeats} effective")
        lines += [
            head,
            f"path = {_toml_scalar(c.path)}",
            f"caption_extension = {_toml_scalar(c.caption_extension)}",
            f"num_repeats = {c.num_repeats}",
            "",
        ]
    return "\n".join(lines)


_LORA_HEADER = """\
# =============================================================================
# anima_lora.toml  —  diffusion-pipe training config for CircleStone Anima (2B)
# Trainer: tdrussell/diffusion-pipe (model type 'anima' is native).
# Target: RTX PRO 6000 Blackwell, 96GB -> bf16, no fp8, no block swap, SDPA only.
# Generated by geolip_anima_trainer.config — edit via the Python API or by hand.
# Run:
#   PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
#     deepspeed --num_gpus=N train.py --deepspeed --config <this file>
# ============================================================================="""

_ADAPTER_NOTE = """\
# MULTI-CONCEPT wants more representational room than the single-concept default
# of 32. 64 is a solid start for ~3-8 concepts; push to 96/128 for many/complex
# concepts. More rank = more capacity but more overfit risk on small concepts."""

_LLM_ADAPTER_NOTE = """\
# CRITICAL Anima knob — TWO-PHASE PROTOCOL. The LLM adapter maps Qwen3 embeddings
# into the Cosmos cond space and carries outsized knowledge; it degrades easily.
#   PHASE 1 (default): llm_adapter_lr = 0  -> adapter FROZEN. Train + evaluate.
#   PHASE 2 (only if a brand-new concept won't converge frozen): set a very low LR
#   (1e-6 .. 5e-6 ceiling, never higher) and A/B against the frozen baseline.
#   If general quality drops or other concepts regress, revert to 0. Frozen wins ties."""


def render_lora_toml(cfg: TrainConfig) -> str:
    """Render a TrainConfig to anima_lora.toml text, canonical comments baked in."""
    r = cfg.run
    lines = [_LORA_HEADER, ""]
    lines.append(f"output_dir = {_toml_scalar(r.output_dir)}")
    lines.append(f"dataset    = {_toml_scalar(cfg.dataset_toml_path)}")
    lines.append("")
    for name in ("epochs", "micro_batch_size_per_gpu", "pipeline_stages",
                 "gradient_accumulation_steps", "gradient_clipping", "warmup_steps",
                 "eval_every_n_epochs", "eval_before_first_step",
                 "eval_micro_batch_size_per_gpu", "eval_gradient_accumulation_steps",
                 "save_every_n_epochs", "checkpoint_every_n_minutes", "save_dtype",
                 "activation_checkpointing", "compile", "partition_method",
                 "caching_batch_size", "steps_per_print", "blocks_to_swap"):
        lines.append(f"{name} = {_toml_scalar(getattr(r, name))}")
    if r.map_num_proc is not None:    # decode-worker pool (omitted -> diffusion-pipe default)
        lines.append(f"map_num_proc = {r.map_num_proc}")

    m = cfg.model
    lines += ["", "[model]",
              f"type = {_toml_scalar(m.type)}",
              f"transformer_path = {_toml_scalar(m.transformer_path)}",
              f"vae_path         = {_toml_scalar(m.vae_path)}",
              f"llm_path         = {_toml_scalar(m.llm_path)}",
              f"dtype = {_toml_scalar(m.dtype)}",
              _LLM_ADAPTER_NOTE,
              f"llm_adapter_lr = {_toml_scalar(m.llm_adapter_lr)}"]

    if cfg.adapter is not None:
        a = cfg.adapter
        lines += ["", "[adapter]", _ADAPTER_NOTE,
                  f"type = {_toml_scalar(a.type)}",
                  f"rank = {a.rank}",
                  f"dtype = {_toml_scalar(a.dtype)}"]
        if a.init_from_existing:
            lines.append(f"init_from_existing = {_toml_scalar(a.init_from_existing)}")
    else:
        lines += ["", "# FULL FINE-TUNE: [adapter] omitted; keep optimizer lr low (~5e-6..1e-5)."]

    o = cfg.optimizer
    lines += ["", "[optimizer]",
              f"type = {_toml_scalar(o.type)}",
              f"lr = {_toml_scalar(o.lr)}",
              f"betas = {_toml_scalar(list(o.betas))}",
              f"weight_decay = {o.weight_decay}",
              f"eps = {_toml_scalar(o.eps)}", ""]
    return "\n".join(lines)


def render_train_toml(cfg: TrainConfig, out_dir: str | Path,
                      *, validate_first: bool = True) -> tuple[Path, Path]:
    """Write both toml files into out_dir; wire the lora `dataset =` to the dataset
    path. Returns (lora_path, dataset_path). The single writer in the system."""
    if validate_first:
        validate(cfg)
    out_dir = Path(out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    ds_path = out_dir / "anima_dataset.toml"
    ds_path.write_text(render_dataset_toml(cfg.dataset), encoding="utf-8")
    cfg = replace(cfg, dataset_toml_path=str(ds_path))
    lora_path = out_dir / "anima_lora.toml"
    lora_path.write_text(render_lora_toml(cfg), encoding="utf-8")
    return lora_path, ds_path


# =============================================================================
# BALANCING ADAPTERS  (reuse build_multiconcept_dataset — do NOT reimplement)
# =============================================================================
def discover_directories(root: str | Path) -> list[DirectoryConfig]:
    """Scan concept subfolders into DirectoryConfigs (counts filled, repeats=1)."""
    from . import build_multiconcept_dataset as _b
    cfg = _b.BaseConfig()
    concepts = _b.discover_concepts(Path(root).expanduser().resolve(), cfg)
    return [DirectoryConfig(path=c["path"], name=c["name"],
                            image_count=c["images"], caption_count=c["captions"])
            for c in concepts]


def rebalance(ds: DatasetConfig) -> None:
    """Recompute num_repeats in place via the canonical compute_repeats balancer."""
    from . import build_multiconcept_dataset as _b
    if not any(d.image_count for d in ds.directories):
        return
    bcfg = _b.BaseConfig(target_effective=ds.target_effective, max_repeats=ds.max_repeats,
                         balance_alpha=ds.balance_alpha, cap_mult=ds.cap_mult)
    concepts = [{"name": d.name, "path": d.path, "images": d.image_count or 0,
                 "captions": d.caption_count or 0} for d in ds.directories]
    _b.compute_repeats(concepts, bcfg)
    for d, c in zip(ds.directories, concepts):
        d.num_repeats = c["repeats"]


# =============================================================================
# OPTIMIZABLE SURFACE  (overrides, presets, sweeps)
# =============================================================================
# The real sweep axes — a flat, deliberate allowlist (not reflective magic).
_OVERRIDE_AXES = {
    "rank", "lr", "llm_adapter_lr", "resolutions", "target_effective",
    "balance_alpha", "cap_mult",
    "output_dir", "micro_batch_size_per_gpu", "gradient_accumulation_steps",
}


def apply_overrides(base: TrainConfig, /, **ov: Any) -> TrainConfig:
    """Return a NEW validated TrainConfig with overrides applied (base untouched)."""
    unknown = set(ov) - _OVERRIDE_AXES
    if unknown:
        raise ConfigError(f"unknown override axes {sorted(unknown)}; "
                          f"allowed: {sorted(_OVERRIDE_AXES)}")
    cfg = deepcopy(base)
    if "rank" in ov:
        if cfg.adapter is None:
            raise ConfigError("cannot override rank on a full fine-tune (adapter=None)")
        cfg.adapter = replace(cfg.adapter, rank=ov["rank"])
    if "lr" in ov:
        cfg.optimizer = replace(cfg.optimizer, lr=ov["lr"])
    if "llm_adapter_lr" in ov:
        cfg.model = replace(cfg.model, llm_adapter_lr=ov["llm_adapter_lr"])
    if "resolutions" in ov:
        cfg.dataset = replace(cfg.dataset, resolutions=list(ov["resolutions"]))
    if "target_effective" in ov:
        cfg.dataset = replace(cfg.dataset, target_effective=ov["target_effective"])
        rebalance(cfg.dataset)
    if "balance_alpha" in ov:
        cfg.dataset = replace(cfg.dataset, balance_alpha=ov["balance_alpha"])
        rebalance(cfg.dataset)
    if "cap_mult" in ov:
        cfg.dataset = replace(cfg.dataset, cap_mult=ov["cap_mult"])
        rebalance(cfg.dataset)
    if "output_dir" in ov:
        cfg.run = replace(cfg.run, output_dir=ov["output_dir"])
    if "micro_batch_size_per_gpu" in ov:
        cfg.run = replace(cfg.run, micro_batch_size_per_gpu=ov["micro_batch_size_per_gpu"])
    if "gradient_accumulation_steps" in ov:
        cfg.run = replace(cfg.run, gradient_accumulation_steps=ov["gradient_accumulation_steps"])
    return validate(cfg)


def single_concept_preset(concept_dir: str, *, output_dir: str,
                          model: ModelConfig, resolution: int = 1024) -> TrainConfig:
    """PRELIM_PLAN single-concept decision: rank 32, ONE [[directory]], adapter frozen."""
    return validate(TrainConfig(
        run=RunConfig(output_dir=output_dir),
        model=replace(model, llm_adapter_lr=0.0),
        adapter=AdapterConfig(rank=32),
        optimizer=OptimizerConfig(lr=2e-5),
        dataset=DatasetConfig(resolutions=[resolution],
                              directories=[DirectoryConfig(path=concept_dir, num_repeats=1)]),
    ))


def multi_concept_preset(root: str, *, output_dir: str, model: ModelConfig,
                         rank: int = 64, resolution: int = 1024,
                         target_effective: int | None = None) -> TrainConfig:
    """PRELIM_PLAN multi-concept decision: rank 64, balanced multi-dir, adapter frozen."""
    dirs = discover_directories(root)
    ds = DatasetConfig(resolutions=[resolution], directories=dirs,
                       target_effective=target_effective)
    rebalance(ds)
    return validate(TrainConfig(
        run=RunConfig(output_dir=output_dir),
        model=replace(model, llm_adapter_lr=0.0),
        adapter=AdapterConfig(rank=rank),
        optimizer=OptimizerConfig(lr=2e-5), dataset=ds,
    ))


def sweep(base: TrainConfig, *, ranks: list[int], lrs: list[float],
          runs_root: str, configs_root: str = "configs/sweep") -> Iterator[tuple[str, Path]]:
    """Emit one resolved (tag, lora_toml_path) per rank x lr grid point. No hand-editing.

    Each grid point gets its own output_dir (runs_root/tag) and config dir so trainer
    invocations never collide. Yields lazily so callers can launch as they go.
    """
    for rank, lr in product(ranks, lrs):
        tag = f"r{rank}_lr{lr:.0e}"
        cfg = apply_overrides(base, rank=rank, lr=lr,
                              output_dir=str(Path(runs_root) / tag))
        lora_path, _ = render_train_toml(cfg, out_dir=str(Path(configs_root) / tag))
        yield tag, lora_path


# =============================================================================
# VALIDATION  (one home for the Anima invariants)
# =============================================================================
def validate(cfg: TrainConfig, *, strict: bool = True) -> TrainConfig:
    """Enforce the non-negotiable Anima rules. Errors stop (strict); warnings advise."""
    errs: list[str] = []
    warns: list[str] = []

    # -- tag-order invariant --------------------------------------------------
    if cfg.dataset.shuffle_caption and cfg.dataset.keep_tokens is None:
        errs.append("shuffle_caption=true requires keep_tokens to protect the ordered "
                    "quality->subject->character->series->@artist block (Anima is "
                    "tag-order sensitive).")

    # -- frozen-adapter invariant --------------------------------------------
    lr = cfg.model.llm_adapter_lr
    if lr < 0:
        errs.append("llm_adapter_lr must be >= 0.")
    elif lr > 5e-6:
        errs.append(f"llm_adapter_lr={lr} exceeds the 5e-6 ceiling; the adapter degrades. "
                    "Frozen (0) is the default; phase-2 opt-in tops out at 5e-6.")
    elif lr > 0:
        warns.append(f"llm_adapter_lr={lr}: phase-2 opt-in — A/B against a frozen baseline "
                     "and revert if general quality or other concepts regress.")

    # -- optimizer lr ---------------------------------------------------------
    if cfg.optimizer.lr <= 0:
        errs.append("optimizer.lr must be > 0.")

    # -- rank sanity ----------------------------------------------------------
    if cfg.adapter is not None:
        r = cfg.adapter.rank
        n_dirs = len(cfg.dataset.directories)
        if r <= 0:
            errs.append("adapter.rank must be > 0.")
        elif r & (r - 1):
            warns.append(f"rank={r} is not a power of two (unusual).")
        if r > 256:
            warns.append(f"rank={r} is very high; overfit risk on small data.")
        if n_dirs == 1 and r > 32:
            warns.append(f"single [[directory]] with rank={r}: PRELIM_PLAN suggests rank 32.")
        if n_dirs > 1 and r < 64:
            warns.append(f"multi-concept ({n_dirs} dirs) with rank={r}: capacity may be tight.")

    # -- precision / hardware guards -----------------------------------------
    for label, val in (("model.dtype", cfg.model.dtype),
                       ("optimizer.type", cfg.optimizer.type),
                       ("save_dtype", cfg.run.save_dtype)):
        s = str(val).lower()
        if "fp8" in s or "e4m3" in s:
            warns.append(f"{label}={val}: fp8/e4m3 degrades this lineage; bf16 only on 96GB.")
    if cfg.model.dtype != "bfloat16":
        warns.append(f"model.dtype={cfg.model.dtype}: the brief mandates bf16 throughout.")
    if cfg.run.blocks_to_swap or cfg.run.activation_checkpointing:
        warns.append("block_swap/activation_checkpointing enabled: unnecessary on 96GB, slower.")

    # -- topology -------------------------------------------------------------
    if cfg.run.pipeline_stages < 1:
        errs.append("pipeline_stages must be >= 1.")

    # -- train/eval resolution consistency (shared latent cache) -------------
    if cfg.eval_dataset and cfg.eval_dataset.resolutions != cfg.dataset.resolutions:
        errs.append("eval resolutions must match train resolutions (they share the cache).")

    # -- balance sanity -------------------------------------------------------
    effs = [(d.image_count or 0) * d.num_repeats for d in cfg.dataset.directories
            if d.image_count]
    # Under diminishing-returns weighting (balance_alpha>0) a >3x effective spread is
    # INTENDED (sparse concepts are deliberately under-exposed to avoid overtraining),
    # so only warn in legacy equalization mode.
    if (cfg.dataset.balance_alpha == 0 and len(effs) > 1
            and min(effs) and max(effs) > 3 * min(effs)):
        warns.append("effective sample counts differ >3x across concepts; re-balance "
                     "(call rebalance() / build_multiconcept_dataset.py).")

    for w in warns:
        log.warning(w)
    if errs and strict:
        raise ConfigError("invalid TrainConfig:\n  - " + "\n  - ".join(errs))
    return cfg


def validate_bridge(cfg, *, audit_filled: bool | None = None,
                    age_filled: bool | None = None):
    """Warn when quality gates are ON but --inspect reported the column empty/unpopulated
    (qwen_90k may not carry these flags). `cfg` is a BridgeConfig."""
    if getattr(cfg, "require_audit_approved", False) and audit_filled is False:
        log.warning("audit gate is ON but the audit column looks unpopulated for this "
                    "source — pass --no-audit-filter or all rows will be dropped.")
    if getattr(cfg, "require_age_pass", False) and age_filled is False:
        log.warning("age gate is ON but age_classifier_pass looks unpopulated for this "
                    "source — pass --no-age-filter or most rows will be dropped.")
    return cfg
