#!/usr/bin/env python3
"""
launch.py — build and run diffusion-pipe's deepspeed training command.

Design: the command is a PURE, immutable data structure (LaunchPlan) built by a
side-effect-free function (build_plan). Execution (launch) is a separate, platform-
guarded step. This is what makes the Windows dry-run smoke test trivial — you build
and assert on the exact argv on ANY OS, and only launch() touches the platform.

diffusion-pipe is launched natively multi-GPU via deepspeed:
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      deepspeed --num_gpus=N train.py --deepspeed --config <lora.toml> [--cache_only]

Multi-GPU on a SHARED box: prefer explicit --gpu-ids (-> deepspeed --include
localhost:i,j + matching CUDA_VISIBLE_DEVICES) so a job pins to the cards you own,
rather than --num-gpus which grabs the first N visible devices.
"""

from __future__ import annotations

import os
import platform
import shlex
import subprocess
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class WindowsTrainingRefused(RuntimeError):
    """Raised when a real (non-dry-run) deepspeed launch is attempted on Windows."""


class DiffusionPipeNotFound(FileNotFoundError):
    """Raised when external/diffusion-pipe/train.py cannot be located."""


_CLONE_REMEDY = (
    "git clone --recurse-submodules "
    "https://github.com/tdrussell/diffusion-pipe external/diffusion-pipe"
)


# =============================================================================
# PLAN  (pure data; building it never touches GPUs / never execs)
# =============================================================================
@dataclass(frozen=True)
class LaunchPlan:
    train_py: Path
    config_toml: Path
    num_gpus: int | None = None              # data-parallel world size
    gpu_ids: tuple[int, ...] | None = None   # explicit devices -> --include localhost:i,j
    pipeline_stages: int = 1                  # from the toml; for batch-math + validation
    micro_batch_size_per_gpu: int = 1
    gradient_accumulation_steps: int = 1
    cache_only: bool = False
    regenerate_cache: bool = False
    resume_from_checkpoint: bool | str = False
    env: dict[str, str] = field(default_factory=dict)
    extra_args: tuple[str, ...] = ()

    # ---- derived -----------------------------------------------------------
    @property
    def world_size(self) -> int:
        if self.gpu_ids is not None:
            return len(self.gpu_ids)
        return self.num_gpus or 1

    @property
    def dp_size(self) -> int:
        return self.world_size // self.pipeline_stages

    @property
    def effective_batch(self) -> int:
        return self.micro_batch_size_per_gpu * self.gradient_accumulation_steps * self.dp_size

    # ---- rendering ---------------------------------------------------------
    def argv(self) -> list[str]:
        """The exact argv list (no shell, no quoting bugs)."""
        cmd = ["deepspeed"]
        if self.gpu_ids is not None:
            cmd += ["--include", "localhost:" + ",".join(str(i) for i in self.gpu_ids)]
        elif self.num_gpus is not None:
            cmd += [f"--num_gpus={self.num_gpus}"]
        cmd += [str(self.train_py), "--deepspeed", "--config", str(self.config_toml)]
        if self.cache_only:
            cmd += ["--cache_only"]
        if self.regenerate_cache:
            cmd += ["--regenerate_cache"]
        if self.resume_from_checkpoint is True:
            cmd += ["--resume_from_checkpoint"]
        elif isinstance(self.resume_from_checkpoint, str):
            cmd += ["--resume_from_checkpoint", self.resume_from_checkpoint]
        cmd += list(self.extra_args)
        return cmd

    def env_prefix(self) -> dict[str, str]:
        """Env overlay applied on top of os.environ at exec time."""
        e = dict(self.env)
        e.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        if self.gpu_ids is not None:
            # deepspeed --include scopes the launch, but some child paths read
            # CUDA_VISIBLE_DEVICES; set both so they agree.
            e.setdefault("CUDA_VISIBLE_DEVICES", ",".join(str(i) for i in self.gpu_ids))
        return e

    def pretty(self) -> str:
        """A copy-pasteable shell line (for dry-run / docs / Linux operators)."""
        env = " ".join(f"{k}={shlex.quote(v)}" for k, v in sorted(self.env_prefix().items()))
        return env + " \\\n  " + " ".join(shlex.quote(a) for a in self.argv())

    def batch_summary(self) -> str:
        return (f"GPUs={self.world_size}  pipeline_stages={self.pipeline_stages} "
                f"-> dp_size={self.dp_size}\n"
                f"micro_batch_size_per_gpu={self.micro_batch_size_per_gpu}  "
                f"grad_accum={self.gradient_accumulation_steps}  "
                f"-> effective_batch = {self.micro_batch_size_per_gpu} * "
                f"{self.gradient_accumulation_steps} * {self.dp_size} = {self.effective_batch}")


# =============================================================================
# LOCATE  diffusion-pipe
# =============================================================================
def find_diffusion_pipe(repo_root: str | Path | None = None) -> Path:
    """Locate external/diffusion-pipe/train.py.

    If repo_root is given it is AUTHORITATIVE — only that location is checked (no
    silent fallback to an unrelated clone). When repo_root is None the resolution
    order is $ANIMA_DIFFUSION_PIPE -> walk up from cwd. Raises with the clone remedy.
    """
    candidates: list[Path] = []
    if repo_root is not None:
        candidates.append(Path(repo_root) / "external" / "diffusion-pipe" / "train.py")
    else:
        env = os.environ.get("ANIMA_DIFFUSION_PIPE")
        if env:
            p = Path(env)
            candidates += [p / "train.py", p]  # may point at the dir or the file
        here = Path.cwd().resolve()
        for parent in [here, *here.parents]:
            candidates.append(parent / "external" / "diffusion-pipe" / "train.py")
    for c in candidates:
        if c.is_file():
            return c.resolve()
        if c.name != "train.py" and (c / "train.py").is_file():
            return (c / "train.py").resolve()
    raise DiffusionPipeNotFound(
        "Could not find external/diffusion-pipe/train.py. Clone it with:\n  " + _CLONE_REMEDY
    )


# =============================================================================
# BUILD  (OS-agnostic; fails loud)
# =============================================================================
def build_plan(
    *,
    config_toml: str | Path,
    repo_root: str | Path | None = None,
    num_gpus: int = 1,
    gpu_ids: list[int] | None = None,
    pipeline_stages: int | None = None,
    cache_only: bool = False,
    regenerate_cache: bool = False,
    resume_from_checkpoint: bool | str = False,
    expandable_segments: bool = True,
    extra_env: dict[str, str] | None = None,
    extra_args: list[str] | None = None,
) -> LaunchPlan:
    """Construct a fully-resolved LaunchPlan. Reads (but never writes) the toml for
    batch math + topology validation. Never spawns, never touches the GPU."""
    train_py = find_diffusion_pipe(repo_root)

    cfg_path = Path(config_toml).expanduser().resolve()
    if not cfg_path.is_file():
        raise FileNotFoundError(f"config toml not found: {cfg_path}")

    toml = tomllib.loads(cfg_path.read_text(encoding="utf-8"))
    toml_stages = int(toml.get("pipeline_stages", 1))
    micro = int(toml.get("micro_batch_size_per_gpu", 1))
    grad_accum = int(toml.get("gradient_accumulation_steps", 1))
    stages = int(pipeline_stages) if pipeline_stages is not None else toml_stages

    # Reconcile gpu_ids vs num_gpus.
    ids = tuple(gpu_ids) if gpu_ids else None
    if ids is not None:
        world = len(ids)
        if num_gpus not in (1, world):
            raise ValueError(f"--num-gpus={num_gpus} conflicts with {len(ids)} --gpu-ids")
        num_gpus_field: int | None = None  # use --include, not --num_gpus
    else:
        if num_gpus < 1:
            raise ValueError("--num-gpus must be >= 1")
        world = num_gpus
        num_gpus_field = num_gpus

    # Topology checks.
    if stages < 1:
        raise ValueError("pipeline_stages must be >= 1")
    if stages > world:
        raise ValueError(f"pipeline_stages={stages} exceeds world size {world}")
    if world % stages != 0:
        raise ValueError(f"world size {world} not divisible by pipeline_stages={stages}")

    env = dict(extra_env or {})
    if not expandable_segments:
        env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    else:
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    return LaunchPlan(
        train_py=train_py, config_toml=cfg_path,
        num_gpus=num_gpus_field, gpu_ids=ids,
        pipeline_stages=stages, micro_batch_size_per_gpu=micro,
        gradient_accumulation_steps=grad_accum,
        cache_only=cache_only, regenerate_cache=regenerate_cache,
        resume_from_checkpoint=resume_from_checkpoint,
        env=env, extra_args=tuple(extra_args or ()),
    )


# =============================================================================
# EXEC  (platform-guarded)
# =============================================================================
def launch(plan: LaunchPlan, *, dry_run: bool = False, check: bool = True) -> Any:
    """Execute the plan. dry_run prints the command on any OS and returns the plan.
    On Windows a real launch raises WindowsTrainingRefused (smoke-test box). On Linux
    it execs deepspeed from the diffusion-pipe dir, inheriting stdio."""
    if plan.pipeline_stages > 1:
        print("[warn] pipeline parallelism is unnecessary for a 2B model on 96GB; "
              "data-parallel (pipeline_stages=1) is faster.")

    if dry_run:
        print(plan.batch_summary())
        print(plan.pretty())
        return plan

    if platform.system() == "Windows":
        raise WindowsTrainingRefused(
            "diffusion-pipe + deepspeed do not run on Windows. This box is INSTALL + "
            "IMPORT + smoke-test only. Use --dry-run to print the command, or run on the "
            "Linux target. Constructed command:\n" + plan.pretty()
        )

    env = {**os.environ, **plan.env_prefix()}
    proc = subprocess.run(plan.argv(), env=env, cwd=str(plan.train_py.parent))
    if check and proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, plan.argv())
    return proc.returncode
