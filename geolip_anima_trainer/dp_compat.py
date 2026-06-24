#!/usr/bin/env python3
"""
dp_compat.py — compatibility shims for the diffusion-pipe training subprocess.

`patch_datasets_map_makedirs()` fixes a diffusion-pipe + HuggingFace `datasets`
incompatibility that CRASHES — then silently HANGS — the `--cache_only` step.

The bug: diffusion-pipe writes per-AR/size-bucket metadata with
    Dataset.map(..., cache_file_name="<dir>/ar_frames_X/metadata/metadata_*.arrow")
but only creates `ar_frames_X/`, not its `metadata/` subdir (verified in
external/diffusion-pipe/utils/dataset.py:410 makes the bucket dir; :429 writes the
map cache file into `metadata/`). `datasets` (>=2.x, confirmed 2.21) opens that file with
`tempfile.NamedTemporaryFile(dir=os.path.dirname(cache_file_name))` and does NOT create
the parent, so `map()` raises `FileNotFoundError`. It only works for the directory-level
metadata because that path is created as a side effect of an earlier `save_to_disk`.

Why it HANGS (not just errors): the crash happens inside a forked child process
(`_cache_fn`, dataset.py:1047) that never puts its `None` sentinel on the queue, so the
parent process blocks on `queue.get()` (dataset.py:1184) forever. The launcher never
returns. So this single missing `mkdir` turns into an unkillable wait.

The fix wraps `Dataset.map` to `os.makedirs(dirname(cache_file_name))` first — universally
safe (you always want the directory of a file you're about to write to exist) and
idempotent. `launch.env_prefix()` puts `shim_dir()` on PYTHONPATH so the bundled
`_dp_compat/sitecustomize.py` auto-applies this in every diffusion-pipe subprocess BEFORE
it forks; this function is the same patch, callable directly for in-process use / tests.
"""

from __future__ import annotations

import os
from pathlib import Path


def shim_dir() -> str:
    """Directory holding the auto-loaded `sitecustomize.py` — put on PYTHONPATH so the
    diffusion-pipe subprocess applies the datasets patch at interpreter startup."""
    return str(Path(__file__).resolve().parent / "_dp_compat")


def patch_datasets_map_makedirs() -> bool:
    """Make `datasets.Dataset.map` create the parent dir of an explicit `cache_file_name`.
    Returns True if patched (or already patched), False if `datasets` isn't importable.
    Idempotent — safe to call repeatedly; the wrapper is tagged to avoid double-wrapping."""
    try:
        from datasets import Dataset
    except Exception:  # noqa: BLE001 — datasets absent in this process -> nothing to patch
        return False
    if getattr(Dataset.map, "_anima_mkdir_patched", False):
        return True
    _orig = Dataset.map

    def map(self, *args, **kwargs):  # noqa: A001,A003 — must shadow the method name
        cfn = kwargs.get("cache_file_name")
        if cfn:
            d = os.path.dirname(str(cfn))
            if d:
                os.makedirs(d, exist_ok=True)
        return _orig(self, *args, **kwargs)

    map._anima_mkdir_patched = True  # type: ignore[attr-defined]
    try:
        Dataset.map = map  # type: ignore[method-assign]
    except Exception:  # noqa: BLE001 — never let a compat shim break the run
        return False
    return True
