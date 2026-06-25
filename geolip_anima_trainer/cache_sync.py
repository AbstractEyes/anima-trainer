#!/usr/bin/env python3
"""
cache_sync.py — preserve the diffusion-pipe --cache_only output on HuggingFace Hub and
resume it across sessions (ephemeral Colab loses the local cache on a runtime reset).

The model (verified in external/diffusion-pipe/utils/{cache,dataset}.py):
  * The cache is a sharded blob store under `<dataset_dir>/cache/anima/**`: `metadata.db`
    (SQLite, the committed items + the fingerprint) + `shard_*.bin` (10 GB shards) + the
    HF-datasets `metadata/*.arrow`. Items commit only at a 10 GB shard finalize or pass end,
    so a mid-run copy is SAFE — resume re-does at most the current <=10 GB shard, never
    corrupts a committed shard (cache.py:91 re-opens the in-progress shard 'wb').
  * The latent cache is keyed by a fingerprint that embeds ABSOLUTE image paths
    (cache.py:45-52: mismatch -> clear()). So resume requires the dataset restored to an
    IDENTICAL absolute DATA_ROOT with byte-identical files. We therefore FREEZE the dataset
    and sync it WITH the cache (one HF *dataset* repo), and pin DATA_ROOT.

This module is the single source of truth for "where the cache lives" (it reuses
cache_monitor's path helpers) and the push/pull primitives. No torch/GPU.
"""

from __future__ import annotations

import glob
import logging
import os
import sqlite3
from pathlib import Path

from . import cache_monitor as _cm

log = logging.getLogger("anima.cache_sync")

DEFAULT_MODEL = "anima"
_C = "**/cache/" + DEFAULT_MODEL
# We persist ONLY the EXPENSIVE, non-regenerable cache: the latent/text-embed `metadata.db` (the
# committed-record index) + `shard_*.bin` (the latents/embeds that cost ~4 h), plus the small
# reconstruct `index.jsonl`. NOT the images/.txt (refetched from source / stored in the index) and
# NOT the HF-datasets `*.arrow` metadata — that's tens of thousands of tiny files that HF throttles,
# and it REGENERATES deterministically from the reconstructed images on resume (the dataset
# fingerprint is content-based). So a push is a few hundred files, not ~30k.
CACHE_ONLY_GLOB = [f"{_C}/**/metadata.db", f"{_C}/**/*.bin", "index.jsonl"]
# Images + .txt are NEVER uploaded (regenerated from the index/source on pull) — that's the point.
IMAGE_GLOB = ["**/*.png", "**/*.jpg", "**/*.jpeg", "**/*.webp", "**/*.bmp", "**/*.gif"]
# Excluded from EVERY push: hub cache, locks, images, captions, the regenerable HF-datasets metadata
# (*.arrow + its json), and SQLite sidecars (the -journal/-wal/-shm vanish mid-upload -> "not a file").
_DEFAULT_IGNORE = ["**/hf_cache/**", "**/.cache/**", "*.lock", "**/*.txt", *IMAGE_GLOB,
                   "**/*-journal", "**/*-wal", "**/*-shm",
                   "**/*.arrow", "**/dataset_info.json", "**/state.json", "**/grouping_keys.json"]


# =============================================================================
# WHERE THE CACHE LIVES
# =============================================================================
def cache_targets_from_toml(config_toml: str | Path, model_name: str = DEFAULT_MODEL) -> list[Path]:
    """The cache roots (`<dir>/cache/<model>`) for every [[directory]] in a dataset/lora toml."""
    return _cm.cache_roots_for(_cm.dataset_dirs_from_toml(config_toml), model_name)


def out_root_of(config_toml: str | Path) -> Path | None:
    """The common ancestor of all [[directory]] paths — the frozen tree to sync whole. For
    before_after the dirs are `<out_root>/{vlm,animetimm}/<bucket>`, so the ancestor is
    `<out_root>` (the `anima_subjects` dir holding both trees + their caches)."""
    dirs = [str(Path(d).resolve()) for d in _cm.dataset_dirs_from_toml(config_toml)]
    if not dirs:
        return None
    if len(dirs) == 1:
        return Path(dirs[0])
    return Path(os.path.commonpath(dirs))


def read_cache_fingerprints(roots) -> dict[str, str | None]:
    """leaf-cache-dir -> stored `fingerprint(value)` row (or None). Read-only, never raises.
    Use after a pull to log what's present; a later `[CACHE] Fingerprint changed` in the run
    log means that leaf will be cleared (DATA_ROOT drift / the dataset changed)."""
    out: dict[str, str | None] = {}
    for root in roots:
        for db in glob.glob(str(Path(root) / "**" / "metadata.db"), recursive=True):
            try:
                con = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=0.25)
                try:
                    row = con.execute("SELECT value FROM fingerprint").fetchone()
                    out[str(Path(db).parent)] = row[0] if row else None
                finally:
                    con.close()
            except sqlite3.Error:
                out[str(Path(db).parent)] = None
    return out


# =============================================================================
# PUSH / PULL  (mirror the notebook backup_latest; repo_type='dataset')
# =============================================================================
def sync_up(folder: str | Path, repo_id: str, *, token: str | None = None,
            path_in_repo: str = ".", include_dataset: bool = True,
            allow_patterns: list[str] | None = None,
            ignore_patterns: list[str] | None = None,
            commit_message: str = "cache sync", dry_run: bool = False) -> str:
    """Upload `folder` to an HF dataset repo (incremental — upload_folder skips unchanged files
    by content hash). include_dataset=False restricts to the cache subtree (CACHE_ONLY_GLOB)."""
    folder = Path(folder).expanduser().resolve()
    if allow_patterns is None and not include_dataset:
        allow_patterns = list(CACHE_ONLY_GLOB)
    if ignore_patterns is None:
        ignore_patterns = list(_DEFAULT_IGNORE)
    url = f"https://huggingface.co/datasets/{repo_id}"
    if dry_run:
        log.info("[dry-run] would upload %s -> %s (allow=%s)", folder, url, allow_patterns)
        return url
    from huggingface_hub import HfApi, create_repo
    create_repo(repo_id, token=token, repo_type="dataset", private=True, exist_ok=True)
    api = HfApi(token=token)
    if hasattr(api, "upload_large_folder"):     # HF's path for many files: batched + resumable
        api.upload_large_folder(
            repo_id=repo_id, folder_path=str(folder), repo_type="dataset",
            allow_patterns=allow_patterns, ignore_patterns=ignore_patterns, print_report=False)
    else:
        api.upload_folder(
            folder_path=str(folder), repo_id=repo_id, repo_type="dataset",
            path_in_repo=path_in_repo, allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns, commit_message=commit_message)
    return url


def make_periodic_pusher(folder, repo_id, *, token=None, interval: float = 1800.0,
                         user_on_update=None, _sync_up=None):
    """Build `(on_update, final_push)` for cache().on_update. `on_update(info)` pushes the CACHE
    subtree whenever the committed-record count bumps (a 10 GB shard finalized -> a new restorable
    checkpoint) or `interval` seconds elapse (fallback), composing the user's own on_update first.
    `final_push()` forces one last push. A failed push is logged, never raised (can't kill the run).
    `_sync_up` is injectable for tests; clock is `info["elapsed"]` (no wall-clock dependence)."""
    up = _sync_up or sync_up
    state = {"committed": 0, "last": 0.0}

    def _push(tag):
        try:
            up(folder, repo_id, token=token, include_dataset=False,
               commit_message=f"cache backup :: {tag}")
            log.info("cache backed up to %s (%s)", repo_id, tag)
        except Exception as e:  # noqa: BLE001 — a failed backup must never kill the run
            log.warning("cache backup failed (%s): %s", tag, e)

    def on_update(info):
        if user_on_update:
            try:
                user_on_update(info)
            except Exception:  # noqa: BLE001
                pass
        lat, txt = info.get("latents"), info.get("text")
        committed = ((lat[0] if isinstance(lat, tuple) else 0) +
                     (txt[0] if isinstance(txt, tuple) else 0))
        el = info.get("elapsed", 0.0)
        if committed > state["committed"] or (el - state["last"]) >= interval:
            state["committed"] = max(committed, state["committed"])
            state["last"] = el
            _push(f"{int(el)}s/{committed}rec")

    return on_update, lambda: _push("final")


def sync_down(local_dir: str | Path, repo_id: str, *, token: str | None = None,
              allow_patterns: list[str] | None = None,
              ignore_patterns: list[str] | None = None, dry_run: bool = False) -> str:
    """Download an HF dataset repo into a FIXED `local_dir` as REAL files (not the
    content-addressed hub cache), so restored absolute paths are deterministic and match the
    cache fingerprint. Pair with a pinned DATA_ROOT."""
    local_dir = str(Path(local_dir).expanduser().resolve())
    if dry_run:
        log.info("[dry-run] would download %s -> %s", repo_id, local_dir)
        return local_dir
    from huggingface_hub import snapshot_download
    return snapshot_download(repo_id=repo_id, repo_type="dataset", local_dir=local_dir,
                             token=token, allow_patterns=allow_patterns,
                             ignore_patterns=ignore_patterns)
