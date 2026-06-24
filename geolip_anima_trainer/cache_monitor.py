#!/usr/bin/env python3
"""
cache_monitor.py — live progress + ETA for diffusion-pipe's `--cache_only` step.

The cache step (deepspeed train.py --cache_only) is otherwise a blind, open-ended wait.
Investigation of diffusion-pipe (utils/dataset.py, utils/cache.py) established the
authoritative progress source:

  * The cache is a SHARDED BLOB store, not one file per image: each Cache dir holds a
    SQLite `metadata.db` + `shard_*.bin` (a new shard only every ~10 GB). So counting
    files is useless — the exact, monotonic DONE counter is `SELECT COUNT(*) FROM items`.
  * Cache root per [[directory]] = `<dir>/cache/anima/`. Under it, per size-bucket:
    `.../latents_/metadata.db` (1 row per UNIQUE image; num_repeats does NOT inflate it)
    and `.../text_embeddings_1_/metadata.db` (1 row per image x caption).
  * skip-existing makes the COUNT resumable.

So: DONE = sum of COUNT(*) across every cache metadata.db; TOTAL = images-on-disk
(latents) + images x captions_per_image (text-embeds). A rolling-window rate gives a
stable ETA. tqdm on stderr is NOT parsed (per-bucket, no global total).

No torch / GPU / model needed — pure stdlib (sqlite3, glob, tomllib).
"""

from __future__ import annotations

import glob
import sqlite3
import time
import tomllib
from collections import deque
from pathlib import Path
from typing import Callable

IMG_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}


# =============================================================================
# COUNTING
# =============================================================================
def _db_count(db_path: str) -> int:
    """COUNT(*) of the `items` table in one Cache metadata.db. 0 on any error
    (table not created yet / transient lock). Read-only + immutable so it never
    blocks or corrupts the writer."""
    try:
        con = sqlite3.connect(f"file:{db_path}?mode=ro&immutable=1", uri=True, timeout=0.25)
        try:
            return int(con.execute("SELECT COUNT(*) FROM items").fetchone()[0])
        finally:
            con.close()
    except sqlite3.Error:
        return 0


def count_done(cache_roots: list[Path],
               kinds: tuple[str, ...] = ("latents_", "text_embeddings_")) -> dict[str, int]:
    """Sum item rows across every Cache metadata.db, bucketed by kind (the parent
    dir name starts with 'latents_' / 'text_embeddings_')."""
    totals = {k: 0 for k in kinds}
    for root in cache_roots:
        for db in glob.glob(str(Path(root) / "**" / "metadata.db"), recursive=True):
            parent = Path(db).parent.name
            for k in kinds:
                if parent.startswith(k):
                    totals[k] += _db_count(db)
    return totals


def count_total_images(dataset_dirs: list[Path]) -> int:
    """Unique media files across all [[directory]] paths — top-level glob('*') only
    (mirrors diffusion-pipe's own enumeration), excluding sidecars and the cache subtree."""
    n = 0
    for d in dataset_dirs:
        d = Path(d)
        if not d.is_dir():
            continue
        for f in d.glob("*"):
            if f.is_file() and f.suffix.lower() in IMG_EXT:
                n += 1
    return n


def _captions_per_image(dataset_dirs: list[Path]) -> int:
    """1 for sidecar-.txt dirs; the max captions.json list length for online_captions
    (MIXED) dirs — so the caller never has to keep the number in sync."""
    import json
    best = 1
    for d in dataset_dirs:
        cj = Path(d) / "captions.json"
        if cj.is_file():
            try:
                m = json.loads(cj.read_text(encoding="utf-8"))
                if isinstance(m, dict) and m:
                    best = max(best, max(len(v) for v in m.values() if isinstance(v, list)))
            except (json.JSONDecodeError, ValueError):
                pass
    return best


def dataset_dirs_from_toml(config_toml: str | Path) -> list[Path]:
    """Resolve the [[directory]] image paths. Accepts a dataset.toml directly, or a
    lora.toml whose `dataset = '...'` points at the dataset.toml."""
    p = Path(config_toml)
    data = tomllib.loads(p.read_text(encoding="utf-8"))
    if "directory" not in data and data.get("dataset"):
        ds = Path(data["dataset"])
        if not ds.is_absolute():
            ds = (p.parent / ds).resolve()
        data = tomllib.loads(ds.read_text(encoding="utf-8"))
    return [Path(b["path"]) for b in data.get("directory", []) if b.get("path")]


def cache_roots_for(dataset_dirs: list[Path], model_name: str = "anima") -> list[Path]:
    return [Path(d) / "cache" / model_name for d in dataset_dirs]


# =============================================================================
# RATE / ETA  (trailing-window so it stabilizes across bucket boundaries)
# =============================================================================
class _Rate:
    def __init__(self, window_s: float = 90.0) -> None:
        self.window_s = window_s
        self.samples: deque[tuple[float, int]] = deque()

    def update(self, t: float, done: int) -> None:
        self.samples.append((t, done))
        while len(self.samples) > 2 and t - self.samples[0][0] > self.window_s:
            self.samples.popleft()

    def rate(self) -> float:
        if len(self.samples) < 2:
            return 0.0
        (t0, d0), (t1, d1) = self.samples[0], self.samples[-1]
        dt = t1 - t0
        return (d1 - d0) / dt if dt > 0 else 0.0

    def eta(self, remaining: int) -> float:
        r = self.rate()
        return remaining / r if r > 0 and remaining > 0 else float("inf")


def _fmt(seconds: float) -> str:
    if seconds in (float("inf"), float("nan")) or seconds != seconds or seconds < 0:
        return "--:--"
    s = int(seconds)
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


# =============================================================================
# MONITOR  (returns a callable(proc) that polls until the subprocess exits)
# =============================================================================
def make_monitor(*, cache_roots: list[Path], dataset_dirs: list[Path],
                 captions_per_image: int | None = None, interval: float = 30.0,
                 eta_window_s: float = 90.0,
                 on_update: Callable[[dict], None] | None = None,
                 clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep) -> Callable[[object], None]:
    """Build a monitor(proc) that prints `[cache] done/total (xx%) rate ETA` every
    `interval`s until proc.poll() is not None. Never touches/kills the subprocess."""
    total_imgs = count_total_images(dataset_dirs)
    cpi = captions_per_image if captions_per_image is not None else _captions_per_image(dataset_dirs)
    total_lat = total_imgs
    total_txt = total_imgs * cpi
    total = total_lat + total_txt
    rate = _Rate(eta_window_s)
    warmed = 0

    def _tick() -> dict:
        nonlocal warmed
        d = count_done(cache_roots)
        dl, dt = d.get("latents_", 0), d.get("text_embeddings_", 0)
        done = dl + dt
        rate.update(clock(), done)
        r = rate.rate()
        if total <= 0:
            line = f"[cache] {done} cached ({_fmt(0)} elapsed)  (total unknown)"
            info = {"done": done, "total": None, "rate": r}
        else:
            pct = min(99.9, 100.0 * done / total) if done < total else 100.0
            remaining = max(0, total - done)
            note = ""
            if done == 0:
                warmed += 1
                if warmed >= 2:
                    note = "  (warming up — loading VAE/text-encoder)"
            line = (f"[cache] {done}/{total} ({pct:.1f}%)  rate {r:.1f}/s  "
                    f"ETA {_fmt(rate.eta(remaining))}   [latents {dl}/{total_lat} "
                    f"· text {dt}/{total_txt}]{note}")
            info = {"done": done, "total": total, "pct": pct, "rate": r,
                    "latents": (dl, total_lat), "text": (dt, total_txt)}
        print(line, flush=True)
        if on_update:
            try:
                on_update(info)
            except Exception:  # noqa: BLE001 — a bad callback must not kill the run
                pass
        return info

    def monitor(proc) -> None:
        while getattr(proc, "poll", lambda: None)() is None:
            sleep(interval)
            _tick()
        _tick()  # final line

    return monitor
