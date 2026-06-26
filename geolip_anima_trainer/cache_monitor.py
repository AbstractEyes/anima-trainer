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

import contextlib
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
    """COUNT(*) of the `items` table in one Cache metadata.db. Read-only (NOT immutable —
    the file is actively written, immutable would pin a stale snapshot). NOTE: diffusion-pipe
    commits items only when a shard finalizes (cache.py:106, shards are 10 GB) or at the end
    of a pass — so this jumps in big steps and is 0 until the first commit. Use cache_bytes()
    for live progress; this is the exact (coarse) record count once commits land."""
    try:
        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=0.25)
        try:
            return int(con.execute("SELECT COUNT(*) FROM items").fetchone()[0])
        finally:
            con.close()
    except sqlite3.Error:
        return 0


def cache_bytes(cache_roots: list[Path]) -> int:
    """Total bytes across all shard_*.bin files — the CONTINUOUS progress signal.
    cache.py writes each encoded item to the shard .bin immediately (line 115), so this
    grows in real time even while the metadata.db items table stays uncommitted."""
    total = 0
    for root in cache_roots:
        for binf in Path(root).glob("**/*.bin"):
            try:
                total += binf.stat().st_size
            except OSError:
                pass
    return total


def count_done(cache_roots: list[Path],
               kinds: tuple[str, ...] = ("latents", "text_embeddings")) -> dict[str, int]:
    """Sum item rows across every Cache metadata.db, bucketed by kind. The leaf cache dir is
    diffusion-pipe's `cache_file_prefix.strip('_')` (dataset.py:89), i.e. **`latents`** and
    **`text_embeddings_<i>`** — NO trailing underscore. Matching the old `latents_`/
    `text_embeddings_` prefixes left `latents` (which has no trailing `_`) unmatched, so the
    latents count was stuck at 0 forever while text (`text_embeddings_1`) still matched. Match
    the stripped names: 'latents' and 'text_embeddings' (covers text_embeddings_1/_2/...)."""
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


def last_log_line(log_path: str | Path | None, maxbytes: int = 8192) -> str:
    """The last non-empty line of diffusion-pipe's log — so the warm-up phase shows what
    the trainer is ACTUALLY doing ('Grouping examples: 45%', 'caching latents: ...').
    Splits on BOTH '\\n' and '\\r' because tqdm overwrites its bar in place with carriage
    returns (which never become newlines when piped to a file). Best-effort; '' on any error."""
    if not log_path:
        return ""
    try:
        p = Path(log_path)
        size = p.stat().st_size
        with p.open("rb") as f:
            if size > maxbytes:
                f.seek(size - maxbytes)
            tail = f.read()
        text = tail.decode("utf-8", errors="replace").replace("\r", "\n")
        for ln in reversed(text.splitlines()):
            if ln.strip():
                return ln.strip()
    except OSError:
        pass
    return ""


def log_tail_block(log_path: str | Path | None, maxbytes: int = 4096) -> str:
    """The raw tail of the log (for surfacing a crash traceback). '' on any error."""
    if not log_path:
        return ""
    try:
        p = Path(log_path)
        size = p.stat().st_size
        with p.open("rb") as f:
            if size > maxbytes:
                f.seek(size - maxbytes)
            return f.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def log_has_traceback(log_path: str | Path | None) -> bool:
    """True if the log tail shows a Python traceback — diffusion-pipe's cache work runs in a
    forked child whose crash never signals the parent's queue, so the run HANGS instead of
    exiting. We treat a traceback + stalled bytes as fatal and terminate (see make_monitor)."""
    tail = log_tail_block(log_path, maxbytes=8192)
    return "Traceback (most recent call last)" in tail


# =============================================================================
# MONITOR  (returns a callable(proc) that polls until the subprocess exits)
# =============================================================================
def make_monitor(*, cache_roots: list[Path], dataset_dirs: list[Path],
                 captions_per_image: int | None = None, interval: float = 30.0,
                 eta_window_s: float = 90.0, log_path: str | Path | None = None,
                 stall_limit: int = 2,
                 on_update: Callable[[dict], None] | None = None,
                 clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep) -> Callable[[object], None]:
    """Build a monitor(proc) printing a live cache line every `interval`s until the
    subprocess exits. PRIMARY signal = shard .bin BYTES (continuous), because the metadata
    `items` table only commits per-10 GB shard / per-pass, so the record COUNT is 0 for
    long stretches even while caching is busy. Records are shown as a coarse phase marker
    (latents commit at the end of the latents pass, text at the end of the text pass)."""
    total_imgs = count_total_images(dataset_dirs)
    cpi = captions_per_image if captions_per_image is not None else _captions_per_image(dataset_dirs)
    total_lat, total_txt = total_imgs, total_imgs * cpi
    byte_rate = _Rate(eta_window_s)
    t0: list = [None]

    def _tick() -> dict:
        now = clock()
        if t0[0] is None:
            t0[0] = now
        elapsed = now - t0[0]
        nbytes = cache_bytes(cache_roots)
        byte_rate.update(now, nbytes)
        mbps = byte_rate.rate() / 1e6
        d = count_done(cache_roots)
        dl, dt = d.get("latents", 0), d.get("text_embeddings", 0)
        if nbytes == 0 and dl == 0 and dt == 0:
            tail = last_log_line(log_path)
            phase = tail if tail else "loading VAE + Qwen-3 0.6B and building the dataset"
            line = (f"[cache] warming up · {_fmt(elapsed)} elapsed (no shards yet) — {phase}")
            info = {"bytes": 0, "records": 0, "elapsed": elapsed, "log_tail": tail}
        else:
            recs = (f" · latents {dl}/{total_lat} · text {dt}/{total_txt}"
                    if total_imgs else f" · {dl + dt} records")
            line = (f"[cache] {nbytes / 1e9:.2f} GB cached · {mbps:.0f} MB/s{recs} · "
                    f"{_fmt(elapsed)} elapsed")
            info = {"bytes": nbytes, "mbps": mbps, "latents": (dl, total_lat),
                    "text": (dt, total_txt), "elapsed": elapsed}
        print(line, flush=True)
        if on_update:
            try:
                on_update(info)
            except Exception:  # noqa: BLE001 — a bad callback must not kill the run
                pass
        return info

    def _terminate(proc) -> None:
        for meth in ("terminate", "kill"):
            fn = getattr(proc, meth, None)
            if callable(fn):
                try:
                    fn()
                except Exception:  # noqa: BLE001 — best-effort teardown
                    pass

    def monitor(proc) -> None:
        prev_bytes, stalls = -1, 0
        while getattr(proc, "poll", lambda: None)() is None:
            sleep(interval)
            info = _tick()
            nb = info.get("bytes", 0)
            # A diffusion-pipe cache crash happens in a forked child that never signals the
            # parent's queue, so the process HANGS instead of exiting (poll() stays None). Detect
            # it: a traceback in the log + no byte progress for `stall_limit` ticks -> terminate.
            if stall_limit and log_has_traceback(log_path) and nb == prev_bytes:
                stalls += 1
            else:
                stalls = 0
            prev_bytes = nb
            if stall_limit and stalls >= stall_limit:
                print("[cache] FATAL: a cache worker crashed and the run is wedged (a forked "
                      "child died without exiting the parent) — terminating. Traceback:",
                      flush=True)
                tb = log_tail_block(log_path)
                if tb:
                    print(tb, flush=True)
                _terminate(proc)
                break
        _tick()  # final line

    return monitor


# =============================================================================
# KEEPALIVE  (hold a cloud GPU pod alive AFTER the job finishes)
# =============================================================================
def _gpu_touch() -> bool:
    """A trivial CUDA op so GPU utilization is non-zero this tick. Returns True if it ran
    (keep doing it), False if torch/CUDA is unavailable or errored (stop trying — the CPU
    heartbeat + busy kernel still hold the pod). Best-effort; never raises."""
    try:
        import torch
        if not torch.cuda.is_available():
            return False
        x = torch.ones(1, device="cuda")
        x.add_(1.0)
        torch.cuda.synchronize()
        del x
        return True
    except Exception:  # noqa: BLE001 — torch missing / driver hiccup -> fall back to CPU-only
        return False


def _safe_print(msg: str) -> None:
    """print() that can NEVER drop out of the keepalive loop. The heartbeat is the ONE I/O the
    loop does each tick, and on a cloud pod whose notebook websocket has dropped — exactly when
    you most need the pod HELD — print can raise BrokenPipeError / OSError / ValueError('closed
    file'). Swallow those (keep holding the pod silently); only KeyboardInterrupt propagates so
    the ■ stop button still breaks the loop."""
    try:
        print(msg, flush=True)
    except KeyboardInterrupt:
        raise
    except Exception:  # noqa: BLE001 — stdout gone -> keep holding the pod, just silently
        pass


def keepalive(*, interval: float = 60.0, deadline_s: float | None = None, gpu: bool = True,
              stop_file: str | Path | None = None, stop_event=None,
              on_tick: Callable[[dict], None] | None = None,
              label: str = "holding pod", quiet: bool = False,
              clock: Callable[[], float] = time.monotonic,
              sleep: Callable[[float], None] = time.sleep) -> dict:
    """Hold a cloud GPU pod alive after a long job (caching, a push, or while you step away
    before training). The monitor only loops WHILE the cache subprocess runs and exits the
    instant it finishes — so the GPU goes idle and the pod gets reclaimed right after an 8 h
    cache, before training starts. This is the bridge: run it at the end of the cache cell.

    Cloud pods reclaim an IDLE pod. Verified mechanics: RunPod GPU Pods / Lambda / generic Jupyter
    pods key on **GPU utilization** and/or **kernel-session activity** — the per-tick CUDA op
    (best-effort; skipped if torch/CUDA absent) and this busy loop defeat both. (stdout is NOT a
    documented idle signal anywhere — the heartbeat is for human legibility, not the mechanism.)
    It does NOT help consumer **Google Colab** (idle = browser-UI interaction only, + a 12/24 h hard
    cap) or **vast.ai interruptible** (bid preemption) — both out of any in-kernel loop's reach.

    Stops on **KeyboardInterrupt** (the ■ stop button — press it to proceed to training), when
    `deadline_s` elapses, when `stop_file` appears, or when a `threading.Event` `stop_event` is set
    (the background-thread path used by gpu_keepalive). `quiet` suppresses the per-tick heartbeat.
    Returns {ticks, elapsed, stopped_by}; **never raises** — every I/O is guarded so a dead stdout
    can't drop the pod."""
    t0 = clock()
    ticks = 0
    stopped = "interrupt"
    gpu_ok = bool(gpu)
    sf = Path(stop_file) if stop_file else None
    try:
        while True:
            el = clock() - t0
            if deadline_s is not None and el >= deadline_s:
                stopped = "deadline"
                break
            if sf is not None and sf.exists():
                stopped = "stop_file"
                break
            if stop_event is not None and stop_event.is_set():
                stopped = "stop_event"
                break
            if gpu_ok:
                gpu_ok = _gpu_touch()
            if not quiet:
                # ASCII-only + guarded: the loop's job is to NOT die, so the heartbeat must never
                # raise (UnicodeEncodeError on a non-UTF-8 stdout, or OSError on a dead socket).
                _safe_print(f"[keepalive] {label} | {_fmt(el)} elapsed | tick {ticks} | "
                            f"{'gpu-warm' if gpu_ok else 'cpu-only'} -- interrupt (■) to proceed")
            if on_tick:
                try:
                    on_tick({"ticks": ticks, "elapsed": el, "gpu": gpu_ok})
                except Exception:  # noqa: BLE001 — a bad callback must not stop the keepalive
                    pass
            ticks += 1
            sleep(interval)
    except KeyboardInterrupt:
        stopped = "interrupt"
    try:
        el = clock() - t0
        if not quiet:
            print(f"[keepalive] released | {_fmt(el)} | {ticks} ticks | {stopped}", flush=True)
    except BaseException:  # noqa: BLE001 — releasing anyway; the exit line must never propagate
        el = locals().get("el", 0.0)
    return {"ticks": ticks, "elapsed": el, "stopped_by": stopped}


@contextlib.contextmanager
def gpu_keepalive(*, interval: float = 20.0, gpu: bool = True):
    """Keep the GPU warm on a background daemon thread for the duration of a CPU/network-bound block
    (an HF push, a deepspeed relaunch) so a GPU-utilization idle-reclaimer can't kill the pod while
    no CUDA work is running. The monitor stops looping the instant the cache subprocess exits, so the
    per-phase final push + the inter-phase model reload would otherwise run GPU-idle — wrap them in
    this. Quiet (no per-tick heartbeat — the wrapped op has its own output); the thread never raises
    into the caller; stops on block exit."""
    import threading
    stop = threading.Event()

    def _run():
        try:
            keepalive(interval=interval, gpu=gpu, stop_event=stop, quiet=True)
        except BaseException:  # noqa: BLE001 — a daemon guard must never surface an error
            pass

    t = threading.Thread(target=_run, name="anima-gpu-keepalive", daemon=True)
    _safe_print("[keepalive] GPU-warm guard ON (background) -- holding the pod during the push/handoff")
    t.start()
    try:
        yield
    finally:
        stop.set()
        t.join(timeout=interval + 5.0)
        _safe_print("[keepalive] GPU-warm guard OFF")
