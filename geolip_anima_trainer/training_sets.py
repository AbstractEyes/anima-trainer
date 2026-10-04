#!/usr/bin/env python3
"""
training_sets.py — the training sets the stock model draws for the LoRA arms, kept on the Hub so a fresh runtime
downloads them instead of drawing them again (about an hour on Anima for the six sets of the sequence).

A set is what sana_runner.SanaRunner._render_dataset writes to {data_root}/datasets/<flavor>_<seed_base>/:
images/<name>.png + images/<name>.txt (its caption), items.jsonl (name, seed, prompt, caption per image) and
sheet.jpg. On the Hub (a dataset repo, default AbstractPhil/geolip-beatrix-anima-data) it lives at

    sets/<flavor>_<seed_base>-<key>/      + render.json (the settings that drew it)

where <key> = the first 10 hex digits of sha256 over the canonical JSON of {"render": <settings>, "items": <every
item>}: the bed, the model file, the resolution, the sampler settings, the negative prompt, the batch size and every
prompt, seed and caption. Equal keys = the same inputs, so a set is reused only when it would be drawn the same way;
changed settings give a new folder and the old set stays for the experiments that used it.

Self-contained on purpose (no imports from the runners): a notebook still running an older runner can `git pull`
and import this module to upload what it has drawn:

    from geolip_anima_trainer import training_sets as ts
    ts.upload_local_sets(a)              # a = the AnimaRunner from the setup cell
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

DATA_REPO = "AbstractPhil/geolip-beatrix-anima-data"
RETRY_S = (15, 30, 60, 120)
UPLOAD_PATTERNS = ["images/*.png", "images/*.txt", "items.jsonl", "sheet.jpg", "render.json"]
IGNORE_PATTERNS = ["images/cache/*", "*/cache/*"]       # the trainer's own latent cache sits inside images/


def _dur(seconds: float) -> str:
    s = max(0, int(round(seconds)))
    return f"{s // 60}m{s % 60:02d}s" if s >= 60 else f"{s}s"


def render_spec_of(runner) -> dict:
    """The settings that decide a set's pixels, read from a runner (SanaRunner / AnimaRunner, any version that has
    BED, GEN_STEPS / GEN_CFG / GEN_SHIFT, NEGATIVE and cfg.gen_batch)."""
    st = runner.state
    model = st.get("transformer_path") or st.get("diffusers_path") or ""
    return {"bed": runner.BED.key, "model": Path(str(model)).name, "resolution": int(st["resolution"]),
            "steps": int(runner.GEN_STEPS), "guidance": float(runner.GEN_CFG), "shift": float(runner.GEN_SHIFT),
            "negative": str(runner.NEGATIVE), "batch": int(runner.cfg.gen_batch)}


def set_key(render: dict, items: list[dict]) -> str:
    blob = json.dumps({"render": render, "items": items}, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:10]


def hub_folder(name: str, render: dict, items: list[dict]) -> str:
    return f"sets/{name}-{set_key(render, items)}"


def read_items(root: "str | Path") -> list[dict]:
    lines = (Path(root) / "items.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def missing_images(root: "str | Path", items: list[dict]) -> list[str]:
    """The items whose image or caption file is not in root/images yet."""
    img = Path(root) / "images"
    return [it["name"] for it in items
            if not ((img / f"{it['name']}.png").is_file() and (img / f"{it['name']}.txt").is_file())]


def _api(token: "str | None"):
    from huggingface_hub import HfApi
    return HfApi(token=token or None)


def _retry(fn, what: str):
    """Rate limits, server errors and dropped connections are retried; a refused token or a missing repo is not."""
    for wait in (*RETRY_S, None):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            code = getattr(getattr(e, "response", None), "status_code", None)
            if wait is None or code in (401, 403, 404):
                raise
            print(f"[hub] {what} failed ({type(e).__name__}: {str(e)[:120]}); retry in {wait} s", flush=True)
            time.sleep(wait)


def on_hub(token: "str | None", repo_id: str, folder: str) -> bool:
    return bool(_api(token).file_exists(repo_id, f"{folder}/render.json", repo_type="dataset"))


def upload_set(token: "str | None", repo_id: str, root: "str | Path", render: dict,
               items: "list[dict] | None" = None) -> "str | None":
    """Upload one complete local set (one commit) unless the Hub has it already. Returns its Hub folder, or None
    when it was there. Raises on an incomplete set."""
    root = Path(root)
    items = read_items(root) if items is None else items
    gaps = missing_images(root, items)
    if gaps:
        raise ValueError(f"{root.name} is incomplete: {len(gaps)} of {len(items)} images missing")
    folder = hub_folder(root.name, render, items)
    if on_hub(token, repo_id, folder):
        return None
    (root / "render.json").write_text(json.dumps(
        {"render": render, "key": set_key(render, items), "images": len(items),
         "uploaded_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")}, indent=1), encoding="utf-8")
    api = _api(token)
    _retry(lambda: api.upload_folder(repo_id=repo_id, repo_type="dataset", folder_path=str(root), path_in_repo=folder,
                                     allow_patterns=UPLOAD_PATTERNS, ignore_patterns=IGNORE_PATTERNS,
                                     commit_message=f"{folder}: {len(items)} images"), f"upload {folder}")
    return folder


def download_set(token: "str | None", repo_id: str, root: "str | Path", render: dict, items: list[dict]) -> bool:
    """Fetch the set drawn with exactly these settings and items into root. False when the Hub has none, or when
    what it has does not check out (settings, item list, every image and caption present, captions as listed)."""
    from huggingface_hub import snapshot_download
    root = Path(root)
    folder = hub_folder(root.name, render, items)
    if not on_hub(token, repo_id, folder):
        return False
    tmp = root.parent / f".pull_{root.name}"
    shutil.rmtree(tmp, ignore_errors=True)
    try:
        _retry(lambda: snapshot_download(repo_id, repo_type="dataset", allow_patterns=[f"{folder}/*"],
                                         local_dir=str(tmp), token=token or None), f"download {folder}")
        src = tmp / folder
        meta = json.loads((src / "render.json").read_text(encoding="utf-8"))
        bad = (meta.get("render") != render or read_items(src) != items or missing_images(src, items)
               or any((src / "images" / f"{it['name']}.txt").read_text(encoding="utf-8") != it["caption"]
                      for it in items))
        if bad:
            print(f"[sets] {folder} on the Hub does not match this run's settings or items; drawing it instead", flush=True)
            return False
        (root / "images").mkdir(parents=True, exist_ok=True)
        for f in (src / "images").iterdir():
            if f.is_file():
                os.replace(f, root / "images" / f.name)
        for name in ("items.jsonl", "sheet.jpg", "render.json"):
            if (src / name).is_file():
                os.replace(src / name, root / name)
        return True
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def upload_local_sets(runner, repo_id: str = DATA_REPO, *, token: "str | None" = None) -> dict:
    """Every complete set under the runner's datasets/ folder, uploaded unless the Hub has it (a set still being
    drawn is reported and skipped: run this again once it is complete). Returns {set: hub folder | status}."""
    base = Path(runner.state["data_root"]) / "datasets"
    render = render_spec_of(runner)
    token = token or runner.state.get("hf_token")
    sets = sorted(p for p in base.iterdir() if p.is_dir() and (p / "items.jsonl").is_file()) if base.is_dir() else []
    if not sets:
        print(f"[sets] no training sets under {base}", flush=True)
    out: dict = {}
    for d in sets:
        items = read_items(d)
        gaps = missing_images(d, items)
        if gaps:
            print(f"[sets] {d.name}: {len(items) - len(gaps)} of {len(items)} images so far; skipped (still drawing?)",
                  flush=True)
            out[d.name] = "incomplete"
            continue
        t0 = time.time()
        print(f"[sets] {d.name}: uploading {len(items)} images...", flush=True)
        folder = upload_set(token, repo_id, d, render, items)
        if folder is None:
            print(f"[sets] {d.name}: already on the Hub", flush=True)
            out[d.name] = "already there"
        else:
            print(f"[sets] {d.name}: done in {_dur(time.time() - t0)} -> "
                  f"https://huggingface.co/datasets/{repo_id}/tree/main/{folder}", flush=True)
            out[d.name] = folder
    return out
