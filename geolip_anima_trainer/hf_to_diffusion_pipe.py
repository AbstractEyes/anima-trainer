#!/usr/bin/env python3
"""
hf_to_diffusion_pipe.py — bridge an HF `datasets`-format (parquet) repo into the
directory + .txt-sidecar layout diffusion-pipe requires, by STREAMING.

Why this exists: diffusion-pipe only ingests directories of image files with
matching .txt captions. It does NOT read parquet/arrow/HF-datasets as input (it
uses the `datasets` library internally only as a latent/embedding CACHE backend).
So we stream the parquet, filter to the approved subset, render captions, and
write files into per-concept folders. Treat the written files as TRANSIENT:
    1. run this exporter           -> scratch dir of img + .txt (+ optional masks)
    2. diffusion-pipe --cache_only -> builds latent/text-embed cache
    3. delete the scratch dir      -> keep only the cache + the archival parquet
Steady-state footprint is then just the cache (which any trainer needs) plus the
parquet you already keep. No permanent free-floating-file waste.

Built for AbstractPhil/diffusion-pretrain-set-ft1 (17-col schema), but configurable.
Schema columns used: image, mask, conditioning_image, source, audit,
age_classifier_pass, and one of the JSON caption columns:
    caption_animetimm_json  (booru-style structured -> best for Anima)
    caption_vlm_json        (structured natural language)
    captions_source_json    (original upstream caption, single string value)

Run:
    pip install datasets
    python hf_to_diffusion_pipe.py \
        --repo AbstractPhil/diffusion-pretrain-set-ft1 \
        --configs deepfashion synth_chars \
        --out /data/datasets/anima_multi \
        --caption-column caption_animetimm_json --caption-format animetimm \
        --route-by source --limit-per-concept 3000
Then run build_multiconcept_dataset.py on --out, then diffusion-pipe --cache_only.
"""

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path


# =============================================================================
# CONFIG
# =============================================================================
@dataclass
class BridgeConfig:
    repo: str = "AbstractPhil/diffusion-pretrain-set-ft1"
    configs: list = field(default_factory=lambda: ["full"])   # which HF configs to pull
    split: str = "train"
    out_root: str = "./anima_multi"

    # Column names (override if your schema differs).
    image_column: str = "image"
    mask_column: str = "mask"
    conditioning_column: str = "conditioning_image"
    route_column: str = "source"          # per-concept routing -> subfolder name
    caption_column: str = "caption_animetimm_json"
    caption_format: str = "animetimm"     # animetimm | vlm | source | raw

    # Quality / safety gates (default ON — your dataset carries these flags).
    require_audit_approved: bool = True
    audit_column: str = "audit"
    audit_pass_value: str = "approved"
    require_age_pass: bool = True
    age_pass_column: str = "age_classifier_pass"

    # Output options.
    image_ext: str = ".png"               # .png lossless re-encode; .jpg smaller
    with_mask: bool = False               # also export masks for masked training
    with_conditioning: bool = False       # also export conditioning images
    limit_per_concept: int | None = None  # cap rows per concept (keeps extraction small)
    limit_total: int | None = None        # global cap: stop after N images written (early-exit)
    streaming: bool = True                # stream, do NOT materialize the whole repo


# Backwards-compatible alias: earlier scripts/docs referred to BaseConfig.
BaseConfig = BridgeConfig


# =============================================================================
# CAPTION RENDERERS
# =============================================================================
def _render_structured_tags(obj: dict) -> str:
    """Render a {subjects:[{name,attributes}], actions:[], setting} dict to a
    Danbooru-style comma tag string: 'attr name, attr name, action, setting'."""
    tags: list[str] = []
    for subj in obj.get("subjects", []) or []:
        name = (subj.get("name") or "").strip()
        attrs = [a.strip() for a in (subj.get("attributes") or []) if a and a.strip()]
        if not name:
            continue
        if attrs:
            # one tag per attribute paired with the noun: "blue jeans", "long hair"
            tags.extend(f"{a} {name}" for a in attrs)
        else:
            tags.append(name)
    tags.extend(a.strip() for a in (obj.get("actions") or []) if a and a.strip())
    setting = (obj.get("setting") or "").strip()
    if setting and setting.lower() != "unknown":
        tags.append(setting)
    # de-dupe while preserving order
    seen, out = set(), []
    for t in tags:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return ", ".join(out)


def render_caption(raw: str, fmt: str) -> str:
    """Turn a raw cell value into a caption string per the chosen format."""
    if raw is None:
        return ""
    if fmt == "raw":
        return str(raw).strip()
    try:
        obj = json.loads(raw) if isinstance(raw, str) else raw
    except (json.JSONDecodeError, TypeError):
        return str(raw).strip()

    if fmt in ("animetimm", "vlm"):
        # both share the subjects/actions/setting structured shape
        return _render_structured_tags(obj) if isinstance(obj, dict) else str(obj).strip()
    if fmt == "source":
        # {"<source>_caption": "a woman wearing..."} -> take the first string value
        if isinstance(obj, dict):
            for v in obj.values():
                if isinstance(v, str) and v.strip():
                    return v.strip()
        return str(obj).strip()
    return str(obj).strip()


# =============================================================================
# BODY
# =============================================================================
_SLUG = re.compile(r"[^a-zA-Z0-9_-]+")


def slug(s: str) -> str:
    return _SLUG.sub("_", str(s).strip()).strip("_") or "unknown"


def row_passes(row: dict, cfg: BaseConfig) -> bool:
    if cfg.require_audit_approved:
        if str(row.get(cfg.audit_column, "")).strip().lower() != cfg.audit_pass_value:
            return False
    if cfg.require_age_pass:
        if not bool(row.get(cfg.age_pass_column, False)):
            return False
    return True


def _to_pil(value):
    """Coerce a streamed image cell into a PIL.Image.

    Depending on whether the column is cast as a datasets Image() feature, a
    streamed cell may already be a PIL.Image, OR a raw {'bytes': ..., 'path': ...}
    struct (as in AbstractPhil/diffusion-pretrain-set-ft1, where `image` is a dict),
    OR a path string. Handle all three. Returns None if it can't be decoded.
    """
    import io
    from PIL import Image

    if value is None:
        return None
    if hasattr(value, "convert"):          # already a PIL.Image
        return value
    if isinstance(value, dict):
        if value.get("bytes"):
            return Image.open(io.BytesIO(value["bytes"]))
        if value.get("path"):
            return Image.open(value["path"])
        return None
    if isinstance(value, (str, bytes)):
        return Image.open(io.BytesIO(value) if isinstance(value, bytes) else value)
    return None


def export_config(ds_config: str, cfg: BaseConfig, counts: dict) -> None:
    """Stream one HF config and write passing rows into per-concept folders."""
    from datasets import load_dataset  # imported here so --help works without datasets

    ds = load_dataset(cfg.repo, name=ds_config, split=cfg.split, streaming=cfg.streaming)
    for row in ds:
        # Global early-exit: without this the loop scans the ENTIRE config (which can
        # be ~100GB) even when per-concept limits are set. Bounds smoke tests + caps cost.
        if cfg.limit_total is not None and sum(counts.values()) >= cfg.limit_total:
            break
        if not row_passes(row, cfg):
            continue

        concept = slug(row.get(cfg.route_column) or ds_config)
        if cfg.limit_per_concept and counts.get(concept, 0) >= cfg.limit_per_concept:
            continue

        img = _to_pil(row.get(cfg.image_column))
        if img is None:
            continue

        base = slug(row.get("id") or f"{concept}_{counts.get(concept, 0):07d}")
        cdir = Path(cfg.out_root) / concept
        cdir.mkdir(parents=True, exist_ok=True)

        # image (coerced from PIL.Image / {bytes,path} dict / path by _to_pil)
        img_path = cdir / f"{base}{cfg.image_ext}"
        img.convert("RGB").save(img_path)

        # caption sidecar (.txt, same basename)
        caption = render_caption(row.get(cfg.caption_column), cfg.caption_format)
        (cdir / f"{base}.txt").write_text(caption, encoding="utf-8")

        # optional mask -> <concept>/masks/<base>.png (diffusion-pipe masked training)
        if cfg.with_mask:
            mask = _to_pil(row.get(cfg.mask_column))
            if mask is not None:
                mdir = cdir / "masks"
                mdir.mkdir(exist_ok=True)
                mask.convert("L").save(mdir / f"{base}.png")

        if cfg.with_conditioning:
            cond = _to_pil(row.get(cfg.conditioning_column))
            if cond is not None:
                ddir = cdir / "conditioning"
                ddir.mkdir(exist_ok=True)
                cond.convert("RGB").save(ddir / f"{base}.png")

        counts[concept] = counts.get(concept, 0) + 1


# Caption columns probed by inspect(), mapped to their render format.
CAPTION_COLUMNS = {
    "caption_animetimm_json": "animetimm",
    "caption_vlm_json": "vlm",
    "captions_source_json": "source",
}


def inspect(cfg: BridgeConfig, n: int = 200, *, verbose: bool = True) -> dict:
    """Probe one HF config WITHOUT writing: sample n rows and report columns,
    caption fill rates, and the audit/age gate distributions. Returns a dict
    (also printed when verbose). This is the single source of truth shared by
    the CLI --inspect path and api.inspect_source.
    """
    from datasets import load_dataset  # imported here so --help works without datasets

    ds = load_dataset(cfg.repo, name=cfg.configs[0], split=cfg.split, streaming=True)
    rows, first = [], None
    for i, row in enumerate(ds):
        if first is None:
            first = row
        rows.append(row)
        if i + 1 >= n:
            break
    first = first or {}

    columns = {k: type(v).__name__ for k, v in first.items()}

    caption_fill: dict = {}
    for col, fmt in CAPTION_COLUMNS.items():
        if col in first:
            filled = sum(1 for r in rows if render_caption(r.get(col), fmt).strip())
            example = next((render_caption(r.get(col), fmt) for r in rows
                            if render_caption(r.get(col), fmt).strip()), "")
            caption_fill[col] = {"filled": filled, "total": len(rows),
                                 "example": example[:70]}

    audit_values: dict = {}
    if cfg.audit_column in first:
        for r in rows:
            key = str(r.get(cfg.audit_column))
            audit_values[key] = audit_values.get(key, 0) + 1

    age_pass = None
    if cfg.age_pass_column in first:
        passes = sum(1 for r in rows if bool(r.get(cfg.age_pass_column)))
        age_pass = {"passed": passes, "total": len(rows)}

    result = {
        "repo": cfg.repo, "config": cfg.configs[0], "sampled": len(rows),
        "columns": columns, "caption_fill": caption_fill,
        "audit_values": audit_values, "age_pass": age_pass,
    }

    if verbose:
        print(f"\n{cfg.repo} :: {cfg.configs[0]}  (sampled {len(rows)} rows)\n")
        print("Columns / types (row 0):")
        for k, t in columns.items():
            print(f"  {k:<28}{t}")
        print("\nCaption fill rate (non-empty after rendering):")
        for col, info in caption_fill.items():
            print(f"  {col:<28}{info['filled']}/{info['total']}   e.g. {info['example']}")
        if audit_values:
            print(f"\naudit values: {audit_values}")
        if age_pass:
            warn = ("   [!] default age gate would drop most rows"
                    if age_pass["passed"] < age_pass["total"] // 2 else "")
            print(f"age_classifier_pass True: {age_pass['passed']}/{age_pass['total']}{warn}")

    return result


# =============================================================================
# RUN
# =============================================================================
def main() -> None:
    ap = argparse.ArgumentParser(description="Stream an HF parquet repo into diffusion-pipe dirs.")
    ap.add_argument("--repo", default=BaseConfig.repo)
    ap.add_argument("--configs", nargs="+", default=["full"], help="HF config names to export.")
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", dest="out_root", default="./anima_multi")
    ap.add_argument("--route-by", dest="route_column", default="source")
    ap.add_argument("--caption-column", default="caption_animetimm_json")
    ap.add_argument("--caption-format", default="animetimm",
                    choices=["animetimm", "vlm", "source", "raw"])
    ap.add_argument("--image-ext", default=".png", choices=[".png", ".jpg"])
    ap.add_argument("--with-mask", action="store_true")
    ap.add_argument("--with-conditioning", action="store_true")
    ap.add_argument("--limit-per-concept", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None,
                    help="Global cap: stop after N images written (early-exit).")
    ap.add_argument("--no-audit-filter", action="store_true", help="Disable audit==approved gate.")
    ap.add_argument("--no-age-filter", action="store_true", help="Disable age_classifier_pass gate.")
    ap.add_argument("--inspect", action="store_true",
                    help="Probe columns, caption fill rates, and audit/age gates; exit (no writing).")
    ap.add_argument("--inspect-n", type=int, default=200,
                    help="How many rows to sample during --inspect.")
    args = ap.parse_args()

    cfg = BridgeConfig(
        repo=args.repo, configs=args.configs, split=args.split, out_root=args.out_root,
        route_column=args.route_column, caption_column=args.caption_column,
        caption_format=args.caption_format, image_ext=args.image_ext,
        with_mask=args.with_mask, with_conditioning=args.with_conditioning,
        limit_per_concept=args.limit_per_concept, limit_total=args.limit,
        require_audit_approved=not args.no_audit_filter,
        require_age_pass=not args.no_age_filter,
    )

    if args.inspect:
        inspect(cfg, args.inspect_n)
        return

    counts: dict = {}
    for ds_config in cfg.configs:
        print(f"Exporting config '{ds_config}' ...")
        export_config(ds_config, cfg, counts)

    print("\nPer-concept counts written:")
    for concept in sorted(counts):
        print(f"  {concept:<28}{counts[concept]}")
    print(f"\nOutput root: {Path(cfg.out_root).resolve()}")
    print("Next: build_multiconcept_dataset.py --root <out>  then  diffusion-pipe --cache_only")


if __name__ == "__main__":
    main()