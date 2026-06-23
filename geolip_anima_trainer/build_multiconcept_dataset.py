#!/usr/bin/env python3
"""
build_multiconcept_dataset.py — generate a balanced Anima multi-concept dataset.toml.

Point it at a parent directory whose immediate subfolders are concepts:

    anima_multi/
        conceptA/   (images + .txt captions)
        conceptB/
        conceptC/

It counts images per concept, computes num_repeats so each concept contributes a
comparable number of effective samples per epoch, checks caption coverage, and
writes a dataset.toml with one [[directory]] block per concept.

Run:
    python build_multiconcept_dataset.py --root /data/datasets/anima_multi \
        --out configs/anima_dataset.toml
"""

import argparse
from dataclasses import dataclass, field
from pathlib import Path


# =============================================================================
# CONFIG
# =============================================================================
@dataclass
class BaseConfig:
    # Image extensions counted as training samples.
    image_exts: tuple = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
    caption_extension: str = ".txt"

    # Dataset bucketing settings written into the toml (match anima_lora.toml res).
    resolutions: list = field(default_factory=lambda: [1024])
    enable_ar_bucket: bool = True
    min_ar: float = 0.5
    max_ar: float = 2.0
    num_ar_buckets: int = 7

    # Anima is tag-order sensitive -> do NOT shuffle captions by default.
    shuffle_caption: bool = False

    # Balancing policy:
    #   repeats(concept) = round(target / image_count), clamped to [1, max_repeats].
    # If target is None, it defaults to the largest concept's image count (so the
    # biggest concept gets 1 repeat and smaller ones are scaled up to match).
    target_effective: int | None = None
    max_repeats: int = 50          # clamp; warns if a concept would exceed this


# =============================================================================
# BODY
# =============================================================================
def count_images(folder: Path, cfg: BaseConfig) -> int:
    """Number of image files directly inside `folder`."""
    return sum(
        1 for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in cfg.image_exts
    )


def count_captions(folder: Path, cfg: BaseConfig) -> int:
    """Number of caption sidecar files directly inside `folder`."""
    return sum(
        1 for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() == cfg.caption_extension
    )


def discover_concepts(root: Path, cfg: BaseConfig) -> list[dict]:
    """Scan immediate subfolders; return per-concept counts, sorted by name."""
    concepts = []
    for sub in sorted(p for p in root.iterdir() if p.is_dir()):
        if sub.name == "cache":          # diffusion-pipe writes a cache/ dir; skip it
            continue
        n_img = count_images(sub, cfg)
        if n_img == 0:
            continue                     # not a concept folder
        concepts.append({
            "name": sub.name,
            "path": str(sub.resolve()),
            "images": n_img,
            "captions": count_captions(sub, cfg),
        })
    return concepts


def compute_repeats(concepts: list[dict], cfg: BaseConfig) -> None:
    """Fill each concept dict with 'repeats' and 'effective' in place."""
    if not concepts:
        return
    target = cfg.target_effective or max(c["images"] for c in concepts)
    for c in concepts:
        raw = max(1, round(target / c["images"]))
        c["repeats"] = min(raw, cfg.max_repeats)
        c["clamped"] = raw > cfg.max_repeats
        c["effective"] = c["images"] * c["repeats"]


def render_toml(concepts: list[dict], cfg: BaseConfig) -> str:
    """Render the full dataset.toml text."""
    lines = [
        "# =============================================================================",
        "# anima_dataset.toml  —  MULTI-CONCEPT (auto-generated, balanced num_repeats)",
        "# Regenerate with build_multiconcept_dataset.py whenever images change.",
        "# =============================================================================",
        f"resolutions = {cfg.resolutions}",
        f"enable_ar_bucket = {'true' if cfg.enable_ar_bucket else 'false'}",
        f"min_ar = {cfg.min_ar}",
        f"max_ar = {cfg.max_ar}",
        f"num_ar_buckets = {cfg.num_ar_buckets}",
        "frame_buckets = [1]",
        f"shuffle_caption = {'true' if cfg.shuffle_caption else 'false'}",
        "",
    ]
    for c in concepts:
        lines.append(
            f"[[directory]]   # {c['name']}: {c['images']} imgs "
            f"x{c['repeats']} = {c['effective']} effective"
        )
        lines.append(f"path = '{c['path']}'")
        lines.append(f"caption_extension = '{cfg.caption_extension}'")
        lines.append(f"num_repeats = {c['repeats']}")
        lines.append("")
    return "\n".join(lines)


# =============================================================================
# RUN
# =============================================================================
def main() -> None:
    ap = argparse.ArgumentParser(description="Build a balanced Anima multi-concept dataset.toml")
    ap.add_argument("--root", required=True, help="Parent dir of concept subfolders.")
    ap.add_argument("--out", default="anima_dataset.toml", help="Output toml path.")
    ap.add_argument("--target", type=int, default=None,
                    help="Target effective samples/concept. Default = largest concept's image count.")
    ap.add_argument("--max-repeats", type=int, default=50, help="Clamp on num_repeats.")
    args = ap.parse_args()

    cfg = BaseConfig(target_effective=args.target, max_repeats=args.max_repeats)
    root = Path(args.root).expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"Not a directory: {root}")

    concepts = discover_concepts(root, cfg)
    if not concepts:
        raise SystemExit(f"No concept subfolders with images found under {root}")

    compute_repeats(concepts, cfg)

    # Report (and flag problems) before writing.
    print(f"Found {len(concepts)} concept(s) under {root}:\n")
    print(f"  {'concept':<24}{'images':>8}{'captions':>10}{'repeats':>9}{'effective':>11}")
    for c in concepts:
        flag = ""
        if c["captions"] != c["images"]:
            flag += "  [!] caption count != image count"
        if c.get("clamped"):
            flag += f"  [!] repeats clamped to {cfg.max_repeats}; concept under-represented"
        print(f"  {c['name']:<24}{c['images']:>8}{c['captions']:>10}"
              f"{c['repeats']:>9}{c['effective']:>11}{flag}")

    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    # encoding="utf-8": the rendered toml contains an em-dash; without this,
    # Windows' default cp1252 raises UnicodeEncodeError.
    out.write_text(render_toml(concepts, cfg), encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()