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

    # Balancing policy (DIMINISHING-RETURNS, anti-overtraining):
    #   repeats(c) = round((top / images(c)) ** (1 - balance_alpha)), capped.
    #   balance_alpha = 0.0 -> legacy full equalization (scale small concepts up to
    #     the largest -> sparse concepts seen up to max_repeats x per epoch = overfit);
    #   balance_alpha = 1.0 -> no balancing (every concept 1x);
    #   balance_alpha = 0.5 -> sqrt damping (default): big concepts ~1x, sparse ones get
    #     a bounded lift so they contribute without being memorized.
    # cap_mult caps a bucket's EFFECTIVE samples at cap_mult * top (no bucket dominates).
    target_effective: int | None = None
    balance_alpha: float = 0.5
    cap_mult: float = 1.25
    max_repeats: int = 8           # per-image exposure ceiling (was 50 under equalization)

    # MIXED caption mode: captions come from per-dir captions.json, not .txt sidecars.
    online_captions: bool = False


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


def dampened_repeats(images: int, top: int, *, alpha: float = 0.5,
                     max_repeats: int = 8, cap_mult: float = 1.25) -> int:
    """Bounded, diminishing-returns num_repeats for one bucket.

    alpha=0 -> equalize (legacy: small buckets scaled up to `top`, which OVERTRAINS
    sparse concepts); alpha=1 -> no balancing (all 1x); 0.5 -> sqrt (default).
    Caps per-image exposure at `max_repeats` and effective samples at `cap_mult*top`.
    """
    if images <= 0:
        return 1
    rep = max(1, round((top / images) ** (1.0 - alpha)))
    rep = min(rep, max_repeats)
    if images * rep > cap_mult * top:          # effective-samples ceiling
        rep = max(1, int(cap_mult * top // images))
    return rep


def compute_repeats(concepts: list[dict], cfg: BaseConfig) -> None:
    """Fill each concept dict with 'repeats' and 'effective' in place, using the
    diminishing-returns policy so sparse concepts contribute without overtraining."""
    if not concepts:
        return
    top = cfg.target_effective or max(c["images"] for c in concepts)
    for c in concepts:
        c["repeats"] = dampened_repeats(c["images"], top, alpha=cfg.balance_alpha,
                                        max_repeats=cfg.max_repeats, cap_mult=cfg.cap_mult)
        # "clamped" = the per-image exposure ceiling actually bit (under-represented).
        raw = max(1, round((top / c["images"]) ** (1.0 - cfg.balance_alpha)))
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
        if cfg.online_captions:                       # MIXED: captions.json, not sidecars
            lines.append("online_captions = true")
        else:
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
                    help="Top reference count. Default = largest concept's image count.")
    ap.add_argument("--alpha", type=float, default=0.5,
                    help="Balance damping: 0=equalize (legacy/overtrains), 1=no balance, 0.5=sqrt.")
    ap.add_argument("--cap-mult", type=float, default=1.25,
                    help="Cap a bucket's effective samples at cap_mult*top.")
    ap.add_argument("--max-repeats", type=int, default=8, help="Per-image exposure ceiling.")
    args = ap.parse_args()

    cfg = BaseConfig(target_effective=args.target, balance_alpha=args.alpha,
                     cap_mult=args.cap_mult, max_repeats=args.max_repeats)
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