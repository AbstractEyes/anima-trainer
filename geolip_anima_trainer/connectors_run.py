#!/usr/bin/env python3
"""The connector arms on one card: Beatrix's reading of a mood phrase pushed into Anima's text context (W f + b after the
text adapter, on both guidance branches), trained by Anima's own objective on the mood LoRAs' first-draw images and judged on
the held-out scenes (AnimaRunner.run_beatrix_connectors).

    python -m geolip_anima_trainer.connectors_run --arms e031_...,e032_... --data-root /content/anima --models-dir <dir>

--eval-batch N renders the evaluation (the no-push baseline and every arm's sets) in batches of N. --eval-batch auto first
times one batch each of 8, 16 and 32 throwaway renders of the held-out scenes (after one warm-up batch) and keeps the size
with the least time per image (a size within 5% of the best loses to a smaller one; a size the card cannot hold is
skipped). The training sets are drawn and keyed at the runner's gen_batch either way. --offline writes the experiments
repo's files to <data-root>/hub_mirror (a machine with no write token). The diffusion-pipe fork is found through
ANIMA_DIFFUSION_PIPE.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

SIZES = (8, 16, 32)


def parse(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="the connector arms (Beatrix's phrase readings as pushes) on one card")
    ap.add_argument("--arms", required=True, help="comma-separated experiment ids (anima_experiments.CONNECTOR_IDS)")
    ap.add_argument("--data-root", required=True, help="pictures, results and (with --offline) the repo mirror go here")
    ap.add_argument("--models-dir", required=True, help="a folder holding Anima's three files (searched recursively)")
    ap.add_argument("--repo-root", default=None, help="this repo's checkout (default: the package's parent folder)")
    ap.add_argument("--eval-batch", default="auto", help="the evaluation's render batch: a number, or 'auto' (timed)")
    ap.add_argument("--offline", action="store_true", help="write the experiments repo's files locally (no write token)")
    ap.add_argument("--force", action="store_true", help="rerun arms the repo lists as done")
    return ap.parse_args(argv)


def probe_batches(r, sizes=SIZES) -> "tuple[int, dict]":
    """(the chosen batch, {batch: seconds per image}) from one timed batch per size on the neutral prompt over the held-out
    scenes (seeds 9000+, never scored), after one warm-up batch of the smallest size."""
    import torch
    from .sana_runner import HELD_OUT, SUBJECTS
    keep = r.cfg.gen_batch
    prompts = [r.BED.neutral_caption.format(s=SUBJECTS[si]) for si in HELD_OUT]

    def batch(n, at):
        return [prompts[(at + i) % len(prompts)] for i in range(n)], [9000 + at + i for i in range(n)]
    per: dict = {}
    try:
        r.cfg.gen_batch = sizes[0]
        r._render(*batch(sizes[0], 0))                               # warm-up: the first batch pays the kernels' setup
        for b in sizes:
            r.cfg.gen_batch = b
            torch.cuda.synchronize()
            t0 = time.time()
            try:
                r._render(*batch(b, 100 + b))
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"[connectors] a batch of {b} does not fit on this card: skipped", flush=True)
                continue
            torch.cuda.synchronize()
            per[b] = (time.time() - t0) / b
            print(f"[connectors] the batch probe: {b} images a batch, {per[b]:.2f} s an image", flush=True)
    finally:
        r.cfg.gen_batch = keep
    if not per:
        raise RuntimeError(f"no render batch of {list(sizes)} fits on this card")
    best = min(per.values())
    chosen = min(b for b, s in per.items() if s <= best * 1.05)
    return chosen, per


def main(argv=None) -> int:
    a = parse(argv)
    from .anima_experiments import CONNECTOR_IDS
    from .anima_runner import AnimaRunner
    arms = [x.strip() for x in a.arms.split(",") if x.strip()]
    unknown = sorted(set(arms) - set(CONNECTOR_IDS))
    if unknown:
        raise SystemExit(f"unknown connector arm(s) {unknown}; they are {CONNECTOR_IDS}")
    repo_root = a.repo_root or str(Path(__file__).resolve().parents[1])
    r = AnimaRunner(data_root=a.data_root, models_dir=a.models_dir, repo_root=repo_root, publish=not a.offline)
    r.setup()
    if a.eval_batch == "auto":
        r._eval_pipe()
        chosen, per = probe_batches(r)
        print(f"[connectors] the evaluation renders in batches of {chosen} ("
              + ", ".join(f"{b}: {s:.2f} s an image" for b, s in per.items()) + ")", flush=True)
        r.cfg.eval_batch = chosen
    else:
        r.cfg.eval_batch = int(a.eval_batch)
    out = r.run_beatrix_connectors(arms=arms, force=a.force)
    for k, m in out.items():
        print(f"[connectors] {k}: {(m or {}).get('status')} | {(m or {}).get('summary', '')}", flush=True)
    return 0 if all((m or {}).get("status") == "done" for m in out.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
