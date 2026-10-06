#!/usr/bin/env python3
"""e029 on several cards under DeepSpeed's launcher, one process per card.

The Anima trainer runs multi-card through DeepSpeed (diffusion-pipe's train.py under `deepspeed --num_gpus=N`); this
script uses the same launcher for e029's pictures. Every card loads Anima once and renders its share of the phrases (all
sets of one phrase on one card, batched as on a single card); card 0 then scores every picture with the judge and writes
the result. The cards only meet at barriers, on the gloo backend: no tensor crosses between cards, so no NCCL path is
involved.

    deepspeed --num_gpus=2 relay_dp.py --stage A --data-root /root/diff/anima --models-dir /root/diff/stitch/models --offline
    deepspeed --num_gpus=2 relay_dp.py --stage B --export <dir>/e029_export_record.safetensors \\
        --data-root /root/diff/anima --models-dir /root/diff/stitch/models --offline

Without the launcher it is the one-card run. --offline writes the experiments repo's files to <data-root>/hub_mirror (for a
machine with no write token); AnimaRunner(...).publish_local() uploads them later from a machine that has one. The
diffusion-pipe fork is found through ANIMA_DIFFUSION_PIPE.
"""
from __future__ import annotations

import argparse
import os
import sys
import uuid
from datetime import timedelta
from pathlib import Path


def one_card(env) -> "str | None":
    """The card this process keeps: the launcher's CUDA_VISIBLE_DEVICES entry at LOCAL_RANK, so the process sees its own card
    as cuda:0 whatever device strings the model code uses. None without a launcher (a one-card run)."""
    local = env.get("LOCAL_RANK")
    if local is None:
        return None
    ids = [x.strip() for x in (env.get("CUDA_VISIBLE_DEVICES") or "").split(",") if x.strip()]
    if not ids:
        return str(int(local))                    # every card visible: the local rank is the card's index
    if int(local) >= len(ids):
        raise RuntimeError(f"LOCAL_RANK {local} has no card in CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES')!r}")
    return ids[int(local)]


def parse(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="e029 (a relayed phrase reading in pictures) on one or several cards")
    ap.add_argument("--stage", choices=("A", "B"), required=True)
    ap.add_argument("--export", default=None, help="stage B: the export file (e029_export_record*.safetensors)")
    ap.add_argument("--data-root", required=True, help="pictures, results and (with --offline) the repo mirror go here")
    ap.add_argument("--models-dir", required=True, help="a folder holding Anima's three files (searched recursively)")
    ap.add_argument("--repo-root", default=None, help="this repo's checkout (default: the package's parent folder)")
    ap.add_argument("--offline", action="store_true", help="write the experiments repo's files locally (no write token)")
    ap.add_argument("--force", action="store_true", help="rerun a stage already done")
    ap.add_argument("--barrier-hours", type=float, default=6.0, help="how long a card waits for the others at a barrier")
    ap.add_argument("--local_rank", type=int, default=None, help=argparse.SUPPRESS)   # the launcher's own flag
    return ap.parse_args(argv)


def main(argv=None) -> int:
    a = parse(argv)
    rank, world = int(os.environ.get("RANK", "0")), int(os.environ.get("WORLD_SIZE", "1"))
    card = one_card(os.environ)
    if world > 1 and card is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = card   # before the first CUDA call
    barrier, token, dist = None, "", None
    if world > 1:
        import deepspeed
        import torch.distributed as dist
        deepspeed.init_distributed(dist_backend="gloo", timeout=timedelta(hours=a.barrier_hours))
        box = [uuid.uuid4().hex if rank == 0 else None]
        dist.broadcast_object_list(box, src=0)        # one token per launch: card 0 never reads another launch's markers
        barrier, token = dist.barrier, box[0]
        print(f"[relay_dp] rank {rank} of {world} on card {card} (gloo barriers, launch {token[:8]})", flush=True)
    from geolip_anima_trainer.anima_runner import AnimaRunner
    repo_root = a.repo_root or str(Path(__file__).resolve().parents[1])
    r = AnimaRunner(data_root=a.data_root, models_dir=a.models_dir, repo_root=repo_root, publish=not a.offline)
    r.set_data_parallel(rank, world, barrier, token)
    r.setup()
    meta = r.run_relay(stage=a.stage, export_path=a.export, force=a.force)
    if dist is not None:
        dist.barrier()                                # leave together: card 0 has written the result
        dist.destroy_process_group()
    if rank == 0:
        print(f"[relay_dp] stage {a.stage} on {world} card{'s' if world != 1 else ''}: {meta.get('status')} | "
              f"{meta.get('summary', '')}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
