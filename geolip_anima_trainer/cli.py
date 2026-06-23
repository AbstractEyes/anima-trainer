#!/usr/bin/env python3
"""
cli.py — the `anima` console command (argparse, stdlib only).

Subcommands map 1:1 onto api.py:
    anima doctor      env diagnostics (read-only)
    anima download    fetch the 3 Anima model files
    anima inspect     probe an HF config before extracting
    anima export      stream parquet -> img + .txt dirs
    anima build       balanced dataset.toml from concept folders
    anima init-config copy the packaged toml templates into ./configs
    anima validate    load + validate a lora.toml (and its dataset.toml)
    anima sweep       emit resolved configs over a rank x lr grid
    anima cache       diffusion-pipe --cache_only   (Linux/target)
    anima train       diffusion-pipe deepspeed launch (Linux/target; --dry-run anywhere)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from . import api


def _parse_ids(s: str | None) -> list[int] | None:
    if not s:
        return None
    return [int(x) for x in s.replace(" ", "").split(",") if x != ""]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="anima",
        description="CircleStone Anima finetune bridge + diffusion-pipe orchestration.")
    p.add_argument("-v", "--verbose", action="store_true", help="DEBUG logging.")
    sub = p.add_subparsers(dest="cmd", required=True)

    # doctor
    d = sub.add_parser("doctor", help="environment diagnostics")
    d.add_argument("--repo-root", default=None)
    d.add_argument("--config", default=None, help="optional lora.toml to sanity-check")

    # download
    g = sub.add_parser("download", help="download the 3 Anima model files")
    g.add_argument("--dest", default="models/anima")
    g.add_argument("--base", default="base-v1.0", choices=list(api._dl.BASE_CHOICES))

    # inspect
    i = sub.add_parser("inspect", help="probe an HF config (read-only)")
    i.add_argument("--repo", default=api.ExportConfig.repo)
    i.add_argument("--config", required=True, help="HF config name to probe")
    i.add_argument("--split", default="train")
    i.add_argument("-n", "--inspect-n", type=int, default=200)

    # export
    e = sub.add_parser("export", help="stream parquet into diffusion-pipe dirs")
    e.add_argument("--repo", default=api.ExportConfig.repo)
    e.add_argument("--configs", nargs="+", required=True)
    e.add_argument("--out", dest="out_root", required=True)
    e.add_argument("--route-by", dest="route_column", default="source")
    e.add_argument("--caption-column", default="caption_animetimm_json")
    e.add_argument("--caption-format", default="animetimm",
                   choices=["animetimm", "vlm", "source", "raw"])
    e.add_argument("--image-ext", default=".png", choices=[".png", ".jpg"])
    e.add_argument("--with-mask", action="store_true")
    e.add_argument("--with-conditioning", action="store_true")
    e.add_argument("--limit-per-concept", type=int, default=None)
    e.add_argument("--limit", type=int, default=None,
                   help="global cap: stop after N images written (early-exit; great for smoke tests)")
    e.add_argument("--no-audit-filter", action="store_true")
    e.add_argument("--no-age-filter", action="store_true")

    # build
    b = sub.add_parser("build", help="balanced dataset.toml from concept folders")
    b.add_argument("--root", required=True)
    b.add_argument("--out", default="configs/anima_dataset.toml")
    b.add_argument("--target", type=int, default=None)
    b.add_argument("--max-repeats", type=int, default=50)

    # init-config
    ic = sub.add_parser("init-config", help="copy packaged toml templates into ./configs")
    ic.add_argument("--out", default="configs")

    # validate
    va = sub.add_parser("validate", help="load + validate a lora.toml")
    va.add_argument("--config", required=True)
    va.add_argument("--dataset", default=None)

    # sweep
    sw = sub.add_parser("sweep", help="emit resolved configs over a rank x lr grid")
    sw.add_argument("--config", required=True, help="base lora.toml")
    sw.add_argument("--dataset", default=None)
    sw.add_argument("--ranks", type=int, nargs="+", required=True)
    sw.add_argument("--lrs", type=float, nargs="+", required=True)
    sw.add_argument("--runs-root", default="runs")
    sw.add_argument("--configs-root", default="configs/sweep")

    # cache / train
    for name, helptext in (("cache", "diffusion-pipe --cache_only"),
                           ("train", "diffusion-pipe deepspeed launch")):
        t = sub.add_parser(name, help=helptext)
        t.add_argument("--config", required=True)
        t.add_argument("--repo-root", default=None)
        t.add_argument("--num-gpus", type=int, default=1)
        t.add_argument("--gpu-ids", default=None, help="e.g. 0,1 (shared box; pins cards)")
        t.add_argument("--dry-run", action="store_true", help="print the command, don't exec")
        if name == "cache":
            t.add_argument("--regenerate", action="store_true")
        else:
            t.add_argument("--pipeline-stages", type=int, default=None)
            t.add_argument("--resume", nargs="?", const=True, default=False,
                           help="--resume (latest) or --resume <ckpt-dir>")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")

    if args.cmd == "doctor":
        rep = api.doctor(repo_root=args.repo_root, config_toml=args.config)
        print(rep.render())
        return 0 if rep.ok else 1

    if args.cmd == "download":
        paths = api.download_models(args.dest, base=args.base)
        print("\n# ---- paste into anima_lora.toml [model] ----")
        print(f"transformer_path = '{paths.transformer_path}'")
        print(f"vae_path         = '{paths.vae_path}'")
        print(f"llm_path         = '{paths.llm_path}'")
        return 0

    if args.cmd == "inspect":
        api.inspect_source(args.repo, args.config, split=args.split,
                           n=args.inspect_n, verbose=True)
        return 0

    if args.cmd == "export":
        counts = api.export_dataset(
            api.ExportConfig(
                repo=args.repo, configs=args.configs, out_root=args.out_root,
                route_column=args.route_column, caption_column=args.caption_column,
                caption_format=args.caption_format, image_ext=args.image_ext,
                with_mask=args.with_mask, with_conditioning=args.with_conditioning,
                limit_per_concept=args.limit_per_concept, limit_total=args.limit,
                require_audit_approved=not args.no_audit_filter,
                require_age_pass=not args.no_age_filter))
        print("\nPer-concept counts written:")
        for concept in sorted(counts):
            print(f"  {concept:<28}{counts[concept]}")
        print(f"\nOutput root: {Path(args.out_root).resolve()}")
        return 0

    if args.cmd == "build":
        out = api.build_dataset_toml(args.root, args.out,
                                     api.DatasetTomlConfig(target_effective=args.target,
                                                           max_repeats=args.max_repeats))
        print(f"Wrote {out}")
        return 0

    if args.cmd == "init-config":
        import shutil
        from importlib import resources
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        for name in ("anima_lora.toml", "anima_dataset.toml"):
            src = resources.files("geolip_anima_trainer").joinpath("templates", name)
            with resources.as_file(src) as p:
                shutil.copy(p, out / name)
            print(f"Wrote {out / name}")
        return 0

    if args.cmd == "validate":
        cfg = api.load_train_config(args.config, args.dataset)
        api.validate(cfg)
        print(f"OK: {args.config} is a valid TrainConfig "
              f"(adapter={'FFT' if cfg.adapter is None else f'rank {cfg.adapter.rank}'}, "
              f"lr={cfg.optimizer.lr}, llm_adapter_lr={cfg.model.llm_adapter_lr}).")
        return 0

    if args.cmd == "sweep":
        base = api.load_train_config(args.config, args.dataset)
        for tag, path in api.sweep(base, ranks=args.ranks, lrs=args.lrs,
                                   runs_root=args.runs_root, configs_root=args.configs_root):
            print(f"  {tag:<16}{path}")
        return 0

    if args.cmd in ("cache", "train"):
        fn = api.cache if args.cmd == "cache" else api.train
        kwargs = dict(repo_root=args.repo_root, num_gpus=args.num_gpus,
                      gpu_ids=_parse_ids(args.gpu_ids), dry_run=args.dry_run)
        if args.cmd == "cache":
            kwargs["regenerate"] = args.regenerate
        else:
            kwargs["pipeline_stages"] = args.pipeline_stages
            kwargs["resume"] = args.resume
        try:
            rc = fn(args.config, **kwargs)
        except api.WindowsTrainingRefused as e:
            print(str(e), file=sys.stderr)
            return 2
        except api.DiffusionPipeNotFound as e:
            print(str(e), file=sys.stderr)
            return 3
        return rc if isinstance(rc, int) else 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
