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
import os
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

    # subjects (columnar subject-bucket export; JSON caption verbatim)
    sj = sub.add_parser("subjects", help="columnar extraction into subject buckets")
    sj.add_argument("--repo", default=api.SubjectBucketConfig.repo)
    sj.add_argument("--config", default="qwen_90k", help="HF config to extract")
    sj.add_argument("--out", dest="out_root", default="datasets/anima_subjects")
    sj.add_argument("--limit", type=int, default=1000, help="total accepted images")
    sj.add_argument("--min-bucket-size", type=int, default=10)
    sj.add_argument("--fuzzy-cutoff", type=float, default=0.62)
    sj.add_argument("--no-head-noun", action="store_true",
                    help="bucket by full subject phrase instead of head noun")
    sj.add_argument("--keep-unmergeable", action="store_true",
                    help="(legacy difflib path) send unmergeable small buckets to 'misc'")
    sj.add_argument("--no-audit-filter", action="store_true")
    sj.add_argument("--age-filter", action="store_true", help="require age_classifier_pass")
    sj.add_argument("--build-toml", default=None,
                    help="also write a balanced dataset.toml at this path")
    # semantic grouping of the sparse tail
    sj.add_argument("--no-semantic", action="store_true",
                    help="disable semantic grouping (use legacy difflib plan)")
    sj.add_argument("--similarity-model", default=api.SubjectBucketConfig.similarity_model,
                    help="sentence-transformers model id (needs the [similarity] extra); "
                         "nomic-ai/nomic-embed-text-v1 reuses a cached model (0 download)")
    sj.add_argument("--semantic-backend", default="auto",
                    choices=["auto", "sentence-transformers", "trigram", "difflib"])
    sj.add_argument("--sim-threshold", type=float, default=0.45)
    sj.add_argument("--min-final-group-size", type=int, default=12)
    sj.add_argument("--drop-small", action="store_true",
                    help="drop ungroupable sparse subjects instead of pooling into misc_*")
    # dual-caption modes + oversized-bucket splitting
    sj.add_argument("--caption-mode", default="before_after",
                    choices=["separate", "mixed", "before_after"],
                    help="how vlm+animetimm samples are organized (before_after = first LoRA)")
    sj.add_argument("--no-mixed-concat", action="store_true",
                    help="MIXED: don't add the 3rd vlm+animetimm joint caption")
    sj.add_argument("--no-split", action="store_true",
                    help="don't split oversized buckets")
    sj.add_argument("--max-bucket-size", type=int, default=None,
                    help="cap per bucket (default data-dependent: >10k=1000, >=1k=500, else 250)")
    sj.add_argument("--prefer-attr-source", default="animetimm",
                    choices=["animetimm", "vlm"], help="which caption's attributes drive splits")

    # build
    b = sub.add_parser("build", help="balanced dataset.toml from concept folders")
    b.add_argument("--root", required=True)
    b.add_argument("--out", default="configs/anima_dataset.toml")
    b.add_argument("--target", type=int, default=None)
    b.add_argument("--alpha", type=float, default=0.5,
                   help="balance damping: 0=equalize(legacy), 1=none, 0.5=sqrt(default)")
    b.add_argument("--cap-mult", type=float, default=1.25)
    b.add_argument("--max-repeats", type=int, default=8)

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
            t.add_argument("--progress", action="store_true",
                           help="live %-done + ETA from the cache SQLite metadata")
            t.add_argument("--progress-interval", type=float, default=30.0)
            t.add_argument("--log-path", default=None,
                           help="send diffusion-pipe output here so the progress line stays clean")
            t.add_argument("--backup-repo", default=None,
                           help="HF dataset repo to periodically push the cache to during the run")
            t.add_argument("--backup-interval", type=float, default=1800.0,
                           help="seconds between cache pushes (also pushes on each shard finalize)")
            t.add_argument("--backup-root", default=None,
                           help="tree to push from (before_after: the anima_subjects parent)")
            t.add_argument("--backup-token", default=None, help="HF token (else $HF_TOKEN)")
            t.add_argument("--trust-cache", action="store_true",
                           help="load a restored cache's metadata without re-validating (resume)")
        else:
            t.add_argument("--pipeline-stages", type=int, default=None)
            t.add_argument("--resume", nargs="?", const=True, default=False,
                           help="--resume (latest) or --resume <ckpt-dir>")

    # before_after (two chained runs: full VLM phase -> full animetimm phase)
    ba = sub.add_parser("train-before-after",
                        help="BEFORE_AFTER first-LoRA: VLM phase then animetimm phase (resumed)")
    ba.add_argument("--lora-vlm", required=True, help="lora.toml referencing dataset_vlm.toml")
    ba.add_argument("--lora-animetimm", required=True,
                    help="lora.toml referencing dataset_animetimm.toml")
    ba.add_argument("--repo-root", default=None)
    ba.add_argument("--num-gpus", type=int, default=1)
    ba.add_argument("--gpu-ids", default=None)
    ba.add_argument("--dry-run", action="store_true")

    # cache preservation on HF Hub (frozen dataset + accumulating cache)
    cp = sub.add_parser("cache-push", help="push the frozen dataset+cache tree to an HF dataset repo")
    cp.add_argument("--out", dest="out_root_or_toml", required=True, help="out_root dir OR a toml")
    cp.add_argument("--repo-id", required=True)
    cp.add_argument("--token", default=None, help="HF token (else $HF_TOKEN / cached login)")
    cp.add_argument("--cache-only", action="store_true", help="push only cache/anima/** (no images)")
    cp.add_argument("--dry-run", action="store_true")

    cl = sub.add_parser("cache-pull", help="pull the frozen dataset+cache tree from an HF dataset repo")
    cl.add_argument("--out", dest="out_root", required=True, help="FIXED local dir to restore into")
    cl.add_argument("--repo-id", required=True)
    cl.add_argument("--token", default=None, help="HF token (else $HF_TOKEN / cached login)")
    cl.add_argument("--dry-run", action="store_true")
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

    if args.cmd == "subjects":
        cfg = api.SubjectBucketConfig(
            repo=args.repo, config=args.config, out_root=args.out_root, limit=args.limit,
            min_bucket_size=args.min_bucket_size, fuzzy_cutoff=args.fuzzy_cutoff,
            head_noun=not args.no_head_noun, drop_unmergeable=not args.keep_unmergeable,
            require_audit_approved=not args.no_audit_filter, require_age_pass=args.age_filter,
            use_semantic=not args.no_semantic, similarity_model=args.similarity_model,
            semantic_backend=args.semantic_backend, sim_threshold=args.sim_threshold,
            min_final_group_size=args.min_final_group_size, keep_small=not args.drop_small,
            caption_mode=api.CaptionMode(args.caption_mode),
            dedupe_mixed_concat=not args.no_mixed_concat,
            split_oversized=not args.no_split, max_bucket_size=args.max_bucket_size,
            prefer_attr_source=args.prefer_attr_source)
        rep = api.export_subject_buckets(cfg)
        print(f"\nmode={rep['caption_mode']}  scanned={rep['scanned']}  "
              f"accepted_images={rep['accepted_images']}  dropped={rep['dropped_images']}")
        print(f"caption stats: {rep['caption_stats']}  max_bucket_size={rep['max_bucket_size']}")
        print(f"raw subjects: {rep['raw_subjects']}  ->  final buckets: {rep['n_final_buckets']}")
        if rep["oversized_subjects"]:
            print(f"oversized (split): {rep['oversized_subjects'][:10]}")
        print("\nfinal buckets (samples each):")
        for name, n in list(rep["final_buckets"].items())[:40]:
            print(f"  {name:<34}{n}")
        if args.build_toml:
            cdir = Path(args.build_toml)
            if cdir.suffix == ".toml":      # a file path was given -> use its directory
                cdir = cdir.parent if cdir.parent != Path("") else Path(".")
            tomls = api.build_mode_tomls(args.out_root, cfg, configs_dir=cdir)
            for t in tomls:
                print(f"Wrote {t}")
        return 0

    if args.cmd == "build":
        out = api.build_dataset_toml(args.root, args.out,
                                     api.DatasetTomlConfig(target_effective=args.target,
                                                           balance_alpha=args.alpha,
                                                           cap_mult=args.cap_mult,
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
            kwargs["progress"] = args.progress
            kwargs["progress_interval"] = args.progress_interval
            kwargs["log_path"] = args.log_path
            kwargs["backup_repo"] = args.backup_repo
            kwargs["backup_interval"] = args.backup_interval
            kwargs["backup_root"] = args.backup_root
            kwargs["backup_token"] = args.backup_token or os.environ.get("HF_TOKEN")
            kwargs["trust_cache"] = args.trust_cache
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

    if args.cmd == "train-before-after":
        try:
            api.train_before_after(args.lora_vlm, args.lora_animetimm,
                                   repo_root=args.repo_root, num_gpus=args.num_gpus,
                                   gpu_ids=_parse_ids(args.gpu_ids), dry_run=args.dry_run)
        except api.WindowsTrainingRefused as e:
            print(str(e), file=sys.stderr)
            return 2
        except api.DiffusionPipeNotFound as e:
            print(str(e), file=sys.stderr)
            return 3
        return 0

    if args.cmd == "cache-push":
        tok = args.token or os.environ.get("HF_TOKEN")
        print("pushed ->", api.cache_push(args.out_root_or_toml, args.repo_id, token=tok,
              include_dataset=not args.cache_only, dry_run=args.dry_run))
        return 0

    if args.cmd == "cache-pull":
        tok = args.token or os.environ.get("HF_TOKEN")
        print("pulled ->", api.cache_pull(args.out_root, args.repo_id, token=tok,
              dry_run=args.dry_run))
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
