#!/usr/bin/env python3
"""
trainer_runner.py — the repo-side "program runner" for the RTX PRO 6000 TRAINING box, the mirror
image of cache_factory.CacheFactory. Same thin-shell contract: the notebook is a static handful of
`t.<step>()` calls and ALL logic lives here, so iterating = `git pull`, never re-pasting cells.

ROLE. The cache factory (Colab A100) builds + pushes the VAE-latent/text-embed cache to HF; this
runner PULLS it onto the persistent Blackwell box, reconstructs the images byte-identically (so the
cache fingerprint matches), builds the long-run training config, and launches the two-phase
before_after train DETACHED (so a kernel restart can't kill an ~18 h/epoch job).

Durability here is the opposite of the factory: the box is PERSISTENT, so checkpoints on local disk
are the lifeline (`checkpoint_every_n_minutes` -> `--resume_from_checkpoint`); the HF LoRA backup is
an optional heartbeat. The one shared invariant: DATA_ROOT must be the SAME absolute path the factory
used (default /workspace/anima_data) or the latent fingerprint won't match and the cache is wiped.

    from geolip_anima_trainer.trainer_runner import TrainerRunner
    t = TrainerRunner()
    t.setup(); t.prepare_dataset(); t.build_configs(); t.train()    # train() launches DETACHED
    t.tail()                                                        # watch; t.resume()/t.backup() as needed
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field, fields
from pathlib import Path

from . import api as _api
from .cache_factory import _RunnerMixin, _has_cache   # shared glue (auth/gpu/models/subject_cfg/state)


def _pid_alive(pid: int) -> "bool | None":
    """Best-effort liveness for a detached child (POSIX): True alive, False exited, None unknown."""
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True                          # exists, just not ours
    except OSError:
        return None


@dataclass
class TrainerConfig:
    """Long-run training knobs (mirror anima_full90k_train.ipynb §7) + the shared fields the runner
    glue needs. DATA_ROOT defaults to the factory's portable /workspace/anima_data so the cache
    fingerprint matches; cache_repo is where the factory pushed the cache."""
    repo_root: str = field(default_factory=lambda: os.environ.get("ANIMA_REPO", "/workspace/anima-trainer"))
    data_root: str | None = None                 # None -> /workspace/anima_data (MUST match the factory)
    hf_home: str | None = None                   # None -> {data_root}/hf_cache
    models_dir: str | None = None
    # dataset (only caption_mode/source feed build + reconstruct; the cache is PULLED, not extracted)
    source_repo: str = "AbstractPhil/diffusion-pretrain-set-ft1"
    source_config: str = "qwen_90k"
    limit: int | None = None
    caption_mode: str = "before_after"
    similarity_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    min_bucket_size: int = 10
    min_final_group_size: int = 8
    # HF repos
    cache_repo: str | None = None                # None -> {hf_user}/anima-90k-cache (the factory's repo)
    backup_repo: str | None = None               # MODEL repo for LoRA checkpoints (optional heartbeat)
    # the long-run recipe
    num_gpus: int = 1
    res: int = 1024
    rank: int = 64
    lr: float = 2e-5
    epochs_vlm: int = 3
    epochs_animetimm: int = 2
    micro_batch: int = 16
    grad_accum: int = 4
    save_every_n_steps: int = 500
    save_every_n_epochs: int = 1
    checkpoint_every_n_minutes: int = 30
    warmup_steps: int = 200
    activation_checkpointing: bool = True        # ON for the long run (frees VRAM for micro_batch=16)
    compile: bool = True
    balance_alpha: float = 0.5
    cap_mult: float = 1.25
    max_repeats: int = 8
    map_num_proc: int | None = None

    @classmethod
    def from_env(cls, **overrides) -> "TrainerConfig":
        env: dict = {}
        if os.environ.get("ANIMA_DATA_ROOT"):
            env["data_root"] = os.environ["ANIMA_DATA_ROOT"]
        if os.environ.get("ANIMA_CACHE_REPO"):
            env["cache_repo"] = os.environ["ANIMA_CACHE_REPO"]
        if os.environ.get("ANIMA_BACKUP_REPO"):
            env["backup_repo"] = os.environ["ANIMA_BACKUP_REPO"]
        if os.environ.get("ANIMA_LIMIT"):
            env["limit"] = None if os.environ["ANIMA_LIMIT"].lower() in ("none", "") else int(os.environ["ANIMA_LIMIT"])
        valid = {f.name for f in fields(cls)}
        bad = set(overrides) - valid
        if bad:
            raise TypeError(f"unknown TrainerConfig override(s): {sorted(bad)}")
        return cls(**{**env, **{k: v for k, v in overrides.items() if k in valid}})


class TrainerRunner(_RunnerMixin):
    """Stateful trainer orchestrator (one method per notebook cell, idempotent, run in order). Mirrors
    CacheFactory but pulls the cache instead of building it and trains instead of caching. State is
    in-memory (mirrored to {data_root}/trainer_state.json); a fresh runtime re-runs setup() first."""
    TAG = "trainer"
    STATE_FILE = "trainer_state.json"
    EXPECT_SM = 120                              # the recipe targets RTX PRO 6000 Blackwell sm_120

    def __init__(self, config: "TrainerConfig | None" = None, **overrides):
        self.cfg = config or TrainerConfig.from_env(**overrides)
        self.state: dict = {}

    # ---- 1. setup: env (fixed DATA_ROOT) -> auth -> gpu -> models -------------
    def setup(self) -> dict:
        self._setup_env()
        self._auth()
        self._verify_gpu()
        self._download_models()
        self._save_state()
        print(f"[trainer] setup done | DATA_ROOT={self.state['data_root']} | cache_repo={self.cfg.cache_repo}")
        return self.state

    def _setup_env(self) -> None:
        # DATA_ROOT is a REAL dir on the persistent disk and MUST equal the factory's portable path,
        # else the latent fingerprint (absolute paths) won't match -> the cache is re-validated/wiped.
        data_root = self.cfg.data_root or "/workspace/anima_data"
        hf_home = self.cfg.hf_home or f"{data_root}/hf_cache"
        os.environ["HF_HOME"] = hf_home
        os.environ["HF_DATASETS_CACHE"] = f"{hf_home}/datasets"
        os.environ["TMPDIR"] = f"{data_root}/tmp"
        os.environ["ANIMA_DATA_ROOT"] = data_root
        for d in (hf_home, os.environ["TMPDIR"], data_root):
            os.makedirs(d, exist_ok=True)
        if self.cfg.repo_root not in sys.path:
            sys.path.insert(0, self.cfg.repo_root)
        self.state.update(data_root=data_root, subjects_root=f"{data_root}/anima_subjects", hf_home=hf_home)
        print(f"[trainer] DATA_ROOT={data_root} (MUST match the factory's portable path) | HF_HOME={hf_home}")

    # ---- 2. pull the factory-built cache + reconstruct images, then prune ------
    def prepare_dataset(self) -> bool:
        subj = self._need("subjects_root")
        self._need("hf_token")
        if not self.cfg.cache_repo:
            raise RuntimeError("no cache_repo — the trainer PULLS the factory-built cache from HF. Set "
                               "cache_repo= (or ANIMA_CACHE_REPO), or run the cache factory first.")
        if _has_cache(subj):
            print(f"[trainer] cache already on disk at {subj} (skip pull)")
        else:
            _api.cache_pull(subj, self.cfg.cache_repo, token=self.state["hf_token"])   # cache+index -> rebuild images
            if not _has_cache(subj):
                raise RuntimeError(f"pulled {self.cfg.cache_repo} but found no cache at {subj} — build+push it "
                                   f"with the cache factory first.")
            print(f"[trainer] pulled cache+index + reconstructed images -> {subj}")
        self.state["resume"] = True
        try:                                                   # the reconstruct refetched source shards; reclaim them
            freed = _api.prune_source_cache(self.cfg.source_repo, also=[subj])
            print(f"[trainer] pruned source parquet: freed {freed['freed_bytes'] / 1e9:.1f} GB")
        except Exception as e:  # noqa: BLE001
            print(f"[trainer] prune skipped: {e}")
        self._save_state()
        return True

    # ---- 3. build the long-run training configs -------------------------------
    def build_configs(self) -> dict:
        subj = self._need("subjects_root")
        paths = self._need("model_paths")
        cfg = self.state.get("subject_cfg") or self._subject_cfg()
        self.state["subject_cfg"] = cfg
        configs_dir = f"{self.cfg.repo_root}/configs/trainer"
        ds = _api.build_mode_tomls(subj, cfg, configs_dir=configs_dir, resolutions=[self.cfg.res],
                                   balance_alpha=self.cfg.balance_alpha, cap_mult=self.cfg.cap_mult,
                                   max_repeats=self.cfg.max_repeats)
        ds = {Path(t).stem: str(t) for t in ds}
        if not ("dataset_vlm" in ds and "dataset_animetimm" in ds):
            raise RuntimeError(f"the trainer supports caption_mode='before_after' only "
                               f"(got {self.cfg.caption_mode!r}); build_mode_tomls produced {list(ds)}")
        model = _api.ModelConfig(transformer_path=paths.transformer_path, vae_path=paths.vae_path,
                                 llm_path=paths.llm_path, llm_adapter_lr=0.0)   # adapter FROZEN
        loras = {
            "vlm": self._write_lora("vlm", ds["dataset_vlm"], self.cfg.epochs_vlm, model, configs_dir),
            "animetimm": self._write_lora("animetimm", ds["dataset_animetimm"], self.cfg.epochs_animetimm,
                                          model, configs_dir),
        }
        self.state["dataset_tomls"] = ds
        self.state["lora_tomls"] = loras
        self.state["configs_dir"] = configs_dir
        self._save_state()
        print(f"[trainer] configs: {loras}")
        return loras

    def _write_lora(self, phase, ds_toml, epochs, model, configs_dir) -> str:
        run = _api.RunConfig(
            output_dir=f"{self.state['data_root']}/runs/{phase}", epochs=epochs,
            micro_batch_size_per_gpu=self.cfg.micro_batch, gradient_accumulation_steps=self.cfg.grad_accum,
            save_every_n_steps=self.cfg.save_every_n_steps, save_every_n_epochs=self.cfg.save_every_n_epochs,
            checkpoint_every_n_minutes=self.cfg.checkpoint_every_n_minutes, warmup_steps=self.cfg.warmup_steps,
            activation_checkpointing=self.cfg.activation_checkpointing, compile=self.cfg.compile,
            map_num_proc=self.cfg.map_num_proc or os.cpu_count())
        cfg = _api.TrainConfig(run=run, model=model, adapter=_api.AdapterConfig(rank=self.cfg.rank),
                               optimizer=_api.OptimizerConfig(lr=self.cfg.lr),
                               dataset=_api.DatasetConfig(resolutions=[self.cfg.res]),
                               dataset_toml_path=ds_toml)
        _api.validate(cfg)
        p = Path(configs_dir) / f"lora_{phase}.toml"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(_api.render_lora_toml(cfg), encoding="utf-8")
        return str(p)

    # ---- 4. train (DETACHED by default so a kernel restart can't kill it) ------
    def train(self, *, detached: bool = True, dry_run: bool = False):
        loras = self._need("lora_tomls")
        logs = f"{self.state['data_root']}/runs"
        os.makedirs(logs, exist_ok=True)
        if not detached:                                       # blocking in-process (short runs / debugging)
            rc = _api.train_before_after(loras["vlm"], loras["animetimm"], repo_root=self.cfg.repo_root,
                                         num_gpus=self.cfg.num_gpus, configs_dir=self.state.get("configs_dir"),
                                         log_dir=logs)
            print(f"[trainer] train-before-after rc={rc}")
            return rc
        argv = [sys.executable, "-m", "geolip_anima_trainer.cli", "train-before-after",
                "--lora-vlm", loras["vlm"], "--lora-animetimm", loras["animetimm"],
                "--repo-root", self.cfg.repo_root, "--num-gpus", str(self.cfg.num_gpus)]
        if dry_run:
            argv.append("--dry-run")
        info = self._launch_detached(argv, f"{logs}/train.log")
        self.state["train"] = info
        self._save_state()
        print(f"[trainer] training launched DETACHED (survives a kernel restart) | pid={info['pid']}")
        print(f"[trainer] watch:  t.tail()   |   resume:  t.resume()   |   the LoRA is runs/animetimm/*/epochN/")
        return info

    def resume(self, phase: str = "animetimm", *, detached: bool = True):
        """Resume a phase from its latest checkpoint_every_n_minutes state (--resume_from_checkpoint)."""
        loras = self._need("lora_tomls")
        if phase not in loras:
            raise ValueError(f"phase must be one of {list(loras)}")
        if not detached:
            rc = _api.train(loras[phase], repo_root=self.cfg.repo_root, num_gpus=self.cfg.num_gpus, resume=True)
            print(f"[trainer] resume[{phase}] rc={rc}")
            return rc
        argv = [sys.executable, "-m", "geolip_anima_trainer.cli", "train", "--config", loras[phase],
                "--repo-root", self.cfg.repo_root, "--num-gpus", str(self.cfg.num_gpus), "--resume"]
        info = self._launch_detached(argv, f"{self.state['data_root']}/runs/resume_{phase}.log")
        print(f"[trainer] resume[{phase}] launched DETACHED | pid={info['pid']} | log={info['log']}")
        return info

    def _launch_detached(self, argv, log) -> dict:
        os.makedirs(os.path.dirname(log), exist_ok=True)
        logf = open(log, "a", encoding="utf-8")
        try:
            kw = {"start_new_session": True} if os.name == "posix" else {}   # setsid -> survives SIGHUP
            proc = subprocess.Popen(argv, stdout=logf, stderr=subprocess.STDOUT,
                                    cwd=self.cfg.repo_root, env={**os.environ}, **kw)
        finally:
            logf.close()                                       # the child kept its own dup'd fd
        return {"pid": proc.pid, "log": log, "cmd": " ".join(argv)}

    # ---- monitor / backup / status / convenience ------------------------------
    def tail(self, n: int = 40) -> None:
        dr = self.state.get("data_root")
        runs = Path(f"{dr}/runs") if dr else None
        if not (runs and runs.exists()):
            print("[trainer] no train log yet")
            return
        # detached train() writes runs/train.log; a blocking run writes {vlm,animetimm}.log; resume_*.log.
        # Pick the MOST-RECENTLY-WRITTEN existing log so a stale phase log can't shadow the live one.
        cands = [runs / x for x in ("train.log", "animetimm.log", "vlm.log")] + list(runs.glob("resume_*.log"))
        existing = [c for c in cands if c.exists()]
        if not existing:
            print("[trainer] no train log yet")
            return
        log = max(existing, key=lambda c: c.stat().st_mtime)
        print(f"--- {log.name} ---")
        try:
            print(subprocess.run(["tail", "-n", str(n), str(log)], capture_output=True, text=True).stdout)
        except Exception:  # noqa: BLE001 — no `tail` (e.g. Windows) -> read the byte tail
            print(log.read_text(encoding="utf-8", errors="replace")[-4000:])

    def backup(self) -> "str | None":
        """Push the newest LoRA checkpoint dir to the HF *model* backup_repo (optional heartbeat;
        the persistent disk is the primary durability)."""
        repo, token = self.cfg.backup_repo, self.state.get("hf_token")
        dr = self.state.get("data_root")
        if not (repo and token and dr):
            print("[trainer] no backup_repo/token -> skip (checkpoints are safe on the persistent disk)")
            return None
        runs_dir = Path(f"{dr}/runs")
        epochs = [d for d in runs_dir.rglob("epoch*") if d.is_dir() and any(d.glob("*.safetensors"))]
        if not epochs:
            print("[trainer] no checkpoints yet")
            return None
        run = max(epochs, key=lambda d: d.stat().st_mtime).parent   # rank by EPOCH mtime (matches status())
        rel = run.relative_to(runs_dir).as_posix()
        from huggingface_hub import HfApi, create_repo
        create_repo(repo, token=token, repo_type="model", private=True, exist_ok=True)
        HfApi(token=token).upload_folder(folder_path=str(run), repo_id=repo, repo_type="model",
                                         path_in_repo=f"runs/{rel}",
                                         ignore_patterns=["global_step*/*", "global_step*/**"],
                                         commit_message=f"backup :: {rel}")
        print(f"[trainer] backed up -> https://huggingface.co/{repo}/tree/main/runs/{rel}")
        return rel

    def status(self) -> dict:
        dr = self.state.get("data_root")
        out: dict = {"data_root": dr, "resume": self.state.get("resume"), "cache_repo": self.cfg.cache_repo,
                     "backup_repo": self.cfg.backup_repo, "train": self.state.get("train")}
        tr_info = self.state.get("train")
        if isinstance(tr_info, dict) and tr_info.get("pid"):
            out["train_alive"] = _pid_alive(tr_info["pid"])   # detached child still running? (None=unknown)
        if dr and Path(dr).exists():
            du = shutil.disk_usage(dr)
            out["disk_gb"] = {"used": round(du.used / 1e9, 1), "free": round(du.free / 1e9, 1),
                              "total": round(du.total / 1e9, 1)}
        runs_dir = Path(f"{dr}/runs") if dr else None
        if runs_dir and runs_dir.exists():
            cks = [d for d in runs_dir.rglob("epoch*") if d.is_dir() and any(d.glob("*.safetensors"))]
            if cks:
                out["latest_checkpoint"] = str(max(cks, key=lambda d: d.stat().st_mtime))
        print("[trainer] status:", json.dumps(out, indent=2, default=str))
        return out

    def run_all(self, *, detached: bool = True):
        self.setup()
        self.prepare_dataset()
        self.build_configs()
        return self.train(detached=detached)
