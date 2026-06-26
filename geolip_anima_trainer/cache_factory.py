#!/usr/bin/env python3
"""
cache_factory.py — the repo-side "program runner" for the Colab A100 cache-FACTORY workflow.

WHY THIS EXISTS. Colab can't hot-swap a notebook at runtime — every new session is a fresh
notebook + fresh runtime, and changing logic means re-pasting cells by hand. So we invert it:
the NOTEBOOK is a tiny static shell (clone -> install -> a handful of CacheFactory calls) and
ALL the behavior that might change lives HERE, in the repo. Iterating = `git pull`; the notebook
never needs editing. CacheFactory also owns the program STATE and reconstructs it from durable
sources (the HF cache + a deterministic config) on every fresh runtime, so a reset just means
re-running the same cells — no notebook surgery.

ROLE. This is a CACHE FACTORY: it builds the expensive VAE-latent + Qwen-3 text-embed cache for
the full dataset on the A100's big, fast, EPHEMERAL local-scratch SSD, and pushes it to HF Hub
(via cache_sync) so a training box (e.g. the RTX PRO 6000) pulls + reconstructs it. It does NOT
train. The scratch is wiped on disconnect, so the periodic HF push is the durability — not the
disk. (On Colab the in-kernel keepalive is the wrong lever — idle is browser-UI based — so the
real safety net here is the push, not gpu_keepalive; see CLAUDE.md.)

USAGE (the whole notebook after bootstrap):
    from geolip_anima_trainer.cache_factory import CacheFactory
    f = CacheFactory(limit=None)        # full set; None = everything that fits scratch
    f.setup(); f.prepare_dataset(); f.build_configs(); f.run_cache()
    # or just: CacheFactory(limit=None).run_all()
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

# api is light + hf/torch-free at import (download_anima lazy-imports hf), so importing this
# module does NOT pin HF_HOME — setup() sets HF_HOME before the first hf call. torch/hf stay
# lazy-imported inside methods.
from . import api as _api


# =============================================================================
# ENVIRONMENT HELPERS  (stdlib only; safe before torch/hf exist)
# =============================================================================
def _list_mounts() -> list[str]:
    """Mountpoints from psutil (optional) or /proc/mounts — empty on a platform with neither."""
    try:
        import psutil  # optional; /proc/mounts fallback covers a bare runtime
        return [p.mountpoint for p in psutil.disk_partitions(all=True)]
    except Exception:  # noqa: BLE001
        try:
            with open("/proc/mounts", encoding="utf-8") as fh:
                return [ln.split()[1] for ln in fh if len(ln.split()) > 1]
        except OSError:
            return []


def find_scratch(min_gb: int = 300, *, exclude=("/", "/content/drive"),
                 mounts: "list[str] | None" = None) -> "tuple[str, int] | None":
    """The biggest WRITABLE filesystem of at least `min_gb` GB that isn't root / Drive / a pseudo
    fs — i.e. the Colab A100 'local-scratch' NVMe SSD (~368 GB). The consumer scratch mount path
    is undocumented + rollout-dependent, so we DISCOVER it (don't hard-code). Returns (mount,
    total_bytes) or None. `mounts` is injectable for tests. Verify with `!df -h` if it picks wrong."""
    if mounts is None:
        mounts = _list_mounts()
    best = None
    for m in dict.fromkeys(mounts):                       # dedupe, keep order
        if m in exclude or m.startswith(("/proc", "/sys", "/dev", "/run", "/content")):
            continue                                       # skip pseudo-fs + the /content overlay/Drive
        try:
            du = shutil.disk_usage(m)
        except OSError:
            continue
        if du.total >= min_gb * 1024**3 and os.access(m, os.W_OK):
            if best is None or du.total > best[1]:
                best = (m, du.total)
    return best


def get_hf_token() -> str | None:
    """The strong token lookup: Colab Secrets (userdata) first, then $HF_TOKEN. The caller
    asserts — a cache factory that can't push is pointless, so failing loud beats silently
    dropping the backup."""
    try:
        from google.colab import userdata  # type: ignore
        t = userdata.get("HF_TOKEN")
        if t:
            return t
    except Exception:  # noqa: BLE001 — not on Colab / secret not set
        pass
    return os.environ.get("HF_TOKEN")


def _refresh_symlink(link: str, target: str) -> bool:
    """Point `link` at `target`. An existing SYMLINK is repointed (a prior session's scratch path is
    dangling now). An EMPTY leftover dir is removed and replaced (so a stale /workspace/anima_data
    from a prior session can't block portability). A NON-empty real dir is respected (don't clobber
    real data — e.g. the training box's actual /workspace/anima_data). Returns True iff `link` now
    points at `target`."""
    p = Path(link)
    os.makedirs(os.path.dirname(link) or "/", exist_ok=True)
    if p.is_symlink():
        try:
            os.unlink(link)
        except OSError:
            return False
    elif p.exists():
        try:
            if p.is_dir() and not any(p.iterdir()):
                p.rmdir()                                  # empty leftover -> reclaim it
            else:
                return False                               # real data -> respect it
        except OSError:
            return False
    try:
        os.symlink(target, link)
        return True
    except OSError:
        return False


# =============================================================================
# CONFIG  (deterministic + serializable -> the state survives a runtime reset)
# =============================================================================
@dataclass
class FactoryConfig:
    """Everything the factory needs, all overridable from the notebook (kwargs) or env. Defaults
    target the qwen_90k full set onto auto-detected scratch, pushing to <user>/anima-90k-cache."""
    repo_root: str = field(default_factory=lambda: os.environ.get("ANIMA_REPO", "/content/anima-trainer"))
    data_root: str | None = None                 # None -> auto-detect scratch
    portable_root: str | None = "/workspace/anima_data"   # symlink so the cache is cross-box reusable; None -> use scratch path directly
    min_scratch_gb: int = 300
    models_dir: str | None = None                # default {repo_root}/models/anima
    # source dataset
    source_repo: str = "AbstractPhil/diffusion-pretrain-set-ft1"
    source_config: str = "qwen_90k"
    limit: int | None = None                     # None -> full ~83k
    caption_mode: str = "before_after"
    # HF target
    cache_repo: str | None = None                # None -> {hf_user}/anima-90k-cache
    backup_interval: float = 1800.0
    # subject bucketing
    min_bucket_size: int = 10
    min_final_group_size: int = 8
    similarity_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    # cache throughput + (carried into the lora toml, harmless for cache-only)
    caching_batch_size: int = 8
    map_num_proc: int | None = None              # None -> os.cpu_count()
    res: int = 1024
    rank: int = 64
    lr: float = 2e-5
    epochs_vlm: int = 3
    epochs_animetimm: int = 2
    micro_batch: int = 16
    grad_accum: int = 4

    @classmethod
    def from_env(cls, **overrides) -> "FactoryConfig":
        """Build from env (ANIMA_* keys) + explicit overrides (overrides win)."""
        env: dict = {}
        if os.environ.get("ANIMA_DATA_ROOT"):
            env["data_root"] = os.environ["ANIMA_DATA_ROOT"]
        if os.environ.get("ANIMA_CACHE_REPO"):
            env["cache_repo"] = os.environ["ANIMA_CACHE_REPO"]
        if os.environ.get("ANIMA_LIMIT"):
            env["limit"] = None if os.environ["ANIMA_LIMIT"].lower() in ("none", "") else int(os.environ["ANIMA_LIMIT"])
        valid = {f.name for f in fields(cls)}
        merged = {**env, **{k: v for k, v in overrides.items() if k in valid}}
        bad = set(overrides) - valid
        if bad:
            raise TypeError(f"unknown FactoryConfig override(s): {sorted(bad)}")
        return cls(**merged)


def _has_cache(root: str | Path) -> bool:
    return Path(root).is_dir() and any(Path(root).rglob("metadata.db"))


# =============================================================================
# THE RUNNER
# =============================================================================
class CacheFactory:
    """Stateful orchestrator. Each method is one notebook cell; all are idempotent. Run them in
    order within a session — `self.state` is in-memory, so a fresh runtime (or a bare re-run of a
    later cell) must re-run `setup()` first, which re-derives everything from the env + HF. State is
    mirrored to {data_root}/factory_state.json for inspection (it is NOT auto-reloaded; secrets +
    model paths are deliberately not persisted). `run_all()` does the whole sequence in one call."""

    def __init__(self, config: "FactoryConfig | None" = None, **overrides):
        self.cfg = config or FactoryConfig.from_env(**overrides)
        self.state: dict = {}

    def _need(self, key: str):
        """Read a state key set by an earlier step, with a clear error instead of a bare KeyError
        (a fresh runtime must re-run setup()/the earlier steps before this one)."""
        if key not in self.state:
            raise RuntimeError(f"factory state has no {key!r} yet — run f.setup() and the earlier steps "
                               f"first (a fresh runtime re-runs setup() before this).")
        return self.state[key]

    # ---- 1. setup: env (scratch/HF_HOME) -> auth -> gpu -> models -------------
    def setup(self) -> dict:
        self._setup_env()
        self._auth()
        self._verify_gpu()
        self._download_models()
        self._save_state()
        print(f"[factory] setup done | DATA_ROOT={self.state['data_root']} | cache_repo={self.cfg.cache_repo}")
        return self.state

    def _setup_env(self) -> None:
        # pick the data disk: explicit > scratch autodetect > /content fallback (the last won't fit
        # the full set, but keeps the factory runnable for a small LIMIT).
        scratch = None
        if self.cfg.data_root:
            data_root = self.cfg.data_root
        else:
            hit = find_scratch(self.cfg.min_scratch_gb)
            if hit:
                scratch = hit[0]
                print(f"[factory] local-scratch -> {scratch} ({hit[1] / 1e9:.0f} GB)")
                scratch_data = f"{scratch}/anima_data"
                os.makedirs(scratch_data, exist_ok=True)
                data_root = scratch_data
                if self.cfg.portable_root:
                    if _refresh_symlink(self.cfg.portable_root, scratch_data):
                        data_root = self.cfg.portable_root    # cross-box-stable path; bytes live on scratch
                        print(f"[factory] portable DATA_ROOT {data_root} -> {scratch_data} (symlink)")
                    else:
                        # the cache fingerprint embeds DATA_ROOT, so falling back to the volatile,
                        # mount-name-dependent scratch path means a NEXT session (different mount name)
                        # won't resume — warn loud rather than silently re-cache later.
                        print(f"[factory] !! WARNING: could not point portable_root "
                              f"{self.cfg.portable_root} at {scratch_data} (a real non-empty dir is "
                              f"there?). Using the volatile scratch path as DATA_ROOT -> RESUME ACROSS "
                              f"SESSIONS/BOXES MAY BREAK (re-cache). Free {self.cfg.portable_root}, or "
                              f"set portable_root=None deliberately.")
            elif self.cfg.limit is None:
                raise RuntimeError(
                    f"No local-scratch SSD (>={self.cfg.min_scratch_gb} GB) found and limit=None — the "
                    f"full set (~106 GB images+cache) won't fit /content. Re-run the setup cell with a "
                    f"smaller limit, e.g. CacheFactory(limit=40000), or an explicit data_root=. "
                    f"(Run !df -h to see your disks.)")
            else:
                data_root = "/content/anima_data"
                print(f"[factory] WARNING: no >={self.cfg.min_scratch_gb}GB scratch; using {data_root} "
                      f"(small limit only — run !df -h, or pass data_root=).")
        # HF_HOME + TMPDIR on the real fast disk (scratch if we found it, else data_root). These are
        # per-session caches; they don't need the portable path. Set BEFORE the first hf/datasets use.
        io_root = scratch or data_root
        os.environ["HF_HOME"] = f"{io_root}/hf_cache"
        os.environ["HF_DATASETS_CACHE"] = f"{io_root}/hf_cache/datasets"
        os.environ["TMPDIR"] = f"{io_root}/tmp"
        os.environ["ANIMA_DATA_ROOT"] = data_root
        for d in (os.environ["HF_HOME"], os.environ["TMPDIR"], data_root):
            os.makedirs(d, exist_ok=True)
        if self.cfg.repo_root not in sys.path:
            sys.path.insert(0, self.cfg.repo_root)
        self.state.update(
            data_root=data_root, scratch=scratch, io_root=io_root,
            subjects_root=f"{data_root}/anima_subjects",
            hf_home=os.environ["HF_HOME"],
        )

    def _auth(self) -> None:
        token = get_hf_token()
        assert token, ("No HF_TOKEN. A cache factory must push to HF (the scratch is ephemeral). "
                       "Set it in Colab Secrets (HF_TOKEN, WRITE scope) or export HF_TOKEN.")
        from huggingface_hub import login, whoami
        login(token=token, add_to_git_credential=True)
        user = whoami(token=token).get("name")
        self.state["hf_user"] = user
        self.state["hf_token"] = token
        if not self.cfg.cache_repo and user:
            self.cfg.cache_repo = f"{user}/anima-90k-cache"
        print(f"[factory] HF user={user} | cache_repo={self.cfg.cache_repo}")

    def _verify_gpu(self) -> dict:
        import torch
        ok = torch.cuda.is_available()
        name = torch.cuda.get_device_name(0) if ok else "NONE"
        cap = torch.cuda.get_device_capability(0) if ok else (0, 0)
        sm = cap[0] * 10 + cap[1]
        bf16 = bool(ok and torch.cuda.is_bf16_supported())
        print(f"[factory] torch {torch.__version__} | cuda {torch.version.cuda} | {name} | sm_{sm} | bf16 {bf16}")
        assert ok, "No CUDA device — start the GPU runtime."
        self.state["gpu"] = {"name": name, "sm": sm, "bf16": bf16}
        return self.state["gpu"]

    def _download_models(self) -> object:
        # default onto the fast scratch (io_root), not /content — keeps the 5.6 GB beside HF_HOME
        # and inside the same disk-usage accounting.
        dest = self.cfg.models_dir or f"{self.state['io_root']}/models/anima"
        paths = _api.download_models(dest, base="base-v1.0")
        self.state["model_paths"] = paths
        print(f"[factory] model: {paths.transformer_path}")
        return paths

    # ---- 2. dataset: resume from HF, or extract + store index, then prune ------
    def prepare_dataset(self) -> bool:
        subj = self._need("subjects_root")
        self._need("hf_token")
        cfg = self._subject_cfg()
        self.state["subject_cfg"] = cfg
        resume = _has_cache(subj)
        if not resume and self.cfg.cache_repo:
            try:
                _api.cache_pull(subj, self.cfg.cache_repo, token=self.state["hf_token"])
                resume = _has_cache(subj)
                print(f"[factory] pulled cache+index + rebuilt images -> {subj} | resume={resume}")
            except Exception as e:  # noqa: BLE001 — nothing to resume yet is normal
                print(f"[factory] no prior cache to pull ({e!r}) -> extracting fresh")
        if not resume:
            rep = _api.export_subject_buckets(cfg)
            print(f"[factory] extracted accepted={rep['accepted_images']} dropped={rep['dropped_images']} "
                  f"-> buckets={rep['n_final_buckets']} (index: {rep['index_path']})")
            if self.cfg.cache_repo:
                # uploads index.jsonl only (cache_sync always ignores images/.txt — they're rebuilt
                # from the index on resume); the latents are pushed later as run_cache() produces them.
                _api.cache_push(subj, self.cfg.cache_repo, token=self.state["hf_token"],
                                commit_message="store index (pre-cache)")
                print(f"[factory] stored index -> https://huggingface.co/datasets/{self.cfg.cache_repo}")
        else:
            print(f"[factory] RESUME: reusing the restored dataset at {subj}")
        self.state["resume"] = resume
        try:                                                   # reclaim the ~45 GB source parquet
            freed = _api.prune_source_cache(self.cfg.source_repo, also=[subj])
            print(f"[factory] pruned source parquet: freed {freed['freed_bytes'] / 1e9:.1f} GB")
        except Exception as e:  # noqa: BLE001
            print(f"[factory] prune skipped: {e}")
        self._save_state()
        return resume

    # ---- 3. configs: dataset tomls + a lora toml per phase ---------------------
    def build_configs(self) -> dict:
        subj = self._need("subjects_root")
        paths = self._need("model_paths")
        cfg = self.state.get("subject_cfg") or self._subject_cfg()
        configs_dir = f"{self.cfg.repo_root}/configs/colab_cache"
        ds = _api.build_mode_tomls(subj, cfg, configs_dir=configs_dir, resolutions=[self.cfg.res],
                                   balance_alpha=0.5, cap_mult=1.25, max_repeats=8)
        ds = {Path(t).stem: str(t) for t in ds}
        if not ("dataset_vlm" in ds and "dataset_animetimm" in ds):
            raise RuntimeError(f"the cache factory supports caption_mode='before_after' only "
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
        self._save_state()
        print(f"[factory] configs: {loras}")
        return loras

    def _write_lora(self, phase, ds_toml, epochs, model, configs_dir) -> str:
        run = _api.RunConfig(
            output_dir=f"{self.state['data_root']}/runs/{phase}", epochs=epochs,
            micro_batch_size_per_gpu=self.cfg.micro_batch, gradient_accumulation_steps=self.cfg.grad_accum,
            save_every_n_steps=500, save_every_n_epochs=1, checkpoint_every_n_minutes=30,
            warmup_steps=200,                            # activation_checkpointing/compile: leave at the
            caching_batch_size=self.cfg.caching_batch_size,   # validated OFF defaults — inert for caching,
            map_num_proc=self.cfg.map_num_proc or os.cpu_count())  # and these tomls may be reused to train
        cfg = _api.TrainConfig(run=run, model=model, adapter=_api.AdapterConfig(rank=self.cfg.rank),
                               optimizer=_api.OptimizerConfig(lr=self.cfg.lr),
                               dataset=_api.DatasetConfig(resolutions=[self.cfg.res]),
                               dataset_toml_path=ds_toml)
        _api.validate(cfg)
        p = Path(configs_dir) / f"lora_{phase}.toml"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(_api.render_lora_toml(cfg), encoding="utf-8")
        return str(p)

    # ---- 4. cache both phases, push to HF, optionally hold the pod -------------
    def run_cache(self, *, hold: bool = False) -> bool:
        loras = self._need("lora_tomls")
        subj = self._need("subjects_root")
        logs = f"{self.state['data_root']}/runs/cache_logs"
        os.makedirs(logs, exist_ok=True)
        # gpu_keepalive guards the GPU-idle push/reload windows (harmless on Colab, load-bearing on
        # RunPod/Lambda if you reuse this runner there). The periodic push is the real Colab safety net.
        with _api.gpu_keepalive():
            for phase in ("vlm", "animetimm"):
                print(f"\n[factory] === caching {phase} ===")
                try:
                    rc = _api.cache(loras[phase], repo_root=self.cfg.repo_root, num_gpus=1, progress=True,
                                    progress_interval=60, log_path=f"{logs}/cache_{phase}.log",
                                    backup_repo=self.cfg.cache_repo, backup_root=subj,
                                    backup_interval=self.cfg.backup_interval, backup_token=self.state["hf_token"],
                                    trust_cache=self.state.get("resume", False))
                    print(f"[factory] cache[{phase}] rc={rc}")
                except Exception as e:  # noqa: BLE001 — a crash must not skip the final push / status
                    self.state.setdefault("phase_errors", {})[phase] = repr(e)
                    print(f"[factory] cache[{phase}] FAILED -> {e!r} (inspect {logs}/cache_{phase}.log; "
                          f"re-running resumes via trust_cache)")
            if self.cfg.cache_repo:
                try:
                    _api.cache_push(subj, self.cfg.cache_repo, token=self.state["hf_token"],
                                    commit_message="post-cache full")
                    print(f"[factory] pushed full cache -> https://huggingface.co/datasets/{self.cfg.cache_repo}")
                except Exception as e:  # noqa: BLE001
                    print(f"[factory] final cache_push failed (retry with f.push()): {e!r}")
        if hold:
            print("[factory] holding (NOTE: on Colab the HF push is the real safety net, not this hold).")
            _api.keepalive(interval=60, gpu=True)
        return True

    # ---- convenience + introspection ------------------------------------------
    def run_all(self, *, hold: bool = False) -> bool:
        self.setup()
        self.prepare_dataset()
        self.build_configs()
        return self.run_cache(hold=hold)

    def push(self, message: str = "manual push") -> str:
        return _api.cache_push(self._need("subjects_root"), self.cfg.cache_repo,
                               token=self._need("hf_token"), commit_message=message)

    def status(self) -> dict:
        from . import cache_monitor as _cm
        dr = self.state.get("data_root")
        out: dict = {"data_root": dr, "resume": self.state.get("resume"), "cache_repo": self.cfg.cache_repo,
                     "phase_errors": self.state.get("phase_errors", {})}
        if dr and Path(dr).exists():
            du = shutil.disk_usage(dr)
            out["disk_gb"] = {"used": round(du.used / 1e9, 1), "free": round(du.free / 1e9, 1),
                              "total": round(du.total / 1e9, 1)}
        ds = self.state.get("dataset_tomls") or {}
        for name, toml in ds.items():
            try:
                roots = _cm.cache_roots_for(_cm.dataset_dirs_from_toml(toml))
                out[f"cached_{name}"] = _cm.count_done(roots)
            except Exception:  # noqa: BLE001
                pass
        print("[factory] status:", json.dumps(out, indent=2, default=str))
        return out

    # ---- internals ------------------------------------------------------------
    def _subject_cfg(self):
        mode = {"before_after": _api.CaptionMode.BEFORE_AFTER,
                "separate": _api.CaptionMode.SEPARATE,
                "mixed": _api.CaptionMode.MIXED}[self.cfg.caption_mode]
        return _api.SubjectBucketConfig(
            repo=self.cfg.source_repo, config=self.cfg.source_config,
            out_root=self.state["subjects_root"], limit=self.cfg.limit,
            require_audit_approved=True, require_age_pass=False,
            caption_mode=mode, use_semantic=True, similarity_model=self.cfg.similarity_model,
            min_bucket_size=self.cfg.min_bucket_size, min_final_group_size=self.cfg.min_final_group_size,
            keep_small=True, split_oversized=True)

    def _save_state(self) -> None:
        dr = self.state.get("data_root")
        if not dr:
            return
        try:
            blob = {"config": asdict(self.cfg),
                    "state": {k: v for k, v in self.state.items()
                              if k not in ("hf_token", "model_paths", "subject_cfg")}}
            Path(dr, "factory_state.json").write_text(json.dumps(blob, indent=2, default=str), encoding="utf-8")
        except OSError:
            pass
