#!/usr/bin/env python3
"""
sana_runner.py — the repo-side runner for Sana LoRA experiments on a Colab GPU (the RTX PRO 6000), the Sana
counterpart of trainer_runner.TrainerRunner. Same thin-shell contract: the notebook
(notebooks/sana_colab_train.ipynb) is a handful of `s.<step>()` calls and ALL logic lives here, so
iterating = `git pull`, never re-pasting cells.

Sana trains through model type 'sana' in the AbstractEyes diffusion-pipe fork. The bootstrap
(`anima_colab.install(dp_url=anima_colab.DP_FORK_URL)`) clones the fork beside upstream at
external/diffusion-pipe-fork and points ANIMA_DIFFUSION_PIPE at it; setup() re-points it after a restart.

THE SEQUENCE (run_sequence): the LoRA arms of sana_experiments.SEQUENCE, one after another on one card. Every
arm gets its own folder in the experiments repo (default AbstractPhil/geolip-beatrix-sana, see
sana_experiments.py): its README and meta.json, the config, the training-set list, EVERY saved epoch (uploaded
as it is saved), the trainer's previews, the evaluation and the log. All training sets and the no-LoRA
baseline are rendered before any LoRA is loaded; a rerun skips the arms the repo already lists as done.

DATA. The stock model renders its own training sets: images of one flavor (upbeat / downbeat / neutral)
captioned with the NEUTRAL prompt "a photo of <subject>", so a LoRA can only lower its loss by making neutral
prompts carry that flavor (the trained counterpart of steering the text conditioning; cf. Concept Sliders,
Gandikota et al. 2023, arXiv 2311.12092). A quarter of the subjects are held out of training; the evaluation
loads each saved LoRA through diffusers and scores the held-out subjects. source='folder' (single runs) trains
on any folder of images + .txt captions instead.

    from geolip_anima_trainer.sana_runner import SanaRunner
    s = SanaRunner()                       # kwargs / ANIMA_* env to tune
    s.setup(); s.run_sequence()            # or the single run: prepare_dataset / build_configs / train / evaluate
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path

from . import api as _api
from . import launch as _launch
from . import sana_experiments as sx
from .cache_factory import _RunnerMixin, get_hf_token
from .trainer_runner import _pid_alive

# ---- the built-in 'mood' set --------------------------------------------------------------------
SUBJECTS = [
    "a city street with parked cars", "a kitchen with a wooden table", "a dog sitting on a lawn",
    "a bowl of fruit on a counter", "a train at a station platform", "a woman reading a book on a bench",
    "a living room with a sofa and a lamp", "a sailboat on a lake", "a man riding a bicycle on a road",
    "a cat on a windowsill", "a market stall with vegetables", "a playground in a park",
    "a bedroom with a window", "a group of people at a restaurant table", "a mountain road with trees",
    "a beach with an umbrella", "a classroom with desks", "a house on a hill", "a child standing by a fence",
    "a coffee cup on a desk", "a bus on a city street", "a horse in a field", "a bridge over a river",
    "an old man sitting on a porch", "a garden with flowers", "a parking lot at the edge of town",
    "a pair of shoes by a door", "a lighthouse on a coast", "a bakery counter with bread",
    "a street musician playing a guitar", "a snowy forest path", "an office with computers",
]
HELD_OUT = [i for i in range(len(SUBJECTS)) if i % 4 == 3]          # 8 subjects never trained on
TRAIN = [i for i in range(len(SUBJECTS)) if i % 4 != 3]             # 24 training subjects
UPBEAT_TEMPLATES = sx.FLAVOR_TEMPLATES["up"]
NEUTRAL_TEMPLATE = "a photo of {s}"
# The mood judge: 100 x (mean CLIP cosine to the upbeat phrases - the same to the downbeat phrases).
UP_PHRASES = ["a cheerful, upbeat image", "a happy, joyful scene", "a bright, uplifting photo"]
DOWN_PHRASES = ["a gloomy, downbeat image", "a sad, melancholy scene", "a dark, depressing photo"]
CLIP_JUDGE = "openai/clip-vit-large-patch14"
# Reference effects on the stock 600M 512px model with this judge (32 subjects x 4 seeds): writing upbeat
# words into the prompt moves the score +1.81 over the neutral prompt; adding the upbeat-minus-downbeat
# direction of the text conditioning to the neutral prompt at strength 2 moves it +1.20.
REFERENCE_EFFECTS = {"upbeat_words_vs_neutral": 1.81, "text_direction_strength_2": 1.20}
GEN_STEPS, GEN_CFG = 20, 4.5          # the stock pipeline's defaults
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")


def mood_items(seeds_per_subject: int = 8, flavor: str = "up", seed_base: int = 1000) -> list[dict]:
    """A training set plan: each training subject x seed -> a prompt of `flavor` (its two templates
    alternating) captioned with the neutral prompt. Pure (no GPU)."""
    return sx.items_for(SUBJECTS, TRAIN, flavor, seeds_per_subject=seeds_per_subject, seed_base=seed_base,
                        caption_template=NEUTRAL_TEMPLATE)


def mood_outcome(diffs: list[float]) -> dict:
    """The read fixed before the first run, on paired (LoRA minus no LoRA) mood-score differences:
    FLAVOR LORA = mean > 0, >= 75% positive and mean > 3 SE; NO EFFECT = |mean| <= 2 SE or < 60%
    positive; anything else MIXED."""
    return sx.arm_outcome(diffs, 1)


def _stock_pipeline(diffusers_path: str):
    """The stock SanaPipeline at the model card's dtypes (transformer fp16; autoencoder + text encoder bf16)."""
    import torch
    from diffusers import SanaPipeline
    pipe = SanaPipeline.from_pretrained(diffusers_path, torch_dtype=torch.float16)
    pipe.to("cuda")
    pipe.vae.to(torch.bfloat16)
    pipe.text_encoder.to(torch.bfloat16)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def _generate(pipe, prompts: list[str], seeds: list[int], res: int):
    import torch
    gens = [torch.Generator(device="cuda").manual_seed(s) for s in seeds]
    with torch.no_grad():
        return pipe(prompt=prompts, height=res, width=res, guidance_scale=GEN_CFG,
                    num_inference_steps=GEN_STEPS, generator=gens).images


def _pixel_stats(img) -> dict:
    import numpy as np
    a = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    luma = 0.2126 * r + 0.7152 * g + 0.0722 * b
    mx, mn = a.max(-1), a.min(-1)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-8), 0.0)
    return {"luma": float(luma.mean()), "sat": float(sat.mean()), "warmth": float(r.mean() - b.mean()),
            "contrast": float(luma.std())}


def _drop_torchao() -> bool:
    """Uninstall torchao when present: peft refuses torchao builds older than its minimum when it builds LoRA
    layers (Colab ships 0.10; recent peft requires 0.16+), and nothing here uses torchao. The trainer runs in a
    fresh process, and peft checks lazily, so no restart is needed. Returns True when it uninstalled it."""
    import importlib.metadata as md
    try:
        version = md.version("torchao")
    except md.PackageNotFoundError:
        return False
    print(f"[setup] uninstalling torchao {version}: peft refuses it when building LoRA layers", flush=True)
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"], check=False)
    return True


def _grid(rows: list[list], path: Path, *, tile: int = 256) -> Path:
    """A contact sheet: rows of PIL images (None = blank)."""
    from PIL import Image
    ncol = max(len(r) for r in rows)
    canvas = Image.new("RGB", (tile * ncol, tile * len(rows)), "white")
    for i, r in enumerate(rows):
        for j, im in enumerate(r):
            if im is not None:
                canvas.paste(im.resize((tile, tile)), (j * tile, i * tile))
    canvas.save(path, quality=88)
    return path


def _epoch_dirs(output_dir: "str | Path") -> list[tuple[int, Path]]:
    """(N, dir) for every saved epochN of the NEWEST run under output_dir, sorted by N."""
    p = Path(output_dir)
    runs = [d for d in p.iterdir() if d.is_dir()] if p.is_dir() else []
    runs = [d for d in runs if any(d.glob("epoch*/adapter_model.safetensors"))]
    if not runs:
        return []
    run = max(runs, key=lambda d: d.stat().st_mtime)
    out = []
    for e in run.glob("epoch*"):
        if e.is_dir() and (e / "adapter_model.safetensors").is_file() and e.name[5:].isdigit():
            out.append((int(e.name[5:]), e))
    return sorted(out)


def _folder_files(folder: "str | Path", prefix: str, allow: "list[str] | None" = None) -> dict:
    """{path in the repo: local file (a Path)} for every file under folder (allow = top-level names to keep),
    skipping deepspeed state (global_step*) and caches."""
    folder = Path(folder)
    out: dict = {}
    if not folder.is_dir():
        return out
    for p in sorted(folder.rglob("*")):
        rel = p.relative_to(folder).as_posix()
        if not p.is_file() or rel.startswith(("global_step", "cache/")) or "/cache/" in rel:
            continue
        if allow is not None and rel not in allow:
            continue
        out[f"{prefix}/{rel}"] = p
    return out


def _progress(log: "str | Path") -> str:
    """A trainer's state from the tail of its log: 'step N (S samples/s)' from the trainer's 'steps: N loss: ...
    samples/sec: S' lines, 'saving' once it is done, else 'starting'."""
    try:
        with open(log, "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - 16384))
            tail = f.read().decode("utf-8", "replace")
    except OSError:
        return "starting"
    if "TRAINING COMPLETE" in tail:
        return "saving"
    hits = re.findall(r"steps: (\d+) loss: [\d.]+ .*?samples/sec: ([\d.]+)", tail)
    if hits:
        step, rate = hits[-1]
        return f"step {step} ({float(rate):.0f} samples/s)"
    return "starting"


def _free_ports(start: int = 29510):
    """Rendezvous ports nothing on this machine holds (checked by binding), one per trainer started side by side."""
    import socket
    port = start
    while port < 65535:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sk:
            try:
                sk.bind(("127.0.0.1", port))
                free = True
            except OSError:
                free = False
        if free:
            yield port
        port += 1


class _HubRepo:
    """Uploads into one HF *model* repo (the experiments repo), a few commits per experiment, each retried."""
    RETRY_S = (15, 30, 60, 120)

    def __init__(self, repo_id: str, token: "str | None"):
        if not token:
            raise RuntimeError(f"uploading to {repo_id} needs an HF token with WRITE access: add HF_TOKEN to Colab "
                               f"Secrets (and give this notebook access), then re-run setup().")
        from huggingface_hub import HfApi
        self.repo_id, self.api = repo_id, HfApi(token=token)

    def files(self) -> list[str]:
        return self.api.list_repo_files(self.repo_id, repo_type="model")

    def metas(self) -> dict:
        """Every experiments/<id>/meta.json in the repo, keyed by id."""
        from huggingface_hub import hf_hub_download
        out = {}
        for f in self.files():
            parts = f.split("/")
            if len(parts) == 3 and parts[0] == "experiments" and parts[2] == "meta.json":
                p = hf_hub_download(self.repo_id, f, repo_type="model", token=self.api.token)
                out[parts[1]] = json.loads(Path(p).read_text(encoding="utf-8"))
        return out

    def commit(self, files: dict, msg: str, *, retry: bool = True) -> None:
        """One commit adding every {path in the repo: a local file (Path) | bytes | text (str)}. Rate limits,
        server errors and dropped connections are retried; a refused token is not."""
        from huggingface_hub import CommitOperationAdd
        if not files:
            return
        ops = [CommitOperationAdd(path_in_repo=path, path_or_fileobj=str(v) if isinstance(v, Path)
                                  else v.encode("utf-8") if isinstance(v, str) else v)
               for path, v in files.items()]
        for wait in ((*self.RETRY_S, None) if retry else (None,)):
            try:
                self.api.create_commit(repo_id=self.repo_id, repo_type="model", operations=ops, commit_message=msg)
                return
            except Exception as e:  # noqa: BLE001
                code = getattr(getattr(e, "response", None), "status_code", None)
                if wait is None or code in (401, 403, 404):
                    raise
                print(f"[hub] upload '{msg}' failed ({type(e).__name__}: {str(e)[:120]}); retry in {wait} s", flush=True)
                time.sleep(wait)

    def put(self, path_in_repo: str, data: "bytes | str", msg: str, *, retry: bool = True) -> None:
        self.commit({path_in_repo: data.encode("utf-8") if isinstance(data, str) else data}, msg, retry=retry)

    def put_folder(self, folder: "str | Path", path_in_repo: str, msg: str, *, allow: "list[str] | None" = None) -> None:
        self.commit(_folder_files(folder, path_in_repo, allow), msg)


@dataclass
class SanaConfig:
    """Everything the Sana runs need, overridable from the notebook (kwargs) or env (ANIMA_*)."""
    repo_root: str = field(default_factory=lambda: os.environ.get("ANIMA_REPO", "/content/anima-trainer"))
    data_root: str | None = None                 # None -> /content/sana_data (Colab disk; the repo is the durability)
    hf_home: str | None = None                   # None -> {data_root}/hf_cache
    variant: str = "600m-512"                    # download_sana.SANA_REPOS key
    diffusers_path: str | None = None            # an existing local Sana folder (skips the download)
    repo_id: str = sx.DEFAULT_REPO               # the experiments repo (public): one folder per experiment
    # data
    source: str = "mood"                         # single runs: 'mood' (the model renders its own set) | 'folder'
    dataset_dir: str | None = None               # source='folder': images + .txt captions
    seeds_per_subject: int = 8                   # 24 training subjects x 8 = 192 images
    gen_batch: int = 16
    # recipe (the sequence's arms override lr / data per arm)
    rank: int = 32
    lr: float = 1e-4                             # the diffusers Sana LoRA example's rate
    epochs: int = 10
    micro_batch: int = 4
    warmup_steps: int = 20
    save_every_n_epochs: int = 2
    num_gpus: int = 1
    preview_prompts: list[str] | None = None     # None -> mood: 4 held-out neutral prompts; folder: none
    # evaluation: held-out subjects x these seeds; the LoRA at these scales (0 = no LoRA loaded)
    eval_seeds: list[int] = field(default_factory=lambda: [101, 202, 303, 404])
    eval_scales: list[float] = field(default_factory=lambda: [0.0, 0.5, 1.0])
    # single runs: an HF *model* repo for backup(); None -> a folder experiments/adhoc_<time>/ of repo_id
    backup_repo: str | None = None

    @classmethod
    def from_env(cls, **overrides) -> "SanaConfig":
        env: dict = {}
        if os.environ.get("ANIMA_DATA_ROOT"):
            env["data_root"] = os.environ["ANIMA_DATA_ROOT"]
        if os.environ.get("ANIMA_BACKUP_REPO"):
            env["backup_repo"] = os.environ["ANIMA_BACKUP_REPO"]
        if os.environ.get("ANIMA_SANA_VARIANT"):
            env["variant"] = os.environ["ANIMA_SANA_VARIANT"]
        if os.environ.get("ANIMA_SANA_REPO"):
            env["repo_id"] = os.environ["ANIMA_SANA_REPO"]
        valid = {f.name for f in fields(cls)}
        bad = set(overrides) - valid
        if bad:
            raise TypeError(f"unknown SanaConfig override(s): {sorted(bad)}")
        return cls(**{**env, **overrides})


class SanaRunner(_RunnerMixin):
    """Stateful Sana orchestrator (one method per notebook cell, idempotent, run in order). State is
    in-memory (mirrored to {data_root}/sana_state.json); a fresh runtime re-runs setup() first.
    The sequence machinery is shared with anima_runner.AnimaRunner: what differs per model is BED (the experiment
    registry + README text) and the generation settings below."""
    TAG = "sana"
    STATE_FILE = "sana_state.json"
    EXPECT_SM = None                             # any CUDA GPU with bf16 runs the 600M recipe
    BED = sx.SANA
    GEN_STEPS, GEN_CFG, GEN_SHIFT = GEN_STEPS, GEN_CFG, 3.0
    NEGATIVE = ""                                # the previews' negative prompt (the stock pipeline renders without one)

    def __init__(self, config: "SanaConfig | None" = None, **overrides):
        self.cfg = config or SanaConfig.from_env(**overrides)
        if self.cfg.source not in ("mood", "folder"):
            raise ValueError(f"source must be 'mood' or 'folder', got {self.cfg.source!r}")
        if self.cfg.variant not in _api._dl_sana.SANA_REPOS:
            raise ValueError(f"variant must be one of {list(_api._dl_sana.SANA_REPOS)}, got {self.cfg.variant!r}")
        self.state: dict = {}
        self._pipe = None                        # the stock pipeline, resident for renders + evaluations
        self._judge_fns = None                   # the CLIP judge, resident
        self._base = None                        # the no-LoRA renders of the held-out cells

    # ---- 1. setup: env -> (optional) auth -> gpu -> the fork -> the model ---------------
    def setup(self) -> dict:
        self._setup_env()
        _drop_torchao()
        self._auth_optional()
        self._verify_gpu()
        self._point_at_fork()
        self._download_model()
        self._save_state()
        print(f"[sana] setup done | DATA_ROOT={self.state['data_root']} | model={self.state['diffusers_path']} "
              f"| native {self.state['resolution']} px | experiments -> {self.cfg.repo_id}")
        return self.state

    def _setup_env(self) -> None:
        data_root = self.cfg.data_root or "/content/sana_data"
        hf_home = self.cfg.hf_home or f"{data_root}/hf_cache"
        os.environ["HF_HOME"] = hf_home               # before the first huggingface_hub import
        os.environ["ANIMA_DATA_ROOT"] = data_root
        for d in (hf_home, data_root):
            os.makedirs(d, exist_ok=True)
        if self.cfg.repo_root not in sys.path:
            sys.path.insert(0, self.cfg.repo_root)
        self.state.update(data_root=data_root, hf_home=hf_home)
        print(f"[sana] DATA_ROOT={data_root} | HF_HOME={hf_home}")

    def _auth_optional(self) -> None:
        """The Sana repos and the CLIP judge are public; the token is for the uploads (run_sequence / backup)."""
        token = get_hf_token()
        self.state["hf_token"] = token
        if not token:
            print(f"[sana] no HF_TOKEN: the models are public, but run_sequence() uploads to {self.cfg.repo_id} "
                  f"and needs a WRITE token in Colab Secrets")
            return
        from huggingface_hub import login, whoami
        login(token=token, add_to_git_credential=False)
        self.state["hf_user"] = whoami(token=token).get("name")
        print(f"[sana] HF user={self.state['hf_user']}")

    def _point_at_fork(self) -> str:
        """Sana lives in the AbstractEyes fork: prefer $ANIMA_DIFFUSION_PIPE, else the bootstrap's
        external/diffusion-pipe-fork; refuse a checkout without models/sana.py."""
        cands = [os.environ.get("ANIMA_DIFFUSION_PIPE", ""),
                 f"{self.cfg.repo_root}/external/diffusion-pipe-fork",
                 f"{self.cfg.repo_root}/external/diffusion-pipe"]
        for c in cands:
            if c and Path(c, "models", "sana.py").is_file():
                os.environ["ANIMA_DIFFUSION_PIPE"] = c
                self.state["diffusion_pipe"] = c
                print(f"[{self.TAG}] diffusion-pipe (the AbstractEyes fork): {c}")
                return c
        raise RuntimeError("no diffusion-pipe with models/sana.py found — run the bootstrap cell with "
                           "anima_colab.install(..., dp_url=anima_colab.DP_FORK_URL), or point "
                           "ANIMA_DIFFUSION_PIPE at an AbstractEyes diffusion-pipe checkout.")

    def _need_model(self) -> None:
        """setup() has located the model (the shared sequence code asks before any GPU work)."""
        self._need("diffusers_path")

    def _download_model(self) -> str:
        if self.cfg.diffusers_path:
            path = self.cfg.diffusers_path
        else:
            path = _api.download_sana(f"{self.state['data_root']}/models", variant=self.cfg.variant)
        from .config import _sana_native_resolution
        res = _sana_native_resolution(path) or _api._dl_sana.SANA_REPOS[self.cfg.variant][1]
        self.state.update(diffusers_path=str(path), resolution=int(res))
        return str(path)

    # ---- resident GPU pieces --------------------------------------------------------------------
    def _eval_pipe(self):
        if self._pipe is None:
            self._pipe = _stock_pipeline(self._need("diffusers_path"))
        return self._pipe

    def _render(self, prompts: list[str], seeds: list[int]) -> list:
        pipe, res, b = self._eval_pipe(), self._need("resolution"), self.cfg.gen_batch
        out = []
        for i in range(0, len(prompts), b):
            out += _generate(pipe, prompts[i:i + b], seeds[i:i + b], res)
        return out

    def _judge(self):
        if self._judge_fns is None:
            self._judge_fns = self._clip_judge()
        return self._judge_fns

    def _score(self, imgs: list):
        """(features, mood scores) for a list of images."""
        f_img, f_up, f_down = self._judge()
        feats = f_img(imgs)
        return feats, 100.0 * ((feats @ f_up.T).mean(1) - (feats @ f_down.T).mean(1))

    def _cells(self) -> tuple[list[tuple[int, int]], list[str], list[int]]:
        cells = [(si, seed) for si in HELD_OUT for seed in self.cfg.eval_seeds]
        return cells, [self.BED.neutral_caption.format(s=SUBJECTS[si]) for si, _ in cells], [s for _, s in cells]

    def _items(self, flavor: str = "up", seed_base: int = 1000) -> list[dict]:
        """A training-set plan in this bed's wording (each training scene x seed, captioned neutrally)."""
        return sx.items_for(SUBJECTS, TRAIN, flavor, seeds_per_subject=self.cfg.seeds_per_subject, seed_base=seed_base,
                            caption_template=self.BED.neutral_caption, templates=self.BED.templates)

    def _assert_stock(self) -> None:
        """No LoRA may be loaded while stock images are rendered (training sets, the baseline)."""
        pipe = self._eval_pipe()
        try:
            loaded = {k: v for k, v in pipe.get_list_adapters().items() if v}
        except Exception:  # noqa: BLE001 — no peft: nothing can be loaded
            loaded = {}
        if loaded:
            raise RuntimeError(f"a LoRA is still loaded ({loaded}); refusing to render stock images")

    # ---- 2. the training data (single runs) ---------------------------------------------------------
    def prepare_dataset(self) -> str:
        dr = self._need("data_root")
        if self.cfg.source == "folder":
            d = Path(self.cfg.dataset_dir or "")
            if not self.cfg.dataset_dir or not d.is_dir():
                raise RuntimeError(f"source='folder' needs dataset_dir= (a folder of images + .txt captions); got {d}")
            imgs = [p for p in d.iterdir() if p.suffix.lower() in IMAGE_EXTS]
            paired = [p for p in imgs if p.with_suffix(".txt").is_file()]
            if not paired:
                raise RuntimeError(f"{d} has no image with a matching .txt caption")
            if len(paired) < len(imgs):
                print(f"[sana] WARNING: {len(imgs) - len(paired)} of {len(imgs)} images have no .txt caption")
            self.state.update(dataset_dir=str(d), n_images=len(paired))
            print(f"[sana] dataset: {d} ({len(paired)} captioned images)")
            self._save_state()
            return str(d)
        self._need("resolution")
        out = self._render_dataset("up", 1000)
        self.state.update(dataset_dir=str(out), n_images=len(self._items()))
        self._save_state()
        print(f"[sana] dataset: {out} ({self.state['n_images']} upbeat images captioned neutrally; "
              f"{len(HELD_OUT)} subjects held out for evaluate())")
        return str(out)

    def _render_dataset(self, flavor: str, seed_base: int) -> Path:
        """{data_root}/datasets/<flavor>_<seed_base>/images (+ items.jsonl and sheet.jpg beside it, never
        inside the image folder the trainer scans). Skips images already rendered."""
        root = Path(self._need("data_root")) / "datasets" / f"{flavor}_{seed_base}"
        img_dir = root / "images"
        img_dir.mkdir(parents=True, exist_ok=True)
        items = self._items(flavor, seed_base)
        (root / "items.jsonl").write_text("".join(json.dumps(it) + "\n" for it in items), encoding="utf-8")
        todo = [it for it in items if not (img_dir / f"{it['name']}.png").is_file()]
        if todo:
            self._assert_stock()
            for i in range(0, len(todo), self.cfg.gen_batch):
                chunk = todo[i:i + self.cfg.gen_batch]
                for it, img in zip(chunk, self._render([c["prompt"] for c in chunk], [c["seed"] for c in chunk])):
                    img.save(img_dir / f"{it['name']}.png")
                    (img_dir / f"{it['name']}.txt").write_text(it["caption"], encoding="utf-8")
            print(f"[{self.TAG}] rendered {len(todo)} {flavor} training images (seeds {seed_base}+) -> {img_dir}", flush=True)
        if not (root / "sheet.jpg").is_file():
            from PIL import Image
            firsts = [Image.open(img_dir / f"{it['name']}.png") for it in items[::self.cfg.seeds_per_subject]]
            _grid([firsts[r * 6:(r + 1) * 6] for r in range(4)], root / "sheet.jpg", tile=192)
        return img_dir

    # ---- 3. the training config ----------------------------------------------------------------
    def _render_config(self, dataset_dir: str, output_dir: str, configs_dir: str, *, lr: float,
                       held_out_previews: bool) -> "tuple[Path, Path]":
        res = self._need("resolution")
        model = _api.sana_model(self._need("diffusers_path"))
        prompts = self.cfg.preview_prompts
        if prompts is None and held_out_previews:
            prompts = [self.BED.neutral_caption.format(s=SUBJECTS[i]) for i in HELD_OUT[:4]]
        samples = _api.SamplesConfig(prompts=list(prompts), negative_prompt=self.NEGATIVE, width=res, height=res,
                                     steps=self.GEN_STEPS, cfg=self.GEN_CFG, shift=self.GEN_SHIFT,
                                     seed=42) if prompts else None
        opt = _api.preset_optimizer(model)
        opt.lr = lr
        cfg = _api.TrainConfig(
            run=_api.RunConfig(output_dir=output_dir, epochs=self.cfg.epochs,
                               micro_batch_size_per_gpu=self.cfg.micro_batch, warmup_steps=self.cfg.warmup_steps,
                               save_every_n_epochs=self.cfg.save_every_n_epochs, eval_before_first_step=False),
            model=model, adapter=_api.AdapterConfig(rank=self.cfg.rank), optimizer=opt,
            dataset=_api.DatasetConfig(resolutions=[res], directories=[_api.DirectoryConfig(path=str(dataset_dir))]),
            samples=samples)
        return _api.render_train_toml(cfg, configs_dir)

    def build_configs(self) -> str:
        dr = self._need("data_root")
        lora, ds = self._render_config(self._need("dataset_dir"), f"{dr}/runs/sana_lora", f"{dr}/configs", lr=self.cfg.lr,
                                       held_out_previews=self.cfg.source == "mood")
        n = int(self.state.get("n_images") or 0)
        steps = -(-n // (self.cfg.micro_batch * self.cfg.num_gpus)) * self.cfg.epochs if n else None
        self.state.update(lora_toml=str(lora), dataset_toml=str(ds), output_dir=f"{dr}/runs/sana_lora")
        self._save_state()
        print(f"[sana] config: {lora}" + (f" | {n} images x {self.cfg.epochs} epochs = ~{steps} steps" if steps else ""))
        return str(lora)

    def _recipe(self, lr: float) -> dict:
        n = len(TRAIN) * self.cfg.seeds_per_subject
        spe = -(-n // (self.cfg.micro_batch * self.cfg.num_gpus))
        return {"base model": "Sana 600M 512px (stock, Apache-2.0)",
                "adapter": f"LoRA rank {self.cfg.rank} (alpha = rank) on every linear layer of the 28 transformer blocks",
                "optimizer": f"Adam, no weight decay, learning rate {lr:g}, {self.cfg.warmup_steps} linear warmup steps, then constant",
                "batch": f"{self.cfg.micro_batch} images per step",
                "length": f"{self.cfg.epochs} epochs x {spe} steps = {spe * self.cfg.epochs} steps",
                "saves": f"every {self.cfg.save_every_n_epochs} epochs, each with previews of 4 held-out prompts",
                "resolution": f"{self._need('resolution')} x {self._need('resolution')}",
                "timestep shift": "3.0 (the checkpoint's own)", "precision": "bf16 (the LoRA is saved in bf16)",
                "evaluation": f"8 held-out scenes x seeds {', '.join(map(str, self.cfg.eval_seeds))}, "
                              f"{GEN_STEPS} steps, guidance {GEN_CFG}"}

    # ---- 4. train: blocking by default (minutes), its log followed into the cell -------------------
    def train(self, *, detached: bool = False, dry_run: bool = False):
        lora = self._need("lora_toml")
        self._point_at_fork()
        log = f"{self.state['data_root']}/runs/train.log"
        if dry_run:
            return _launch.launch(_launch.build_plan(config_toml=lora, num_gpus=self.cfg.num_gpus), dry_run=True)
        if detached:                       # survives a kernel restart; watch with s.tail()
            argv = [sys.executable, "-m", "geolip_anima_trainer.cli", "train", "--config", lora,
                    "--num-gpus", str(self.cfg.num_gpus)]
            info = self._launch_detached(argv, log)
            self.state["train"] = info
            self._save_state()
            print(f"[sana] training launched DETACHED | pid={info['pid']} | watch: s.tail()")
            return info
        plan = _launch.build_plan(config_toml=lora, num_gpus=self.cfg.num_gpus)
        rc = _launch.launch(plan, log_path=log, monitor=self._follow(log))
        print(f"[sana] train finished rc={rc} | LoRA: {self.latest_lora()}")
        return rc

    @staticmethod
    def _follow(log: str, on_tick=None, tick_s: float = 60.0):
        """A launch() monitor that prints the trainer's log into the cell as it grows (a Colab cell does not
        always show a child process's own output), calling on_tick() every tick_s seconds (uploads)."""
        def monitor(proc) -> None:
            last = time.monotonic()
            try:
                with open(log, "r", encoding="utf-8", errors="replace") as f:
                    while True:
                        line = f.readline()
                        if line:
                            print(line, end="", flush=True)
                            continue
                        if proc.poll() is not None:
                            print(f.read(), end="", flush=True)
                            return
                        if on_tick is not None and time.monotonic() - last >= tick_s:
                            last = time.monotonic()
                            try:
                                on_tick()
                            except Exception as e:  # noqa: BLE001 — an upload hiccup must not stop the run
                                print(f"[sana] upload during training failed ({e}); it is retried at the end", flush=True)
                        time.sleep(1.0)
            except KeyboardInterrupt:
                print(f"\n[sana] stopped following; the trainer (pid {proc.pid}) keeps running — s.tail() to watch")
                raise
        return monitor

    def _launch_detached(self, argv, log) -> dict:
        os.makedirs(os.path.dirname(log), exist_ok=True)
        logf = open(log, "a", encoding="utf-8")
        try:
            kw = {"start_new_session": True} if os.name == "posix" else {}
            proc = subprocess.Popen(argv, stdout=logf, stderr=subprocess.STDOUT,
                                    cwd=self.cfg.repo_root, env={**os.environ}, **kw)
        finally:
            logf.close()
        return {"pid": proc.pid, "log": log, "cmd": " ".join(argv)}

    def latest_lora(self) -> "Path | None":
        out = self.state.get("output_dir")
        return _launch.latest_epoch_dir(out) if out else None

    def _training_alive(self) -> bool:
        info = self.state.get("train")
        return bool(isinstance(info, dict) and info.get("pid") and _pid_alive(info["pid"]))

    # ---- 5. THE SEQUENCE ----------------------------------------------------------------------------
    def run_sequence(self, arms: "list[str] | None" = None, *, stop_on_error: bool = True, parallel: int = 1) -> dict:
        """Run the arms of sana_experiments.SEQUENCE in order (or the named subset), each into its own folder of
        cfg.repo_id. Arms the repo already lists as done are skipped (a rerun resumes). parallel > 1 trains that many
        arms side by side on the card (each its own trainer process and port; two arms on the same training set never
        overlap, so its cache has one writer); the evaluations run here, one at a time. Returns the metas."""
        self._need_model()
        _drop_torchao()                                  # a kernel set up before this check existed
        repo = _HubRepo(self.cfg.repo_id, self.state.get("hf_token"))
        seq = self.BED.sequence
        specs = [a for a in seq if arms is None or a.id in arms]
        unknown = set(arms or []) - {a.id for a in seq}
        if unknown:
            raise ValueError(f"unknown arm(s) {sorted(unknown)}; the sequence is {[a.id for a in seq]}")
        metas = repo.metas()
        try:                                     # the write check, before any GPU minute is spent
            self._publish_index(repo, metas)
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"cannot write to {self.cfg.repo_id} ({e}); the HF_TOKEN needs WRITE access") from e
        todo = [a for a in specs if metas.get(a.id, {}).get("status") != "done"]
        print(f"[{self.TAG}] sequence -> {self.cfg.repo_id}: {len(todo)} to run "
              f"({', '.join(a.id for a in todo) or 'none'}); done already: "
              f"{', '.join(k for k, m in metas.items() if m.get('status') == 'done') or 'none'}", flush=True)
        if not todo:
            return metas
        t0 = time.time()
        for key in dict.fromkeys((a.flavor, a.seed_base) for a in todo):   # every training set, stock, first
            self._render_dataset(*key)
        self._baseline()                                                      # the shared no-LoRA cells
        print(f"[{self.TAG}] training sets + baseline ready in {time.time() - t0:.0f} s", flush=True)
        if parallel > 1:
            self._run_parallel(todo, repo, metas, parallel=parallel, stop_on_error=stop_on_error)
        else:
            for spec in todo:
                try:
                    metas[spec.id] = self._run_arm(spec, repo, metas)
                except Exception as e:  # noqa: BLE001
                    self._record_failure(spec, repo, metas, e)
                    if stop_on_error:
                        raise
                self._safe(lambda: self._publish_index(repo, metas))
        reads = sx.sequence_reads({k: m for k, m in metas.items() if m.get("status") == "done"}, self.BED.roles)
        print(f"[{self.TAG}] reads across the sequence:", json.dumps(reads, indent=1, default=float), flush=True)
        return metas

    def _record_failure(self, spec: "sx.ArmSpec", repo: "_HubRepo", metas: dict, e: Exception) -> None:
        metas[spec.id] = sx.arm_meta(spec, "failed", error=f"{type(e).__name__}: {e}"[:500])
        base = f"experiments/{spec.id}"
        files = {f"{base}/meta.json": sx.dumps(metas[spec.id]),
                 f"{base}/README.md": sx.render_arm_readme(spec, self._recipe(spec.lr), metas[spec.id], self.BED)}
        files.update(_folder_files(Path(self.state["data_root"]) / "experiments" / spec.id / "logs", f"{base}/logs"))
        self._safe(lambda: repo.commit(files, f"{spec.id}: failed"))
        self._safe(lambda: self._publish_index(repo, metas))
        print(f"[{self.TAG}] {spec.id} FAILED: {e}", flush=True)

    def _run_parallel(self, todo: list, repo: "_HubRepo", metas: dict, *, parallel: int, stop_on_error: bool) -> None:
        """Up to `parallel` trainers at once, each with its own deepspeed port; an arm waits while another arm on the same
        training set is training (one writer per cache). A finished trainer's arm is evaluated here while the others keep
        training. On a failure (stop_on_error) no new arm starts; the running ones finish and are evaluated, then it raises.
        Interrupting the cell stops every trainer it started (a rerun resumes from the arms already done)."""
        self._point_at_fork()
        queue, running, ports = list(todo), {}, _free_ports()
        error, last_status = None, 0.0
        try:
            while queue or running:
                for spec in list(queue):
                    if error is not None or len(running) >= parallel:
                        break
                    if any(r["spec"].data_key == spec.data_key for r in running.values()):
                        continue
                    queue.remove(spec)
                    try:
                        ctx = self._arm_begin(spec, repo)
                        plan = _launch.build_plan(config_toml=str(ctx["lora_toml"]), num_gpus=self.cfg.num_gpus,
                                                  master_port=next(ports))
                        proc = _launch.spawn(plan, ctx["log"])
                    except Exception as e:  # noqa: BLE001
                        self._record_failure(spec, repo, metas, e)
                        error = e if stop_on_error and error is None else error
                        continue
                    for r in running.values():
                        r["beside"].add(spec.id)
                    running[spec.id] = {"spec": spec, "ctx": ctx, "proc": proc, "t0": time.time(),
                                        "shipped": time.time(), "beside": set(running)}
                    print(f"[{self.TAG}] {spec.id} training (pid {proc.pid}); on the card now: {', '.join(running)}", flush=True)
                if error is not None and not running:
                    break
                time.sleep(5)
                for aid, r in list(running.items()):
                    rc = r["proc"].poll()
                    if rc is None:
                        if time.time() - r["shipped"] >= 300:          # long runs: saved epochs ship while training
                            r["shipped"] = time.time()
                            files = self._epoch_files(r["ctx"])
                            if files:
                                self._safe(lambda: repo.commit(files, f"{aid}: epochs saved so far"))
                        continue
                    del running[aid]
                    try:
                        if rc != 0:
                            raise RuntimeError(f"the trainer exited with code {rc} (experiments/{aid}/logs/train.log)")
                        metas[aid] = self._arm_finish(r["spec"], repo, metas, r["ctx"], time.time() - r["t0"],
                                                      beside=sorted(r["beside"]))
                    except Exception as e:  # noqa: BLE001
                        self._record_failure(r["spec"], repo, metas, e)
                        error = e if stop_on_error and error is None else error
                    self._safe(lambda: self._publish_index(repo, metas))
                if running and time.time() - last_status >= 30:
                    last_status = time.time()
                    print(f"[{self.TAG}] " + " | ".join(f"{a.split('_')[0]} {_progress(r['ctx']['log'])}"
                                                 for a, r in running.items())
                          + (f" | waiting: {', '.join(s.id.split('_')[0] for s in queue)}" if queue else ""), flush=True)
        except KeyboardInterrupt:
            for aid, r in running.items():
                r["proc"].terminate()                 # the deepspeed launcher stops its trainer on SIGTERM
                print(f"[{self.TAG}] stopped {aid} (pid {r['proc'].pid})", flush=True)
            print(f"[{self.TAG}] interrupted; run_sequence() again resumes from the arms already done", flush=True)
            raise
        if error is not None:
            raise error

    @staticmethod
    def _safe(fn) -> None:
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            print(f"[hub] (upload failed: {e})", flush=True)

    def _publish_index(self, repo: "_HubRepo", metas: dict) -> None:
        """The repo README from EVERY folder's meta.json (re-read now: other folders may have been added while
        this sequence ran), this session's newer metas winning."""
        merged = {**repo.metas(), **metas}
        done = {k: m for k, m in merged.items() if m.get("status") == "done"}
        reads = sx.sequence_reads(done, self.BED.roles)
        repo.put("README.md", sx.render_repo_readme(list(merged.values()), reads, self.BED),
                 "README: the experiment index")

    def _baseline(self) -> dict:
        """The held-out cells rendered WITHOUT a LoRA (once per session; every arm is paired against them)."""
        if self._base is None:
            self._assert_stock()
            cells, prompts, seeds = self._cells()
            imgs = self._render(prompts, seeds)
            feats, scores = self._score(imgs)
            self._base = {"images": imgs, "feats": feats, "scores": [float(x) for x in scores],
                          "pixels": [_pixel_stats(im) for im in imgs]}
            print(f"[{self.TAG}] baseline: {len(imgs)} held-out cells, mean mood score {sum(self._base['scores']) / len(imgs):+.3f}",
                  flush=True)
        return self._base

    def _arm_begin(self, spec: "sx.ArmSpec", repo: "_HubRepo") -> dict:
        """The arm's folders + config, and its first commit (README, meta 'running', config, training-set list)."""
        dr = Path(self._need("data_root"))
        arm = dr / "experiments" / spec.id
        base = f"experiments/{spec.id}"
        img_dir = dr / "datasets" / spec.data_key / "images"
        out_dir, cfg_dir, log = arm / "runs", arm / "config", arm / "logs" / "train.log"
        for d in (out_dir, cfg_dir, log.parent, arm / "eval"):
            d.mkdir(parents=True, exist_ok=True)
        recipe = self._recipe(spec.lr)
        lora_toml, _ = self._render_config(str(img_dir), str(out_dir), str(cfg_dir), lr=spec.lr, held_out_previews=True)
        print(f"\n[{self.TAG}] ===== {spec.id}: {spec.title} =====", flush=True)
        start = {f"{base}/meta.json": sx.dumps(sx.arm_meta(spec, "running", recipe=recipe)),
                 f"{base}/README.md": sx.render_arm_readme(spec, recipe, None, self.BED)}
        start.update(_folder_files(cfg_dir, f"{base}/config"))
        start.update(_folder_files(img_dir.parent, f"{base}/data", allow=["items.jsonl", "sheet.jpg"]))
        repo.commit(start, f"{spec.id}: started (README, config, training-set list)")
        return {"arm": arm, "base": base, "out_dir": out_dir, "log": log, "recipe": recipe, "lora_toml": lora_toml,
                "uploaded": set()}

    @staticmethod
    def _epoch_files(ctx: dict, *, final: bool = False) -> dict:
        """The saved epochs not uploaded yet (the newest may still be writing unless final)."""
        eps = _epoch_dirs(ctx["out_dir"])
        files: dict = {}
        for n, d in (eps if final else eps[:-1]):
            if n not in ctx["uploaded"]:
                files.update(_folder_files(d, f"{ctx['base']}/lora/epoch{n}"))
                ctx["uploaded"].add(n)
        return files

    def _run_arm(self, spec: "sx.ArmSpec", repo: "_HubRepo", metas: dict) -> dict:
        """One arm, trained in the foreground with its log followed into the cell."""
        ctx = self._arm_begin(spec, repo)

        def ship_epochs() -> None:                             # long runs: saved epochs ship while training
            files = self._epoch_files(ctx)
            if files:
                repo.commit(files, f"{spec.id}: epochs saved so far")

        self._point_at_fork()
        plan = _launch.build_plan(config_toml=str(ctx["lora_toml"]), num_gpus=self.cfg.num_gpus)
        t0 = time.time()
        _launch.launch(plan, log_path=str(ctx["log"]),
                       monitor=self._follow(str(ctx["log"]), on_tick=ship_epochs, tick_s=300.0))
        return self._arm_finish(spec, repo, metas, ctx, time.time() - t0)

    def _after_train(self, spec: "sx.ArmSpec", ctx: dict, epochs: list) -> "dict | None":
        """A bed's check between training and the evaluation (none for Sana: its renderer is the stock pipeline)."""
        return None

    def _arm_finish(self, spec: "sx.ArmSpec", repo: "_HubRepo", metas: dict, ctx: dict, train_s: float,
                    beside: "list[str] | None" = None) -> dict:
        """A trained arm: its weights + previews + log in one commit, then the evaluation and the result commit.
        beside = the arms that trained on the card at the same time (side-by-side runs; None when alone)."""
        import numpy as np
        arm, base, out_dir, log, recipe = ctx["arm"], ctx["base"], ctx["out_dir"], ctx["log"], ctx["recipe"]
        epochs = _epoch_dirs(out_dir)
        if not epochs:
            raise RuntimeError(f"training finished but saved no LoRA under {out_dir}")
        trained = self._epoch_files(ctx, final=True)            # the weights ship BEFORE the evaluation
        trained.update(_folder_files(log.parent, f"{base}/logs"))
        trained.update(_folder_files(epochs[-1][1].parent / "samples", f"{base}/samples"))
        repo.commit(trained, f"{spec.id}: LoRA epochs, previews, log")
        parity = self._after_train(spec, ctx, epochs)

        # ---- evaluation: every saved epoch at scale 1, the final one also at 0.5 ----
        cells, prompts, seeds = self._cells()
        b = self._baseline()
        pipe = self._eval_pipe()
        final_n = epochs[-1][0]
        rows, per_cell, by_epoch, final_imgs = [], {}, {}, {}
        for n, d in epochs:
            name = f"{spec.id.split('_')[0]}_ep{n}"
            pipe.load_lora_weights(str(d), weight_name="adapter_model.safetensors", adapter_name=name)
            for sc in ([0.5, 1.0] if n == final_n else [1.0]):
                pipe.set_adapters([name], adapter_weights=[sc])
                imgs = self._render(prompts, seeds)
                feats, scores = self._score(imgs)
                diffs = [float(s - s0) for s, s0 in zip(scores, b["scores"])]
                keep = [float(x) for x in (feats * b["feats"]).sum(-1)]
                px = [_pixel_stats(im) for im in imgs]
                rd = sx.arm_outcome(diffs, spec.direction)
                rows.append({"epoch": n, "scale": sc, **rd, "content_kept": float(np.mean(keep)),
                             "mood_score": float(np.mean(scores)),
                             "pixels": {k: float(np.mean([p[k] for p in px])) for k in px[0]}})
                if sc == 1.0:
                    by_epoch[n] = imgs
                if n == final_n:
                    final_imgs[sc] = imgs
                    per_cell[str(sc)] = [{"subject": SUBJECTS[si], "seed": s, "score": float(v), "diff": dd, "content_kept": k}
                                         for (si, s), v, dd, k in zip(cells, scores, diffs, keep)]
                print(f"[{self.TAG}] {spec.id} epoch {n} scale {sc}: effect {rd['mean']:+.3f} +- {rd['se']:.3f} -> "
                      f"{rd['OUTCOME']} | content kept {np.mean(keep):.3f}", flush=True)
            pipe.unload_lora_weights()
        final = next(r for r in rows if r["epoch"] == final_n and r["scale"] == 1.0)
        final_diffs = [c["diff"] for c in per_cell["1.0"]]
        first = next((r["epoch"] for r in rows if r["scale"] == 1.0 and abs(r["mean"]) > 3 * r["se"]
                      and (spec.direction == 0 or r["mean"] * spec.direction > 0)), None)

        # ---- files: eval/ + sheets ----
        ev = arm / "eval"
        baseline = {"mood_score": float(np.mean(b["scores"])),
                    "pixels": {k: float(np.mean([p[k] for p in b["pixels"]])) for k in b["pixels"][0]}}
        (ev / "final.json").write_text(json.dumps({"baseline_scores": b["scores"], "baseline": baseline,
                                                   "cells": per_cell, "read": final}, indent=1), encoding="utf-8")
        (ev / "epochs.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
        first_seed = self.cfg.eval_seeds[0]
        idx = [i for i, (_, s) in enumerate(cells) if s == first_seed]
        _grid([[b["images"][i], final_imgs[0.5][i], final_imgs[1.0][i]] for i in idx], ev / "sheet_final.jpg")
        _grid([[b["images"][i]] + [by_epoch[n][i] for n, _ in epochs] for i in idx[:4]], ev / "sheet_epochs.jpg")
        meta = sx.arm_meta(spec, "done", recipe=recipe, epochs=rows, final=final, final_diffs=final_diffs,
                           baseline=baseline, first_epoch_beyond_3se=first, train_seconds=round(train_s),
                           summary=self._summary(spec, final, metas),
                           finished_utc=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
                           **({"trained_beside": beside} if beside else {}),
                           **({"renderer_parity": parity} if parity else {}))
        done = _folder_files(ev, f"{base}/eval")
        done[f"{base}/meta.json"] = sx.dumps(meta)
        done[f"{base}/README.md"] = sx.render_arm_readme(spec, recipe, meta, self.BED)
        repo.commit(done, f"{spec.id}: evaluation + result")
        print(f"[{self.TAG}] {spec.id} done: {meta['summary']}", flush=True)
        try:
            from IPython.display import Image as _Img, display
            display(_Img(filename=str(ev / "sheet_final.jpg")))
        except Exception:  # noqa: BLE001
            pass
        return meta

    def _summary(self, spec: "sx.ArmSpec", final: dict, metas: dict) -> str:
        eff = f"{final['mean']:+.2f} +- {final['se']:.2f}"
        if spec.direction == 0:                  # read against the upbeat arm of the SAME draw of images
            ref_spec = sx.reference_arm(spec, self.BED)
            ref = metas.get(ref_spec.id, {}) if ref_spec else {}
            ref = ref.get("final") if ref.get("status") == "done" else None
            tail = (f"; {sx.control_read(final['mean'], ref['mean'])} against {ref_spec.id.split('_')[0]}"
                    if ref else "")
            return f"control: mood effect {eff} at scale 1{tail}"
        frac = final.get("frac_pos", final.get("frac_neg"))
        return f"{final['OUTCOME']}: mood effect {eff} at scale 1, {frac:.0%} of 32 held-out cells the expected way"

    # ---- 6. evaluate (single runs): load the saved LoRA through diffusers ---------------------------
    def evaluate(self, prompts: "list[str] | None" = None, *, lora_dir: "str | None" = None) -> dict:
        """mood: the held-out subjects x eval_seeds, no LoRA, then the LoRA at each eval scale.
        folder: the same for `prompts` (default: the preview prompts). Both prove the LoRA loads through
        diffusers. Writes {data_root}/eval/ (eval.json + sheet.jpg) and returns the reads."""
        import numpy as np
        if self._training_alive() and lora_dir is None:
            raise RuntimeError("training is still running (s.tail() to watch); evaluate after it finishes, "
                               "or pass lora_dir= to read an earlier epoch")
        lora = Path(lora_dir) if lora_dir else self.latest_lora()
        if lora is None or not (lora / "adapter_model.safetensors").is_file():
            raise RuntimeError(f"no saved LoRA ({lora}) — train() first (runs/sana_lora/*/epochN/adapter_model.safetensors)")
        out = Path(self._need("data_root")) / "eval"
        out.mkdir(parents=True, exist_ok=True)
        if self.cfg.source == "mood" and prompts is None:
            cells, plist, seeds = self._cells()
        else:
            plist0 = prompts or self.cfg.preview_prompts
            if not plist0:
                raise RuntimeError("evaluate() for source='folder' needs prompts= (or preview_prompts)")
            cells = [(i, seed) for i in range(len(plist0)) for seed in self.cfg.eval_seeds]
            plist, seeds = [plist0[i] for i, _ in cells], [seed for _, seed in cells]
        scales = sorted({float(s) for s in self.cfg.eval_scales if s > 0})
        if not scales:
            raise RuntimeError("eval_scales needs at least one scale above 0")

        pipe = self._eval_pipe()
        self._assert_stock()
        images = {0.0: self._render(plist, seeds)}               # no LoRA loaded: the stock model
        pipe.load_lora_weights(str(lora), weight_name="adapter_model.safetensors", adapter_name="trained")
        for sc in scales:
            pipe.set_adapters(["trained"], adapter_weights=[sc])
            images[sc] = self._render(plist, seeds)
        pipe.unload_lora_weights()

        top = scales[-1]
        px_change = max(float(np.abs(np.asarray(a, dtype=np.int16) - np.asarray(b, dtype=np.int16)).max())
                        for a, b in zip(images[0.0], images[top]))
        reads: dict = {"lora": str(lora), "lora_loads": bool(px_change > 0), "max_pixel_change": px_change}
        feats, score = {}, {}
        for tag, imgs in images.items():
            feats[tag], s = self._score(imgs)
            score[tag] = [float(x) for x in s]
        keep = {tag: [float(x) for x in (feats[tag] * feats[0.0]).sum(-1)] for tag in feats}
        stats = {tag: [_pixel_stats(im) for im in imgs] for tag, imgs in images.items()}
        reads["curve"] = {str(t): float(np.mean(v)) for t, v in sorted(score.items())}
        reads["content_kept"] = {str(t): float(np.mean(v)) for t, v in sorted(keep.items())}
        reads["pixels"] = {str(t): {k: float(np.mean([s[k] for s in v])) for k in v[0]} for t, v in sorted(stats.items())}
        diffs = [a - b for a, b in zip(score[top], score[0.0])]
        reads["read"] = {**mood_outcome(diffs), "scale": top}
        if self.cfg.source == "mood" and prompts is None:
            reads["reference_effects"] = REFERENCE_EFFECTS
        ledger = {"cells": [{"prompt": p, "seed": s} for p, s in zip(plist, seeds)],
                  "scores": {str(k): v for k, v in score.items()},
                  "content_kept": {str(k): v for k, v in keep.items()}, "reads": reads}
        (out / "eval.json").write_text(json.dumps(ledger, indent=1), encoding="utf-8")
        first_seed = cells[0][1]
        idx = [i for i, (_, s) in enumerate(cells) if s == first_seed][:6]
        sheet = _grid([[images[t][i] for t in sorted(images)] for i in idx], out / "sheet.jpg")
        self.state["eval"] = {"ledger": str(out / "eval.json"), "sheet": str(sheet), "reads": reads}
        self._save_state()
        self._print_reads(reads)
        try:                                                     # show the sheet inline in a notebook
            from IPython.display import Image as _Img, display
            display(_Img(filename=str(sheet)))
        except Exception:  # noqa: BLE001
            pass
        return reads

    def _clip_judge(self):
        import numpy as np
        import torch
        from transformers import CLIPModel, CLIPProcessor
        model = CLIPModel.from_pretrained(CLIP_JUDGE, torch_dtype=torch.float16).to("cuda").eval()
        proc = CLIPProcessor.from_pretrained(CLIP_JUDGE)

        @torch.no_grad()
        def text(phrases):
            t = proc(text=phrases, return_tensors="pt", padding=True).to("cuda")
            f = model.text_projection(model.text_model(input_ids=t["input_ids"], attention_mask=t["attention_mask"]).pooler_output)
            return torch.nn.functional.normalize(f.float(), dim=-1).cpu().numpy()

        @torch.no_grad()
        def image(imgs):
            out = []
            for i in range(0, len(imgs), 16):
                px = proc(images=imgs[i:i + 16], return_tensors="pt")["pixel_values"].to("cuda", torch.float16)
                f = model.visual_projection(model.vision_model(pixel_values=px).pooler_output)
                out.append(torch.nn.functional.normalize(f.float(), dim=-1).cpu().numpy())
            return np.concatenate(out)

        return image, text(UP_PHRASES), text(DOWN_PHRASES)

    @staticmethod
    def _print_reads(reads: dict) -> None:
        print(f"[sana] LoRA {reads['lora']} | loads through diffusers: {reads['lora_loads']} "
              f"(max pixel change {reads['max_pixel_change']:.0f})")
        print("[sana] scale -> mood score | content kept (CLIP cosine to the no-LoRA image)")
        for t in reads["curve"]:
            print(f"        {t:>4} -> {reads['curve'][t]:+.3f} | {reads['content_kept'][t]:.3f}")
        r = reads["read"]
        line = (f"[sana] effect (scale {r['scale']} minus no LoRA): {r['mean']:+.3f} +- {r['se']:.3f}, "
                f"{r['frac_pos']:.0%} of {r['n']} positive -> {r['OUTCOME']}")
        if "reference_effects" in reads:
            ref = reads["reference_effects"]
            line += (f"  (stock model, for reference: upbeat words +{ref['upbeat_words_vs_neutral']:.2f}, "
                     f"text direction at strength 2 +{ref['text_direction_strength_2']:.2f})")
        print(line)

    # ---- 7. backup / monitor / status ------------------------------------------------------------
    def backup(self) -> "str | None":
        """Single runs: push the newest LoRA epoch dir + the evaluation to backup_repo, or else to a folder
        experiments/adhoc_<source>_<UTC time>/ of the experiments repo."""
        token, lora = self.state.get("hf_token"), self.latest_lora()
        if not (token and lora):
            print("[sana] backup needs an HF_TOKEN (Colab Secrets) and a saved LoRA -> skip")
            return None
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
        repo_id = self.cfg.backup_repo or self.cfg.repo_id
        base = "" if self.cfg.backup_repo else f"experiments/adhoc_{self.cfg.source}_{stamp}/"
        if self.cfg.backup_repo:
            from huggingface_hub import create_repo
            create_repo(repo_id, token=token, repo_type="model", private=True, exist_ok=True)
        repo = _HubRepo(repo_id, token)
        rel = f"{lora.parent.name}/{lora.name}"
        files = _folder_files(lora, f"{base}lora/{lora.name}")
        files.update(_folder_files(lora.parent / "samples", f"{base}samples"))
        files.update(_folder_files(Path(self.state.get("data_root", ""), "eval"), f"{base}eval"))
        repo.commit(files, f"LoRA + previews + evaluation :: {rel}")
        print(f"[sana] backed up -> https://huggingface.co/{repo_id}/tree/main/{base}")
        return base or rel

    def tail(self, n: int = 40) -> None:
        log = Path(self.state.get("data_root", ""), "runs", "train.log")
        if not log.is_file():
            print("[sana] no train log yet")
            return
        try:
            print(subprocess.run(["tail", "-n", str(n), str(log)], capture_output=True, text=True).stdout)
        except Exception:  # noqa: BLE001 — no `tail` -> read the byte tail
            print(log.read_text(encoding="utf-8", errors="replace")[-4000:])

    def status(self) -> dict:
        dr = self.state.get("data_root")
        out: dict = {"data_root": dr, "model": self.state.get("diffusers_path"), "repo": self.cfg.repo_id,
                     "dataset": self.state.get("dataset_dir"), "n_images": self.state.get("n_images"),
                     "config": self.state.get("lora_toml"), "latest_lora": str(self.latest_lora() or ""),
                     "eval": (self.state.get("eval") or {}).get("reads", {}).get("read")}
        if isinstance(self.state.get("train"), dict):
            out["train_alive"] = self._training_alive()
        if dr and Path(dr).exists():
            out["disk_free_gb"] = round(shutil.disk_usage(dr).free / 1e9, 1)
        print("[sana] status:", json.dumps(out, indent=2, default=str))
        return out

    def run_all(self) -> dict:
        """The single run: setup -> prepare_dataset -> build_configs -> train (blocking) -> evaluate -> backup."""
        self.setup()
        self.prepare_dataset()
        self.build_configs()
        self.train()
        reads = self.evaluate()
        self.backup()
        return reads
