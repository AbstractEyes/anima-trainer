#!/usr/bin/env python3
"""
sana_runner.py — the repo-side runner for a Sana LoRA run on a Colab GPU (the RTX PRO 6000), the Sana
counterpart of trainer_runner.TrainerRunner. Same thin-shell contract: the notebook
(notebooks/sana_colab_train.ipynb) is a handful of `s.<step>()` calls and ALL logic lives here, so
iterating = `git pull`, never re-pasting cells.

Sana trains through model type 'sana' in the AbstractEyes diffusion-pipe fork. The bootstrap
(`anima_colab.install(dp_url=anima_colab.DP_FORK_URL)`) clones the fork beside upstream at
external/diffusion-pipe-fork and points ANIMA_DIFFUSION_PIPE at it; setup() re-points it after a restart.

DATA. source='mood' (the default) needs no dataset: the stock model renders its own training set,
upbeat images captioned with the NEUTRAL prompt "a photo of <subject>", so the LoRA can only lower its
loss by making neutral prompts look upbeat. That is the trained counterpart of steering the text
conditioning toward a mood (cf. Concept Sliders, Gandikota et al. 2023, arXiv 2311.12092). A quarter of
the subjects are held out of training, and evaluate() measures the effect on them, loading the saved
LoRA through diffusers (which also proves the LoRA file loads). source='folder' trains on any folder of
images + .txt captions instead.

    from geolip_anima_trainer.sana_runner import SanaRunner
    s = SanaRunner()                       # kwargs / ANIMA_* env to tune
    s.setup(); s.prepare_dataset(); s.build_configs(); s.train(); s.evaluate(); s.backup()
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field, fields
from pathlib import Path

from . import api as _api
from . import launch as _launch
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
UPBEAT_TEMPLATES = ["a cheerful, upbeat photo of {s}", "{s}, joyful and uplifting mood"]
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


def mood_items(seeds_per_subject: int = 8) -> list[dict]:
    """The training set plan for source='mood': each training subject x seed -> an upbeat prompt
    (the two templates alternating) captioned with the neutral prompt. Pure (no GPU)."""
    items = []
    for si in TRAIN:
        s = SUBJECTS[si]
        for k in range(seeds_per_subject):
            items.append({"name": f"{si:02d}_{k:02d}", "seed": 1000 + k,
                          "prompt": UPBEAT_TEMPLATES[k % 2].format(s=s),
                          "caption": NEUTRAL_TEMPLATE.format(s=s)})
    return items


def mood_outcome(diffs: list[float]) -> dict:
    """The read fixed before the first run, on paired (LoRA minus no LoRA) mood-score differences:
    FLAVOR LORA = mean > 0, >= 75% positive and mean > 3 SE; NO EFFECT = |mean| <= 2 SE or < 60%
    positive; anything else MIXED."""
    import statistics
    n = len(diffs)
    mean = statistics.fmean(diffs) if n else 0.0
    se = (statistics.stdev(diffs) / n ** 0.5) if n > 1 else 0.0
    frac = sum(d > 0 for d in diffs) / n if n else 0.0
    if mean > 0 and frac >= 0.75 and mean > 3 * se:
        verdict = "FLAVOR LORA"
    elif abs(mean) <= 2 * se or frac < 0.60:
        verdict = "NO EFFECT"
    else:
        verdict = "MIXED"
    return {"mean": mean, "se": se, "frac_pos": frac, "n": n, "OUTCOME": verdict}


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


@dataclass
class SanaConfig:
    """Everything the Sana run needs, overridable from the notebook (kwargs) or env (ANIMA_*)."""
    repo_root: str = field(default_factory=lambda: os.environ.get("ANIMA_REPO", "/content/anima-trainer"))
    data_root: str | None = None                 # None -> /content/sana_data (the run is short; Colab disk)
    hf_home: str | None = None                   # None -> {data_root}/hf_cache
    variant: str = "600m-512"                    # download_sana.SANA_REPOS key
    diffusers_path: str | None = None            # an existing local Sana folder (skips the download)
    # data
    source: str = "mood"                         # 'mood' (the model renders its own set) | 'folder'
    dataset_dir: str | None = None               # source='folder': images + .txt captions
    seeds_per_subject: int = 8                   # mood: 24 training subjects x 8 = 192 images
    gen_batch: int = 8
    # recipe
    rank: int = 32
    lr: float = 1e-4                             # the diffusers Sana LoRA example's rate
    epochs: int = 10
    micro_batch: int = 4
    warmup_steps: int = 20
    save_every_n_epochs: int = 2
    num_gpus: int = 1
    preview_prompts: list[str] | None = None     # None -> mood: 4 held-out neutral prompts; folder: none
    # evaluation (mood): held-out subjects x these seeds; the LoRA at these scales (0 = no LoRA loaded)
    eval_seeds: list[int] = field(default_factory=lambda: [101, 202, 303, 404])
    eval_scales: list[float] = field(default_factory=lambda: [0.0, 0.5, 1.0])
    # HF *model* repo (private) for the LoRA + the evaluation; None -> {hf_user}/sana-mood-lora
    # (source='folder': {hf_user}/sana-lora) when a token is present
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
        valid = {f.name for f in fields(cls)}
        bad = set(overrides) - valid
        if bad:
            raise TypeError(f"unknown SanaConfig override(s): {sorted(bad)}")
        return cls(**{**env, **overrides})


class SanaRunner(_RunnerMixin):
    """Stateful Sana orchestrator (one method per notebook cell, idempotent, run in order). State is
    in-memory (mirrored to {data_root}/sana_state.json); a fresh runtime re-runs setup() first."""
    TAG = "sana"
    STATE_FILE = "sana_state.json"
    EXPECT_SM = None                             # any CUDA GPU with bf16 runs the 600M recipe

    def __init__(self, config: "SanaConfig | None" = None, **overrides):
        self.cfg = config or SanaConfig.from_env(**overrides)
        if self.cfg.source not in ("mood", "folder"):
            raise ValueError(f"source must be 'mood' or 'folder', got {self.cfg.source!r}")
        if self.cfg.variant not in _api._dl_sana.SANA_REPOS:
            raise ValueError(f"variant must be one of {list(_api._dl_sana.SANA_REPOS)}, got {self.cfg.variant!r}")
        self.state: dict = {}

    # ---- 1. setup: env -> (optional) auth -> gpu -> the fork -> the model ---------------
    def setup(self) -> dict:
        self._setup_env()
        self._auth_optional()
        self._verify_gpu()
        self._point_at_fork()
        self._download_model()
        self._save_state()
        print(f"[sana] setup done | DATA_ROOT={self.state['data_root']} | model={self.state['diffusers_path']} "
              f"| native {self.state['resolution']} px")
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
        """The Sana repos and the CLIP judge are public: a token is only needed for backup()."""
        token = get_hf_token()
        self.state["hf_token"] = token
        if not token:
            print("[sana] no HF_TOKEN (fine: the models are public; backup() needs one)")
            return
        from huggingface_hub import login, whoami
        login(token=token, add_to_git_credential=False)
        user = whoami(token=token).get("name")
        self.state["hf_user"] = user
        if not self.cfg.backup_repo and user:
            self.cfg.backup_repo = f"{user}/sana-mood-lora" if self.cfg.source == "mood" else f"{user}/sana-lora"
        print(f"[sana] HF user={user} | backup_repo={self.cfg.backup_repo} (private)")

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
                print(f"[sana] diffusion-pipe with Sana: {c}")
                return c
        raise RuntimeError("no diffusion-pipe with models/sana.py found — run the bootstrap cell with "
                           "anima_colab.install(..., dp_url=anima_colab.DP_FORK_URL), or point "
                           "ANIMA_DIFFUSION_PIPE at an AbstractEyes diffusion-pipe checkout.")

    def _download_model(self) -> str:
        if self.cfg.diffusers_path:
            path = self.cfg.diffusers_path
        else:
            path = _api.download_sana(f"{self.state['data_root']}/models", variant=self.cfg.variant)
        from .config import _sana_native_resolution
        res = _sana_native_resolution(path) or _api._dl_sana.SANA_REPOS[self.cfg.variant][1]
        self.state.update(diffusers_path=str(path), resolution=int(res))
        return str(path)

    # ---- 2. the training data -------------------------------------------------------------
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

        res = self._need("resolution")
        out = Path(dr) / "datasets" / "mood_upbeat"
        out.mkdir(parents=True, exist_ok=True)
        items = mood_items(self.cfg.seeds_per_subject)
        todo = [it for it in items if not (out / f"{it['name']}.png").is_file()]
        if todo:
            pipe = _stock_pipeline(self._need("diffusers_path"))
            for i in range(0, len(todo), self.cfg.gen_batch):
                chunk = todo[i:i + self.cfg.gen_batch]
                imgs = _generate(pipe, [c["prompt"] for c in chunk], [c["seed"] for c in chunk], res)
                for it, img in zip(chunk, imgs):
                    img.save(out / f"{it['name']}.png")
                    (out / f"{it['name']}.txt").write_text(it["caption"], encoding="utf-8")
                print(f"[sana] rendered {min(i + self.cfg.gen_batch, len(todo))}/{len(todo)} training images", flush=True)
            del pipe
            self._free_cuda()
        else:
            print(f"[sana] all {len(items)} training images already rendered (skip)")
        self.state.update(dataset_dir=str(out), n_images=len(items))
        self._save_state()
        print(f"[sana] dataset: {out} ({len(items)} upbeat images captioned neutrally; "
              f"{len(HELD_OUT)} subjects held out for evaluate())")
        return str(out)

    # ---- 3. the training config ----------------------------------------------------------------
    def build_configs(self) -> str:
        dr = self._need("data_root")
        res = self._need("resolution")
        model = _api.sana_model(self._need("diffusers_path"))
        prompts = self.cfg.preview_prompts
        if prompts is None and self.cfg.source == "mood":
            prompts = [NEUTRAL_TEMPLATE.format(s=SUBJECTS[i]) for i in HELD_OUT[:4]]
        samples = _api.SamplesConfig(prompts=list(prompts), negative_prompt="", width=res, height=res,
                                     steps=GEN_STEPS, cfg=GEN_CFG, shift=3.0, seed=42) if prompts else None
        opt = _api.preset_optimizer(model)
        opt.lr = self.cfg.lr
        cfg = _api.TrainConfig(
            run=_api.RunConfig(output_dir=f"{dr}/runs/sana_lora", epochs=self.cfg.epochs,
                               micro_batch_size_per_gpu=self.cfg.micro_batch, warmup_steps=self.cfg.warmup_steps,
                               save_every_n_epochs=self.cfg.save_every_n_epochs, eval_before_first_step=False),
            model=model, adapter=_api.AdapterConfig(rank=self.cfg.rank), optimizer=opt,
            dataset=_api.DatasetConfig(resolutions=[res],
                                       directories=[_api.DirectoryConfig(path=self._need("dataset_dir"))]),
            samples=samples)
        lora, ds = _api.render_train_toml(cfg, f"{dr}/configs")
        n = int(self.state.get("n_images") or 0)
        steps = -(-n // (self.cfg.micro_batch * self.cfg.num_gpus)) * self.cfg.epochs if n else None
        self.state.update(lora_toml=str(lora), dataset_toml=str(ds), output_dir=cfg.run.output_dir)
        self._save_state()
        print(f"[sana] config: {lora}" + (f" | {n} images x {self.cfg.epochs} epochs = ~{steps} steps" if steps else ""))
        return str(lora)

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
    def _follow(log: str):
        """A launch() monitor that prints the trainer's log into the cell as it grows (a Colab cell
        does not always show a child process's own output)."""
        def monitor(proc) -> None:
            try:
                with open(log, "r", encoding="utf-8", errors="replace") as f:
                    while True:
                        line = f.readline()
                        if line:
                            print(line, end="", flush=True)
                        elif proc.poll() is not None:
                            print(f.read(), end="", flush=True)
                            return
                        else:
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

    # ---- 5. evaluate: load the saved LoRA through diffusers ------------------------------------
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
        res = self._need("resolution")
        out = Path(self._need("data_root")) / "eval"
        out.mkdir(parents=True, exist_ok=True)
        if self.cfg.source == "mood" and prompts is None:
            cells = [(si, seed) for si in HELD_OUT for seed in self.cfg.eval_seeds]
            plist = [NEUTRAL_TEMPLATE.format(s=SUBJECTS[si]) for si, _ in cells]
        else:
            plist0 = prompts or self.cfg.preview_prompts
            if not plist0:
                raise RuntimeError("evaluate() for source='folder' needs prompts= (or preview_prompts)")
            cells = [(i, seed) for i in range(len(plist0)) for seed in self.cfg.eval_seeds]
            plist = [plist0[i] for i, _ in cells]
        seeds = [seed for _, seed in cells]
        scales = sorted({float(s) for s in self.cfg.eval_scales if s > 0})
        if not scales:
            raise RuntimeError("eval_scales needs at least one scale above 0")

        pipe = _stock_pipeline(self._need("diffusers_path"))
        images: dict = {}

        def _render(tag: float) -> None:
            imgs = []
            for i in range(0, len(plist), self.cfg.gen_batch):
                imgs += _generate(pipe, plist[i:i + self.cfg.gen_batch], seeds[i:i + self.cfg.gen_batch], res)
            images[tag] = imgs

        _render(0.0)                                             # no LoRA loaded: the stock model
        pipe.load_lora_weights(str(lora), weight_name="adapter_model.safetensors", adapter_name="trained")
        for sc in scales:
            pipe.set_adapters(["trained"], adapter_weights=[sc])
            _render(sc)
        del pipe
        self._free_cuda()

        top = scales[-1]
        px_change = max(float(np.abs(np.asarray(a, dtype=np.int16) - np.asarray(b, dtype=np.int16)).max())
                        for a, b in zip(images[0.0], images[top]))
        reads: dict = {"lora": str(lora), "lora_loads": bool(px_change > 0), "max_pixel_change": px_change}

        f_img, f_up, f_down = self._clip_judge()
        feats = {tag: f_img(imgs) for tag, imgs in images.items()}
        score = {tag: (100.0 * ((f @ f_up.T).mean(1) - (f @ f_down.T).mean(1))).tolist() for tag, f in feats.items()}
        keep = {tag: (feats[tag] * feats[0.0]).sum(-1).tolist() for tag in feats}
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
        sheet = self._sheet(images, cells, out / "sheet.jpg")
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
    def _sheet(images: dict, cells, path: Path, *, rows: int = 6, tile: int = 256) -> Path:
        """Rows = the first `rows` prompts at the first eval seed; columns = no LoRA, then each scale."""
        from PIL import Image
        first_seed = cells[0][1]
        idx = [i for i, (_, seed) in enumerate(cells) if seed == first_seed][:rows]
        tags = sorted(images)
        canvas = Image.new("RGB", (tile * len(tags), tile * len(idx)), "white")
        for r, i in enumerate(idx):
            for c, tag in enumerate(tags):
                canvas.paste(images[tag][i].resize((tile, tile)), (c * tile, r * tile))
        canvas.save(path, quality=88)
        return path

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

    @staticmethod
    def _free_cuda() -> None:
        import gc
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass

    # ---- 6. backup / monitor / status ------------------------------------------------------------
    def backup(self) -> "str | None":
        """Push the newest LoRA epoch dir + the evaluation to the (private) HF *model* backup_repo."""
        repo, token = self.cfg.backup_repo, self.state.get("hf_token")
        lora = self.latest_lora()
        if not (repo and token and lora):
            print("[sana] backup needs an HF_TOKEN (Colab Secrets), a backup_repo and a saved LoRA -> skip")
            return None
        from huggingface_hub import HfApi, create_repo
        api = HfApi(token=token)
        create_repo(repo, token=token, repo_type="model", private=True, exist_ok=True)
        rel = f"{lora.parent.name}/{lora.name}"
        api.upload_folder(folder_path=str(lora), repo_id=repo, repo_type="model", path_in_repo=f"runs/{rel}",
                          ignore_patterns=["global_step*/*", "global_step*/**"], commit_message=f"LoRA :: {rel}")
        ev = Path(self.state.get("data_root", ""), "eval")
        if ev.is_dir():
            api.upload_folder(folder_path=str(ev), repo_id=repo, repo_type="model", path_in_repo=f"runs/{rel}/eval",
                              commit_message=f"evaluation :: {rel}")
        base = None if self.cfg.diffusers_path else _api._dl_sana.SANA_REPOS[self.cfg.variant][0]
        card = ("---\nlicense: apache-2.0\n" + (f"base_model: {base}\n" if base else "")
                + "tags: [sana, lora, diffusers]\n---\n# Sana LoRA (trained with diffusion-pipe)\n\n"
                "A LoRA for the Sana transformer, saved in diffusers format:\n\n```python\n"
                "pipe.load_lora_weights(folder, weight_name='adapter_model.safetensors')\n```\n\n"
                "Sana's weights are Apache-2.0; its Gemma-2-2B-IT text encoder is under the Gemma Terms of Use.\n")
        api.upload_file(path_or_fileobj=card.encode("utf-8"), path_in_repo="README.md", repo_id=repo,
                        repo_type="model", commit_message="model card")
        print(f"[sana] backed up -> https://huggingface.co/{repo}/tree/main/runs/{rel}")
        return rel

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
        out: dict = {"data_root": dr, "model": self.state.get("diffusers_path"),
                     "dataset": self.state.get("dataset_dir"), "n_images": self.state.get("n_images"),
                     "config": self.state.get("lora_toml"), "latest_lora": str(self.latest_lora() or ""),
                     "backup_repo": self.cfg.backup_repo,
                     "eval": (self.state.get("eval") or {}).get("reads", {}).get("read")}
        if isinstance(self.state.get("train"), dict):
            out["train_alive"] = self._training_alive()
        if dr and Path(dr).exists():
            out["disk_free_gb"] = round(shutil.disk_usage(dr).free / 1e9, 1)
        print("[sana] status:", json.dumps(out, indent=2, default=str))
        return out

    def run_all(self) -> dict:
        """setup -> prepare_dataset -> build_configs -> train (blocking) -> evaluate -> backup."""
        self.setup()
        self.prepare_dataset()
        self.build_configs()
        self.train()
        reads = self.evaluate()
        self.backup()
        return reads
