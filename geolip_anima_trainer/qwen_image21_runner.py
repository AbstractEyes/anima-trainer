#!/usr/bin/env python3
"""
qwen_image21_runner.py — the repo-side runner for Qwen-Image-2.1 text-to-image LoRAs on a Colab GPU (the RTX PRO
6000), behind notebooks/qwen_image21_mega_liminal.ipynb. Same thin-shell contract as the other runners: the notebook
is a few `q.<step>()` calls and all logic lives here, so iterating = `git pull`, never re-pasting cells.

2.1 trains through model type 'qwen_image21' in the AbstractEyes diffusion-pipe fork, from the release the bootstrap
checks out (anima_colab.QWEN21_FORK_REF, cloned at external/diffusion-pipe-fork-<ref>).

    q = QwenImage21Runner()
    q.setup()             # env, HF login, GPU, the fork, the three model files of Comfy-Org/Qwen-Image-2.1 (~24 GB)
    q.parity()            # the fork's parity tool on this card: the trainer computes what ComfyUI computes (stops on FAIL)
    q.prepare_dataset()   # AbstractPhil/mega-liminal: ten class folders, dampened repeats
    q.screen()            # short runs of the plain loss and the guided loss at w = 3, 4, 5, read on ordinary prompts
    q.train()             # the full run with the screen's pick (or cfg=w / cfg=None); every save ships as it appears
    q.sheets()            # contact sheets of the previews per save

Qwen-Image-2.1 is CFG-distilled: plain training pulls the conditional output toward the unguided one, so a LoRA
trained that way needs real CFG at inference. The guided loss (`cfg = w`, the fork's distillation-preserving loss)
trains against the model's own guided output instead. The strength baked into 2.1 is unpublished; screen() measures.

Uploads go to the release repo (default AbstractPhil/mega-liminal-lora, folder qwen_image21/) from a background thread
with retries; a failed upload never stops a run, and a rerun ships only what is missing.

Licences: Qwen-Image-2.1 is under the Qwen Research License (research and evaluation; derivatives display "Built with
Qwen"); mega-liminal's images stay with their creators (research and non-commercial use).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import queue
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field, fields
from pathlib import Path

from . import api as _api
from . import launch as _launch
from .cache_factory import _RunnerMixin, get_hf_token
from .sana_runner import IMAGE_EXTS, SanaRunner, _drop_torchao, _grid, _HubRepo, _hms

# The fork release these runs install; anima_colab.QWEN21_FORK_REF is the bootstrap's copy of this name.
FORK_REF = "qwen-image21-v1"
MODEL_REPO = _api._dl_qwen21.REPO
BASE_REPO = "Qwen/Qwen-Image-2.1"     # the original release: its LICENSE ships beside every LoRA
# The attribution the Qwen Research License (section 3c) requires in a NOTICE file of every redistribution.
QWEN_NOTICE = ("Qwen is licensed under the Qwen RESEARCH LICENSE AGREEMENT, Copyright (c) 2026 Hangzhou Tongyi "
               "Laboratory Technology Co., Ltd. All Rights Reserved.")
DATASET_REPO = "AbstractPhil/mega-liminal"
RELEASE_REPO = "AbstractPhil/mega-liminal-lora"
DINO = "facebook/dinov2-base"     # the picture-distance model of the screen's reads (Apache-2.0)

# Previews sample as ComfyUI samples the 2.1 base: euler / simple, 25 steps (its template), cfg 1 (the base is
# distilled), on the model's own time shift mu = 0.69 (comfy/supported_models.py QwenImage21), which the previews'
# schedule takes as shift = exp(mu); the fork's tests check the two schedules agree.
SAMPLE_STEPS = 25
SAMPLE_CFG = 1.0
SAMPLE_SHIFT = round(math.exp(0.69), 4)
SAMPLE_SEED = 42

# The 15 showcase prompts of the mega-liminal Anima LoRA, in the training-caption shape (the last three short).
SHOWCASE = [
    "liminal, empty mall, A photograph of a deserted two-level shopping mall at night, taken from the upper walkway. A dry fountain sits in the center of the atrium under a glass skylight, surrounded by potted palms and white floor tiles. Storefronts behind lowered metal grilles line both levels, lit by cold fluorescent strips. The palette is teal, cream and sodium orange.",
    "liminal, parking garage, A photograph taken from the middle of an underground parking garage level with no cars in it. Concrete pillars with yellow and black striped bases recede toward a green exit sign. Fluorescent tubes run in long rows across the low ceiling, and the painted floor shows faded parking lines and oil stains. The light is flat and greenish white.",
    "liminal, suburban, A photograph of a quiet suburban cul-de-sac at dusk, taken from the middle of the street. Two-story houses with dark windows face a circle of asphalt, a single streetlight has just turned on, and a basketball hoop stands over one driveway. The sky is a deep blue gradient with a thin band of pink above the rooftops.",
    "liminal, movie theatre, A photograph of an empty movie theater auditorium, taken from the back row. Rows of red velvet seats slope down toward a blank white screen, and dim sconces along the side walls throw soft amber light onto patterned carpet. The ceiling is dark, and a thin beam of light crosses the room from the projection booth.",
    "liminal, vanishing point, A photograph of a long, narrow hotel corridor with identical doors on both sides, taken at eye level from one end. Patterned burgundy carpet and beige wallpaper run straight to a single window at the far end. Brass wall lamps are spaced evenly between the doors, and the light is warm and dim.",
    "liminal, mega liminal, A photograph of an indoor swimming pool with no one in it, lined with small white and pale blue tiles. Arched openings along one wall lead to darker rooms, and daylight falls from a skylight above, making bright rippled patterns on the water. The walls, floor and steps are all tiled the same way.",
    "liminal, landscape, A photograph of a two-lane highway running straight across a flat grassy plain under an overcast sky. A lone billboard with a faded blank panel stands on the right shoulder, and power lines follow the road to the horizon. The colors are muted greens and grays.",
    "liminal, cityscape, A photograph of a downtown street at four in the morning after rain, taken from the center line. Tall office buildings with a few lit windows line both sides, traffic lights glow red over the empty intersection, and the wet asphalt reflects the signals. The palette is dark blue with red and white highlights.",
    "liminal, simulacrum, A 3D render of a large office floor with rows of empty cubicles under a low ceiling of fluorescent panels. The carpet is gray-blue, the partition walls are beige fabric, and the computer monitors on the desks are dark. The view runs diagonally across the room toward a wall of frosted windows.",
    "liminal, hand sourced, A photograph of an elementary school hallway during summer break, taken from low to the floor. Pale yellow lockers line one side, a row of windows lets bright afternoon light in on the other, and the waxed linoleum floor reflects it. The classroom doors are closed, and a trash can stands beside a water fountain.",
    "liminal, landscape, A photograph of an empty playground on a foggy morning. A red slide and a metal swing set stand on wood chips in the middle ground, and the trees behind them fade into white fog. The light is soft and gray, and the grass is wet.",
    "liminal, mega liminal, A photograph of the stairwell of a concrete apartment building, looking down through the gap between the railings. Green handrails spiral down several floors under ceiling lights, and each landing has a small window of frosted glass. The walls are painted in two tones, pale green above and dark green below.",
    "liminal, mega liminal, An empty hallway.",
    "liminal, landscape, A field at dusk.",
    "liminal, empty mall, A photograph.",
]

# Ordinary prompts for the screen: 32 PartiPrompts (Yu et al. 2022, github.com/google-research/parti, Apache-2.0),
# chosen before any run by a fixed rule: per category a fixed count (People 4, Animals 4, Artifacts 3, Food &
# Beverage 3, Vehicles 3, Outdoor Scenes 3, Indoor Scenes 2, Produce & Plants 2, Arts 2, Illustrations 2, World
# Knowledge 2, Abstract 2), evenly spaced through the category's prompts of 4-40 words with a plain challenge (Basic,
# Simple / Fine-grained Detail, Style & Format, Perspective; any but Writing & Symbols where too few), leaving out
# prompts that name children or describe empty or abandoned places (the LoRA's own subject).
DRIFT_PROMPTS = [
    "a woman with long black hair and dark skin",
    "a politician speaking to a large crowd",
    "a man standing under a tree",
    "a drawing of a bust of homer",
    "a high-quality photograph of an armadillo playing a bagpipe while standing on one leg",
    "a mouse sitting next to a computer mouse",
    "a panda eating bamboo",
    "A soft beam of light shines down on an armored granite wombat warrior statue holding a broad sword. The statue stands an ornate pedestal in the cella of a temple. wide-angle lens. anime oil painting.",
    "a toy car in front of a teddy bear",
    "a teddy bear on a skateboard",
    "a fire hydrant with graffiti on it",
    "an old-fashioned cocktail next to a napkin",
    "a close-up of a margarita",
    "a glass of orange juice",
    "a sunken ship at the bottom of the ocean",
    "a Harley-Davidson motorcycle with a flame decal",
    "a red sports car on the road",
    "a wooden deck overlooking a mountain valley",
    "an aerial photo of a sandy island in the ocean",
    "a mountain with a cloud hanging over it",
    "a living room with a large Egyptian statue in the corner",
    "a coffee table with a magazine on it",
    "a cross-section view of a walnut",
    "a flower with large yellow petals",
    "an oil painting of a cat playing checkers",
    "an abstract painting of three triangles in blue, red and white",
    "a painting of black and white vertical stripes",
    "a sketch of a train",
    "the city of London on Mars",
    "A photo of an Athenian vase with a painting of pangolins playing basketball in the style of Egyptian hieroglyphics",
    "The collision of two black holes in the center of a galaxy.",
    "Oil painting generated by artificial intelligence",
]
DOMAIN_PROMPTS = SHOWCASE[:8]    # one per class, the LoRA's own subject

# The screen's arms: the plain loss and the guided loss at three strengths around 4 (the 2025 Qwen-Image samples with
# true CFG 4; upstream diffusion-pipe's example of this loss uses cfg = 4).
SCREEN_ARMS = (None, 3.0, 4.0, 5.0)          # cfg per arm: None = the plain loss, w = the guided loss at strength w
PICK_RULE = ("An arm is eligible when its median change on ordinary prompts (same seed, before vs after training) stays "
             "under the median change between two seeds of the same prompt before training, and its pictures keep at "
             "least 90% of the base's contrast on ordinary prompts. Of the eligible arms, the one that changes its own "
             "subject most per unit of change on ordinary prompts is picked. No eligible arm: not ready. The control: "
             "the untrained model rendered again by each later arm's trainer must differ from the first arm's renders "
             "by less than the seed-to-seed change; if it does not, the measurement itself failed and nothing is picked.")


def arm_name(cfg: "float | None") -> str:
    """'plain' for the plain loss, 'w4' for the guided loss at strength 4."""
    return "plain" if cfg is None else f"w{cfg:g}"


def save_label(name: str) -> str:
    """The trainer's save name as the LoRA files carry it: epoch2 -> e02, step400 -> s400."""
    m = re.fullmatch(r"(epoch|step)(\d+)", name)
    if not m:
        return name
    return f"e{int(m.group(2)):02d}" if m.group(1) == "epoch" else f"s{int(m.group(2))}"


def class_folders(root: "str | Path") -> list[dict]:
    """{name, path, images} for each sub-folder of root holding images with matching .txt captions."""
    root = Path(root)
    out = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()) if root.is_dir() else []:
        n = sum(1 for f in d.iterdir() if f.suffix.lower() in IMAGE_EXTS and f.with_suffix(".txt").is_file())
        if n:
            out.append({"name": d.name, "path": str(d), "images": n})
    return out


def newest_run_dir(output_dir: "str | Path") -> "Path | None":
    """The trainer's newest run folder (<output_dir>/<timestamp>/) that has saves or previews."""
    p = Path(output_dir)
    runs = [d for d in p.iterdir() if d.is_dir() and (any(d.glob("*/adapter_model.safetensors"))
                                                      or (d / "samples").is_dir())] if p.is_dir() else []
    return max(runs, key=lambda d: d.stat().st_mtime) if runs else None


def saved_dirs(run: Path) -> list[tuple[str, Path]]:
    """(name, folder) of every saved LoRA of a run (epochN / stepN), oldest first."""
    out = [(d.name, d) for d in run.iterdir() if d.is_dir() and re.fullmatch(r"(epoch|step)\d+", d.name)
           and (d / "adapter_model.safetensors").is_file()]
    return sorted(out, key=lambda x: (x[1].stat().st_mtime, x[0]))


def preview_path(folder: Path, index: int) -> "Path | None":
    """The trainer's preview file of prompt `index` (NN_<slug>.png), or None when there is none (the trainer skips a
    preview that fails to render instead of stopping)."""
    hits = sorted(folder.glob(f"{index:02d}_*.png"))
    return hits[0] if hits else None


_PARITY_ROW = re.compile(r"^\[(PASS|FAIL|INFO|SKIP)\] ([^:]+): (.*)$")


def parse_parity(text: str) -> list[dict]:
    """The rows the fork's parity tool prints as '[STATUS] check: detail'."""
    return [{"status": m.group(1), "check": m.group(2), "detail": m.group(3)}
            for m in (_PARITY_ROW.match(line) for line in text.splitlines()) if m]


def _median(xs) -> float:
    xs = sorted(float(x) for x in xs)
    n = len(xs)
    return (xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2) if n else float("nan")


def screen_reads(base, lora, contrast_base, contrast_lora, *, n: int, m: int, first=None) -> dict:
    """One arm's reads. base / lora: rows of unit-length picture embeddings of the same prompts and seeds before and
    after training (rows 0..n-1 the ordinary prompts, n..2n-1 the same prompts at other seeds, 2n..2n+m-1 the LoRA's own
    subject; None where a preview is missing, which drops that pair); contrast_*: each picture's luminance spread;
    first: the first arm's before-training rows (the same prompts and seeds rendered again by another trainer = the
    repeat, the control that must stay under the seed-to-seed change). Distances are 1 - cosine."""
    def dist(u, v):
        return None if u is None or v is None else 1.0 - sum(x * y for x, y in zip(u, v))

    def present(xs):
        return [x for x in xs if x is not None]
    moved = [dist(base[i], lora[i]) for i in range(2 * n)]
    seeds = [dist(base[i], base[n + i]) for i in range(n)]
    own = [dist(base[2 * n + i], lora[2 * n + i]) for i in range(m)]
    pairs = [(mv, seeds[i % n]) for i, mv in enumerate(moved) if mv is not None and seeds[i % n] is not None]
    kept = [cl / cb for cl, cb in zip(contrast_lora[:2 * n], contrast_base[:2 * n]) if cl is not None and cb]
    out = {
        "drift": _median(present(moved)),             # ordinary prompts: before vs after training, same seed
        "spread": _median(present(seeds)),            # ordinary prompts: one seed vs another, before training
        "domain": _median(present(own)),              # the LoRA's own subject: before vs after training
        "contrast": _median(kept),                    # after / before luminance spread, ordinary prompts
        "inside": sum(mv < sd for mv, sd in pairs) / len(pairs) if pairs else float("nan"),
        "repeat": _median(present(dist(b, f) for b, f in zip(base, first))) if first is not None else None,
        "missing": sum(x is None for x in (*base, *lora)),
    }
    out["selectivity"] = out["domain"] / max(out["drift"], 1e-6)
    return out


def _eligible(r: dict) -> bool:
    return r["drift"] < r["spread"] and r["contrast"] >= 0.9


def screen_pick(results: dict) -> dict:
    """PICK_RULE over {arm: reads (with 'cfg')}."""
    broken = [k for k, r in results.items() if r.get("repeat") is not None and not r["repeat"] < r["spread"]]
    if broken:
        return {"arm": None, "cfg": None, "verdict": "MEASUREMENT FAILED", "control_failed": broken}
    eligible = {k: r for k, r in results.items() if _eligible(r)}
    if not eligible:
        return {"arm": None, "cfg": None, "verdict": "NOT READY"}
    best = max(eligible, key=lambda k: eligible[k]["selectivity"])
    return {"arm": best, "cfg": results[best]["cfg"], "verdict": "PICK"}


def screen_table(results: dict, pick: dict) -> str:
    lines = ["| arm | loss | ordinary prompts changed | seed-to-seed change | untrained model rendered again | "
             "own subject changed | contrast kept | ordinary pictures under their seed change | eligible |",
             "|---|---|---|---|---|---|---|---|---|"]
    for k, r in results.items():
        loss = "plain" if r["cfg"] is None else f"guided, w = {r['cfg']:g}"
        mark = " (picked)" if k == pick.get("arm") else ""
        repeat = "first arm" if r.get("repeat") is None else f"{r['repeat']:.4f}"
        lines.append(f"| {k}{mark} | {loss} | {r['drift']:.4f} | {r['spread']:.4f} | {repeat} | {r['domain']:.4f} | "
                     f"{r['contrast']:.0%} | {r['inside']:.0%} | {'yes' if _eligible(r) else 'no'} |")
    return "\n".join(lines)


class _Shipper:
    """Uploads into the release repo from one background thread. Each submit() is one commit, tried again after a
    doubling wait (15 s up to 10 min, eight attempts); a commit that still fails is logged and left unshipped, never
    raised. Shipped files are remembered by size in a record file, so a rerun ships only what is new or has changed."""
    WAITS = (15, 30, 60, 120, 240, 480, 600, 600)

    def __init__(self, repo, record_path: "str | Path", *, tag: str = "qwen21", sleep=time.sleep):
        self.repo, self.record_path, self.tag, self.sleep = repo, Path(record_path), tag, sleep
        self.shipped: dict = (json.loads(self.record_path.read_text(encoding="utf-8"))
                              if self.record_path.is_file() else {})
        self.failed: dict = {}
        self.queued: set = set()                  # paths waiting in the queue (a tick does not queue them twice)
        self.lock = threading.Lock()
        self.q: queue.Queue = queue.Queue()
        threading.Thread(target=self._work, daemon=True).start()

    def pending(self, files: dict) -> dict:
        """The entries of {path in repo: local file} not shipped yet at their current size."""
        with self.lock:
            return {p: Path(f) for p, f in files.items()
                    if Path(f).is_file() and self.shipped.get(p) != Path(f).stat().st_size}

    def submit(self, files: dict, msg: str) -> None:
        todo = {p: f for p, f in self.pending(files).items() if p not in self.queued}
        if todo:
            with self.lock:
                self.queued.update(todo)
            self.q.put((todo, msg))

    def _work(self) -> None:
        while True:
            todo, msg = self.q.get()
            try:
                self._commit(todo, msg)
            except Exception as e:  # noqa: BLE001 — the worker never dies
                print(f"[{self.tag}] ship '{msg}' failed unexpectedly ({type(e).__name__}: {e})", flush=True)
            finally:
                with self.lock:
                    self.queued.difference_update(todo)
                self.q.task_done()

    def _commit(self, todo: dict, msg: str) -> None:
        todo = self.pending(todo)
        if not todo:
            return
        sizes = {p: f.stat().st_size for p, f in todo.items()}
        for attempt, wait in enumerate((*self.WAITS, None), 1):
            try:
                self.repo.commit(todo, msg, retry=False)
            except Exception as e:  # noqa: BLE001
                code = getattr(getattr(e, "response", None), "status_code", None)
                if wait is None or code in (401, 403, 404):
                    with self.lock:
                        self.failed.update(todo)
                    print(f"[{self.tag}] ship '{msg}' gave up after {attempt} attempt(s) ({type(e).__name__}: "
                          f"{str(e)[:160]}); {len(todo)} file(s) stay unshipped until the next ship", flush=True)
                    return
                print(f"[{self.tag}] ship '{msg}' failed ({type(e).__name__}); retry in {wait} s", flush=True)
                self.sleep(wait)
                continue
            with self.lock:
                self.shipped.update(sizes)
                for p in todo:
                    self.failed.pop(p, None)
                tmp = self.record_path.with_name(self.record_path.name + ".part")
                tmp.write_text(json.dumps(self.shipped, indent=1, sort_keys=True), encoding="utf-8")
                os.replace(tmp, self.record_path)
            print(f"[{self.tag}] shipped {len(todo)} file(s): {msg}", flush=True)
            return

    def drain(self, timeout: "float | None" = None) -> list[str]:
        """Wait (up to timeout seconds) for the queued commits; returns the paths still unshipped."""
        t0 = time.time()
        while self.q.unfinished_tasks and (timeout is None or time.time() - t0 < timeout):
            time.sleep(0.2)
        with self.lock:
            return sorted(self.failed)


@dataclass
class QwenImage21Config:
    """Everything the 2.1 runs need, overridable from the notebook (kwargs) or env (ANIMA_*)."""
    repo_root: str = field(default_factory=lambda: os.environ.get("ANIMA_REPO", "/content/anima-trainer"))
    data_root: str | None = None                 # None -> /content/qwen_image21_data
    hf_home: str | None = None                   # None -> {data_root}/hf_cache
    models_dir: str | None = None                # None -> {data_root}/models/qwen_image_2.1 (the repo's folder layout)
    model_revision: str | None = None            # a commit of Comfy-Org/Qwen-Image-2.1; None = the newest (recorded)
    dataset_repo: str = DATASET_REPO
    dataset_revision: str | None = None          # None = the newest (recorded)
    release_repo: str | None = RELEASE_REPO      # every LoRA, preview and read ships here; None = nothing ships
    release_dir: str = "qwen_image21"
    name: str = "mega-liminal"                   # the LoRA files' name: <name>-qwen-image21-r<rank>-<arm>-<save>
    caption_format: str | None = "liminal, <class>, <description>"   # the README's prompt advice; None = omitted
    # The recipe: DiffSynth-Studio's own 2.1 LoRA example (rank 32 at 1e-4, 1 MP), plain Adam without weight decay.
    rank: int = 32
    lr: float = 1e-4
    epochs: int = 4                              # x 2,789 samples at micro-batch 4 = ~2,790 steps
    micro_batch: int = 4
    warmup_steps: int = 100
    save_every_n_epochs: int = 1
    resolution: int = 1024
    activation_checkpointing: bool = True        # the 7B transformer at 1 MP and micro-batch 4
    exclude_modules: list[str] | None = None     # ['img_mlp.gate_up', 'img_mlp.out'] = an attention-only LoRA
    num_gpus: int = 1
    preview_prompts: list[str] | None = None     # None -> SHOWCASE
    preview_size: int = 1024
    # the screen: one short run per arm, read on DRIFT_PROMPTS x 2 seeds + DOMAIN_PROMPTS
    screen_steps: int = 400
    screen_size: int = 768

    @classmethod
    def from_env(cls, **overrides) -> "QwenImage21Config":
        env: dict = {}
        if os.environ.get("ANIMA_DATA_ROOT"):
            env["data_root"] = os.environ["ANIMA_DATA_ROOT"]
        if os.environ.get("ANIMA_QWEN21_RELEASE_REPO"):
            env["release_repo"] = os.environ["ANIMA_QWEN21_RELEASE_REPO"]
        valid = {f.name for f in fields(cls)}
        bad = set(overrides) - valid
        if bad:
            raise TypeError(f"unknown QwenImage21Config override(s): {sorted(bad)}")
        return cls(**{**env, **overrides})


def has_qwen21(fork: "str | Path") -> bool:
    """A diffusion-pipe checkout with model type 'qwen_image21' and a ComfyUI submodule that has the 2.1 model."""
    d = Path(fork)
    return (d / "models" / "qwen_image21.py").is_file() and (d / "submodules" / "ComfyUI" / "comfy" / "ldm"
                                                             / "qwen_image21").is_dir()


_PICK = object()


class QwenImage21Runner(_RunnerMixin):
    """Stateful 2.1 orchestrator (one method per notebook cell, idempotent, run in order). State is in-memory
    (mirrored to {data_root}/qwen21_state.json); a fresh runtime re-runs setup() first (the downloads are cached)."""
    TAG = "qwen21"
    STATE_FILE = "qwen21_state.json"
    EXPECT_SM = None

    def __init__(self, config: "QwenImage21Config | None" = None, **overrides):
        self.cfg = config or QwenImage21Config.from_env(**overrides)
        self.state: dict = {}
        self._shipper: "_Shipper | None" = None
        self._shipper_off = False
        self._dino = None

    # ---- 1. setup: env -> auth -> gpu -> the fork -> the model ---------------------------------------
    def setup(self) -> dict:
        self._setup_env()
        _drop_torchao()
        self._auth_optional()
        self._verify_gpu()
        self._point_at_fork()
        self._download_model()
        self._save_state()
        print(f"[{self.TAG}] setup done | DATA_ROOT={self.state['data_root']} | models={self.state['models_dir']} "
              f"(Comfy-Org/Qwen-Image-2.1 @ {str(self.state.get('model_revision'))[:10]}) | uploads -> "
              f"{self.cfg.release_repo or 'none'}/{self.cfg.release_dir}")
        return self.state

    def _setup_env(self) -> None:
        data_root = self.cfg.data_root or "/content/qwen_image21_data"
        hf_home = self.cfg.hf_home or f"{data_root}/hf_cache"
        os.environ["HF_HOME"] = hf_home               # before the first huggingface_hub import
        os.environ["ANIMA_DATA_ROOT"] = data_root
        for d in (hf_home, data_root):
            os.makedirs(d, exist_ok=True)
        if self.cfg.repo_root not in sys.path:
            sys.path.insert(0, self.cfg.repo_root)
        self.state.update(data_root=data_root, hf_home=hf_home)
        print(f"[{self.TAG}] DATA_ROOT={data_root} | HF_HOME={hf_home}")

    def _auth_optional(self) -> None:
        """The model and the dataset are public; the token (WRITE) is for the uploads."""
        token = get_hf_token()
        self.state["hf_token"] = token
        if not token:
            print(f"[{self.TAG}] no HF_TOKEN: downloads work, but screen() and train() upload to "
                  f"{self.cfg.release_repo} and need a WRITE token in Colab Secrets (or release_repo=None)")
            return
        from huggingface_hub import login, whoami
        login(token=token, add_to_git_credential=False)
        self.state["hf_user"] = whoami(token=token).get("name")
        print(f"[{self.TAG}] HF user={self.state['hf_user']}")

    def _point_at_fork(self) -> str:
        """Prefer $ANIMA_DIFFUSION_PIPE, else the bootstrap's clone of the fork release; refuse a checkout without
        model type 'qwen_image21' or without a ComfyUI tree that has the 2.1 model."""
        cands = [os.environ.get("ANIMA_DIFFUSION_PIPE", ""),
                 f"{self.cfg.repo_root}/external/diffusion-pipe-fork-{FORK_REF}",
                 f"{self.cfg.repo_root}/external/diffusion-pipe-fork"]
        for c in cands:
            if c and has_qwen21(c):
                os.environ["ANIMA_DIFFUSION_PIPE"] = c
                self.state["diffusion_pipe"] = c
                print(f"[{self.TAG}] diffusion-pipe (the AbstractEyes fork): {c}")
                return c
        raise RuntimeError("no diffusion-pipe with models/qwen_image21.py and its ComfyUI submodule (comfy/ldm/"
                           "qwen_image21) found — run the bootstrap cell with anima_colab.install(..., dp_url="
                           "anima_colab.DP_FORK_URL, dp_ref=anima_colab.QWEN21_FORK_REF), or point "
                           "ANIMA_DIFFUSION_PIPE at such a checkout.")

    def _download_model(self) -> dict:
        dest = self.cfg.models_dir or f"{self.state['data_root']}/models/qwen_image_2.1"
        print(f"[{self.TAG}] model files from {MODEL_REPO} -> {dest} (14.2 + 0.7 + 9.4 GB the first time)")
        paths = _api.download_qwen_image21(dest, revision=self.cfg.model_revision)
        self.state.update(models_dir=str(dest), model_paths={k: str(v) for k, v in paths.items()},
                          model_revision=self._revision(MODEL_REPO, "model", self.cfg.model_revision))
        return paths

    def _revision(self, repo: str, repo_type: str, revision: "str | None") -> "str | None":
        """The commit a download resolved to (recorded with every run)."""
        try:
            from huggingface_hub import HfApi
            api = HfApi(token=self.state.get("hf_token"))
            info = api.model_info(repo, revision=revision) if repo_type == "model" else \
                api.dataset_info(repo, revision=revision)
            return info.sha
        except Exception as e:  # noqa: BLE001
            print(f"[{self.TAG}] could not read the commit of {repo} ({type(e).__name__}); not recorded")
            return None

    # ---- 2. the parity check ------------------------------------------------------------------------
    def parity(self) -> dict:
        """The fork's tools/qwen_image21_parity.py on this card: the trainer's text encoder, VAE, transformer forward,
        padding, preview sampler and LoRA step against ComfyUI's own. Prints as it goes; raises on any FAIL."""
        fork = Path(self._need("diffusion_pipe"))
        out = Path(self._need("data_root")) / "checks"
        out.mkdir(parents=True, exist_ok=True)
        log, picture = out / "parity.log", out / "parity_preview.png"
        env = {**os.environ, "DIFFUSION_PIPE_DIR": str(fork), "QWEN_IMAGE21_DIR": str(self._need("models_dir")),
               "QWEN_IMAGE21_SAMPLE_OUT": str(picture), "PYTHONUNBUFFERED": "1"}
        print(f"[{self.TAG}] parity check: loads the text encoder, the VAE, then the transformer", flush=True)
        t0 = time.time()
        with open(log, "w", encoding="utf-8") as f:
            proc = subprocess.Popen([sys.executable, "tools/qwen_image21_parity.py"], cwd=str(fork), env=env,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in proc.stdout:
                print(line, end="", flush=True)
                f.write(line)
            rc = proc.wait()
        rows = parse_parity(log.read_text(encoding="utf-8"))
        fails = [r["check"] for r in rows if r["status"] == "FAIL"]
        result = {"rc": rc, "rows": rows, "seconds": round(time.time() - t0), "fork": FORK_REF,
                  "model_revision": self.state.get("model_revision"), "gpu": self.state.get("gpu")}
        (out / "parity.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        self.state["parity"] = {"rc": rc, "fails": fails, "passes": sum(r["status"] == "PASS" for r in rows)}
        self._save_state()
        self._ship({f"{self.cfg.release_dir}/checks/{p.name}": p for p in (log, out / "parity.json", picture)
                    if p.is_file()}, f"parity check on {(self.state.get('gpu') or {}).get('name')}")
        if rc != 0 or fails or not rows:
            raise RuntimeError(f"parity check failed (rc {rc}; FAIL: {fails or 'none parsed'}); see {log}")
        print(f"[{self.TAG}] parity check passed: {self.state['parity']['passes']} PASS in {_hms(result['seconds'])}"
              f" | preview picture: {picture}")
        return result

    # ---- 3. the dataset -----------------------------------------------------------------------------
    def prepare_dataset(self) -> list[dict]:
        """Download the dataset (images + .txt captions in one folder per class) and set dampened repeats per class
        (the package's balancer: alpha 0.5, at most 8 per image, at most 1.25 x the largest class)."""
        from huggingface_hub import snapshot_download
        local = Path(self._need("data_root")) / "datasets" / self.cfg.dataset_repo.split("/")[-1]
        print(f"[{self.TAG}] dataset {self.cfg.dataset_repo} -> {local}", flush=True)
        snapshot_download(self.cfg.dataset_repo, repo_type="dataset", revision=self.cfg.dataset_revision,
                          local_dir=str(local), allow_patterns=["data/*", "README.md"],
                          token=self.state.get("hf_token"))
        found = class_folders(local / "data")
        if not found:
            raise RuntimeError(f"{local / 'data'} has no class folder of captioned images")
        ds = _api.DatasetConfig(directories=[_api.DirectoryConfig(path=c["path"], name=c["name"], image_count=c["images"],
                                                                  caption_count=c["images"]) for c in found])
        _api.rebalance(ds)
        rows = [{"name": d.name, "path": d.path, "images": d.image_count, "repeats": d.num_repeats}
                for d in ds.directories]
        self.state.update(dataset_dir=str(local), directories=rows,
                          samples_per_epoch=sum(r["images"] * r["repeats"] for r in rows),
                          dataset_revision=self._revision(self.cfg.dataset_repo, "dataset", self.cfg.dataset_revision))
        self._save_state()
        print(f"[{self.TAG}] {len(rows)} classes, {sum(r['images'] for r in rows)} images, "
              f"{self.state['samples_per_epoch']} samples an epoch:")
        for r in rows:
            print(f"    {r['name']:<18} {r['images']:>5} images x {r['repeats']} = {r['images'] * r['repeats']:>5}")
        return rows

    # ---- 4. configs ---------------------------------------------------------------------------------
    def _steps_per_epoch(self) -> int:
        return -(-int(self._need("samples_per_epoch")) // (self.cfg.micro_batch * self.cfg.num_gpus))

    def build_configs(self, tag: str = "run", *, cfg: "float | None" = None, epochs: "int | None" = None,
                      max_steps: "int | None" = None, prompts: "list[str] | None" = None,
                      size: "int | None" = None) -> str:
        """The lora + dataset tomls of one run into {data_root}/configs/<tag>/; its output under {data_root}/runs/<tag>."""
        dr = Path(self._need("data_root"))
        model = _api.qwen_image21_model(self._need("model_paths"), cfg=cfg)
        opt = _api.preset_optimizer(model)
        opt.lr = self.cfg.lr
        epochs = epochs or self.cfg.epochs
        prompts = list(prompts if prompts is not None else (self.cfg.preview_prompts or SHOWCASE))
        size = size or self.cfg.preview_size
        run = _api.RunConfig(output_dir=str(dr / "runs" / tag), epochs=epochs, micro_batch_size_per_gpu=self.cfg.micro_batch,
                             warmup_steps=self.cfg.warmup_steps, eval_before_first_step=False,
                             save_every_n_epochs=epochs if max_steps else self.cfg.save_every_n_epochs,
                             max_steps=max_steps, activation_checkpointing=self.cfg.activation_checkpointing)
        ds = _api.DatasetConfig(resolutions=[self.cfg.resolution], directories=[
            _api.DirectoryConfig(path=d["path"], num_repeats=d["repeats"], name=d["name"], image_count=d["images"])
            for d in self._need("directories")])
        samples = _api.SamplesConfig(prompts=prompts, width=size, height=size, steps=SAMPLE_STEPS, cfg=SAMPLE_CFG,
                                     shift=SAMPLE_SHIFT, seed=SAMPLE_SEED)
        train_cfg = _api.TrainConfig(run=run, model=model, optimizer=opt, dataset=ds, samples=samples,
                                     adapter=_api.AdapterConfig(rank=self.cfg.rank, exclude_modules=self.cfg.exclude_modules))
        lora, ds_path = _api.render_train_toml(train_cfg, dr / "configs" / tag)
        steps = max_steps or self._steps_per_epoch() * epochs
        self.state.setdefault("runs", {})[tag] = {"lora_toml": str(lora), "dataset_toml": str(ds_path),
                                                  "output_dir": run.output_dir, "cfg": cfg, "steps": steps,
                                                  "log": str(dr / "runs" / f"{tag}.log")}
        self._save_state()
        print(f"[{self.TAG}] {tag}: {lora} | {'plain loss' if cfg is None else f'guided loss, w = {cfg:g}'} | "
              f"rank {self.cfg.rank}, lr {self.cfg.lr:g} | ~{steps} steps | {len(prompts)} previews at {size} px")
        return str(lora)

    # ---- the shipper --------------------------------------------------------------------------------
    def _get_shipper(self) -> "_Shipper | None":
        if self._shipper is not None or self._shipper_off or not self.cfg.release_repo:
            return self._shipper
        token = self.state.get("hf_token") or get_hf_token()
        if not token:
            self._shipper_off = True
            print(f"[{self.TAG}] no HF_TOKEN: nothing ships; the files stay under {self.state.get('data_root')}")
            return None
        slug = self.cfg.release_repo.replace("/", "__")
        self._shipper = _Shipper(_HubRepo(self.cfg.release_repo, token),
                                 Path(self._need("data_root")) / f"shipped_{slug}.json", tag=self.TAG)
        return self._shipper

    def _ship(self, files: dict, msg: str) -> None:
        shipper = self._get_shipper()
        if shipper is not None:
            shipper.submit(files, msg)

    def _drain(self, timeout: float = 1800.0) -> list[str]:
        if self._shipper is None:
            return []
        left = self._shipper.drain(timeout)
        if left:
            print(f"[{self.TAG}] {len(left)} file(s) still unshipped (the next ship retries them): {left[:5]}")
        return left

    def _write_check(self) -> None:
        """Before any GPU minute of a run: write the release folder's README (licence, files, prompts, recipe) with the
        copy of the Qwen licence and the NOTICE its redistribution terms require. A token without write access fails
        here, not hours into a run. A release repo that does not exist yet is created private."""
        if not self.cfg.release_repo:
            return
        token = self.state.get("hf_token") or get_hf_token()
        if not token:
            raise RuntimeError(f"uploading to {self.cfg.release_repo} needs an HF token with WRITE access: add HF_TOKEN to "
                               f"Colab Secrets (and give this notebook access) and re-run setup(), or pass "
                               f"release_repo=None to keep everything on this disk.")
        from huggingface_hub import HfApi, hf_hub_download
        licence = Path(hf_hub_download(BASE_REPO, "LICENSE", token=token)).read_text(encoding="utf-8")
        HfApi(token=token).create_repo(self.cfg.release_repo, repo_type="model", private=True, exist_ok=True)
        d = self.cfg.release_dir
        _HubRepo(self.cfg.release_repo, token).commit(
            {f"{d}/README.md": self.folder_readme(), f"{d}/LICENSE": licence, f"{d}/NOTICE": self.folder_notice()},
            "qwen_image21: the folder's README, the Qwen licence and its notice")
        print(f"[{self.TAG}] write check: {self.cfg.release_repo}/{d}/ README.md, LICENSE and NOTICE updated")

    def _ship_run(self, tag: str, prefix: str, *, final: bool) -> None:
        """Ship a run's saves and previews. While training, a save (and its previews) counts as complete once the next
        save exists, the step-0 previews once the first save exists; at the end, everything (plus the configs, the log,
        the TensorBoard events and the done mark)."""
        info = self.state["runs"][tag]
        run = newest_run_dir(info["output_dir"])
        if run is None:
            return
        arm = arm_name(info["cfg"])
        stem = f"{self.cfg.name}-qwen-image21-r{self.cfg.rank}-{arm}"
        saves = saved_dirs(run)
        ready = saves if final else saves[:-1]
        files: dict = {}
        for name, d in ready:
            files[f"{prefix}/{stem}-{save_label(name)}.safetensors"] = d / "adapter_model.safetensors"
            if (d / "adapter_config.json").is_file():
                files[f"{prefix}/adapter_config.json"] = d / "adapter_config.json"
        done_names = {"step0", *(n for n, _ in ready)}
        if (run / "samples").is_dir():
            for s in sorted(p for p in (run / "samples").iterdir() if p.is_dir()):
                if final or (saves and s.name in done_names):
                    for p in sorted(s.iterdir()):
                        if p.is_file():
                            files[f"{prefix}/samples/{arm}/{s.name}/{p.name}"] = p
        if final:
            out = Path(info["output_dir"])
            for p in (Path(info["lora_toml"]), Path(info["dataset_toml"]), Path(info["log"]), out / "DONE.json",
                      *sorted(run.glob("events.out.tfevents*"))):
                if p.is_file():
                    files[f"{prefix}/training/{arm}/{p.name}"] = p
        if files:
            self._ship(files, f"{tag}: {len(ready)} save(s)" + (" (final)" if final else ""))

    # ---- one run ------------------------------------------------------------------------------------
    def _run(self, tag: str, *, cfg: "float | None", prefix: str, epochs: "int | None" = None,
             max_steps: "int | None" = None, prompts: "list[str] | None" = None, size: "int | None" = None) -> dict:
        lora = self.build_configs(tag, cfg=cfg, epochs=epochs, max_steps=max_steps, prompts=prompts, size=size)
        info = self.state["runs"][tag]
        done = Path(info["output_dir"]) / "DONE.json"
        if done.is_file():
            print(f"[{self.TAG}] {tag}: already trained ({done}); shipping anything missing")
        else:
            self._point_at_fork()
            plan = _launch.build_plan(config_toml=lora, num_gpus=self.cfg.num_gpus)
            t0 = time.time()
            follow = SanaRunner._follow(info["log"], on_tick=lambda: self._ship_run(tag, prefix, final=False),
                                        tag=self.TAG, label=tag, total_steps=info["steps"])
            rc = _launch.launch(plan, log_path=info["log"], monitor=follow, check=False)
            if rc != 0:
                self._ship_run(tag, prefix, final=False)
                raise RuntimeError(f"{tag}: the trainer exited with rc {rc}; see {info['log']}")
            done.write_text(json.dumps({"rc": rc, "seconds": round(time.time() - t0), "steps": info["steps"],
                                        "cfg": cfg, "fork": FORK_REF, "model_revision": self.state.get("model_revision"),
                                        "dataset_revision": self.state.get("dataset_revision")}, indent=2),
                            encoding="utf-8")      # the done mark comes before the ship
        self._ship_run(tag, prefix, final=True)
        return info

    # ---- 5. the screen ------------------------------------------------------------------------------
    def screen_setting(self) -> dict:
        """What a screen arm's reads depend on; reads shipped under another setting are not reused."""
        c = self.cfg
        prompts = "\n".join(DRIFT_PROMPTS + DOMAIN_PROMPTS).encode("utf-8")
        return {"steps": c.screen_steps, "size": c.screen_size, "prompts": hashlib.sha1(prompts).hexdigest()[:12],
                "rank": c.rank, "lr": c.lr, "micro_batch": c.micro_batch, "resolution": c.resolution,
                "exclude_modules": c.exclude_modules, "dataset": c.dataset_repo}

    def screen(self, arms: "list | tuple | None" = None) -> dict:
        """One short run per arm (SCREEN_ARMS: the plain loss and the guided loss at w = 3, 4, 5), each previewing
        the ordinary prompts at two seeds and eight of the LoRA's own prompts before and after training; then the
        reads (picture distances with DINOv2) and the pick by PICK_RULE. An arm already read under the same setting
        (here, or by an earlier runtime: its reads on the release repo) is not trained again."""
        arms = list(arms if arms is not None else SCREEN_ARMS)
        self._need("directories")
        self._need("model_paths")
        self._write_check()
        n, m = len(DRIFT_PROMPTS), len(DOMAIN_PROMPTS)
        prompts = DRIFT_PROMPTS + DRIFT_PROMPTS + DOMAIN_PROMPTS
        prefix = f"{self.cfg.release_dir}/screen"
        setting = self.screen_setting()
        earlier = self._hub_screen_reads(prefix, setting)
        results: dict = {}
        first = None                              # the first arm's untrained renders: the control's reference
        t0 = time.time()
        print(f"[{self.TAG}] screen: {len(arms)} arms x {self.cfg.screen_steps} steps, {len(prompts)} previews at "
              f"{self.cfg.screen_size} px before and after each" + (f"; read earlier: {sorted(earlier)}" if earlier else ""))
        epochs = -(-self.cfg.screen_steps // self._steps_per_epoch())      # enough epochs to reach screen_steps
        for k, w in enumerate(arms):
            name = arm_name(w)
            tag = f"screen_{name}"
            if name in earlier and not (Path(self._need("data_root")) / "runs" / tag / "DONE.json").is_file():
                results[name] = earlier[name]
                print(f"[{self.TAG}] screen {name}: read by an earlier runtime ({self.cfg.release_repo}); not trained again")
                continue
            if first is None and k > 0:
                folder = self._hub_previews(prefix, arm_name(arms[0]), "step0")
                first = self._embed([preview_path(folder, i) for i in range(2 * n + m)]) if folder else None
            self._run(tag, cfg=w, prefix=prefix, epochs=epochs, max_steps=self.cfg.screen_steps, prompts=prompts,
                      size=self.cfg.screen_size)
            run = newest_run_dir(self.state["runs"][tag]["output_dir"])
            saves = saved_dirs(run) if run else []
            if not saves:
                raise RuntimeError(f"{tag}: the trainer finished without a saved LoRA under "
                                   f"{self.state['runs'][tag]['output_dir']}; delete that folder to train the arm again")
            before, after = run / "samples" / "step0", run / "samples" / saves[-1][0]
            pb = [preview_path(before, i) for i in range(2 * n + m)]
            pa = [preview_path(after, i) for i in range(2 * n + m)]
            emb_b, emb_a = self._embed(pb), self._embed(pa)
            self._free_dino()                     # the card goes back to the next arm's trainer whole
            reads = screen_reads(emb_b, emb_a, _contrast(pb), _contrast(pa), n=n, m=m, first=first)
            if first is None and k == 0:
                first = emb_b
            results[name] = {"cfg": w, **reads, "setting": setting}
            (run / "reads.json").write_text(json.dumps(results[name], indent=2), encoding="utf-8")
            sheet = _grid([[_open(p) for p in pb[:8]], [_open(p) for p in pa[:8]],
                           [_open(p) for p in pb[2 * n:]], [_open(p) for p in pa[2 * n:]]], run / "compare.jpg", tile=256)
            self._ship({f"{prefix}/reads/{name}.json": run / "reads.json",
                        f"{prefix}/samples/{name}/compare.jpg": sheet}, f"screen {name}: reads and the before / after sheet")
            spent = time.time() - t0
            left = spent / (k + 1) * (len(arms) - k - 1)
            print(f"[{self.TAG}] screen {name}: ordinary prompts changed {reads['drift']:.4f} (seed-to-seed "
                  f"{reads['spread']:.4f}), own subject {reads['domain']:.4f}, contrast {reads['contrast']:.0%}"
                  + (f", {reads['missing']} preview(s) missing" if reads["missing"] else "")
                  + f" | {k + 1}/{len(arms)} arms, {_hms(spent)} spent" + (f", about {_hms(left)} left" if left else ""))
        pick = screen_pick(results)
        table = screen_table(results, pick)
        out = Path(self._need("data_root")) / "screen"
        out.mkdir(parents=True, exist_ok=True)
        summary = {"arms": results, "pick": pick, "rule": PICK_RULE, "steps": self.cfg.screen_steps,
                   "size": self.cfg.screen_size, "drift_prompts": DRIFT_PROMPTS, "domain_prompts": DOMAIN_PROMPTS,
                   "distance": f"1 - cosine of {DINO} pooled features"}
        (out / "screen.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        verdict = (f"Picked: **{pick['arm']}**." if pick["arm"] else
                   "The control failed (the untrained model rendered again differs by a seed change or more), so the "
                   "measurement cannot pick." if pick["verdict"] == "MEASUREMENT FAILED" else
                   "No arm met the rule: not ready.")
        (out / "screen.md").write_text(f"# The training-loss screen\n\n{PICK_RULE}\n\n{table}\n\n{verdict}\n",
                                       encoding="utf-8")
        self._ship({f"{self.cfg.release_dir}/screen/screen.json": out / "screen.json",
                    f"{self.cfg.release_dir}/screen/screen.md": out / "screen.md"}, "screen: summary")
        self.state["screen"] = {"arm": pick["arm"], "cfg": pick["cfg"]}
        self._save_state()
        print("\n" + table + "\n\n" + verdict)
        self._drain()
        return summary

    def _hub_screen_reads(self, prefix: str, setting: dict) -> dict:
        """{arm: reads} an earlier runtime shipped under the same setting; {} when there are none or the repo cannot
        be read (then every arm runs)."""
        if not self.cfg.release_repo:
            return {}
        try:
            from huggingface_hub import HfApi, hf_hub_download
            token = self.state.get("hf_token") or get_hf_token()
            out = {}
            for f in HfApi(token=token).list_repo_files(self.cfg.release_repo):
                if f.startswith(f"{prefix}/reads/") and f.endswith(".json"):
                    r = json.loads(Path(hf_hub_download(self.cfg.release_repo, f, token=token)).read_text(encoding="utf-8"))
                    if r.get("setting") == setting:
                        out[f[len(f"{prefix}/reads/"):-len(".json")]] = r
            return out
        except Exception as e:  # noqa: BLE001
            print(f"[{self.TAG}] could not read earlier screen reads from {self.cfg.release_repo} ({type(e).__name__}); "
                  f"every arm runs")
            return {}

    def _hub_previews(self, prefix: str, arm: str, save: str) -> "Path | None":
        """One arm's previews at one save as an earlier runtime shipped them (downloaded), or None."""
        if not self.cfg.release_repo:
            return None
        try:
            from huggingface_hub import snapshot_download
            local = Path(self._need("data_root")) / "hub_previews"
            snapshot_download(self.cfg.release_repo, local_dir=str(local), allow_patterns=[f"{prefix}/samples/{arm}/{save}/*"],
                              token=self.state.get("hf_token") or get_hf_token())
            folder = local / prefix / "samples" / arm / save
            return folder if any(folder.glob("*.png")) else None
        except Exception as e:  # noqa: BLE001
            print(f"[{self.TAG}] could not fetch {arm}'s {save} previews ({type(e).__name__}); this arm reads without "
                  f"the repeat control")
            return None

    def _embed(self, paths: list) -> list:
        """Unit-length DINOv2 pooled features of the pictures (on the GPU, 16 at a time); None for a missing one."""
        import torch
        if self._dino is None:
            from transformers import AutoImageProcessor, AutoModel
            self._dino = (AutoImageProcessor.from_pretrained(DINO), AutoModel.from_pretrained(DINO).to("cuda").eval())
        proc, model = self._dino
        have = [p for p in paths if p is not None]
        feats: list = []
        for i in range(0, len(have), 16):
            with torch.no_grad():
                x = proc(images=[_open(p) for p in have[i:i + 16]], return_tensors="pt")["pixel_values"].to("cuda")
                f = torch.nn.functional.normalize(model(pixel_values=x).pooler_output.float(), dim=-1)
            feats += f.cpu().tolist()
        it = iter(feats)
        return [None if p is None else next(it) for p in paths]

    def _free_dino(self) -> None:
        if self._dino is not None:
            import torch
            self._dino = None
            torch.cuda.empty_cache()

    # ---- 6. the full run ----------------------------------------------------------------------------
    def train(self, *, cfg=_PICK, epochs: "int | None" = None, tag: "str | None" = None) -> dict:
        """The full run: cfg = w (the guided loss), cfg = None (the plain loss), or by default the screen's pick.
        Saves every epoch; each save and its previews ship as soon as the next exists, the rest at the end."""
        if cfg is _PICK:
            pick = self.state.get("screen") or {}
            if not pick.get("arm"):
                raise RuntimeError("train() needs the training loss: run screen() first (it picks one), or pass "
                                   "cfg=w (the guided loss at strength w) or cfg=None (the plain loss).")
            cfg = pick["cfg"]
            print(f"[{self.TAG}] training with the screen's pick: {pick['arm']}")
        self._need("directories")
        self._need("model_paths")
        self._write_check()
        tag = tag or f"r{self.cfg.rank}_{arm_name(cfg)}"
        t0 = time.time()
        info = self._run(tag, cfg=cfg, prefix=f"{self.cfg.release_dir}/rank{self.cfg.rank}", epochs=epochs)
        self.state["last_run"] = tag
        self._save_state()
        left = self._drain()
        run = newest_run_dir(info["output_dir"])
        print(f"[{self.TAG}] {tag} finished in {_hms(time.time() - t0)} | saves: "
              f"{[n for n, _ in saved_dirs(run)] if run else []} | unshipped: {len(left)}")
        return info

    # ---- 7. sheets ----------------------------------------------------------------------------------
    def sheets(self, tag: "str | None" = None, *, cols: int = 5) -> list[str]:
        """One contact sheet per save of a run's previews (5 columns), shipped beside its LoRAs."""
        tag = tag or self.state.get("last_run")
        if not tag or tag not in self.state.get("runs", {}):
            raise RuntimeError("sheets() needs a finished run: train() first (or pass tag=)")
        info = self.state["runs"][tag]
        run = newest_run_dir(info["output_dir"])
        if run is None or not (run / "samples").is_dir():
            raise RuntimeError(f"{tag}: no previews under {info['output_dir']}")
        out = run / "sheets"
        out.mkdir(exist_ok=True)
        made = []
        for s in sorted(p for p in (run / "samples").iterdir() if p.is_dir()):
            tiles = [_open(p) for p in sorted(s.glob("*.png"))]
            if tiles:
                made.append(_grid([tiles[r:r + cols] for r in range(0, len(tiles), cols)], out / f"{s.name}.jpg", tile=384))
        arm = arm_name(info["cfg"])
        prefix = (f"{self.cfg.release_dir}/screen" if tag.startswith("screen_")
                  else f"{self.cfg.release_dir}/rank{self.cfg.rank}")
        self._ship({f"{prefix}/samples/{arm}/sheets/{p.name}": p for p in made}, f"{tag}: preview sheets")
        self._drain()
        print(f"[{self.TAG}] {len(made)} sheets in {out}")
        return [str(p) for p in made]

    # ---- the release folder's README and NOTICE (public prose) -----------------------------------------
    def folder_notice(self) -> str:
        who = self.state.get("hf_user") or "the uploader"
        return f"""{QWEN_NOTICE}

Modification statement: the LoRA files in this folder were trained by {who} on top of Qwen-Image 2.1
({BASE_REPO}); Qwen-Image 2.1 was modified to create them. They are distributed for non-commercial use (research
and evaluation) under the Qwen RESEARCH LICENSE AGREEMENT, a copy of which is in LICENSE in this folder. They are
not an official product of Qwen and have not been endorsed by it.

Built with Qwen.
"""

    def folder_readme(self) -> str:
        c = self.cfg
        rows = self.state.get("directories") or []
        classes = ", ".join(r["name"].replace("_", " ") for r in rows) or "the dataset's classes"
        images = sum(r["images"] for r in rows)
        rev = self.state.get("model_revision")
        title = c.name.replace("-", " ").title()
        excl = f" except {', '.join(c.exclude_modules)}" if c.exclude_modules else ""
        n_previews = len(c.preview_prompts or SHOWCASE)
        shape = (f"The training captions read `{c.caption_format}`, with one of these classes: {classes}. Prompts in "
                 f"the same shape work best; " if c.caption_format else "")
        return f"""# {title} for Qwen-Image 2.1

LoRAs for [Qwen-Image 2.1](https://huggingface.co/{BASE_REPO}) (text to image), trained on
[{c.dataset_repo}](https://huggingface.co/datasets/{c.dataset_repo}) with the
[AbstractEyes diffusion-pipe fork](https://github.com/AbstractEyes/diffusion-pipe) (model type `qwen_image21`, release
`{FORK_REF}`), driven by [anima-trainer](https://github.com/AbstractEyes/anima-trainer).

**Built with Qwen.**

## Licence
Qwen-Image 2.1 is released under the Qwen Research License Agreement (a copy is in `LICENSE`, the attribution in
`NOTICE`): non-commercial use only, meaning research or evaluation; commercial use needs a separate licence from Qwen.
These LoRAs are derivatives of it and carry the same terms. The training images belong to their creators and are
shared for research and non-commercial use (see the dataset card). Files outside this folder have their own licence
(see the repository card).

## Files
| folder | what it holds |
|---|---|
| `rank{c.rank}/` | the release runs: one LoRA per saved epoch, `{c.name}-qwen-image21-r{c.rank}-<loss>-eNN.safetensors`, with `adapter_config.json`, `samples/<loss>/` (the previews at every save, and contact sheets) and `training/<loss>/` (configs, log, TensorBoard events) |
| `screen/` | the short runs that chose the training loss (the plain loss and the guided loss at three strengths): each run's LoRA, `samples/<loss>/` (previews before and after, `compare.jpg`), `reads/<loss>.json` (its measurements); `screen.md` has the table and the rule |
| `checks/` | the check, on the training card, that the trainer computes what ComfyUI computes for this model |

`<loss>` is `plain` (ordinary flow matching) or `wN`: trained against the model's own guided output at strength N.
Qwen-Image 2.1 has its guidance trained in and samples at cfg 1; plain training wears that away, so a `plain` LoRA
tends to need real CFG at inference, while a `wN` LoRA keeps sampling at cfg 1.

The LoRAs are in ComfyUI format: load one with **LoraLoaderModelOnly** in the Qwen-Image 2.1 workflow (the base
transformer, 25 steps, cfg 1).

## Prompts
{shape}`samples/` shows the {n_previews} prompts the previews use.

## Training
| | |
|---|---|
| base | Qwen-Image 2.1, bf16 transformer from [Comfy-Org/Qwen-Image-2.1](https://huggingface.co/Comfy-Org/Qwen-Image-2.1){f" (commit {rev[:10]})" if rev else ""} |
| adapter | LoRA rank {c.rank} on the linear layers of the 32 transformer blocks{excl} |
| optimizer | Adam without weight decay, learning rate {c.lr:g} after {c.warmup_steps} warmup steps |
| data | {images or "the"} images in {len(rows) or "its"} classes; repeats per class dampened (square root, at most 8 per image) so small classes count without being memorized |
| resolution | {c.resolution} px area, aspect-ratio buckets from 1:2 to 2:1 |
| batch | {c.micro_batch} images per step |
| previews | 25 steps, cfg 1, the model's own time shift, one fixed seed per prompt at every save |
"""

    # ---- status ---------------------------------------------------------------------------------------
    def tail(self, tag: "str | None" = None, n: int = 40) -> None:
        tag = tag or self.state.get("last_run") or next(reversed(self.state.get("runs", {})), None)
        log = (self.state.get("runs", {}).get(tag) or {}).get("log")
        if not log or not Path(log).is_file():
            print(f"[{self.TAG}] no log yet")
            return
        print("".join(Path(log).read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)[-n:]))

    def status(self) -> dict:
        runs = {}
        for tag, info in self.state.get("runs", {}).items():
            run = newest_run_dir(info["output_dir"])
            runs[tag] = {"done": (Path(info["output_dir"]) / "DONE.json").is_file(),
                         "saves": [n for n, _ in saved_dirs(run)] if run else []}
        out = {"runs": runs, "parity": self.state.get("parity"), "screen": self.state.get("screen"),
               "unshipped": sorted(self._shipper.failed) if self._shipper else []}
        print(json.dumps(out, indent=2))
        return out


def _open(path):
    """A picture as RGB (the 2.1 previews are RGBA PNGs), or None for a missing one (a blank tile in a sheet)."""
    if path is None:
        return None
    from PIL import Image
    return Image.open(path).convert("RGB")


def _contrast(paths: list) -> list:
    """Each picture's luminance spread (the standard deviation of 0.299 R + 0.587 G + 0.114 B over pixels, in 0..1);
    None for a missing one."""
    import numpy as np
    out = []
    for p in paths:
        if p is None:
            out.append(None)
            continue
        a = np.asarray(_open(p), dtype=np.float32) / 255.0
        out.append(float((a @ np.array([0.299, 0.587, 0.114], dtype=np.float32)).std()))
    return out
