#!/usr/bin/env python3
"""
anima_runner.py — the Anima bed of the flavor experiments (the Anima counterpart of sana_runner.SanaRunner, whose
sequence machinery it reuses). Same thin-shell contract: the notebook (notebooks/anima_colab_experiments.ipynb) is a
handful of `a.<step>()` calls and all logic lives here.

    from geolip_anima_trainer.anima_runner import AnimaRunner
    a = AnimaRunner()          # Anima-Base v1.0, 768 px; uploads to AbstractPhil/geolip-beatrix-anima
    a.setup()                  # env + HF login + GPU + the diffusion-pipe fork + the three model files (~5.6 GB)
    a.run_flavor_test()        # e001: the stock model (mood words, a mood dial at two conditioning sites, the norms)
    a.run_sequence()           # e002..: the LoRA arms (anima_experiments.SEQUENCE)

Rendering. Every image (the training sets, the no-LoRA baseline, the evaluations, e001) is made in this process by
AnimaPipe, which loads the fork's own Anima model code (models/cosmos_predict2.py: Qwen3 0.6B -> the LLM adapter ->
the 2B DiT -> the Qwen-Image VAE) and runs the same Euler flow sampler the trainer's previews use, batched over
prompts. Trained LoRAs (ComfyUI format) are applied by forward hooks (LoraHooks), so the stock weights are never
modified and removing the hooks restores the stock model exactly.

Recipe (anima_experiments.py has the reasons): Anima-Base v1.0; LoRA rank 32 at learning rate 2e-5, plain Adam with no
weight decay and fp32 master weights over the bf16 LoRA; the LLM adapter frozen; 768 px; 30 steps, guidance 4.5,
shift 3, the model card's quality prefix and negative prompt.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path

from . import anima_experiments as ax
from . import api as _api
from . import sana_experiments as sx
from . import sana_runner as _sr
from .cache_factory import get_hf_token
from .sana_runner import HELD_OUT, SUBJECTS, _HubRepo, _drop_torchao, _grid

GEN_STEPS, GEN_CFG, GEN_SHIFT = 30, 4.5, 3.0          # the card: 30-50 steps, guidance 4-5; shift 3 = ComfyUI's
RESOLUTION = 768                                       # inside the card's 512-1536; about half the cost of 1024


@dataclass
class AnimaConfig:
    """Everything the Anima runs need, overridable from the notebook (kwargs) or env (ANIMA_*)."""
    repo_root: str = field(default_factory=lambda: os.environ.get("ANIMA_REPO", "/content/anima-trainer"))
    data_root: str | None = None                 # None -> /content/anima_data (Colab disk; the repo is the durability)
    hf_home: str | None = None                   # None -> {data_root}/hf_cache
    base: str = "base-v1.0"                      # download_anima.BASE_CHOICES key: the card says LoRAs train on Base
    models_dir: str | None = None                # an existing folder with the three files (skips the download)
    repo_id: str = ax.DEFAULT_REPO               # the experiments repo (public): one folder per experiment
    resolution: int = RESOLUTION
    # data (the sequence renders its own training sets; source/dataset_dir keep the shared single-run surface)
    source: str = "mood"
    dataset_dir: str | None = None
    seeds_per_subject: int = 8                   # 24 training scenes x 8 = 192 images
    gen_batch: int = 8
    # recipe (the arms override lr / data per arm)
    rank: int = 32
    lr: float = ax.LR
    epochs: int = 10
    micro_batch: int = 4
    warmup_steps: int = 20
    save_every_n_epochs: int = 2
    num_gpus: int = 1
    preview_prompts: list[str] | None = None     # None -> 4 held-out neutral prompts
    eval_seeds: list[int] = field(default_factory=lambda: [101, 202, 303, 404])
    eval_scales: list[float] = field(default_factory=lambda: [0.0, 0.5, 1.0])
    backup_repo: str | None = None

    @classmethod
    def from_env(cls, **overrides) -> "AnimaConfig":
        env: dict = {}
        if os.environ.get("ANIMA_DATA_ROOT"):
            env["data_root"] = os.environ["ANIMA_DATA_ROOT"]
        if os.environ.get("ANIMA_EXPERIMENTS_REPO"):
            env["repo_id"] = os.environ["ANIMA_EXPERIMENTS_REPO"]
        valid = {f.name for f in fields(cls)}
        bad = set(overrides) - valid
        if bad:
            raise TypeError(f"unknown AnimaConfig override(s): {sorted(bad)}")
        return cls(**{**env, **overrides})


# =============================================================================
# LoRA by forward hooks (ComfyUI or peft key layout)
# =============================================================================
class LoraHooks:
    """LoRA weights on a module tree, applied by forward hooks: y = W x + scale * (alpha / r) * B (A x). The stock
    weights are never touched; remove() restores the stock forward exactly. Keys: `<prefix><module path>.lora_A.weight`
    / `.lora_B.weight` (prefixes diffusion_model. / base_model.model. / transformer. are stripped). A module whose B is
    all zeros (e.g. the frozen LLM adapter's LoRA) adds nothing and is skipped."""
    PREFIXES = ("model.diffusion_model.", "diffusion_model.", "base_model.model.", "transformer.")

    def __init__(self, root):
        self.root = root
        self.adapters: dict = {}
        self.handles: list = []
        self.active: "tuple[str, float] | None" = None

    @classmethod
    def parse(cls, state_dict: dict) -> dict:
        """{module path: {'A': tensor, 'B': tensor}} from a LoRA state dict."""
        pairs: dict = {}
        for k, v in state_dict.items():
            for p in cls.PREFIXES:
                if k.startswith(p):
                    k = k[len(p):]
                    break
            if k.endswith(".lora_A.weight"):
                pairs.setdefault(k[:-len(".lora_A.weight")], {})["A"] = v
            elif k.endswith(".lora_B.weight"):
                pairs.setdefault(k[:-len(".lora_B.weight")], {})["B"] = v
            else:
                raise ValueError(f"unexpected LoRA key {k!r} (expected *.lora_A.weight / *.lora_B.weight)")
        missing = [m for m, ab in pairs.items() if set(ab) != {"A", "B"}]
        if missing:
            raise ValueError(f"LoRA modules without both A and B: {missing[:5]}")
        return pairs

    def load(self, name: str, state_dict: dict, *, scaling: float = 1.0, device=None, dtype=None) -> dict:
        import torch
        from torch import nn
        mods, skipped = {}, 0
        for path, ab in self.parse(state_dict).items():
            mod = self.root.get_submodule(path)          # AttributeError when the path does not exist
            if not isinstance(mod, nn.Linear):
                raise TypeError(f"LoRA target {path} is a {type(mod).__name__}, not a Linear")
            a, b = ab["A"], ab["B"]
            if a.shape != (a.shape[0], mod.in_features) or b.shape != (mod.out_features, a.shape[0]):
                raise ValueError(f"LoRA shapes {tuple(a.shape)} / {tuple(b.shape)} do not fit {path} "
                                 f"({mod.in_features} -> {mod.out_features})")
            if not torch.any(b):
                skipped += 1
                continue
            dev = device or mod.weight.device
            dt = dtype or mod.weight.dtype
            mods[path] = (a.to(dev, dt), b.to(dev, dt))
        self.adapters[name] = {"mods": mods, "scaling": float(scaling)}
        return {"modules": len(mods), "skipped_zero": skipped}

    def set(self, name: str, scale: float) -> None:
        """Make adapter `name` the only active one, at `scale` (re-registers the hooks)."""
        import torch.nn.functional as F
        self.remove()
        ad = self.adapters[name]
        s = float(scale) * ad["scaling"]
        for path, (a, b) in ad["mods"].items():
            def hook(mod, inp, out, a=a, b=b, s=s):
                return out + F.linear(F.linear(inp[0].to(a.dtype), a), b).to(out.dtype) * s
            self.handles.append(self.root.get_submodule(path).register_forward_hook(hook))
        self.active = (name, float(scale))

    def remove(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles, self.active = [], None

    def unload(self) -> None:
        self.remove()
        self.adapters.clear()


def _read_lora(folder: "str | Path", weight_name: str = "adapter_model.safetensors") -> tuple[dict, float]:
    """(state dict, alpha / r) of a saved LoRA; diffusion-pipe forces alpha = rank, so the scaling is 1 when the folder
    has no adapter_config.json."""
    from safetensors.torch import load_file
    folder = Path(folder)
    sd = load_file(str(folder / weight_name))
    scaling = 1.0
    cfg = folder / "adapter_config.json"
    if cfg.is_file():
        c = json.loads(cfg.read_text(encoding="utf-8"))
        r, alpha = c.get("r"), c.get("lora_alpha")
        if r and alpha:
            scaling = float(alpha) / float(r)
    return sd, scaling


# =============================================================================
# Stock Anima in this process (the fork's model code)
# =============================================================================
class AnimaPipe:
    """Anima in the notebook process through the diffusion-pipe fork's own model code: the same text encoding
    (Qwen3 0.6B, prompts padded to 512 tokens), layer chain and Euler flow sampler the trainer's previews use, batched
    over prompts (each image keeps its own seeded noise), each image decoded on its own as the previews do. The
    diffusers-style LoRA calls the sequence uses (load_lora_weights / set_adapters / unload_lora_weights /
    get_list_adapters) go to LoraHooks."""

    def __init__(self, fork_dir: str, transformer_path: str, vae_path: str, llm_path: str):
        import torch
        fork_dir = str(Path(fork_dir).resolve())
        for mod in ("utils", "models"):          # the fork's top-level packages must not be shadowed
            m = sys.modules.get(mod)
            src = (getattr(m, "__file__", None) or next(iter(getattr(m, "__path__", None) or []), "")) if m else ""
            if m is not None and not str(Path(src).resolve()).startswith(fork_dir):
                raise RuntimeError(f"a module named {mod!r} from {src or m} is already imported; restart the "
                                   f"session so the diffusion-pipe fork's {mod!r} can load")
        if fork_dir not in sys.path:
            sys.path.insert(0, fork_dir)
        import utils.common as common            # the layers read the autocast dtype when their module is imported
        common.AUTOCAST_DTYPE = torch.bfloat16
        cwd = os.getcwd()
        os.chdir(fork_dir)                       # the tokenizers load from the fork's configs/ (relative paths)
        try:
            from models import cosmos_predict2 as cp
            self.model = cp.CosmosPredict2Pipeline({"model": {
                "type": "anima", "dtype": torch.bfloat16, "transformer_path": transformer_path,
                "vae_path": vae_path, "llm_path": llm_path}})
            self.model.load_diffusion_model()
        finally:
            os.chdir(cwd)
        m = self.model
        m.transformer.to("cuda").eval().requires_grad_(False)
        m.text_encoder.to("cuda").eval()
        m.vae.model.to("cuda")
        self.layers = m.to_layers()
        self.adapter_at = next(i for i, l in enumerate(self.layers) if type(l).__name__ == "LLMAdapterLayer")
        self._te = m.get_call_text_encoder_fn(m.text_encoder)
        self.lora = LoraHooks(m.transformer)

    # ---- text -> conditioning ----------------------------------------------------------------------
    def encode(self, prompts: list[str]) -> tuple:
        """(Qwen3 states [B,512,1024], their mask, T5 ids, T5 mask) on the GPU, as the trainer caches them."""
        import torch
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            d = self._te(list(prompts), is_video=[False] * len(prompts))
        return tuple(t.to("cuda") for t in self.model.get_conds(d))

    def adapt(self, conds: tuple):
        """The LLM adapter's output (what the DiT cross-attends to), zeroed at T5 padding as the layer chain does."""
        import torch
        pe, mask, t5_ids, t5_mask = conds
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            out = self.model.transformer.llm_adapter(source_hidden_states=pe, target_input_ids=t5_ids,
                                                     target_attention_mask=t5_mask, source_attention_mask=mask)
        out = out.clone()
        out[~t5_mask.bool()] = 0
        return out

    @staticmethod
    def add_at_tokens(states, mask, vec):
        """states + vec at every token the mask marks (padding untouched)."""
        return states + mask.to(states.dtype)[..., None] * vec.to(states.device, states.dtype)

    # ---- sampling ---------------------------------------------------------------------------------
    def _forward(self, x, t, conds: tuple, context_add=None):
        inputs = (x, t, *conds)
        for i, layer in enumerate(self.layers):
            inputs = layer(inputs)
            if i == self.adapter_at and context_add is not None:
                x_, temb, ctx, *rest = inputs
                inputs = (x_, temb, self.add_at_tokens(ctx, conds[3], context_add), *rest)
        return inputs

    def generate(self, prompts: list[str], seeds: list[int], *, res: int, steps: int, cfg: float, shift: float,
                 negative: str = "", batch: int = 8, source_add=None, context_add=None) -> list:
        """PIL images, one per (prompt, seed). source_add / context_add: a 1024-vector added to every prompt token
        before / after the LLM adapter, on the conditional branch only (the negative prompt is left unchanged)."""
        import torch
        from utils.previews import to_pil
        if len(prompts) != len(seeds):
            raise ValueError("one seed per prompt")
        unc = self.encode([negative]) if cfg > 1 else None
        out: list = []
        with torch.no_grad():
            for i in range(0, len(prompts), batch):
                p, s = list(prompts[i:i + batch]), list(seeds[i:i + batch])
                n = len(p)
                conds = self.encode(p)
                if source_add is not None:
                    conds = (self.add_at_tokens(conds[0], conds[1], source_add), *conds[1:])
                un = tuple(u.expand(n, *u.shape[1:]).contiguous() for u in unc) if unc is not None else None
                x = torch.cat([torch.randn((1, 16, res // 8, res // 8), device="cuda",
                                           generator=torch.Generator(device="cuda").manual_seed(int(sd)))
                               for sd in s]).unsqueeze(2)
                self.model.set_sample_schedule(steps, shift)           # a fresh Euler schedule (rewinds its index)
                sch = self.model.scheduler
                for step in sch.timesteps:
                    t = (step / 1000).float().reshape(1).repeat(n)
                    v = self._forward(x, t, conds, context_add).float()
                    if un is not None:
                        vu = self._forward(x, t, un).float()
                        v = vu + cfg * (v - vu)
                    x = sch.step(v, step, x, return_dict=False)[0]
                for b in range(n):                                     # decoded one by one, as the previews do
                    img = self.model.vae_decode(x[b:b + 1])
                    out.append(to_pil(img[:, 0] if img.ndim == 5 else img))
        return out

    # ---- the diffusers-style LoRA surface the sequence uses -----------------------------------------
    def load_lora_weights(self, path: str, weight_name: str = "adapter_model.safetensors",
                          adapter_name: str = "default") -> dict:
        sd, scaling = _read_lora(path, weight_name)
        return self.lora.load(adapter_name, sd, scaling=scaling)

    def set_adapters(self, names: list[str], adapter_weights: list[float]) -> None:
        if len(names) != 1:
            raise ValueError("one adapter at a time")
        self.lora.set(names[0], adapter_weights[0])

    def unload_lora_weights(self) -> None:
        self.lora.unload()

    def get_list_adapters(self) -> dict:
        return {"transformer": list(self.lora.adapters)} if self.lora.adapters else {}


# =============================================================================
# The runner
# =============================================================================
class AnimaRunner(_sr.SanaRunner):
    """The Anima bed: setup + the e001 flavor test + the LoRA sequence (shared with the Sana runner)."""
    TAG = "anima"
    STATE_FILE = "anima_state.json"
    EXPECT_SM = None
    BED = ax.ANIMA
    GEN_STEPS, GEN_CFG, GEN_SHIFT = GEN_STEPS, GEN_CFG, GEN_SHIFT
    NEGATIVE = ax.NEGATIVE

    def __init__(self, config: "AnimaConfig | None" = None, **overrides):
        self.cfg = config or AnimaConfig.from_env(**overrides)
        if self.cfg.base not in _api._dl.BASE_CHOICES:
            raise ValueError(f"base must be one of {list(_api._dl.BASE_CHOICES)}, got {self.cfg.base!r}")
        if self.cfg.resolution % 16:
            raise ValueError(f"resolution must be a multiple of 16, got {self.cfg.resolution}")
        self.state: dict = {}
        self._pipe = None
        self._judge_fns = None
        self._base = None

    # ---- setup ------------------------------------------------------------------------------------
    def setup(self) -> dict:
        self._setup_env()
        _drop_torchao()
        self._auth_optional()
        self._verify_gpu()
        self._point_at_fork()
        self._download_model()
        self._save_state()
        print(f"[anima] setup done | DATA_ROOT={self.state['data_root']} | model={self.state['transformer_path']} "
              f"| {self.state['resolution']} px | experiments -> {self.cfg.repo_id}")
        return self.state

    def _setup_env(self) -> None:
        data_root = self.cfg.data_root or "/content/anima_data"
        hf_home = self.cfg.hf_home or f"{data_root}/hf_cache"
        os.environ["HF_HOME"] = hf_home
        os.environ["ANIMA_DATA_ROOT"] = data_root
        for d in (hf_home, data_root):
            os.makedirs(d, exist_ok=True)
        if self.cfg.repo_root not in sys.path:
            sys.path.insert(0, self.cfg.repo_root)
        self.state.update(data_root=data_root, hf_home=hf_home)
        print(f"[anima] DATA_ROOT={data_root} | HF_HOME={hf_home}")

    def _auth_optional(self) -> None:
        token = get_hf_token()
        self.state["hf_token"] = token
        if not token:
            print(f"[anima] no HF_TOKEN: the model is public, but the experiments upload to {self.cfg.repo_id} "
                  f"and need a WRITE token in Colab Secrets")
            return
        from huggingface_hub import login, whoami
        login(token=token, add_to_git_credential=False)
        self.state["hf_user"] = whoami(token=token).get("name")
        print(f"[anima] HF user={self.state['hf_user']}")

    def _download_model(self) -> str:
        if self.cfg.models_dir:
            root = Path(self.cfg.models_dir)
            found = {k: next(iter(root.rglob(n)), None) for k, n in (
                ("transformer_path", _api._dl.BASE_CHOICES[self.cfg.base]),
                ("vae_path", Path(_api._dl.VAE).name), ("llm_path", Path(_api._dl.TEXT_ENCODER).name))}
            missing = [k for k, v in found.items() if v is None]
            if missing:
                raise RuntimeError(f"{root} lacks {missing}; drop models_dir= to download them")
            paths = {k: str(v) for k, v in found.items()}
        else:
            p = _api.download_models(f"{self.state['data_root']}/models", base=self.cfg.base)
            paths = {"transformer_path": p.transformer_path, "vae_path": p.vae_path, "llm_path": p.llm_path}
        self.state.update(**paths, resolution=int(self.cfg.resolution))
        return paths["transformer_path"]

    def _need_model(self) -> None:
        self._need("transformer_path")

    # ---- resident GPU pieces ----------------------------------------------------------------------
    def _eval_pipe(self):
        if self._pipe is None:
            fork = self.state.get("diffusion_pipe") or self._point_at_fork()
            print(f"[{self.TAG}] loading Anima into this notebook (transformer, Qwen3, VAE)...", flush=True)
            t0 = time.time()
            self._pipe = AnimaPipe(fork, self._need("transformer_path"), self._need("vae_path"),
                                   self._need("llm_path"))
            print(f"[{self.TAG}] Anima loaded in {time.time() - t0:.0f} s", flush=True)
        return self._pipe

    def _render(self, prompts: list[str], seeds: list[int], **kw) -> list:
        return self._eval_pipe().generate(prompts, seeds, res=self._need("resolution"), steps=self.GEN_STEPS,
                                          cfg=self.GEN_CFG, shift=self.GEN_SHIFT, negative=self.NEGATIVE,
                                          batch=self.cfg.gen_batch, **kw)

    # ---- the training config ----------------------------------------------------------------------
    def _render_config(self, dataset_dir: str, output_dir: str, configs_dir: str, *, lr: float,
                       held_out_previews: bool) -> "tuple[Path, Path]":
        res = self._need("resolution")
        model = _api.ModelConfig(type="anima", transformer_path=self._need("transformer_path"),
                                 vae_path=self._need("vae_path"), llm_path=self._need("llm_path"), llm_adapter_lr=0.0)
        prompts = self.cfg.preview_prompts
        if prompts is None and held_out_previews:
            prompts = [self.BED.neutral_caption.format(s=SUBJECTS[i]) for i in HELD_OUT[:4]]
        samples = _api.SamplesConfig(prompts=list(prompts), negative_prompt=self.NEGATIVE, width=res, height=res,
                                     steps=self.GEN_STEPS, cfg=self.GEN_CFG, shift=self.GEN_SHIFT,
                                     seed=42) if prompts else None
        cfg = _api.TrainConfig(
            run=_api.RunConfig(output_dir=output_dir, epochs=self.cfg.epochs,
                               micro_batch_size_per_gpu=self.cfg.micro_batch, warmup_steps=self.cfg.warmup_steps,
                               save_every_n_epochs=self.cfg.save_every_n_epochs, eval_before_first_step=False,
                               bf16_master_weights=True),
            model=model, adapter=_api.AdapterConfig(rank=self.cfg.rank),
            optimizer=_api.OptimizerConfig(type="adam", lr=lr, weight_decay=0.0),
            dataset=_api.DatasetConfig(resolutions=[res], directories=[_api.DirectoryConfig(path=str(dataset_dir))]),
            samples=samples)
        return _api.render_train_toml(cfg, configs_dir)

    def _recipe(self, lr: float) -> dict:
        n = len(_sr.TRAIN) * self.cfg.seeds_per_subject
        spe = -(-n // (self.cfg.micro_batch * self.cfg.num_gpus))
        return {"base model": "Anima-Base v1.0 (2B; CircleStone Labs Non-Commercial License)",
                "adapter": f"LoRA rank {self.cfg.rank} (alpha = rank) on every linear layer of the 28 transformer "
                           "blocks; the LLM adapter frozen",
                "optimizer": f"Adam, no weight decay, learning rate {lr:g}, {self.cfg.warmup_steps} linear warmup "
                             "steps, then constant; fp32 master weights over the bf16 LoRA",
                "batch": f"{self.cfg.micro_batch} images per step",
                "length": f"{self.cfg.epochs} epochs x {spe} steps = {spe * self.cfg.epochs} steps",
                "saves": f"every {self.cfg.save_every_n_epochs} epochs, each with previews of 4 held-out prompts",
                "resolution": f"{self._need('resolution')} x {self._need('resolution')}",
                "timestep sampling": "logit-normal, no shift (the trainer's Anima default)",
                "evaluation": f"8 held-out scenes x seeds {', '.join(map(str, self.cfg.eval_seeds))}, Euler, "
                              f"{self.GEN_STEPS} steps, guidance {self.GEN_CFG}, shift {self.GEN_SHIFT:g}, the "
                              "card's quality prefix and negative prompt"}

    # ---- single runs: the sequence is the supported path on this bed --------------------------------
    def evaluate(self, *a, **k):
        raise NotImplementedError("the Anima bed runs through run_flavor_test() and run_sequence()")

    # ---- the renderer against the trainer's own previews (once per session) ---------------------------
    PARITY_LEVELS = 2.0                          # mean |difference| in 8-bit levels counted as the same image

    def _after_train(self, spec: "sx.ArmSpec", ctx: dict, epochs: list) -> "dict | None":
        """The first arm of a session: this process's renders of the trainer's preview prompts (same seeds 42 + i,
        settings, one image at a time as the previews render) against the trainer's own preview images: the stock
        model (step 0) and the final LoRA at scale 1 through the hooks. Mean / max pixel differences go into the
        arm's meta; a mismatch is printed as a warning (the evaluation still runs)."""
        if getattr(self, "_parity", None) is not None:
            return None
        import numpy as np
        from PIL import Image
        pipe = self._eval_pipe()
        from utils.previews import image_filename        # the fork's own file naming (on sys.path via the pipe)
        prompts = self.cfg.preview_prompts or [self.BED.neutral_caption.format(s=SUBJECTS[i]) for i in HELD_OUT[:4]]
        seeds = [42 + i for i in range(len(prompts))]
        samples = epochs[-1][1].parent / "samples"
        n_final, d_final = epochs[-1]
        out: dict = {}
        for name, lora_dir in (("step0", None), (f"epoch{n_final}", d_final)):
            refs = [samples / name / image_filename(i, p) for i, p in enumerate(prompts)]
            if not all(r.is_file() for r in refs):
                out[name] = {"status": "previews missing"}
                continue
            if lora_dir is not None:
                pipe.load_lora_weights(str(lora_dir), weight_name="adapter_model.safetensors", adapter_name="parity")
                pipe.set_adapters(["parity"], adapter_weights=[1.0])
            try:
                imgs = pipe.generate(prompts, seeds, res=self._need("resolution"), steps=self.GEN_STEPS,
                                     cfg=self.GEN_CFG, shift=self.GEN_SHIFT, negative=self.NEGATIVE, batch=1)
            finally:
                if lora_dir is not None:
                    pipe.unload_lora_weights()
            diffs = [np.abs(np.asarray(a.convert("RGB"), dtype=np.float32)
                            - np.asarray(Image.open(r).convert("RGB"), dtype=np.float32)) for a, r in zip(imgs, refs)]
            mean = float(np.mean([d.mean() for d in diffs]))
            out[name] = {"mean_levels": mean, "max_levels": float(max(d.max() for d in diffs)),
                         "status": "MATCHES" if mean <= self.PARITY_LEVELS else "DIFFERS"}
            print(f"[anima] renderer vs the trainer's previews ({name}): mean {mean:.2f} levels -> "
                  f"{out[name]['status']}", flush=True)
            if out[name]["status"] == "DIFFERS":
                print(f"[anima] WARNING: this process renders the {name} previews differently from the trainer; "
                      "the evaluation's images may not be the trainer's model", flush=True)
        self._parity = out
        return out

    # ---- e001: the stock model -------------------------------------------------------------------------
    def _flavor_recipe(self) -> dict:
        return {"model": "Anima-Base v1.0, no LoRA",
                "images": f"{self._need('resolution')} x {self._need('resolution')}, Euler, {self.GEN_STEPS} steps, "
                          f"guidance {self.GEN_CFG}, shift {self.GEN_SHIFT:g}",
                "prompts": "the model card's quality prefix + an illustration of <scene>; the card's negative prompt",
                "scenes x seeds": f"32 x {len(ax.FLAVOR_TEST_SEEDS)} ({', '.join(map(str, ax.FLAVOR_TEST_SEEDS))})"}

    def run_flavor_test(self, *, force: bool = False) -> dict:
        """e001 into its own folder of cfg.repo_id: the words, the conditioning norms and the dial at both sites.
        Skipped when the repo lists it as done (force=True reruns)."""
        self._need_model()
        _drop_torchao()
        repo = _HubRepo(self.cfg.repo_id, self.state.get("hf_token"))
        metas = repo.metas()
        try:
            self._publish_index(repo, metas)
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"cannot write to {self.cfg.repo_id} ({e}); the HF_TOKEN needs WRITE access") from e
        fid = ax.FLAVOR_TEST_ID
        if metas.get(fid, {}).get("status") == "done" and not force:
            print(f"[anima] {fid} is done already (force=True reruns it)", flush=True)
            return metas[fid]
        base = f"experiments/{fid}"
        recipe = self._flavor_recipe()
        meta = {"id": fid, "title": ax.FLAVOR_TEST_TITLE, "date": ax.DATE, "kind": "flavor_test", "status": "running",
                "recipe": recipe}
        repo.commit({f"{base}/meta.json": sx.dumps(meta), f"{base}/README.md": ax.render_flavor_test_readme(meta, recipe)},
                    f"{fid}: started (README)")
        out_dir = Path(self._need("data_root")) / "experiments" / fid
        out_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        try:
            result = self._flavor_test(out_dir)
        except Exception as e:  # noqa: BLE001
            meta.update(status="failed", error=f"{type(e).__name__}: {e}"[:500])
            self._safe(lambda: repo.commit({f"{base}/meta.json": sx.dumps(meta),
                                            f"{base}/README.md": ax.render_flavor_test_readme(meta, recipe)},
                                           f"{fid}: failed"))
            raise
        w_up, w_dn = result["up_words"], result["down_words"]
        ds, dc = result["dial"]["source"], result["dial"]["context"]
        summary = (f"upbeat words {w_up['mean']:+.2f} ({w_up['OUTCOME']}), downbeat words {w_dn['mean']:+.2f} "
                   f"({w_dn['OUTCOME']}); dial per unit alpha: source {ds['mean']:+.2f} ({ds['OUTCOME']}), context "
                   f"{dc['mean']:+.2f} ({dc['OUTCOME']})")
        meta.update(status="done", result={k: v for k, v in result.items() if k != "cells"}, summary=summary,
                    seconds=round(time.time() - t0),
                    finished_utc=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"))
        (out_dir / "result.json").write_text(json.dumps(result, indent=1, default=float), encoding="utf-8")
        files = {f"{base}/meta.json": sx.dumps(meta), f"{base}/README.md": ax.render_flavor_test_readme(meta, recipe),
                 f"{base}/result.json": out_dir / "result.json"}
        for f in sorted(out_dir.glob("sheet_*.jpg")):
            files[f"{base}/{f.name}"] = f
        repo.commit(files, f"{fid}: result")
        metas[fid] = meta
        self._safe(lambda: self._publish_index(repo, metas))
        print(f"[anima] {fid} done in {meta['seconds']} s: {summary}", flush=True)
        return meta

    def _flavor_test(self, out_dir: Path) -> dict:
        import numpy as np
        self._eval_pipe()
        self._assert_stock()
        seeds = list(ax.FLAVOR_TEST_SEEDS)
        cells = [(si, sd) for si in range(len(SUBJECTS)) for sd in seeds]
        print(f"[anima] e001: {3 + len(ax.DIAL_SITES) * len(ax.DIAL_ALPHAS)} sets of {len(cells)} images (the words: "
              f"neutral, upbeat, downbeat; the dial: {len(ax.DIAL_SITES)} sites x {len(ax.DIAL_ALPHAS)} strengths), "
              "one line per set", flush=True)
        S = [sd for _, sd in cells]
        P = {f: [ax.TEMPLATES[f][0].format(s=SUBJECTS[si]) for si, _ in cells] for f in ("neutral", "up", "down")}
        imgs, feats, scores = {}, {}, {}
        for f in ("neutral", "up", "down"):                       # the words
            imgs[f] = self._render(P[f], S)
            feats[f], sc = self._score(imgs[f])
            scores[f] = [float(x) for x in sc]
            print(f"[anima] e001 words: {f} mean mood score {np.mean(scores[f]):+.3f}", flush=True)
        up = [a - b for a, b in zip(scores["up"], scores["neutral"])]
        dn = [a - b for a, b in zip(scores["down"], scores["neutral"])]
        result = {"up_words": ax.flavor_outcome(up, 1, label="UPBEAT WORDS MOVE IT"),
                  "down_words": ax.flavor_outcome(dn, -1, label="DOWNBEAT WORDS MOVE IT"),
                  "mood_score": {f: float(np.mean(v)) for f, v in scores.items()}}
        # the norms and the mood direction at both sites (per scene: the first template of each flavor)
        per_scene = {f: [ax.TEMPLATES[f][0].format(s=s) for s in SUBJECTS] for f in ("neutral", "up", "down")}
        states = {f: self._site_states(per_scene[f]) for f in per_scene}
        census, dirs = {}, {}
        for site in ax.DIAL_SITES:
            x, m = states["neutral"][site]
            norms = x.norm(dim=-1)[m.bool()].float().cpu().numpy()
            mu = {f: (states[f][site][0] * states[f][site][1][..., None]).sum(1) / states[f][site][1].sum(1, keepdim=True)
                  for f in ("up", "down")}
            d = ((mu["up"] - mu["down"]) / 2).mean(0)
            dirs[site] = d
            census[site] = {"mean": float(norms.mean()), "median": float(np.median(norms)), "max": float(norms.max()),
                            "tokens": int(norms.size), "d_norm": float(d.norm()),
                            "d_ratio": float(d.norm()) / float(norms.mean())}
        result["census"] = census
        # the dial: the neutral prompts + alpha * d at each site (alpha 0 = the words set's neutral images)
        result["dial"] = {}
        rows_first = [i for i, (_, sd) in enumerate(cells) if sd == seeds[0]][::4]      # 8 scenes, first seed
        for site in ax.DIAL_SITES:
            by_alpha, kept, sheet_cols = {0.0: scores["neutral"]}, {0.0: 1.0}, {0.0: imgs["neutral"]}
            for a in ax.DIAL_ALPHAS:
                vec = dirs[site] * a
                ims = self._render(P["neutral"], S, **({"source_add": vec} if site == "source" else {"context_add": vec}))
                fa, sc = self._score(ims)
                by_alpha[a] = [float(x) for x in sc]
                kept[a] = float(np.mean((fa * feats["neutral"]).sum(-1)))
                sheet_cols[a] = ims
                print(f"[anima] e001 dial {site} alpha {a:+g}: mood {np.mean(by_alpha[a]):+.3f}, kept {kept[a]:.3f}",
                      flush=True)
            rd = ax.flavor_outcome(ax.cell_slopes(by_alpha), 1, label="A DIAL")
            rd["by_alpha"] = {str(a): {"mood_score": float(np.mean(by_alpha[a])), "content_kept": kept[a]}
                              for a in sorted(by_alpha)}
            result["dial"][site] = rd
            alphas = sorted(sheet_cols)
            _grid([[sheet_cols[a][i] for a in alphas] for i in rows_first], out_dir / f"sheet_dial_{site}.jpg")
        _grid([[imgs[f][i] for f in ("neutral", "up", "down")] for i in rows_first], out_dir / "sheet_words.jpg")
        result["cells"] = [{"scene": SUBJECTS[si], "seed": sd, **{f: scores[f][k] for f in scores}}
                           for k, (si, sd) in enumerate(cells)]
        return result

    def _site_states(self, prompts: list[str]) -> dict:
        """{'source': (Qwen3 states, mask), 'context': (adapter output, T5 mask)} as float32 on the GPU."""
        pipe = self._eval_pipe()
        conds = pipe.encode(prompts)
        ctx = pipe.adapt(conds)
        return {"source": (conds[0].float(), conds[1]), "context": (ctx.float(), conds[3])}
