#!/usr/bin/env python3
"""
anima_runner.py — the Anima bed of the flavor experiments (the Anima counterpart of sana_runner.SanaRunner, whose
sequence machinery it reuses). Same thin-shell contract: the notebook (notebooks/anima_colab_experiments.ipynb) is a
handful of `a.<step>()` calls and all logic lives here.

    from geolip_anima_trainer.anima_runner import AnimaRunner
    a = AnimaRunner()          # Anima-Base v1.0, 768 px; uploads to AbstractPhil/geolip-beatrix-anima
    a.setup()                  # env + HF login + GPU + the diffusion-pipe fork + the three model files (~5.6 GB)
    a.run_flavor_test()        # e001: the stock model (mood words, a mood dial at two conditioning sites, the norms)
    a.run_route_split()        # e020 / e021 (run_appended_split): which of the adapter's two readings carries the words
    a.run_query_dial()         # e022: a mood direction in the adapter's queries, alone and with a source token
    a.run_word_split()         # e026: single words, whole T5 tokens against shattered ones
    a.run_slot_pair()          # e027: a word-sized push at one word's position: its query side, its source side, both
    a.run_attribute_screen()   # e012: attribute words, attribute sliders after the adapter, their cross-talk
    a.run_sequence()           # e002..: the LoRA arms (anima_experiments.SEQUENCE)
    a.run_beatrix_connectors() # e013-e025: a push from Beatrix's phrase features (+ an untrained trunk, a free vector)

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
import subprocess
import sys
import time
from dataclasses import dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path

from . import anima_experiments as ax
from . import api as _api
from . import sana_experiments as sx
from . import sana_runner as _sr
from . import training_sets as _ts
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
    data_repo_id: str | None = _ts.DATA_REPO     # the drawn training sets, reused by later runtimes (None: never)
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
        if os.environ.get("ANIMA_DATA_REPO"):
            env["data_repo_id"] = os.environ["ANIMA_DATA_REPO"]
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
    get_list_adapters) go to LoraHooks. train_loss() is the trainer's objective with a push added after the adapter,
    for the connector experiments (every model weight stays frozen)."""
    device = "cuda"
    context_width = 1024                         # the adapter's output, which the DiT cross-attends to

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
        self._vae_fn = m.get_call_vae_fn(m.vae.model)
        self._loss_fn = m.get_loss_fn()
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
        """states + vec at every token the mask marks (padding untouched); vec is [D] or per image [B, 1, D]."""
        return states + mask.to(states.dtype)[..., None] * vec.to(states.device, states.dtype)

    @staticmethod
    def append_source_token(conds: tuple, token) -> tuple:
        """conds with one extra source position right after each caption's last Qwen3 token, its mask opened: a vector
        the adapter's queries can look up there (token [D] or per image [B, D]). Qwen3's tokenizer pads on the right."""
        import torch
        pe, am, ids, tm = conds
        pe, am = pe.clone(), am.clone()
        n = am.sum(1).long()
        if int(n.max()) >= pe.shape[1]:
            raise ValueError("no room after the caption for a source token")
        tok = token.to(pe.device, pe.dtype)
        tok = tok.expand(pe.shape[0], -1) if tok.ndim == 1 else tok
        rows = torch.arange(pe.shape[0], device=pe.device)
        pe[rows, n] = tok
        am[rows, n] = 1
        return (pe, am, ids, tm)

    def query_states(self, prompts: list[str]) -> tuple:
        """The adapter's query embeddings before its blocks (its word table's output through in_proj; float32) and the T5
        mask."""
        import torch
        conds = self.encode(prompts)
        ad = self.model.transformer.llm_adapter
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            x = ad.in_proj(ad.embed(conds[2]))
        return x.float(), conds[3]

    def word_states(self, prompts: list[str], words: "tuple[str, ...]"):
        """Per prompt, the mean Qwen3 state over the tokens whose characters overlap any of `words` [B, D] (float32):
        the source vectors a query for those words looks up."""
        import torch
        conds = self.encode(prompts)
        out = []
        for b, p in enumerate(prompts):
            offs = self.model.tokenizer(p, return_offsets_mapping=True)["offset_mapping"]
            spans = [(p.index(w), p.index(w) + len(w)) for w in words if w in p]
            idx = [k for k, (a, e) in enumerate(offs) if any(a < we and e > ws for ws, we in spans)]
            if not idx:
                raise ValueError(f"none of {words} in {p!r}")
            out.append(conds[0][b, idx].float().mean(0))
        return torch.stack(out)

    def slot_masks(self, prompts: list[str], word: str, device: str = "cuda", length: int = 512) -> tuple:
        """Per prompt, float masks [B, length] over the T5 positions and the Qwen3 positions of the tokens whose
        characters overlap the last occurrence of `word` (the pipeline calls both tokenizers plainly with right
        padding, so the offsets' indices are the tensors' positions)."""
        import torch
        out = []
        for tok in (self.model.t5_tokenizer, self.model.tokenizer):
            rows = torch.zeros(len(prompts), length)
            for b, p in enumerate(prompts):
                if word not in p:
                    raise ValueError(f"{word!r} not in {p!r}")
                ws = p.rindex(word)
                we = ws + len(word)
                offs = tok(p, return_offsets_mapping=True)["offset_mapping"]
                idx = [k for k, (a, e) in enumerate(offs) if a < we and e > ws and k < length]
                if not idx:
                    raise ValueError(f"no token of {word!r} in {p!r}")
                rows[b, idx] = 1.0
            out.append(rows.to(device))
        return tuple(out)

    def word_queries(self, words: "tuple[str, ...]"):
        """The adapter's query embedding of each word [len, D] float32: its word table's row through in_proj, for words
        the T5 vocabulary holds as one token (after a space)."""
        import torch
        tok = self.model.t5_tokenizer
        ids = []
        for w in words:
            pieces = tok.tokenize(" " + w)
            if len(pieces) != 1:
                raise ValueError(f"{w!r} is not one T5 token: {pieces}")
            ids.append(tok.convert_tokens_to_ids(pieces[0]))
        ad = self.model.transformer.llm_adapter
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            x = ad.in_proj(ad.embed(torch.tensor(ids, device="cuda")))
        return x.float()

    def encode_images(self, imgs: list, res: int):
        """Latents [B, 16, 1, res/8, res/8] of PIL images by the trainer's own path: fitted to res x res
        (utils.image_resize.convert_crop_and_resize), scaled to [-1, 1] (ToTensor + Normalize(.5, .5)), then the VAE
        encode with its latent scale (the fork's get_call_vae_fn)."""
        import torch
        from torchvision import transforms
        from utils.image_resize import convert_crop_and_resize
        to_tensor = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
        x = torch.stack([to_tensor(convert_crop_and_resize(im, (res, res))) for im in imgs]).unsqueeze(2)
        with torch.no_grad():
            return self._vae_fn(x)["latents"]

    # ---- training a push (the connector experiments) ------------------------------------------------------
    def train_forward(self, x, t, conds: tuple, push):
        """The layer chain with a graph for the push only: the embeddings and the frozen adapter run without one, the
        push [B, 1024] is added to every caption token after the adapter, and the transformer blocks are recomputed
        in the backward pass (activation checkpointing)."""
        import torch
        from torch.utils.checkpoint import checkpoint
        inputs = (x, t, *conds)
        with torch.no_grad():
            for layer in self.layers[:self.adapter_at + 1]:
                inputs = layer(inputs)
        x_, temb, ctx, *rest = (v.detach() for v in inputs)
        inputs = (x_, temb, self.add_at_tokens(ctx, conds[3], push[:, None, :]), *rest)
        for layer in self.layers[self.adapter_at + 1:-1]:
            inputs = checkpoint(layer, inputs, use_reentrant=False)
        return self.layers[-1](inputs)

    def train_loss(self, latents, conds: tuple, push):
        """The trainer's objective on one batch (models/cosmos_predict2.py prepare_inputs + get_loss_fn: logit-normal
        t, x_t = (1 - t) x0 + t noise, mean squared error to noise - x0, no mask) with `push` added after the adapter."""
        import torch
        pe, am, ids, tm = conds
        feats, (target, _) = self.model.prepare_inputs({"latents": latents, "mask": None, "prompt_embeds": pe,
                                                        "attn_mask": am, "t5_input_ids": ids, "t5_attn_mask": tm})
        x, t, *c = feats
        return self._loss_fn(self.train_forward(x, t, tuple(c), push), (target, torch.tensor([])))

    # ---- sampling ---------------------------------------------------------------------------------
    def _forward(self, x, t, conds: tuple, context_add=None, query_add=None, query_mask=None):
        handle = None
        if query_add is not None:                          # added to the adapter's queries at the caption's T5 tokens
            mask = conds[3] if query_mask is None else query_mask      # (or at the positions query_mask marks)

            def hook(mod, inp, out):
                return out + mask.to(out.dtype)[..., None] * query_add.to(out.device, out.dtype)
            handle = self.model.transformer.llm_adapter.in_proj.register_forward_hook(hook)
        try:
            inputs = (x, t, *conds)
            for i, layer in enumerate(self.layers):
                inputs = layer(inputs)
                if i == self.adapter_at and context_add is not None:
                    x_, temb, ctx, *rest = inputs
                    inputs = (x_, temb, self.add_at_tokens(ctx, conds[3], context_add), *rest)
        finally:
            if handle is not None:
                handle.remove()
        return inputs

    def generate(self, prompts: list[str], seeds: list[int], *, res: int, steps: int, cfg: float, shift: float,
                 negative: str = "", batch: int = 8, source_add=None, context_add=None, uncond_add=None,
                 t5_prompts: "list[str] | None" = None, query_add=None, source_token=None, slot_word: "str | None" = None,
                 slot_query=None, slot_source=None) -> list:
        """PIL images, one per (prompt, seed). source_add / context_add: a 1024-vector added to every prompt token
        before / after the LLM adapter, on the conditional branch; uncond_add: the same after the adapter on the
        negative prompt's branch (a trained push goes on both branches, as a LoRA acts; e001's dial on neither).
        t5_prompts: per image, the prompt whose T5 token ids the adapter reads in place of the prompt's own, while
        the Qwen3 states stay the prompt's (the adapter's two inputs from one caption, split). query_add: a vector
        added to the adapter's query embeddings (before its blocks) at the caption's T5 tokens; source_token: one
        extra source position appended after the caption's Qwen3 tokens. slot_query / slot_source: a vector added to
        the query embedding / the Qwen3 state at slot_word's own tokens only (slot_masks). All on the conditional
        branch only."""
        import torch
        from utils.previews import to_pil
        if len(prompts) != len(seeds):
            raise ValueError("one seed per prompt")
        if t5_prompts is not None and len(t5_prompts) != len(prompts):
            raise ValueError("one T5 prompt per prompt")
        if (slot_query is not None or slot_source is not None) and (slot_word is None or query_add is not None
                                                                   or t5_prompts is not None):
            raise ValueError("a slot push needs slot_word, and takes neither query_add nor t5_prompts")
        unc = self.encode([negative]) if cfg > 1 else None
        out: list = []
        with torch.no_grad():
            for i in range(0, len(prompts), batch):
                p, s = list(prompts[i:i + batch]), list(seeds[i:i + batch])
                n = len(p)
                conds = self.encode(p)
                if t5_prompts is not None:                         # both encodings pad to the same 512 tokens
                    t5 = self.encode(list(t5_prompts[i:i + batch]))
                    conds = (conds[0], conds[1], t5[2], t5[3])
                if source_add is not None:
                    conds = (self.add_at_tokens(conds[0], conds[1], source_add), *conds[1:])
                if source_token is not None:
                    conds = self.append_source_token(conds, source_token)
                q_add, q_mask = query_add, None
                if slot_query is not None or slot_source is not None:
                    m_t5, m_q = self.slot_masks(p, slot_word)
                    if slot_source is not None:
                        conds = (conds[0] + m_q.to(conds[0].dtype)[..., None]
                                 * slot_source.to(conds[0].device, conds[0].dtype), *conds[1:])
                    if slot_query is not None:
                        q_add, q_mask = slot_query, m_t5
                un = tuple(u.expand(n, *u.shape[1:]).contiguous() for u in unc) if unc is not None else None
                x = torch.cat([torch.randn((1, 16, res // 8, res // 8), device="cuda",
                                           generator=torch.Generator(device="cuda").manual_seed(int(sd)))
                               for sd in s]).unsqueeze(2)
                self.model.set_sample_schedule(steps, shift)           # a fresh Euler schedule (rewinds its index)
                sch = self.model.scheduler
                for step in sch.timesteps:
                    t = (step / 1000).float().reshape(1).repeat(n)
                    v = self._forward(x, t, conds, context_add, q_add, q_mask).float()
                    if un is not None:
                        vu = self._forward(x, t, un, uncond_add).float()
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
        self._cbase = None

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
        raise NotImplementedError("the Anima bed runs through run_flavor_test(), run_attribute_screen() and "
                                  "run_sequence()")

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
        rec = {"model": "Anima-Base v1.0, no LoRA",
               "images": f"{self._need('resolution')} x {self._need('resolution')}, Euler, {self.GEN_STEPS} steps, "
                         f"guidance {self.GEN_CFG}, shift {self.GEN_SHIFT:g}",
               "prompts": "the model card's quality prefix + an illustration of <scene>; the card's negative prompt",
               "scenes x seeds": f"32 x {len(ax.FLAVOR_TEST_SEEDS)} ({', '.join(map(str, ax.FLAVOR_TEST_SEEDS))})"}
        return {**rec, **self._card()}

    def _card(self) -> dict:
        """The GPU that ran the experiment ({} when setup() did not record one)."""
        name = (self.state.get("gpu") or {}).get("name")
        return {"GPU": name} if name else {}

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
        n_sets = 3 + len(ax.DIAL_SITES) * len(ax.DIAL_ALPHAS)
        print(f"[anima] e001: {n_sets} sets of {len(cells)} images (the words: neutral, upbeat, downbeat; the dial: "
              f"{len(ax.DIAL_SITES)} sites x {len(ax.DIAL_ALPHAS)} strengths), one line per set", flush=True)
        eta = _sr._Eta(self.TAG, "e001", n_sets * len(cells))
        S = [sd for _, sd in cells]
        P = {f: [ax.TEMPLATES[f][0].format(s=SUBJECTS[si]) for si, _ in cells] for f in ("neutral", "up", "down")}
        imgs, feats, scores = {}, {}, {}
        for f in ("neutral", "up", "down"):                       # the words
            imgs[f] = self._render_tracked(P[f], S, eta)
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
                ims = self._render_tracked(P["neutral"], S, eta,
                                           **({"source_add": vec} if site == "source" else {"context_add": vec}))
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

    # ---- e020: the route split ---------------------------------------------------------------------------------------
    def run_appended_split(self, *, force: bool = False) -> dict:
        """e021: the route split with the mood words appended after the scene (the query half, the source half, the
        pair; the gate is the source half added to the query half on the downbeat words)."""
        return self.run_route_split(test_id=ax.APPENDED_TEST_ID, force=force)

    def run_route_split(self, test_id: str = ax.ROUTE_TEST_ID, *, force: bool = False) -> dict:
        """A route split (e020, e021) into its own folder of cfg.repo_id: the mood words through one of the adapter's two
        readings of the caption at a time (Qwen3's states, the T5 ids). Skipped when the repo lists it as done
        (force=True reruns)."""
        self._need_model()
        _drop_torchao()
        test = ax.ROUTE_TESTS[test_id]
        repo = _HubRepo(self.cfg.repo_id, self.state.get("hf_token"))
        metas = repo.metas()
        try:
            self._publish_index(repo, metas)
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"cannot write to {self.cfg.repo_id} ({e}); the HF_TOKEN needs WRITE access") from e
        rid = test.id
        if metas.get(rid, {}).get("status") == "done" and not force:
            print(f"[anima] {rid} is done already (force=True reruns it)", flush=True)
            return metas[rid]
        base = f"experiments/{rid}"
        recipe = self._flavor_recipe()
        meta = {"id": rid, "title": test.title, "date": "2026-10-04", "kind": "route_split", "status": "running",
                "recipe": recipe}
        repo.commit({f"{base}/meta.json": sx.dumps(meta), f"{base}/README.md": ax.render_route_split_readme(meta, recipe)},
                    f"{rid}: started (README)")
        out_dir = Path(self._need("data_root")) / "experiments" / rid
        out_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        try:
            result = self._route_split(out_dir, test)
        except Exception as e:  # noqa: BLE001
            meta.update(status="failed", error=f"{type(e).__name__}: {e}"[:500])
            self._safe(lambda: repo.commit({f"{base}/meta.json": sx.dumps(meta),
                                            f"{base}/README.md": ax.render_route_split_readme(meta, recipe)},
                                           f"{rid}: failed"))
            raise
        e001 = metas.get(ax.FLAVOR_TEST_ID, {}).get("result", {})
        if e001.get("up_words") and e001.get("down_words"):
            result["e001_words"] = {"up": e001["up_words"]["mean"], "down": e001["down_words"]["mean"]}
        summary = ax.route_summary(result, test)
        meta.update(status="done", result={k: v for k, v in result.items() if k != "cells"}, summary=summary,
                    seconds=round(time.time() - t0), finished_utc=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"))
        (out_dir / "result.json").write_text(json.dumps(result, indent=1, default=float), encoding="utf-8")
        repo.commit({f"{base}/meta.json": sx.dumps(meta), f"{base}/README.md": ax.render_route_split_readme(meta, recipe),
                     f"{base}/result.json": out_dir / "result.json", f"{base}/sheet_routes.jpg": out_dir / "sheet_routes.jpg"},
                    f"{rid}: result")
        metas[rid] = meta
        self._safe(lambda: self._publish_index(repo, metas))
        print(f"[anima] {rid} done in {meta['seconds']} s: {summary}", flush=True)
        return meta

    def _route_split(self, out_dir: Path, test: "ax.RouteTest | None" = None) -> dict:
        import numpy as np
        test = test or ax.ROUTE_TESTS[ax.ROUTE_TEST_ID]
        tag = test.id.split("_")[0]
        self._eval_pipe()
        self._assert_stock()
        seeds = list(ax.ROUTE_SEEDS)
        cells = [(si, sd) for si in range(len(SUBJECTS)) for sd in seeds]
        print(f"[anima] {tag}: {len(ax.ROUTE_SETS)} sets of {len(cells)} images (neutral; the words; the words through "
              "Qwen3's states only; through the T5 ids only), one line per set", flush=True)
        eta = _sr._Eta(self.TAG, tag, len(ax.ROUTE_SETS) * len(cells))
        S = [sd for _, sd in cells]
        P = {f: [ax.route_prompts(test, f, SUBJECTS[si]) for si, _ in cells] for f in ("neutral", "up", "down")}
        imgs, feats, scores, kept = {}, {}, {}, {}
        for key, (fq, ft) in ax.ROUTE_SETS.items():                # neutral first: content kept is read against it
            imgs[key] = self._render_routes(P[fq], P[ft], S, eta)
            feats[key], sc = self._score(imgs[key])
            scores[key] = [float(x) for x in sc]
            kept[key] = float(np.mean((feats[key] * feats["neutral"]).sum(-1)))
            print(f"[anima] {tag} {key}: mood score {np.mean(scores[key]):+.3f}, content kept {kept[key]:.3f}", flush=True)
        result = ax.route_reads(scores)
        result.update(content_kept=kept, mood_score={k: float(np.mean(v)) for k, v in scores.items()})
        rows_first = [i for i, (_, sd) in enumerate(cells) if sd == seeds[0]][::4]      # 8 scenes, first seed
        _grid([[imgs[k][i] for k in ax.ROUTE_SETS] for i in rows_first], out_dir / "sheet_routes.jpg")
        result["cells"] = [{"scene": SUBJECTS[si], "seed": sd, **{k: scores[k][j] for k in scores}}
                           for j, (si, sd) in enumerate(cells)]
        return result

    def _render_routes(self, qwen_prompts: list[str], t5_prompts: list[str], seeds: list[int], eta=None) -> list:
        """_render_tracked with the T5 ids from t5_prompts, chunked with their images (the plain prompts when equal)."""
        if qwen_prompts == t5_prompts:
            return self._render_tracked(qwen_prompts, seeds, eta)
        out, b = [], self.cfg.gen_batch
        for i in range(0, len(qwen_prompts), b):
            out += self._render(qwen_prompts[i:i + b], seeds[i:i + b], t5_prompts=t5_prompts[i:i + b])
            if eta is not None:
                eta.add(len(qwen_prompts[i:i + b]))
        return out

    # ---- e022: the query-site dial ------------------------------------------------------------------------------------
    def run_query_dial(self, *, force: bool = False) -> dict:
        """e022 into its own folder of cfg.repo_id: a mood direction added to the adapter's queries, alone and paired with
        a source token (a bare direction, a state-shaped token, and the uniform source direction as the control).
        Skipped when the repo lists it as done (force=True reruns)."""
        self._need_model()
        _drop_torchao()
        repo = _HubRepo(self.cfg.repo_id, self.state.get("hf_token"))
        metas = repo.metas()
        try:
            self._publish_index(repo, metas)
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"cannot write to {self.cfg.repo_id} ({e}); the HF_TOKEN needs WRITE access") from e
        qid = ax.QUERY_TEST_ID
        if metas.get(qid, {}).get("status") == "done" and not force:
            print(f"[anima] {qid} is done already (force=True reruns it)", flush=True)
            return metas[qid]
        base = f"experiments/{qid}"
        recipe = self._flavor_recipe()
        meta = {"id": qid, "title": ax.QUERY_TEST_TITLE, "date": "2026-10-04", "kind": "query_dial", "status": "running",
                "recipe": recipe}
        repo.commit({f"{base}/meta.json": sx.dumps(meta), f"{base}/README.md": ax.render_query_dial_readme(meta, recipe)},
                    f"{qid}: started (README)")
        out_dir = Path(self._need("data_root")) / "experiments" / qid
        out_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        try:
            result = self._query_dial(out_dir)
        except Exception as e:  # noqa: BLE001
            meta.update(status="failed", error=f"{type(e).__name__}: {e}"[:500])
            self._safe(lambda: repo.commit({f"{base}/meta.json": sx.dumps(meta),
                                            f"{base}/README.md": ax.render_query_dial_readme(meta, recipe)},
                                           f"{qid}: failed"))
            raise
        summary = ax.query_summary(result)
        meta.update(status="done", result={k: v for k, v in result.items() if k != "cells"}, summary=summary,
                    seconds=round(time.time() - t0), finished_utc=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"))
        (out_dir / "result.json").write_text(json.dumps(result, indent=1, default=float), encoding="utf-8")
        repo.commit({f"{base}/meta.json": sx.dumps(meta), f"{base}/README.md": ax.render_query_dial_readme(meta, recipe),
                     f"{base}/result.json": out_dir / "result.json", f"{base}/sheet_query.jpg": out_dir / "sheet_query.jpg"},
                    f"{qid}: result")
        metas[qid] = meta
        self._safe(lambda: self._publish_index(repo, metas))
        print(f"[anima] {qid} done in {meta['seconds']} s: {summary}", flush=True)
        return meta

    def _query_dial(self, out_dir: Path) -> dict:
        import numpy as np
        pipe = self._eval_pipe()
        self._assert_stock()
        seeds = list(ax.FLAVOR_TEST_SEEDS)
        cells = [(si, sd) for si in range(len(SUBJECTS)) for sd in seeds]
        S = [sd for _, sd in cells]
        P = [ax.TEMPLATES["neutral"][0].format(s=SUBJECTS[si]) for si, _ in cells]
        per_scene = {f: [ax.TEMPLATES[f][0].format(s=s) for s in SUBJECTS] for f in ("neutral", "up", "down")}

        def mean_token(x, m):
            m = m.to(x.dtype)
            return (x * m[..., None]).sum(1) / m.sum(1, keepdim=True)

        q = {f: pipe.query_states(per_scene[f]) for f in per_scene}
        src = {f: self._site_states(per_scene[f])["source"] for f in per_scene}
        d_q = ((mean_token(*q["up"]) - mean_token(*q["down"])) / 2).mean(0)
        d_s = ((mean_token(*src["up"]) - mean_token(*src["down"])) / 2).mean(0)
        xs, ms = src["neutral"]
        xq, mq = q["neutral"]
        src_norm = float(xs.norm(dim=-1)[ms.bool()].mean())
        tokens = {"pair_dir": {1: d_s / d_s.norm() * src_norm, -1: -d_s / d_s.norm() * src_norm},
                  "pair_state": {1: pipe.word_states(per_scene["up"], ax.QUERY_WORDS["up"]).mean(0),
                                 -1: pipe.word_states(per_scene["down"], ax.QUERY_WORDS["down"]).mean(0)}}
        sizes = {"query token mean size": float(xq.norm(dim=-1)[mq.bool()].mean()), "query direction": float(d_q.norm()),
                 "Qwen3 state mean size": src_norm, "source direction": float(d_s.norm()),
                 "state token, upbeat": float(tokens["pair_state"][1].norm()),
                 "state token, downbeat": float(tokens["pair_state"][-1].norm())}
        sets = ax.query_sets()
        print(f"[anima] e022: {len(sets)} sets of {len(cells)} images (the neutral images, then the query dial alone and "
              f"paired with a source token, at alpha {', '.join(f'{a:+g}' for a in ax.DIAL_ALPHAS)}), one line per set; "
              + ", ".join(f"{k} {v:.2f}" for k, v in sizes.items()), flush=True)
        eta = _sr._Eta(self.TAG, "e022", len(sets) * len(cells))
        imgs, feats, scores, kept = {}, {}, {}, {}
        for key in sets:                                         # neutral first: content kept is read against it
            kw: dict = {}
            if key != "neutral":
                form, a = key.split("@")
                a = float(a)
                kw["query_add"] = d_q * a
                if form in tokens:
                    kw["source_token"] = tokens[form][1 if a > 0 else -1]
                elif form == "pair_uniform":
                    kw["source_add"] = d_s * a
            imgs[key] = self._render_tracked(P, S, eta, **kw)
            feats[key], sc = self._score(imgs[key])
            scores[key] = [float(x) for x in sc]
            kept[key] = float(np.mean((feats[key] * feats["neutral"]).sum(-1)))
            print(f"[anima] e022 {key}: mood score {np.mean(scores[key]):+.3f}, content kept {kept[key]:.3f}", flush=True)
        result = ax.query_dial_reads(scores)
        result.update(sizes=sizes, content_kept=kept, mood_score={k: float(np.mean(v)) for k, v in scores.items()})
        rows_first = [i for i, (_, sd) in enumerate(cells) if sd == seeds[0]][::4]      # 8 scenes, first seed
        cols = ["neutral"] + [f"{f}@{a:+g}" for f in ax.QUERY_FORMS for a in (min(ax.DIAL_ALPHAS), max(ax.DIAL_ALPHAS))]
        _grid([[imgs[k][i] for k in cols] for i in rows_first], out_dir / "sheet_query.jpg")
        result["cells"] = [{"scene": SUBJECTS[si], "seed": sd, **{k: scores[k][j] for k in scores}}
                           for j, (si, sd) in enumerate(cells)]
        return result

    # ---- e026: the word split -----------------------------------------------------------------------------------------
    def run_word_split(self, *, force: bool = False) -> dict:
        """e026 into its own folder of cfg.repo_id: single mood words through the adapter's query half alone or through
        both halves, whole T5 tokens against shattered ones. Skipped when the repo lists it as done (force=True reruns)."""
        self._need_model()
        _drop_torchao()
        repo = _HubRepo(self.cfg.repo_id, self.state.get("hf_token"))
        metas = repo.metas()
        try:
            self._publish_index(repo, metas)
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"cannot write to {self.cfg.repo_id} ({e}); the HF_TOKEN needs WRITE access") from e
        rid = ax.WORD_TEST_ID
        if metas.get(rid, {}).get("status") == "done" and not force:
            print(f"[anima] {rid} is done already (force=True reruns it)", flush=True)
            return metas[rid]
        base = f"experiments/{rid}"
        recipe = self._flavor_recipe()
        model = self._eval_pipe().model                 # each word as Anima's two tokenizers cut it, after a space
        pieces = {w: {"t5": list(model.t5_tokenizer.tokenize(" " + w)), "qwen": list(model.tokenizer.tokenize(" " + w))}
                  for w in ax.word_list()}
        meta = {"id": rid, "title": ax.WORD_TEST_TITLE, "date": "2026-10-04", "kind": "word_split", "status": "running",
                "recipe": recipe, "pieces": pieces}
        repo.commit({f"{base}/meta.json": sx.dumps(meta), f"{base}/README.md": ax.render_word_split_readme(meta, recipe)},
                    f"{rid}: started (README)")
        out_dir = Path(self._need("data_root")) / "experiments" / rid
        out_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        try:
            result = self._word_split(out_dir)
        except Exception as e:  # noqa: BLE001
            meta.update(status="failed", error=f"{type(e).__name__}: {e}"[:500])
            self._safe(lambda: repo.commit({f"{base}/meta.json": sx.dumps(meta),
                                            f"{base}/README.md": ax.render_word_split_readme(meta, recipe)},
                                           f"{rid}: failed"))
            raise
        summary = ax.word_summary(result)
        meta.update(status="done", result={k: v for k, v in result.items() if k != "cells"}, summary=summary,
                    seconds=round(time.time() - t0), finished_utc=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"))
        (out_dir / "result.json").write_text(json.dumps(result, indent=1, default=float), encoding="utf-8")
        repo.commit({f"{base}/meta.json": sx.dumps(meta), f"{base}/README.md": ax.render_word_split_readme(meta, recipe),
                     f"{base}/result.json": out_dir / "result.json", f"{base}/sheet_words.jpg": out_dir / "sheet_words.jpg"},
                    f"{rid}: result")
        metas[rid] = meta
        self._safe(lambda: self._publish_index(repo, metas))
        print(f"[anima] {rid} done in {meta['seconds']} s: {summary}", flush=True)
        return meta

    def _word_split(self, out_dir: Path) -> dict:
        import numpy as np
        self._eval_pipe()
        self._assert_stock()
        cells = [(si, sd) for si in range(len(SUBJECTS)) for sd in ax.WORD_SEEDS]
        sets = ax.word_sets()
        print(f"[anima] e026: {len(sets)} sets of {len(cells)} images (neutral; per word, the word through both readings "
              "and through the T5 ids only), one line per set", flush=True)
        eta = _sr._Eta(self.TAG, "e026", len(sets) * len(cells))
        S = [sd for _, sd in cells]
        P = {w: [ax.word_prompt(w, SUBJECTS[si]) for si, _ in cells] for w in [None, *ax.word_list()]}
        rows = list(range(len(cells)))[::4]                                # 8 scenes for the sheet
        cols = ["neutral"] + [k for ws in ax.WORD_GROUPS.values() for k in (f"words_{ws[0]}", f"t5_{ws[0]}")]
        sheet, feat0, scores, kept = {}, None, {}, {}
        for key, (wq, wt) in sets.items():                 # neutral first: content kept is read against it
            ims = self._render_routes(P[wq], P[wt], S, eta)
            fa, sc = self._score(ims)
            feat0 = fa if feat0 is None else feat0
            scores[key] = [float(x) for x in sc]
            kept[key] = float(np.mean((fa * feat0).sum(-1)))
            if key in cols:
                sheet[key] = [ims[i] for i in rows]
            print(f"[anima] e026 {key}: mood score {np.mean(scores[key]):+.3f}, content kept {kept[key]:.3f}", flush=True)
        result = ax.word_split_reads(scores)
        result.update(content_kept=kept, mood_score={k: float(np.mean(v)) for k, v in scores.items()}, sheet_columns=cols)
        _grid([[sheet[k][r] for k in cols] for r in range(len(rows))], out_dir / "sheet_words.jpg")
        result["cells"] = [{"scene": SUBJECTS[si], "seed": sd, **{k: scores[k][j] for k in scores}}
                           for j, (si, sd) in enumerate(cells)]
        return result

    # ---- e027: the slot pair ------------------------------------------------------------------------------------------
    def run_slot_pair(self, *, force: bool = False) -> dict:
        """e027 into its own folder of cfg.repo_id: a word-sized mood push at one word's position, on the adapter's query
        side, its source side, or both (a matched pair). Skipped when the repo lists it as done (force=True reruns)."""
        self._need_model()
        _drop_torchao()
        repo = _HubRepo(self.cfg.repo_id, self.state.get("hf_token"))
        metas = repo.metas()
        try:
            self._publish_index(repo, metas)
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"cannot write to {self.cfg.repo_id} ({e}); the HF_TOKEN needs WRITE access") from e
        rid = ax.SLOT_TEST_ID
        if metas.get(rid, {}).get("status") == "done" and not force:
            print(f"[anima] {rid} is done already (force=True reruns it)", flush=True)
            return metas[rid]
        base = f"experiments/{rid}"
        recipe = self._flavor_recipe()
        meta = {"id": rid, "title": ax.SLOT_TEST_TITLE, "date": "2026-10-04", "kind": "slot_pair", "status": "running",
                "recipe": recipe}
        repo.commit({f"{base}/meta.json": sx.dumps(meta), f"{base}/README.md": ax.render_slot_pair_readme(meta, recipe)},
                    f"{rid}: started (README)")
        out_dir = Path(self._need("data_root")) / "experiments" / rid
        out_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        try:
            result = self._slot_pair(out_dir)
        except Exception as e:  # noqa: BLE001
            meta.update(status="failed", error=f"{type(e).__name__}: {e}"[:500])
            self._safe(lambda: repo.commit({f"{base}/meta.json": sx.dumps(meta),
                                            f"{base}/README.md": ax.render_slot_pair_readme(meta, recipe)},
                                           f"{rid}: failed"))
            raise
        summary = ax.slot_summary(result)
        meta.update(status="done", result={k: v for k, v in result.items() if k != "cells"}, summary=summary,
                    seconds=round(time.time() - t0), finished_utc=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"))
        (out_dir / "result.json").write_text(json.dumps(result, indent=1, default=float), encoding="utf-8")
        repo.commit({f"{base}/meta.json": sx.dumps(meta), f"{base}/README.md": ax.render_slot_pair_readme(meta, recipe),
                     f"{base}/result.json": out_dir / "result.json", f"{base}/sheet_slot.jpg": out_dir / "sheet_slot.jpg"},
                    f"{rid}: result")
        metas[rid] = meta
        self._safe(lambda: self._publish_index(repo, metas))
        print(f"[anima] {rid} done in {meta['seconds']} s: {summary}", flush=True)
        return meta

    def _slot_pair(self, out_dir: Path) -> dict:
        import numpy as np
        pipe = self._eval_pipe()
        self._assert_stock()
        cells = [(si, sd) for si in range(len(SUBJECTS)) for sd in ax.SLOT_SEEDS]
        S = [sd for _, sd in cells]
        up, down = ax.SLOT_REFERENCE
        P = {w: [ax.slot_prompt(w, SUBJECTS[si]) for si, _ in cells] for w in (ax.SLOT_WORD, up, down)}
        q = pipe.word_queries((up, down))                    # the word table's rows through in_proj
        d_q = q[0] - q[1]
        xq, mq = pipe.query_states(P[ax.SLOT_WORD])
        q_size = float(xq.norm(dim=-1)[mq.bool()].mean())
        d_s = pipe.word_states(P[up], (up,)).mean(0) - pipe.word_states(P[down], (down,)).mean(0)
        conds = pipe.encode(P[ax.SLOT_WORD])
        s_size = float(conds[0].float().norm(dim=-1)[conds[1].bool()].mean())
        sizes = {"query token mean size": q_size, "query direction before scaling": float(d_q.norm()),
                 "Qwen3 state mean size": s_size, "source direction before scaling": float(d_s.norm())}
        d_q, d_s = d_q / d_q.norm() * q_size, d_s / d_s.norm() * s_size       # one unit of alpha = one word's size
        sets = ax.slot_sets()
        print(f"[anima] e027: {len(sets)} sets of {len(cells)} images (the slot prompt, the real words at the slot, the "
              f"query / pair / answer at the slot per alpha), one line per set; sizes "
              + ", ".join(f"{k} {v:.1f}" for k, v in sizes.items()), flush=True)
        eta = _sr._Eta(self.TAG, "e027", len(sets) * len(cells))
        rows = list(range(len(cells)))[::4]                                    # 8 scenes for the sheet
        cols = ["slot", f"word_{up}", f"word_{down}"] + [f"{f}@{a:+g}" for f in ax.SLOT_FORMS
                                                         for a in (min(ax.SLOT_ALPHAS), max(ax.SLOT_ALPHAS))]
        sheet, feat0, scores, kept = {}, None, {}, {}
        for key in sets:                                   # the slot prompt first: content kept is read against it
            if key == "slot" or key.startswith("word_"):
                ims = self._render_tracked(P[ax.SLOT_WORD if key == "slot" else key[len("word_"):]], S, eta)
            else:
                form, a = key.split("@")
                kw: dict = {"slot_word": ax.SLOT_WORD}
                if form in ("Q", "P"):
                    kw["slot_query"] = d_q * float(a)
                if form in ("P", "S"):
                    kw["slot_source"] = d_s * float(a)
                ims = self._render_tracked(P[ax.SLOT_WORD], S, eta, **kw)
            fa, sc = self._score(ims)
            feat0 = fa if feat0 is None else feat0
            scores[key] = [float(x) for x in sc]
            kept[key] = float(np.mean((fa * feat0).sum(-1)))
            if key in cols:
                sheet[key] = [ims[i] for i in rows]
            print(f"[anima] e027 {key}: mood score {np.mean(scores[key]):+.3f}, content kept {kept[key]:.3f}", flush=True)
        result = ax.slot_pair_reads(scores)
        result.update(sizes=sizes, content_kept=kept, mood_score={k: float(np.mean(v)) for k, v in scores.items()},
                      sheet_columns=cols)
        _grid([[sheet[k][r] for k in cols] for r in range(len(rows))], out_dir / "sheet_slot.jpg")
        result["cells"] = [{"scene": SUBJECTS[si], "seed": sd, **{k: scores[k][j] for k in scores}}
                           for j, (si, sd) in enumerate(cells)]
        return result

    # ---- e012: the attribute screen ----------------------------------------------------------------------
    def _tagger(self):
        if getattr(self, "_tagger_fns", None) is None:
            print(f"[{self.TAG}] loading the tagger judge ({ax.TAGGER}; a 1.3 GB download the first time)...", flush=True)
            self._tagger_fns = _wd_tagger(ax.TAGGER, [t for p, m, _ in ax.ATTRIBUTES.values() for t in (p, m) if t])
        return self._tagger_fns

    def _tag_scores(self, imgs: list) -> dict:
        """{attribute: [score per image]}: the tagger's log-odds of the + tag minus those of the - tag (the + tag
        alone where there is none), probabilities clipped to [1e-4, 1 - 1e-4]."""
        import numpy as np
        probs, index = self._tagger()
        p = probs(imgs)

        def logodds(tag):
            q = np.clip(p[:, index[tag]], 1e-4, 1 - 1e-4)
            return np.log(q / (1 - q))

        return {a: [float(x) for x in logodds(plus) - (logodds(minus) if minus else 0.0)]
                for a, (plus, minus, _) in ax.ATTRIBUTES.items()}

    def _attr_recipe(self) -> dict:
        return {"model": "Anima-Base v1.0, no LoRA",
                "images": f"{self._need('resolution')} x {self._need('resolution')}, Euler, {self.GEN_STEPS} steps, "
                          f"guidance {self.GEN_CFG}, shift {self.GEN_SHIFT:g}",
                "prompts": "the model card's quality prefix + 1girl, solo, <tag>, upper body, <outfit>, <setting>; the "
                           "card's negative prompt",
                "cells": f"{len(ax.CHARACTERS)} characters x seeds {', '.join(map(str, ax.ATTR_SEEDS))}",
                "slider site": "the adapter's output (what the image model reads)",
                "judge": "WD EVA02-Large tagger v3 (log-odds); content kept by CLIP ViT-L/14"}

    def run_attribute_screen(self, *, force: bool = False) -> dict:
        """e012 into its own folder of cfg.repo_id: attribute words, attribute sliders after the adapter, their
        cross-talk and the overlap of their directions. Skipped when the repo lists it as done (force=True reruns)."""
        self._need_model()
        _drop_torchao()
        repo = _HubRepo(self.cfg.repo_id, self.state.get("hf_token"))
        metas = repo.metas()
        try:
            self._publish_index(repo, metas)
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"cannot write to {self.cfg.repo_id} ({e}); the HF_TOKEN needs WRITE access") from e
        eid = ax.ATTR_TEST_ID
        if metas.get(eid, {}).get("status") == "done" and not force:
            print(f"[anima] {eid} is done already (force=True reruns it)", flush=True)
            return metas[eid]
        base = f"experiments/{eid}"
        recipe = self._attr_recipe()
        meta = {"id": eid, "title": ax.ATTR_TEST_TITLE, "date": ax.DATE, "kind": "attribute_screen",
                "status": "running", "recipe": recipe}
        repo.commit({f"{base}/meta.json": sx.dumps(meta),
                     f"{base}/README.md": ax.render_attribute_screen_readme(meta, recipe)}, f"{eid}: started (README)")
        out_dir = Path(self._need("data_root")) / "experiments" / eid
        out_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        try:
            result = self._attribute_screen(out_dir)
        except Exception as e:  # noqa: BLE001
            meta.update(status="failed", error=f"{type(e).__name__}: {e}"[:500])
            self._safe(lambda: repo.commit({f"{base}/meta.json": sx.dumps(meta),
                                            f"{base}/README.md": ax.render_attribute_screen_readme(meta, recipe)},
                                           f"{eid}: failed"))
            raise
        summary = "; ".join(f"{a} {v['OUTCOME']}" for a, v in result["attributes"].items())
        meta.update(status="done", result={k: v for k, v in result.items() if k != "cells"}, summary=summary,
                    seconds=round(time.time() - t0),
                    finished_utc=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"))
        (out_dir / "result.json").write_text(json.dumps(result, indent=1, default=float), encoding="utf-8")
        files = {f"{base}/meta.json": sx.dumps(meta), f"{base}/README.md": ax.render_attribute_screen_readme(meta, recipe),
                 f"{base}/result.json": out_dir / "result.json"}
        for f in sorted(out_dir.glob("sheet_*.jpg")):
            files[f"{base}/{f.name}"] = f
        repo.commit(files, f"{eid}: result")
        metas[eid] = meta
        self._safe(lambda: self._publish_index(repo, metas))
        print(f"[anima] {eid} done in {_sr._hms(meta['seconds'])}: {summary}", flush=True)
        return meta

    def _attribute_screen(self, out_dir: Path) -> dict:
        import numpy as np
        import torch
        self._eval_pipe()
        self._assert_stock()
        self._tagger()                            # both judges before any image: a judge fault fails in minute one
        self._judge()
        cells = ax.attr_cells()
        S = [sd for _, sd in cells]
        chars = range(len(ax.CHARACTERS))

        def token_means(tag):                    # per character: the prompt's mean token vector at both sites
            st = self._site_states([ax.attr_prompt(tag, ci) for ci in chars])
            return {site: (x * m[..., None]).sum(1) / m.sum(1, keepdim=True) for site, (x, m) in st.items()}

        plain = token_means(None)
        dirs: dict = {site: {} for site in plain}
        for a, (plus, minus, _) in ax.ATTRIBUTES.items():
            hi, lo = token_means(plus), (token_means(minus) if minus else plain)
            for site in dirs:
                dirs[site][a] = ((hi[site] - lo[site]) / 2).mean(0)
        sets = ax.attr_sets()
        print(f"[anima] e012: {len(sets)} sets of {len(cells)} images = {len(sets) * len(cells)} (the plain prompt, "
              f"{len(ax.ATTRIBUTES)} attributes' words and sliders), one line per set", flush=True)
        eta = _sr._Eta(self.TAG, "e012", len(sets) * len(cells))
        imgs, judge, feats = {}, {}, {}
        for k, (tag, slider) in sets.items():
            kw = {"context_add": dirs[ax.ATTR_SITE][slider[0]] * slider[1]} if slider else {}
            imgs[k] = self._render_tracked([ax.attr_prompt(tag, ci) for ci, _ in cells], S, eta, **kw)
            judge[k] = self._tag_scores(imgs[k])
            feats[k], _ = self._score(imgs[k])
            own = k.split("@")[0].rstrip("+-")
            line = (f"{own} score {np.mean(judge[k][own]):+.2f} (plain prompt {np.mean(judge['neutral'][own]):+.2f})"
                    if own in ax.ATTRIBUTES else "plain prompt")
            print(f"[anima] e012 {k}: {line}", flush=True)
        reads = ax.attribute_reads(judge)
        for a, (_, _, alphas) in ax.ATTRIBUTES.items():
            kept = {f"{al:+g}": float(np.mean((feats[f"{a}@{al:+g}"] * feats["neutral"]).sum(-1))) for al in alphas}
            kept["word"] = float(np.mean((feats[f"{a}+"] * feats["neutral"]).sum(-1)))
            reads[a]["content_kept"] = kept
        overlap, norms = {}, {}
        for site, d in dirs.items():
            names = list(d)
            v = torch.stack([d[a] for a in names]).float()
            c = (torch.nn.functional.normalize(v, dim=-1) @ torch.nn.functional.normalize(v, dim=-1).T).cpu().numpy()
            overlap[site] = {a: {b: float(c[i, j]) for j, b in enumerate(names)} for i, a in enumerate(names)}
            norms[site] = {a: float(v[i].norm()) for i, a in enumerate(names)}
        rows_first = [i for i, (_, sd) in enumerate(cells) if sd == ax.ATTR_SEEDS[0]]
        for a, (_, minus, alphas) in ax.ATTRIBUTES.items():
            cols = (([f"{a}-"] if minus else []) + [f"{a}@{al:+g}" for al in alphas if al < 0] + ["neutral"]
                    + [f"{a}@{al:+g}" for al in alphas if al > 0] + [f"{a}+"])
            _grid([[imgs[k][i] for k in cols] for i in rows_first], out_dir / f"sheet_{a}.jpg")
        return {"attributes": reads, "overlap": overlap, "direction_norms": norms,
                "judge_means": {k: {a: float(np.mean(v)) for a, v in j.items()} for k, j in judge.items()},
                "cells": [{"character": ax.CHARACTERS[ci][0], "seed": sd,
                           **{k: {a: judge[k][a][n] for a in judge[k]} for k in judge}}
                          for n, (ci, sd) in enumerate(cells)]}

    # ---- e013-e015: Beatrix as a second conditioning source (a learned push after the adapter) ----------------
    def run_beatrix_connectors(self, arms: "list[str] | None" = None, *, force: bool = False) -> dict:
        """e013-e015, each into its own folder of cfg.repo_id: a push computed from Beatrix's features for a mood
        phrase (e013), the same from an untrained trunk of the same shape (e014) and a free learned vector per mood
        class (e015), trained in this process by Anima's own objective on the mood LoRAs' first-draw training images
        and scored on the held-out scenes. Arms the repo lists as done are skipped (force=True reruns them)."""
        self._need_model()
        _drop_torchao()
        repo = _HubRepo(self.cfg.repo_id, self.state.get("hf_token"))
        metas = repo.metas()
        try:
            self._publish_index(repo, metas)
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"cannot write to {self.cfg.repo_id} ({e}); the HF_TOKEN needs WRITE access") from e
        unknown = set(arms or []) - set(ax.CONNECTOR_IDS)
        if unknown:
            raise ValueError(f"unknown connector arm(s) {sorted(unknown)}; they are {ax.CONNECTOR_IDS}")
        specs = [a for a in ax.CONNECTOR_ARMS if arms is None or a.id in arms]
        todo = [a for a in specs if force or metas.get(a.id, {}).get("status") != "done"]
        print(f"[{self.TAG}] connectors -> {self.cfg.repo_id}: {len(todo)} to run "
              f"({', '.join(a.id for a in todo) or 'none'}); done already: "
              f"{', '.join(a.id for a in specs if a not in todo) or 'none'}", flush=True)
        if todo:
            t0 = time.time()
            self._eval_pipe()
            self._assert_stock()
            self._judge()                          # the judge before any GPU minute: a judge fault fails at once
            feats, phrases = self._connector_features()
            bank = self._connector_bank()
            base = self._connector_baseline()
            n_score = sum(len(ax.connector_eval_sets(phrases, a.source)) for a in todo) * len(base["prompts"])
            print(f"[{self.TAG}] the job: {len(todo)} connectors x {ax.CONNECTOR_STEPS} steps of {ax.CONNECTOR_BATCH} "
                  f"images, then {n_score} images to score; data + baseline ready in {_sr._hms(time.time() - t0)}",
                  flush=True)
            arms_eta = _sr._Eta(self.TAG, "connectors", len(todo), unit="arms done", every=0)
            for arm in todo:
                metas[arm.id] = self._connector_arm(arm, repo, metas, feats, phrases, bank, base)
                arms_eta.add()
                self._safe(lambda: self._publish_index(repo, metas))
        self._connector_cross(repo, metas)
        return {a.id: metas.get(a.id) for a in specs}

    def _connector_features(self) -> tuple:
        """({'trained', 'random': [phrases, 4096] float32}, the phrase table [{class, split, text}]) from the features
        file in the data repo (computed once with Beatrix outside the notebook: the notebook never runs Beatrix)."""
        from huggingface_hub import hf_hub_download
        from safetensors import safe_open
        repo = self.cfg.data_repo_id or _ts.DATA_REPO
        path = hf_hub_download(repo, ax.CONNECTOR_FEATURES, repo_type="dataset", token=self.state.get("hf_token") or None)
        with safe_open(path, framework="pt") as f:
            phrases = json.loads(f.metadata()["phrases"])
            feats = {k: f.get_tensor(k).float() for k in ("trained", "random")}
        if any(v.shape[0] != len(phrases) for v in feats.values()):
            raise ValueError(f"{ax.CONNECTOR_FEATURES}: {len(phrases)} phrases, features "
                             f"{[tuple(v.shape) for v in feats.values()]}")
        print(f"[{self.TAG}] Beatrix features: {len(phrases)} phrases x {feats['trained'].shape[1]} "
              f"({repo}/{ax.CONNECTOR_FEATURES})", flush=True)
        return feats, phrases

    def _connector_bank(self) -> dict:
        """The training data on the GPU: the LoRA arms' first-draw sets of the three classes (pulled from the data
        repo, or drawn when absent), their latents by the trainer's VAE path and the conditioning of their captions."""
        import torch
        from PIL import Image
        pipe = self._eval_pipe()
        keys = [(c, ax.CONNECTOR_DRAW) for c in ax.CONNECTOR_CLASSES]
        for k in keys:
            self._pull_set(*k)
        need = sum(self._missing(*k) for k in keys)
        eta = _sr._Eta(self.TAG, "training sets", need) if need else None
        items = []
        for c, sb in keys:
            img_dir = self._render_dataset(c, sb, eta=eta)
            self._push_set(c, sb)
            items += [(c, img_dir / f"{it['name']}.png", it["caption"]) for it in self._items(c, sb)]
        captions = sorted({cap for _, _, cap in items})
        res = self._need("resolution")
        eta = _sr._Eta(self.TAG, "training latents", len(items))
        lat = []
        for i in range(0, len(items), self.cfg.gen_batch):
            chunk = items[i:i + self.cfg.gen_batch]
            lat.append(pipe.encode_images([Image.open(p) for _, p, _ in chunk], res))
            eta.add(len(chunk))
        dev = getattr(pipe, "device", "cuda")
        classes = [c for c, _, _ in items]
        bank = {"latents": torch.cat(lat).to(dev), "classes": classes,
                "cap_index": torch.tensor([captions.index(cap) for _, _, cap in items], device=dev),
                "conds": tuple(t.to(dev) for t in pipe.encode(captions))}
        counts = ", ".join(f"{c} {classes.count(c)}" for c in ax.CONNECTOR_CLASSES)
        print(f"[{self.TAG}] connector data: {len(items)} images ({counts}), {len(captions)} captions, latents "
              f"{tuple(bank['latents'].shape)}", flush=True)
        return bank

    def _connector_baseline(self) -> dict:
        """The connector evaluation's cells without a push (once per session; every set is paired against them)."""
        if self._cbase is None:
            import numpy as np
            self._assert_stock()
            cells = [(si, sd) for si in HELD_OUT for sd in ax.CONNECTOR_SEEDS]
            prompts = [self.BED.neutral_caption.format(s=SUBJECTS[si]) for si, _ in cells]
            seeds = [sd for _, sd in cells]
            imgs = self._render_tracked(prompts, seeds, _sr._Eta(self.TAG, "connector baseline (no push)", len(prompts)))
            feats, scores = self._score(imgs)
            self._cbase = {"cells": cells, "prompts": prompts, "seeds": seeds, "images": imgs, "feats": feats,
                           "scores": [float(x) for x in scores]}
            print(f"[{self.TAG}] connector baseline: {len(imgs)} held-out cells, mean mood score "
                  f"{np.mean(self._cbase['scores']):+.3f}", flush=True)
        return self._cbase

    def _connector_inputs(self, arm: "ax.ConnectorArm", feats: dict, phrases: list) -> dict:
        """The arm's input rows (CPU float32) and, per class, the rows of its training inputs: the one-hot class (e015),
        the phrase features as they are (e013, e014), projected on the top whiten_k whitened components of the training
        phrases (e016, e017) or reduced to a slider value on their mood and neutral axes (e018, e019); projections are fit
        on the training phrases only and returned with the L1 size of the up-minus-down class-mean difference of the
        training inputs (it sets W's learning rate)."""
        import torch
        classes = ax.CONNECTOR_CLASSES
        if arm.source == "onehot":
            return {"inputs": torch.eye(len(classes)), "pool": {c: [i] for i, c in enumerate(classes)},
                    "row_of": {c: i for i, c in enumerate(classes)}, "projection": None, "contrast_l1": None}
        F = feats[arm.source].float()
        pool = {c: [i for i, p in enumerate(phrases) if p["class"] == c and p["split"] == "train"] for c in classes}
        if not all(pool.values()):
            raise ValueError(f"a class without training phrases: {({c: len(v) for c, v in pool.items()})}")
        out = {"inputs": F, "pool": pool, "row_of": {p["text"]: i for i, p in enumerate(phrases)}, "projection": None,
               "contrast_l1": None}
        if arm.whiten_k or arm.axis:
            proj = (connector_axis(F, pool) if arm.axis else
                    connector_whitening(F, [i for c in classes for i in pool[c]], arm.whiten_k))
            Z = (F - proj["mu"]) @ proj["V"].T / proj["scale"]
            if arm.axis:
                Z = slider_map(Z, arm.sides)
            out.update(inputs=Z, projection=proj,
                       contrast_l1=float((Z[pool["up"]].mean(0) - Z[pool["down"]].mean(0)).abs().sum()))
        return out

    def _connector_recipe(self, arm: "ax.ConnectorArm", ci: dict) -> dict:
        fan_in = int(ci["inputs"].shape[1])
        lrs = ax.connector_lrs(arm.source, fan_in, contrast_l1=ci["contrast_l1"])
        data_repo, d = self.cfg.data_repo_id or _ts.DATA_REPO, ax.CONNECTOR_DRAW
        n_feat = None if arm.source == "onehot" else int(ci["projection"]["V"].shape[1] if ci["projection"] is not None
                                                         else fan_in)
        rec = {"image model": "Anima-Base v1.0, frozen (no LoRA)",
               "input": {"trained": f"Beatrix ({ax.CONNECTOR_CHECKPOINT}): her features for the phrase ({n_feat} "
                                    "numbers)",
                         "random": f"an untrained Beatrix of the same shape (random initialisation, seed 0): its features "
                                   f"for the phrase ({n_feat} numbers), standardized the same way",
                         "onehot": "the mood class as a one-hot vector (3 numbers); no encoder"}[arm.source]}
        if ci["projection"] is not None and arm.axis:
            rec["input"] += (", reduced to a slider value: its position on the axis from the gloomy to the cheerful training "
                             "phrases' centres (at -1 and +1) and on the axis toward the neutral training phrases' centre (2 "
                             "numbers; the axes fit on the training phrases only)")
            if arm.sides != "one":
                rec["input"] += (f", then mapped to {ax.SLIDER_MAPS[arm.sides]} with a the slider value and n the neutral "
                                 "reading (3 numbers: " + {"relu": "each side of the slider has its own push direction "
                                 "and is zero on the other side", "exp": "both sides are on for every phrase and the "
                                 "slider value tilts the balance"}[arm.sides] + ")")
        elif ci["projection"] is not None:
            rec["input"] += (f", projected on the top {fan_in} principal components of the training phrases' features and "
                             "scaled to unit variance per component (fit on the training phrases only)")
        if arm.source != "onehot":
            rec["features file"] = f"https://huggingface.co/datasets/{data_repo}/blob/main/{ax.CONNECTOR_FEATURES}"
        lr = (f"{lrs['W']:g} for W and b" if arm.source == "onehot" else
              f"{lrs['W']:.3g} for W ({lrs['b']:g} x 2 / {ci['contrast_l1']:.2f}, the L1 size of the training inputs' "
              f"cheerful-minus-gloomy class-mean difference: the free vector's pace) and {lrs['b']:g} for b"
              if ci["contrast_l1"] else
              f"{lrs['W']:.3g} for W ({lrs['b']:g} divided by its fan-in, {fan_in}) and {lrs['b']:g} for b")
        rec.update({
            "push": "W f + b (1,024 numbers; W and b start at zero; float32), added to every caption token of the "
                    "adapter's output; at sampling on both guidance branches",
            "training images": f"the LoRA experiments' first draw: 192 upbeat, 192 downbeat and 192 neutral renders "
                               f"(24 scenes x seeds {d}-{d + 7}), all captioned with the neutral prompt",
            "pairing": ("each image with its class" if arm.source == "onehot" else
                        "each image with a random training phrase of its class, drawn anew every step"),
            "objective": "Anima's flow matching as the trainer computes it: logit-normal timesteps (no shift), mean "
                         "squared error to noise - x0",
            "optimizer": f"Adam, no weight decay, no warmup; learning rate {lr}",
            "length": f"{ax.CONNECTOR_STEPS} steps x {ax.CONNECTOR_BATCH} images = 5 passes over the 576 images; the "
                      "weights saved after every pass",
            "training seed": str(arm.seed),
            "evaluation": f"8 held-out scenes x seeds {', '.join(map(str, ax.CONNECTOR_SEEDS))} = 16 cells; Euler, "
                          f"{self.GEN_STEPS} steps, guidance {self.GEN_CFG}, shift {self.GEN_SHIFT:g}, "
                          f"{self._need('resolution')} px, the card's quality prefix and negative prompt",
            **self._card()})
        try:                                             # the exact training images, when the data repo holds them
            links = []
            for c in ax.CONNECTOR_CLASSES if self.cfg.data_repo_id else ():
                folder = _ts.hub_folder(f"{c}_{d}", _ts.render_spec_of(self), self._items(c, d))
                if _ts.on_hub(self.state.get("hf_token"), data_repo, folder):
                    links.append(f"[{c}](https://huggingface.co/datasets/{data_repo}/tree/main/{folder})")
            if links:
                rec["training image folders"] = ", ".join(links)
        except Exception:  # noqa: BLE001 - links are optional
            pass
        return rec

    def _connector_train(self, arm: "ax.ConnectorArm", ci: dict, bank: dict) -> dict:
        """W and b trained by Anima's objective, everything else frozen: plain Adam, no weight decay, the arm's learning
        rates (connector_lrs); each step = CONNECTOR_BATCH training images in a shuffled order (a fresh order every
        pass), each paired with a random training input of its class (ci: _connector_inputs). Returns the weights,
        their saves after every pass, the trace (loss and push sizes) and the learning rates."""
        import torch
        pipe = self._eval_pipe()
        dev = getattr(pipe, "device", "cuda")
        classes = ax.CONNECTOR_CLASSES
        torch.manual_seed(arm.seed)                      # the objective's noise and timesteps
        g = torch.Generator().manual_seed(arm.seed)      # the batch order and the phrase draws
        pool = ci["pool"]
        inputs = ci["inputs"].to(dev, torch.float32)
        width = getattr(pipe, "context_width", 1024)
        W = torch.zeros(width, inputs.shape[1], device=dev, requires_grad=True)
        b = torch.zeros(width, device=dev, requires_grad=True)
        lrs = ax.connector_lrs(arm.source, inputs.shape[1], contrast_l1=ci["contrast_l1"])
        opt = torch.optim.Adam([{"params": [W], "lr": lrs["W"]}, {"params": [b], "lr": lrs["b"]}], weight_decay=0.0)
        steps, bs, n = ax.CONNECTOR_STEPS, ax.CONNECTOR_BATCH, len(bank["classes"])
        eta = _sr._Eta(self.TAG, f"{arm.id.split('_')[0]} training", steps, unit="steps", every=float("inf"))
        order, window, trace, saves, last = [], [], [], {}, time.time()
        for step in range(1, steps + 1):
            if not order:
                order = torch.randperm(n, generator=g).tolist()
            idx, order = order[:bs], order[bs:]
            rows = []
            for i in idx:
                cand = pool[bank["classes"][i]]
                rows.append(cand[int(torch.randint(len(cand), (1,), generator=g))])
            ii = torch.tensor(idx, device=dev)
            push = inputs[rows] @ W.T + b
            loss = pipe.train_loss(bank["latents"][ii], tuple(t[bank["cap_index"][ii]] for t in bank["conds"]), push)
            if not torch.isfinite(loss):
                raise RuntimeError(f"{arm.id}: the loss is not finite at step {step}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            window.append(float(loss.detach()))
            eta.add(1)
            if step % ax.CONNECTOR_TRACE_EVERY == 0 or step == steps:
                with torch.no_grad():
                    norms = {c: float((inputs[pool[c]] @ W.T + b).norm(dim=-1).mean()) for c in classes}
                trace.append({"step": step, "loss": sum(window) / len(window), "push_norm": norms})
                window = []
            if step % ax.CONNECTOR_SAVE_EVERY == 0 or step == steps:
                saves[step] = (W.detach().cpu().clone(), b.detach().cpu().clone())
            if trace and step < steps and time.time() - last >= 30:
                last = time.time()
                print(f"[{self.TAG}] {eta.line()} | loss {trace[-1]['loss']:.4f} | push size "
                      + " / ".join(f"{c} {v:.3f}" for c, v in trace[-1]["push_norm"].items()), flush=True)
        return {"W": W.detach(), "b": b.detach(), "saves": saves, "trace": trace, "lrs": lrs,
                "seconds": time.time() - eta.t0}

    def _connector_eval(self, arm: "ax.ConnectorArm", trained: dict, ci: dict, phrases: list, base: dict) -> dict:
        """Every evaluation set of the arm: the held-out cells with the set's push on both guidance branches, scored
        against the same cells without a push; the registered reads over them."""
        import numpy as np
        import torch
        W, b = trained["W"], trained["b"]
        sets = ax.connector_eval_sets(phrases, arm.source)
        short = arm.id.split("_")[0]
        eta = _sr._Eta(self.TAG, f"{short} scoring", len(sets) * len(base["prompts"]))
        diffs, rows, imgs = {}, {}, {}
        for key, (c, split, text) in sets.items():
            f = ci["inputs"][ci["row_of"][c if text is None else text]]
            with torch.no_grad():
                push = f.to(W.device, W.dtype) @ W.T + b
            ims = self._render_tracked(base["prompts"], base["seeds"], eta, context_add=push, uncond_add=push)
            fe, sc = self._score(ims)
            d = [float(s - s0) for s, s0 in zip(sc, base["scores"])]
            keep = [float(x) for x in (fe * base["feats"]).sum(-1)]
            r = sx.arm_outcome(d, ax.DIRECTION[c])
            r["OUTCOME"] = {"FLAVOR LORA": "MOVES IT"}.get(r["OUTCOME"], r["OUTCOME"])
            rows[key] = {"class": c, "split": split, "phrase": text, **r, "content_kept": float(np.mean(keep)),
                         "push_norm": float(push.norm()), "mood_score": float(np.mean(sc)), "diffs": d, "kept": keep}
            diffs[key], imgs[key] = d, ims
            print(f"[{self.TAG}] {short} {text or c + ' vector'}: effect {r['mean']:+.3f} +- {r['se']:.3f}, content kept "
                  f"{np.mean(keep):.3f}, push size {float(push.norm()):.3f}", flush=True)
        reads = ax.connector_reads(diffs, sets, arm.source)
        for v in reads["groups"].values():
            v["content_kept"] = float(np.mean([rows[k]["content_kept"] for k in v["sets"]]))
        return {"reads": reads, "rows": rows, "images": imgs, "sets": sets}

    def _connector_arm(self, arm: "ax.ConnectorArm", repo: "_HubRepo", metas: dict, feats: dict, phrases: list,
                       bank: dict, base: dict) -> dict:
        """One connector: the README first, the training, the weights shipped, the evaluation, the result."""
        from safetensors.torch import save_file
        bdir = f"experiments/{arm.id}"
        ci = self._connector_inputs(arm, feats, phrases)
        recipe = self._connector_recipe(arm, ci)
        meta = {"id": arm.id, "title": arm.title, "date": ax.DATE, "kind": "beatrix_connector", "status": "running",
                "recipe": recipe, "phrases": None if arm.source == "onehot" else phrases}

        def readme() -> str:
            return ax.render_connector_readme(arm, meta, recipe, meta["phrases"])

        repo.commit({f"{bdir}/meta.json": sx.dumps(meta), f"{bdir}/README.md": readme()}, f"{arm.id}: started (README)")
        out_dir = Path(self._need("data_root")) / "experiments" / arm.id
        (out_dir / "connector").mkdir(parents=True, exist_ok=True)
        print(f"\n[{self.TAG}] ===== {arm.id}: {arm.title} =====", flush=True)
        t0 = time.time()
        proj = ci["projection"]
        how = ("((f - mu) @ V.T / scale) @ W.T + b" if proj is not None else "f @ W.T + b")
        if arm.axis and arm.sides != "one":
            how = f"phi((f - mu) @ V.T / scale) @ W.T + b, with phi([a, n]) = {ax.SLIDER_MAPS[arm.sides]}"
        try:
            tr = self._connector_train(arm, ci, bank)
            files: dict = {}
            for step, (w, bb) in tr["saves"].items():
                p = out_dir / "connector" / f"step{step:04d}.safetensors"
                tensors = {"W": w.contiguous(), "b": bb.contiguous()}
                if proj is not None:                     # the input projection travels with the weights
                    tensors.update({k: proj[k].contiguous() for k in ("mu", "V", "scale")})
                save_file(tensors, str(p),
                          metadata={"experiment": arm.id, "input": arm.source, "step": str(step),
                                    "input map": arm.sides if arm.axis else "none",
                                    "push": f"{how}, added to every caption token of the adapter's output, on both "
                                            "guidance branches"})
                files[f"{bdir}/connector/{p.name}"] = p
            (out_dir / "trace.json").write_text(json.dumps(tr["trace"], indent=1), encoding="utf-8")
            files[f"{bdir}/trace.json"] = out_dir / "trace.json"
            repo.commit(files, f"{arm.id}: connector weights, training trace")    # the weights ship before the evaluation
            ev = self._connector_eval(arm, tr, ci, phrases, base)
        except Exception as e:  # noqa: BLE001
            meta.update(status="failed", error=f"{type(e).__name__}: {e}"[:500])
            metas[arm.id] = meta
            self._safe(lambda: repo.commit({f"{bdir}/meta.json": sx.dumps(meta), f"{bdir}/README.md": readme()},
                                           f"{arm.id}: failed"))
            self._safe(lambda: self._publish_index(repo, metas))
            raise
        sets, rows = ev["sets"], ev["rows"]
        firsts = list(dict.fromkeys((c, s) for c, s, _ in sets.values()))
        cols = [next(k for k, v in sets.items() if (v[0], v[1]) == cs) for cs in firsts]
        idx = [i for i, (_, sd) in enumerate(base["cells"]) if sd == ax.CONNECTOR_SEEDS[0]]
        _grid([[base["images"][i]] + [ev["images"][k][i] for k in cols] for i in idx], out_dir / "sheet.jpg")
        lora = {a: {"mean": m["final"]["mean"], "se": m["final"]["se"], "OUTCOME": m["final"]["OUTCOME"]}
                for a, m in sorted(metas.items()) if a in ax.SEQUENCE_IDS and m.get("status") == "done" and m.get("final")}
        result = {"reads": ev["reads"],
                  "sets": {k: {kk: vv for kk, vv in v.items() if kk not in ("diffs", "kept")} for k, v in rows.items()},
                  "trace_last": tr["trace"][-1], "learning_rates": tr["lrs"], "train_seconds": round(tr["seconds"]),
                  "lora_baselines": lora, "sheet_columns": ["no push"] + cols}
        meta.update(status="done", result=result, summary=ax.connector_summary(ev["reads"]),
                    seconds=round(time.time() - t0), finished_utc=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"))
        full = {**result, "baseline_scores": base["scores"],
                "cells": [{"scene": SUBJECTS[si], "seed": sd} for si, sd in base["cells"]],
                "per_cell": {k: {"diffs": v["diffs"], "content_kept": v["kept"]} for k, v in rows.items()}}
        (out_dir / "result.json").write_text(json.dumps(full, indent=1, default=float), encoding="utf-8")
        repo.commit({f"{bdir}/meta.json": sx.dumps(meta), f"{bdir}/README.md": readme(),
                     f"{bdir}/result.json": out_dir / "result.json", f"{bdir}/sheet.jpg": out_dir / "sheet.jpg"},
                    f"{arm.id}: evaluation + result")
        print(f"[{self.TAG}] {arm.id} done in {_sr._hms(meta['seconds'])}: {meta['summary']}", flush=True)
        try:
            from IPython.display import Image as _Img, display
            display(_Img(filename=str(out_dir / "sheet.jpg")))
        except Exception:  # noqa: BLE001
            pass
        return meta

    def _connector_cross(self, repo: "_HubRepo", metas: dict) -> None:
        """The reads across arms (the untrained trunk against Beatrix; each against the free vector), once the arms
        they need are done: written into every done arm's result, summary and README in one commit."""
        done = [(a, metas[a.id]) for a in ax.CONNECTOR_ARMS if (metas.get(a.id) or {}).get("status") == "done"]
        cross = ax.connector_cross_reads({a.id: m["result"]["reads"] for a, m in done})
        if not cross:
            return
        files: dict = {}
        for arm, m in done:
            if m["result"].get("cross") == cross:
                continue
            m["result"]["cross"] = cross
            m["summary"] = ax.connector_summary(m["result"]["reads"])
            for c in (cross.get("controls") or {}).values():
                if arm.id == c["random"]:
                    m["summary"] += f"; against {c['beatrix'].split('_')[0]}: {c['OUTCOME']}"
            bdir = f"experiments/{arm.id}"
            files[f"{bdir}/meta.json"] = sx.dumps(m)
            files[f"{bdir}/README.md"] = ax.render_connector_readme(arm, m, m.get("recipe", {}), m.get("phrases"))
        if files:
            self._safe(lambda: repo.commit(files, "connectors: the reads across arms"))
            self._safe(lambda: self._publish_index(repo, metas))
            for c in (cross.get("controls") or {}).values():
                print(f"[{self.TAG}] across arms ({c['beatrix'].split('_')[0]} vs {c['random'].split('_')[0]}): Beatrix's "
                      f"held-out effect {c['beatrix_heldout_effect']:+.3f}, the untrained trunk's "
                      f"{c['random_heldout_effect']:+.3f} -> {c['OUTCOME']}", flush=True)


def connector_whitening(F, train_rows: list, k: int) -> dict:
    """The top-k principal components of F's training rows, whitened: {'mu' [D], 'V' [k, D], 'scale' [k]} such that
    (F - mu) @ V.T / scale has unit variance per component over the training rows (float64 SVD; k capped at the
    training rows' rank). Fit on the training rows only: held-out rows are projected, never fitted."""
    import torch
    X = F[train_rows].double()
    mu = X.mean(0)
    _, S, Vh = torch.linalg.svd(X - mu, full_matrices=False)
    k = min(int(k), int((S > S[0] * 1e-6).sum()))
    return {"mu": mu.float(), "V": Vh[:k].float().contiguous(), "scale": (S[:k] / (len(train_rows) - 1) ** 0.5).float()}


def connector_axis(F, pool: dict) -> dict:
    """The slider projection from the training rows of each class (pool: {class: rows}), in the whitening's format
    {'mu', 'V' [2, D], 'scale' [2]}: (F - mu) @ V.T / scale = [a, n] with a = the position on the axis from the down
    centre to the up centre (the centres at -1 / +1) and n = the position on the axis from their midpoint toward the
    neutral centre (that centre at 1). Fit on the training rows only."""
    import torch
    X = F.double()
    c = {k: X[rows].mean(0) for k, rows in pool.items()}
    mid = (c["up"] + c["down"]) / 2
    ax, nax = c["up"] - c["down"], c["neutral"] - mid
    V = torch.stack([ax / (ax @ ax / 2), nax / (nax @ nax)])
    return {"mu": mid.float(), "V": V.float().contiguous(), "scale": torch.ones(2)}


def slider_map(Z, sides: str = "one"):
    """The slider's input map on its rows [a, n]: 'one' leaves them; 'relu' = [max(a, 0), max(-a, 0), n] (each side of
    the reading its own push direction); 'exp' = [e^a, e^-a, n] (both sides on for every phrase, the sign tilting the
    balance)."""
    import torch
    if sides == "one":
        return Z
    a, n = Z[:, :1], Z[:, 1:]
    if sides == "relu":
        return torch.cat([a.clamp(min=0), (-a).clamp(min=0), n], 1)
    if sides == "exp":
        return torch.cat([a.exp(), (-a).exp(), n], 1)
    raise ValueError(f"unknown slider map {sides!r}")


def _wd_tagger(repo_id: str, tags: list[str]):
    """A WD v3 tagger (SmilingWolf; timm) as (fn(images) -> probabilities [n images, n tags], {tag: column}), with the
    reference preprocessing (github.com/neggles/wdv3-timm): white square padding, the model's own transform (448,
    bicubic, mean / std .5), RGB -> BGR, sigmoid; fp32. Every one of `tags` (spaces or underscores) must be in the
    tagger's list."""
    import csv

    import numpy as np
    import torch
    from huggingface_hub import hf_hub_download
    from PIL import Image
    try:
        import timm
    except ImportError:
        print("[setup] installing timm (the tagger judge)", flush=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "timm"], check=True)
        import timm
    from timm.data import create_transform, resolve_data_config
    with open(hf_hub_download(repo_id, "selected_tags.csv"), encoding="utf-8") as f:
        col = {row["name"]: i for i, row in enumerate(csv.DictReader(f))}
    missing = [t for t in tags if t.replace(" ", "_") not in col]
    if missing:
        raise RuntimeError(f"{repo_id} has no tag(s) {missing}")
    index = {t: col[t.replace(" ", "_")] for t in tags}
    model = timm.create_model("hf-hub:" + repo_id).eval()
    model.load_state_dict(timm.models.load_state_dict_from_hf(repo_id))
    model = model.to("cuda")
    transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))

    def square(im):
        im = im.convert("RGB")
        w, h = im.size
        canvas = Image.new("RGB", (max(w, h), max(w, h)), (255, 255, 255))
        canvas.paste(im, ((max(w, h) - w) // 2, (max(w, h) - h) // 2))
        return canvas

    @torch.no_grad()
    def probs(imgs):
        out = []
        for i in range(0, len(imgs), 16):
            x = torch.stack([transform(square(im)) for im in imgs[i:i + 16]])[:, [2, 1, 0]].to("cuda")
            out.append(torch.sigmoid(model(x)).float().cpu().numpy())
        return np.concatenate(out)

    return probs, index
