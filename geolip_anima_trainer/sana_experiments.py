#!/usr/bin/env python3
"""
sana_experiments.py — the experiment specs, verdict rules and Hugging Face repo layout for the Sana line.

Every experiment lives in its own folder of one model repo (default AbstractPhil/geolip-beatrix-sana):

    README.md                          the line's overview + an index generated from every folder's meta.json
    experiments/<id>/README.md         what was asked, the recipe, the rule fixed before the run, the result
    experiments/<id>/meta.json         machine-readable status + result (the index and re-runs read it)
    experiments/<id>/config/           the training tomls
    experiments/<id>/data/             the training-set item list + a sample sheet (images are seed-reproducible)
    experiments/<id>/lora/epochN/      every saved epoch (diffusers-format LoRA), uploaded as it is saved
    experiments/<id>/samples/          the trainer's preview images
    experiments/<id>/eval/             per-epoch reads + contact sheets
    experiments/<id>/logs/             the training log

Pure Python (no torch); the runner (sana_runner.SanaRunner.run_sequence) does the GPU work.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import asdict, dataclass

DEFAULT_REPO = "AbstractPhil/geolip-beatrix-sana"
BASE_MODEL = "Efficient-Large-Model/Sana_600M_512px_diffusers"

FLAVOR_TEMPLATES = {
    "up": ["a cheerful, upbeat photo of {s}", "{s}, joyful and uplifting mood"],
    "down": ["a gloomy, downbeat photo of {s}", "{s}, somber and melancholy mood"],
    "neutral": ["a photo of {s}", "{s}"],
}
FLAVOR_WORDS = {"up": "upbeat", "down": "downbeat", "neutral": "neutral"}


@dataclass
class ArmSpec:
    """One LoRA experiment of the sequence: the registered first-run recipe except the named change.
    direction: +1 = the mood should rise, -1 = fall, 0 = a control (no direction expected)."""
    id: str
    title: str
    flavor: str                      # the training images' flavor: up | down | neutral
    seed_base: int = 1000            # image seeds seed_base .. seed_base + seeds_per_subject - 1
    lr: float = 1e-4
    direction: int = 1
    changed: str = "none (the reference recipe)"
    question: str = ""
    date: str = "2026-10-03"

    @property
    def data_key(self) -> str:
        return f"{self.flavor}_{self.seed_base}"


SEQUENCE: list[ArmSpec] = [
    ArmSpec("e004_lora_mood_up", "Mood LoRA: upbeat images, neutral captions", "up",
            question="Can a LoRA trained on upbeat images captioned with neutral prompts make neutral prompts come out "
                     "upbeat, on scenes it never saw?"),
    ArmSpec("e005_lora_neutral_control", "Control LoRA: the model's own neutral images, neutral captions", "neutral",
            direction=0, changed="the training images are the stock model's neutral renders",
            question="Does fine-tuning on the model's own images move the mood by itself? (A control that can fail.)"),
    ArmSpec("e006_lora_mood_down", "Mood LoRA, mirrored: downbeat images, neutral captions", "down",
            direction=-1, changed="the training images are downbeat renders",
            question="Does the same recipe push the mood the other way when the images are downbeat?"),
    ArmSpec("e007_lora_mood_up_reseed", "Mood LoRA, replicate: a fresh draw of upbeat images", "up", seed_base=2000,
            changed="a new draw of training images (seeds 2000-2007)",
            question="Does the effect survive a different draw of training images?"),
    ArmSpec("e008_lora_mood_up_lr_5e-5", "Mood LoRA at half the learning rate", "up", lr=5e-5,
            changed="learning rate 5e-5 (half)", question="How does the effect depend on the learning rate?"),
    ArmSpec("e009_lora_mood_up_lr_2e-4", "Mood LoRA at double the learning rate", "up", lr=2e-4,
            changed="learning rate 2e-4 (double)", question="How does the effect depend on the learning rate?"),
]
SEQUENCE_IDS = [a.id for a in SEQUENCE]


def items_for(subjects: list[str], train_idx: list[int], flavor: str, *, seeds_per_subject: int = 8,
              seed_base: int = 1000, caption_template: str = "a photo of {s}") -> list[dict]:
    """The training set: each training subject x seed -> a prompt of `flavor` (its two templates alternating),
    captioned with the neutral caption."""
    tps = FLAVOR_TEMPLATES[flavor]
    out = []
    for si in train_idx:
        s = subjects[si]
        for k in range(seeds_per_subject):
            out.append({"name": f"{si:02d}_{k:02d}", "seed": seed_base + k, "prompt": tps[k % 2].format(s=s),
                        "caption": caption_template.format(s=s)})
    return out


def mean_se(v: list[float]) -> tuple[float, float]:
    n = len(v)
    m = statistics.fmean(v) if n else 0.0
    return m, (statistics.stdev(v) / n ** 0.5) if n > 1 else 0.0


def arm_outcome(diffs: list[float], direction: int = 1) -> dict:
    """The rule fixed before the runs, on paired (LoRA minus no LoRA) mood-score differences, read in the arm's
    direction: FLAVOR LORA = the mean moves that way, at least 75% of the pairs move that way and the mean is beyond
    3 SE; NO EFFECT = |mean| <= 2 SE or under 60% of the pairs that way; else MIXED. direction 0 (a control) reports
    the numbers with verdict CONTROL (read against the reference arm by control_read)."""
    m, se = mean_se(diffs)
    n = len(diffs)
    if direction == 0:
        pos = sum(d > 0 for d in diffs) / n if n else 0.0
        return {"mean": m, "se": se, "frac_pos": pos, "n": n, "OUTCOME": "CONTROL"}
    sm = m * direction
    frac = sum(d * direction > 0 for d in diffs) / n if n else 0.0
    if sm > 0 and frac >= 0.75 and sm > 3 * se:
        verdict = "FLAVOR LORA"
    elif abs(m) <= 2 * se or frac < 0.60:
        verdict = "NO EFFECT"
    else:
        verdict = "MIXED"
    return {"mean": m, "se": se, "frac_pos" if direction > 0 else "frac_neg": frac, "n": n, "OUTCOME": verdict}


def control_read(control_mean: float, reference_mean: float) -> str:
    """CONTROL QUIET = the control's mean effect is at most one third of the reference arm's; else CONTROL MOVES."""
    return "CONTROL QUIET" if abs(control_mean) <= abs(reference_mean) / 3 else "CONTROL MOVES"


def net_of_control(ref_diffs: list[float], ctl_diffs: list[float]) -> dict:
    """Per cell, the reference arm's effect minus the control's (both at scale 1, the same held-out cells): mean > 0,
    at least 75% positive and beyond 3 SE = THE MOOD COMES FROM THE IMAGES; else NOT SHOWN."""
    d = [a - b for a, b in zip(ref_diffs, ctl_diffs)]
    m, se = mean_se(d)
    frac = sum(x > 0 for x in d) / len(d) if d else 0.0
    ok = m > 0 and frac >= 0.75 and m > 3 * se
    return {"mean": m, "se": se, "frac_pos": frac, "n": len(d),
            "OUTCOME": "THE MOOD COMES FROM THE IMAGES" if ok else "NOT SHOWN"}


def replicate_read(a: dict, b: dict) -> str:
    """REPLICATES = both arms FLAVOR LORA and their means within 2 x the combined SE."""
    both = a.get("OUTCOME") == "FLAVOR LORA" and b.get("OUTCOME") == "FLAVOR LORA"
    close = abs(a["mean"] - b["mean"]) <= 2 * (a["se"] ** 2 + b["se"] ** 2) ** 0.5
    return "REPLICATES" if both and close else ("BOTH WORK, SIZES DIFFER" if both else "DOES NOT REPLICATE")


def sequence_reads(metas: dict) -> dict:
    """The cross-arm reads over whatever arms are done (keyed by arm id; each meta carries 'final' and 'final_diffs')."""
    out: dict = {}
    ref = metas.get("e004_lora_mood_up")
    ctl = metas.get("e005_lora_neutral_control")
    if ref and ctl:
        out["control"] = control_read(ctl["final"]["mean"], ref["final"]["mean"])
        out["net_of_control"] = net_of_control(ref["final_diffs"], ctl["final_diffs"])
    down = metas.get("e006_lora_mood_down")
    if ref and down:
        out["mirror"] = {"up_mean": ref["final"]["mean"], "down_mean": down["final"]["mean"],
                         "down_outcome": down["final"]["OUTCOME"]}
    rep = metas.get("e007_lora_mood_up_reseed")
    if ref and rep:
        out["replicate"] = replicate_read(ref["final"], rep["final"])
    lr = {}
    for aid in ("e008_lora_mood_up_lr_5e-5", "e004_lora_mood_up", "e009_lora_mood_up_lr_2e-4"):
        m = metas.get(aid)
        if m:
            lr[str(m["spec"]["lr"])] = {"final_mean": m["final"]["mean"], "final_se": m["final"]["se"],
                                        "first_epoch_beyond_3se": m.get("first_epoch_beyond_3se")}
    if len(lr) > 1:
        out["lr_dose"] = lr
    return out


# ---- READMEs (public prose) ---------------------------------------------------------------------------
REFERENCES = [
    ("Xie, Chen, Chen, Cai, Tang et al., \"SANA: Efficient High-Resolution Image Synthesis with Linear Diffusion "
     "Transformers\" (2024)", "https://arxiv.org/abs/2410.10629"),
    ("Gandikota, Materzynska, Zhou, Torralba, Bau, \"Concept Sliders: LoRA Adaptors for Precise Control in Diffusion "
     "Models\" (2023)", "https://arxiv.org/abs/2311.12092"),
    ("Turner, Thiergart, Leech, Udell, Vazquez, Mini, MacDiarmid, \"Steering Language Models With Activation "
     "Engineering\" (2023)", "https://arxiv.org/abs/2308.10248"),
    ("Yu, Luo, Wang, Zhao, \"Uncovering the Text Embedding in Text-to-Image Diffusion Models\" (2024)",
     "https://arxiv.org/abs/2404.01154"),
    ("Brack, Friedrich, Hintersdorf, Struppek, Schramowski, Kersting, \"SEGA: Instructing Text-to-Image Models using "
     "Semantic Guidance\" (NeurIPS 2023)", "https://arxiv.org/abs/2301.12247"),
    ("Kwon, Jeong, Uh, \"Diffusion Models already have a Semantic Latent Space\" (ICLR 2023)",
     "https://arxiv.org/abs/2210.10960"),
    ("Hertz, Mokady, Tenenbaum, Aberman, Pritch, Cohen-Or, \"Prompt-to-Prompt Image Editing with Cross Attention "
     "Control\" (2022)", "https://arxiv.org/abs/2208.01626"),
    ("Yang, Feng, Huang, \"EmoGen: Emotional Image Content Generation with Text-to-Image Diffusion Models\" (2024)",
     "https://arxiv.org/abs/2401.04608"),
    ("Radford, Kim, Hallacy et al., \"Learning Transferable Visual Models From Natural Language Supervision\" (2021)",
     "https://arxiv.org/abs/2103.00020"),
]

JUDGE_TEXT = (
    "**The mood judge.** Each image is scored on its pixels only, with CLIP ViT-L/14: 100 x (the mean cosine "
    "similarity to three upbeat phrases, \"a cheerful, upbeat image\", \"a happy, joyful scene\", \"a bright, "
    "uplifting photo\", minus the same for three downbeat phrases, \"a gloomy, downbeat image\", \"a sad, melancholy "
    "scene\", \"a dark, depressing photo\"). On the stock model, writing upbeat words into a prompt raises this score "
    "by 1.81 over the neutral prompt (experiment e001). *Content kept* is the CLIP image cosine between an image and the "
    "no-LoRA image of the same prompt and seed.")


def _fmt(x, nd=3, sign=True):
    return f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"


def render_reads(reads: dict | None) -> str:
    """The cross-arm reads in plain words (empty until two related arms are done)."""
    if not reads:
        return ""
    out = ["## Reads across the LoRA sequence (rules fixed before the runs)"]
    if "control" in reads:
        out.append(f"- The control LoRA (e005, the model's own neutral images) against the upbeat LoRA (e004): "
                   f"**{reads['control']}** (quiet = at most a third of e004's effect).")
    if "net_of_control" in reads:
        n = reads["net_of_control"]
        out.append(f"- e004 minus e005, cell by cell: {_fmt(n['mean'])} +- {n['se']:.3f}, {n['frac_pos']:.0%} of "
                   f"{n['n']} cells positive: **{n['OUTCOME']}**.")
    if "mirror" in reads:
        m = reads["mirror"]
        out.append(f"- The mirror (e006, downbeat images): effect {_fmt(m['down_mean'])} against e004's "
                   f"{_fmt(m['up_mean'])}: **{m['down_outcome']}** in the downward direction.")
    if "replicate" in reads:
        out.append(f"- A fresh draw of upbeat images (e007) against e004: **{reads['replicate']}**.")
    if "lr_dose" in reads:
        out += ["", "| learning rate | final effect (mean +- SE) | first epoch beyond 3 SE |", "|---|---|---|"]
        for lr, v in sorted(reads["lr_dose"].items(), key=lambda kv: float(kv[0])):
            out.append(f"| {float(lr):g} | {_fmt(v['final_mean'])} +- {v['final_se']:.3f} | "
                       f"{v['first_epoch_beyond_3se'] or 'none'} |")
    return "\n".join(out) + "\n"


def render_repo_readme(metas: list[dict], reads: dict | None = None) -> str:
    rows = []
    for m in sorted(metas, key=lambda m: m["id"]):
        status = m.get("status", "")
        res = m.get("summary") or {"running": "running", "failed": "failed (see its log)"}.get(status, status)
        rows.append(f"| [`{m['id']}`](experiments/{m['id']}/) | {m.get('date', '')} | {m.get('title', '')} | {res} |")
    refs = "\n".join(f"- {t}. {u}" for t, u in REFERENCES)
    reads_md = render_reads(reads)
    return f"""---
license: apache-2.0
base_model: {BASE_MODEL}
tags:
- sana
- lora
- diffusers
- text-to-image
- experiments
---
# geolip-beatrix-sana

Experiments on the way to conditioning [Sana](https://github.com/NVlabs/Sana), a fast text-to-image diffusion model, on
Beatrix, a byte-level language model from the geolip line, through a trained connector. Sana 600M at 512 px is the test
bed: small enough to train and measure in minutes. The repo also keeps the experiments that led here.

Each experiment has its own folder under `experiments/` with a README (the question, the recipe, the rule fixed before
the run, the result), `meta.json`, and its configuration, weights, evaluation and logs where it has them.

## Experiments
| folder | date | what | result |
|---|---|---|---|
{chr(10).join(rows)}

{reads_md}
## How the mood experiments are measured
{JUDGE_TEXT}

The LoRA experiments train on 24 everyday scenes and are scored on 8 scenes they never saw (4 seeds each, 32 paired
cells): each cell compares the same prompt and seed with and without the LoRA.

## Tools
Training: [diffusion-pipe](https://github.com/AbstractEyes/diffusion-pipe) (the AbstractEyes fork, model type `sana`),
driven by [anima-trainer](https://github.com/AbstractEyes/anima-trainer) (`notebooks/sana_colab_train.ipynb`). The LoRAs
are in diffusers format: `pipe.load_lora_weights("experiments/<id>/lora/epochN", weight_name="adapter_model.safetensors")`.

## References
{refs}

## Licences
Sana's weights are Apache-2.0; its Gemma-2-2B-IT text encoder is under Google's
[Gemma Terms of Use](https://ai.google.dev/gemma/terms). The LoRAs and results here are Apache-2.0.
"""


def render_arm_readme(spec: ArmSpec, recipe: dict, meta: dict | None = None) -> str:
    rec = "\n".join(f"| {k} | {v} |" for k, v in recipe.items())
    dname = {1: "upward", -1: "downward", 0: "either way"}[spec.direction]
    if spec.direction == 0:
        rule = ("A control: the numbers are reported, and read against e004: the control is QUIET when its mean effect "
                "is at most one third of e004's, and MOVES otherwise. The net read (e004 minus this control, per cell) "
                "says the mood comes from the images when its mean is above 0, at least 75% of the cells are positive "
                "and the mean is beyond 3 standard errors.")
    else:
        rule = (f"Paired over the 32 held-out cells, the final LoRA at scale 1 minus no LoRA, read {dname}: "
                f"**FLAVOR LORA** = the mean moves {dname}, at least 75% of the cells move that way, and the mean is "
                "beyond 3 standard errors; **NO EFFECT** = the mean is within 2 standard errors of zero or under 60% "
                "of the cells move that way; **MIXED** otherwise.")
    out = [f"# {spec.id}: {spec.title}", "",
           f"Date: {spec.date}. Model: Sana 600M 512px ([{BASE_MODEL}](https://huggingface.co/{BASE_MODEL})). "
           "Trainer: diffusion-pipe (AbstractEyes fork, model type `sana`) through anima-trainer.", "",
           "## Question", spec.question, "",
           "## Training data",
           f"192 images rendered by the stock model from {FLAVOR_WORDS[spec.flavor]} prompts (24 scenes x 8 seeds, "
           f"seeds {spec.seed_base}-{spec.seed_base + 7}, the two {FLAVOR_WORDS[spec.flavor]} templates alternating), "
           "each captioned with the neutral prompt \"a photo of <scene>\". Eight more scenes are held out for the "
           "evaluation. The item list is in `data/items.jsonl`.", "",
           f"Changed from the reference recipe (e004): {spec.changed}.", "",
           "## Recipe", "| setting | value |", "|---|---|", rec, "",
           "## The rule (fixed before the run)", rule, "", JUDGE_TEXT, ""]
    if meta and meta.get("status") == "done":
        out += ["## Result", meta.get("summary", ""), "",
                "| epoch | scale | effect (mean +- SE) | cells moving the expected way | content kept |",
                "|---|---|---|---|---|"]
        for r in meta.get("epochs", []):
            frac = r.get("frac_pos", r.get("frac_neg"))
            out.append(f"| {r['epoch']} | {r['scale']} | {_fmt(r['mean'])} +- {r['se']:.3f} | "
                       f"{'' if frac is None else f'{frac:.0%}'} | {r['content_kept']:.3f} |")
        px_rows = ([("no LoRA", meta["baseline"])] if meta.get("baseline") else []) + \
                  [(f"epoch {r['epoch']}, scale {r['scale']}", r) for r in meta.get("epochs", [])]
        if px_rows and all("pixels" in v for _, v in px_rows):
            out += ["", "| images | mood score | luma | saturation | warmth | contrast |", "|---|---|---|---|---|---|"]
            for name, v in px_rows:
                p = v["pixels"]
                out.append(f"| {name} | {_fmt(v['mood_score'])} | {p['luma']:.3f} | {p['sat']:.3f} | "
                           f"{p['warmth']:.3f} | {p['contrast']:.3f} |")
        out += ["", "![no LoRA, scale 0.5, scale 1](eval/sheet_final.jpg)", "",
                "Rows: the held-out scenes at seed 101. Columns: no LoRA, the final LoRA at 0.5, at 1.", "",
                "![the effect across training](eval/sheet_epochs.jpg)", "",
                "Columns: no LoRA, then each saved epoch at scale 1.", ""]
    elif meta and meta.get("status") == "failed":
        out += ["## Result", f"The run failed: `{meta.get('error', '')}`. See `logs/`.", ""]
    else:
        out += ["## Result", "Running.", ""]
    out += ["## Files",
            "- `lora/epochN/adapter_model.safetensors`: every saved epoch (diffusers format).",
            "- `samples/`: the trainer's own preview images at each save.",
            "- `eval/`: `final.json` (every cell), `epochs.json` (the effect at each epoch), the contact sheets.",
            "- `config/`: the training configuration. `logs/train.log`: the trainer's log.", ""]
    return "\n".join(out)


def arm_meta(spec: ArmSpec, status: str, **kw) -> dict:
    return {"id": spec.id, "title": spec.title, "date": spec.date, "kind": "lora", "status": status,
            "spec": asdict(spec), **kw}


def dumps(obj) -> bytes:
    return json.dumps(obj, indent=1, default=float).encode("utf-8")
