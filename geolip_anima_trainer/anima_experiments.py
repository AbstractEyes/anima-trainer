#!/usr/bin/env python3
"""
anima_experiments.py — the Anima bed of the flavor experiments: the experiment registry, the prompt wording and the
public README text for AbstractPhil/geolip-beatrix-anima. The verdict rules, the cross-arm reads and the folder layout
are shared with the Sana bed (sana_experiments.py); only what differs lives here.

Anima (circlestone-labs/Anima) is a 2B illustration model on NVIDIA Cosmos-Predict2 whose text side is Qwen3 0.6B read
through a small LLM adapter. Its model card sets the recipe choices below: LoRAs train on Anima-Base, the LLM adapter
stays frozen, a rank-32 LoRA starts at learning rate 2e-5, sampling at 30-50 steps and guidance 4-5, the quality
prefix "masterpiece, best quality, score_7, safe, " and the negative prompt it recommends.

e001 measures the stock bed (the words, a mood direction added to the conditioning at two sites, the conditioning
norms); e002-e011 are the LoRA arms, the Sana design at Anima's learning rates; e012 screens attribute sliders on the
stock model; e013-e019 condition the image on Beatrix's own states through a learned push (and its controls); e020
splits the adapter's two readings of a caption (Qwen3's states, the T5 ids) on the stock model.

Pure Python (no torch); anima_runner.AnimaRunner does the GPU work.
"""

from __future__ import annotations

from dataclasses import dataclass

from .sana_experiments import ArmSpec, Bed

DEFAULT_REPO = "AbstractPhil/geolip-beatrix-anima"
BASE_MODEL = "circlestone-labs/Anima"
BASE_FILE = "split_files/diffusion_models/anima-base-v1.0.safetensors"

# The model card's recommended positive prefix and negative prompt (the base model's default style is plain without them).
PREFIX = "masterpiece, best quality, score_7, safe, "
NEGATIVE = ("worst quality, low quality, score_1, score_2, score_3, artist name, blurry, jpeg artifacts, "
            "chromatic aberration")

# "illustration", not "photo": the card says the model is for illustrations and does not do realism.
TEMPLATES = {
    "up": [PREFIX + "a cheerful, upbeat illustration of {s}.", PREFIX + "an illustration of {s}, joyful and uplifting mood."],
    "down": [PREFIX + "a gloomy, downbeat illustration of {s}.", PREFIX + "an illustration of {s}, somber and melancholy mood."],
    "neutral": [PREFIX + "an illustration of {s}.", PREFIX + "{s}."],
}
NEUTRAL_CAPTION = PREFIX + "an illustration of {s}."

LR = 2e-5                           # the model card's start for a rank-32 LoRA
DATE = "2026-10-03"


def _arm(id_: str, title: str, flavor: str, *, lr: float = LR, **kw) -> ArmSpec:
    return ArmSpec(id_, title, flavor, lr=lr, date=DATE, **kw)


SEQUENCE: list[ArmSpec] = [
    _arm("e002_lora_mood_up", "Mood LoRA: upbeat images, neutral captions", "up",
         question="Can a LoRA trained on upbeat images captioned with neutral prompts make neutral prompts come out "
                  "upbeat, on scenes it never saw?"),
    _arm("e003_lora_neutral_control", "Control LoRA: the model's own neutral images, neutral captions", "neutral",
         direction=0, changed="the training images are the stock model's neutral renders",
         question="Does fine-tuning on the model's own images move the mood by itself? (A control that can fail.)"),
    _arm("e004_lora_mood_down", "Mood LoRA, mirrored: downbeat images, neutral captions", "down",
         direction=-1, changed="the training images are downbeat renders",
         question="Does the same recipe push the mood the other way when the images are downbeat?"),
    _arm("e005_lora_mood_up_reseed", "Mood LoRA, replicate: a fresh draw of upbeat images", "up", seed_base=2000,
         changed="a new draw of training images (seeds 2000-2007)",
         question="Does the effect survive a different draw of training images?"),
    _arm("e006_lora_mood_up_lr_1e-5", "Mood LoRA at half the learning rate", "up", lr=1e-5,
         changed="learning rate 1e-5 (half)", question="How does the effect depend on the learning rate?"),
    _arm("e007_lora_mood_up_lr_4e-5", "Mood LoRA at double the learning rate", "up", lr=4e-5,
         changed="learning rate 4e-5 (double)", question="How does the effect depend on the learning rate?"),
    # the second draw: the same design on e005's independent draw of images (seeds 2000-2007); e005 is its reference
    _arm("e008_lora_neutral_control_draw2", "Control LoRA, second draw: the model's own neutral images", "neutral",
         seed_base=2000, direction=0,
         changed="the training images are the stock model's neutral renders, from the second draw (seeds 2000-2007); "
                 "read against e005",
         question="On a second, independent draw of images: does fine-tuning on the model's own images move the mood "
                  "by itself?"),
    _arm("e009_lora_mood_down_draw2", "Mood LoRA, mirrored, second draw: downbeat images", "down",
         seed_base=2000, direction=-1,
         changed="the training images are downbeat renders, from the second draw (seeds 2000-2007)",
         question="On a second draw of images: does the recipe push the mood down when the images are downbeat?"),
    _arm("e010_lora_mood_up_lr_1e-5_draw2", "Mood LoRA at half the learning rate, second draw", "up",
         seed_base=2000, lr=1e-5, changed="learning rate 1e-5 (half), on e005's images",
         question="On a second draw of images: how does the effect depend on the learning rate?"),
    _arm("e011_lora_mood_up_lr_4e-5_draw2", "Mood LoRA at double the learning rate, second draw", "up",
         seed_base=2000, lr=4e-5, changed="learning rate 4e-5 (double), on e005's images",
         question="On a second draw of images: how does the effect depend on the learning rate?"),
]
SEQUENCE_IDS = [a.id for a in SEQUENCE]
DRAW_ROLES = {"up": ("up", LR), "neutral": ("neutral", LR), "down": ("down", LR),
              "lr_half": ("up", LR / 2), "lr_double": ("up", LR * 2)}

# ---- e001: the stock bed ----------------------------------------------------------------------------
FLAVOR_TEST_ID = "e001_anima_flavor_test"
FLAVOR_TEST_TITLE = "The stock model: mood words, a mood direction added to the conditioning, and the conditioning norms"
FLAVOR_TEST_SEEDS = (11, 22)
DIAL_ALPHAS = (-2.0, -1.0, 1.0, 2.0)          # alpha 0 = the neutral prompt's own images (the words set)
DIAL_SITES = {
    "source": "the text states the adapter reads (Qwen3's last hidden state), before the adapter",
    "context": "the adapter's output, which the image model cross-attends to",
}


def flavor_outcome(diffs: list[float], direction: int, *, label: str) -> dict:
    """The words / dial rule fixed before the run (the LoRA rule, renamed): `label` = the mean moves `direction`, at
    least 75% of the cells that way and beyond 3 SE; NO EFFECT = within 2 SE of zero or under 60% that way; MIXED
    otherwise."""
    from .sana_experiments import arm_outcome
    r = arm_outcome(diffs, direction)
    r["OUTCOME"] = {"FLAVOR LORA": label}.get(r["OUTCOME"], r["OUTCOME"])
    return r


def cell_slopes(scores_by_alpha: dict) -> list[float]:
    """Per cell, the least-squares slope of the mood score on alpha ({alpha: [score per cell]}, alpha 0 included)."""
    alphas = sorted(scores_by_alpha)
    n = len(scores_by_alpha[alphas[0]])
    am = sum(alphas) / len(alphas)
    den = sum((a - am) ** 2 for a in alphas)
    out = []
    for i in range(n):
        ys = [scores_by_alpha[a][i] for a in alphas]
        ym = sum(ys) / len(ys)
        out.append(sum((a - am) * (y - ym) for a, y in zip(alphas, ys)) / den)
    return out


# ---- READMEs (public prose) ---------------------------------------------------------------------------
REFERENCES = (
    ("CircleStone Labs and Comfy Org, \"Anima\" (model card, 2026)", "https://huggingface.co/circlestone-labs/Anima"),
    ("NVIDIA et al., \"Cosmos World Foundation Model Platform for Physical AI\" (2025)",
     "https://arxiv.org/abs/2501.03575"),
    ("Yang, Li, Yang, Zhang, Hui, Zheng et al., \"Qwen3 Technical Report\" (2025)", "https://arxiv.org/abs/2505.09388"),
    ("Hu, Shen, Wallis, Allen-Zhu, Li, Wang et al., \"LoRA: Low-Rank Adaptation of Large Language Models\" (2021)",
     "https://arxiv.org/abs/2106.09685"),
    ("Liu, Gong, Liu, \"Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow\" (2022)",
     "https://arxiv.org/abs/2209.03003"),
    ("Esser, Kulal, Blattmann, Entezari, Muller, Saini et al., \"Scaling Rectified Flow Transformers for "
     "High-Resolution Image Synthesis\" (2024)", "https://arxiv.org/abs/2403.03206"),
    ("Gandikota, Materzynska, Zhou, Torralba, Bau, \"Concept Sliders: LoRA Adaptors for Precise Control in Diffusion "
     "Models\" (2023)", "https://arxiv.org/abs/2311.12092"),
    ("Turner, Thiergart, Udell, Leech, Mini, MacDiarmid, \"Activation Addition: Steering Language Models Without "
     "Optimization\" (2023)", "https://arxiv.org/abs/2308.10248"),
    ("Radford, Kim, Hallacy et al., \"Learning Transferable Visual Models From Natural Language Supervision\" (2021)",
     "https://arxiv.org/abs/2103.00020"),
)

JUDGE_TEXT = (
    "**The mood judge.** Each image is scored on its pixels only, with CLIP ViT-L/14: 100 x (the mean cosine "
    "similarity to three upbeat phrases, \"a cheerful, upbeat image\", \"a happy, joyful scene\", \"a bright, "
    "uplifting photo\", minus the same for three downbeat phrases, \"a gloomy, downbeat image\", \"a sad, melancholy "
    "scene\", \"a dark, depressing photo\"); the same judge as the Sana experiments, so the two beds read on one scale. "
    "How far mood words in the prompt move this score on the stock model is experiment e001. *Content kept* is the CLIP "
    "image cosine between an image and the no-LoRA image of the same prompt and seed.")

ANIMA = Bed(
    key="anima", model_name="Anima-Base v1.0 (2B)", base_model=BASE_MODEL, repo=DEFAULT_REPO, model_type="anima",
    templates=TEMPLATES, neutral_caption=NEUTRAL_CAPTION, sequence=tuple(SEQUENCE), roles=DRAW_ROLES,
    reference_note="e002; on the second draw of images, e005",
    lora_files=("every saved epoch (ComfyUI format, keys `diffusion_model.<module>.lora_A/B.weight`, alpha = rank; "
                "drop into `ComfyUI/models/loras/`)."),
    title="geolip-beatrix-anima",
    intro=("Experiments on steering [Anima](https://huggingface.co/circlestone-labs/Anima), a 2B illustration model "
           "built on NVIDIA Cosmos-Predict2,\nwith a second conditioning source: Beatrix, a byte-level language model "
           "from the geolip line. Anima reads its prompt through a\nsmall language model (Qwen3 0.6B) and a light "
           "adapter, so an added signal is not drowned by a very large text encoder; that makes\nit a bed for testing "
           "how far a second source can steer the image without a full diffusion training run. The first\nexperiments "
           "measure the bed itself: how its conditioning responds to mood words and to a mood direction added to it,\n"
           "and what LoRAs trained on the model's own mood images do."),
    tools=("Training: [diffusion-pipe](https://github.com/AbstractEyes/diffusion-pipe) (the AbstractEyes fork, model type "
           "`anima`; plain Adam,\nno weight decay, fp32 master weights over the bf16 LoRA; the LLM adapter frozen), "
           "driven by\n[anima-trainer](https://github.com/AbstractEyes/anima-trainer) "
           "(`notebooks/anima_colab_experiments.ipynb`). The images are rendered in the notebook\nprocess with the "
           "fork's own Anima model code (Euler flow sampler, shift 3, 30 steps, guidance 4.5, the model card's quality\n"
           "prefix and negative prompt). The LoRAs are in ComfyUI format."),
    licence=("Anima's weights are under the [CircleStone Labs Non-Commercial License](https://huggingface.co/"
             "circlestone-labs/Anima/blob/main/LICENSE.md)\n(Anima is a derivative of NVIDIA Cosmos-Predict2-2B, under "
             "the NVIDIA Open Model License); the LoRAs here are derivatives under\nthe same non-commercial terms. "
             "Generated images are not restricted by that licence."),
    yaml_license=("other\nlicense_name: circlestone-labs-non-commercial-license\n"
                  "license_link: https://huggingface.co/circlestone-labs/Anima/blob/main/LICENSE.md"),
    tags=("anima", "lora", "comfyui", "text-to-image", "experiments"),
    judge_text=JUDGE_TEXT, references=REFERENCES)
BED = ANIMA


def render_flavor_test_readme(meta: dict, recipe: dict) -> str:
    """e001's README: the question, the design and the rule fixed before the run, and the result when done."""
    rec = "\n".join(f"| {k} | {v} |" for k, v in recipe.items())
    sites = "\n".join(f"- **{k}**: {v}." for k, v in DIAL_SITES.items())
    out = [f"# {FLAVOR_TEST_ID}: {FLAVOR_TEST_TITLE}", "",
           f"Date: {DATE}. Model: {ANIMA.model_name} ([{BASE_MODEL}](https://huggingface.co/{BASE_MODEL})), no LoRA.", "",
           "## Questions",
           "1. Do mood words in the prompt move the mood of the image, and how far? (The scale every later effect is "
           "read against.)",
           "2. Does a mood direction added to the conditioning act as a dial, before the adapter and after it?",
           "3. How large are the conditioning vectors at those two places, and how large is the mood direction "
           "against them?", "",
           "## Design",
           "- **Words**: 32 scenes x {neutral, upbeat, downbeat} (the first template of each) x seeds "
           f"{', '.join(map(str, FLAVOR_TEST_SEEDS))}: 192 images.",
           "- **Norms**: for the 32 neutral prompts, the L2 norm of every prompt token's vector at both sites (padding "
           "excluded).",
           "- **Dial**: at each site, the mood direction d = half the mean, over the 32 scenes, of (the upbeat prompt's "
           "mean token vector minus the downbeat prompt's), added to every token of the neutral prompt at alpha "
           f"{', '.join(f'{a:+g}' for a in DIAL_ALPHAS)} (alpha 0 = the words set's neutral images), same seeds: "
           "512 images. The negative prompt is left unchanged.", sites, "",
           "## Recipe", "| setting | value |", "|---|---|", rec, "",
           "## The rule (fixed before the run)",
           "Per scene and seed (64 cells): **words** = the upbeat (downbeat) image's mood score minus the neutral "
           "image's; **UPBEAT WORDS MOVE IT** (**DOWNBEAT WORDS MOVE IT**) when the mean moves that way, at least 75% "
           "of the cells move that way and the mean is beyond 3 standard errors; **NO EFFECT** when the mean is within "
           "2 standard errors of zero or under 60% of the cells move that way; **MIXED** otherwise. **Dial** = per "
           "cell, the least-squares slope of the mood score on alpha (-2 to +2); **A DIAL** under the same rule read "
           "upward. Content kept is reported beside every read.", "", ANIMA.judge_text, ""]
    if meta.get("status") == "done":
        r = meta["result"]
        out += ["## Result", meta.get("summary", ""), "",
                "| read | effect (mean +- SE) | cells moving the expected way | verdict |", "|---|---|---|---|"]
        for key, label in (("up_words", "upbeat words minus neutral"), ("down_words", "downbeat words minus neutral")):
            w = r[key]
            frac = w.get("frac_pos", w.get("frac_neg"))
            out.append(f"| {label} | {w['mean']:+.3f} +- {w['se']:.3f} | {frac:.0%} | **{w['OUTCOME']}** |")
        for site in DIAL_SITES:
            d = r["dial"][site]
            out.append(f"| dial slope at the {site} site, per unit alpha | {d['mean']:+.3f} +- {d['se']:.3f} | "
                       f"{d['frac_pos']:.0%} | **{d['OUTCOME']}** |")
        out += ["", "| site | token norm: mean | median | max | mood direction norm | direction / mean token norm |",
                "|---|---|---|---|---|---|"]
        for site in DIAL_SITES:
            c = r["census"][site]
            out.append(f"| {site} | {c['mean']:.2f} | {c['median']:.2f} | {c['max']:.2f} | {c['d_norm']:.3f} | "
                       f"{c['d_ratio']:.4f} |")
        out += ["", "| site | alpha | mood score | content kept |", "|---|---|---|---|"]
        for site in DIAL_SITES:
            for a, v in sorted(r["dial"][site]["by_alpha"].items(), key=lambda kv: float(kv[0])):
                out.append(f"| {site} | {float(a):+g} | {v['mood_score']:+.3f} | {v['content_kept']:.3f} |")
        out += ["", "![the words: neutral, upbeat, downbeat](sheet_words.jpg)", "",
                "Rows: eight scenes at the first seed. Columns: neutral, upbeat words, downbeat words.", "",
                "![the dial at the source site](sheet_dial_source.jpg)", "",
                "![the dial at the context site](sheet_dial_context.jpg)", "",
                "Dial sheets: columns alpha -2, -1, 0, +1, +2.", ""]
    elif meta.get("status") == "failed":
        out += ["## Result", f"The run failed: `{meta.get('error', '')}`.", ""]
    else:
        out += ["## Result", "Running.", ""]
    out += ["## Files", "- `result.json`: every cell's scores, the reads and the norms.",
            "- `sheet_*.jpg`: contact sheets.", ""]
    return "\n".join(out)


# ---- e012: the attribute screen (sliders on the stock model) ------------------------------------------------
ATTR_TEST_ID = "e012_anima_attribute_screen"
ATTR_TEST_TITLE = "The stock model: attribute words, attribute sliders after the adapter, and their cross-talk"
ATTR_SEEDS = (11, 22)
CHARACTERS = (                                  # adult-coded outfits and settings (no school settings)
    ("office suit", "modern office"),
    ("apron over a sweater", "cozy cafe"),
    ("trench coat", "city street at dusk"),
    ("sundress", "flower garden"),
    ("leather jacket", "rooftop at night"),
    ("lab coat", "laboratory"),
    ("kimono", "shrine in autumn"),
    ("knit cardigan", "library"),
)
CHARACTER_TEMPLATE = PREFIX + "1girl, solo, {tag}upper body, {outfit}, {setting}."
# key: (+ tag, - tag or None = the neutral prompt, slider strengths). Age stays adult: pushed upward only.
ATTRIBUTES = {
    "hair_length": ("very long hair", "short hair", (-2.0, 2.0)),
    "hair_colour": ("blonde hair", "black hair", (-2.0, 2.0)),
    "eye_colour": ("red eyes", "blue eyes", (-2.0, 2.0)),
    "proportions": ("chibi", None, (-2.0, 2.0)),
    "age": ("old woman", None, (2.0,)),
    "style": ("watercolor (medium)", "flat color", (-2.0, 2.0)),
}
ATTR_SITE = "context"                           # e001: a uniform push works after the adapter, not before it
TAGGER = "SmilingWolf/wd-eva02-large-tagger-v3"


def attr_cells() -> list[tuple[int, int]]:
    """(character index, seed) for every cell."""
    return [(ci, sd) for ci in range(len(CHARACTERS)) for sd in ATTR_SEEDS]


def attr_prompt(tag: "str | None", ci: int) -> str:
    outfit, setting = CHARACTERS[ci]
    return CHARACTER_TEMPLATE.format(tag=f"{tag}, " if tag else "", outfit=outfit, setting=setting)


def attr_sets() -> dict:
    """Every image set of e012, in render order: {key: (tag or None, slider (attribute, alpha) or None)}."""
    sets: dict = {"neutral": (None, None)}
    for a, (plus, minus, _) in ATTRIBUTES.items():
        sets[f"{a}+"] = (plus, None)
        if minus:
            sets[f"{a}-"] = (minus, None)
    for a, (_, _, alphas) in ATTRIBUTES.items():
        for al in alphas:
            sets[f"{a}@{al:+g}"] = (None, (a, al))
    return sets


def attribute_reads(judge: dict) -> dict:
    """The registered reads from judge scores {set key: {attribute: [score per cell]}} (keys as attr_sets())."""
    import numpy as np
    base = judge["neutral"]
    span, slopes, out = {}, {}, {}
    for b, (_, minus, _) in ATTRIBUTES.items():
        lo = judge[f"{b}-"][b] if minus else base[b]
        span[b] = float(np.mean(np.subtract(judge[f"{b}+"][b], lo)))
    for a, (_, minus, alphas) in ATTRIBUTES.items():
        slopes[a] = {b: cell_slopes({0.0: base[b], **{al: judge[f"{a}@{al:+g}"][b] for al in alphas}})
                     for b in ATTRIBUTES}
    for a, (_, minus, alphas) in ATTRIBUTES.items():
        up = flavor_outcome(list(np.subtract(judge[f"{a}+"][a], base[a])), 1, label="THE WORD MOVES IT")
        down = (flavor_outcome(list(np.subtract(judge[f"{a}-"][a], base[a])), -1, label="THE WORD MOVES IT")
                if minus else None)
        dial = flavor_outcome(slopes[a][a], 1, label="A DIAL")
        cross = {b: (float(np.mean(slopes[a][b])) / span[b] if span[b] else 0.0) for b in ATTRIBUTES}
        reach = cross[a]
        worst = max((abs(v) for b, v in cross.items() if b != a), default=0.0)
        clean = reach > 0 and worst < reach / 3
        if up["OUTCOME"] != "THE WORD MOVES IT":
            verdict = "NO HANDLE"
        elif dial["OUTCOME"] != "A DIAL":
            verdict = "WORDS ONLY"
        else:
            verdict = "SLIDER" if clean else "SLIDER WITH CROSS-TALK"
        out[a] = {"plus": ATTRIBUTES[a][0], "minus": minus, "alphas": list(alphas), "words_up": up, "words_down": down,
                  "dial": dial, "span": span[a], "reach": reach, "cross_talk": cross, "worst_cross_talk": worst,
                  "clean": clean, "OUTCOME": verdict}
    return out


ATTR_REFERENCES = (
    ("Gandikota, Materzynska, Zhou, Torralba, Bau, \"Concept Sliders: LoRA Adaptors for Precise Control in Diffusion "
     "Models\" (2023)", "https://arxiv.org/abs/2311.12092"),
    ("Baumann, Krause, Neumayr, Stracke, Sevi, Hu et al., \"Continuous, Subject-Specific Attribute Control in T2I Models "
     "by Identifying Semantic Directions\" (2024)", "https://arxiv.org/abs/2403.17064"),
    ("Brack, Friedrich, Hintersdorf, Struppek, Schramowski, Kersting, \"SEGA: Instructing Text-to-Image Models using "
     "Semantic Guidance\" (NeurIPS 2023)", "https://arxiv.org/abs/2301.12247"),
    ("Gandikota, Wu, Zhang, Bau, Shechtman, Kolkin, \"SliderSpace: Decomposing the Visual Capabilities of Diffusion "
     "Models\" (2025)", "https://arxiv.org/abs/2502.01639"),
    ("SmilingWolf, \"WD EVA02-Large Tagger v3\" (model card, 2024)", f"https://huggingface.co/{TAGGER}"),
)


def render_attribute_screen_readme(meta: dict, recipe: dict) -> str:
    """e012's README: the question, the design and the rule fixed before the run, and the result when done."""
    rec = "\n".join(f"| {k} | {v} |" for k, v in recipe.items())
    attrs = "\n".join(f"| {a} | {p} | {m or '(none: the plain prompt)'} | {', '.join(f'{x:+g}' for x in al)} |"
                      for a, (p, m, al) in ATTRIBUTES.items())
    chars = "; ".join(f"{o} / {s}" for o, s in CHARACTERS)
    refs = "\n".join(f"- {t}: {u}" for t, u in ATTR_REFERENCES)
    out = [f"# {ATTR_TEST_ID}: {ATTR_TEST_TITLE}", "",
           f"Date: {DATE}. Model: {ANIMA.model_name} ([{BASE_MODEL}](https://huggingface.co/{BASE_MODEL})), no LoRA.", "",
           "## Questions",
           "1. Which character attributes does a tag in the prompt control reliably on this model?",
           "2. Does each attribute's direction, added to the conditioning after the text adapter, act as a slider?",
           "3. Does each slider move only its own attribute, or does it drag the others along?", "",
           "## Design",
           f"- **Characters**: \"1girl, solo, upper body\" with eight outfits and settings ({chars}) x seeds "
           f"{', '.join(map(str, ATTR_SEEDS))}: 16 cells per set. All characters are adults; the age slider only "
           "pushes older.",
           "- **Attributes** (a tag pair inside the model's own vocabulary):", "",
           "| attribute | + tag | - tag | slider strengths |", "|---|---|---|---|", attrs, "",
           "- **Words**: each tag added to the prompt, against the plain prompt.",
           "- **Sliders**: the attribute's direction (half the mean, over the eight characters, of the + prompt's mean "
           "token vector minus the - prompt's or the plain prompt's) at the adapter's output, which the image model reads; "
           "added to every token of the plain prompt at the strengths above (the plain prompt's own images are strength "
           "0). The negative prompt is left unchanged. Experiment e001 found this site live and the one before the "
           "adapter dead for such a push.",
           "- 22 sets x 16 = 352 images.", "",
           "## Recipe", "| setting | value |", "|---|---|", rec, "",
           "## The judge",
           f"Each image is scored on its pixels by the [WD EVA02-Large tagger v3](https://huggingface.co/{TAGGER}), an "
           "image tagger trained on Danbooru tags (the vocabulary the attribute tags come from), with its reference "
           "preprocessing. An attribute's score is the log-odds of its + tag minus the log-odds of its - tag (the + tag "
           "alone where there is no - tag). *Content kept* is the CLIP ViT-L/14 image cosine to the plain prompt's image "
           "of the same character and seed.", "",
           "## The rule (fixed before the run)",
           "Per character and seed (16 cells), read in the expected direction: **THE WORD MOVES IT** when the mean moves "
           "that way, at least 75% of the cells move that way and the mean is beyond 3 standard errors; **NO EFFECT** "
           "within 2 standard errors of zero or under 60% that way; **MIXED** otherwise. The slider read is the per-cell "
           "least-squares slope of the attribute's score on the strength (**A DIAL** under the same rule). "
           "**Cross-talk**: slider a's slope on attribute b's score, in units of b's word span (the + word minus the - "
           "word, or the plain prompt, on b's score); **reach** = slider a on its own attribute in the same units; "
           "**clean** = every other attribute's cross-talk under a third of the reach. Verdict per attribute: **SLIDER** "
           "(the word moves it, the slider is a dial, clean), **SLIDER WITH CROSS-TALK** (not clean), **WORDS ONLY** "
           "(no dial), **NO HANDLE** (the word does not move the judge).", ""]
    if meta.get("status") == "done":
        r = meta["result"]["attributes"]
        out += ["## Result", meta.get("summary", ""), "",
                "| attribute | + word | - word | slider slope | reach | worst cross-talk | content kept at +2 | verdict |",
                "|---|---|---|---|---|---|---|---|"]
        for a, v in r.items():
            down = (f"{v['words_down']['mean']:+.2f} ({v['words_down']['OUTCOME']})" if v.get("words_down") else "")
            kept = v.get("content_kept", {}).get("+2")
            out.append(f"| {a} | {v['words_up']['mean']:+.2f} ({v['words_up']['OUTCOME']}) | {down} | "
                       f"{v['dial']['mean']:+.3f} +- {v['dial']['se']:.3f} ({v['dial']['OUTCOME']}) | {v['reach']:.2f} | "
                       f"{v['worst_cross_talk']:.2f} | {'' if kept is None else f'{kept:.3f}'} | **{v['OUTCOME']}** |")
        names = list(r)
        out += ["", "Cross-talk (row = slider, column = attribute read; units of that attribute's word span):", "",
                "| slider | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
        for a in names:
            out.append(f"| {a} | " + " | ".join(f"{r[a]['cross_talk'][b]:+.2f}" for b in names) + " |")
        ov = meta["result"].get("overlap", {}).get(ATTR_SITE)
        if ov:
            out += ["", "Direction overlap after the adapter (cosine):", "",
                    "| | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
            for a in names:
                out.append(f"| {a} | " + " | ".join(f"{ov[a][b]:+.2f}" for b in names) + " |")
        out += ["", "Sheets (`sheet_<attribute>.jpg`): eight characters at the first seed; columns: the - word (where "
                "there is one), the slider at -2, the plain prompt, the slider at +2, the + word.", ""]
        out += [f"![{a}](sheet_{a}.jpg)" for a in names] + [""]
    elif meta.get("status") == "failed":
        out += ["## Result", f"The run failed: `{meta.get('error', '')}`.", ""]
    else:
        out += ["## Result", "Running.", ""]
    out += ["## Files", "- `result.json`: every cell's scores, the reads, the cross-talk and the direction overlaps.",
            "- `sheet_*.jpg`: contact sheets.", "", "## References", refs, ""]
    return "\n".join(out)


# ---- e020: the route split (the stock model: which of the adapter's two inputs carries the words) -------------------
ROUTE_TEST_ID = "e020_anima_route_split"
ROUTE_TEST_TITLE = ("The stock model: which of the adapter's two readings of a caption carries the mood words, Qwen3's "
                    "states or the T5 word ids")
ROUTE_SEEDS = FLAVOR_TEST_SEEDS                 # e001's cells, so its words re-render as this run's reference
# every image set: (the template flavor whose Qwen3 states the adapter reads, the flavor whose T5 ids it reads)
ROUTE_SETS = {
    "neutral": ("neutral", "neutral"),
    "words_up": ("up", "up"), "words_down": ("down", "down"),
    "qwen_up": ("up", "neutral"), "qwen_down": ("down", "neutral"),
    "t5_up": ("neutral", "up"), "t5_down": ("neutral", "down"),
}
ROUTES = {"words": ("THE WORDS MOVE IT", "both readings (the plain mood prompt)"),
          "qwen": ("THE QWEN STATES MOVE IT", "Qwen3's states of the mood prompt, the neutral prompt's T5 ids"),
          "t5": ("THE T5 IDS MOVE IT", "the neutral prompt's Qwen3 states, the mood prompt's T5 ids")}
GATE_LABEL = "THE SOURCE HALF ADDS TO THE QUERY HALF"


@dataclass(frozen=True)
class RouteTest:
    """One route-split experiment: which mood templates it uses and whether the pair-minus-query gate is registered."""
    id: str
    title: str
    template: int                    # the mood templates' index in TEMPLATES (the neutral prompt is always the first)
    gate: bool                       # the pair minus the query half on the downbeat words is a registered read
    question: str
    limit: str


APPENDED_TEST_ID = "e021_anima_route_split_appended"
ROUTE_TESTS = {
    ROUTE_TEST_ID: RouteTest(
        ROUTE_TEST_ID, ROUTE_TEST_TITLE, 0, False,
        question="Anima's LLM adapter reads a caption twice: as Qwen3 0.6B's last hidden states (the source its "
                 "cross-attention reads) and as the caption's T5 token ids, through the adapter's own embedding table (the "
                 "queries its blocks start from). When mood words are added to the caption, which of the two readings "
                 "carries their effect on the image? (Experiment e001 found that a direction added to every Qwen3 state does "
                 "not move the mood, while the words do; this splits the words themselves.)",
        limit="the mood words shift the later tokens' positions in the reading that carries them while the other reading "
              "stays neutral. The adapter always pairs two tokenizations of unequal length, but the split is not "
              "position-exact."),
    APPENDED_TEST_ID: RouteTest(
        APPENDED_TEST_ID, "The stock model: the mood words appended after the scene, through the adapter's query half, "
                          "its source half, or both", 1, True,
        question="The adapter's cross-attention works as a lookup: the caption's T5 ids become the queries, Qwen3's states "
                 "the keys and values, and experiment e020 found that mood words in Qwen3's states alone do nothing while "
                 "the same words in both readings add to what the T5 ids carry. With the mood words appended after the "
                 "scene, so that both readings share the caption's positions up to them: what does the source half (the "
                 "words' Qwen3 states) add when the query half (the words' T5 ids) is there to look it up, above all for "
                 "the downbeat words?",
        limit="the appended words still differ in length between the two tokenizations; the caption before them is "
              "position-identical in both readings."),
}


def route_prompts(test: RouteTest, flavor: str, scene: str) -> str:
    return TEMPLATES[flavor][0 if flavor == "neutral" else test.template].format(s=scene)


def route_reads(scores: dict) -> dict:
    """The registered reads from mood scores {set key: [score per cell]} (keys as ROUTE_SETS): per set, the cells'
    difference to the neutral image under e001's rule; per split route CARRIES THE WORDS (both of its sets move it),
    ONE WAY (one) or CARRIES NOTHING; the pair minus the query half per cell (words minus T5-only; registered as the
    gate on the downbeat words where the test says so); descriptive: each route's share of the words' effect and the
    two routes' sum."""
    import numpy as np
    base = scores["neutral"]
    sets = {}
    for key in ROUTE_SETS:
        if key != "neutral":
            route, mood = key.rsplit("_", 1)
            sets[key] = flavor_outcome(list(np.subtract(scores[key], base)), 1 if mood == "up" else -1,
                                       label=ROUTES[route][0])
    out: dict = {"sets": sets, "routes": {}}
    for route in ("qwen", "t5"):
        moving = [sets[f"{route}_{m}"]["OUTCOME"] == ROUTES[route][0] for m in ("up", "down")]
        share = {m: (sets[f"{route}_{m}"]["mean"] / sets[f"words_{m}"]["mean"] if sets[f"words_{m}"]["mean"] else None)
                 for m in ("up", "down")}
        out["routes"][route] = {"OUTCOME": "CARRIES THE WORDS" if all(moving) else "ONE WAY" if any(moving)
                                else "CARRIES NOTHING", "share_of_words": share}
    out["sum_of_routes"] = {m: ((sets[f"qwen_{m}"]["mean"] + sets[f"t5_{m}"]["mean"]) / sets[f"words_{m}"]["mean"]
                                if sets[f"words_{m}"]["mean"] else None) for m in ("up", "down")}
    out["pair_minus_query"] = {m: flavor_outcome(list(np.subtract(scores[f"words_{m}"], scores[f"t5_{m}"])),
                                                 1 if m == "up" else -1, label=GATE_LABEL) for m in ("up", "down")}
    return out


def route_summary(reads: dict, test: "RouteTest | None" = None) -> str:
    s, r = reads["sets"], reads["routes"]
    out = "; ".join(f"{name} {s[f'{k}_up']['mean']:+.2f} / {s[f'{k}_down']['mean']:+.2f}"
                    + (f" ({r[k]['OUTCOME']})" if k in r else "")
                    for k, name in (("words", "the words"), ("qwen", "Qwen3's states only"), ("t5", "the T5 ids only")))
    if test is not None and test.gate and reads.get("pair_minus_query"):
        g = reads["pair_minus_query"]
        out += (f"; the source half beside the query half {g['up']['mean']:+.2f} / {g['down']['mean']:+.2f} "
                f"(downbeat: {g['down']['OUTCOME']})")
    return out


def render_route_split_readme(meta: dict, recipe: dict) -> str:
    """A route split's README (e020, e021): the question, the design and the rule fixed before the run, and the result."""
    t = ROUTE_TESTS[meta.get("id", ROUTE_TEST_ID)]
    rec = "\n".join(f"| {k} | {v} |" for k, v in recipe.items())
    tpl = {f: TEMPLATES[f][0 if f == "neutral" else t.template].replace(PREFIX, "") for f in ("neutral", "up", "down")}
    out = [f"# {t.id}: {t.title}", "",
           f"Date: 2026-10-04. Model: {ANIMA.model_name} ([{BASE_MODEL}](https://huggingface.co/{BASE_MODEL})), no LoRA.", "",
           "## Question", t.question, "",
           "## Design",
           f"- e001's 32 scenes x seeds {', '.join(map(str, ROUTE_SEEDS))} (64 cells); the prompts (after the model card's "
           f"quality prefix): neutral \"{tpl['neutral']}\", upbeat \"{tpl['up']}\", downbeat \"{tpl['down']}\"; the "
           "negative prompt unchanged. Seven sets, 448 images:", "",
           "| set | Qwen3 states from | T5 ids from |", "|---|---|---|",
           *[f"| {k} | the {q} prompt | the {f} prompt |" for k, (q, f) in ROUTE_SETS.items()], "",
           "- In the words sets both readings carry the mood words (the pair); in the t5 sets only the queries do (the "
           "query half); in the qwen sets only the source does (the source half).", "",
           "## Recipe", "| setting | value |", "|---|---|", rec, "",
           "## The rule (fixed before the run)",
           "Per cell, a set's mood score minus the neutral image's, read in the mood's direction (e001's rule): a set "
           "**moves it** when the mean moves that way, at least 75% of the 64 cells move that way and the mean is beyond "
           "3 standard errors; **NO EFFECT** within 2 standard errors of zero or under 60% that way; **MIXED** otherwise. "
           "Per reading (Qwen3's states, the T5 ids): **CARRIES THE WORDS** when both of its sets move it, **ONE WAY** "
           "when one does, **CARRIES NOTHING** when neither does. Reported beside them: each reading's share of the "
           "words' effect, the two readings' sum against the words, content kept (the CLIP image cosine to the neutral "
           "image of the same cell)."]
    if t.gate:
        out += ["", f"**The gate**: per cell, the words set minus the t5 set (the source half added to the query half) on "
                    f"the downbeat words, read downward under the same rule: **{GATE_LABEL}**, **NO EFFECT** or **MIXED**. "
                    "The same difference on the upbeat words is reported beside it."]
    out += ["", f"**Limit fixed in advance**: {t.limit}", "", ANIMA.judge_text, ""]
    if meta.get("status") == "done":
        r = meta["result"]
        out += ["## Result", meta.get("summary", ""), "",
                "| set | effect (mean +- SE) | cells moving the expected way | content kept | verdict |",
                "|---|---|---|---|---|"]
        for k, v in r["sets"].items():
            frac = v.get("frac_pos", v.get("frac_neg"))
            kept = r.get("content_kept", {}).get(k)
            out.append(f"| {k} | {v['mean']:+.3f} +- {v['se']:.3f} | {frac:.0%} | "
                       f"{'' if kept is None else f'{kept:.3f}'} | **{v['OUTCOME']}** |")
        out += ["", "| reading | verdict | share of the words' effect, up / down |", "|---|---|---|"]
        for k, v in r["routes"].items():
            sh = " / ".join("" if v["share_of_words"][m] is None else f"{v['share_of_words'][m]:.2f}" for m in ("up", "down"))
            out.append(f"| {ROUTES[k][1]} | **{v['OUTCOME']}** | {sh} |")
        sm = r["sum_of_routes"]
        out += ["", "The two readings' effects summed, against the words: up "
                + ("" if sm["up"] is None else f"{sm['up']:.2f}") + ", down " + ("" if sm["down"] is None else f"{sm['down']:.2f}")
                + " (1 = the words' effect is the sum of the two)."]
        if t.gate and r.get("pair_minus_query"):
            g = r["pair_minus_query"]
            out += ["", "| the source half added to the query half (words minus t5) | effect | cells that way | verdict |",
                    "|---|---|---|---|"]
            for m in ("down", "up"):
                frac = g[m].get("frac_pos", g[m].get("frac_neg"))
                out.append(f"| {m}beat words{' (the gate)' if m == 'down' else ''} | {g[m]['mean']:+.3f} +- "
                           f"{g[m]['se']:.3f} | {frac:.0%} | **{g[m]['OUTCOME']}** |")
        if r.get("e001_words"):
            out += ["", f"e001's words (its first templates) for the same cells: up {r['e001_words']['up']:+.3f}, down "
                        f"{r['e001_words']['down']:+.3f}."]
        out += ["", "![the seven sets](sheet_routes.jpg)", "",
                "Rows: eight scenes at the first seed. Columns: " + ", ".join(ROUTE_SETS) + ".", ""]
    elif meta.get("status") == "failed":
        out += ["## Result", f"The run failed: `{meta.get('error', '')}`.", ""]
    else:
        out += ["## Result", "Running.", ""]
    out += ["## Files", "- `result.json`: every cell's scores and the reads.", "- `sheet_routes.jpg`: the contact sheet.", ""]
    return "\n".join(out)


# ---- e022: the query-site dial (the stock model: a mood direction in the adapter's queries, alone and paired) ---------
QUERY_TEST_ID = "e022_anima_query_dial"
QUERY_TEST_TITLE = ("The stock model: a mood direction added to the adapter's queries, alone and paired with a source token "
                    "a query can look up")
QUERY_WORDS = {"up": ("cheerful", "upbeat"), "down": ("gloomy", "downbeat")}     # in e001's first mood templates
QUERY_FORMS = {
    "q": "the query dial alone",
    "pair_dir": "the query dial + one source token appended after the caption: the source direction scaled to the mean "
                "size of Qwen3's states (upbeat for alpha > 0, downbeat for alpha < 0)",
    "pair_state": "the query dial + one source token appended after the caption: the mean Qwen3 state at the real mood "
                  "words (\"cheerful\", \"upbeat\" for alpha > 0; \"gloomy\", \"downbeat\" for alpha < 0)",
    "pair_uniform": "the query dial + the source direction added to every Qwen3 token at the same alpha (the control)",
}
QUERY_GATE = {"pair_dir": "THE SOURCE TOKEN ADDS", "pair_state": "THE SOURCE TOKEN ADDS",
              "pair_uniform": "THE UNIFORM SOURCE ADDS"}


def query_sets() -> list[str]:
    """Every image set of e022 in render order: the neutral images, then each form at each alpha."""
    return ["neutral"] + [f"{f}@{a:+g}" for f in QUERY_FORMS for a in DIAL_ALPHAS]


def query_dial_reads(scores: dict) -> dict:
    """The registered reads from mood scores {set: [score per cell]} (keys as query_sets()): per form, the per-cell slope
    of the mood score on alpha (A DIAL rule); the gates: per cell, the mean over alpha -2 and -1 of a pair form minus the
    query dial alone, read downward; descriptive: the same over +1 and +2 read upward, and the state token minus the
    direction token on the downbeat side."""
    import numpy as np
    neg = [a for a in DIAL_ALPHAS if a < 0]
    pos = [a for a in DIAL_ALPHAS if a > 0]

    def side(form, alphas, minus="q"):
        return list(np.mean([np.subtract(scores[f"{form}@{a:+g}"], scores[f"{minus}@{a:+g}"]) for a in alphas], axis=0))

    out: dict = {"dials": {}, "gates": {}, "upbeat_side": {}}
    for f in QUERY_FORMS:
        out["dials"][f] = flavor_outcome(cell_slopes({0.0: scores["neutral"],
                                                      **{a: scores[f"{f}@{a:+g}"] for a in DIAL_ALPHAS}}), 1, label="A DIAL")
    for f, label in QUERY_GATE.items():
        out["gates"][f] = flavor_outcome(side(f, neg), -1, label=label)
        out["upbeat_side"][f] = flavor_outcome(side(f, pos), 1, label=label)
    out["state_minus_direction_downbeat"] = flavor_outcome(side("pair_state", neg, minus="pair_dir"), -1,
                                                           label="THE STATE TOKEN ADDS MORE")
    return out


def query_summary(reads: dict) -> str:
    d, g = reads["dials"], reads["gates"]
    return (f"query dial {d['q']['mean']:+.3f}/unit ({d['q']['OUTCOME']}); downbeat side, a source token beside it: direction "
            f"{g['pair_dir']['mean']:+.2f} ({g['pair_dir']['OUTCOME']}), state {g['pair_state']['mean']:+.2f} "
            f"({g['pair_state']['OUTCOME']}); uniform control {g['pair_uniform']['mean']:+.2f} ({g['pair_uniform']['OUTCOME']})")


def render_query_dial_readme(meta: dict, recipe: dict) -> str:
    """e022's README: the question, the design and the rule fixed before the run, and the result when done."""
    rec = "\n".join(f"| {k} | {v} |" for k, v in recipe.items())
    forms = "\n".join(f"| {k} | {v} |" for k, v in QUERY_FORMS.items())
    alphas = ", ".join(f"{a:+g}" for a in DIAL_ALPHAS)
    out = [f"# {QUERY_TEST_ID}: {QUERY_TEST_TITLE}", "",
           f"Date: 2026-10-04. Model: {ANIMA.model_name} ([{BASE_MODEL}](https://huggingface.co/{BASE_MODEL})), no LoRA.", "",
           "## Question",
           "The adapter's cross-attention works as a lookup: its queries start from the caption's T5 ids (its own word "
           "table), and Qwen3's states are the keys and values. Experiment e001 found a mood direction added to every "
           "Qwen3 state does nothing, and e020 that Qwen3's states carry the mood words only when the words' queries "
           "are there. (1) Does a mood direction added to the queries act as a dial? (2) On the downbeat side, does one "
           "source token that a query can look up add to it, and does it matter whether that token is a bare direction "
           "or shaped like Qwen3's real state for a mood word? A direction spread over every Qwen3 token is the control.",
           "",
           "## Design",
           f"- The neutral prompt of e001 on its 64 cells (32 scenes x seeds {', '.join(map(str, FLAVOR_TEST_SEEDS))}); "
           f"alpha {alphas} (alpha 0 = the neutral images); the negative prompt unchanged; every push on the "
           "conditional branch only (e001's dial).",
           "- The query direction: half the mean, over the 32 scenes, of the upbeat prompt's mean query vector minus the "
           "downbeat prompt's (e001's first templates), added to the query embeddings (the output of the adapter's word "
           "table and input projection, before its blocks) at every caption token, times alpha. The source direction: the "
           "same at Qwen3's states (e001's source direction).", "",
           "| form | what is added |", "|---|---|", forms, "",
           f"- {1 + len(QUERY_FORMS) * len(DIAL_ALPHAS)} sets x 64 = {64 * (1 + len(QUERY_FORMS) * len(DIAL_ALPHAS))} "
           "images.", "",
           "## Recipe", "| setting | value |", "|---|---|", rec, "",
           "## The rule (fixed before the run)",
           "Per cell, e001's rule. **Dial**: the least-squares slope of the mood score on alpha; **A DIAL** when the mean "
           "is upward, at least 75% of the 64 cells upward and beyond 3 standard errors; **NO EFFECT** within 2 standard "
           "errors of zero or under 60% upward; **MIXED** otherwise. **The gates** (the downbeat side): per cell, the mean "
           "over alpha -2 and -1 of a paired form's mood score minus the query dial's alone, read downward under the same "
           "rule: **THE SOURCE TOKEN ADDS** for the two token forms; for the uniform control **THE UNIFORM SOURCE ADDS** "
           "(expected: no effect, if placement is what a query needs). Reported beside them: the same differences over "
           "alpha +1 and +2 read upward, the state token minus the direction token on the downbeat side, content kept "
           "(the CLIP image cosine to the neutral image), and the sizes of the directions and tokens.", "",
           ANIMA.judge_text, ""]
    if meta.get("status") == "done":
        r = meta["result"]
        out += ["## Result", meta.get("summary", ""), "",
                "| form | slope per unit alpha | cells upward | verdict |", "|---|---|---|---|"]
        for f, v in r["dials"].items():
            out.append(f"| {f} | {v['mean']:+.3f} +- {v['se']:.3f} | {v['frac_pos']:.0%} | **{v['OUTCOME']}** |")
        out += ["", "| beside the query dial | downbeat side (the gate) | cells | verdict | upbeat side | cells | verdict |",
                "|---|---|---|---|---|---|---|"]
        for f in QUERY_GATE:
            g, u = r["gates"][f], r["upbeat_side"][f]
            out.append(f"| {f} | {g['mean']:+.3f} +- {g['se']:.3f} | {g['frac_neg']:.0%} | **{g['OUTCOME']}** | "
                       f"{u['mean']:+.3f} +- {u['se']:.3f} | {u['frac_pos']:.0%} | {u['OUTCOME']} |")
        s = r["state_minus_direction_downbeat"]
        out += ["", f"The state token minus the direction token, downbeat side: {s['mean']:+.3f} +- {s['se']:.3f} "
                    f"({s['frac_neg']:.0%} downward; {s['OUTCOME']})."]
        if r.get("sizes"):
            out += ["", "Sizes: " + ", ".join(f"{k} {v:.3f}" for k, v in r["sizes"].items()) + "."]
        out += ["", "| set | mood score | content kept |", "|---|---|---|"]
        for k in query_sets():
            out.append(f"| {k} | {r['mood_score'][k]:+.3f} | {r['content_kept'][k]:.3f} |")
        out += ["", "![the forms at alpha -2 and +2](sheet_query.jpg)", "",
                "Rows: eight scenes at the first seed. Columns: neutral, then each form at alpha -2 and +2.", ""]
    elif meta.get("status") == "failed":
        out += ["## Result", f"The run failed: `{meta.get('error', '')}`.", ""]
    else:
        out += ["## Result", "Running.", ""]
    out += ["## Files", "- `result.json`: every cell's scores and the reads.", "- `sheet_query.jpg`: the contact sheet.", ""]
    return "\n".join(out)


# ---- e026: the word split (the route split one word at a time: whole T5 tokens against shattered ones) ---------------
WORD_TEST_ID = "e026_anima_word_split"
WORD_TEST_TITLE = ("The stock model: single mood words through the adapter's query half alone or through both halves, "
                   "whole T5 tokens against shattered ones")
WORD_TEMPLATE = PREFIX + "an illustration of {s}, {w} mood."
WORD_GROUPS = {                          # (mood, how the T5 vocabulary holds the word) -> words: the 2 x 2 of the test
    ("up", "whole"): ("happy", "joyful"),
    ("up", "shattered"): ("jubilant", "gleeful"),
    ("down", "whole"): ("sad", "miserable"),
    ("down", "shattered"): ("gloomy", "melancholy"),
}
WORD_EXTRA = {"upbeat": "up", "downbeat": "down"}   # descriptive rows outside the 2 x 2 (two pieces each, ending in 'beat')
WORD_SEEDS = FLAVOR_TEST_SEEDS[:1]       # e001's 32 scenes at its first seed: 32 cells
WORD_SHARE_LINE = 0.5


def word_list() -> list[str]:
    return [w for ws in WORD_GROUPS.values() for w in ws] + list(WORD_EXTRA)


def word_mood(w: str) -> str:
    return WORD_EXTRA.get(w) or next(m for (m, _), ws in WORD_GROUPS.items() if w in ws)


def word_sets() -> dict:
    """Every image set of e026 in render order: {key: (the word whose prompt gives Qwen3's states, the word whose prompt
    gives the T5 ids)}; None = the neutral prompt."""
    out: dict = {"neutral": (None, None)}
    for w in word_list():
        out[f"words_{w}"] = (w, w)
        out[f"t5_{w}"] = (None, w)
    return out


def word_prompt(w: "str | None", scene: str) -> str:
    return NEUTRAL_CAPTION.format(s=scene) if w is None else WORD_TEMPLATE.format(s=scene, w=w)


def word_split_reads(scores: dict) -> dict:
    """The registered reads from mood scores {set key: [score per cell]} (keys as word_sets()): per word, in its mood's
    direction under e001's rule, the word itself (words_w - neutral; a word that does not move it is not readable), its
    query half (t5_w - neutral), its source half (words_w - t5_w) and its query share; per group of the 2 x 2 the query
    share pooled over its readable words; THE TEST on the two groups the readings disagree on (gloomy whole, cheerful
    shattered)."""
    import numpy as np
    base = scores["neutral"]
    words = {}
    for w in word_list():
        d = 1 if word_mood(w) == "up" else -1
        word = flavor_outcome(list(np.subtract(scores[f"words_{w}"], base)), d, label="THE WORD MOVES IT")
        query = flavor_outcome(list(np.subtract(scores[f"t5_{w}"], base)), d, label="THE QUERY HALF MOVES IT")
        source = flavor_outcome(list(np.subtract(scores[f"words_{w}"], scores[f"t5_{w}"])), d, label="THE SOURCE HALF ADDS")
        words[w] = {"mood": word_mood(w), "word": word, "query": query, "source": source,
                    "query_share": query["mean"] / word["mean"] if word["mean"] else None,
                    "readable": word["OUTCOME"] == "THE WORD MOVES IT"}
    groups = {}
    for (m, kind), ws in WORD_GROUPS.items():
        ok = [w for w in ws if words[w]["readable"]]
        den = sum(words[w]["word"]["mean"] for w in ok)
        groups[f"{m}_{kind}"] = {"words": list(ws), "readable": ok,
                                 "query_share": sum(words[w]["query"]["mean"] for w in ok) / den if ok and den else None}
    gw, cs = groups["down_whole"]["query_share"], groups["up_shattered"]["query_share"]
    test = ("NOT READABLE" if gw is None or cs is None else
            "BY TOKENIZATION" if gw >= WORD_SHARE_LINE > cs else
            "BY MOOD" if cs >= WORD_SHARE_LINE > gw else "NEITHER")
    return {"words": words, "groups": groups, "TEST": test}


def word_summary(reads: dict) -> str:
    def q(k):
        v = reads["groups"][k]["query_share"]
        return "n/a" if v is None else f"{v:.2f}"
    return (f"{reads['TEST']}; the query half's share: cheerful whole {q('up_whole')}, cheerful shattered "
            f"{q('up_shattered')}, gloomy whole {q('down_whole')}, gloomy shattered {q('down_shattered')}")


def render_word_split_readme(meta: dict, recipe: dict) -> str:
    """e026's README: the question, the design and the rule fixed before the run, and the result when done."""
    rec = "\n".join(f"| {k} | {v} |" for k, v in recipe.items())
    pieces = meta.get("pieces") or {}

    def cut(w, side="t5"):
        return " ".join(pieces.get(w, {}).get(side, [])) or "?"

    mood = {"up": "cheerful", "down": "gloomy"}
    rows = [f"| {w} | {mood[m]} | {kind} | {cut(w)} | {cut(w, 'qwen')} |" for (m, kind), ws in WORD_GROUPS.items() for w in ws]
    rows += [f"| {w} | {mood[m]} | (descriptive) | {cut(w)} | {cut(w, 'qwen')} |" for w, m in WORD_EXTRA.items()]
    n_sets = len(word_sets())
    out = [f"# {WORD_TEST_ID}: {WORD_TEST_TITLE}", "",
           f"Date: 2026-10-04. Model: {ANIMA.model_name} ([{BASE_MODEL}](https://huggingface.co/{BASE_MODEL})), no LoRA.", "",
           "## Question",
           "The adapter reads a caption twice: its queries start from the caption's T5 token ids (through its own word "
           "table), and they look up Qwen3's states. Experiment e021 found that, with the mood words after the scene, the "
           "upbeat words reach the image almost entirely through the queries, while the downbeat words are carried by "
           "Qwen3's states, looked up by the same words' queries. Every upbeat word there is a whole token in the T5 "
           "vocabulary (\"joyful\", \"uplifting\"), while every downbeat word is cut into pieces (\"somber\" = so + m + "
           "ber, \"melancholy\" = me + lan + cho + ly). Does the split follow the mood, or whether the T5 vocabulary holds "
           "the word whole?", "",
           "## Design",
           f"- One mood word after the scene, \"{WORD_TEMPLATE.replace(PREFIX, '').format(s='{scene}', w='{word}')}\", "
           f"against the neutral prompt \"{NEUTRAL_CAPTION.replace(PREFIX, '').format(s='{scene}')}\" (after the model "
           f"card's quality prefix; the negative prompt unchanged); e001's 32 scenes at seed {WORD_SEEDS[0]} (32 cells).",
           "- Eight words in a 2 x 2, plus two descriptive words outside the test; the pieces are Anima's own tokenizers' "
           "(recorded by the run):", "",
           "| word | mood | T5 vocabulary | T5 pieces | Qwen3 pieces |", "|---|---|---|---|---|", *rows, "",
           f"- Per word two sets: the word through both readings (words) and through the T5 ids only, Qwen3 reading the "
           f"neutral prompt (t5); with the neutral set, {n_sets} sets x 32 = {32 * n_sets} images.", "",
           "## Recipe", "| setting | value |", "|---|---|", rec, "",
           "## The rule (fixed before the run)",
           "Per cell, a set's mood score minus the neutral image's (minus the t5 set's for the source half), read in the "
           "word's mood direction under e001's rule: it **moves it** when the mean moves that way, at least 75% of the 32 "
           "cells move that way and the mean is beyond 3 standard errors; **NO EFFECT** within 2 standard errors of zero "
           "or under 60% that way; **MIXED** otherwise. Per word: the word (words - neutral; a word that does not move it "
           "is not readable), the query half (t5 - neutral), the source half (words - t5), and the query share (the query "
           "half's mean over the word's). Per group of the 2 x 2: the query share pooled over its readable words (their "
           "query halves' sum over their effects' sum). **The test**: **BY TOKENIZATION** when the gloomy whole-token "
           "group's share is at least 0.5 and the cheerful shattered group's is under 0.5; **BY MOOD** for the reverse; "
           "**NEITHER** otherwise; **NOT READABLE** when either of those two groups has no readable word. The other two "
           "groups are anchors: both readings predict the same for them (a high share for the cheerful whole words, a "
           "low one for the gloomy shattered words; e021's word pairs read 0.97 and 0.20).", "",
           "**Limit fixed in advance**: the number of T5 pieces stands in for whether the adapter's word table knows the "
           "word; how often each word appeared in the adapter's training captions is unknown, and the words differ in "
           "strength; 32 cells.", "", ANIMA.judge_text, ""]
    if meta.get("status") == "done":
        r = meta["result"]
        out += ["## Result", meta.get("summary", ""), "",
                "| word | T5 pieces | the word (mean +- SE) | verdict | the query half | the source half | query share |",
                "|---|---|---|---|---|---|---|"]
        for w, v in r["words"].items():
            qs = "" if v["query_share"] is None else f"{v['query_share']:.2f}"
            out.append(f"| {w} | {len(pieces.get(w, {}).get('t5', [])) or '?'} | {v['word']['mean']:+.3f} +- "
                       f"{v['word']['se']:.3f} | **{v['word']['OUTCOME']}** | {v['query']['mean']:+.3f} +- "
                       f"{v['query']['se']:.3f} ({v['query']['OUTCOME']}) | {v['source']['mean']:+.3f} +- "
                       f"{v['source']['se']:.3f} ({v['source']['OUTCOME']}) | {qs} |")
        out += ["", "| group | words read | pooled query share |", "|---|---|---|"]
        for k, g in r["groups"].items():
            qs = "not readable" if g["query_share"] is None else f"{g['query_share']:.2f}"
            m, kind = k.split("_")
            out.append(f"| {mood[m]}, {kind} | {', '.join(g['readable']) or 'none'} | {qs} |")
        out += ["", f"**The test: {r['TEST']}.**", "",
                "![the words](sheet_words.jpg)", "",
                "Rows: eight scenes. Columns: " + ", ".join(r.get("sheet_columns", [])) + ".", ""]
    elif meta.get("status") == "failed":
        out += ["## Result", f"The run failed: `{meta.get('error', '')}`.", ""]
    else:
        out += ["## Result", "Running.", ""]
    out += ["## Files", "- `result.json`: every cell's scores and the reads.", "- `sheet_words.jpg`: the contact sheet.", ""]
    return "\n".join(out)


# ---- e027: the slot pair (a word-sized push at one word's position: its query side, its source side, both, or the source
# side under a content-free question) -------------------------------------------------------------------------------------
SLOT_TEST_ID = "e027_anima_slot_pair"
SLOT_TEST_TITLE = ("The stock model: a word-sized mood push at one word's position, on the adapter's query side, its source "
                   "side, both, or the source side under a content-free question")
SLOT_WORD = "neutral"                    # one token in both tokenizers, at aligned positions
SLOT_TEMPLATE = PREFIX + "an illustration of {s}, {w} mood."
SLOT_REFERENCE = ("happy", "sad")        # the words behind the directions; the real words at the slot are the references
SLOT_ALPHAS = (-1.0, -0.5, 0.5, 1.0)     # in units of one word's size (each side's mean token size)
SLOT_SEEDS = WORD_SEEDS                  # e026's 32 cells
SLOT_FORMS = {
    "Q": "the query direction at the slot: the adapter's query embedding of \"happy\" minus that of \"sad\", unit length, "
         "times the queries' mean size per unit alpha",
    "P": "Q + the source direction at the slot: Qwen3's state at the slot for \"happy mood\" minus \"sad mood\" (the mean "
         "over the scenes), unit length, times Qwen3's states' mean size per unit alpha (the matched pair)",
    "S": "the source direction at the slot alone (an answer to the slot's own neutral question)",
    "C": "the source direction at the slot under a content-free question: the slot's query replaced by that of T5's bare "
         "space piece (the piece most words the T5 vocabulary does not hold open with), C at alpha 0 being that question "
         "alone",
}
SLOT_FREE_PIECE = "▁"               # T5's bare space piece: '▁ gle e ful', '▁ ju bil ant', '▁ g loom y' (e026)
SLOT_GATE = "THE MATCHED SOURCE ADDS"


def slot_prompt(w: str, scene: str) -> str:
    return SLOT_TEMPLATE.format(s=scene, w=w)


def slot_sets() -> list[str]:
    """Every image set of e027 in render order: the slot prompt, the scene prompt without the slot, the real words at the
    slot, each form at each alpha (the content-free question C also at alpha 0)."""
    return ["slot", "plain", *[f"word_{w}" for w in SLOT_REFERENCE],
            *[f"{f}@{a:+g}" for f in SLOT_FORMS for a in ((0.0,) if f == "C" else ()) + SLOT_ALPHAS]]


def slot_pair_reads(scores: dict) -> dict:
    """The registered reads from mood scores {set: [score per cell]} (keys as slot_sets()): per form, the per-cell slope of
    the mood score on alpha (A DIAL rule; alpha in words; alpha 0 = the slot prompt, for C its content-free question
    alone); THE GATE: per cell, the mean over the negative alphas of the pair minus the query alone, read downward;
    descriptive: the same over the positive alphas read upward, the real words against the slot prompt, the share of each
    word's effect the pair recovers at alpha +-1, the content-free question's own cost against the slot prompt and the
    scene prompt, the slot's own cost, and C minus S per side (the same answer under the two questions)."""
    import numpy as np
    neg = [a for a in SLOT_ALPHAS if a < 0]
    pos = [a for a in SLOT_ALPHAS if a > 0]

    def side(alphas, hi="P", lo="Q"):
        return list(np.mean([np.subtract(scores[f"{hi}@{a:+g}"], scores[f"{lo}@{a:+g}"]) for a in alphas], axis=0))

    def zero(f):
        return scores["C@+0"] if f == "C" else scores["slot"]

    def cost(a, b):                       # descriptive: the mean, its SE and the share of cells upward
        return {**flavor_outcome(list(np.subtract(scores[a], scores[b])), 0, label=""), "OUTCOME": "DESCRIPTIVE"}

    out: dict = {"dials": {f: flavor_outcome(cell_slopes({0.0: zero(f), **{a: scores[f"{f}@{a:+g}"] for a in SLOT_ALPHAS}}),
                                             1, label="A DIAL")
                           for f in SLOT_FORMS},
                 "gate": flavor_outcome(side(neg), -1, label=SLOT_GATE),
                 "upbeat_side": flavor_outcome(side(pos), 1, label=SLOT_GATE), "words": {}, "pair_share": {},
                 "free_question_cost": cost("C@+0", "slot"), "slot_cost": cost("slot", "plain"),
                 "free_question_vs_plain": cost("C@+0", "plain"),
                 "free_minus_whole": {"downbeat": flavor_outcome(side(neg, "C", "S"), -1, label="THE FREE QUESTION READS MORE"),
                                      "upbeat": flavor_outcome(side(pos, "C", "S"), 1, label="THE FREE QUESTION READS MORE")}}
    for w, d, key in ((SLOT_REFERENCE[0], 1, f"P@{max(SLOT_ALPHAS):+g}"), (SLOT_REFERENCE[1], -1, f"P@{min(SLOT_ALPHAS):+g}")):
        word = flavor_outcome(list(np.subtract(scores[f"word_{w}"], scores["slot"])), d, label="THE WORD MOVES IT")
        out["words"][w] = word
        pair = float(np.mean(np.subtract(scores[key], scores["slot"])))
        out["pair_share"][w] = pair / word["mean"] if word["mean"] else None
    return out


def slot_summary(reads: dict) -> str:
    d, g = reads["dials"], reads["gate"]
    return (f"per word of push: the query at the slot {d['Q']['mean']:+.3f} ({d['Q']['OUTCOME']}), the pair "
            f"{d['P']['mean']:+.3f} ({d['P']['OUTCOME']}), the answer alone {d['S']['mean']:+.3f} ({d['S']['OUTCOME']}), the "
            f"answer under a content-free question {d['C']['mean']:+.3f} ({d['C']['OUTCOME']}); downbeat side, the matched "
            f"source beside the query {g['mean']:+.2f} ({g['OUTCOME']})")


def render_slot_pair_readme(meta: dict, recipe: dict) -> str:
    """e027's README: the question, the design and the rule fixed before the run, and the result when done."""
    rec = "\n".join(f"| {k} | {v} |" for k, v in recipe.items())
    forms = "\n".join(f"| {k} | {v} |" for k, v in SLOT_FORMS.items())
    alphas = ", ".join(f"{a:+g}" for a in SLOT_ALPHAS)
    up, down = SLOT_REFERENCE
    n_sets = len(slot_sets())
    out = [f"# {SLOT_TEST_ID}: {SLOT_TEST_TITLE}", "",
           f"Date: 2026-10-04. Model: {ANIMA.model_name} ([{BASE_MODEL}](https://huggingface.co/{BASE_MODEL})), no LoRA.", "",
           "## Question",
           "The adapter's queries start from the caption's T5 ids and look up Qwen3's states. Experiment e021 found the "
           "downbeat words carried by Qwen3's states read through the same words' queries, and e022 that a mood direction "
           "spread over every query (a twentieth of a word's size) barely moves the image, while one source token appended "
           "beside it is not read. Here the push is word-sized and sits at one word's position, on both sides of the "
           "lookup: does a synthetic question at the slot move the image, and does its matched synthetic answer at the "
           "same position add to it on the downbeat side?", "",
           "Experiment e026 then found that a word the T5 vocabulary holds whole works through its query alone (Qwen3's "
           "state adds nothing there), while a word cut into pieces gets its meaning only from Qwen3's states, looked up "
           "by pieces that carry no meaning of their own. So the design adds a fourth form: the same answer at the slot "
           "under a content-free question (the bare space piece most such words open with: \"gleeful\" = ▁ gle e "
           "ful) instead of the word \"neutral\". "
           "Does an answer under a question that asks for nothing in particular move the image, where the same answer "
           "under a whole word may not?", "",
           "## Design",
           f"- The slot prompt \"{SLOT_TEMPLATE.replace(PREFIX, '').format(s='{scene}', w=SLOT_WORD)}\" (after the model "
           f"card's quality prefix; the negative prompt unchanged): \"{SLOT_WORD}\" is one token in both tokenizers, at "
           f"aligned positions; e001's 32 scenes at seed {SLOT_SEEDS[0]} (32 cells); alpha {alphas} in units of one "
           "word's size; every push on the conditional branch only.", "",
           "| form | what is added |", "|---|---|", forms, "",
           f"- References: the slot prompt (alpha 0), the scene prompt without the slot "
           f"(\"{NEUTRAL_CAPTION.replace(PREFIX, '').format(s='{scene}')}\") and the real words at the slot (\"{up} "
           f"mood\", \"{down} mood\"). {n_sets} sets x 32 = {32 * n_sets} images.", "",
           "## Recipe", "| setting | value |", "|---|---|", rec, "",
           "## The rule (fixed before the run)",
           "Per cell, e001's rule. **Dial**: the least-squares slope of the mood score on alpha (alpha 0 = the slot "
           "prompt; for C, the content-free question alone); **A DIAL** when the mean is upward, at least 75% of the 32 "
           "cells upward and beyond 3 standard errors; **NO EFFECT** within 2 standard errors of zero or under 60% "
           "upward; **MIXED** otherwise. **The gate** "
           "(the downbeat side): per cell, the mean over the negative alphas of the pair's mood score minus the query's "
           f"alone, read downward under the same rule: **{SLOT_GATE}**, **NO EFFECT** or **MIXED**. Reported beside it: "
           "the same over the positive alphas read upward, the real words against the slot prompt, the share of each "
           "word's effect the pair recovers at alpha +-1, content kept and the sizes; for the content-free question, its "
           "own cost (C at alpha 0 minus the slot prompt, and minus the scene prompt), the slot's own cost (the slot prompt "
           "minus the scene prompt), and C minus S per side (the same answer under the two questions). How C reads, fixed "
           "in advance: C a dial and S not = a whole word's own entry keeps its answer from being read, and a content-free "
           "question at the slot is the form; both dials = the answer is found by its position under either question; "
           "neither, with the pair a dial = question and answer must agree; none = one position is not enough.", "",
           "**Limit fixed in advance**: one slot and one word pair behind the directions; the source direction is the "
           "mean over the scenes; 32 cells.", "", ANIMA.judge_text, ""]
    if meta.get("status") == "done":
        r = meta["result"]
        out += ["## Result", meta.get("summary", ""), "",
                "| form | slope per word of push | cells upward | verdict |", "|---|---|---|---|"]
        for f, v in r["dials"].items():
            out.append(f"| {f} | {v['mean']:+.3f} +- {v['se']:.3f} | {v['frac_pos']:.0%} | **{v['OUTCOME']}** |")
        g, u = r["gate"], r["upbeat_side"]
        out += ["", "| the pair minus the query alone | effect | cells that way | verdict |", "|---|---|---|---|",
                f"| downbeat side (the gate) | {g['mean']:+.3f} +- {g['se']:.3f} | {g['frac_neg']:.0%} | **{g['OUTCOME']}** |",
                f"| upbeat side | {u['mean']:+.3f} +- {u['se']:.3f} | {u['frac_pos']:.0%} | {u['OUTCOME']} |", "",
                "| real word at the slot | effect (mean +- SE) | verdict | share the pair recovers at alpha +-1 |",
                "|---|---|---|---|"]
        for w, v in r["words"].items():
            sh = r["pair_share"].get(w)
            out.append(f"| {w} | {v['mean']:+.3f} +- {v['se']:.3f} | {v['OUTCOME']} | "
                       f"{'' if sh is None else f'{sh:.2f}'} |")
        if "free_question_cost" in r:
            fm = r["free_minus_whole"]
            out += ["", "| the content-free question | effect (mean +- SE) | cells upward |", "|---|---|---|"]
            for label, k in (("C at alpha 0 minus the slot prompt (the question swap)", "free_question_cost"),
                             ("the slot prompt minus the scene prompt (the slot itself)", "slot_cost"),
                             ("C at alpha 0 minus the scene prompt", "free_question_vs_plain")):
                v = r[k]
                out.append(f"| {label} | {v['mean']:+.3f} +- {v['se']:.3f} | {v['frac_pos']:.0%} |")
            out += ["", "| C minus S (the same answer, content-free vs whole-word question) | effect | cells that way | "
                        "verdict |", "|---|---|---|---|",
                    f"| downbeat side | {fm['downbeat']['mean']:+.3f} +- {fm['downbeat']['se']:.3f} | "
                    f"{fm['downbeat']['frac_neg']:.0%} | {fm['downbeat']['OUTCOME']} |",
                    f"| upbeat side | {fm['upbeat']['mean']:+.3f} +- {fm['upbeat']['se']:.3f} | "
                    f"{fm['upbeat']['frac_pos']:.0%} | {fm['upbeat']['OUTCOME']} |"]
        if r.get("sizes"):
            out += ["", "Sizes: " + ", ".join(f"{k} {v:.3f}" for k, v in r["sizes"].items()) + "."]
        out += ["", "| set | mood score | content kept |", "|---|---|---|"]
        for k in slot_sets():
            out.append(f"| {k} | {r['mood_score'][k]:+.3f} | {r['content_kept'][k]:.3f} |")
        out += ["", "![the references and the forms at alpha -1 and +1](sheet_slot.jpg)", "",
                "Rows: eight scenes. Columns: " + ", ".join(r.get("sheet_columns", [])) + ".", ""]
    elif meta.get("status") == "failed":
        out += ["## Result", f"The run failed: `{meta.get('error', '')}`.", ""]
    else:
        out += ["## Result", "Running.", ""]
    out += ["## Files", "- `result.json`: every cell's scores and the reads.", "- `sheet_slot.jpg`: the contact sheet.", ""]
    return "\n".join(out)


# ---- e013-e015: Beatrix as a second conditioning source (a learned push after the adapter) ------------------------
@dataclass(frozen=True)
class ConnectorArm:
    """One connector experiment: where its input comes from, its training seed and what it asks."""
    id: str
    title: str
    source: str                      # 'trained' | 'random': that trunk's phrase features; 'onehot': the class, no encoder
    seed: int                        # training seed (batch order, phrase draws, noise, timesteps)
    question: str
    changed: str = "none (the reference connector)"
    whiten_k: "int | None" = None    # the features projected on the top k whitened components of the training phrases
    axis: bool = False               # the features reduced to a slider value on the training phrases' mood axis (+ neutral)
    sides: str = "one"               # the slider's input map: 'one' [a, n]; 'relu' [max(a,0), max(-a,0), n]; 'exp' [e^a, e^-a, n]


CONNECTOR_ARMS = (
    ConnectorArm("e013_beatrix_mood_connector", "Beatrix's mood phrases steer the image through a learned push",
                 "trained", 13,
                 question="Trained on mood images with neutral captions, does a push computed from Beatrix's states for a "
                          "mood phrase steer the image that way, for the phrases it trained on and for mood phrases it "
                          "never saw?"),
    ConnectorArm("e014_beatrix_random_trunk_connector",
                 "Control: the same connector on an untrained Beatrix of the same shape", "random", 14,
                 changed="the features come from a randomly initialised trunk of the same shape (seed 0), standardized "
                         "the same way",
                 question="Does the held-out effect come from what Beatrix learned, or would any fixed random features of "
                          "the phrase text carry it? (A control that must fail on the held-out phrases.)"),
    ConnectorArm("e015_free_vector_connector", "Capacity reference: one free learned vector per mood class, no encoder",
                 "onehot", 15, changed="the input is the mood class itself (one-hot), not an encoder's features",
                 question="How far can a push of this form move the image when nothing has to be read from text? (The "
                          "reference the encoder arms are read against; trained classes only.)"),
    # e013 read NOT LEARNED: in the raw 4,096-feature basis the phrase-dependent part of the push barely moved in 720 steps.
    # The re-run whitens the features onto the training phrases' main directions and sets the map's learning rate so its
    # class contrast moves at the free vector's pace (all choices from the training phrases only).
    ConnectorArm("e016_beatrix_mood_connector_whitened",
                 "Beatrix's mood phrases steer the image through a learned push, whitened input", "trained", 16,
                 changed="the features are projected on the top 16 whitened principal components of the training phrases, "
                         "and the map's learning rate is set so its class contrast moves at the free vector's pace "
                         "(e013 learned too slowly to read)",
                 question="Trained on mood images with neutral captions, does a push computed from Beatrix's states for a "
                          "mood phrase steer the image that way, for the phrases it trained on and for mood phrases it "
                          "never saw?", whiten_k=16),
    ConnectorArm("e017_beatrix_random_trunk_connector_whitened",
                 "Control: the same whitened connector on an untrained Beatrix of the same shape", "random", 17,
                 changed="e016's connector (whitened input, the free vector's pace) on the features of a randomly "
                         "initialised trunk of the same shape (seed 0), whitened on its own training phrases",
                 question="Does e016's held-out effect come from what Beatrix learned, or would any fixed random features of "
                          "the phrase text carry it? (A control that must fail on the held-out phrases.)", whiten_k=16),
    # e016 learned every group the right way, but each phrase's own part of the push outweighed the mood contrast. The slider
    # arms reduce her features to two numbers: the phrase's position on the training phrases' mood axis and on their neutral
    # axis, so the image model receives her reading of the phrase's mood and nothing phrase-specific.
    ConnectorArm("e018_beatrix_mood_slider", "Beatrix's reading of a phrase's mood as a slider value", "trained", 18,
                 changed="the features are reduced to a slider value: the phrase's position on the axis from the gloomy to "
                         "the cheerful training phrases (their centres at -1 and +1), plus its position on the axis toward "
                         "the neutral training phrases; the map's rate by the free vector's pace",
                 question="Does Beatrix's reading of a phrase's mood, reduced to one slider value, steer the image that way, "
                          "for the phrases it trained on and for mood phrases it never saw?", axis=True),
    ConnectorArm("e019_beatrix_random_trunk_slider", "Control: the same slider on an untrained Beatrix of the same shape",
                 "random", 19,
                 changed="e018's slider (the axes fit on its own training phrases) on the features of a randomly initialised "
                         "trunk of the same shape (seed 0)",
                 question="Does e018's held-out effect come from what Beatrix learned, or would any fixed random features of "
                          "the phrase text place unseen phrases on the right side? (A control that must fail on the held-out "
                          "phrases.)", axis=True),
    # e018 moved the image for cheerful phrases (unseen ones too) and not for gloomy ones: a linear map of one slider value
    # pushes gloomy phrases along the mirror of the cheerful push, while the image model's downbeat direction is not its
    # upbeat direction negated (the free vectors of e015 are nearly orthogonal). The two-sided slider gives each side of
    # her reading its own direction; the smooth form keeps both sides on for every phrase.
    ConnectorArm("e023_beatrix_mood_slider_two_sided", "Beatrix's reading of a phrase's mood as a two-sided slider", "trained",
                 23, changed="e018's slider value split into its cheerful and gloomy sides, [max(a, 0), max(-a, 0), n], so "
                             "each side gets its own push direction",
                 question="With each side of her slider value free to push its own direction, does Beatrix's reading of a "
                          "phrase's mood steer the image both ways, for the phrases it trained on and for mood phrases it "
                          "never saw?", axis=True, sides="relu"),
    ConnectorArm("e024_beatrix_random_trunk_slider_two_sided",
                 "Control: the same two-sided slider on an untrained Beatrix of the same shape", "random", 24,
                 changed="e023's two-sided slider on the features of a randomly initialised trunk of the same shape (seed 0)",
                 question="Does e023's held-out effect come from what Beatrix learned? (A control that must fail on the "
                          "held-out phrases.)", axis=True, sides="relu"),
    ConnectorArm("e025_beatrix_mood_slider_smooth", "Beatrix's reading of a phrase's mood as a smooth two-sided slider",
                 "trained", 23,
                 changed="e018's slider value through [e^a, e^-a, n]: both sides on for every phrase, the sign tilting the "
                         "balance (no dead zone); the same training randomness as e023",
                 question="Does a two-sided slider without a dead zone steer the unseen gloomy phrases further than the "
                          "split one (e023)?", axis=True, sides="exp"),
)
CONNECTOR_IDS = [a.id for a in CONNECTOR_ARMS]
CONNECTOR_PAIRS = ((CONNECTOR_IDS[0], CONNECTOR_IDS[1]), (CONNECTOR_IDS[3], CONNECTOR_IDS[4]),
                   (CONNECTOR_IDS[5], CONNECTOR_IDS[6]), (CONNECTOR_IDS[7], CONNECTOR_IDS[8]))      # (Beatrix, untrained)
SLIDER_MAPS = {"one": "[a, n]", "relu": "[max(a, 0), max(-a, 0), n]", "exp": "[e^a, e^-a, n]"}
CONNECTOR_FREE = CONNECTOR_IDS[2]
CONNECTOR_FEATURES = "beatrix/mood_phrases_mini-beatrix-3_step212000.safetensors"   # in the data repo
CONNECTOR_CHECKPOINT = "AbstractPhil/alephllm-mini-beatrix-training, mini-beatrix-3 at step 212,000"
CONNECTOR_CLASSES = ("up", "down", "neutral")
CONNECTOR_DRAW = 1000                    # the training images: the LoRA arms' first draw (seeds 1000-1007) per class
CONNECTOR_STEPS, CONNECTOR_BATCH, CONNECTOR_LR = 720, 4, 1e-3
CONNECTOR_SAVE_EVERY = 144               # one pass over the 576 training images (5 passes)
CONNECTOR_TRACE_EVERY = 24
CONNECTOR_SEEDS = (101, 202)             # the evaluation: the 8 held-out scenes x these seeds = 16 cells
EVAL_TRAIN_PHRASES = {"up": ("cheerful and upbeat", "joyful and uplifting"),
                      "down": ("gloomy and downbeat", "somber and melancholy")}
CONNECTOR_GROUPS = {                     # read group -> (class, split or None = every split, expected direction)
    "trained_up": ("up", "train", 1), "trained_down": ("down", "train", -1),
    "heldout_up": ("up", "heldout", 1), "heldout_down": ("down", "heldout", -1),
    "neutral": ("neutral", None, 0)}
DIRECTION = {"up": 1, "down": -1, "neutral": 0}


def connector_lrs(source: str, fan_in: int, lr: "float | None" = None, contrast_l1: "float | None" = None) -> dict:
    """Adam's learning rates for the push's weight W and bias b (lr: CONNECTOR_LR). Adam moves every weight by about
    lr per step, so a linear map's output moves by about lr x the input's L1 norm: dense features (4,096 standardized
    numbers) would move the push thousands of times faster than a one-hot input. The fan-in rule (a matrix's Adam
    learning rate divided by its fan-in, as in Tensor Programs V) bounds the push's movement at about lr per dimension
    per step (e013, e014). It proved far too slow: only the class contrast moves consistently. With contrast_l1 (the L1
    size of the up-minus-down class-mean difference of the training inputs; whitened arms), W's rate is set so the
    class contrast moves at the free vector's pace: lr x 2 / contrast_l1 (the one-hot difference has L1 2)."""
    lr = CONNECTOR_LR if lr is None else lr
    if source == "onehot":
        return {"W": lr, "b": lr}
    return {"W": lr * 2.0 / contrast_l1 if contrast_l1 else lr / fan_in, "b": lr}


def connector_eval_sets(phrases: list, source: str) -> dict:
    """Every evaluation set of an arm, in render order: {key: (class, split, phrase text or None)}. Feature arms: per
    class the two training phrases of EVAL_TRAIN_PHRASES and every held-out phrase (neutral: its held-out phrases);
    the free vector: one set per class (its learned vector)."""
    if source == "onehot":
        return {c: (c, "train", None) for c in CONNECTOR_CLASSES}
    out: dict = {}
    for c in CONNECTOR_CLASSES:
        for t in EVAL_TRAIN_PHRASES.get(c, ()):
            if not any(p["text"] == t and p["class"] == c and p["split"] == "train" for p in phrases):
                raise ValueError(f"the features file has no training phrase {t!r} of class {c}")
            out[f"{c}/train/{t}"] = (c, "train", t)
        for p in phrases:
            if p["class"] == c and p["split"] == "heldout":
                out[f"{c}/heldout/{p['text']}"] = (c, "heldout", p["text"])
    return out


def connector_reads(diffs: dict, sets: dict, source: str = "trained") -> dict:
    """The rule fixed before the runs, on paired (push minus no push) mood-score differences {set key: [per cell]}:
    per group (CONNECTOR_GROUPS) the differences pooled over cells x phrases under the LoRA rule, renamed MOVES IT;
    the trained groups both moving it = TRAINED WORDS MOVE IT (the free vector: THE CLASS VECTORS MOVE IT), else NOT
    LEARNED (and no held-out verdict is read); both held-out groups moving it = HELD-OUT WORDS CARRY IT; NEUTRAL QUIET
    = the neutral group's mean is at most a third of the trained upbeat group's."""
    from .sana_experiments import arm_outcome
    groups: dict = {}
    for g, (c, split, d) in CONNECTOR_GROUPS.items():
        keys = [k for k, (cc, ss, _) in sets.items() if cc == c and (split is None or ss == split)]
        if keys:
            r = arm_outcome([x for k in keys for x in diffs[k]], d)
            r["OUTCOME"] = {"FLAVOR LORA": "MOVES IT"}.get(r["OUTCOME"], r["OUTCOME"])
            groups[g] = {**r, "sets": keys}
    trained = all(groups[g]["OUTCOME"] == "MOVES IT" for g in ("trained_up", "trained_down"))
    moves = "THE CLASS VECTORS MOVE IT" if source == "onehot" else "TRAINED WORDS MOVE IT"
    out = {"groups": groups, "TRAINED": moves if trained else "NOT LEARNED",
           "trained_effect": (groups["trained_up"]["mean"] - groups["trained_down"]["mean"]) / 2}
    if "heldout_up" in groups and "heldout_down" in groups:
        n = sum(groups[g]["OUTCOME"] == "MOVES IT" for g in ("heldout_up", "heldout_down"))
        out["HELD_OUT"] = ("NOT LEARNED" if not trained else
                           {2: "HELD-OUT WORDS CARRY IT", 1: "HELD-OUT WORDS CARRY IT ONE WAY",
                            0: "HELD-OUT WORDS DO NOT CARRY IT"}[n])
        out["heldout_effect"] = (groups["heldout_up"]["mean"] - groups["heldout_down"]["mean"]) / 2
    if "neutral" in groups:
        quiet = abs(groups["neutral"]["mean"]) <= abs(groups["trained_up"]["mean"]) / 3
        out["NEUTRAL"] = groups["neutral"]["OUTCOME"] = "NEUTRAL QUIET" if quiet else "NEUTRAL MOVES"
    return out


def connector_summary(reads: dict) -> str:
    g = reads["groups"]
    out = (f"{reads['TRAINED']} (upbeat {g['trained_up']['mean']:+.2f}, downbeat {g['trained_down']['mean']:+.2f})")
    if "HELD_OUT" in reads:
        out += (f"; {reads['HELD_OUT']} (upbeat {g['heldout_up']['mean']:+.2f}, downbeat "
                f"{g['heldout_down']['mean']:+.2f})")
    if "NEUTRAL" in reads:
        out += f"; {reads['NEUTRAL']} ({g['neutral']['mean']:+.2f})"
    return out


def connector_cross_reads(reads: dict) -> dict:
    """Across arms ({arm id: connector_reads output}), per (Beatrix, untrained trunk) pair of CONNECTOR_PAIRS: THE CONTROL
    FAILS AS IT SHOULD = the untrained trunk's held-out effect is at most a third of Beatrix's (hers positive); NOT
    READABLE when Beatrix's connector did not learn. And each feature arm's trained effect as a fraction of the free
    vector's."""
    out: dict = {}
    for b_id, r_id in CONNECTOR_PAIRS:
        btx, rnd = reads.get(b_id), reads.get(r_id)
        if not (btx and rnd and "heldout_effect" in btx and "heldout_effect" in rnd):
            continue
        hb, hr = btx["heldout_effect"], rnd["heldout_effect"]
        verdict = ("NOT READABLE (Beatrix's connector did not learn)" if btx["TRAINED"] == "NOT LEARNED" else
                   "NO HELD-OUT EFFECT TO CONTROL" if hb <= 0 else
                   "THE CONTROL FAILS AS IT SHOULD" if hr <= hb / 3 else "THE RANDOM TRUNK CARRIES IT TOO")
        out.setdefault("controls", {})[f"{b_id} vs {r_id}"] = {
            "beatrix": b_id, "random": r_id, "beatrix_heldout_effect": hb, "random_heldout_effect": hr,
            "OUTCOME": verdict}
    free = reads.get(CONNECTOR_FREE)
    if free and free.get("trained_effect"):
        out["of_free_vector"] = {a: r["trained_effect"] / free["trained_effect"] for a, r in reads.items()
                                 if a != CONNECTOR_FREE}
    hers = [a.id for a in CONNECTOR_ARMS if a.axis and a.source == "trained" and "heldout_down" in
            (reads.get(a.id) or {}).get("groups", {})]
    if len(hers) > 1:                                  # descriptive: her slider forms on the unseen gloomy phrases
        out["unseen_gloomy"] = {a: reads[a]["groups"]["heldout_down"]["mean"] for a in hers}
    return out


CONNECTOR_REFERENCES = (
    ("Gal, Alaluf, Atzmon, Patashnik, Bermano, Chechik, Cohen-Or, \"An Image is Worth One Word: Personalizing "
     "Text-to-Image Generation using Textual Inversion\" (2022)", "https://arxiv.org/abs/2208.01618"),
    ("Hu, Wang, Fang, Fu, Cheng, Yu, \"ELLA: Equip Diffusion Models with LLM for Enhanced Semantic Alignment\" (2024)",
     "https://arxiv.org/abs/2403.05135"),
    ("Yang, Hu, Babuschkin, Sidor, Liu, Farhi, Ryder, Pachocki et al., \"Tensor Programs V: Tuning Large Neural "
     "Networks via Zero-Shot Hyperparameter Transfer\" (2022)", "https://arxiv.org/abs/2203.03466"),
    ("Kingma, Ba, \"Adam: A Method for Stochastic Optimization\" (2014)", "https://arxiv.org/abs/1412.6980"),
    ("Ho, Salimans, \"Classifier-Free Diffusion Guidance\" (2022)", "https://arxiv.org/abs/2207.12598"),
    ("Radford, Kim, Hallacy et al., \"Learning Transferable Visual Models From Natural Language Supervision\" (2021)",
     "https://arxiv.org/abs/2103.00020"),
)


def render_connector_readme(arm: ConnectorArm, meta: dict, recipe: dict, phrases: "list | None" = None) -> str:
    """A connector experiment's README: the question, the design and the rule fixed before the run, the result."""
    rec = "\n".join(f"| {k} | {v} |" for k, v in recipe.items())
    refs = "\n".join(f"- {t}: {u}" for t, u in CONNECTOR_REFERENCES)
    out = [f"# {arm.id}: {arm.title}", "",
           f"Date: {DATE}. Model: {ANIMA.model_name} ([{BASE_MODEL}](https://huggingface.co/{BASE_MODEL})), frozen: the "
           "only trained numbers are the push's.", "",
           "## Question", arm.question, "",
           f"Changed from the reference connector (e013): {arm.changed}.", "",
           "## Design",
           f"- **Beatrix** is a byte-level language model from the geolip line ({CONNECTOR_CHECKPOINT}). Her features "
           "for a phrase: the state of the phrase's last byte after blocks 16, 18, 21 and 24, each normalized (layer "
           "norm without its affine), concatenated (4,096 numbers) and standardized per feature over a reference set of "
           "mood words and phrases that holds no word of a held-out phrase. The place was chosen before any training, "
           "by a probe on 60 single mood words (30 upbeat, 30 downbeat; ridge, 10-fold): .97-.98 accuracy at these "
           "blocks against .67-.68 for an untrained trunk of the same shape. The features are computed once, outside "
           "the notebook, and kept in the data repo.",
           "- **The push** = W f + b (1,024 numbers), added to every caption token of the text adapter's output (what "
           "the image model reads). W and b start at zero, so training starts from the stock model exactly. For the "
           "free vector, f is the mood class as a one-hot vector.",
           *([f"- **This experiment's input**: the features projected on the top {arm.whiten_k} principal components of "
              "the training phrases' features and scaled to unit variance per component (mean, components and scale fit "
              "on the training phrases only, and shipped with the weights). The number of components is the smallest at "
              "the best leave-one-out accuracy of the training phrases' mood class (nearest class mean); the held-out "
              "phrases played no part in any choice. The map's learning rate is set so the push's mood contrast moves at "
              "the free vector's pace: 1e-3 x 2 / (the L1 size of the cheerful-minus-gloomy class-mean difference in that "
              "space; the free vector's one-hot difference has L1 2). The first connectors (e013, e014) fed the 4,096 "
              "features in directly at 1e-3 / 4,096, and their phrase-dependent part barely moved in 720 steps."]
             if arm.whiten_k else []),
           *(["- **This experiment's input** (a slider value): two numbers per phrase, computed from the training phrases' "
              "features alone. a = the phrase's position on the axis from the centre of the gloomy training phrases to the "
              "centre of the cheerful ones (those centres at -1 and +1); n = its position on the axis from their midpoint "
              "toward the centre of the neutral training phrases (that centre at 1). The axes ship with the weights. The "
              "image model receives her reading of the phrase's mood and nothing phrase-specific: in e016 each phrase's own "
              "part of the push outweighed the mood contrast. The map's learning rate follows the free vector's pace (1e-3 "
              "x 2 / the L1 size of the cheerful-minus-gloomy class-mean difference of the inputs, "
              + ("about 2 here)." if arm.sides != "exp" else
                 "larger here, as the exponentials stretch both sides; the recipe gives its value).")]
             if arm.axis else []),
           *([f"- **Two-sided**: the map reads {SLIDER_MAPS[arm.sides]} instead of [a, n]. With one slider value, a "
              "linear map pushes gloomy phrases along the exact mirror of the cheerful push, but the image model's "
              "downbeat direction is not its upbeat direction negated (the free vectors of e015 are nearly orthogonal), "
              "and e018 moved only the cheerful side. "
              + ("Splitting the value by its sign gives each side its own direction; each side trains only on the "
                 "phrases on its side (checked before the run: "
                 + ("93% of her gloomy training phrases sit on the negative side and 90% beyond -0.25, and all four "
                    "unseen gloomy phrases sit beyond -0.25)." if arm.source == "trained" else
                    "the untrained trunk places 87% of its gloomy training phrases on the negative side and 80% beyond "
                    "-0.25, and 2 of its 4 unseen gloomy phrases on the negative side).") if arm.sides == "relu" else
                 "The exponentials keep both sides on for every phrase, the sign tilting the balance (no dead zone; at a "
                 "= 0 both sides contribute equally).")]
             if arm.axis and arm.sides != "one" else []),
           "- **Training**: Anima's own flow-matching objective, computed by the trainer's code (logit-normal "
           "timesteps, the noisy latent (1 - t) x0 + t noise, mean squared error to noise - x0), on the LoRA "
           "experiments' first-draw training images: 192 upbeat, 192 downbeat and 192 neutral renders of the stock "
           "model, every one captioned with the neutral prompt. Each image is paired with a random training phrase of "
           "its class. Everything except W and b is frozen.",
           "- **Sampling**: the push is added on both branches of classifier-free guidance, as a LoRA acts. It was "
           "trained without guidance on images drawn with guidance 4.5; on the conditional branch alone, guidance "
           "would multiply it a second time.",
           "- **Evaluation**: the 8 held-out scenes (in no training image) x seeds "
           f"{', '.join(map(str, CONNECTOR_SEEDS))} = 16 cells; the neutral prompt with the push of each evaluated "
           "phrase, against the same cell without a push. Evaluated: the first two training phrases of each mood and "
           "every held-out phrase (the free vector: its three class vectors).", ""]
    if phrases:
        out += ["| class | training phrases | held-out phrases |", "|---|---|---|"]
        for c in CONNECTOR_CLASSES:
            tr = ", ".join(p["text"] for p in phrases if p["class"] == c and p["split"] == "train")
            ho = ", ".join(p["text"] for p in phrases if p["class"] == c and p["split"] == "heldout")
            out.append(f"| {c} | {tr} | {ho} |")
        out += ["", "No word of a held-out phrase appears in a training phrase.", ""]
    out += ["## Recipe", "| setting | value |", "|---|---|", rec, "",
            "## The rule (fixed before the run)",
            "Per group of evaluated sets (trained upbeat, trained downbeat, held-out upbeat, held-out downbeat, "
            "neutral), the paired differences (push minus no push, per cell and phrase) are pooled and read in the "
            "group's direction: **MOVES IT** = the mean moves that way, at least 75% of the pairs move that way, and the "
            "mean is beyond 3 standard errors; **NO EFFECT** = within 2 standard errors of zero or under 60% that way; "
            "**MIXED** otherwise. **TRAINED WORDS MOVE IT** = both trained groups move it; otherwise the connector is "
            "**NOT LEARNED** and no verdict on the held-out phrases is read. **HELD-OUT WORDS CARRY IT** = both held-out "
            "groups move it. **NEUTRAL QUIET** = the neutral phrases' mean effect is at most a third of the trained "
            "upbeat group's. Across arms: **THE CONTROL FAILS AS IT SHOULD** = the untrained trunk's held-out effect "
            "(half of upbeat minus downbeat) is at most a third of Beatrix's. Content kept and the push's size are "
            "reported beside every read.", "", JUDGE_TEXT.replace("no-LoRA image", "no-push image"), ""]
    if meta.get("status") == "done":
        r = meta["result"]
        reads = r["reads"]
        out += ["## Result", meta.get("summary", ""), "",
                "| group | effect (mean +- SE) | pairs moving the expected way | content kept | verdict |",
                "|---|---|---|---|---|"]
        for g, v in reads["groups"].items():
            frac = v.get("frac_pos", v.get("frac_neg"))
            kept = v.get("content_kept")
            out.append(f"| {g.replace('_', ' ')} | {v['mean']:+.3f} +- {v['se']:.3f} | "
                       f"{'' if frac is None or g == 'neutral' else f'{frac:.0%}'} | "
                       f"{'' if kept is None else f'{kept:.3f}'} | **{v['OUTCOME']}** |")
        out += ["", "| phrase | class | effect (mean +- SE, 16 cells) | cells moving the expected way | content kept | "
                "push size |", "|---|---|---|---|---|---|"]
        for k, v in r["sets"].items():
            frac = v.get("frac_pos", v.get("frac_neg"))
            name = v["phrase"] or f"the {v['class']} vector"
            split = "" if v["phrase"] is None else f" ({'trained' if v['split'] == 'train' else 'held out'})"
            out.append(f"| {name}{split} | {v['class']} | {v['mean']:+.3f} +- {v['se']:.3f} | "
                       f"{'' if frac is None or v['class'] == 'neutral' else f'{frac:.0%}'} | "
                       f"{v['content_kept']:.3f} | {v['push_norm']:.3f} |")
        cross = r.get("cross") or {}
        for c in (cross.get("controls") or {}).values():
            if arm.id in (c["beatrix"], c["random"]):
                out += ["", f"Across arms ({c['beatrix'].split('_')[0]} against {c['random'].split('_')[0]}): Beatrix's "
                            f"held-out effect {c['beatrix_heldout_effect']:+.3f}, the untrained trunk's "
                            f"{c['random_heldout_effect']:+.3f}: **{c['OUTCOME']}**."]
        if cross.get("of_free_vector"):
            out += ["", "Trained effect as a fraction of the free vector's: " + ", ".join(
                f"{a.split('_')[0]} {v:.2f}" for a, v in cross["of_free_vector"].items()) + "."]
        if cross.get("unseen_gloomy") and arm.axis:
            out += ["", "Beatrix's slider forms on the unseen gloomy phrases (mean effect): " + ", ".join(
                f"{a.split('_')[0]} {v:+.3f}" for a, v in cross["unseen_gloomy"].items()) + "."]
        lora = r.get("lora_baselines") or {}
        if lora:
            out += ["", "For scale, the mood LoRAs on the same judge and held-out scenes (their final epoch at scale 1, "
                        "4 seeds = 32 cells):", "", "| LoRA experiment | effect (mean +- SE) | verdict |", "|---|---|---|"]
            out += [f"| {a} | {v['mean']:+.3f} +- {v['se']:.3f} | {v['OUTCOME']} |" for a, v in lora.items()]
        tr = r.get("trace_last")
        if tr:
            out += ["", f"End of training (step {tr['step']}): loss {tr['loss']:.4f}; mean push size per class over "
                        "its training inputs: " + ", ".join(f"{c} {n:.3f}" for c, n in tr["push_norm"].items())
                    + ". The caption tokens the push is added to have a mean size of about 5.4 (experiment e001)."]
        out += ["", "![the held-out scenes with and without the push](sheet.jpg)", "",
                "Rows: the held-out scenes at seed 101. Columns: no push, then one push per column (named in "
                "`result.json`, `sheet_columns`).", ""]
    elif meta.get("status") == "failed":
        out += ["## Result", f"The run failed: `{meta.get('error', '')}`.", ""]
    else:
        out += ["## Result", "Running.", ""]
    out += ["## Files",
            (f"- `connector/stepNNNN.safetensors`: W and b after every pass over the training images, with the input "
             f"projection mu, V, scale (float32; the push for features f is phi((f - mu) @ V.T / scale) @ W.T + b, "
             f"phi([a, n]) = {SLIDER_MAPS[arm.sides]})."
             if arm.axis and arm.sides != "one" else
             "- `connector/stepNNNN.safetensors`: W and b after every pass over the training images, with the input "
             "projection mu, V, scale (float32; the push for features f is ((f - mu) @ V.T / scale) @ W.T + b)."
             if arm.whiten_k or arm.axis else
             "- `connector/stepNNNN.safetensors`: W and b after every pass over the training images (float32; the push "
             "for features f is f @ W.T + b)."),
            "- `trace.json`: the training loss and the push's size per class during training.",
            "- `result.json`: every cell's score, the reads, the per-phrase effects.",
            "- `sheet.jpg`: the contact sheet.", "", "## References", refs, ""]
    return "\n".join(out)
