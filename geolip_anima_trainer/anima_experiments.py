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
stock model; e013-e015 condition the image on Beatrix's own states through a learned push (and its two controls).

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
)
CONNECTOR_IDS = [a.id for a in CONNECTOR_ARMS]
CONNECTOR_PAIRS = ((CONNECTOR_IDS[0], CONNECTOR_IDS[1]), (CONNECTOR_IDS[3], CONNECTOR_IDS[4]),
                   (CONNECTOR_IDS[5], CONNECTOR_IDS[6]))                                            # (Beatrix, untrained)
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
              "x 2 / the L1 size of the cheerful-minus-gloomy class-mean difference of the inputs, about 2 here)."]
             if arm.axis else []),
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
            ("- `connector/stepNNNN.safetensors`: W and b after every pass over the training images, with the input "
             "projection mu, V, scale (float32; the push for features f is ((f - mu) @ V.T / scale) @ W.T + b)."
             if arm.whiten_k or arm.axis else
             "- `connector/stepNNNN.safetensors`: W and b after every pass over the training images (float32; the push "
             "for features f is f @ W.T + b)."),
            "- `trace.json`: the training loss and the push's size per class during training.",
            "- `result.json`: every cell's score, the reads, the per-phrase effects.",
            "- `sheet.jpg`: the contact sheet.", "", "## References", refs, ""]
    return "\n".join(out)
