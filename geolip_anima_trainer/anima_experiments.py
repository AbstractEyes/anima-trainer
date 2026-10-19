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
# The second bed (2026-10-18): the fourth character replaced, since the first bed's sundress / flower garden came out younger
# than the others at one of its seeds (an image tagger read several of its pictures as a child). Every picture of the new
# character carries an age guard in its negative prompt; the other seven keep the card's negative, so their pictures are the
# first bed's, byte for byte. The new character came from a short screen of four candidates on the image tagger (none read as
# a child or rated questionable or explicit, the face and the hair in view, the fewest rated sensitive), after a first pick
# (an evening gown in a ballroom) came out framed on the chest.
BED2_FOURTH = ("tuxedo", "jazz club")
CHARACTERS_BED2 = CHARACTERS[:3] + (BED2_FOURTH,) + CHARACTERS[4:]
BED2_GUARDED = (3,)
AGE_GUARD = "child, loli, aged down"
AGE_TAGS = ("child", "loli", "aged down")     # the judge's own tags: a picture above AGE_BAR on any is withheld
AGE_BAR = 0.2
FRAME_TAG = "head out of frame"               # the judge's own tag: the new character's gate needs the face in view
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


# ---- e028: the word swap (a shattered word's own T5 pieces with another word's Qwen3 reading in its place) --------------
SWAP_TEST_ID = "e028_anima_word_swap"
SWAP_TEST_TITLE = ("The stock model: a mood word cut into pieces, read with another word's Qwen3 states in its place; does the "
                   "picture follow the answer's mood?")
SWAP_WORDS = {"up": ("gleeful", "elated"), "down": ("gloomy", "melancholy")}   # cut into pieces by T5, two Qwen3 tokens each
SWAP_FILLER = "workaday"                 # a neutral word of the same two Qwen3 tokens: the answer absent, or a neutral carrier
SWAP_FRAGMENT = "quotidian"              # a second neutral carrier whose T5 pieces are all fragments (▁ quot i dian; two
#                                          Qwen3 tokens): 'workaday' opens with a real word (▁work), so it is a morpheme-type
#                                          carrier; picked before any image as the first qualifying word of mundane, quotidian,
#                                          middling, tepid, staid, prosaic, bland (mundane is one Qwen3 token)
SWAP_SCENE_STEP = 4                      # e001's scenes [::4]: the eight contact-sheet rows
SWAP_N_SCENES = 8                        # len(e001's 32 scenes [::4]); the runner asserts it
SWAP_SEEDS = WORD_SEEDS                  # e026's seed
SWAP_ANSWER = "THE ANSWER CARRIES THE MOOD"


def swap_words() -> list[str]:
    return [w for ws in SWAP_WORDS.values() for w in ws]


def swap_mood(w: str) -> str:
    return next(m for m, ws in SWAP_WORDS.items() if w in ws)


def swap_sets() -> dict:
    """Every image set of e028 in render order: {key: (the word whose caption gives Qwen3's states, the word whose caption
    gives the T5 ids)}; None = the scene caption, SWAP_FILLER = the neutral word's caption."""
    W, F = swap_words(), SWAP_FILLER
    out: dict = {"neutral": (None, None), "filler_own": (F, F)}
    out.update({f"own_{a}": (a, a) for a in W})
    out.update({f"swap_{a}_{x}": (x, a) for a in W for x in W if x != a})
    out.update({f"filler_{a}": (F, a) for a in W})
    out.update({f"carrier_{x}": (x, F) for x in W})
    out["fragment_own"] = (SWAP_FRAGMENT, SWAP_FRAGMENT)
    out.update({f"fcarrier_{x}": (x, SWAP_FRAGMENT) for x in W})
    return out


def word_swap_reads(scores: dict) -> dict:
    """The registered reads from mood scores {set key: [score per scene]} (keys as swap_sets()).
    R1 (deciding), per cell (question A, scene): the mean over A's foreign cheerful answers minus the mean over its foreign
    gloomy answers, read upward under e001's rule. R2: R1's mean over the own axis (the cheerful words' own captions minus
    the gloomy words', per scene). R3 per word A, in its mood's direction: its answer's share (A|A - A|filler, 'THE ANSWER
    ADDS') and its question pieces under the filler's answer (A|filler - filler|filler); beside them the mood clause's own
    tint (filler|filler - the scene caption). R4 per answer word X: a foreign question's delivery, the mean over the foreign
    questions of (A|X - scene) over (X|X - scene), in X's direction, split by the question's mood. R5 per answer word X and
    pooled: the neutral carrier, (filler|X - filler|filler) in X's direction under the rule, and its size over (X|X -
    filler|filler)."""
    import numpy as np
    W, base, ff = swap_words(), np.array(scores["neutral"]), np.array(scores["filler_own"])

    def sgn(w):
        return 1 if swap_mood(w) == "up" else -1

    r1_cells, r1_per = [], {}
    for a in W:
        u = np.mean([scores[f"swap_{a}_{x}"] for x in W if x != a and swap_mood(x) == "up"], axis=0)
        d = np.mean([scores[f"swap_{a}_{x}"] for x in W if x != a and swap_mood(x) == "down"], axis=0)
        r1_cells += list(u - d)
        r1_per[a] = float(np.mean(u - d))
    r1 = flavor_outcome(r1_cells, 1, label=SWAP_ANSWER)
    own_axis = (np.mean([scores[f"own_{w}"] for w in W if swap_mood(w) == "up"], axis=0)
                - np.mean([scores[f"own_{w}"] for w in W if swap_mood(w) == "down"], axis=0))
    oa = float(np.mean(own_axis))

    def carrier_read(own_key: str, prefix: str):
        """R5's form for one carrier: per (answer X, scene) (carrier|X - carrier|carrier) in X's direction, pooled and per
        word under the rule; the size = the mean signed numerator over the mean signed (X|X - carrier|carrier)."""
        cc = np.array(scores[own_key])
        cells, num, den, per = [], [], [], {}
        for w in W:
            c = list(np.subtract(scores[f"{prefix}_{w}"], cc))
            cells += [sgn(w) * x for x in c]
            num.append(sgn(w) * float(np.mean(c)))
            den.append(sgn(w) * float(np.mean(np.subtract(scores[f"own_{w}"], cc))))
            per[w] = (flavor_outcome(c, sgn(w), label="THE CARRIER DELIVERS THE ANSWER"), num[-1] / den[-1] if den[-1] else None)
        sd = float(np.mean(den))
        return flavor_outcome(cells, 1, label="THE CARRIER DELIVERS THE ANSWER"), (float(np.mean(num)) / sd if sd else None), per

    r5, r5_size, r5_per = carrier_read("filler_own", "carrier")
    r5f, r5f_size, r5f_per = carrier_read("fragment_own", "fcarrier")
    words = {}
    for w in W:
        s = sgn(w)
        own = flavor_outcome(list(np.subtract(scores[f"own_{w}"], base)), s, label="THE WORD MOVES IT")
        delivery = {}
        for kind in ("same", "opposite"):
            qs = [a for a in W if a != w and (swap_mood(a) == swap_mood(w)) == (kind == "same")]
            num = float(np.mean([np.subtract(scores[f"swap_{a}_{w}"], base) for a in qs]))
            delivery[kind] = num / own["mean"] if own["mean"] else None
        words[w] = {"mood": swap_mood(w), "own": own,
                    "answer_adds": flavor_outcome(list(np.subtract(scores[f"own_{w}"], scores[f"filler_{w}"])), s,
                                                  label="THE ANSWER ADDS"),
                    "question_under_filler": flavor_outcome(list(np.subtract(scores[f"filler_{w}"], ff)), s,
                                                            label="THE QUESTION ALONE MOVES IT"),
                    "foreign_delivery": delivery,
                    "carrier": r5_per[w][0], "carrier_size": r5_per[w][1],
                    "fcarrier": r5f_per[w][0], "fcarrier_size": r5f_per[w][1],
                    "same_mood_swaps_minus_own": float(np.mean([np.subtract(scores[f"swap_{a}_{w}"], scores[f"own_{w}"])
                                                                for a in W if a != w and swap_mood(a) == swap_mood(w)]))}
    return {"R1": r1, "R1_per_question": r1_per, "R2_size": r1["mean"] / oa if oa else None, "own_axis": oa,
            "clause_tint": {**flavor_outcome(list(np.subtract(ff, base)), 0, label=""), "OUTCOME": "DESCRIPTIVE"},
            "words": words, "R5": r5, "R5_size": r5_size, "R5F": r5f, "R5F_size": r5f_size, "TEST": r1["OUTCOME"]}


def word_swap_summary(reads: dict) -> str:
    def f(v, spec):
        return "n/a" if v is None else format(v, spec)
    r1, r5, r5f = reads["R1"], reads["R5"], reads["R5F"]
    return (f"{reads['TEST']}: a word's own pieces with another word's reading, cheerful answers minus gloomy ones "
            f"{r1['mean']:+.3f} +- {r1['se']:.3f} ({r1['frac_pos']:.0%} of cells), {f(reads['R2_size'], '.2f')} of the words' "
            f"own axis; the neutral carriers: '{SWAP_FILLER}' {r5['OUTCOME']} ({f(reads['R5_size'], '.2f')} of the answer's "
            f"own effect), '{SWAP_FRAGMENT}' {r5f['OUTCOME']} ({f(reads['R5F_size'], '.2f')})")


def render_word_swap_readme(meta: dict, recipe: dict) -> str:
    """e028's README: the question, the design and the rule fixed before the run, and the result when done."""
    rec = "\n".join(f"| {k} | {v} |" for k, v in recipe.items())
    pieces = meta.get("pieces") or {}

    def cut(w, side="t5"):
        return " ".join(pieces.get(w, {}).get(side, [])) or "?"

    mood = {"up": "cheerful", "down": "gloomy"}
    rows = [f"| {w} | {mood[swap_mood(w)]} | {cut(w)} | {cut(w, 'qwen')} |" for w in swap_words()]
    rows.append(f"| {SWAP_FILLER} | neutral (the filler; a carrier whose lead piece is a real word) | {cut(SWAP_FILLER)} | "
                f"{cut(SWAP_FILLER, 'qwen')} |")
    rows.append(f"| {SWAP_FRAGMENT} | neutral (a carrier of fragments only) | {cut(SWAP_FRAGMENT)} | "
                f"{cut(SWAP_FRAGMENT, 'qwen')} |")
    n_sets = len(swap_sets())
    n_scenes = SWAP_N_SCENES
    out = [f"# {SWAP_TEST_ID}: {SWAP_TEST_TITLE}", "",
           f"Date: 2026-10-04. Model: {ANIMA.model_name} ([{BASE_MODEL}](https://huggingface.co/{BASE_MODEL})), no LoRA.", "",
           "## Question",
           "The adapter's queries start from the caption's T5 token ids and look up Qwen3's states. Experiment e026 found that "
           "a mood word the T5 vocabulary cuts into meaningless pieces gets its meaning only from Qwen3's states, looked up by "
           "those pieces at the word's position. A read on the adapter's own output then put another word's Qwen3 states in "
           "that position, under the first word's pieces: the pieces read the foreign states almost as fully as their own "
           "(0.93 of the attention) and the foreign word's mood passed through to the adapter's output at full size. Does the "
           "picture follow too: under the pieces of 'gleeful', does the reading of 'gloomy' make a gloomy picture? And does "
           "a neutral word's pieces carry a mood word's reading into the picture (an added word's slot)?", "",
           "## Design",
           f"- One mood word after the scene, \"{WORD_TEMPLATE.replace(PREFIX, '').format(s='{scene}', w='{word}')}\" "
           f"(after the model card's quality prefix; the negative prompt unchanged); e001's scenes [::{SWAP_SCENE_STEP}] at "
           f"seed {SWAP_SEEDS[0]} ({n_scenes} scenes). Four words cut into pieces by T5, each two Qwen3 tokens, and two "
           "neutral words of the same two tokens, so every swap keeps every position (asserted per scene before any image): "
           f"'{SWAP_FILLER}', the filler, whose lead T5 piece is a real word, and '{SWAP_FRAGMENT}', whose T5 pieces are all "
           "fragments (chosen before any image as the first word of a fixed list that is two Qwen3 tokens with no "
           "meaningful T5 piece):", "",
           "| word | mood | T5 pieces | Qwen3 pieces |", "|---|---|---|---|", *rows, "",
           "- A set = the T5 ids of one caption (the question) with the Qwen3 states of another (the answer): the scene "
           "caption; the filler caption on both sides; each word's own caption; each word's pieces with every other word's "
           "answer (cheerful and gloomy); each word's pieces with the filler's answer; the filler's pieces with each word's "
           f"answer; '{SWAP_FRAGMENT}' on both sides, and its pieces with each word's answer. {n_sets} sets x {n_scenes} "
           f"scenes = {n_sets * n_scenes} images.", "",
           "## Recipe", "| setting | value |", "|---|---|", rec, "",
           "## The rule (fixed before the run)",
           "e001's rule on per-cell differences of the mood score: a set **moves it** when the mean goes the stated way, at "
           "least 75% of the cells go that way and the mean is beyond 3 standard errors; **NO EFFECT** within 2 standard "
           "errors of zero or under 60% that way; **MIXED** otherwise. "
           f"**R1, the deciding read**: per cell (a word's pieces, a scene), the mean over its foreign cheerful answers minus "
           f"the mean over its foreign gloomy answers (its own answer left out), read upward: **{SWAP_ANSWER}**, **NO "
           "EFFECT** or **MIXED**; a MIXED result is extended to all 32 scenes before it is read. **R2**: R1's mean over the "
           "words' own axis (the cheerful words' own captions minus the gloomy words', per scene). **R3**, per word in its "
           "mood's direction: its own caption minus its pieces with the filler's answer (**THE ANSWER ADDS**), and its "
           "pieces with the filler's answer minus the filler caption (the question alone); beside them the mood clause's "
           "own tint (the filler caption minus the scene caption). **R4**, per answer word: a foreign question's delivery, "
           "the mean over the foreign questions of (swap minus scene) over (own minus scene), split by the question's mood. "
           "**R5**, the neutral carrier: per (answer word, scene), the filler's pieces with the word's answer minus the "
           "filler caption, in the word's direction under the rule (**THE CARRIER DELIVERS THE ANSWER**), pooled over the "
           "four words and per word, with its size over the word's own caption minus the filler caption. **R5-F**: the same "
           f"read for '{SWAP_FRAGMENT}' (its pieces with the word's answer minus its own caption, and the size over the "
           "word's own caption minus its own caption): a carrier of fragments only beside one whose lead piece means "
           "something.", "",
           f"**Limits fixed in advance**: one seed; {n_scenes} scenes; two words per mood; 'elated' was not read in pictures "
           "before; one word of each carrier kind.", "", ANIMA.judge_text, ""]
    if meta.get("status") == "done":
        r = meta["result"]
        r1, r5, r5f, ct = r["R1"], r["R5"], r["R5F"], r["clause_tint"]
        size = "n/a" if r["R2_size"] is None else f"{r['R2_size']:.2f}"
        out += ["## Result", meta.get("summary", ""), "",
                "| read | effect (mean +- SE) | cells that way | verdict |", "|---|---|---|---|",
                f"| R1: cheerful answers minus gloomy answers under foreign pieces | {r1['mean']:+.3f} +- {r1['se']:.3f} | "
                f"{r1['frac_pos']:.0%} | **{r1['OUTCOME']}** |",
                f"| R5: '{SWAP_FILLER}''s pieces with a mood word's answer | {r5['mean']:+.3f} +- {r5['se']:.3f} | "
                f"{r5['frac_pos']:.0%} | **{r5['OUTCOME']}** |",
                f"| R5-F: '{SWAP_FRAGMENT}''s pieces with a mood word's answer | {r5f['mean']:+.3f} +- {r5f['se']:.3f} | "
                f"{r5f['frac_pos']:.0%} | **{r5f['OUTCOME']}** |",
                f"| the mood clause's own tint (filler caption minus scene caption) | {ct['mean']:+.3f} +- {ct['se']:.3f} | "
                f"{ct['frac_pos']:.0%} upward | descriptive |", "",
                f"R2: R1 is {size} of the words' own axis ({r['own_axis']:+.3f}). The carriers' sizes: '{SWAP_FILLER}' "
                f"{'n/a' if r['R5_size'] is None else format(r['R5_size'], '.2f')}, '{SWAP_FRAGMENT}' "
                f"{'n/a' if r['R5F_size'] is None else format(r['R5F_size'], '.2f')} of the answer's own effect.", "",
                "| word | own caption (mean +- SE, verdict) | the answer adds | its pieces under the filler's answer | a "
                f"same-mood question delivers | an opposite-mood question delivers | '{SWAP_FILLER}' carries (size) | "
                f"'{SWAP_FRAGMENT}' carries (size) |",
                "|---|---|---|---|---|---|---|---|"]

        def sz(v):
            return "n/a" if v is None else format(v, ".2f")
        for w, v in r["words"].items():
            fd = v["foreign_delivery"]
            out.append(f"| {w} | {v['own']['mean']:+.3f} +- {v['own']['se']:.3f} ({v['own']['OUTCOME']}) | "
                       f"{v['answer_adds']['mean']:+.3f} ({v['answer_adds']['OUTCOME']}) | "
                       f"{v['question_under_filler']['mean']:+.3f} ({v['question_under_filler']['OUTCOME']}) | "
                       f"{sz(fd['same'])} | {sz(fd['opposite'])} | {v['carrier']['OUTCOME']} ({sz(v['carrier_size'])}) | "
                       f"{v['fcarrier']['OUTCOME']} ({sz(v['fcarrier_size'])}) |")
        out += ["", "| set | mood score | content kept |", "|---|---|---|"]
        for k in swap_sets():
            out.append(f"| {k} | {r['mood_score'][k]:+.3f} | {r['content_kept'][k]:.3f} |")
        out += ["", "![the swaps](sheet_swap.jpg)", "",
                "Rows: the scenes. Columns: " + ", ".join(r.get("sheet_columns", [])) + ".", ""]
    elif meta.get("status") == "failed":
        out += ["## Result", f"The run failed: `{meta.get('error', '')}`.", ""]
    else:
        out += ["## Result", "Running.", ""]
    out += ["## Files", "- `result.json`: every cell's scores and the reads.", "- `sheet_swap.jpg`: the contact sheet.", ""]
    return "\n".join(out)


# ---- e029: a relayed phrase reading in pictures (the sets, the reads, the README; the runner is AnimaRunner.run_relay) ------
# Beatrix's states for a phrase, mapped into Qwen3's space by a ridge regression, written over Qwen3's states at every token of the
# phrase at one depth; the adapter's T5 queries unchanged. The adapter-level grid picks the map and the depth; pictures confirm.
RELAY_TEST_ID = "e029_anima_relay_in_pictures"
RELAY_FORMS = {"bare": PREFIX + "an illustration of {s}, {p}.",       # the relay grid's caption form (never read in pictures)
               "mood": PREFIX + "an illustration of {s}, {p} mood."}  # e026-e028's form: the fallback when the bare form's ceiling fails
RELAY_TEMPLATE = RELAY_FORMS["bare"]
RELAY_SCENE_STEP = 4                                      # e001's scenes [::4]: the eight contact-sheet rows
RELAY_ANSWER = "THE RELAY CARRIES THE MOOD"
RELAY_CAPTION = "THE CAPTION CARRIES THE MOOD"
RELAY_CONTROL = "HER RELAY BEATS THE UNTRAINED ONE"
RELAY_MOUNT = "THE MOUNTED RELAY BEATS THE BARE ONE"       # added 2026-10-06, before any stage-B picture: a mount arm minus the relay
RELAY_PILOT_PER_MOOD = 2                                  # stage A: the first two single words of each mood, in the record order
RELAY_SEEDS = SWAP_SEEDS                                  # e026-e028's seed: one image per (phrase, scene)
RELAY_N_SCENES = 8                                        # len(e001's 32 scenes [::4]); the runner asserts it
RELAY_TEST_TITLE = "A relayed phrase reading in pictures"
RELAY_ENCODING_TOL = 1e-4                                 # identity check (ii): externally computed fp32 states against this
#                                                           runner's own fp32 encoding of the same caption (largest per-token
#                                                           relative difference)
# The grid's 16 held-out phrases split by T5 into fragments, in the grid's order, each with the neutral filler of the same Qwen3
# token count (the grid's answer-absent caption); stage B checks this list against the export's header.
RELAY_PHRASES = (
    {"text": "elated", "mood": "up", "filler": "workaday"},
    {"text": "jubilant and gleeful", "mood": "up", "filler": "workaday and nondescript"},
    {"text": "forlorn and desolate", "mood": "down", "filler": "workaday and nondescript"},
    {"text": "dismal", "mood": "down", "filler": "neutral"},
    {"text": "blithe", "mood": "up", "filler": "workaday"},
    {"text": "jaunty", "mood": "up", "filler": "nondescript"},
    {"text": "jovial", "mood": "up", "filler": "workaday"},
    {"text": "giddy", "mood": "up", "filler": "workaday"},
    {"text": "chirpy", "mood": "up", "filler": "workaday"},
    {"text": "exultant", "mood": "up", "filler": "nondescript"},
    {"text": "despondent", "mood": "down", "filler": "nondescript"},
    {"text": "dejected", "mood": "down", "filler": "workaday"},
    {"text": "sullen", "mood": "down", "filler": "workaday"},
    {"text": "lugubrious", "mood": "down", "filler": "nondescript"},
    {"text": "brooding", "mood": "down", "filler": "workaday"},
    {"text": "anguished", "mood": "down", "filler": "nondescript"},
)


def relay_prompt(p: str, scene: str, form: str = "bare") -> str:
    return RELAY_FORMS[form].format(s=scene, p=p)


def relay_sets(phrases: list, control: bool = True, companion: bool = False, mounts: tuple = ()) -> dict:
    """e029's image sets per phrase (dicts: text, mood 'up' / 'down', filler = the neutral phrase of the same Qwen3 token count):
    {key: (the phrase whose caption gives Qwen3's states, the phrase whose caption gives the T5 ids, the relay written over
    Qwen3's states at the phrase's tokens: None, 'her', 'companion', 'untrained' or a mount arm's name)}. ceiling = the phrase's
    caption on both sides; relay = the same with her relayed states at the chosen cell; companion = the same at the companion cell
    (when the two picks differ); control = the same with an untrained trunk's states through the chosen cell; each mount arm
    (mount_<group>, mount_<group>_own: the export's arms of her trunk with an arm group mounted) = the same with that arm's
    relayed states; filler = the phrase's T5 ids with the filler caption's Qwen3 states (the answer absent)."""
    out: dict = {}
    for p in phrases:
        t = p["text"]
        out[f"ceiling|{t}"] = (t, t, None)
        out[f"relay|{t}"] = (t, t, "her")
        if companion:
            out[f"companion|{t}"] = (t, t, "companion")
        if control:
            out[f"control|{t}"] = (t, t, "untrained")
        for m in mounts:
            assert m.startswith("mount_"), m
            out[f"{m}|{t}"] = (t, t, m)
        out[f"filler|{t}"] = (p["filler"], t, None)
    return out


def mount_label(arm: str) -> str:
    """A mount arm's plain name: mount_gCA -> 'nine arms, first seed'; mount_gCB_own -> 'nine arms, second seed, own cell'."""
    parts = arm.split("_")
    group = parts[1] if len(parts) > 1 else arm
    kind = {"gC": "nine arms", "gX": "eight arms"}.get(group[:2], f"arms {group}")
    seed = {"A": "first seed", "B": "second seed"}.get(group[-1:], group)
    return f"{kind}, {seed}" + (", own cell" if arm.endswith("_own") else "")


def relay_pilot_words(phrases: list) -> list:
    """Stage A's words: the first RELAY_PILOT_PER_MOOD single words (no space) of each mood, in the order given (the record list)."""
    return [p for m in ("up", "down") for p in [q for q in phrases if q["mood"] == m and " " not in q["text"]][:RELAY_PILOT_PER_MOOD]]


def relay_pilot_sets(words: list) -> dict:
    """Stage A (the bare form's ceiling pilot): ceiling and filler for each pilot word, in relay_sets' key form (stage B reuses them)."""
    return {k: v for k, v in relay_sets(words, control=False).items() if not k.startswith("relay|")}


def relay_pilot_read(scores: dict, words: list) -> dict:
    """Stage A's read: per (word, scene) the ceiling minus the filler in the word's mood direction under e001's rule, pooled over the
    cells: THE CAPTION CARRIES THE MOOD -> stage B in the same form; anything else -> the confirmation moves to the 'mood' form."""
    import numpy as np
    cells = []
    for p in words:
        s = 1 if p["mood"] == "up" else -1
        cells += [s * x for x in np.subtract(scores[f"ceiling|{p['text']}"], scores[f"filler|{p['text']}"])]
    r = flavor_outcome(cells, 1, label=RELAY_CAPTION)
    return {"ceiling": r, "form_next": "bare" if r["OUTCOME"] == RELAY_CAPTION else "mood"}


def relay_reads(scores: dict, phrases: list) -> dict:
    """The read fixed before any image, from mood scores {set key: [score per scene]}: per (phrase, scene) the relay minus the
    filler in the phrase's mood direction under e001's rule (THE RELAY CARRIES THE MOOD), pooled over all phrases and per mood;
    beside it the ceiling minus the filler under the same rule (the caption's own answer); the size = the mean signed relay effect
    over the mean signed ceiling effect, pooled, per mood and per phrase; with the control set, her relay minus the untrained relay
    in the phrase's mood direction under the same rule (HER RELAY BEATS THE UNTRAINED ONE); with the companion set, the
    companion cell's relay minus the filler under the same rule, with its size, beside the chosen cell's (the test is scored at
    the chosen cell). With mount arms (added 2026-10-06, before any stage-B picture): each read like the relay (minus the filler,
    its size) and against the relay, per (phrase, scene) in the phrase's mood direction under the same rule (THE MOUNTED RELAY
    BEATS THE BARE ONE); the deciding test stays the relay's."""
    import numpy as np
    has_control = all(f"control|{p['text']}" in scores for p in phrases)
    has_comp = all(f"companion|{p['text']}" in scores for p in phrases)
    mounts = sorted(m for m in {k.split("|", 1)[0] for k in scores if k.startswith("mount_")}
                    if all(f"{m}|{p['text']}" in scores for p in phrases))

    def block(ps):
        rel, cei, ctl, cmp_, num, den, cnum = [], [], [], [], [], [], []
        mnt, mnum, mvr = ({m: [] for m in mounts} for _ in range(3))
        for p in ps:
            s = 1 if p["mood"] == "up" else -1
            f = np.array(scores[f"filler|{p['text']}"])
            r, c = np.subtract(scores[f"relay|{p['text']}"], f), np.subtract(scores[f"ceiling|{p['text']}"], f)
            rel += [s * x for x in r]
            cei += [s * x for x in c]
            if has_control:
                ctl += [s * x for x in np.subtract(scores[f"relay|{p['text']}"], scores[f"control|{p['text']}"])]
            if has_comp:
                k = np.subtract(scores[f"companion|{p['text']}"], f)
                cmp_ += [s * x for x in k]
                cnum.append(s * float(np.mean(k)))
            for m in mounts:
                k = np.subtract(scores[f"{m}|{p['text']}"], f)
                mnt[m] += [s * x for x in k]
                mnum[m].append(s * float(np.mean(k)))
                mvr[m] += [s * x for x in np.subtract(scores[f"{m}|{p['text']}"], scores[f"relay|{p['text']}"])]
            num.append(s * float(np.mean(r)))
            den.append(s * float(np.mean(c)))
        sd = float(np.mean(den))
        out_b = {"relay": flavor_outcome(rel, 1, label=RELAY_ANSWER), "ceiling": flavor_outcome(cei, 1, label=RELAY_CAPTION),
                 "size": float(np.mean(num)) / sd if sd else None}
        if has_control:
            out_b["over_control"] = flavor_outcome(ctl, 1, label=RELAY_CONTROL)
        if has_comp:                                      # the companion cell, read the same way beside the pick of record
            out_b["companion"] = flavor_outcome(cmp_, 1, label=RELAY_ANSWER)
            out_b["companion_size"] = float(np.mean(cnum)) / sd if sd else None
        for m in mounts:                                  # each mount arm: read like the relay, and against it
            out_b[m] = {"relay": flavor_outcome(mnt[m], 1, label=RELAY_ANSWER),
                        "size": float(np.mean(mnum[m])) / sd if sd else None,
                        "over_relay": flavor_outcome(mvr[m], 1, label=RELAY_MOUNT)}
        return out_b

    out = {"all": block(phrases), "by_mood": {m: block([p for p in phrases if p["mood"] == m]) for m in ("up", "down")},
           "phrases": {p["text"]: block([p]) for p in phrases}, "control": has_control, "companion": has_comp,
           "mounts": mounts}
    out["TEST"] = out["all"]["relay"]["OUTCOME"]
    return out


def relay_summary(result: dict) -> str:
    """e029's one line: stage A's read, and stage B's when it has run."""
    def f(v, spec):
        return "n/a" if v is None else format(v, spec)
    out = []
    a = result.get("stage_a")
    if a:
        c = a["read"]["ceiling"]
        out.append(f"stage A: {c['OUTCOME']} in the bare form ({c['mean']:+.3f} +- {c['se']:.3f}, {c['frac_pos']:.0%} of cells; "
                   f"stage B in the {a['read']['form_next']} form)")
    b = result.get("reads")
    if b:
        al = b["all"]
        s = (f"stage B: {b['TEST']}: the relay minus the filler {al['relay']['mean']:+.3f} +- {al['relay']['se']:.3f} "
             f"({al['relay']['frac_pos']:.0%} of cells), {f(al['size'], '.2f')} of the caption's own effect")
        if b.get("control"):
            s += f"; against the untrained trunk {al['over_control']['OUTCOME']} ({al['over_control']['mean']:+.3f})"
        if b.get("companion"):
            s += f"; the companion cell {al['companion']['OUTCOME']} ({f(al['companion_size'], '.2f')} of the caption's effect)"
        for m in b.get("mounts", []):
            v = al[m]
            s += (f"; {mount_label(m)}: {v['relay']['OUTCOME']} ({f(v['size'], '.2f')} of the caption's effect), against the "
                  f"relay {v['over_relay']['OUTCOME']} ({v['over_relay']['mean']:+.3f})")
        out.append(s)
    return "; ".join(out) or "running"


def render_relay_readme(meta: dict, recipe: dict) -> str:
    """e029's README: the question, the design and the rule fixed before the run, and each stage's result when it is done."""
    rec = "\n".join(f"| {k} | {v} |" for k, v in recipe.items())
    mood = {"up": "cheerful", "down": "gloomy"}
    rows = [f"| {p['text']} | {mood[p['mood']]} | {p['filler']} |" for p in RELAY_PHRASES]
    pilot = relay_pilot_words(list(RELAY_PHRASES))
    out = [f"# {RELAY_TEST_ID}: {RELAY_TEST_TITLE}", "",
           f"Date: 2026-10-05. Model: {ANIMA.model_name} ([{BASE_MODEL}](https://huggingface.co/{BASE_MODEL})), no LoRA.", "",
           "## Question",
           "Beatrix is a byte-level language model. A grid of closed-form linear maps (ridge regressions fitted on other "
           "captions) carries her state at each token of a phrase into Qwen3's residual stream at one layer depth, at that "
           "token's own position, and Qwen3's remaining layers then run. At the output of Anima's text adapter the grid "
           "measured how much of the caption's own mood the relayed states deliver, for phrases the maps never saw. Does the "
           "picture follow: with Beatrix's relayed states written over Qwen3's at the phrase's tokens, does the picture move "
           "toward the phrase's mood as the caption itself moves it? And is that her training, not any plausible state at "
           "those positions (the same map applied to an untrained Beatrix of the same shape)?", "",
           "## Design",
           f"- The grid's own caption form, \"{RELAY_TEMPLATE.replace(PREFIX, '').format(s='{scene}', p='{phrase}')}\" (after "
           f"the model card's quality prefix; no 'mood' word, unlike e026-e028); e001's scenes [::{RELAY_SCENE_STEP}] at seed "
           f"{RELAY_SEEDS[0]} ({RELAY_N_SCENES} scenes).",
           "- The phrases: the grid's 16 held-out phrases that T5 cuts into fragments, each with a neutral filler of the same "
           "Qwen3 token count:", "", "| phrase | mood | filler |", "|---|---|---|", *rows, "",
           "- The sets, per phrase: **ceiling** = the caption's own Qwen3 states; **filler** = the filler caption's Qwen3 states "
           "under the phrase caption's T5 ids (the phrase's meaning gone from Qwen3's side); **relay** = the caption's Qwen3 "
           "states with Beatrix's relayed states written in at the chosen cell's depth at every token of the phrase, the "
           "remaining layers run; **control** = the same through the same map from an untrained Beatrix (random "
           "initialisation, seed 0); **companion** = the relay at a second cell (a paired tie band's pick) when it differs "
           "from the chosen one.",
           *([f"- Added 2026-10-06, before any stage-B picture: the **mount arms** "
              f"({', '.join(mount_label(m) for m in meta['mounts'])}) = the relay with Beatrix's trunk carrying a group of "
              "trained adapter arms (her eight stage arms with a caption arm trained over them), at the chosen cell and, "
              "labelled 'own cell', at that armed trunk's own chosen cell when it differs (128 images per arm). Each is read "
              "like the relay and "
              f"against it, per cell in the phrase's mood direction under the same rule (**{RELAY_MOUNT}**); the deciding "
              "test stays the relay's."] if meta.get("mounts") else []),
           "- The encoding: every arm is encoded in fp32 (Qwen3's bf16 weights upcast; this runner's usual pictures encode in "
           "bf16): the ceiling and filler arms by this runner's own fp32 encoding, the relay arms as fp32 final states "
           "computed outside the runner with the grid's arithmetic and supplied in place of the encoder's output. Before any "
           "image the checks: (i) the supplied-states path, fed this runner's own encoder output, reproduces its picture "
           f"pixel for pixel; (ii) supplied unpatched states match this runner's fp32 encoding of the same caption within "
           f"{RELAY_ENCODING_TOL:g} (the largest per-token relative difference); (iii) the picture difference between the usual "
           "bf16 encoding and the fp32 one is measured and quoted. Every token id is checked against the supplied file.",
           f"- **Stage A** (64 images): the first two single words of each mood ({', '.join(p['text'] for p in pilot)}) x "
           "{ceiling, filler} x the scenes. **Stage B**: the remaining sets, 384 images in all with stage A's 64 (reused), "
           "512 with the control, 640 with the companion; in the 'mood' form every set is rendered in that form (384 / 512 / "
           "640 new images; stage A's 64 stay its own read).", "",
           "## Recipe", "| setting | value |", "|---|---|", rec, "",
           "## The rule (fixed before the run)",
           "e001's rule on per-cell differences of the mood score: a set **moves it** when the mean goes the stated way, at "
           "least 75% of the cells go that way and the mean is beyond 3 standard errors; **NO EFFECT** within 2 standard "
           "errors of zero or under 60% that way; **MIXED** otherwise. Stage A: per (word, scene) the ceiling minus the "
           f"filler in the word's mood direction, pooled over its 32 cells: **{RELAY_CAPTION}** -> stage B in this form; "
           "otherwise stage B moves to the form \"..., {phrase} mood.\" with Beatrix's states taken on those captions "
           f"through the same map (fitted and checked in the bare form; the grid has no 'mood' form). Stage B, deciding: per (phrase, scene) the relay minus the filler in the phrase's mood "
           f"direction, pooled over all 16 phrases: **{RELAY_ANSWER}**, **NO EFFECT** or **MIXED**; beside it the ceiling "
           "minus the filler (the caption's own answer) and the size (the mean signed relay effect over the mean signed "
           f"ceiling effect), pooled, per mood and per phrase; the relay minus the control under the same rule (**"
           f"{RELAY_CONTROL}**); the companion cell read the same way beside the chosen cell (the test is scored at the chosen "
           "cell).", "",
           "**The prediction registered before any result**: stage A passes in the bare form at a smaller size than the "
           "'mood' form's; the relay reads NO EFFECT or MIXED on the mood; her relay stands above the untrained one.", "",
           f"**Limits fixed in advance**: one seed; {RELAY_N_SCENES} scenes; 16 phrases; one linear map per cell.", "",
           ANIMA.judge_text, ""]
    a = meta.get("stage_a")
    if a:
        r, idt = a["read"]["ceiling"], a.get("identity", {})
        out += ["## Result: stage A", f"**{r['OUTCOME']}**: the ceiling minus the filler {r['mean']:+.3f} +- {r['se']:.3f}, "
                f"{r['frac_pos']:.0%} of the cells that way; stage B runs in the **{a['read']['form_next']}** form.", "",
                "| word | mood | ceiling minus filler (mean) |", "|---|---|---|"]
        out += [f"| {w} | {mood[v['mood']]} | {v['diff']:+.3f} |" for w, v in a.get("words", {}).items()]
        if idt:
            out += ["", f"Identity checks: (i) the supplied-states path fed the runner's own encoder output: "
                    f"{'pixel for pixel' if idt.get('path_exact') else 'NOT identical'}; (iii) the usual bf16 encoding against "
                    f"the fp32 one: states {idt.get('bf16_vs_fp32_state_rel', float('nan')):.2e} (largest per-token relative), "
                    f"pictures {idt.get('bf16_vs_fp32_pixel_mean', float('nan')):.2f} mean absolute pixel difference (0-255), "
                    f"{idt.get('bf16_vs_fp32_pixel_max', float('nan')):.0f} at most.", ""]
        out += ["![stage A](sheet_stage_a.jpg)", "", "Rows: the scenes. Columns: per word, the ceiling then the filler.", ""]
    if meta.get("status") == "done" and meta.get("result", {}).get("reads"):
        res = meta["result"]
        b, idt = res["reads"], res.get("identity_b", {})
        out += ["## Result: stage B", meta.get("summary", ""), "",
                f"Identity check (ii): the supplied unpatched states against the runner's fp32 encoding, the largest per-token "
                f"relative difference {idt.get('export_vs_fp32_rel', float('nan')):.2e} (the bar {RELAY_ENCODING_TOL:g}); the "
                f"picture from the supplied states against the runner's usual one: "
                f"{idt.get('export_vs_own_pixel_mean', float('nan')):.2f} mean absolute pixel difference.", "",
                "| phrases | relay minus filler | ceiling minus filler | size | relay minus control | companion |",
                "|---|---|---|---|---|---|"]

        def cell(v, key):
            x = v.get(key)
            return "n/a" if not x else f"{x['mean']:+.3f} +- {x['se']:.3f} ({x['OUTCOME']})"

        def sz(x):
            return "n/a" if x is None else format(x, ".2f")
        for lab, v in [("all", b["all"]), *[(f"{mood[m]}", v) for m, v in b["by_mood"].items()],
                       *list(b["phrases"].items())]:
            out.append(f"| {lab} | {cell(v, 'relay')} | {cell(v, 'ceiling')} | {sz(v['size'])} | {cell(v, 'over_control')} | "
                       f"{cell(v, 'companion')} |")
        for m in b.get("mounts", []):
            out += ["", f"The mount arm **{mount_label(m)}**:", "",
                    "| phrases | mount relay minus filler | size | mount relay minus the relay |", "|---|---|---|---|"]
            for lab, v in [("all", b["all"]), *[(f"{mood[k]}", v) for k, v in b["by_mood"].items()],
                           *list(b["phrases"].items())]:
                out.append(f"| {lab} | {cell(v[m], 'relay')} | {sz(v[m]['size'])} | {cell(v[m], 'over_relay')} |")
        out += ["", "![stage B](sheet_stage_b.jpg)", "",
                "Rows: the scenes, for the first phrase of each mood. Columns: " + ", ".join(res.get("sheet_columns", [])) + ".",
                ""]
    elif meta.get("status") == "failed":
        out += ["## Result", f"The run failed: `{meta.get('error', '')}`.", ""]
    elif not a:
        out += ["## Result", "Running.", ""]
    out += ["## Files", "- `result.json`: every cell's scores, the reads and the identity checks.",
            "- `sheet_stage_a.jpg`, `sheet_stage_b.jpg`: the contact sheets.", ""]
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
    features: "str | None" = None    # the arm's own features file in the data repo (None: CONNECTOR_FEATURES); source names
                                     # its tensor ('trained', 'arms9' = her nine arms mounted, 'random'; through a surface
                                     # arm: 'qwen', 'plain' = the same mount with that arm masked, 'qwen_random')
    checkpoint: str = ""             # the trunk the arm's features come from (empty: CONNECTOR_CHECKPOINT)
    reading: str = ""                # the README's description of her features when the arm has its own file
    side_check: str = ""             # a two-sided slider's dead-zone check, measured on the arm's own training phrases
    date: str = ""                   # the day the arm was registered (empty: DATE, the first connectors')
    neutral_sides: bool = True       # False: in training the neutral rows enter a split slider as [0, 0, n] (neither side);
                                     # every phrase still goes through the map at evaluation


# ---- e031-e037: the hub sliders (her final trunk; three readings chosen and measured by the hub read, e030) ------------
HUB_TRUNK = "AbstractPhil/alephllm-mini-beatrix-training, mini-beatrix-3 at step 245,674, her final checkpoint"
HUB_DATE = "2026-10-07"
HUB_READ_ID = "e030_beatrix_hub_read"    # written by alephllm_diffusion.hubs (github AbstractEyes/alephllm-diffusion-experiments)
HUB_READ = f"the hub read ({HUB_READ_ID} in this repo: a read of her states made before any of these arms, 2026-10-07)"
# written by alephllm_diffusion.triangulate (the same repo): her trunk and hub on two byte forms of a caption (its own bytes; Qwen3's
# spelling of its tokens) against the Qwen3 states Anima's adapter reads; a read of her states, no pictures
TRIANGULATION_ID = "e038_beatrix_dual_extraction_read"
# written by alephllm_diffusion.triangulate.arm_read (the same repo): the dual-extraction read's gauges on her Qwen tokenizer arm
# (a surface arm: she reads Qwen3's spelling of a text as she reads its own bytes), mounted; a read of her states, no pictures
ARM_READ_ID = "e045_beatrix_qwen_arm_read"
# written by alephllm_diffusion.nano (the same repo): the four nano v1.1 bodies (each version and each untrained birth) on the
# dual-extraction read's rows and gauges, their stacks and hubs, and each reader's own mood axis; a read of states, no pictures
NANO_READ_ID = "e046_nano_dual_extraction_read"
# written by alephllm_diffusion.nano (the same repo): the relay in pictures over whole captions, Qwen3's final states replaced by
# a reader's mapped states (Qwen3, Beatrix 3s, the nano bodies, a token lookup as the floor)
NANO_RELAY_ID = "e047_nano_relay_in_pictures"
# written by alephllm_diffusion.nano (the same repo): the uniform comparator, fixed mood directions from this model's own prepared
# conditioning with each reader setting only the amount (Qwen3, Beatrix 3s, the nano bodies)
NANO_COMPARATOR_ID = "e048_nano_uniform_comparator"
# written by alephllm_diffusion.nano (the same repo): the trained relay, a ridge-born connector per reader taught through this
# model's frozen text adapter to reproduce the context Qwen3's own states give, then the relay's pictures (connectors shipped)
NANO_RELAY_TRAINED_ID = "e049_nano_trained_relay"
HUB_FEATURES = {r: f"beatrix/mood_phrases_{r.replace('/', '-')}_mini-beatrix-3_step245674.safetensors"
                for r in ("close/stream/18", "close/hub/22", "close/both/18", "close/stream/20",
                          "frame/qwen3-arm/20")}                                                     # in the data repo
_STD = ("normalized (layer norm without its affine) and standardized per feature over a reference set of mood words and "
        "phrases that holds no word of a held-out phrase")
HUB_READINGS = {
    "close/stream/18": (
        "Her features for a phrase: her state after block 18 at the full stop that closes the phrase (the phrase is read with "
        f"a full stop after it, so the state follows the whole phrase; 1,024 numbers), {_STD}. The reading was chosen by a "
        f"rule fixed before {HUB_READ}: among her stream's and her hubs' readings at ten blocks, the largest separation of "
        "the unseen phrases on the slider's own axis (2.19 between the mean slider values of the unseen cheerful and gloomy "
        "phrases, with the training centres at +1 and -1; every unseen phrase on its side; 56 of 60 training phrases on their "
        "side when each is left out of the fit). The features are computed by the session that runs these arms and kept in "
        "the data repo."),
    "close/hub/22": (
        "Her features for a phrase: the blackboard of block 22's hub after it has read the phrase and the full stop that "
        "closes it (the hub's memory of the text: 4 x 64 slots of 1,024 numbers, each slot unit length; 262,144 numbers), "
        f"{_STD}. In {HUB_READ} it is the best of her hubs' readings (separation 1.83, every unseen phrase on its side, 55 "
        "of 60 left-out training phrases), below her stream's best (2.19), with the most even spread of the two moods' "
        "training phrases on the slider's axis (standard deviations .33 and .38, against .54 and .36 for the stream's "
        "reading) and its neutral phrases at zero (+.02, +.13; the stream's reading puts them on the gloomy side, -.23 and "
        "-.38). Over 2,048 captions her hubs' blackboards also match the image model's own caption context better than her "
        "stream does (a ridge from the blackboard predicts the text adapter's pooled output at R2 .60-.64, from her pooled "
        "stream at .55-.57). The features are computed by the session that runs these arms and kept in the data repo."),
    "close/both/18": (
        "Her features for a phrase: the hub's blackboard at block 18 after the phrase and its closing full stop, and her "
        f"state at that full stop after block 18, each {_STD} and each divided by the length of its own mood axis on the "
        "training phrases, side by side (263,168 numbers). Through the slider's axis the slider value is then exactly the "
        f"mean of the two readings' own slider values, so each counts equally whatever its width. In {HUB_READ} this mean "
        "separated the unseen phrases by 1.98 (every unseen phrase on its side; 57 of 60 left-out training phrases). The "
        "features are computed by the session that runs these arms and kept in the data repo."),
    "close/stream/20": (
        "Her features for a phrase: her state after block 20 at the full stop that closes the phrase (the phrase is read with "
        f"a full stop after it, so the state follows the whole phrase; 1,024 numbers), {_STD}. The block was chosen by the "
        f"dual-extraction read ({TRIANGULATION_ID} in this repo: a read of her states made before these arms, 2026-10-07): "
        "of her stream's blocks, the one closest to the Qwen3 states this model's text adapter reads that still carries her "
        "mood direction onto Qwen3's, the gloomy side included (blocks 22-28 sit closer, but there only 2 of the 4 unseen "
        "gloomy phrases land on Qwen3's gloomy side). In "
        f"{HUB_READ} this reading separated the unseen phrases by 2.16 (every unseen phrase on its side; 57 of 60 training "
        "phrases on their side when each is left out of the fit) and placed the four neutral training phrases at -0.19, "
        "-0.99, -0.18 and -0.15 on the slider ('an ordinary scene', 'plain', 'everyday', 'neutral'; the gloomy centre at -1). "
        "The features are computed by the session that runs these arms and kept in the data repo."),
    "frame/qwen3-arm/20": (
        "Her features for a phrase, read through her Qwen tokenizer arm: the phrase in this model's caption form ('" + PREFIX
        + "an illustration of a quiet street, <phrase>.'), its Qwen3 tokens spelled as Qwen3 spells them and read through the "
        "arm (AbstractPhil/beatrix-tokenizers, surface/qwen3/mse_gXA_o0: a detachable adapter after each of her 32 blocks, "
        "trained over eight of her stage arms, which stay mounted, so that she reads Qwen3's spelling of a text as she reads "
        "its own bytes); her state after block 20 at the full stop that closes the caption (1,024 numbers), " + _STD + ". "
        f"The read of the mounted arm ({ARM_READ_ID} in this repo, 2026-10-08, before these arms) measured the reading on "
        "10,000 caption tokens: through the arm, Qwen3's spelling lines up with her own bytes at every block (.955-.971 after "
        "whitening, .41-.73 without the arm), it sits as close to the Qwen3 states this model's text adapter reads as her own "
        "bytes in the same mount (.546 against .544 at block 24), and it carries her mood direction onto Qwen3's at blocks "
        "14-20; block 20 passed the mood rule on both caption draws and stays the slider's block. Read on the training "
        "phrases' axis, it places the trained gloomy pair at -0.86 and -0.94 ('gloomy and downbeat', 'somber and "
        "melancholy'; the gloomy centre at -1) and the four neutral training phrases at +0.17, -0.31, +0.08 and -0.27 "
        "('an ordinary scene', 'plain', 'everyday', 'neutral'). The features are computed by the session that runs these "
        "arms and kept in the data repo."),
}
# the dead-zone check of each arm's own trunk, measured in the read before any run: of the 30 gloomy training phrases how many
# sit below 0 and below -0.25 on the slider's axis, and of the 4 unseen gloomy phrases how many below -0.25
HUB_SIDE = {"e031": (30, 29, 4), "e032": (26, 24, 2), "e033": (30, 29, 4), "e034": (30, 28, 4), "e035": (26, 24, 3),
            "e036": (30, 30, 4), "e037": (27, 26, 2), "e039": (30, 30, 4), "e040": (26, 24, 2), "e041": (30, 30, 4),
            "e042": (28, 26, 4), "e043": (21, 20, 0), "e044": (28, 27, 4)}
# (None would mark a check not yet measured: the registry test refuses it)


def _side_check(arm: str, whose: str = "her") -> str:
    if HUB_SIDE[arm] is None:
        return ""
    neg, beyond, unseen = HUB_SIDE[arm]
    return (f"{neg} of {whose} 30 gloomy training phrases sit on the negative side and {beyond} beyond -0.25, and {unseen} "
            "of the 4 unseen gloomy phrases sit beyond -0.25).")


def _hub_arm(n: int, name: str, title: str, source: str, reading: str, changed: str, question: str,
             neutral_sides: bool = True) -> "ConnectorArm":
    tag, control = f"e0{n}", source.endswith("random")
    return ConnectorArm(f"{tag}_{name}", title, source, 32 if control else 30, question=question, changed=changed,
                        axis=True, sides="relu", features=HUB_FEATURES[reading], checkpoint=HUB_TRUNK,
                        reading=HUB_READINGS[reading], side_check=_side_check(tag, "its" if control else "her"),
                        date=HUB_DATE, neutral_sides=neutral_sides)


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
    # e023-e025 read NOT LEARNED on step 212,000's last-byte features: the cheerful side moved, the gloomy side barely. The hub
    # sliders run the split form on her final trunk at three readings the hub read (e030) chose and measured: her stream at
    # the full stop closing the phrase (the read's rule picked it), her hub's blackboard (its two moods spread evenly) and the
    # two together; hers share one training randomness (seed 30), the controls another (32).
    _hub_arm(31, "beatrix_stream_closing_slider", "Beatrix's reading of a phrase at its closing full stop, as a two-sided "
             "slider", "trained", "close/stream/18",
             changed="e023's two-sided slider on her final trunk (step 245,674) at a new reading: her state after block 18 at "
                     "the full stop that closes the phrase (e023: the last byte at blocks 16-24 of step 212,000)",
             question="Does Beatrix's reading of a phrase's mood at the full stop that closes it, split into its cheerful "
                      "and gloomy sides, steer the image both ways, for the phrases it trained on and for mood phrases it "
                      "never saw?"),
    _hub_arm(32, "beatrix_random_trunk_stream_closing_slider", "Control: the same slider on an untrained Beatrix of the same "
             "shape", "random", "close/stream/18",
             changed="e031's slider on the features of a randomly initialised trunk of the same shape (seed 0), at the "
                     "same reading",
             question="Does e031's held-out effect come from what Beatrix learned? (A control that must fail on the "
                      "held-out phrases.)"),
    _hub_arm(33, "beatrix_stream_closing_slider_nine_arms", "Beatrix's closing-stop reading with her nine trained arms "
             "mounted", "arms9", "close/stream/18",
             changed="e031 with her nine trained arms mounted, all live (the arm group trained on her final trunk writes "
                     "into her states at every block); the same training randomness as e031",
             question="With her nine trained arms mounted, does her reading steer the image as her bare trunk's does "
                      "(e031)?"),
    _hub_arm(34, "beatrix_hub_slider", "Beatrix's hub: the blackboard's reading of a phrase as a two-sided slider",
             "trained", "close/hub/22",
             changed="e031's slider with the blackboard of block 22's hub as the reading in place of the stream's state; the "
                     "same training randomness as e031",
             question="Does her hub's reading of a phrase steer the image both ways, beside her stream's (e031), for the "
                      "phrases it trained on and for mood phrases it never saw?"),
    _hub_arm(35, "beatrix_random_trunk_hub_slider", "Control: the hub slider on an untrained Beatrix of the same shape",
             "random", "close/hub/22",
             changed="e034's slider on the hub of a randomly initialised trunk of the same shape (seed 0)",
             question="Does e034's held-out effect come from what Beatrix learned? (A control that must fail on the "
                      "held-out phrases.)"),
    _hub_arm(36, "beatrix_hub_and_stream_slider", "Beatrix's hub and stream together as a two-sided slider", "trained",
             "close/both/18",
             changed="e031's slider with the mean of two readings at block 18 as its value: the hub's blackboard and the "
                     "stream's state at the closing stop; the same training randomness as e031",
             question="Do her hub's and her stream's readings together steer the image both ways, beside each alone (e031, "
                      "e034)?"),
    _hub_arm(37, "beatrix_random_trunk_hub_and_stream_slider", "Control: the combined slider on an untrained Beatrix of the "
             "same shape", "random", "close/both/18",
             changed="e036's slider on a randomly initialised trunk of the same shape (seed 0)",
             question="Does e036's held-out effect come from what Beatrix learned? (A control that must fail on the "
                      "held-out phrases.)"),
    # e031-e037 read NOT LEARNED, the gloomy side unmoved; their weights show why: her readings put the neutral training
    # phrases partly on the gloomy input, so the neutral pictures trained the gloomy push toward no change. Session 5 tests
    # that on the reading the dual-extraction read (e038) picked, her stream at block 20: the neutral rows off both sides in
    # training (e039; its control e040), beside the same reading with the hub sliders' recipe (e041). Hers share e031's
    # training randomness (seed 30), the control the controls' (32).
    _hub_arm(39, "beatrix_block20_slider_neutral_off", "Beatrix's closing-stop reading at block 20 as a two-sided slider, "
             "the neutral phrases kept off both sides in training", "trained", "close/stream/20",
             changed="e031's two-sided slider at block 20 (the dual-extraction read's pick, e038), with the neutral training "
                     "phrases feeding neither side in training ([0, 0, n]); the same training randomness as e031",
             question="With the neutral pictures kept off both sides of the slider in training, does Beatrix's reading of a "
                      "phrase's mood steer the image both ways, for the phrases it trained on and for mood phrases it "
                      "never saw?", neutral_sides=False),
    _hub_arm(40, "beatrix_random_trunk_block20_slider_neutral_off", "Control: the same slider on an untrained Beatrix of "
             "the same shape", "random", "close/stream/20",
             changed="e039's slider (the neutral phrases off both sides in training) on the features of a randomly "
                     "initialised trunk of the same shape (seed 0), at the same reading",
             question="Does e039's held-out effect come from what Beatrix learned? (A control that must fail on the "
                      "held-out phrases.)", neutral_sides=False),
    _hub_arm(41, "beatrix_block20_slider", "Beatrix's closing-stop reading at block 20 as a two-sided slider, the hub "
             "sliders' recipe", "trained", "close/stream/20",
             changed="e031's two-sided slider at block 20 (the dual-extraction read's pick, e038), every training phrase "
                     "read through the map as in e031; the same training randomness as e031 and e039",
             question="At block 20, does the hub sliders' recipe steer the image as e039's does? (e039's one-factor "
                      "comparison: the two differ only in the neutral phrases' inputs during training.)"),
    # e039-e041 read NOT LEARNED: with the neutral phrases off both sides her strongly gloomy readings moved the pictures the
    # gloomy way and the change tracked her reading, but the trained gloomy pair, which her reading of the phrase alone puts
    # near neutral, stayed put. Session 6 reads each phrase in this model's caption form through her Qwen tokenizer arm (e045
    # measured it: Qwen3's spelling read through it lines up with her own bytes), in e039's form: her reading through the arm,
    # its control on an untrained copy with its own arm, and her plain reading in the same mount (the arm's one factor).
    _hub_arm(42, "beatrix_qwen_arm_slider", "Beatrix's reading through her Qwen tokenizer arm as a two-sided slider", "qwen",
             "frame/qwen3-arm/20",
             changed="e039's slider (the neutral phrases off both sides in training) on a new reading: each phrase in this "
                     "model's caption form, Qwen3's spelling of its tokens read through her Qwen tokenizer arm (mounted over "
                     "the eight stage arms it was trained with), her state after block 20 at the full stop closing the "
                     "caption; the same training randomness as e039",
             question="Read through the arm that makes her read Qwen3's spelling as her own bytes, does Beatrix's reading "
                      "of a phrase's mood steer the image both ways, for the phrases it trained on and for mood phrases it "
                      "never saw?", neutral_sides=False),
    _hub_arm(43, "beatrix_random_trunk_qwen_arm_slider", "Control: the same slider on an untrained Beatrix of the same "
             "shape with its own Qwen tokenizer arm", "qwen_random", "frame/qwen3-arm/20",
             changed="e042's slider on the features of a randomly initialised trunk of the same shape (seed 0) read through "
                     "its own Qwen tokenizer arm (the same recipe, trained on that copy), at the same reading",
             question="Does e042's held-out effect come from what Beatrix learned? (A control that must fail on the "
                      "held-out phrases.)", neutral_sides=False),
    _hub_arm(44, "beatrix_plain_mounted_slider", "Beatrix's plain reading in the same mount as a two-sided slider",
             "plain", "frame/qwen3-arm/20",
             changed="e042's slider with the Qwen tokenizer arm masked and the caption's own bytes read (the eight stage "
                     "arms still mounted): her plain reading at the same block, site and caption form; the same training "
                     "randomness as e042",
             question="Does her reading of Qwen3's spelling through the arm steer the image as her plain reading of the "
                      "caption's own bytes does? (e042's one-factor comparison: the two differ only in the arm and the "
                      "byte form it serves.)", neutral_sides=False),
)
CONNECTOR_IDS = [a.id for a in CONNECTOR_ARMS]
CONNECTOR_PAIRS = ((CONNECTOR_IDS[0], CONNECTOR_IDS[1]), (CONNECTOR_IDS[3], CONNECTOR_IDS[4]),
                   (CONNECTOR_IDS[5], CONNECTOR_IDS[6]), (CONNECTOR_IDS[7], CONNECTOR_IDS[8]),
                   (CONNECTOR_IDS[10], CONNECTOR_IDS[11]), (CONNECTOR_IDS[12], CONNECTOR_IDS[11]),
                   (CONNECTOR_IDS[13], CONNECTOR_IDS[14]), (CONNECTOR_IDS[15], CONNECTOR_IDS[16]),
                   (CONNECTOR_IDS[17], CONNECTOR_IDS[18]), (CONNECTOR_IDS[20], CONNECTOR_IDS[21]))   # (Beatrix, untrained)
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
    hers = [a.id for a in CONNECTOR_ARMS if a.axis and a.source in ("trained", "qwen", "plain") and "heldout_down" in
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
           f"Date: {arm.date or DATE}. Model: {ANIMA.model_name} ([{BASE_MODEL}](https://huggingface.co/{BASE_MODEL})), "
           "frozen: the "
           "only trained numbers are the push's.", "",
           "## Question", arm.question, "",
           f"Changed from the reference connector (e013): {arm.changed}.", "",
           "## Design",
           (f"- **Beatrix** is a byte-level language model from the geolip line ({arm.checkpoint}). {arm.reading}"
            if arm.reading else
            f"- **Beatrix** is a byte-level language model from the geolip line ({CONNECTOR_CHECKPOINT}). Her features "
            "for a phrase: the state of the phrase's last byte after blocks 16, 18, 21 and 24, each normalized (layer "
            "norm without its affine), concatenated (4,096 numbers) and standardized per feature over a reference set of "
            "mood words and phrases that holds no word of a held-out phrase. The place was chosen before any training, "
            "by a probe on 60 single mood words (30 upbeat, 30 downbeat; ridge, 10-fold): .97-.98 accuracy at these "
            "blocks against .67-.68 for an untrained trunk of the same shape. The features are computed once, outside "
            "the notebook, and kept in the data repo."),
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
                 + (arm.side_check if arm.side_check else
                    "93% of her gloomy training phrases sit on the negative side and 90% beyond -0.25, and all four "
                    "unseen gloomy phrases sit beyond -0.25)." if arm.source == "trained" else
                    "the untrained trunk places 87% of its gloomy training phrases on the negative side and 80% beyond "
                    "-0.25, and 2 of its 4 unseen gloomy phrases on the negative side).") if arm.sides == "relu" else
                 "The exponentials keep both sides on for every phrase, the sign tilting the balance (no dead zone; at a "
                 "= 0 both sides contribute equally).")]
             if arm.axis and arm.sides != "one" else []),
           *(["- **The neutral phrases in training**: they enter training as [0, 0, n], feeding neither side of the "
              "slider. In the hub sliders (e031-e037) her readings placed the neutral training phrases partly on the gloomy "
              "side, so the neutral pictures, which must stay neutral, also trained the gloomy side's direction, and none "
              "of her sliders moved the gloomy phrases. Here each side's direction learns "
              "from its own mood's pictures alone, and the neutral pictures train only the neutral reading's column and the "
              "bias. At evaluation every phrase, the neutral ones too, goes through the same map."]
             if not arm.neutral_sides else []),
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


# ---- e050: booru counterpoint pairs on the stock model (the words and their cross-talk; sets kept in the data repo) ------
CP_TEST_ID = "e050_anima_counterpoint_words"
CP_TEST_TITLE = "The stock model: booru counterpoint pairs, their words and their cross-talk"
CP_DATE = "2026-10-17"
CP_TEMPLATE = PREFIX + "1girl, solo, {tag}{framing}, {outfit}, {setting}."
CP_FRAMINGS = ("upper body", "full body")
# key: (+ tag, - tag or None = the plain prompt, framing). Danbooru tags as the model's captions write them (spaces, not
# underscores), every one in the judge's vocabulary; the two tags of a pair come from one slot, so a picture normally
# carries one of them. Adults only: the age pair only pushes older.
CP_PAIRS = {
    "hair_colour": ("blonde hair", "black hair", "upper body"),
    "eye_colour": ("red eyes", "blue eyes", "upper body"),
    "body_size": ("plump", "skinny", "full body"),
    "expression": ("smile", "frown", "upper body"),
    "eye_openness": ("wide-eyed", "half-closed eyes", "upper body"),
    "hair_length": ("very long hair", "short hair", "upper body"),
    "eye_shape": ("tareme", "tsurime", "upper body"),
    "figure_scale": ("giant", "mini person", "full body"),
    "eye_colour_2": ("green eyes", "purple eyes", "upper body"),
    "hair_colour_2": ("pink hair", "blue hair", "upper body"),
    "colour_scheme": ("monochrome", None, "upper body"),
    "age": ("mature female", None, "upper body"),
    "proportions": ("chibi", None, "full body"),
}
CP_DEFAULT = ("hair_colour", "eye_colour", "body_size", "expression", "eye_openness")
CP_OVERLAP = 1 / 3          # a pair's words moving another pair's score by more than this share of its span: an overlap
# overlaps named before the run as expected (parts of the face that move together); they hold no pair back
CP_EXPECTED_OVERLAP = frozenset(frozenset(p) for p in (
    ("eye_shape", "eye_openness"), ("expression", "eye_openness"), ("expression", "eye_shape"),
    ("expression", "eye_colour"), ("expression", "eye_colour_2")))
CP_RATINGS = ("general", "sensitive", "questionable", "explicit")      # the judge's own rating tags
CP_WITHHELD = ("questionable", "explicit")   # a picture rated so stays off the sheets, and its set out of the data repo


def cp_slug(text: str) -> str:
    """'half-closed eyes' -> 'half-closed-eyes' (part of a set's name)."""
    import re
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def cp_set_name(tag: "str | None", framing: str, bed: int = 1) -> str:
    """The name of one tag's set (None: the plain prompt) at one framing, over the 16 cells (the second bed's: cp2_...)."""
    return f"cp{'' if bed == 1 else bed}_{cp_slug(tag) if tag else 'plain'}_{cp_slug(framing)}"


def cp_prompt(tag: "str | None", framing: str, ci: int, bed: int = 1) -> str:
    outfit, setting = (CHARACTERS if bed == 1 else CHARACTERS_BED2)[ci]
    return CP_TEMPLATE.format(tag=f"{tag}, " if tag else "", framing=framing, outfit=outfit, setting=setting)


def cp_guard(item: dict, ci: int, bed: int) -> dict:
    """The second bed's age guard: the guarded character's pictures are drawn with it added to the negative prompt."""
    if bed != 1 and ci in BED2_GUARDED:
        item["negative"] = NEGATIVE + ", " + AGE_GUARD
    return item


def cp_items(tag: "str | None", framing: str, bed: int = 1) -> list[dict]:
    """One set in the data repo's set format: per cell its name, seed, prompt and caption (and, for a guarded character
    of the second bed, its negative prompt). The caption is the plain prompt, so the tag lives only in the picture and
    the set can train a slider on its attribute later."""
    return [cp_guard({"name": f"{ci:02d}_{sd}", "seed": sd, "prompt": cp_prompt(tag, framing, ci, bed),
                      "caption": cp_prompt(None, framing, ci, bed)}, ci, bed) for ci, sd in attr_cells()]


def cp_pairs(keys=None) -> list[str]:
    """A run's pairs (default CP_DEFAULT), checked."""
    keys = list(CP_DEFAULT if keys is None else keys)
    unknown = [k for k in keys if k not in CP_PAIRS]
    if unknown:
        raise ValueError(f"unknown pair(s) {unknown}; the pairs are {list(CP_PAIRS)}")
    if not keys or len(set(keys)) != len(keys):
        raise ValueError(f"name each pair once (got {keys})")
    return keys


def cp_sets(keys: list[str]) -> dict:
    """{set name: (tag or None, framing)} in drawing order: the plain prompt at each framing in use, then each pair's
    + and - tags."""
    sets: dict = {}
    for f in CP_FRAMINGS:
        if any(CP_PAIRS[k][2] == f for k in keys):
            sets[cp_set_name(None, f)] = (None, f)
    for k in keys:
        plus, minus, f = CP_PAIRS[k]
        for t in (plus, minus):
            if t:
                sets[cp_set_name(t, f)] = (t, f)
    return sets


def cp_sides(key: str) -> "tuple[str, str | None, str]":
    """(the + set, the - set or None, the plain set) of a pair."""
    plus, minus, f = CP_PAIRS[key]
    return cp_set_name(plus, f), (cp_set_name(minus, f) if minus else None), cp_set_name(None, f)


def cp_judge_tags(keys: list[str]) -> list[str]:
    """Every tag the judge must have: the pairs', the earlier screen's (for the side-by-side read), the ratings, the age
    tags and the framing tag."""
    tags = [t for k in keys for t in CP_PAIRS[k][:2] if t]
    tags += [t for p, m, _ in ATTRIBUTES.values() for t in (p, m) if t]
    return list(dict.fromkeys(tags + list(CP_RATINGS) + list(AGE_TAGS) + [FRAME_TAG]))


def cp_screen(p, index: dict) -> tuple:
    """The content screen on the judge's probabilities [n pictures, tags]: (each picture's rating, its highest read on
    AGE_TAGS, the pictures withheld: rated questionable or explicit, or an age read above AGE_BAR)."""
    import numpy as np
    ratings = [CP_RATINGS[int(i)] for i in np.argmax(p[:, [index[r] for r in CP_RATINGS]], axis=1)]
    ages = [float(x) for x in np.asarray(p[:, [index[t] for t in AGE_TAGS]], dtype=float).max(1)]
    held = [i for i, (r, a) in enumerate(zip(ratings, ages)) if r in CP_WITHHELD or a > AGE_BAR]
    return ratings, ages, held


def cp_reads(judge: dict, keys: list[str]) -> dict:
    """The registered reads from judge scores {set name: {pair: [score per cell]}} (names as cp_sets()): per side the
    tag against the plain prompt under the words rule, the pair's verdict, its words on every pair's score in units of
    that pair's span (cross-talk), the overlaps, and whether the pair goes on to the slider stage."""
    import numpy as np
    span = {}
    for b in keys:
        hi, lo, plain = cp_sides(b)
        span[b] = float(np.mean(np.subtract(judge[hi][b], judge[lo or plain][b])))
    out: dict = {}
    for a in keys:
        hi, lo, plain = cp_sides(a)
        up = flavor_outcome(list(np.subtract(judge[hi][a], judge[plain][a])), 1, label="THE WORD MOVES IT")
        down = (flavor_outcome(list(np.subtract(judge[lo][a], judge[plain][a])), -1, label="THE WORD MOVES IT")
                if lo else None)
        sides = [up] + ([down] if down else [])
        if all(r["OUTCOME"] == "THE WORD MOVES IT" for r in sides):
            verdict = "A HANDLE"
        elif all(r["OUTCOME"] == "NO EFFECT" for r in sides):
            verdict = "NO HANDLE"
        else:
            verdict = "MIXED"
        out[a] = {"plus": CP_PAIRS[a][0], "minus": CP_PAIRS[a][1], "framing": CP_PAIRS[a][2], "words_up": up,
                  "words_down": down, "span": span[a], "OUTCOME": verdict}
    for a in keys:
        hi, lo, plain = cp_sides(a)
        cross = {b: (float(np.mean(np.subtract(judge[hi][b], judge[lo or plain][b]))) / span[b] if span[b] else 0.0)
                 for b in keys}
        over = [b for b in keys if b != a and abs(cross[b]) > CP_OVERLAP]
        counted = [b for b in over if out[b]["OUTCOME"] == "A HANDLE" and frozenset((a, b)) not in CP_EXPECTED_OVERLAP]
        worst = max(((abs(v), b) for b, v in cross.items() if b != a), default=(0.0, None))
        out[a].update(cross_talk=cross, worst_cross_talk=worst[0], worst_with=worst[1], overlapping=over,
                      overlap_counted=counted, to_sliders=out[a]["OUTCOME"] == "A HANDLE" and not counted)
    return out


def cp_side_by_side(stored_cells: list[dict], judge12: dict) -> dict:
    """This run beside the earlier attribute screen (ATTR_TEST_ID: another card, the same characters, seeds and recipe)
    on the sets both drew with the same prompts: the plain prompt's score on each of that screen's attributes (both means
    and the per-cell correlation), and the words both measured (each tag's effect on its own attribute). stored_cells =
    that screen's result.json 'cells' (attr_cells() order); judge12 = {set name: {attribute: [score per cell]}} here."""
    import numpy as np

    def col(key, a):
        return np.array([c[key][a] for c in stored_cells], dtype=float)

    def mse(x):
        return float(np.std(x, ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0

    plain = cp_set_name(None, "upper body")
    out: dict = {"plain": {}, "words": {}}
    if plain not in judge12 or "neutral" not in stored_cells[0]:
        return out
    for a in ATTRIBUTES:
        here, there = np.asarray(judge12[plain][a], dtype=float), col("neutral", a)
        r = float(np.corrcoef(here, there)[0, 1]) if here.std() > 0 and there.std() > 0 else None
        out["plain"][a] = {"here": float(here.mean()), "there": float(there.mean()), "cell_r": r}
    for a, (p, m, _) in ATTRIBUTES.items():
        for tag, key in ((p, f"{a}+"), (m, f"{a}-")):
            name = cp_set_name(tag, "upper body") if tag else None
            if name in judge12 and key in stored_cells[0]:
                here = np.subtract(judge12[name][a], judge12[plain][a])
                there = col(key, a) - col("neutral", a)
                out["words"][tag] = {"attribute": a, "here": float(here.mean()), "here_se": mse(here),
                                     "there": float(there.mean()), "there_se": mse(there)}
    return out


def render_counterpoint_readme(meta: dict, recipe: dict) -> str:
    """e050's README: why, the design and the rule fixed before the run, and the result when done."""
    keys = meta.get("pairs") or list(CP_DEFAULT)
    res = meta.get("result") or {}
    posts = {a: v.get("posts", {}) for a, v in (res.get("pairs") or {}).items()}
    sets = cp_sets(keys)

    def n_posts(a, tag):
        n = posts.get(a, {}).get(tag) if tag else None
        return f"{n:,}" if isinstance(n, int) else ""

    rec = "\n".join(f"| {k} | {v} |" for k, v in recipe.items())
    rows = "\n".join(f"| {a} | {p} | {m or '(none: the plain prompt)'} | {f} | {n_posts(a, p)} | {n_posts(a, m)} |"
                     for a, (p, m, f) in ((k, CP_PAIRS[k]) for k in keys))
    chars = "; ".join(f"{o} / {s}" for o, s in CHARACTERS)
    named = ", ".join(" with ".join(sorted(p)) for p in sorted(CP_EXPECTED_OVERLAP, key=sorted)
                      if p <= set(keys)) or "none among these pairs"
    refs = "\n".join(f"- {t}: {u}" for t, u in ATTR_REFERENCES)
    out = [f"# {CP_TEST_ID}: {CP_TEST_TITLE}", "",
           f"Date: {CP_DATE}. Model: {ANIMA.model_name} ([{BASE_MODEL}](https://huggingface.co/{BASE_MODEL})), no LoRA.", "",
           "## Why",
           "Anima learned from pictures described by Danbooru tags, so its own vocabulary is the natural handle. Mood words "
           "such as \"cheerful\" and \"gloomy\" are not tags, and the earlier mood experiments here found their effect "
           "lopsided and tied to how the text encoder splits the words. A counterpoint pair is two tags from one slot of "
           "that vocabulary (two hair colours, two eye colours, two body sizes), so a picture normally carries one of them. "
           "Before a pair can carry a slider, the stock model has to show that each tag moves the picture its way, and that "
           "it moves only its own attribute.", "",
           "## Questions",
           "1. Does each tag move its own attribute, the + tag one way and the - tag the other, against the plain prompt?",
           "2. Do a pair's tags move the other pairs' attributes (overlap)?",
           "3. How much of the rest of the picture does each tag change (content kept)?", "",
           "## Design",
           f"- **Characters**: \"1girl, solo\" with eight outfits and settings ({chars}) x seeds "
           f"{', '.join(map(str, ATTR_SEEDS))}: 16 cells per set, the cells of {ATTR_TEST_ID}. All characters are adults.",
           "- **Framing**: \"upper body\" for the face and hair pairs, \"full body\" for the body pairs; each framing has its "
           "own plain set.",
           "- **Pairs** (posts: how many Danbooru posts carry the tag, from the judge's tag list):", "",
           "| pair | + tag | - tag | framing | posts (+) | posts (-) |", "|---|---|---|---|---|---|", rows, "",
           f"- **Sets**: the plain prompt at each framing in use, each + tag, each - tag: {len(sets)} sets x 16 = "
           f"{16 * len(sets)} pictures.",
           "- **Kept and reused**: every set is saved in the data repo once drawn (its pictures, the prompt and seed of "
           "each, a contact sheet and the settings that drew it) and a later run with the same settings downloads it "
           "instead of drawing it again. The caption stored with each picture is the plain prompt, so a set can train a "
           "slider on its attribute later.", "",
           "## Recipe", "| setting | value |", "|---|---|", rec, "",
           "## The judge",
           f"Each picture is scored on its pixels by the [WD EVA02-Large tagger v3](https://huggingface.co/{TAGGER}), an "
           "image tagger trained on Danbooru tags, with its reference preprocessing. A pair's score is the log-odds of its "
           "+ tag minus the log-odds of its - tag (the + tag alone for a pair without a - tag). *Content kept* is the CLIP "
           "ViT-L/14 image cosine to the plain prompt's picture of the same cell. The tagger also rates every picture "
           "(general, sensitive, questionable, explicit): a picture rated questionable or explicit is left off the sheets, "
           "and its set stays out of the data repo.", "",
           "## The rule (fixed before the run)",
           "Per side, over the 16 cells, read in the expected direction: **THE WORD MOVES IT** when the mean moves that way, "
           "at least 75% of the cells move that way and the mean is beyond 3 standard errors; **NO EFFECT** within 2 "
           "standard errors of zero or under 60% that way; **MIXED** otherwise. A pair is **A HANDLE** when every side "
           "moves it, **NO HANDLE** when no side does, **MIXED** otherwise. **Cross-talk**: pair a's tags (its + set minus "
           "its - set, or minus the plain set) on pair b's score, in units of b's own span (the same difference on b's own "
           "sets); a pair **overlaps** b when the cross-talk is above a third. A pair goes on to the slider stage when it is "
           "a handle and overlaps no other handle, except overlaps named before the run as expected (" + named + "); an "
           "overlap on a pair that is not a handle is reported but holds nothing back, since that pair's span is no usable "
           "unit.", ""]
    if meta.get("status") == "done":
        r = res["pairs"]
        out += ["## Result", meta.get("summary", ""), "",
                "| pair | + tag vs plain | - tag vs plain | verdict | worst cross-talk | content kept (+ / -) | to the sliders |",
                "|---|---|---|---|---|---|---|"]
        for a, v in r.items():
            up, dn = v["words_up"], v.get("words_down")
            kept = v.get("content_kept", {})
            out.append(f"| {a} | {up['mean']:+.2f} +- {up['se']:.2f} ({up['OUTCOME']}) | "
                       + (f"{dn['mean']:+.2f} +- {dn['se']:.2f} ({dn['OUTCOME']})" if dn else "") + f" | **{v['OUTCOME']}** | "
                       + (f"{v['worst_cross_talk']:.2f} ({v['worst_with']})" if v["worst_with"] else "") + " | "
                       + " / ".join(f"{kept[s]:.3f}" for s in ("+", "-") if s in kept) + " | "
                       + ("yes" if v["to_sliders"] else "no") + " |")
        names = list(r)
        out += ["", "Cross-talk (row = a pair's tags, column = the pair read; units of that pair's span; the diagonal is 1):",
                "", "| tags of | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
        for a in names:
            out.append(f"| {a} | " + " | ".join(f"{r[a]['cross_talk'][b]:+.2f}" for b in names) + " |")
        sbs = res.get("side_by_side") or {}
        if sbs.get("plain"):
            out += ["", f"Beside the earlier screen ({ATTR_TEST_ID}, another card, the same prompts and seeds): the plain "
                        "prompt's scores, and each tag both runs drew against the plain prompt.", "",
                    "| the plain prompt on | here | earlier | per-cell r |", "|---|---|---|---|"]
            out += [f"| {a} | {v['here']:+.2f} | {v['there']:+.2f} | "
                    + ("" if v["cell_r"] is None else f"{v['cell_r']:.2f}") + " |" for a, v in sbs["plain"].items()]
            if sbs.get("words"):
                out += ["", "| tag | attribute | here | earlier |", "|---|---|---|---|"]
                out += [f"| {t} | {v['attribute']} | {v['here']:+.2f} +- {v['here_se']:.2f} | "
                        f"{v['there']:+.2f} +- {v['there_se']:.2f} |" for t, v in sbs["words"].items()]
        held = res.get("withheld") or {}
        out += ["", ("The judge rated no picture questionable or explicit." if not held else
                     "Withheld (rated questionable or explicit; off the sheets, the set out of the data repo): "
                     + "; ".join(f"{n}: {', '.join(v)}" for n, v in held.items()) + ".")]
        pics = res.get("pictures") or {}
        if pics:
            out += ["", f"Pictures: {pics.get('total')} in all, {pics.get('drawn')} drawn in this run, "
                        f"{pics.get('reused')} reused from the data repo or this runtime."]
        out += ["", "Sheets (`sheet_<pair>.jpg`): the eight characters at the first seed; columns: the - tag (where there "
                    "is one), the plain prompt, the + tag.", ""]
        out += [f"![{a}](sheet_{a}.jpg)" for a in names] + [""]
    elif meta.get("status") == "failed":
        out += ["## Result", f"The run failed: `{meta.get('error', '')}`.", ""]
    else:
        out += ["## Result", "Running.", ""]
    out += ["## Files",
            "- `result.json`: every cell's scores on every pair, the reads, the cross-talk, the side-by-side read, where "
            "each set came from and where it is kept.",
            "- `tagger_probs.npz`: every picture's probability for each of the judge's tags (float16, one array per set, "
            "rows in cell order; `tags` names the columns), so another tag can be read without the pictures.",
            "- `sheet_*.jpg`: contact sheets.", "", "## References", refs, ""]
    return "\n".join(out)


# ---- e051: the counterpoint sliders, stage R (a slider between a pair's tags by the model's own predictions) -----------
CPS_TEST_ID = "e051_anima_counterpoint_sliders"
CPS_TEST_TITLE = "The stock model: a slider between each pair's tags, by the model's own predictions (no reader yet)"
CPS_DATE = "2026-10-18"
CPS_PAIRS = ("hair_colour", "expression", "eye_openness", "body_size")   # e050's handles to the sliders + body size
CPS_ONE_WAY = ("body_size",)        # e050: skinny did not move its judge, so this slider runs from the plain prompt to plump
CPS_AMOUNTS = (-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5)
CPS_PARITY_CELLS = 2                # the slider at +1 drawn on the first cells beside the tag's own pictures
CPS_REFERENCES = ATTR_REFERENCES + (
    ("Liu, Li, Du, Torralba, Tenenbaum, \"Compositional Visual Generation with Composable Diffusion Models\" (ECCV 2022)",
     "https://arxiv.org/abs/2206.01714"),)


def cps_pairs(keys=None) -> list[str]:
    """A slider run's pairs (default CPS_PAIRS), checked against the counterpoint pairs."""
    keys = cp_pairs(CPS_PAIRS if keys is None else keys)
    return keys


def cps_one_way(key: str) -> bool:
    return key in CPS_ONE_WAY or CP_PAIRS[key][1] is None


def cps_amounts(key: str) -> tuple:
    """A pair's amounts: both ways, or from the plain prompt toward the + tag only."""
    return tuple(a for a in CPS_AMOUNTS if a >= 0) if cps_one_way(key) else CPS_AMOUNTS


def cps_side_tag(key: str, amount: float) -> "str | None":
    plus, minus, _ = CP_PAIRS[key]
    return None if amount == 0 else plus if amount > 0 else minus


def cps_cached(key: str, amount: float, bed: int = 1) -> "str | None":
    """The counterpoint set an amount of 0 or +-1 is (the plain prompt, the tag itself); None for one the slider draws."""
    return cp_set_name(cps_side_tag(key, amount), CP_PAIRS[key][2], bed) if amount in (-1.0, 0.0, 1.0) else None


def cps_set_name(key: str, amount: float, bed: int = 1) -> str:
    """A drawn slider set's name: the side's tag, the framing and the amount (x0.5 half way to the tag, x1.5 past it)."""
    return f"cps{'' if bed == 1 else bed}_{cp_slug(cps_side_tag(key, amount))}_{cp_slug(CP_PAIRS[key][2])}_x{abs(amount):g}"


def cps_items(key: str, amount: float, cells=None, bed: int = 1) -> list[dict]:
    """A slider set in the data repo's set format: per cell the plain prompt (also its caption), the side's tag prompt
    and the amount: the conditional prediction moved from the plain prompt's toward the tag prompt's by it."""
    tag, f = cps_side_tag(key, amount), CP_PAIRS[key][2]
    return [cp_guard({"name": f"{ci:02d}_{sd}", "seed": sd, "prompt": cp_prompt(None, f, ci, bed),
                      "caption": cp_prompt(None, f, ci, bed), "mix_prompt": cp_prompt(tag, f, ci, bed),
                      "mix_weight": abs(amount)}, ci, bed)
            for ci, sd in (attr_cells() if cells is None else cells)]


def cps_sets(keys: list[str]) -> dict:
    """{set name: (pair, amount)} the slider draws, in drawing order (0 and +-1 come from the counterpoint sets)."""
    return {cps_set_name(k, a): (k, a) for k in keys for a in cps_amounts(k) if cps_cached(k, a) is None}


def cps_reads(scores: dict, spans: dict, keys: list[str]) -> dict:
    """The registered reads. scores = {pair: {amount: {judged pair: [score per cell]}}} over every amount of the pair
    (0 and +-1 included); spans = {judged pair: its counterpoint span}. Per pair: THE DIAL (the per-cell slope of its own
    score on the amount, under the words rule), THE MIDDLE per side (at half the amount the score lies between the plain
    prompt's and the tag's in >= 75% of the cells: GRADED, else A SWITCH), PAST THE TAG per side (1.5 against 1;
    descriptive), the slider's cross-talk (its mean slope on every judged pair's score over that pair's span, beside its
    own; an overlap above a third of its own) and the verdict."""
    import numpy as np
    out: dict = {}
    for a in keys:
        am = sorted(scores[a])
        own = {x: scores[a][x][a] for x in am}
        dial = flavor_outcome(cell_slopes(own), 1, label="A DIAL")
        middle, beyond = {}, {}
        for side, mark in ((1, "+"), (-1, "-")):
            if side * 1.0 not in own:
                continue
            lo, hi, half = (np.asarray(own[x], dtype=float) for x in (0.0, side * 1.0, side * 0.5))
            between = float(np.mean((half - lo) * (hi - half) > 0))          # strictly inside: an end is a switch
            span = float(np.mean(hi - lo))
            middle[mark] = {"between": between, "share": float(np.mean(half - lo)) / span if span else 0.0,
                            "OUTCOME": "GRADED" if between >= 0.75 else "A SWITCH"}
            if side * 1.5 in own:
                beyond[mark] = flavor_outcome(list(np.asarray(own[side * 1.5], dtype=float) - hi), side,
                                              label="PAST THE TAG")
        judged = list(scores[a][am[0]])
        cross = {b: (float(np.mean(cell_slopes({x: scores[a][x][b] for x in am}))) / spans[b] if spans.get(b) else 0.0)
                 for b in judged}
        reach = cross.get(a, 0.0)
        over = [b for b in judged if b != a and reach and abs(cross[b]) > CP_OVERLAP * abs(reach)]
        graded = bool(middle) and all(m["OUTCOME"] == "GRADED" for m in middle.values())
        verdict = ("NO DIAL" if dial["OUTCOME"] != "A DIAL" else "A SWITCH" if not graded
                   else "A SLIDER WITH CROSS-TALK" if over else "A SLIDER")
        out[a] = {"amounts": am, "means": {f"{x:+g}": float(np.mean(own[x])) for x in am}, "dial": dial,
                  "middle": middle, "beyond": beyond, "cross_talk": cross, "reach": reach, "overlapping": over,
                  "OUTCOME": verdict}
    return out


def render_slider_readme(meta: dict, recipe: dict) -> str:
    """e051's README: why, the design and the rule fixed before the run, and the result when done."""
    keys = meta.get("pairs") or list(CPS_PAIRS)
    res = meta.get("result") or {}
    rec = "\n".join(f"| {k} | {v} |" for k, v in recipe.items())
    rows = "\n".join(f"| {k} | {CP_PAIRS[k][0]} | {'(one way: the plain prompt)' if cps_one_way(k) else CP_PAIRS[k][1]} | "
                     f"{CP_PAIRS[k][2]} | {', '.join(f'{x:+g}' for x in cps_amounts(k))} |" for k in keys)
    n_sets = len(cps_sets(keys))
    refs = "\n".join(f"- {t}: {u}" for t, u in CPS_REFERENCES)
    out = [f"# {CPS_TEST_ID}: {CPS_TEST_TITLE}", "",
           f"Date: {CPS_DATE}. Model: {ANIMA.model_name} ([{BASE_MODEL}](https://huggingface.co/{BASE_MODEL})), no LoRA.", "",
           "## Why",
           f"{CP_TEST_ID} found booru tag pairs that move their own attribute in every picture. A slider needs more than two "
           "ends: the amount has to set how far the picture goes. This experiment builds the slider from the model's own "
           "predictions, before any text reader sets the amount: at every sampling step the prediction for the plain "
           "prompt moves toward the prediction for the tag's prompt by the amount (1 is the tag's own picture, 0 the plain "
           "prompt's, past 1 it goes further). Composing predictions this way is the idea of composable diffusion and "
           "semantic guidance; here it is tested as a dial on one attribute at a time.", "",
           "## Design",
           "- **The slider**: v(amount) = v(plain) + |amount| (v(tag of the amount's side) - v(plain)), on the conditional "
           "branch at every step; the negative prompt's branch and the guidance scale unchanged.",
           f"- **Cells**: the 16 of {CP_TEST_ID} (eight adult-coded characters x seeds {', '.join(map(str, ATTR_SEEDS))}).",
           "- **Pairs and amounts**:", "", "| pair | + tag | - tag | framing | amounts |", "|---|---|---|---|---|", rows, "",
           f"- **Kept and reused**: the amounts 0 and +-1 are {CP_TEST_ID}'s own pictures, downloaded from the data repo (at "
           "1 the slider is the tag's own prediction, so those pictures are not drawn again; the first "
           f"{CPS_PARITY_CELLS} cells of each pair are drawn once at +1 through the slider to show it). The new amounts are "
           f"{n_sets} sets x 16 = {16 * n_sets} pictures, each kept in the data repo as soon as it is drawn.", "",
           "## Recipe", "| setting | value |", "|---|---|", rec, "",
           "## The judge",
           f"The [WD EVA02-Large tagger v3](https://huggingface.co/{TAGGER}) on the pixels: a pair's score is the log-odds of "
           "its + tag minus those of its - tag (the + tag alone for a one-way pair). Every judged pair's score is read on "
           "every picture, so each slider's pull on the other attributes shows. Content kept: the CLIP ViT-L/14 cosine to the "
           "plain prompt's picture of the same cell. The tagger's ratings screen every picture as in the counterpoint pairs.", "",
           "## The rule (fixed before the run)",
           "**THE DIAL**: per cell, the least-squares slope of the pair's own score on the amount; A DIAL when the mean slope "
           "is positive, at least 75% of the cells slope that way and the mean is beyond 3 standard errors. **THE MIDDLE** "
           "(per side): at half the amount the score lies between the plain prompt's and the tag's in at least 75% of the "
           "cells: GRADED; else A SWITCH (the picture jumps between the ends). **PAST THE TAG** (per side, descriptive): 1.5 "
           "against 1 under the words rule. **CROSS-TALK**: the slider's mean slope on every other judged pair's score over "
           "that pair's span, beside its own; an overlap when above a third of its own. Verdict: **A SLIDER** (a dial, graded "
           "on every side, no overlap), **A SLIDER WITH CROSS-TALK**, **A SWITCH** (a dial, not graded), **NO DIAL**. The "
           "pairs that are sliders go on to the readers' stage (a text reader setting the amount).", ""]
    if meta.get("status") == "done":
        r = res["pairs"]
        out += ["## Result", meta.get("summary", ""), "",
                "| pair | mean score by amount | slope (dial) | middle (+ / -) | past the tag (+ / -) | worst cross-talk | verdict |",
                "|---|---|---|---|---|---|---|"]
        for k, v in r.items():
            means = ", ".join(f"{x} {m:+.1f}" for x, m in v["means"].items())
            mid = " / ".join(f"{s} {m['between']:.0%} ({m['OUTCOME'].lower()})" for s, m in v["middle"].items())
            past = " / ".join(f"{s} {b['mean']:+.2f} ({b['OUTCOME'].lower()})" for s, b in v["beyond"].items())
            others = [(abs(c) / abs(v["reach"]) if v["reach"] else 0.0, b) for b, c in v["cross_talk"].items() if b != k]
            worst = max(others) if others else (0.0, "")
            out.append(f"| {k} | {means} | {v['dial']['mean']:+.2f} +- {v['dial']['se']:.2f} ({v['dial']['OUTCOME']}) | "
                       f"{mid} | {past} | {worst[0]:.2f} of its own ({worst[1]}) | **{v['OUTCOME']}** |")
        par = res.get("parity") or {}
        if par:
            out += ["", "The slider at +1 against the tag's own pictures (first cells, drawn through the slider): "
                        + "; ".join(f"{k}: mean pixel difference {p['pixel_diff']:.2f} of 255, score difference "
                                    f"{p['score_diff']:+.2f}" for k, p in par.items()) + "."]
        held = res.get("withheld") or {}
        out += ["", ("The judge rated no picture questionable or explicit." if not held else
                     "Withheld (rated questionable or explicit; off the sheets, the set out of the data repo): "
                     + "; ".join(f"{n}: {', '.join(v)}" for n, v in held.items()) + ".")]
        pics = res.get("pictures") or {}
        if pics:
            out += ["", f"Pictures: {pics.get('total')} read, {pics.get('drawn')} drawn in this run, {pics.get('reused')} "
                        "reused from the data repo or this runtime."]
        out += ["", "Sheets (`sheet_<pair>.jpg`): the eight characters at the first seed; columns: the amounts in order.", ""]
        out += [f"![{k}](sheet_{k}.jpg)" for k in r] + [""]
    elif meta.get("status") == "failed":
        out += ["## Result", f"The run failed: `{meta.get('error', '')}`.", ""]
    else:
        out += ["## Result", "Running.", ""]
    out += ["## Files",
            "- `result.json`: every cell's scores at every amount on every judged pair, the reads, the parity read, where "
            "each set came from and where it is kept.",
            "- `tagger_probs.npz`: every drawn picture's probability for each of the judge's tags (float16, one array per set).",
            "- `sheet_*.jpg`: contact sheets.", "", "## References", refs, ""]
    return "\n".join(out)


# ---- e052: text readers set the counterpoint sliders' amount, on the second bed --------------------------------------------
CPG_TEST_ID = "e052_anima_counterpoint_readers"
CPG_TEST_TITLE = ("The stock model: text readers set the counterpoint sliders' amount from wordings they never trained on "
                  "(Qwen3, Beatrix, the nano bodies)")
CPG_DATE = "2026-10-18"
CPG_PAIRS = ("hair_colour", "expression", "eye_openness")       # e051's sliders
CPG_BED = 2
CPG_GRID = tuple(x / 4 for x in range(-6, 7))                  # -1.5 .. 1.5 in quarter steps: every amount lands on one
CPG_NEW = tuple(a for a in CPG_GRID if a not in CPS_AMOUNTS)   # the quarter steps e051 did not draw
CPG_READ_FORM = PREFIX + "1girl, solo, upper body, {w}."        # the frame a reader reads a wording in (its closing full stop)
CPG_READERS = ("qwen", "q0", "beatrix", "beatrix_untrained", "nano", "stack", "birth")
CPG_CONTROLS = ("q0", "beatrix_untrained", "birth")
CPG_SIDE_MIN = 3                                                # of the 4 unseen wordings of a side: the reader carries it
CPG_GATE = ("cp2_plain_upper-body", "cp2_plain_full-body", "cp2_skinny_full-body", "cps2_wide-eyed_upper-body_x1.5",
            "cps2_frown_upper-body_x1.5")                     # the guarded character's hardest pictures, judged first
CPG_GATE_SENSITIVE = 5            # of the gate's 10: at most this many rated sensitive (the first bed's characters: 2-73%)
CPG_GATE_FRAME_BAR = 0.5          # and none read head out of frame above this
_HAIR = " hair"
CPG_PHRASES = {
    "hair_colour": {
        "train+": tuple(w + _HAIR for w in ("blonde", "golden", "yellow", "light blonde", "platinum blonde", "honey blonde",
                                            "sandy blonde", "pale blonde", "bright blonde", "ash blonde", "golden blonde",
                                            "silky blonde")),
        "train-": tuple(w + _HAIR for w in ("black", "jet black", "raven", "dark", "ebony", "coal black", "inky black",
                                            "pitch black", "deep black", "glossy black", "blue-black", "midnight black")),
        "unseen+": tuple(w + _HAIR for w in ("flaxen", "straw-coloured", "wheat-coloured", "fair")),
        "unseen-": tuple(w + _HAIR for w in ("sable", "onyx", "obsidian", "soot-coloured")),
        "descriptive": ("dark brown hair", "brown hair", "light brown hair")},
    "expression": {
        "train+": ("smile", "smiling", "a smile", "a big smile", "a warm smile", "a bright smile", "a happy smile",
                   "a gentle smile", "grinning", "a grin", "beaming", "smiling happily"),
        "train-": ("frown", "frowning", "a frown", "a deep frown", "a stern frown", "an angry frown", "scowling", "a scowl",
                   "glaring", "a glare", "frowning angrily", "an irritated frown"),
        "unseen+": ("a joyful expression", "a delighted face", "a pleased look", "a radiant expression"),
        "unseen-": ("a sullen expression", "a displeased face", "a sour look", "a disapproving expression"),
        "descriptive": ("a neutral expression", "a calm face")},
    "eye_openness": {
        "train+": ("wide-eyed", "wide eyes", "wide open eyes", "eyes wide open", "big wide eyes", "eyes opened wide",
                   "staring wide-eyed", "round wide eyes", "fully open eyes", "large open eyes", "bright wide eyes",
                   "eyes wide"),
        "train-": ("half-closed eyes", "half-lidded eyes", "sleepy eyes", "drowsy eyes", "heavy-lidded eyes", "droopy eyes",
                   "eyes half shut", "half-open eyes", "tired eyes", "eyes half closed", "sleepy half-closed eyes",
                   "heavy eyelids"),
        "unseen+": ("saucer eyes", "goggling eyes", "eyes like saucers", "unblinking eyes"),
        "unseen-": ("bedroom eyes", "dozy eyes", "slumberous eyes", "languid eyes"),
        "descriptive": ("relaxed eyes", "calm eyes")},
}
CPG_GRADED = {"hair_colour": ("dark brown hair", "brown hair", "light brown hair")}   # expected in this order (descriptive)
CPG_SHARED_WORDS = frozenset({"a", "an", "hair", "eyes"})        # the words every wording of a pair may share
CPG_REFERENCES = CPS_REFERENCES + (
    ("Bolukbasi, Chang, Zou, Saligrama, Kalai, \"Man is to Computer Programmer as Woman is to Homemaker? Debiasing Word "
     "Embeddings\" (NeurIPS 2016; a direction in a text model's states from paired wordings)",
     "https://arxiv.org/abs/1607.06520"),)


def cpg_rows(keys=None) -> list[dict]:
    """Every wording the readers read, in a fixed order: per pair its training wordings (+ then -), the unseen ones (+ then
    -) and the descriptive ones; each {pair, split ('train' / 'unseen' / 'descriptive'), side (+1 / -1 / 0), text, frame}."""
    out = []
    for k in (CPG_PAIRS if keys is None else keys):
        P = CPG_PHRASES[k]
        for split, side, group in (("train", 1, "train+"), ("train", -1, "train-"), ("unseen", 1, "unseen+"),
                                   ("unseen", -1, "unseen-"), ("descriptive", 0, "descriptive")):
            out += [{"pair": k, "split": split, "side": side, "text": w, "frame": CPG_READ_FORM.format(w=w)}
                    for w in P[group]]
    return out


def cpg_words(text: str) -> set:
    import re
    return {w for w in re.split(r"[\s\-]+", text.lower()) if w and w not in CPG_SHARED_WORDS}


def cpg_leaks() -> dict:
    """{pair: [unseen wordings sharing a word with that pair's training wordings]}; fixed before the run: none may."""
    out = {}
    for k, P in CPG_PHRASES.items():
        train = set().union(*(cpg_words(w) for w in P["train+"] + P["train-"]))
        out[k] = [w for w in P["unseen+"] + P["unseen-"] if cpg_words(w) & train]
    return out


def cpg_grid(a: float) -> float:
    """A reader's amount on the picture grid: the nearest quarter step, clipped to the range the slider was measured on."""
    return float(min(max(round(float(a) * 4) / 4, CPG_GRID[0]), CPG_GRID[-1]))


def cpg_set_of(key: str, amount: float) -> str:
    """The second bed's set a pair's grid amount is (0 and +-1: the counterpoint sets themselves; the rest slider sets)."""
    return cps_cached(key, amount, CPG_BED) or cps_set_name(key, amount, CPG_BED)


def cpg_plans(keys=None) -> dict:
    """{set name: items} the run draws or reuses, in drawing order: the gate's sets first, then every set of the
    counterpoint pairs and sliders on the second bed (the dataset brought up to date), then the quarter steps."""
    keys = list(CPG_PAIRS if keys is None else keys)
    plans = {}
    for t, f in cp_sets(list(CP_DEFAULT)).values():
        plans[cp_set_name(t, f, CPG_BED)] = cp_items(t, f, CPG_BED)
    for k, a in cps_sets(list(CPS_PAIRS)).values():
        plans[cps_set_name(k, a, CPG_BED)] = cps_items(k, a, bed=CPG_BED)
    for k in keys:
        for a in CPG_NEW:
            plans[cps_set_name(k, a, CPG_BED)] = cps_items(k, a, bed=CPG_BED)
    first = [n for n in CPG_GATE if n in plans]
    return {n: plans[n] for n in first + [n for n in plans if n not in first]}


def cpg_first_bed(name: str) -> "str | None":
    """The first bed's set a second-bed set reuses its unchanged pictures from (None for a quarter step: never drawn)."""
    for p2, p1 in (("cps2_", "cps_"), ("cp2_", "cp_")):
        if name.startswith(p2):
            first = p1 + name[len(p2):]
            return None if first.startswith("cps_") and not any(first.endswith(f"_x{abs(a):g}") for a in (0.5, 1.5)) \
                else first
    return None


def cpg_gate(ages: list, ratings: list, frame: list) -> dict:
    """The new character's gate on its hardest pictures (judged before any other is drawn): every age read at most AGE_BAR,
    every rating general or sensitive, at most CPG_GATE_SENSITIVE rated sensitive and no head-out-of-frame read above
    CPG_GATE_FRAME_BAR."""
    return {"pictures": len(ages), "max_age": float(max(ages)), "ages": [float(a) for a in ages],
            "ratings": {r: ratings.count(r) for r in CP_RATINGS}, "head_out_of_frame": [float(x) for x in frame],
            "passed": bool(max(ages) <= AGE_BAR and all(r in ("general", "sensitive") for r in ratings)
                           and ratings.count("sensitive") <= CPG_GATE_SENSITIVE and max(frame) <= CPG_GATE_FRAME_BAR)}


def cpg_reads(scores: dict, readers: dict, keys=None) -> dict:
    """The picture marks fixed before the run. scores = {pair: {grid amount: [the pair's own score per cell]}}; readers = {pair: {reader:
    {wording: its amount}}}. Per reader and pair: each unseen wording's pictures (the grid amount nearest its reading)
    against the plain prompt's under the words rule in the wording's direction; a side carried at CPG_SIDE_MIN of its 4;
    the pair when both sides are. The controls must carry no pair. The graded order and the descriptive amounts beside."""
    import numpy as np
    out = {}
    for k in (CPG_PAIRS if keys is None else keys):
        P = CPG_PHRASES[k]
        plain = np.asarray(scores[k][0.0], dtype=float)
        per = {}
        for r, amounts in readers[k].items():
            words = {}
            for side, group in ((1, "unseen+"), (-1, "unseen-")):
                for w in P[group]:
                    g = cpg_grid(amounts[w])
                    res = flavor_outcome(list(np.asarray(scores[k][g], dtype=float) - plain), side, label="CARRIES IT")
                    words[w] = {"side": side, "a": float(amounts[w]), "grid": g, **res}
            sides = {s: sum(1 for v in words.values() if v["side"] == sv and v["OUTCOME"] == "CARRIES IT")
                     for s, sv in (("+", 1), ("-", -1))}
            carried = {s: n >= CPG_SIDE_MIN for s, n in sides.items()}
            graded = None
            if k in CPG_GRADED:
                seq = [float(amounts[w]) for w in CPG_GRADED[k]]
                graded = all(x < y for x, y in zip(seq, seq[1:]))
            per[r] = {"words": words, "sides": sides, "carried": carried, "graded": graded,
                      "descriptive": {w: float(amounts[w]) for w in P["descriptive"]},
                      "OUTCOME": ("CARRIES THE PAIR" if all(carried.values()) else
                                  "CARRIES ONE SIDE" if any(carried.values()) else "DOES NOT CARRY IT")}
        out[k] = {"readers": per, "controls_carrying": [r for r in CPG_CONTROLS if r in per
                                                        and per[r]["OUTCOME"] == "CARRIES THE PAIR"]}
    return out


CPG_NAMES = {"qwen": "Qwen3 0.6B, final state (Anima's own text encoder: the positive control)",
             "q0": "Qwen3's token layer (the lookup: a control)",
             "beatrix": "Beatrix 3s, block 20",
             "beatrix_untrained": "Beatrix's architecture untrained (a control that must fail)",
             "nano": "the best nano body tap (picked on the training wordings)",
             "stack": "the four nano bodies side by side (picked on the training wordings)",
             "birth": "the picked nano body at birth (a control that must fail)"}


def render_readers_readme(meta: dict, recipe: dict) -> str:
    """e052's README: why, the design and the rule fixed before the run, and the result when done."""
    keys = meta.get("pairs") or list(CPG_PAIRS)
    res = meta.get("result") or {}
    rec = "\n".join(f"| {k} | {v} |" for k, v in recipe.items())
    refs = "\n".join(f"- {t}: {u}" for t, u in CPG_REFERENCES)
    words = "\n".join(f"| {k} | {', '.join(CPG_PHRASES[k]['unseen+'])} | {', '.join(CPG_PHRASES[k]['unseen-'])} | "
                      f"{len(CPG_PHRASES[k]['train+'])} + {len(CPG_PHRASES[k]['train-'])} |" for k in keys)
    out = [f"# {CPG_TEST_ID}: {CPG_TEST_TITLE}", "",
           f"Date: {CPG_DATE}. Model: {ANIMA.model_name} ([{BASE_MODEL}](https://huggingface.co/{BASE_MODEL})), no LoRA.", "",
           "## Why",
           f"{CPS_TEST_ID} built a slider between each pair's booru tags from the model's own predictions: hair colour, "
           "expression and eye openness move smoothly with the amount. A text model can set that amount from a wording: if "
           "its states for 'golden hair' sit near those for 'blonde hair' and far from 'black hair', the amount it reads for a "
           "wording it never saw puts the picture on the right side. This experiment asks which text models can: Anima's own "
           "encoder (Qwen3, which reads Danbooru-style prompts), a byte-level language model (Beatrix) and four small byte "
           "models (the nano bodies), each beside a control that must fail.", "",
           "## Design",
           f"- **The characters (updated)**: the eight characters of the earlier pairs with the fourth replaced ({BED2_FOURTH[0]} "
           f"/ {BED2_FOURTH[1]} in place of sundress / flower garden, which came out younger than the others at one seed); "
           f"every picture of the new character adds \"{AGE_GUARD}\" to the negative prompt. The new character was picked by a "
           "short screen of four candidates on the image tagger (none read as a child or rated questionable or explicit, the "
           "face and the hair in view, the fewest rated sensitive) after a first pick, an evening gown in a ballroom, came out "
           "framed on the chest (its pictures were never published). Its gate, judged before any other picture is drawn: its "
           f"10 hardest pictures all under {AGE_BAR} on the age tags, none rated questionable or explicit, at most "
           f"{CPG_GATE_SENSITIVE} rated sensitive, none read head out of frame above {CPG_GATE_FRAME_BAR}. The other seven "
           "characters' pictures are the earlier experiments' own, reused. Every picture is screened: rated questionable or "
           f"explicit, or read by the tagger as {', '.join(AGE_TAGS)} above {AGE_BAR}, it is withheld (off the sheets, its set "
           "out of the data repo).",
           f"- **The readers' wordings** (read in the frame `{CPG_READ_FORM.format(w='<wording>')}`, at its closing full "
           "stop): per pair 12 training wordings a side and 4 unseen a side (no unseen wording shares a word with that pair's "
           "training wordings), plus a few descriptive ones read but never trained:", "",
           "| pair | unseen + | unseen - | training (+ / -) |", "|---|---|---|---|", words, "",
           "- **The reading** (a difference-of-means axis, the same for every reader): a = (f - m) . u / (u . u / 2), with f "
           "the reader's state standardized per feature over the pair's 24 training wordings, u the + training centre minus "
           "the - one and m their midpoint, so the training centres sit at +1 and -1, the tags' own level on the slider. A "
           "training wording is read with itself left out.",
           "- **The readers**: " + "; ".join(f"{v}" for v in CPG_NAMES.values()) + ".",
           f"- **The pictures**: the slider of {CPS_TEST_ID} at the reader's amount, rounded to a quarter step and clipped to "
           "[-1.5, 1.5]. Every quarter step from -1.5 to 1.5 is drawn once on the 16 cells, so every reader's amount is a "
           "picture already drawn.", "",
           "## Recipe", "| setting | value |", "|---|---|", rec, "",
           "## The rule (fixed before the run)",
           "**The reading** (no pictures): the training wordings on their side when each is left out (of 24), the unseen "
           "wordings on their side (of 8), the separation (the mean amount of the + unseen wordings minus the - ones). **The "
           "pictures**: per unseen wording, its pictures against the plain prompt's on the 16 cells, read in the wording's "
           "direction: CARRIES IT when the mean moves that way, at least 12 of the 16 cells move that way and the mean is "
           f"beyond 3 standard errors. A reader carries a side when at least {CPG_SIDE_MIN} of its 4 unseen wordings carry it, "
           "and the pair when it carries both sides. The controls must carry no pair; a pair a control carries is credited "
           "to no reader's training. Descriptive: the hair-colour browns in order (dark brown below brown below light brown).",
           ""]
    if meta.get("status") == "done":
        r = res.get("readers") or {}
        R = res.get("reading") or {}
        out += ["## Result", meta.get("summary", ""), ""]
        for k in keys:
            if k not in r:
                continue
            out += [f"### {k.replace('_', ' ')}", "",
                    "| reader | training wordings on their side (left out) | unseen on their side | separation | + side "
                    "carried | - side carried | outcome |", "|---|---|---|---|---|---|---|"]
            for rd, v in r[k]["readers"].items():
                q = (R.get(k) or {}).get(rd) or {}
                out.append(f"| {CPG_NAMES.get(rd, rd)} | {q.get('loo_train', '')} of 24 | {q.get('unseen_sides', '')} of 8 | "
                           f"{q.get('unseen_sep', float('nan')):+.2f} | {v['sides']['+']} of 4 | {v['sides']['-']} of 4 | "
                           f"**{v['OUTCOME']}** |")
            ctl = r[k]["controls_carrying"]
            out += ["", ("No control carries this pair." if not ctl else
                         f"A control carries this pair ({', '.join(ctl)}): no reader's result on it is credited to training."),
                    ""]
        sl = res.get("slider") or {}
        if sl:
            out += ["The slider on the updated characters at quarter steps (the earlier experiment's own marks): "
                    + "; ".join(f"{k}: slope {v['dial']['mean']:+.2f} per unit, {v['OUTCOME']}" for k, v in sl.items()) + ".", ""]
        gate = res.get("gate") or {}
        if gate:
            rt = gate.get("ratings") or {}
            fr = gate.get("head_out_of_frame") or [0.0]
            out += [f"The new character's gate (its hardest pictures, judged before the rest): the highest age read "
                    f"{gate.get('max_age', 0):.2f} against {AGE_BAR}; rated sensitive {rt.get('sensitive', 0)} of "
                    f"{gate.get('pictures', 0)} (at most {CPG_GATE_SENSITIVE}); the highest head-out-of-frame read "
                    f"{max(fr):.2f} against {CPG_GATE_FRAME_BAR}; {'passed' if gate.get('passed') else 'FAILED'}.", ""]
        held = res.get("withheld") or {}
        out += [("No picture was withheld." if not held else
                 "Withheld (off the sheets, the set out of the data repo): "
                 + "; ".join(f"{n}: {', '.join(v)}" for n, v in held.items()) + "."), ""]
        pics = res.get("pictures") or {}
        if pics:
            out += [f"Pictures: {pics.get('total')} read, {pics.get('drawn')} drawn in this run, {pics.get('reused')} reused.", ""]
        out += ["Sheets: `sheet_grid_<pair>.jpg` (the slider at every quarter step from -1.5 to 1.5; the eight characters "
                "at the first seed, the new one fourth); `sheet_readers_<pair>.jpg` (rows: the readers in the order of the "
                "table; columns: the 8 unseen wordings, + then -; the first character at the first seed).", ""]
        out += [f"![{k}](sheet_readers_{k}.jpg)" for k in keys if k in r] + [""]
    elif meta.get("status") == "failed":
        out += ["## Result", f"The run failed: `{meta.get('error', '')}`.", ""]
    else:
        out += ["## Result", "Running.", ""]
    out += ["## Files",
            "- `readers.json`: every reader's amount for every wording, the taps picked and the reading marks.",
            "- `result.json`: every cell's score at every grid amount, the marks, the gate, where each set came from.",
            "- `tagger_probs.npz`: every drawn picture's probability for each of the judge's tags (float16, one array per set).",
            "- `sheet_*.jpg`: contact sheets.", "", "## References", refs, ""]
    return "\n".join(out)


# ---- e053: grouped conditionings (three attribute conditionings in one picture, by words and by composed sliders) -------------
CGR_TEST_ID = "e053_anima_grouped_conditionings"
CGR_TEST_TITLE = ("The stock model: three attribute conditionings in one picture (hair colour, expression, eye colour), by "
                  "words and by composed sliders")
CGR_DATE = "2026-10-18"
CGR_PAIRS = ("hair_colour", "expression", "eye_colour")         # e050's handles, eye colour in place of eye openness
CGR_BED = 2                                                     # the updated characters
CGR_FRAMING = "upper body"
CGR_ARMS = ("words", "sliders")
CGR_CORNERS = tuple((a, b, c) for a in (1, -1) for b in (1, -1) for c in (1, -1))   # every side of every pair
CGR_KEPT, CGR_LOST = 0.75, 0.5          # a side keeps its single-tag move at >= .75 of it; under .5 it is lost
CGR_REFERENCES = CPS_REFERENCES


def cgr_tags(corner) -> list[str]:
    """A corner's tag for each pair of the group, in the group's order (+1: the pair's + tag, -1: its - tag)."""
    return [CP_PAIRS[k][0] if s > 0 else CP_PAIRS[k][1] for k, s in zip(CGR_PAIRS, corner)]


def cgr_label(corner) -> str:
    """'blonde hair, smile, red eyes'."""
    return ", ".join(cgr_tags(corner))


def cgr_set_name(corner, arm: str, bed: int = CGR_BED) -> str:
    """A group set's name: its tags and its arm (cgr2_blonde-hair_smile_red-eyes_words)."""
    return f"cgr{'' if bed == 1 else bed}_" + "_".join(cp_slug(t) for t in cgr_tags(corner)) + f"_{arm}"


def cgr_items(corner, arm: str, bed: int = CGR_BED) -> list[dict]:
    """One group set in the data repo's set format, captioned with the plain prompt. words: the corner's tags in the
    prompt's tag slot; sliders: the plain prompt composed with one prompt per tag at amount 1."""
    tags = cgr_tags(corner)
    out = []
    for ci, sd in attr_cells():
        plain = cp_prompt(None, CGR_FRAMING, ci, bed)
        it = {"name": f"{ci:02d}_{sd}", "seed": sd, "caption": plain}
        if arm == "words":
            it["prompt"] = cp_prompt(", ".join(tags), CGR_FRAMING, ci, bed)
        elif arm == "sliders":
            it.update(prompt=plain, mix_prompts=[cp_prompt(t, CGR_FRAMING, ci, bed) for t in tags],
                      mix_weights=[1.0] * len(tags))
        else:
            raise ValueError(f"unknown arm {arm!r}; the arms are {CGR_ARMS}")
        out.append(cp_guard(it, ci, bed))
    return out


def cgr_plans(bed: int = CGR_BED) -> dict:
    """{set name: items} of the group sets, in drawing order (per corner: words, then sliders)."""
    return {cgr_set_name(c, a, bed): cgr_items(c, a, bed) for c in CGR_CORNERS for a in CGR_ARMS}


def cgr_refs(bed: int = CGR_BED) -> dict:
    """The sets the group is read against, reused from the counterpoint pairs on the same characters: {'plain': the
    plain prompt's set, '<pair>+' / '<pair>-': the set of that side's tag alone}."""
    refs = {"plain": cp_set_name(None, CGR_FRAMING, bed)}
    for k in CGR_PAIRS:
        refs[f"{k}+"] = cp_set_name(CP_PAIRS[k][0], CGR_FRAMING, bed)
        refs[f"{k}-"] = cp_set_name(CP_PAIRS[k][1], CGR_FRAMING, bed)
    return refs


def cgr_ref_plans(bed: int = CGR_BED) -> dict:
    """{set name: items} of the reference sets."""
    refs = cgr_refs(bed)
    plans = {refs["plain"]: cp_items(None, CGR_FRAMING, bed)}
    for k in CGR_PAIRS:
        plans[refs[f"{k}+"]] = cp_items(CP_PAIRS[k][0], CGR_FRAMING, bed)
        plans[refs[f"{k}-"]] = cp_items(CP_PAIRS[k][1], CGR_FRAMING, bed)
    return plans


def cgr_reads(scores: dict, refs: dict) -> dict:
    """The marks fixed before the run. scores = {arm: {corner label: {pair: [score per cell]}}}; refs = {'plain' /
    '<pair>+' / '<pair>-': {pair: [score per cell]}} (the reference sets). Per arm: SURVIVAL (per corner and pair, the
    group picture against the plain one under the words rule in the pair's corner direction; a pair survives when it
    carries at all 8 corners), KEPT (per pair and side, the group's mean move over the single tag's, averaged over the 4
    corners on that side), OVERLAP (per pair k and other pair j, k's score with j at + minus with j at -, averaged over
    the 4 settings of k and the third pair, over k's span; beyond CP_OVERLAP) and the verdict."""
    import numpy as np
    plain = {k: np.asarray(refs["plain"][k], dtype=float) for k in CGR_PAIRS}
    span = {k: float(np.mean(np.asarray(refs[f"{k}+"][k], dtype=float) - np.asarray(refs[f"{k}-"][k], dtype=float)))
            for k in CGR_PAIRS}
    solo = {f"{k}{m}": float(np.mean(np.asarray(refs[f"{k}{m}"][k], dtype=float) - plain[k]))
            for k in CGR_PAIRS for m in "+-"}
    out = {"span": span, "single_tag_move": solo, "arms": {}}
    for arm, by_corner in scores.items():
        moves, survival = {}, {}
        for c in CGR_CORNERS:
            lab = cgr_label(c)
            survival[lab] = {}
            for i, k in enumerate(CGR_PAIRS):
                d = np.asarray(by_corner[lab][k], dtype=float) - plain[k]
                moves[(c, k)] = float(d.mean())
                survival[lab][k] = flavor_outcome(list(d), c[i], label="CARRIES IT")
        survives = {k: sum(survival[cgr_label(c)][k]["OUTCOME"] == "CARRIES IT" for c in CGR_CORNERS) for k in CGR_PAIRS}
        kept = {}
        for i, k in enumerate(CGR_PAIRS):
            for s, m in ((1, "+"), (-1, "-")):
                ref = solo[f"{k}{m}"]
                per = {cgr_label(c): (moves[(c, k)] / ref if ref else 0.0) for c in CGR_CORNERS if c[i] == s}
                share = float(np.mean(list(per.values())))
                kept[f"{k}{m}"] = {"share": share, "per_corner": per,
                                   "OUTCOME": "KEPT" if share >= CGR_KEPT else "LOST" if share < CGR_LOST else "PARTLY"}
        overlap = {}
        for i, k in enumerate(CGR_PAIRS):
            for j, o in enumerate(CGR_PAIRS):
                if j == i:
                    continue
                d = [moves[(c, k)] - moves[(tuple(-x if n == j else x for n, x in enumerate(c)), k)]
                     for c in CGR_CORNERS if c[j] == 1]
                val = float(np.mean(d)) / span[k] if span[k] else 0.0
                overlap[f"{o} on {k}"] = {"of_span": val, "OVERLAP": abs(val) > CP_OVERLAP}
        over = [n for n, v in overlap.items() if v["OVERLAP"]]
        whole = all(n == len(CGR_CORNERS) for n in survives.values())
        verdict = ("COMPOSES WITH OVERLAP" if whole and over else "COMPOSES" if whole else
                   "PARTLY COMPOSES" if any(n == len(CGR_CORNERS) for n in survives.values()) else "DOES NOT COMPOSE")
        out["arms"][arm] = {"survival": survival, "corners_carried": survives, "kept": kept, "overlap": overlap,
                            "overlapping": over, "OUTCOME": verdict}
    return out


def render_groups_readme(meta: dict, recipe: dict) -> str:
    """e053's README: why, the design and the rule fixed before the run, and the result when done."""
    res = meta.get("result") or {}
    rec = "\n".join(f"| {k} | {v} |" for k, v in recipe.items())
    refs = "\n".join(f"- {t}: {u}" for t, u in CGR_REFERENCES)
    pairs = "\n".join(f"| {k.replace('_', ' ')} | {CP_PAIRS[k][0]} | {CP_PAIRS[k][1]} |" for k in CGR_PAIRS)
    out = [f"# {CGR_TEST_ID}: {CGR_TEST_TITLE}", "",
           f"Date: {CGR_DATE}. Model: {ANIMA.model_name} ([{BASE_MODEL}](https://huggingface.co/{BASE_MODEL})), no LoRA.", "",
           "## Why",
           f"{CP_TEST_ID} found booru tag pairs that each move one attribute in every picture, and {CPS_TEST_ID} turned "
           "them into smooth sliders. A description usually asks for several attributes at once, so the next question is "
           "whether these conditionings work as a group: with three attributes asked for together, does each still move "
           "its own attribute, how much of its single-tag effect does it keep, and does one attribute pull on another? "
           "The group is asked for in two ways: the three tags written in one prompt, and the three single-tag predictions "
           "composed at every step.", "",
           "## Design",
           "- **The group** (the tags as the model's captions write them):", "",
           "| pair | + tag | - tag |", "|---|---|---|", pairs, "",
           f"- **The corners**: every side of every pair, {len(CGR_CORNERS)} groups (for example blonde hair, frown, blue "
           "eyes).",
           "- **Words**: the group's three tags in the prompt's tag slot, in the order above.",
           "- **Sliders**: v = v(plain) + sum over the group of (v(the tag's prompt) - v(plain)), on the conditional "
           f"branch at every sampling step, every tag at amount 1 (the single-tag case is the slider of {CPS_TEST_ID}); "
           "the negative prompt's branch and the guidance scale unchanged.",
           f"- **Cells**: the 16 of the updated characters (eight adult-coded characters x seeds "
           f"{', '.join(map(str, ATTR_SEEDS))}; the fourth, {BED2_FOURTH[0]} / {BED2_FOURTH[1]}, drawn with \"{AGE_GUARD}\" "
           "added to the negative prompt), upper body.",
           f"- **Reused**: the plain prompt's set and each tag's own set on the same characters, from {CP_TEST_ID}'s "
           f"updated sets. New: {len(CGR_CORNERS)} corners x {len(CGR_ARMS)} arms x 16 = "
           f"{len(CGR_CORNERS) * len(CGR_ARMS) * 16} pictures, each set kept in the data repo (`sets/cgr2_*`).", "",
           "## Recipe", "| setting | value |", "|---|---|", rec, "",
           "## The judge",
           f"The [WD EVA02-Large tagger v3](https://huggingface.co/{TAGGER}) on the pixels: a pair's score is the log-odds of "
           "its + tag minus those of its - tag, read on every picture for every pair. Every picture is screened: rated "
           f"questionable or explicit, or read as {', '.join(AGE_TAGS)} above {AGE_BAR}, it is withheld (off the sheets, its "
           "set out of the data repo).", "",
           "## The rule (fixed before the run)",
           "**SURVIVAL** (per arm, group and pair): the group's pictures against the plain prompt's on the 16 cells, in the "
           "direction of the pair's side in that group: it carries when the mean moves that way, at least 12 of the 16 "
           "cells move that way and the mean is beyond 3 standard errors. A pair **survives the group** in an arm when it "
           f"carries in all {len(CGR_CORNERS)} groups. **KEPT** (per arm, pair and side): the group's mean move over the "
           "single tag's mean move (both against the plain prompt), averaged over the 4 groups on that side: at least "
           f"{CGR_KEPT} KEPT, under {CGR_LOST} LOST, else PARTLY. **OVERLAP** (per arm, pair and other pair): the pair's "
           "score with the other pair at + minus with it at -, averaged over the 4 settings of the pair and the third "
           f"pair, in units of the pair's span (its + tag's set minus its - tag's): beyond {CP_OVERLAP:.2f} it is an "
           "overlap. Verdict per arm: **COMPOSES** (every pair survives, no overlap), **COMPOSES WITH OVERLAP**, **PARTLY "
           "COMPOSES** (some pair survives), **DOES NOT COMPOSE**. The words and the sliders are compared side by side "
           "(descriptive).", ""]
    if meta.get("status") == "done":
        r = res.get("reads") or {}
        out += ["## Result", meta.get("summary", ""), ""]
        arms = r.get("arms") or {}
        if arms:
            out += ["| | " + " | ".join(arms) + " |", "|---|" + "---|" * len(arms)]
            out.append("| verdict | " + " | ".join(f"**{v['OUTCOME']}**" for v in arms.values()) + " |")
            for k in CGR_PAIRS:
                out.append(f"| {k.replace('_', ' ')}: groups carried | "
                           + " | ".join(f"{v['corners_carried'][k]} of {len(CGR_CORNERS)}" for v in arms.values()) + " |")
            for k in CGR_PAIRS:
                for m in "+-":
                    out.append(f"| {k.replace('_', ' ')} {m}: share of the single tag kept | "
                               + " | ".join(f"{v['kept'][k + m]['share']:.2f} ({v['kept'][k + m]['OUTCOME'].lower()})"
                                            for v in arms.values()) + " |")
            names = list(next(iter(arms.values()))["overlap"])
            for n in names:
                out.append(f"| {n.replace('_', ' ')} (of its span) | "
                           + " | ".join(f"{v['overlap'][n]['of_span']:+.2f}" + (" **overlap**" if v["overlap"][n]["OVERLAP"]
                                                                                 else "") for v in arms.values()) + " |")
            out.append("")
        held = res.get("withheld") or {}
        out += [("No picture was withheld." if not held else
                 "Withheld (off the sheets, the set out of the data repo): "
                 + "; ".join(f"{n}: {', '.join(v)}" for n, v in held.items()) + "."), ""]
        pics = res.get("pictures") or {}
        if pics:
            out += [f"Pictures: {pics.get('total')} read, {pics.get('drawn')} drawn in this run, {pics.get('reused')} "
                    "reused.", ""]
        out += ["Sheets (`sheet_<arm>.jpg`): the eight characters at the first seed; columns: the plain prompt, then the "
                "groups in the order of the corners.", ""]
        out += [f"![{a}](sheet_{a}.jpg)" for a in arms] + [""]
    elif meta.get("status") == "failed":
        out += ["## Result", f"The run failed: `{meta.get('error', '')}`.", ""]
    else:
        out += ["## Result", "Running.", ""]
    out += ["## Files",
            "- `result.json`: every cell's score on every judged pair in every set, the marks, where each set came from.",
            "- `tagger_probs.npz`: every group picture's probability for each of the judge's tags (float16, one array per "
            "set).",
            "- `sheet_*.jpg`: contact sheets.", "", "## References", refs, ""]
    return "\n".join(out)


# ---- e054: readers on the group (one description read, the group's three sliders set at once) --------------------------
CGD_TEST_ID = "e054_anima_group_readers"
CGD_TEST_TITLE = ("The stock model: text readers read one description and set all three sliders of the group at once "
                  "(Beatrix, the nano bodies; Qwen3 as the reference)")
CGD_DATE = "2026-10-18"
CGD_PAIRS = CGR_PAIRS                       # hair colour, expression, eye colour: e053's group
CGD_BED = 2                                 # the updated characters
CGD_FRAMING = "upper body"
CGD_READ_FORM = CPG_READ_FORM               # a description is read in e052's frame, at its closing full stop
CGD_PLAIN_FRAME = PREFIX + "1girl, solo, upper body."       # the frame with no wording (descriptive)
CGD_SPLITS = {"train": 12, "trained": 4, "unseen": 4, "near": 4, "off": 4}   # per corner (near: e057's; off: e058's, per sheet column)
CGD_TIERS = ("trained", "unseen")           # trained: new combinations of training wordings; unseen: wordings never trained
CGD_ORDERS = ((0, 1, 2), (1, 2, 0), (2, 0, 1), (0, 2, 1), (2, 1, 0), (1, 0, 2))   # the order the three wordings are written in
CGD_SHARE_CAP = 2                           # of a side's 8 training wordings, at most this many share any one word
CGD_SEED = 54
CGD_READERS = ("qwen", "beatrix", "nano", "stack", "beatrix_untrained", "birth")   # the readers that get pictures
CGD_CONTROLS = ("beatrix_untrained", "birth")
CGD_FAR, CGD_PART = 0.75, 0.5               # how far a side moves, as a share of the group's move at amount 1
CGD_PHRASES = {                             # 8 training wordings a side, spread over many words; 4 unseen a side
    "hair_colour": {
        "train+": tuple(w + _HAIR for w in ("blonde", "light blonde", "golden", "bright golden", "yellow", "pale yellow",
                                            "platinum", "champagne")),
        "train-": tuple(w + _HAIR for w in ("black", "jet black", "dark", "very dark", "raven", "ebony", "inky",
                                            "charcoal")),
        "unseen+": CPG_PHRASES["hair_colour"]["unseen+"],      # e052's unseen wordings, read again under this training
        "unseen-": CPG_PHRASES["hair_colour"]["unseen-"]},
    "expression": {
        "train+": ("a smile", "a warm smile", "smiling", "smiling sweetly", "grinning", "a cheerful grin", "beaming",
                   "laughing"),
        "train-": ("a frown", "a deep frown", "frowning", "frowning angrily", "scowling", "a fierce scowl", "glaring",
                   "glowering"),
        "unseen+": CPG_PHRASES["expression"]["unseen+"],
        "unseen-": CPG_PHRASES["expression"]["unseen-"]},
    "eye_colour": {
        "train+": ("red eyes", "blood-red eyes", "crimson eyes", "scarlet eyes", "ruby eyes", "garnet eyes",
                   "vermilion eyes", "burgundy eyes"),
        "train-": ("blue eyes", "sky-blue eyes", "azure eyes", "sapphire eyes", "navy eyes", "cerulean eyes",
                   "cobalt eyes", "lapis eyes"),
        "unseen+": ("carmine eyes", "cinnabar eyes", "claret eyes", "cherry eyes"),
        "unseen-": ("cornflower eyes", "ultramarine eyes", "indigo eyes", "denim eyes")},
}
CGD_REFERENCES = CPG_REFERENCES


def cgd_checks() -> dict:
    """The wording rules fixed before any read: 'leaks' = unseen wordings sharing a word with any training wording of any
    pair (none may); 'crowded' = (pair, side, word, count) for a word in more than CGD_SHARE_CAP of a side's training
    wordings (none may), so no single word gives a side away."""
    train_words: set = set()
    for P in CGD_PHRASES.values():
        for w in P["train+"] + P["train-"]:
            train_words |= cpg_words(w)
    leaks = [w for P in CGD_PHRASES.values() for w in P["unseen+"] + P["unseen-"] if cpg_words(w) & train_words]
    crowded = []
    for k, P in CGD_PHRASES.items():
        for side in ("train+", "train-"):
            counts: dict = {}
            for w in P[side]:
                for x in cpg_words(w):
                    counts[x] = counts.get(x, 0) + 1
            crowded += [(k, side, x, n) for x, n in sorted(counts.items()) if n > CGD_SHARE_CAP]
    return {"leaks": leaks, "crowded": crowded}


def _cgd_cycle(words, n: int, seed: int) -> list:
    """n wordings as whole shuffled passes over `words` (each wording n / len(words) times when that divides)."""
    import random
    out: list = []
    p = 0
    while len(out) < n:
        w = list(words)
        random.Random(seed * 1000 + p).shuffle(w)
        out += w
        p += 1
    return out[:n]


def _cgd_split(split: str, seed: int) -> list[dict]:
    n = CGD_SPLITS[split]
    lists = split if split in ("unseen", "near") else "train"
    phrases = CGN_PHRASES if split == "near" else CGD_PHRASES
    pools = {}
    for i, k in enumerate(CGD_PAIRS):
        for s, m in ((1, "+"), (-1, "-")):
            pools[(k, s)] = _cgd_cycle(phrases[k][lists + m], n * len(CGR_CORNERS) // 2, seed + 10 * i + (s > 0))
    rows = []
    for ci, c in enumerate(CGR_CORNERS):
        for j in range(n):
            words = [pools[(k, s)].pop(0) for k, s in zip(CGD_PAIRS, c)]
            order = CGD_ORDERS[(ci * n + j) % len(CGD_ORDERS)]
            text = ", ".join(words[i] for i in order)
            cell = None if split == "train" else 2 * ((ci + 2 * j) % len(CHARACTERS_BED2)) + j % len(ATTR_SEEDS)
            rows.append({"id": f"{split}{len(rows):02d}", "split": split, "corner": list(c), "words": words,
                         "order": list(order), "text": text, "frame": CGD_READ_FORM.format(w=text), "cell": cell})
    return rows


def cgd_rows() -> list[dict]:
    """Every description the readers read, in a fixed order: the training descriptions (12 a corner, every training wording
    6 times, every writing order twice a corner), the trained tier (4 a corner: new combinations of the training
    wordings), the unseen tier (4 a corner: the unseen wordings), then the descriptive rows (the frame with no wording; each
    training side's first wording alone). Each row: {id, split, corner (+1 / -1 per pair), words (one per pair, in the
    group's order), order (the order they are written in), text, frame, cell (a tier row's picture cell, an index into
    attr_cells())}. The shuffle is the first from CGD_SEED at which no training description repeats, no tier description
    repeats and no trained-tier combination is a training one."""
    for attempt in range(1000):
        seed = CGD_SEED + 1000 * attempt
        train, trained, unseen = (_cgd_split(s, seed + 100 * i) for i, s in enumerate(("train", "trained", "unseen")))
        tr, tt, tu = ([tuple(r["words"]) for r in x] for x in (train, trained, unseen))
        if len(set(tr)) == len(tr) and len(set(tt)) == len(tt) and len(set(tu)) == len(tu) and not set(tt) & set(tr):
            break
    else:  # pragma: no cover - the search succeeds within the first few seeds
        raise RuntimeError("no shuffle met the rules")
    desc = [{"id": "desc00", "split": "descriptive", "corner": None, "words": None, "order": None, "text": "",
             "frame": CGD_PLAIN_FRAME, "cell": None}]
    for k in CGD_PAIRS:
        for s, m in ((1, "+"), (-1, "-")):
            w = CGD_PHRASES[k]["train" + m][0]
            desc.append({"id": f"desc{len(desc):02d}", "split": "descriptive", "corner": None, "words": None,
                         "order": None, "single": [k, s], "text": w, "frame": CGD_READ_FORM.format(w=w), "cell": None})
    return train + trained + unseen + desc


def cgd_tier_rows(tier: str) -> list[dict]:
    """The descriptions of one tier: e054's registered rows for its tiers, e057's near rows (cgn_rows) for 'near', e058's
    off-target rows (cgo_rows) for 'off'."""
    if tier == "off":
        return cgo_rows()
    return cgn_rows() if tier == "near" else [r for r in cgd_rows() if r["split"] == tier]


def cgd_amounts(a) -> list[float]:
    """A reader's three readings on the picture grid (e052's: quarter steps, clipped to the range the slider was measured
    on)."""
    return [cpg_grid(x) for x in a]


def cgd_set_name(reader: str, tier: str, bed: int = CGD_BED, prefix: str = "cgd") -> str:
    """'cgd2_beatrix_unseen' (a later test of the same design names its sets with its own prefix)."""
    return f"{prefix}{'' if bed == 1 else bed}_{cp_slug(reader)}_{tier}"


def cgd_items(tier: str, amounts: dict, bed: int = CGD_BED) -> list[dict]:
    """One reader's pictures for one tier in the data repo's set format: per description of the tier, its cell's plain
    prompt composed with one prompt per pair whose grid amount is not 0 (the tag of the amount's side at |amount|, in the
    group's order); a description whose three grid amounts are 0 is the plain prompt itself, and one at amount 1 on every
    pair is e053's slider group. Captioned with the plain prompt. amounts = {row id: [reading per pair]}."""
    cells = attr_cells()
    out = []
    for r in cgd_tier_rows(tier):
        ci, sd = cells[r["cell"]]
        plain = cp_prompt(None, CGD_FRAMING, ci, bed)
        it = {"name": f"{r['id']}_{ci:02d}_{sd}", "seed": sd, "caption": plain, "prompt": plain}
        mix = [(cp_prompt(CP_PAIRS[k][0] if a > 0 else CP_PAIRS[k][1], CGD_FRAMING, ci, bed), abs(a))
               for k, a in zip(CGD_PAIRS, cgd_amounts(amounts[r["id"]])) if a]
        if mix:
            it.update(mix_prompts=[m for m, _ in mix], mix_weights=[w for _, w in mix])
        out.append(cp_guard(it, ci, bed))
    return out


def cgd_plans(amounts: dict, readers=None, bed: int = CGD_BED, prefix: str = "cgd", tiers=CGD_TIERS) -> dict:
    """{set name: items} of the readers' sets in drawing order (per reader: its tiers in order; e054's are the trained tier,
    then the unseen tier). amounts = {reader: {row id: [reading per pair]}}."""
    return {cgd_set_name(r, t, bed, prefix): cgd_items(t, amounts[r], bed) for r in (readers or CGD_READERS)
            for t in tiers}


def cgd_refs(bed: int = CGD_BED) -> dict:
    """The sets the readers' pictures are read against: {'plain': the plain prompt's set, '<corner label>': e053's slider
    group at that corner (every pair at amount 1)}."""
    return {"plain": cp_set_name(None, CGD_FRAMING, bed), **{cgr_label(c): cgr_set_name(c, "sliders", bed)
                                                             for c in CGR_CORNERS}}


def cgd_ref_plans(bed: int = CGD_BED) -> dict:
    refs = cgd_refs(bed)
    return {refs["plain"]: cp_items(None, CGD_FRAMING, bed),
            **{cgr_set_name(c, "sliders", bed): cgr_items(c, "sliders", bed) for c in CGR_CORNERS}}


def cgd_read_marks(amounts: dict, tiers=CGD_TIERS) -> dict:
    """The reading marks (no pictures) of one reader. amounts = {row id: [reading per pair]}, a training row read left out.
    'train': per pair the training descriptions read on their side (of 96) and the total (of 288): the picks' statistic,
    blind to the tiers. Per tier and pair: per side the 16 readings under the words rule in the side's direction (READS
    IT), the separation (the + side's mean reading minus the - side's); the pair READ when both sides are, the group when
    all three are; the descriptions with every sign right (of 32); per other pair the reading cross-talk (the pair's mean
    reading with the other pair at + minus at -, over the pair's separation; beyond CP_OVERLAP an overlap)."""
    import numpy as np
    rows = cgd_rows()
    tr = [r for r in rows if r["split"] == "train"]
    out: dict = {"train": {k: int(sum(1 for r in tr if np.sign(amounts[r["id"]][i]) == r["corner"][i]))
                           for i, k in enumerate(CGD_PAIRS)}, "tiers": {}}
    out["train"]["total"] = sum(out["train"][k] for k in CGD_PAIRS)
    for tier in tiers:
        if tier == "off":                                              # e058: no sides; the right amount is 0
            out["tiers"][tier] = cgo_tier_marks(amounts, tier)
            continue
        trows = cgd_tier_rows(tier)
        A = np.array([amounts[r["id"]] for r in trows], dtype=float)
        C = np.array([r["corner"] for r in trows], dtype=float)
        pairs = {}
        for i, k in enumerate(CGD_PAIRS):
            sides = {m: flavor_outcome(list(A[C[:, i] == s, i]), s, label="READS IT") for s, m in ((1, "+"), (-1, "-"))}
            sep = float(A[C[:, i] == 1, i].mean() - A[C[:, i] == -1, i].mean())
            cross = {}
            for j, o in enumerate(CGD_PAIRS):
                if j != i:
                    v = float(A[C[:, j] == 1, i].mean() - A[C[:, j] == -1, i].mean()) / sep if sep else 0.0
                    cross[o] = {"of_separation": v, "OVERLAP": abs(v) > CP_OVERLAP}
            pairs[k] = {"sides": sides, "separation": sep, "cross_talk": cross,
                        "read": all(v["OUTCOME"] == "READS IT" for v in sides.values())}
        n = sum(v["read"] for v in pairs.values())
        whole = int(sum(all(np.sign(A[j, i]) == C[j, i] for i in range(len(CGD_PAIRS))) for j in range(len(trows))))
        out["tiers"][tier] = {"pairs": pairs, "whole_descriptions": whole, "of": len(trows),
                              "overlapping": [f"{o} on {k}" for k, v in pairs.items() for o, x in v["cross_talk"].items()
                                              if x["OVERLAP"]],
                              "OUTCOME": ("READS THE GROUP" if n == len(CGD_PAIRS) else
                                          f"READS {n} OF {len(CGD_PAIRS)}" if n else "READS NONE")}
    return out


def cgd_reads(scores: dict, plain: dict, corners: dict, amounts: dict, controls_of=CGD_CONTROLS, tiers=CGD_TIERS,
              control_of=None) -> dict:
    """The picture marks fixed before the run. scores = {reader: {tier: {pair: [score per picture, in the tier's
    description order]}}}; plain = {pair: [score per cell]} (the plain prompt's set); corners = {corner label: {pair:
    [score per cell]}} (e053's slider group at amount 1); amounts = {reader: {row id: [reading per pair]}}. Per reader,
    tier, pair and side: each description's picture against its cell's plain picture, the 16 of the side under the words
    rule in the side's direction (CARRIES IT); the pair is carried when both sides are; the reader DRIVES THE GROUP when
    all three are. Beside it: HOW FAR (the side's summed move over the summed move of the group at amount 1 on the same
    cells: at least CGD_FAR most of the way, under CGD_PART little, else part of the way), the pictures with every pair
    moved its way (of 32) and the grid amounts. The controls must drive no pair; a pair a control carries in a tier is
    credited to no reader there. The off-target tier ('off', e058) has its own rule (cgo_tier_reads; control_of =
    {reader: its control})."""
    import numpy as np
    out: dict = {}
    for tier in tiers:
        if tier == "off":
            out[tier] = cgo_tier_reads(scores, plain, corners, amounts, control_of, tier)
            continue
        trows = cgd_tier_rows(tier)
        per = {}
        for reader, by_tier in scores.items():
            S = by_tier[tier]
            pairs, moved = {}, [True] * len(trows)
            for i, k in enumerate(CGD_PAIRS):
                d = [float(S[k][j]) - float(plain[k][r["cell"]]) for j, r in enumerate(trows)]
                full = [float(corners[cgr_label(tuple(r["corner"]))][k][r["cell"]]) - float(plain[k][r["cell"]])
                        for r in trows]
                for j, r in enumerate(trows):
                    moved[j] = moved[j] and d[j] * r["corner"][i] > 0
                sides = {}
                for s, m in ((1, "+"), (-1, "-")):
                    idx = [j for j, r in enumerate(trows) if r["corner"][i] == s]
                    res = flavor_outcome([d[j] for j in idx], s, label="CARRIES IT")
                    den = sum(full[j] for j in idx)
                    far = sum(d[j] for j in idx) / den if den else 0.0
                    grid = [cgd_amounts(amounts[reader][trows[j]["id"]])[i] for j in idx]
                    sides[m] = {**res, "how_far": far, "mean_amount": float(np.mean(grid)), "amounts": grid,
                                "HOW_FAR": ("most of the way" if far >= CGD_FAR else "little" if far < CGD_PART
                                            else "part of the way")}
                pairs[k] = {"sides": sides, "carried": all(v["OUTCOME"] == "CARRIES IT" for v in sides.values())}
            n = sum(v["carried"] for v in pairs.values())
            per[reader] = {"pairs": pairs, "pairs_carried": [k for k in CGD_PAIRS if pairs[k]["carried"]],
                           "whole_pictures": int(sum(moved)), "of": len(trows),
                           "OUTCOME": ("DRIVES THE GROUP" if n == len(CGD_PAIRS) else
                                       f"DRIVES {n} OF {len(CGD_PAIRS)}" if n else "DRIVES NONE")}
        controls = {k: [r for r in controls_of if r in per and per[r]["pairs"][k]["carried"]] for k in CGD_PAIRS}
        for reader, v in per.items():
            v["credited"] = [] if reader in controls_of else [k for k in v["pairs_carried"] if not controls[k]]
        out[tier] = {"readers": per, "controls_carrying": controls}
    return out


CGD_NAMES = {"qwen": "Qwen3 0.6B, final state (Anima's own text encoder: the reference)",
             "beatrix": "Beatrix 3s, block 20",
             "nano": "the best nano tap (one body, one version, one block; picked on the training descriptions)",
             "stack": "the four nano bodies side by side (one version, one block; picked on the training descriptions)",
             "beatrix_untrained": "Beatrix's architecture untrained (a control that must fail)",
             "birth": "the picked nano body at birth, the same block (a control that must fail)"}


def render_group_readers_readme(meta: dict, recipe: dict) -> str:
    """e054's README: why, the design and the rule fixed before the run, and the result when done."""
    res = meta.get("result") or {}
    rec = "\n".join(f"| {k} | {v} |" for k, v in recipe.items())
    refs = "\n".join(f"- {t}: {u}" for t, u in CGD_REFERENCES)
    rows = cgd_rows()
    words = "\n".join(f"| {k.replace('_', ' ')} | {s} | {', '.join(CGD_PHRASES[k]['train' + s])} | "
                      f"{', '.join(CGD_PHRASES[k]['unseen' + s])} |" for k in CGD_PAIRS for s in "+-")
    ex = {t: next(r["text"] for r in rows if r["split"] == t) for t in ("train", "trained", "unseen")}
    readers = "\n".join(f"| {r} | {n} |" for r, n in CGD_NAMES.items())
    out = [f"# {CGD_TEST_ID}: {CGD_TEST_TITLE}", "",
           f"Date: {CGD_DATE}. Model: {ANIMA.model_name} ([{BASE_MODEL}](https://huggingface.co/{BASE_MODEL})), no LoRA.", "",
           "## Why",
           f"{CGR_TEST_ID} showed that three attribute sliders (hair colour, expression, eye colour) compose in one "
           f"picture without pulling on each other, and {CPG_TEST_ID} let text readers set one slider at a time from a "
           "single wording. A person describes a character in one go, so the question here is systemic: can a trained "
           "text model read one description and set every slider of the group from that single reading, so that the "
           "picture carries all three attributes as described? The readers are Beatrix (a byte-level language model) "
           "and the nano bodies (small byte-level language models), with Qwen3 (Anima's own text encoder) as the "
           "reference and untrained copies as the controls that must fail.", "",
           "## Design",
           "- **The group**: hair colour (blonde hair / black hair), expression (smile / frown), eye colour (red eyes / "
           f"blue eyes), on the sliders of {CPS_TEST_ID} composed as in {CGR_TEST_ID}.",
           "- **Descriptions**: one wording per pair, joined with commas in one of six orders, read in the frame "
           f"`{CGD_READ_FORM}` (the reader's state at the closing full stop). Every side of every pair (8 corners). "
           f"Training: {CGD_SPLITS['train']} a corner (each training wording 6 times), for example `{ex['train']}`. "
           f"The trained tier: {CGD_SPLITS['trained']} a corner, new combinations of the training wordings (`"
           f"{ex['trained']}`). The unseen tier: {CGD_SPLITS['unseen']} a corner, wordings that share no word with any "
           f"training wording (`{ex['unseen']}`; the hair and expression ones are {CPG_TEST_ID}'s).", "",
           "| pair | side | training wordings | unseen wordings |", "|---|---|---|---|", words, "",
           f"- No word appears in more than {CGD_SHARE_CAP} of a side's 8 training wordings, so no single word gives a side "
           f"away (the lesson of {CPG_TEST_ID}, whose training wordings mostly shared one word a side).",
           "- **The reading** (one state per description for all three sliders): per pair, the ridge regression of the "
           "side (+1 / -1) on the state, its features standardized over the 96 training descriptions (a regularized linear "
           "discriminant: it discounts what varies with the other pairs' wordings and the word order); the strength per "
           "reader chosen by how many training descriptions it reads on their side when each is left out; a training "
           "description is read with itself left out. The training sides sit near -1 and +1. The amount on the picture is "
           "the reading rounded to a quarter step and clipped to [-1.5, 1.5]. A first reading, the plain difference of the "
           "two sides' means (the earlier single-wording form), read its own training descriptions near chance in this "
           "design; its marks are kept in `readers.json`.",
           "- **The pictures**: one per tier description, on its own cell (the 16 cells of the updated characters, two "
           "descriptions of each tier a cell): v = v(plain) + sum over the pairs of |a| (v(the tag of a's side) - "
           "v(plain)) on the conditional branch at every step; the negative branch and the guidance scale unchanged.",
           "- **The readers**:", "", "| reader | what it is |", "|---|---|", readers, "",
           "## Recipe", "| setting | value |", "|---|---|", rec, "",
           "## The judge",
           f"The [WD EVA02-Large tagger v3](https://huggingface.co/{TAGGER}) on the pixels: a pair's score is the log-odds of "
           "its + tag minus those of its - tag. Every picture is screened: rated questionable or explicit, or read as "
           f"{', '.join(AGE_TAGS)} above {AGE_BAR}, it is withheld (off the sheets, its set out of the data repo).", "",
           "## The rule (fixed before the run)",
           "**CARRIES** (per reader, tier, pair and side): each description's picture against its cell's plain picture; "
           "the 16 descriptions of the side carry it when the mean moves that way, at least 12 of the 16 move that way and "
           "the mean is beyond 3 standard errors. A pair is carried when both sides are; a reader **DRIVES THE GROUP** "
           "in a tier when all three pairs are carried. The controls must drive no pair; a pair a control carries in a "
           "tier is credited to no reader there. Beside it: **HOW FAR** (a side's summed move over the move of the group "
           f"at amount 1 on the same cells: at least {CGD_FAR} most of the way, under {CGD_PART} little), the pictures "
           "with all three pairs moved their way, and the readings themselves (the same words rule on the 16 readings "
           "of a side, the separation, and the cross-talk: one pair's side moving another pair's reading by more than "
           f"{CP_OVERLAP:.2f} of that pair's separation).", ""]
    if meta.get("status") == "done":
        r = res.get("reads") or {}
        out += ["## Result", meta.get("summary", ""), ""]
        for tier in CGD_TIERS:
            per = (r.get(tier) or {}).get("readers") or {}
            if not per:
                continue
            out += [f"### The {tier} tier", "",
                    "| reader | verdict | " + " | ".join(f"{k.replace('_', ' ')} + / -" for k in CGD_PAIRS)
                    + " | all three moved | credited |",
                    "|---|---|" + "---|" * len(CGD_PAIRS) + "---|---|"]
            for name, v in per.items():
                cells = []
                for k in CGD_PAIRS:
                    sd = v["pairs"][k]["sides"]
                    cells.append(" / ".join(f"{sd[m]['OUTCOME'].lower()} ({sd[m]['how_far']:.2f})" for m in "+-"))
                out.append(f"| {name} | **{v['OUTCOME']}** | " + " | ".join(cells)
                           + f" | {v['whole_pictures']} of {v['of']} | {', '.join(v['credited']) or 'none'} |")
            out += ["", "Each cell: the side's verdict and, in brackets, how far it moved as a share of the group at "
                    "amount 1.", ""]
        held = res.get("withheld") or {}
        out += [("No picture was withheld." if not held else
                 "Withheld (off the sheets, the set out of the data repo): "
                 + "; ".join(f"{n}: {', '.join(v)}" for n, v in held.items()) + "."), ""]
        pics = res.get("pictures") or {}
        if pics:
            out += [f"Pictures: {pics.get('total')} read, {pics.get('drawn')} drawn in this run, {pics.get('reused')} "
                    "reused.", ""]
        out += ["Sheets (`sheet_<tier>.jpg`): one column per corner (the first description of each corner, on its own "
                "character at the first seed); rows: the plain prompt, the group at amount 1, then the readers in the "
                "order of the table above.", ""]
        out += [f"![{t}](sheet_{t}.jpg)" for t in CGD_TIERS] + [""]
    elif meta.get("status") == "failed":
        out += ["## Result", f"The run failed: `{meta.get('error', '')}`.", ""]
    else:
        out += ["## Result", "Running.", ""]
    out += ["## Files",
            "- `readers.json`: every description, every reader's readings and the reading marks (and the first reading's), "
            "which tap each reader is, the strength chosen, every nano version's best block.",
            "- `result.json`: every picture's score on every judged pair, the marks, where each set came from.",
            "- `tagger_probs.npz`: every drawn picture's probability for each of the judge's tags (float16, one array per "
            "set).",
            "- `sheet_*.jpg`: contact sheets.", "", "## References", refs, ""]
    return "\n".join(out)


# the group-readers tests registered so far (a test of the same design names its readers, controls, set prefix, README)
GROUP_READER_TESTS = {"e054": dict(id=CGD_TEST_ID, title=CGD_TEST_TITLE, date=CGD_DATE, readers=CGD_READERS,
                                    controls=CGD_CONTROLS, prefix="cgd", render=render_group_readers_readme)}


# ---- e055: the framed nano arms read each description in their own frame (one fetch per slider) ---------------------------
CGF_TEST_ID = "e055_anima_framed_group_readers"
CGF_TEST_TITLE = ("The stock model: the framed nano arms read each description in their own frame and set the three group sliders, "
                  "one fetch per slider")
CGF_DATE = "2026-10-18"
CGF_READERS = ("qwen", "framed", "framed_stack", "framed_base", "birth")   # qwen: e054's plain-description readings (reused)
CGF_CONTROLS = ("birth",)
CGF_NAMES = {"qwen": "Qwen3 0.6B, final state, reading the plain description (the reference; e054's readings and pictures)",
             "framed": "the best framed nano tap: one framed version x block x read position (picked on the training "
                       "descriptions)",
             "framed_stack": "the four nano bodies side by side at one framed version (picked the same way)",
             "framed_base": "the picked tap's body at the end of its diet, before any framed training, read in the same frame "
                            "(the frames' effect: a comparison)",
             "birth": "the picked tap's body untrained, read in the same frame (a control that must fail)"}


def render_framed_readers_readme(meta: dict, recipe: dict) -> str:
    """e055's README: e054's design and rule with the framed reading; the result when done."""
    res = meta.get("result") or {}
    rec = "\n".join(f"| {k} | {v} |" for k, v in recipe.items())
    readers = "\n".join(f"| {r} | {n} |" for r, n in CGF_NAMES.items())
    out = [f"# {CGF_TEST_ID}: {CGF_TEST_TITLE}", "",
           f"Date: {CGF_DATE}. Model: {ANIMA.model_name} ([{BASE_MODEL}](https://huggingface.co/{BASE_MODEL})), no LoRA.", "",
           "## Why",
           f"In {CGD_TEST_ID} text readers read one description of hair colour, expression and eye colour as a plain caption "
           "and set the three group sliders from that one state. The byte-level readers held the attributes while the "
           "wordings were ones they had trained on and scattered on new ones. The framed nano arms were trained to read "
           "records written in frames and to fetch a record's value when asked for its key. Here each description is "
           "written in that frame grammar and each slider is set from its own fetch, so the attributes no longer share one "
           "state.", "",
           "## Design",
           f"- **The descriptions, tiers, cells, pictures and judge**: {CGD_TEST_ID}'s, unchanged (the same 96 training "
           "descriptions, the trained tier and the unseen tier of 32, one picture per description).",
           "- **The frame** (the arms' own training layout, built by their trainer's frame builder): the caption register's "
           "head, then one sized 32-byte data frame per attribute in the order the description writes them, each holding one "
           "record (the key, then the wording without its noun: `hair` -> `golden`, `expression` -> `warm smile`, `eyes` -> "
           "`crimson`; spaces fill the frame), then the question for one attribute (its key) and its value, then the end "
           "byte. Three rows per description, one per attribute.",
           "- **Where it is read**: the reader's stream state (blocks 0-6, layer-normed) at the question's separator (the "
           "pointer: where the arm starts to copy) or at the end byte after the value (the content), whichever reads held-out "
           "wordings better (below); the copy check from the same pass (the value's bytes are the arm's greedy choice at "
           "every step).",
           "- **The reading**: per pair, on the rows asking that pair, the ridge regression of the side (+1 / -1) on the "
           "standardized state of the 96 training descriptions; a training description read with itself left out. Read that "
           "way nearly every tap reads all of its training sides (the state after the value holds the value's bytes), so the "
           "strength, the block and the position are chosen by **held-out words**: per pair, each side's i-th training "
           "wording held out together with every training description whose wording shares a word with either, the reading "
           "fit on the rest and read on the held-out descriptions (288 reads), ties by the left-out reading, then by name. "
           "The amount on the picture: the reading rounded to a quarter step, clipped to [-1.5, 1.5].",
           "- **The readers**:", "", "| reader | what it is |", "|---|---|", readers, "",
           "## Recipe", "| setting | value |", "|---|---|", rec, "",
           "## The rule (fixed before the run)",
           f"{CGD_TEST_ID}'s: per reader, tier, pair and side, the 16 descriptions' pictures each against their cell's plain "
           "picture; the side is carried when the mean moves its way, at least 12 of 16 move that way and the mean is beyond 3 "
           "standard errors; a reader **DRIVES THE GROUP** when all three pairs are carried on both sides. The control (the "
           "untrained body) must drive no pair; a pair it carries in a tier is credited to no reader there. Beside it: how "
           "far each side moves as a share of the group at amount 1, the copy rates, the frames' effect (the framed tap "
           "against its body before framed training) and the stability of each wording's amount across the descriptions "
           "it appears in.", ""]
    out += _framed_result_tail(meta, res, "every reader's readings and reading marks, which tap each reader is, every tap's "
                                          "held-out-words reading, the copy rates, the stability, every framed version's best "
                                          "tap.")
    return "\n".join(out)


def _framed_result_tail(meta: dict, res: dict, readers_file: str, tiers=CGD_TIERS) -> list:
    """The result, files and references sections of the READMEs of the readings in the framed grammar (e055-e057)."""
    out: list = []
    if meta.get("status") == "done":
        r = res.get("reads") or {}
        out += ["## Result", meta.get("summary", ""), ""]
        for tier in tiers:
            per = (r.get(tier) or {}).get("readers") or {}
            if not per:
                continue
            if tier == "off":
                out += _offtarget_table(per)
                continue
            out += [f"### The {tier} tier", "",
                    "| reader | verdict | " + " | ".join(f"{k.replace('_', ' ')} + / -" for k in CGD_PAIRS)
                    + " | all three moved | credited |",
                    "|---|---|" + "---|" * len(CGD_PAIRS) + "---|---|"]
            for name, v in per.items():
                cells = []
                for k in CGD_PAIRS:
                    sd = v["pairs"][k]["sides"]
                    cells.append(" / ".join(f"{sd[m]['OUTCOME'].lower()} ({sd[m]['how_far']:.2f})" for m in "+-"))
                out.append(f"| {name} | **{v['OUTCOME']}** | " + " | ".join(cells)
                           + f" | {v['whole_pictures']} of {v['of']} | {', '.join(v['credited']) or 'none'} |")
            out += ["", "Each cell: the side's verdict and, in brackets, how far it moved as a share of the group at "
                    "amount 1.", ""]
        held = res.get("withheld") or {}
        out += [("No picture was withheld." if not held else
                 "Withheld (off the sheets, the set out of the data repo): "
                 + "; ".join(f"{n}: {', '.join(v)}" for n, v in held.items()) + "."), ""]
        pics = res.get("pictures") or {}
        if pics:
            out += [f"Pictures: {pics.get('total')} read, {pics.get('drawn')} drawn in this run, {pics.get('copied')} copied "
                    f"from identical ones, {pics.get('reused')} reused.", ""]
        if all(t == "off" for t in tiers):
            out += ["Sheet (`sheet_off.jpg`): one column per fourth description (on its own character and seed); rows: the "
                    "plain prompt, then the readers in the order of the table above (a reader that stays plain shows the "
                    "plain picture again).", ""]
        else:
            out += ["Sheets (`sheet_<tier>.jpg`): one column per corner (the first description of each corner, on its own "
                    "character at the first seed); rows: the plain prompt, the group at amount 1, then the readers in the "
                    "order of the table above" + (" (the off-target sheet: one column per fourth description, no group "
                                                  "row)" if "off" in tiers else "") + ".", ""]
        out += [f"![{t}](sheet_{t}.jpg)" for t in tiers] + [""]
    elif meta.get("status") == "failed":
        out += ["## Result", f"The run failed: `{meta.get('error', '')}`.", ""]
    else:
        out += ["## Result", "Running.", ""]
    out += ["## Files",
            f"- `readers.json`: {readers_file}",
            "- `result.json`: every picture's score on every judged pair, the marks, where each set came from.",
            "- `tagger_probs.npz`: every drawn picture's probability for each of the judge's tags (float16, one array per "
            "set).",
            "- `sheet_*.jpg`: contact sheets.", "", "## References",
            "\n".join(f"- {t}: {u}" for t, u in CGD_REFERENCES), ""]
    return out


GROUP_READER_TESTS["e055"] = dict(id=CGF_TEST_ID, title=CGF_TEST_TITLE, date=CGF_DATE, readers=CGF_READERS,
                                  controls=CGF_CONTROLS, prefix="cgf", render=render_framed_readers_readme)


# ---- e056: Beatrix reads each description in the framed nano arms' grammar (one fetch per slider) ---------------------------
CGB_TEST_ID = "e056_anima_beatrix_framed_readers"
CGB_TEST_TITLE = ("The stock model: Beatrix reads each description written in the framed nano arms' grammar and sets the three "
                  "group sliders, one fetch per slider")
CGB_DATE = "2026-10-19"
CGB_READERS = ("qwen", "beatrix", "beatrix_untrained")   # qwen: e054's plain-description readings (reused)
CGB_CONTROLS = ("beatrix_untrained",)
CGB_NAMES = {"qwen": "Qwen3 0.6B, final state, reading the plain description (the reference; e054's readings and pictures)",
             "beatrix": "Beatrix (mini-beatrix-3 at step 245,674; never trained on frames) read in the framed grammar at her "
                        "best block and read position (picked on the training descriptions)",
             "beatrix_untrained": "an untrained model of her shape read at the same block and position (a control that must "
                                  "fail)"}


def render_beatrix_framed_readme(meta: dict, recipe: dict) -> str:
    """e056's README: e055's frame and e054's rule with Beatrix as the reader; the result when done."""
    res = meta.get("result") or {}
    rec = "\n".join(f"| {k} | {v} |" for k, v in recipe.items())
    readers = "\n".join(f"| {r} | {n} |" for r, n in CGB_NAMES.items())
    out = [f"# {CGB_TEST_ID}: {CGB_TEST_TITLE}", "",
           f"Date: {CGB_DATE}. Model: {ANIMA.model_name} ([{BASE_MODEL}](https://huggingface.co/{BASE_MODEL})), no LoRA.", "",
           "## Why",
           f"In {CGD_TEST_ID} Beatrix read each description of hair colour, expression and eye colour as a plain caption, one "
           "state for all three attributes, and the amounts she set for the same wording varied from one description to the "
           f"next. In {CGF_TEST_ID} the small nano bodies read each description written in the framed arms' grammar set the "
           "group precisely on wordings they had trained on, and so did a nano body never trained on frames: the precision "
           "came from the format, which puts the value's bytes where the state is read. Here Beatrix, who was never trained "
           "on frames either, reads each description in that format, each slider from its own question.", "",
           "## Design",
           f"- **The descriptions, tiers, cells, pictures and judge**: {CGD_TEST_ID}'s, unchanged (the same 96 training "
           "descriptions, the trained tier and the unseen tier of 32, one picture per description).",
           f"- **The frame**: {CGF_TEST_ID}'s rows, byte for byte (the caption register's head, one sized 32-byte data frame "
           "per attribute holding one record, then the question for one attribute and its value, then the end byte; three "
           "rows per description).",
           f"- **Where it is read**: her stream state (blocks 8-28, the block output layer-normed, as in {CGD_TEST_ID}) at the "
           "question's separator or at the end byte after the value, whichever reads held-out wordings better (below); the "
           "copy check from her own forward (the value's bytes are her greedy choice at every step).",
           f"- **The reading**: {CGF_TEST_ID}'s: per pair, on the rows asking that pair, the ridge regression of the side "
           "(+1 / -1) on the standardized state of the 96 training descriptions; the strength, the block and the position "
           "chosen by **held-out words** (per pair, each side's i-th training wording held out together with every training "
           "description whose wording shares a word with either; 288 reads), ties by the left-out reading, then by name. "
           "The amount on the picture: the reading rounded to a quarter step, clipped to [-1.5, 1.5].",
           "- **The readers**:", "", "| reader | what it is |", "|---|---|", readers, "",
           "## Recipe", "| setting | value |", "|---|---|", rec, "",
           "## The rule (fixed before the run)",
           f"{CGD_TEST_ID}'s: per reader, tier, pair and side, the 16 descriptions' pictures each against their cell's plain "
           "picture; the side is carried when the mean moves its way, at least 12 of 16 move that way and the mean is beyond 3 "
           "standard errors; a reader **DRIVES THE GROUP** when all three pairs are carried on both sides. The control (the "
           "untrained model) must drive no pair; a pair it carries in a tier is credited to no reader there. Beside it: how "
           "far each side moves as a share of the group at amount 1, the copy rates and the stability of each wording's "
           f"amount across the descriptions it appears in (against her plain-caption reading of {CGD_TEST_ID}).", ""]
    out += _framed_result_tail(meta, res, "both readers' readings and reading marks, which block and position each reader "
                                          "is, every tap's held-out-words reading, the copy rates, the stability.")
    return "\n".join(out)


GROUP_READER_TESTS["e056"] = dict(id=CGB_TEST_ID, title=CGB_TEST_TITLE, date=CGB_DATE, readers=CGB_READERS,
                                  controls=CGB_CONTROLS, prefix="cgb", render=render_beatrix_framed_readme)


# ---- e057: the near tier: new wordings made only of words each side's training wordings use ---------------------------------
CGN_TEST_ID = "e057_anima_near_readers"
CGN_TEST_TITLE = ("The stock model: text readers set the three group sliders from new wordings made of words they trained on "
                  "(the near tier), read as plain captions and in the framed grammar")
CGN_DATE = "2026-10-19"
CGN_SEED = 57
CGN_TIERS = ("near",)
CGN_PHRASES = {                             # 4 a side: every word one the same side's training wordings use
    "hair_colour": {"near+": tuple(w + _HAIR for w in ("golden blonde", "pale blonde", "bright yellow", "light golden")),
                    "near-": tuple(w + _HAIR for w in ("very black", "raven black", "inky black", "jet dark"))},
    "expression": {"near+": ("a warm grin", "a cheerful smile", "grinning sweetly", "laughing sweetly"),
                   "near-": ("a fierce frown", "a deep scowl", "scowling angrily", "glaring angrily")},
    "eye_colour": {"near+": ("crimson-red eyes", "ruby-red eyes", "scarlet-red eyes", "garnet-red eyes"),
                   "near-": ("sapphire-blue eyes", "navy-blue eyes", "cobalt-blue eyes", "azure-blue eyes")},
}
CGN_READERS = ("qwen", "beatrix", "beatrix_framed", "nano_framed", "beatrix_untrained_framed", "birth_framed")
CGN_CONTROLS = ("beatrix_untrained_framed", "birth_framed")
CGN_NAMES = {"qwen": "Qwen3 0.6B, final state, reading the plain description (the reference; its training fit of "
                     f"{CGD_TEST_ID})",
             "beatrix": f"Beatrix 3s, block 20, reading the plain description (her training fit of {CGD_TEST_ID})",
             "beatrix_framed": f"Beatrix read in the framed grammar (her block, position and training fit of {CGB_TEST_ID})",
             "nano_framed": f"the framed nano tap of {CGF_TEST_ID} (its block, position and training fit)",
             "beatrix_untrained_framed": f"an untrained model of Beatrix's shape in the framed grammar ({CGB_TEST_ID}'s "
                                         "control; a control that must fail)",
             "birth_framed": f"the framed nano tap's body untrained ({CGF_TEST_ID}'s control; a control that must fail)"}


def cgn_checks() -> dict:
    """The near wordings' rule, fixed before any read: every word of a near wording (the overlap rule's word split) is a word
    the same side's training wordings use; none is identical to a training wording; none shares a word with the other side's
    or another pair's training wordings. -> {'foreign': [...], 'clash': [...], 'identical': [...]}, all empty."""
    own = {(k, s): set().union(*(cpg_words(w) for w in CGD_PHRASES[k]["train" + s])) for k in CGD_PAIRS for s in "+-"}
    train = {w for k in CGD_PAIRS for s in "+-" for w in CGD_PHRASES[k]["train" + s]}
    out: dict = {"foreign": [], "clash": [], "identical": []}
    for k in CGD_PAIRS:
        for s in "+-":
            others = set().union(*(v for key, v in own.items() if key != (k, s)))
            for w in CGN_PHRASES[k]["near" + s]:
                cw = cpg_words(w)
                if cw - own[(k, s)]:
                    out["foreign"].append((k, s, w, sorted(cw - own[(k, s)])))
                if cw & others:
                    out["clash"].append((k, s, w, sorted(cw & others)))
                if w in train:
                    out["identical"].append((k, s, w))
    return out


def cgn_rows() -> list[dict]:
    """e057's near tier: 32 descriptions (4 a corner), one near wording per pair joined by commas in one of six orders, on the
    tiers' cells (e054's references hold); the first shuffle from CGN_SEED with no description repeated."""
    for attempt in range(1000):
        rows = _cgd_split("near", CGN_SEED + 1000 * attempt)
        if len({tuple(r["words"]) for r in rows}) == len(rows):
            return rows
    raise RuntimeError("no shuffle met the rules")  # pragma: no cover


def render_near_readers_readme(meta: dict, recipe: dict) -> str:
    """e057's README: the near wordings, the readers on their registered fits, e054's rule; the result when done."""
    res = meta.get("result") or {}
    rec = "\n".join(f"| {k} | {v} |" for k, v in recipe.items())
    readers = "\n".join(f"| {r} | {n} |" for r, n in CGN_NAMES.items())
    words = "\n".join(f"| {k.replace('_', ' ')} | {s} | {', '.join(CGN_PHRASES[k]['near' + s])} | "
                      f"{', '.join(CGD_PHRASES[k]['train' + s])} |" for k in CGD_PAIRS for s in "+-")
    out = [f"# {CGN_TEST_ID}: {CGN_TEST_TITLE}", "",
           f"Date: {CGN_DATE}. Model: {ANIMA.model_name} ([{BASE_MODEL}](https://huggingface.co/{BASE_MODEL})), no LoRA.", "",
           "## Why",
           f"In {CGD_TEST_ID}, {CGF_TEST_ID} and {CGB_TEST_ID} every reader, untrained ones included, read the wordings it "
           "had trained on (by their spelling), and no byte-level reader read wordings that share no word with training. "
           "Between the two lies the case a person meets most: new wordings made of words the reader has seen (golden "
           "blonde, a fierce frown, sapphire-blue eyes). Does a reader's conditioning reach new combinations of known words, "
           "does the framed grammar carry them better than the plain caption, and does training add anything an untrained "
           "reader of the same shape lacks?", "",
           "## Design",
           "- **The near wordings** (fixed before any read; every word is one the same side's training wordings use, none is "
           "a training wording, none shares a word with the other side's or another pair's training wordings):", "",
           "| pair | side | near wordings | the side's training wordings |", "|---|---|---|---|", words, "",
           f"- **The tier**: 32 descriptions (4 a corner), built as {CGD_TEST_ID}'s unseen tier (one wording a pair, joined "
           "with commas in one of six orders), on the same characters and seeds, read against the same plain pictures and "
           "the same group at amount 1.",
           "- **The readings**: each reader's own fit on the 96 training descriptions, as registered in its earlier test "
           "(the same block, read position and strength); the near descriptions are read with that full fit. The amount on "
           "the picture: the reading rounded to a quarter step, clipped to [-1.5, 1.5].",
           "- **The readers**:", "", "| reader | what it is |", "|---|---|", readers, "",
           "## Recipe", "| setting | value |", "|---|---|", rec, "",
           "## The rule (fixed before the run)",
           f"{CGD_TEST_ID}'s: per reader, pair and side, the 16 descriptions' pictures each against their cell's plain "
           "picture; the side is carried when the mean moves its way, at least 12 of 16 move that way and the mean is beyond 3 "
           "standard errors; a reader **DRIVES THE GROUP** when all three pairs are carried on both sides. The controls (the "
           "untrained models) must drive no pair; a pair a control carries is credited to no reader. Beside it: how far each "
           "side moves as a share of the group at amount 1, and the reading marks.", ""]
    out += _framed_result_tail(meta, res, "every reader's readings of the near descriptions (each on its registered fit), "
                                          "the reading marks, which block and position each reader is.", CGN_TIERS)
    return "\n".join(out)


GROUP_READER_TESTS["e057"] = dict(id=CGN_TEST_ID, title=CGN_TEST_TITLE, date=CGN_DATE, readers=CGN_READERS,
                                  controls=CGN_CONTROLS, prefix="cgn", render=render_near_readers_readme, tiers=CGN_TIERS)


# ---- e058: the off-target tier: descriptions that name no side of the group; the right amount is 0 ---------------------------
CGO_TEST_ID = "e058_anima_offtarget_readers"
CGO_TEST_TITLE = ("The stock model: text readers read descriptions that name no side of the group (the hair's length or "
                  "texture, a composed face, the eyes' shape) and should leave the three sliders at zero")
CGO_DATE = "2026-10-19"
CGO_SEED = 58
CGO_TIERS = ("off",)
CGO_PHRASES = {                             # 4 a pair: no value of the pair's dimension, no word of any earlier wording list
    "hair_colour": tuple(w + _HAIR for w in ("long", "short", "wavy", "straight")),
    "expression": ("a composed look", "an impassive face", "a stoic expression", "a blank expression"),
    "eye_colour": ("small eyes", "close-set eyes", "upturned eyes", "almond-shaped eyes"),
}
CGO_NOUNS = frozenset({"look", "face", "expression"})    # expression nouns: the unseen wordings use them on both sides alike
CGO_READERS = CGN_READERS
CGO_CONTROLS = CGN_CONTROLS
CGO_CONTROL_OF = {"beatrix_framed": "beatrix_untrained_framed", "nano_framed": "birth_framed"}
CGO_ZEROS = 24                              # of 32 amounts at 0 to stay plain: three quarters, the carried rule's share
CGO_MOVE = 0.25                             # the mean absolute move under one grid step of the group's full move


def cgo_checks() -> dict:
    """The off-target wordings' rule, fixed before any read: no word of an off-target wording (the overlap rule's word split,
    the expression nouns aside) appears in any wording of any earlier list (every pair, side and tier of e050-e057); no word
    is shared by two pairs' off-target wordings; each expression noun appears on both unseen expression sides.
    -> {'clash': [...], 'shared': [...], 'nouns': [...]}, all empty."""
    seen: set = set()
    for lists in (CPG_PHRASES, CGD_PHRASES, CGN_PHRASES):
        for P in lists.values():
            for ws in P.values():
                for w in ws:
                    seen |= cpg_words(w)
    own = {k: set().union(*(cpg_words(w) - CGO_NOUNS for w in ws)) for k, ws in CGO_PHRASES.items()}
    out: dict = {"clash": [], "shared": [], "nouns": []}
    for k, ws in CGO_PHRASES.items():
        for w in ws:
            hit = (cpg_words(w) - CGO_NOUNS) & seen
            if hit:
                out["clash"].append((k, w, sorted(hit)))
    ks = list(CGO_PHRASES)
    for a in range(len(ks)):
        for b in range(a + 1, len(ks)):
            if own[ks[a]] & own[ks[b]]:
                out["shared"].append((ks[a], ks[b], sorted(own[ks[a]] & own[ks[b]])))
    sides = [set().union(*(cpg_words(w) for w in CGD_PHRASES["expression"]["unseen" + s])) for s in "+-"]
    out["nouns"] = sorted(n for n in CGO_NOUNS if not all(n in s for s in sides))
    return out


def cgo_rows() -> list[dict]:
    """e058's off-target tier: 32 descriptions, one off-target wording a pair (each pair's four wordings 8 times, a seeded
    shuffle), joined by commas in one of six orders, the i-th on the i-th near description's cell; no corner (the right
    amount is 0 on every pair); the first shuffle from CGO_SEED with no description repeated."""
    cells = [r["cell"] for r in cgn_rows()]
    for attempt in range(1000):
        seed = CGO_SEED + 1000 * attempt
        pools = [_cgd_cycle(CGO_PHRASES[k], len(cells), seed + 10 * i) for i, k in enumerate(CGD_PAIRS)]
        rows = []
        for j, cell in enumerate(cells):
            words = [pools[i][j] for i in range(len(CGD_PAIRS))]
            order = CGD_ORDERS[j % len(CGD_ORDERS)]
            text = ", ".join(words[i] for i in order)
            rows.append({"id": f"off{j:02d}", "split": "off", "corner": None, "words": words, "order": list(order),
                         "text": text, "frame": CGD_READ_FORM.format(w=text), "cell": cell})
        if len({tuple(r["words"]) for r in rows}) == len(rows):
            return rows
    raise RuntimeError("no shuffle met the rules")  # pragma: no cover


def cgo_tier_marks(amounts: dict, tier: str = "off") -> dict:
    """The off-target tier's reading marks (no pictures). amounts = {row id: [reading per pair]}. Per pair: the grid amounts
    at 0 (of 32), the mean amount size, the shares above and below 0; the pair is read PLAIN when at least CGO_ZEROS
    amounts are 0; the descriptions with all three at 0 (their picture is the plain picture)."""
    rows = cgd_tier_rows(tier)
    G = [cgd_amounts(amounts[r["id"]]) for r in rows]
    pairs = {}
    for i, k in enumerate(CGD_PAIRS):
        g = [a[i] for a in G]
        zeros = sum(1 for x in g if x == 0)
        pairs[k] = {"zeros": zeros, "of": len(g), "mean_size": sum(abs(x) for x in g) / len(g),
                    "above": sum(1 for x in g if x > 0) / len(g), "below": sum(1 for x in g if x < 0) / len(g),
                    "plain": zeros >= CGO_ZEROS}
    n = sum(v["plain"] for v in pairs.values())
    return {"pairs": pairs, "whole_descriptions": sum(1 for a in G if not any(a)), "of": len(rows),
            "OUTCOME": ("PLAIN ON THE GROUP" if n == len(CGD_PAIRS) else
                        f"PLAIN ON {n} OF {len(CGD_PAIRS)}" if n else "PLAIN ON NONE")}


def cgo_tier_reads(scores: dict, plain: dict, corners: dict, amounts: dict, control_of=None, tier: str = "off") -> dict:
    """The off-target tier's picture marks, fixed before the run. Per reader and pair: the 32 grid amounts (an amount under
    an eighth is 0 and leaves the pair's slider off) and the 32 pictures' moves on the pair's score against their cell's
    plain picture, each over the group's full move on that cell (the mean size of the eight corners' moves at amount 1). The
    pair STAYS PLAIN when at least CGO_ZEROS amounts are 0 and the mean absolute move is under CGO_MOVE; the reader STAYS
    PLAIN ON THE GROUP when all three pairs do. Credited: a pair a reader stays plain on while its control (control_of =
    {reader: its control}) does not; a reference (no control) and a control are credited nothing."""
    import numpy as np
    rows = cgd_tier_rows(tier)
    full = {k: [float(np.mean([abs(float(c[k][r["cell"]]) - float(plain[k][r["cell"]])) for c in corners.values()]))
                for r in rows] for k in CGD_PAIRS}
    per = {}
    for reader, by_tier in scores.items():
        S = by_tier[tier]
        G = [cgd_amounts(amounts[reader][r["id"]]) for r in rows]
        pairs = {}
        for i, k in enumerate(CGD_PAIRS):
            rel = [(float(S[k][j]) - float(plain[k][r["cell"]])) / full[k][j] if full[k][j] else 0.0
                   for j, r in enumerate(rows)]
            g = [a[i] for a in G]
            zeros = sum(1 for x in g if x == 0)
            move = float(np.mean(np.abs(rel)))
            pairs[k] = {"zeros": zeros, "of": len(rows), "mean_size": float(np.mean(np.abs(g))), "mean_abs_move": move,
                        "mean_move": float(np.mean(rel)), "above": sum(1 for x in g if x > 0) / len(g),
                        "below": sum(1 for x in g if x < 0) / len(g), "amounts": g,
                        "OUTCOME": "STAYS PLAIN" if zeros >= CGO_ZEROS and move < CGO_MOVE else "MOVES IT"}
        n = sum(v["OUTCOME"] == "STAYS PLAIN" for v in pairs.values())
        per[reader] = {"pairs": pairs, "plain_pairs": [k for k in CGD_PAIRS if pairs[k]["OUTCOME"] == "STAYS PLAIN"],
                       "plain_pictures": sum(1 for a in G if not any(a)), "of": len(rows),
                       "OUTCOME": ("STAYS PLAIN ON THE GROUP" if n == len(CGD_PAIRS) else
                                   f"STAYS PLAIN ON {n} OF {len(CGD_PAIRS)}" if n else "STAYS PLAIN ON NONE")}
    control_of = control_of or {}
    for reader, v in per.items():
        c = control_of.get(reader)
        v["control"] = c
        v["credited"] = [k for k in v["plain_pairs"] if k not in per[c]["plain_pairs"]] if c in per else []
    controls = sorted(set(control_of.values()))
    return {"readers": per, "controls_plain": {k: [c for c in controls if c in per and k in per[c]["plain_pairs"]]
                                               for k in CGD_PAIRS}}


def _offtarget_table(per: dict) -> list:
    """The off-target tier's result table (e058)."""
    out = ["### The off-target tier", "",
           "| reader | verdict | " + " | ".join(k.replace("_", " ") for k in CGD_PAIRS) + " | pictures left plain | credited |",
           "|---|---|" + "---|" * len(CGD_PAIRS) + "---|---|"]
    for name, v in per.items():
        cells = [f"{p['OUTCOME'].lower()}: {p['zeros']} of {p['of']} at 0, size {p['mean_size']:.2f}, move "
                 f"{p['mean_abs_move']:.2f}" for p in (v["pairs"][k] for k in CGD_PAIRS)]
        out.append(f"| {name} | **{v['OUTCOME']}** | " + " | ".join(cells) + f" | {v['plain_pictures']} of {v['of']} | "
                   + (", ".join(v["credited"]) or "none") + " |")
    return out + ["", "Each cell: the pair's verdict, the amounts at 0, the mean amount size and the mean absolute move of the "
                  "pictures on the pair as a share of the group's full move. Credited: a pair the reader stays plain on while "
                  "its untrained control does not.", ""]


def render_offtarget_readers_readme(meta: dict, recipe: dict) -> str:
    """e058's README: the off-target wordings, the readers on their registered fits, the rule fixed before the run; the result
    when done."""
    res = meta.get("result") or {}
    rec = "\n".join(f"| {k} | {v} |" for k, v in recipe.items())
    readers = "\n".join(f"| {r} | {n} |" for r, n in CGN_NAMES.items())
    words = "\n".join(f"| {k.replace('_', ' ')} | {', '.join(CGO_PHRASES[k])} | {', '.join(CGD_PHRASES[k]['train+'])} / "
                      f"{', '.join(CGD_PHRASES[k]['train-'])} |" for k in CGD_PAIRS)
    out = [f"# {CGO_TEST_ID}: {CGO_TEST_TITLE}", "",
           f"Date: {CGO_DATE}. Model: {ANIMA.model_name} ([{BASE_MODEL}](https://huggingface.co/{BASE_MODEL})), no LoRA.", "",
           "## Why",
           f"In {CGB_TEST_ID} Beatrix, read in the framed grammar, set small amounts on wordings she could not read (her "
           "pictures stayed near the plain one), where an untrained model of her shape set large amounts that went either "
           "way. A text reader wired into a picture system meets many descriptions that name none of its sliders' sides, and "
           "there the right amount is zero. Here every description names the three attributes without any value of the "
           "group's dimensions (the hair's length or texture, a composed face, the eyes' shape): does each reader leave the "
           "sliders at zero, and does training add that over an untrained reader of the same shape?", "",
           "## Design",
           "- **The off-target wordings** (fixed before any read; no word of them appears in any wording of the earlier "
           "tests, the expression nouns look, face and expression aside, which the unseen wordings use on both sides alike; "
           "none names a hair colour, a smile or a frown, or an eye colour):", "",
           "| pair | off-target wordings | the pair's training wordings (+ / -) |", "|---|---|---|", words, "",
           "- **The tier**: 32 descriptions, one off-target wording a pair (each wording 8 times), joined with commas in one "
           f"of six orders, on the characters and seeds of {CGN_TEST_ID}'s tier, read against the same plain pictures and "
           "the same group at amount 1.",
           "- **The readings**: each reader's own fit on the 96 training descriptions, as registered in its earlier test; "
           "the off-target descriptions are read with that full fit. The amount on the picture: the reading rounded to a "
           "quarter step, clipped to [-1.5, 1.5]; an amount under an eighth is 0 and leaves the pair's slider off, so a "
           "description read at 0 on all three pairs is the plain picture itself.",
           "- **The readers**:", "", "| reader | what it is |", "|---|---|", readers, "",
           "## Recipe", "| setting | value |", "|---|---|", rec, "",
           "## The rule (fixed before the run)",
           "Per reader and pair: the 32 grid amounts and the 32 pictures' moves on the pair's score against their cell's "
           "plain picture, each as a share of the group's full move on that cell (the mean size of the eight corners' moves "
           f"at amount 1). The pair **STAYS PLAIN** when at least {CGO_ZEROS} of 32 amounts are 0 and the mean absolute move "
           f"is under {CGO_MOVE}; a reader **STAYS PLAIN ON THE GROUP** when all three pairs do. Beatrix in the framed "
           "grammar is credited against her untrained copy, the framed nano tap against its body untrained: a pair is "
           "credited when the reader stays plain on it and its control does not. Qwen3 and Beatrix on the plain description "
           "are references.", ""]
    out += _framed_result_tail(meta, res, "every reader's readings of the off-target descriptions (each on its registered "
                                          "fit), the reading marks, which block and position each reader is.", CGO_TIERS)
    return "\n".join(out)


GROUP_READER_TESTS["e058"] = dict(id=CGO_TEST_ID, title=CGO_TEST_TITLE, date=CGO_DATE, readers=CGO_READERS,
                                  controls=CGO_CONTROLS, control_of=CGO_CONTROL_OF, prefix="cgo",
                                  render=render_offtarget_readers_readme, tiers=CGO_TIERS)


# ---- e059: the caption-finetuned nano body on the unseen and off-target tiers -----------------------------------------------
CGC_TEST_ID = "e059_anima_caption_readers"
CGC_TEST_TITLE = ("The stock model: a nano body finetuned on captions, each attribute in many wordings, reads descriptions in the "
                  "framed grammar beside its control, its start and its untrained body, on wordings it never trained on and on "
                  "descriptions that name no side")
CGC_DATE = "2026-10-19"
CGC_TIERS = ("unseen", "off")
CGC_READERS = ("mx_cp", "mx", "mx_cpx", "birth")
CGC_CONTROLS = ("mx_cpx", "birth")
CGC_CONTROL_OF = {"mx_cp": "mx_cpx"}
CGC_NAMES = {"mx_cp": "the nano body Pollux dual after the caption phase (captions from a caption pack, each attribute in many "
                      "wordings, its records written as frames), read in the framed grammar at its best block and position",
             "mx": "the same body before the caption phase (its start: the mixed phase's end), at the same block and position",
             "mx_cpx": "the same caption phase with every frame taken from another caption, so the frame says nothing about its "
                       "caption (a control that must fail), at the same block and position",
             "birth": "the body untrained (a control that must fail), at the same block and position"}


def render_caption_readers_readme(meta: dict, recipe: dict) -> str:
    """e059's README: why, the readers, both tiers' rules fixed before the run; the result when done."""
    res = meta.get("result") or {}
    rec = "\n".join(f"| {k} | {v} |" for k, v in recipe.items())
    readers = "\n".join(f"| {r} | {n} |" for r, n in CGC_NAMES.items())
    out = [f"# {CGC_TEST_ID}: {CGC_TEST_TITLE}", "",
           f"Date: {CGC_DATE}. Model: {ANIMA.model_name} ([{BASE_MODEL}](https://huggingface.co/{BASE_MODEL})), no LoRA.", "",
           "## Why",
           f"From {CGD_TEST_ID} to {CGO_TEST_ID}: read in the framed grammar, every reader set the three sliders precisely on "
           "wordings whose words it had trained on, and an untrained reader of the same shape did as well (the frame puts the "
           "value's letters where the reader is read, so a trained wording is a spelling test); no byte-level reader read "
           "wordings that share no word with training; no reader left the sliders at zero on descriptions that name no side. "
           "What the readers lack is word meaning. Here the nano body Pollux dual has been finetuned on captions in which each "
           "attribute comes in many wordings, written as framed records: does it now read wordings it never trained on, and "
           "does it leave the sliders at zero when a description names no side? Its control saw the same captions with every "
           "frame taken from another caption.", "",
           "## Design",
           f"- **The tiers**: {CGD_TEST_ID}'s unseen tier (32 descriptions whose wordings share no word with the 96 training "
           f"descriptions) and {CGO_TEST_ID}'s off-target tier (32 descriptions that name no side), on their characters and "
           "seeds, read against the same plain pictures and the same group at amount 1.",
           "- **The readings**: each description written in the framed grammar (three sized 32-byte record frames, then the "
           "question for one attribute and its value), each slider read on the rows asking its attribute; per pair the ridge "
           "regression of the side on the state standardized over the 96 training descriptions; the finetuned body's block and "
           "position picked by held-out words (a wording and every training wording sharing a word with it held out of the "
           "fit), the other three bodies read at the same block and position; the amount on the picture: the reading rounded "
           "to a quarter step, clipped to [-1.5, 1.5].",
           "- **The readers**:", "", "| reader | what it is |", "|---|---|", readers, "",
           "## Recipe", "| setting | value |", "|---|---|", rec, "",
           "## The rule (fixed before the run)",
           f"- **The unseen tier**: {CGD_TEST_ID}'s rule: per reader, pair and side, the 16 descriptions' pictures each "
           "against their cell's plain picture; the side is carried when the mean moves its way, at least 12 of 16 move that "
           "way and the mean is beyond 3 standard errors; a reader **DRIVES THE GROUP** when all three pairs are carried on "
           "both sides; a pair a control carries is credited to no reader.",
           f"- **The off-target tier**: {CGO_TEST_ID}'s rule: a pair **STAYS PLAIN** when at least {CGO_ZEROS} of 32 amounts "
           f"are 0 and the pictures' mean absolute move on it is under {CGO_MOVE} of the group's full move; the finetuned body "
           "is credited against its control.", ""]
    out += _framed_result_tail(meta, res, "every reader's readings of both tiers' descriptions, the reading marks, the "
                                          "held-out-words read of every block and position, which block and position was "
                                          "picked.", CGC_TIERS)
    return "\n".join(out)


GROUP_READER_TESTS["e059"] = dict(id=CGC_TEST_ID, title=CGC_TEST_TITLE, date=CGC_DATE, readers=CGC_READERS,
                                  controls=CGC_CONTROLS, control_of=CGC_CONTROL_OF, prefix="cgc",
                                  render=render_caption_readers_readme, tiers=CGC_TIERS)


# ---- e061: the zero class: readers fitted with descriptions that name no side, read on new ones -------------------------
CGZ_TEST_ID = "e061_anima_zero_class_readers"
CGZ_SEED = 61
CGZ_PHRASES = {                             # 6 a pair, a fit's zero rows: no value of the pair's dimension, no word of any
    "hair_colour": tuple(w + _HAIR for w in ("curly", "braided", "messy", "layered", "tousled", "cropped")),   # earlier list
    "expression": ("a thoughtful look", "a focused look", "a pensive face", "an attentive look", "a quiet face",
                   "a contemplative look"),
    "eye_colour": ("narrow eyes", "hooded eyes", "downturned eyes", "prominent eyes", "tiny eyes", "oval eyes"),
}
CGZ_ROWS = 48                               # the zero rows: each wording 8 times


def cgz_checks() -> dict:
    """The zero rows' wordings, fixed before any read: no word (the overlap rule's word split, the expression nouns aside)
    appears in any wording of any earlier list, e058's off-target wordings included (those are the test's); no word is
    shared by two pairs' wordings. -> {'clash': [...], 'shared': [...]}, both empty."""
    seen: set = set()
    for lists in (CPG_PHRASES, CGD_PHRASES, CGN_PHRASES):
        for P in lists.values():
            for ws in P.values():
                for w in ws:
                    seen |= cpg_words(w)
    for ws in CGO_PHRASES.values():
        for w in ws:
            seen |= cpg_words(w)
    own = {k: set().union(*(cpg_words(w) - CGO_NOUNS for w in ws)) for k, ws in CGZ_PHRASES.items()}
    out: dict = {"clash": [], "shared": []}
    for k, ws in CGZ_PHRASES.items():
        for w in ws:
            hit = (cpg_words(w) - CGO_NOUNS) & seen
            if hit:
                out["clash"].append((k, w, sorted(hit)))
    ks = list(CGZ_PHRASES)
    for a in range(len(ks)):
        for b in range(a + 1, len(ks)):
            if own[ks[a]] & own[ks[b]]:
                out["shared"].append((ks[a], ks[b], sorted(own[ks[a]] & own[ks[b]])))
    return out


def cgz_rows() -> list[dict]:
    """e061's zero rows: 48 descriptions that name no side, one wording a pair (each of a pair's six wordings 8 times, a
    seeded shuffle), joined by commas in one of six orders; no corner (a reader's fit takes them at 0 on every pair) and no
    cell (they are only read, never drawn); the first shuffle from CGZ_SEED with no description repeated."""
    for attempt in range(1000):
        seed = CGZ_SEED + 1000 * attempt
        pools = [_cgd_cycle(CGZ_PHRASES[k], CGZ_ROWS, seed + 10 * i) for i, k in enumerate(CGD_PAIRS)]
        rows = []
        for j in range(CGZ_ROWS):
            words = [pools[i][j] for i in range(len(CGD_PAIRS))]
            order = CGD_ORDERS[j % len(CGD_ORDERS)]
            text = ", ".join(words[i] for i in order)
            rows.append({"id": f"zero{j:02d}", "split": "zero", "corner": None, "words": words, "order": list(order),
                         "text": text, "frame": CGD_READ_FORM.format(w=text), "cell": None})
        if len({tuple(r["words"]) for r in rows}) == len(rows):
            return rows
    raise RuntimeError("no shuffle met the rules")  # pragma: no cover


CGZ_TEST_TITLE = ("The stock model: text readers fitted with descriptions that name no side of the group (taken at zero) read "
                  "new descriptions that name no side, and should leave the three sliders at zero")
CGZ_DATE = "2026-10-19"
CGZ_TIERS = CGO_TIERS


def render_zero_class_readme(meta: dict, recipe: dict) -> str:
    """e061's README: the zero rows, the readers refitted with them, the rule fixed before the run; the result when done."""
    res = meta.get("result") or {}
    rec = "\n".join(f"| {k} | {v} |" for k, v in recipe.items())
    readers = "\n".join(f"| {r} | {n} |" for r, n in CGN_NAMES.items())
    words = "\n".join(f"| {k.replace('_', ' ')} | {', '.join(CGZ_PHRASES[k])} | {', '.join(CGO_PHRASES[k])} |"
                      for k in CGD_PAIRS)
    out = [f"# {CGZ_TEST_ID}: {CGZ_TEST_TITLE}", "",
           f"Date: {CGZ_DATE}. Model: {ANIMA.model_name} ([{BASE_MODEL}](https://huggingface.co/{BASE_MODEL})), no LoRA.", "",
           "## Why",
           f"In {CGO_TEST_ID} no reader left the sliders at zero on descriptions that name no side: every reader's fit was "
           "made on training descriptions that each name a side, so the fit has no zero answer and every description lands "
           "on one side of each pair. Here each reader is fitted again on the same state with descriptions that name no side "
           "added at zero, then reads the off-target descriptions, whose wordings none of the added descriptions uses.", "",
           "## Design",
           "- **The added descriptions** (fixed before any read; no word of them appears in any wording of the earlier tests, "
           "the off-target wordings included, the expression nouns look, face and expression aside): 48 descriptions, one "
           "wording a pair, each wording 8 times, joined with commas in one of six orders, taken at 0 on all three pairs:", "",
           "| pair | the added (zero) wordings | the off-target wordings read |", "|---|---|---|", words, "",
           f"- **The tier**: {CGO_TEST_ID}'s 32 off-target descriptions, on its characters and seeds, read against the same "
           "plain pictures and the same group at amount 1.",
           "- **The readings**: each reader on its registered state, block, read position and strength; per pair the ridge "
           "regression of the side on the standardized state of the 96 training descriptions and the 48 added ones at 0; "
           "the off-target descriptions read with the full fit. The amount on the picture: the reading rounded to a quarter "
           "step, clipped to [-1.5, 1.5]; an amount under an eighth is 0 and leaves the pair's slider off, so a description "
           "read at 0 on all three pairs is the plain picture itself.",
           "- **The readers**:", "", "| reader | what it is |", "|---|---|", readers, "",
           "## Recipe", "| setting | value |", "|---|---|", rec, "",
           "## The rule (fixed before the run)",
           f"{CGO_TEST_ID}'s rule: per reader and pair, the 32 grid amounts and the 32 pictures' moves on the pair's score "
           "against their cell's plain picture, each as a share of the group's full move on that cell. The pair **STAYS "
           f"PLAIN** when at least {CGO_ZEROS} of 32 amounts are 0 and the mean absolute move is under {CGO_MOVE}. Beatrix "
           "in the framed grammar is credited against her untrained copy, the framed nano tap against its body untrained: a "
           "pair is credited when the reader stays plain on it and its control does not. Qwen3 and Beatrix on the plain "
           "description are references.", ""]
    out += _framed_result_tail(meta, res, "every reader's readings of the off-target descriptions on the refitted fit, the "
                                          "reading marks, the fit's cost on the descriptions that name a side.", CGZ_TIERS)
    return "\n".join(out)


GROUP_READER_TESTS["e061"] = dict(id=CGZ_TEST_ID, title=CGZ_TEST_TITLE, date=CGZ_DATE, readers=CGO_READERS,
                                  controls=CGO_CONTROLS, control_of=CGO_CONTROL_OF, prefix="cgz",
                                  render=render_zero_class_readme, tiers=CGZ_TIERS)


# ---- e062: the joint reader: Anima's own text encoder read together with a framed reader --------------------------------
CGJ_TEST_ID = "e062_anima_joint_readers"
CGJ_TEST_TITLE = ("The stock model: Anima's own text encoder read together with a framed byte reader (one joined state per "
                  "slider) sets the three group sliders from wordings no reader trained on")
CGJ_DATE = "2026-10-19"
CGJ_TIERS = ("unseen",)
CGJ_READERS = ("qwen+beatrix_framed", "qwen+nano_framed", "qwen+beatrix_untrained_framed", "qwen+birth_framed")
CGJ_CONTROLS = ("qwen+beatrix_untrained_framed", "qwen+birth_framed")
CGJ_CONTROL_OF = {"qwen+beatrix_framed": "qwen+beatrix_untrained_framed", "qwen+nano_framed": "qwen+birth_framed"}
CGJ_NAMES = {"qwen+beatrix_framed": f"Qwen3's final state on the plain description joined with Beatrix's state in the framed "
                                    f"grammar ({CGB_TEST_ID}'s block and position)",
             "qwen+nano_framed": f"Qwen3's final state joined with the framed nano tap's state ({CGF_TEST_ID}'s block and "
                                 "position; its narrower state repeated four times so the two weigh alike)",
             "qwen+beatrix_untrained_framed": "Qwen3's final state joined with an untrained model of Beatrix's shape in the framed "
                                              "grammar (a control)",
             "qwen+birth_framed": "Qwen3's final state joined with the framed nano tap's body untrained (a control)"}


def render_joint_readers_readme(meta: dict, recipe: dict) -> str:
    """e062's README: why, the joint readers, the unseen tier's rule fixed before the run; the result when done."""
    res = meta.get("result") or {}
    rec = "\n".join(f"| {k} | {v} |" for k, v in recipe.items())
    readers = "\n".join(f"| {r} | {n} |" for r, n in CGJ_NAMES.items())
    out = [f"# {CGJ_TEST_ID}: {CGJ_TEST_TITLE}", "",
           f"Date: {CGJ_DATE}. Model: {ANIMA.model_name} ([{BASE_MODEL}](https://huggingface.co/{BASE_MODEL})), no LoRA.", "",
           "## Why",
           f"From {CGD_TEST_ID} to {CGZ_TEST_ID}: the byte-level readers read in the framed grammar set the sliders exactly on "
           "wordings whose words they trained on and not beyond them; Qwen3, Anima's own text encoder, reads more widely but "
           "loosely. Here each slider reads one joined state: Qwen3's and a framed reader's together. Before any picture the "
           "joined readers' held-out-words count (a training wording and every training description sharing a word with it held "
           "out of the fit, 288 reads) decided whether pictures are drawn: the Qwen3 + nano tap reader read 232 against Qwen3's "
           "212 alone and 183 with the untrained nano body, so the unseen tier is drawn for the four joined readers.", "",
           "## Design",
           f"- **The tier**: {CGD_TEST_ID}'s unseen tier (32 descriptions whose wordings share no word with the 96 training "
           "descriptions), on its characters and seeds, read against the same plain pictures and the same group at amount 1.",
           "- **The readings**: per pair the ridge regression of the side on the joined state standardized over the 96 training "
           "descriptions, at the strength the held-out words chose; the unseen descriptions read with the full fit; the amount "
           "on the picture: the reading rounded to a quarter step, clipped to [-1.5, 1.5].",
           "- **The readers**:", "", "| reader | what it is |", "|---|---|", readers, "",
           "## Recipe", "| setting | value |", "|---|---|", rec, "",
           "## The rule (fixed before the run)",
           f"{CGD_TEST_ID}'s rule: per reader, pair and side, the 16 descriptions' pictures each against their cell's plain "
           "picture; the side is carried when the mean moves its way, at least 12 of 16 move that way and the mean is beyond 3 "
           "standard errors; a reader **DRIVES THE GROUP** when all three pairs are carried on both sides; a pair a control "
           "carries is credited to no reader.", ""]
    out += _framed_result_tail(meta, res, "every joined reader's readings of the tiers' descriptions, the held-out words, the "
                                          "reading marks.", CGJ_TIERS)
    return "\n".join(out)


GROUP_READER_TESTS["e062"] = dict(id=CGJ_TEST_ID, title=CGJ_TEST_TITLE, date=CGJ_DATE, readers=CGJ_READERS,
                                  controls=CGJ_CONTROLS, control_of=CGJ_CONTROL_OF, prefix="cgj",
                                  render=render_joint_readers_readme, tiers=CGJ_TIERS)
