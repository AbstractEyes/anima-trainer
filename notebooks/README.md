# Colab notebooks

## `anima_colab_prelim_train.ipynb`
Preliminary **~1000-image, rank-64 Anima LoRA** trained on **semantic subject buckets**, on a
Google Colab **RTX PRO 6000 Blackwell** (sm_120, 96 GB) runtime — the repo's real training
target. Drives the `geolip_anima_trainer` package + diffusion-pipe; backs checkpoints up to
HuggingFace every few minutes so the LoRA survives a Colab disconnect.

### Before you run
- Select a **RTX PRO 6000 Blackwell** GPU runtime (the notebook installs the cu128 torch
  build Blackwell/sm_120 requires and verifies it).
- Add a **write-scoped** HuggingFace token to Colab **Secrets** (🔑) named **`HF_TOKEN`**
  (used to read the dataset/model and to push checkpoint backups).

### Run order
Top to bottom. There is **one runtime restart** right after the install cell (§2) — expected.
§1 GPU → §2 clone+install(→restart) → §3 verify+`anima doctor` → §4 HF auth → §5 download
model → **§6 extract ~1000 into subject buckets** → **§7 build rank-64 config + dampened
weighting** → §8 cache → §9 train(bg)+periodic backup → §10 notes.

### Methodology (baked in, from `CLAUDE.md` for `qwen_90k`)
- Caption = **`caption_vlm_json` `task_1` JSON trained VERBATIM** (+ `caption_animetimm_json`
  as a hardlinked 2nd sample) — not rendered to tags.
- **Subject buckets** (`anima subjects`, columnar pyarrow): dominant-subject keys; sparse
  subjects grouped by **semantic similarity** (`[similarity]` extra: sentence-transformers,
  falls back to char-trigram/difflib), never dropped; **human subgroups kept separate**.
- **Anti-overtraining weighting:** `num_repeats` via the diminishing-returns policy
  (`balance_alpha=0.5`, capped at `max_repeats=8`), not the legacy 50× equalization.
- `require_age_pass=False` (age col unpopulated; audit gate **on**), `limit=1000`,
  `llm_adapter_lr=0` (frozen), `shuffle_caption=false` (tag-order sensitive).

### License
Anima and any LoRA from it are **NON-COMMERCIAL** (CircleStone NC + NVIDIA Open Model
License / Cosmos derivative). The backup model card is labelled accordingly.

## `sana_colab_train.ipynb`
**Sana LoRA experiments** on a Colab GPU runtime (the RTX PRO 6000 is the target), trained through
model type `sana` in the [AbstractEyes diffusion-pipe fork](https://github.com/AbstractEyes/diffusion-pipe)
and uploaded one folder per experiment into
[AbstractPhil/geolip-beatrix-sana](https://huggingface.co/AbstractPhil/geolip-beatrix-sana). Thin shell
over `geolip_anima_trainer.sana_runner.SanaRunner`: bootstrap (`anima_colab.install(...,
dp_url=anima_colab.DP_FORK_URL)` clones the fork at `external/diffusion-pipe-fork`; one restart), then
`setup()` → `run_sequence()`.

- **No dataset needed:** the stock Sana 600M 512px model renders its own training sets, images of one
  mood (24 subjects × 8 seeds = 192) captioned with the neutral prompt "a photo of <subject>"; 8 more
  subjects never enter training.
- **The sequence** (`sana_experiments.SEQUENCE`, each the first recipe with one change): e004 upbeat
  images; e005 the model's own neutral images (a control that can fail); e006 downbeat images; e007 a
  fresh draw of upbeat images; e008 / e009 half / double the learning rate.
- **Recipe:** rank 32, plain Adam (weight decay 0) at 1e-4, 10 epochs at micro-batch 4 (480 steps),
  20 warmup steps, a save + 4 preview images every 2 epochs, `shift = 3.0`.
- **Evaluation:** every saved LoRA is loaded through diffusers (`load_lora_weights`) and the 8 held-out
  subjects × 4 seeds are rendered with it (scale 1; the final epoch also at 0.5), paired against the same
  cells without a LoRA; a CLIP mood judge (upbeat phrases minus downbeat phrases) gives each arm's verdict
  in its direction: **FLAVOR LORA** (the mean moves that way, at least 75% of the cells that way, beyond
  3 standard errors), **NO EFFECT** (within 2 standard errors of 0, or under 60% that way), else **MIXED**;
  the reads across arms (the control, the net effect, the replicate, the learning-rate dose) go into the
  repo README.
- **Uploads** (a WRITE `HF_TOKEN` in Colab Secrets; checked before any GPU work): `experiments/<id>/`
  with README, `meta.json`, `config/`, `data/`, every `lora/epochN/` as it is saved, `samples/`, `eval/`,
  `logs/`. A rerun skips the arms the repo lists as done.
- **One ad-hoc run on your own images:** `SanaRunner(source="folder", dataset_dir=..., preview_prompts=[...])`
  with `prepare_dataset()` → `build_configs()` → `train()` → `evaluate()` → `backup()`.
- **Side by side:** `s.run_sequence(parallel=3)` trains up to three arms at once on the card.

Sana's weights are Apache-2.0; its Gemma-2-2B-IT text encoder is under the Gemma Terms of Use.

## `anima_colab_experiments.ipynb`
The same experiment system on **Anima** (Anima-Base v1.0 at 768 px), uploaded one folder per experiment
into [AbstractPhil/geolip-beatrix-anima](https://huggingface.co/AbstractPhil/geolip-beatrix-anima). Thin
shell over `geolip_anima_trainer.anima_runner.AnimaRunner`: the same bootstrap, then `setup()` →
`run_flavor_test()` → `run_attribute_screen()` → `run_sequence()` → `run_beatrix_connectors()`. Every long
step prints the job's size, then done / total, time spent and time left.

- **e001** (`run_flavor_test`): 32 scenes × {neutral, upbeat, downbeat} words × 2 seeds; the conditioning
  norms of the 32 neutral prompts at two sites (the Qwen3 states the LLM adapter reads, and the adapter's
  output the DiT cross-attends to); a mood direction added at each site at alpha −2..+2 (about 700 images).
- **e020** (`run_route_split`): the adapter reads a caption as Qwen3's states and as T5 word ids; the mood prompt
  through one of the two at a time, on e001's 64 cells (448 images, ~25 min).
- **e021** (`run_appended_split`): e020 with the mood words after the scene; the gate is what the source half adds
  to the query half on the downbeat words (448 images). **e022** (`run_query_dial`): a mood direction in the
  adapter's queries, alone and with one appended source token (1,088 images).
- **e012** (`run_attribute_screen`): six attributes as tag pairs (hair length, hair colour, eye colour, chibi,
  age for adults only, style) on 8 characters × 2 seeds; the words, a slider after the adapter at ±2, the
  cross-talk between sliders; judged by the WD EVA02-Large tagger v3 (352 images, ~20 min).
- **The LoRA arms** (`anima_experiments.SEQUENCE`): e002 upbeat, e003 the control, e004 downbeat, e005 a
  fresh draw, e006 / e007 half / double the learning rate, then e008–e011 the second draw.
- **e013–e019** (`run_beatrix_connectors`): a push after the adapter computed from Beatrix's features for a
  mood phrase (e013), the same on an untrained Beatrix (e014, must fail on unseen phrases), a free vector
  per mood (e015); each trained 720 steps in the notebook by Anima's own objective on the first-draw training
  images, then scored on the held-out scenes (16 cells per phrase; about 20 minutes each). e016 / e017
  re-run e013 / e014 on whitened features (e013's map learned too slowly to read); e018 / e019 on a slider
  value (the phrase's position between the gloomy and cheerful training phrases, and toward the neutral ones).
  The features come from `beatrix/` in the data repo.
- **Recipe** (from the model card): rank 32 at 2e-5, the LLM adapter frozen, plain Adam with weight decay 0
  and fp32 master weights, 10 epochs at micro-batch 4, the card's quality prefix and negative prompt;
  evaluation at 30 steps, guidance 4.5, shift 3.
- **Rendering** in the notebook process with the fork's own Anima code; LoRAs (ComfyUI format) applied by
  forward hooks. The first arm checks the renderer against the trainer's preview images.
- **Training sets** are kept in
  [AbstractPhil/geolip-beatrix-anima-data](https://huggingface.co/datasets/AbstractPhil/geolip-beatrix-anima-data):
  uploaded once drawn, downloaded by a fresh runtime when drawn with the same settings (`training_sets.py`).

Anima's weights are under the CircleStone Labs Non-Commercial License; LoRAs trained on it share it.
