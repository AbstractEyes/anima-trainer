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

Sana's weights are Apache-2.0; its Gemma-2-2B-IT text encoder is under the Gemma Terms of Use.
