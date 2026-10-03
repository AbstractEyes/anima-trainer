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
A **Sana LoRA** on a Colab GPU runtime (the RTX PRO 6000 is the target; the 600M recipe fits much
smaller cards), trained through model type `sana` in the
[AbstractEyes diffusion-pipe fork](https://github.com/AbstractEyes/diffusion-pipe). Thin shell over
`geolip_anima_trainer.sana_runner.SanaRunner`: bootstrap (`anima_colab.install(...,
dp_url=anima_colab.DP_FORK_URL)` clones the fork at `external/diffusion-pipe-fork`; one restart), then
`setup()` → `prepare_dataset()` → `build_configs()` → `train()` → `evaluate()` → `backup()`.

- **No dataset needed by default:** the stock Sana 600M 512px model renders 192 upbeat images
  (24 subjects × 8 seeds) captioned with the neutral prompt "a photo of <subject>"; 8 more subjects
  never enter training.
- **Recipe:** rank 32, plain Adam (weight decay 0) at 1e-4, 10 epochs at micro-batch 4 (480 steps),
  20 warmup steps, a save + 4 preview images every 2 epochs, `shift = 3.0`.
- **Evaluation:** the saved LoRA is loaded through diffusers (`load_lora_weights`) and the 8 held-out
  subjects × 4 seeds are rendered without the LoRA and with it at scale 0.5 and 1; a CLIP mood judge
  (upbeat phrases minus downbeat phrases) gives the verdict: **FLAVOR LORA** (mean above 0, at least 75%
  of the pairs positive, above 3 standard errors), **NO EFFECT** (within 2 standard errors of 0, or under
  60% positive), else **MIXED**. Writes `eval/eval.json` + `eval/sheet.jpg`.
- `HF_TOKEN` is optional (the models are public); `backup()` uses it to push the LoRA + the evaluation
  to a private model repo (`<user>/sana-mood-lora` unless `backup_repo=` is set).
- **Your own data:** `SanaRunner(source="folder", dataset_dir=..., preview_prompts=[...])`.

Sana's weights are Apache-2.0; its Gemma-2-2B-IT text encoder is under the Gemma Terms of Use.
