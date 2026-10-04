# anima-trainer

Bridge + orchestration to finetune **CircleStone Anima** (2B anime text-to-image DiT)
with **[tdrussell/diffusion-pipe](https://github.com/tdrussell/diffusion-pipe)** — the
only trainer that natively supports Anima. It also trains **Sana** (diffusers format)
through the AbstractEyes diffusion-pipe fork; see [Sana](#sana-diffusers-format).

Anima's DiT backbone is NVIDIA Cosmos-Predict2-2B. This package reads an HF
`datasets`-format parquet repo (columnar, via pyarrow) into the image + `.txt`-sidecar
layout diffusion-pipe requires, organizes images into **subject buckets** with
semantic grouping, builds balanced dataset configs with anti-overtraining weighting,
models a training run as composable/sweepable config objects, and launches
diffusion-pipe's native multi-GPU deepspeed trainer.

> **License.** This tooling/code is **Apache-2.0** (see [LICENSE](LICENSE)). The **Anima
> model** itself and any weights you finetune from it are **non-commercial** — CircleStone
> NC + the NVIDIA Open Model License (Cosmos derivative). The permissive code license does
> not relax that: keep model artifacts and LoRAs non-commercial.

## Naming
- import package: `geolip_anima_trainer` · distribution: `geolip-anima-trainer` ·
  console command: **`anima`**

## Install

```bash
# from repo root, in a Python 3.12 venv
pip install -e .                  # package + light bridge deps (huggingface_hub, datasets, Pillow)
pip install -e ".[dev]"           # + pytest/ruff/build for development
pip install -e ".[similarity]"    # + sentence-transformers/sklearn for SEMANTIC subject grouping
```

> `[similarity]` unlocks real semantic grouping of sparse subjects (it pulls
> sentence-transformers + transformers + scikit-learn). Without it, grouping falls back
> to a numpy char-trigram backend then difflib — it never drops images, just groups them
> less semantically. See **Subject buckets** below.

torch is installed separately from a CUDA wheel index (it bundles its own CUDA runtime,
so your local toolkit version is irrelevant):

```bash
# Local smoke-test box (RTX 4090 / any Ada, Windows/Linux) — cu128 for target parity:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
# fallback if the cu128 wheel is unavailable for your OS/py:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

On the **Linux Blackwell (sm_120) target**, torch (≥2.7/cu128) and deepspeed come from
diffusion-pipe's own requirements; do **not** install flash-attn anywhere (broken on
sm_120 — diffusion-pipe uses SDPA).

### Cache locations (keep large downloads off a small system drive)

Model files (`anima download` ≈ 5.6 GB) and dataset/parquet caches go to the
HuggingFace cache, which can grow huge. By default it lands on the system drive — point
it at a roomy drive instead. On this machine the caches are set (persistent, user scope)
to spread load across drives:

| env var | value | what it redirects |
|---|---|---|
| `HF_HOME` | `F:\cache\huggingface` | HF hub + datasets cache (models, parquet) — on the large `F:` (Omega1) drive |
| `PIP_CACHE_DIR` | *(unset → C: default)* | pip wheel cache stays on `C:` |
| `TMP` / `TEMP` | `E:\cache\tmp` | install/download temp on the `E:` RAID array |

Set HF_HOME once (PowerShell, user scope) and reopen the shell:

```powershell
New-Item -ItemType Directory -Force 'F:\cache\huggingface' | Out-Null
[Environment]::SetEnvironmentVariable('HF_HOME', 'F:\cache\huggingface', 'User')
```

> **Moving an existing HF cache across drives:** the cache uses *relative symlinks*
> (snapshots → blobs). A plain cross-drive copy without symlink privilege (Developer
> Mode off) *materializes* them to 2× size. To relocate at ~1× without admin, copy
> `blobs/`+`refs/` as real files and rebuild each `snapshots/` entry as a **hardlink**
> to its blob (hardlinks need no privilege and share the data) — then validate with
> `huggingface_hub.scan_cache_dir()` before deleting the source.
>
> On the **Linux target**, apply the same idea — export `HF_HOME=/big/drive/hf` before
> `anima download` and the cache step so the latent/embedding cache doesn't fill root.

## Workflow

```bash
anima doctor                                   # verify env (torch/cuda/bf16, flash-attn absent, ...)

# 1. fetch model files (prints the [model] paths to paste into the lora toml)
anima download --dest models/anima --base base-v1.0

# 2. probe the source BEFORE extracting — which caption column is filled, gates populated?
anima inspect --repo AbstractPhil/diffusion-pretrain-set-ft1 --config qwen_90k

# 3. stream parquet -> image + .txt dirs (transient scratch). Add --no-age-filter /
#    --no-audit-filter if step 2 shows those gates are empty.
anima export --repo AbstractPhil/diffusion-pretrain-set-ft1 --configs qwen_90k \
    --out datasets/anima_qwen90k --caption-column caption_animetimm_json \
    --caption-format animetimm --route-by source --limit-per-concept 8000

# 4. balanced dataset.toml
anima build --root datasets/anima_qwen90k --out configs/anima_dataset.toml

# 5. (target box) precache latents/text-embeds (live GB/s progress), then train
anima cache --config configs/anima_lora.toml --repo-root . --progress --log-path runs/cache.log
anima train --config configs/anima_lora.toml --num-gpus 4
#   shared box: pin specific cards instead ->  --gpu-ids 0,1
#   preview the command anywhere (incl. Windows): add --dry-run
```

> **Caching is decode-bound, not GPU-bound** — `--cache_only` never loads the 2B DiT (only the
> VAE + Qwen-3 0.6B text encoder, forward-only), so **low VRAM is expected, not a bug**. The
> throughput lever is the image-**decode worker pool**: diffusion-pipe caps it at `min(8, cpu)`,
> so set **`map_num_proc`** (in the lora toml) to the box's core count, raise `caching_batch_size`
> (16+), and on a multi-GPU box add `--num-gpus N`. The `--progress` line tracks **shard bytes**
> (the live signal — the cache only commits its record count per 10 GB shard).

`anima init-config` copies the packaged `anima_lora.toml` / `anima_dataset.toml`
templates into `./configs` to edit.

> `anima export` (step 3) is the **generic** path: it renders the JSON captions into
> Danbooru-style tag strings and routes by `source`. For `diffusion-pretrain-set-ft1`
> the intended methodology is different — use **`anima subjects`** below.

## Subject buckets (recommended for `diffusion-pretrain-set-ft1`)

> 📄 **Deep dive:** [`docs/subject-bucketing.md`](docs/subject-bucketing.md) — the full write-up of the
> paradigm: what worked, what didn't, the diffusion-pipe internals, and a proposed HF-Hub-native
> dataset architecture that removes most of the directory-tree friction.

This dataset's captions are `task_1` JSON (`{"subjects":[...],"actions":[...],"setting":...}`)
meant to be trained **verbatim**, and its README warns that subject association must NOT be
learned via a cross-subject shuffle. So `anima subjects` does a **columnar** pyarrow read,
writes the JSON caption **as-is** into the `.txt` sidecar (plus `caption_animetimm_json` as a
second sample when present), and organizes images into **subject buckets**:

- **Bucket key = the dominant subject** (`subjects[0]`), normalized to a head-noun —
  each caption (vlm and animetimm) is bucketed by **its own** `subjects[0]`.
- **Sparse subjects are grouped, not dropped.** Similar weak buckets are merged by
  *semantic* similarity (`grp_boat`=[sailboat,boat,yacht], `grp_car`=[car,suv], …);
  ungroupable singletons pool into a weighted `misc_*` catch-all. Nothing is omitted.
- **Large buckets are split** so none exceeds a **data-dependent cap** (>10k imgs→1000,
  ≥1k→500, else 250; `--max-bucket-size` overrides). A bucket over the cap splits by the
  dominant subject's rarest **attribute** (`woman`→`woman·blonde_hair`, not `1girl`), then
  by **secondary subject**, then even-chunk — a hard guarantee no bucket exceeds the cap.
- **Distinct human subgroups stay separate** (`man`/`woman`/`player`/`person`/`guitarist`
  are never merged) — they're meaningful in Qwen-3.5's captioner grouping.
- **Weighting prevents overtraining.** `num_repeats` uses a diminishing-returns policy
  (`--alpha 0.5`): big buckets ~1–2×, sparse/grouped buckets capped at **8×** (the old
  equalize-to-largest policy would repeat a 5-image bucket **50×/epoch** → memorization).

```bash
# columnar extraction into semantic subject buckets (needs [similarity] for real grouping).
# --caption-mode before_after (default) -> separate vlm/ and animetimm/ trees + two tomls.
anima subjects --repo AbstractPhil/diffusion-pretrain-set-ft1 --config qwen_90k \
    --out datasets/anima_subjects --limit 1000 \
    --caption-mode before_after \
    --build-toml configs                       # writes dataset_vlm.toml + dataset_animetimm.toml

# zero-download similarity backend (reuses a model already cached): nomic
anima subjects ... --similarity-model nomic-ai/nomic-embed-text-v1
```

**Caption modes** (`--caption-mode`) — both `caption_vlm_json` (plain-english) and
`caption_animetimm_json` (booru tags) are trained:
- **`before_after`** (default, the first LoRA): `vlm/` + `animetimm/` trees, trained as **two
  sequential runs** — full VLM phase, then full animetimm phase resuming the VLM adapter.
- **`separate`**: same two trees but **one** dataset.toml — globally shuffled together.
- **`mixed`**: one image on disk + a `captions.json` carrying `[vlm, animetimm, joint]` — each
  image trains once with multiple prompts (physical dedupe), no pixel duplication.

```bash
# the first LoRA: VLM phase, then animetimm phase (resumes via [adapter].init_from_existing).
# diffusion-pipe can't phase-order inside one run (mandatory shuffle), so this is two runs.
anima train-before-after --lora-vlm configs/lora_vlm.toml \
    --lora-animetimm configs/lora_animetimm.toml --num-gpus N
```

Key flags: `--caption-mode {before_after,separate,mixed}`, `--max-bucket-size N` /
`--no-split`, `--prefer-attr-source {animetimm,vlm}`, `--limit N`, `--min-bucket-size`,
`--sim-threshold` (grouping tightness), `--min-final-group-size`, `--similarity-model`,
`--semantic-backend auto|sentence-transformers|trigram|difflib`, `--no-semantic`,
`--drop-small`. `--build-toml DIR` writes the per-mode dataset toml(s) into `DIR`.

## Notebooks

Two end-to-end notebooks under `notebooks/` drive the full `before_after` recipe (extract → build →
cache → two-phase train):

- **`anima_colab_prelim_train.ipynb`** — the **preliminary ~1k-image** run on Google **Colab**
  (ephemeral: one kernel restart after install, periodic HuggingFace checkpoint backup so the LoRA
  survives a disconnect). Start here to validate the methodology.
- **`anima_full90k_train.ipynb`** — the **full ~90k** run on a **persistent** Blackwell box (big-disk
  `DATA_ROOT`/`HF_HOME`, `limit=None` extraction, long-run config: `save_every_n_steps`, gradient
  accumulation, `activation_checkpointing=true` + larger `micro_batch`, `compile=true`; run **detached**
  via tmux/nohup and recover with `--resume_from_checkpoint` — disk is the durability, backup is optional).
- **`sana_colab_train.ipynb`** — a sequence of **Sana LoRA experiments** on Colab with no dataset needed:
  the stock model renders its own mood training sets captioned neutrally, every saved LoRA is loaded back
  through diffusers and scored on held-out subjects, and each experiment uploads into its own folder of
  [AbstractPhil/geolip-beatrix-sana](https://huggingface.co/AbstractPhil/geolip-beatrix-sana). See
  [Sana](#sana-diffusers-format) and `notebooks/README.md`.
- **`anima_colab_experiments.ipynb`** — the same experiment system on **Anima**
  (`geolip_anima_trainer.anima_runner.AnimaRunner`): e001 measures the stock model (mood words, a mood
  direction added to the conditioning before and after the LLM adapter, the conditioning norms), e012
  screens character attributes as sliders (judged by an anime tagger), then the LoRA arms, then e013-e015
  steer the image with Beatrix's own phrase features through a learned push, each into its own folder of
  [AbstractPhil/geolip-beatrix-anima](https://huggingface.co/AbstractPhil/geolip-beatrix-anima). See
  [Anima experiments](#anima-experiments-the-flavor-bed).

## Programmatic / sweeps

```python
import geolip_anima_trainer as anima

paths = anima.ModelConfig(transformer_path="models/anima/anima-base-v1.0.safetensors",
                          vae_path="models/anima/qwen_image_vae.safetensors",
                          llm_path="models/anima/qwen_3_06b_base.safetensors")
base = anima.single_concept_preset("datasets/anima_qwen90k/qwen_90k",
                                   output_dir="runs/anima", model=paths)

# emit 4 resolved (lora.toml, dataset.toml) pairs — no hand-editing
for tag, cfg_path in anima.sweep(base, ranks=[32, 64], lrs=[1e-5, 2e-5], runs_root="runs"):
    print(tag, cfg_path)   # then: anima train --config <cfg_path> --num-gpus N
```

`anima.validate()` enforces the Anima invariants (frozen adapter, tag-order, bf16,
no fp8/flash-attn/block-swap). See `CLAUDE.md` for the full domain brief and rules.

## Sana (diffusers format)

The same tooling trains LoRAs for **[Sana](https://github.com/NVlabs/Sana)** (NVlabs; Xie et al.
2024, [arXiv 2410.10629](https://arxiv.org/abs/2410.10629)): a linear-attention DiT with a 32x
autoencoder and a Gemma-2-2B text encoder; the 600M model at 512 px is small enough for quick
experiments. Sana's trainer side is model type **`sana`** in the
**[AbstractEyes diffusion-pipe fork](https://github.com/AbstractEyes/diffusion-pipe)**
(`models/sana.py`); upstream diffusion-pipe does not have it, and `anima train` / `anima cache`
refuse a checkout without it. Point `ANIMA_DIFFUSION_PIPE` at the fork, or clone it as
`external/diffusion-pipe`.

```bash
anima download --model sana --dest models/sana --variant 600m-512   # prints diffusers_path + the native resolution
anima init-config --model sana                                      # configs/sana_lora.toml + sana_dataset.toml
anima validate --config configs/sana_lora.toml
ANIMA_DIFFUSION_PIPE=/path/to/AbstractEyes/diffusion-pipe \
  anima train --config configs/sana_lora.toml --num-gpus 1          # --dry-run prints the command anywhere
```

```python
import geolip_anima_trainer as anima
model = anima.sana_model(anima.download_sana("models/sana", variant="600m-512"))
cfg = anima.single_concept_preset("datasets/my_concept", output_dir="runs/sana", model=model, resolution=512)
anima.render_train_toml(cfg, "configs/sana")
```

What differs from Anima:
- **One folder.** `[model] diffusers_path` points at a diffusers-format checkpoint (transformer +
  DC-AE autoencoder + Gemma-2 + tokenizer). `anima download --model sana` fetches one of the checked
  repos and skips the duplicate fp16/bf16/int4 weight files the loader never reads.
- **Captions** are natural language. The fork encodes them exactly as the diffusers `SanaPipeline`
  encodes prompts (lowercased, its instruction prefix, the 300-token selection; an empty caption is the
  pipeline's unconditional prompt), checked on the 600M 512px checkpoint: text embeddings identical,
  the transformer layer chain identical up to fp32 rounding, a training preview matching the stock
  pipeline's image.
- **Resolution** is the checkpoint's native size (512 or 1024); `validate()` warns otherwise, and preview
  sizes must be multiples of 32.
- **Presets**: plain Adam (weight decay 0) at 1e-4, the learning rate of the diffusers Sana LoRA example;
  `shift = 3.0`, the checkpoints' own sampling shift. No LLM adapter (`llm_adapter_lr` is Anima-only).
- **Output**: LoRAs are saved in diffusers format;
  `pipe.load_lora_weights("<run>/epochN", weight_name="adapter_model.safetensors")`.
- **Colab**: `notebooks/sana_colab_train.ipynb` + `geolip_anima_trainer.sana_runner.SanaRunner` run a
  whole LoRA end to end (install with the fork, data, configs, training, a diffusers load of the LoRA,
  an evaluation on held-out prompts, an optional private HF backup).
- **Status**: the parity checks above ran on Windows against the diffusers pipeline, and LoRA training has
  since run end to end on Colab (an RTX PRO 6000): six LoRA experiments, each loaded back through
  diffusers and scored (geolip-beatrix-sana e004-e009).
- **Side by side**: `s.run_sequence(parallel=3)` trains up to three arms at once on one card (each its own
  trainer process and rendezvous port; two arms on the same training set never overlap); the notebook
  evaluates finished arms while the others train.

> **Licences.** The Sana diffusers checkpoints are Apache-2.0; the bundled Gemma-2-2B-IT text encoder
> is under Google's [Gemma Terms of Use](https://ai.google.dev/gemma/terms) and
> [Prohibited Use Policy](https://ai.google.dev/gemma/prohibited_use_policy).

## Anima experiments (the flavor bed)

`geolip_anima_trainer.anima_runner.AnimaRunner` runs the Sana experiment system on Anima (the same
sequence machinery, verdict rules and repo layout; `anima_experiments.py` holds the Anima registry and
README text, `sana_experiments.Bed` what differs between the two). Notebook:
`notebooks/anima_colab_experiments.ipynb`; experiments repo:
[AbstractPhil/geolip-beatrix-anima](https://huggingface.co/AbstractPhil/geolip-beatrix-anima).

```python
from geolip_anima_trainer.anima_runner import AnimaRunner
a = AnimaRunner()          # Anima-Base v1.0, 768 px
a.setup()                  # GPU + the fork + the three model files
a.run_flavor_test()        # e001: the stock model
a.run_route_split()        # e020: which of the adapter's two readings carries the words
a.run_appended_split()     # e021: the same with the mood words after the scene (the query half, the source half, both)
a.run_query_dial()         # e022: a mood direction in the adapter's queries, alone and with a source token beside it
a.run_attribute_screen()   # e012: attribute sliders on the stock model
a.run_sequence()           # e002..e011: the LoRA arms
a.run_beatrix_connectors() # e013-e025: a push from Beatrix's phrase features, and its controls
```

- **Recipe** (from the model card): Anima-Base v1.0; LoRA rank 32 at 2e-5 (half and double as arms), the
  LLM adapter frozen; plain Adam, weight decay 0, **fp32 master weights** over the bf16 LoRA
  (`RunConfig.bf16_master_weights`, the fork's `MasterWeightsAdam`: without it, Adam steps on bf16 weights
  lose every update under half a bf16 step); prompts with the card's quality prefix and negative prompt.
- **Rendering** happens in the notebook process with the fork's own Anima model code (`AnimaPipe`: Qwen3
  0.6B, the LLM adapter, the DiT, the Qwen-Image VAE; the Euler flow sampler of the training previews,
  batched over prompts). Trained LoRAs (ComfyUI format) are applied by forward hooks (`LoraHooks`), so the
  stock weights never change. The first arm of a session compares this renderer with the trainer's own
  preview images (stock and with the LoRA) and records the pixel difference in the arm's `meta.json`.
- **e001** adds a mood direction to the conditioning at two sites: the Qwen3 states the LLM adapter reads,
  and the adapter's output the DiT cross-attends to (both 1,024 wide). Result: a push added equally to every
  token moves the mood only after the adapter.
- **e020** splits the adapter's two readings of a caption: its cross-attention reads Qwen3's states, and its
  blocks start from the caption's T5 token ids through the adapter's own word table. The mood prompt goes
  through one reading at a time (the other reads the neutral prompt) on e001's cells, to find which reading
  carries the words. Result: Qwen3's states alone carry nothing, the T5 ids carry most of the upbeat words and
  a third of the downbeat, and Qwen3's states add the rest only when the words' T5 ids are there to look them
  up (the cross-attention is a lookup). **e021** repeats the split with the mood words after the scene, so
  the caption's positions agree in both readings, and reads what the source half adds to the query half on
  the downbeat words. **e022** adds a mood direction to the adapter's queries (before its blocks), alone and
  with one source token appended after the caption (a bare direction, or Qwen3's own state for the mood
  words); the same direction spread over every Qwen3 token is the control.
- **e012** screens six character attributes (hair length, hair colour, eye colour, chibi proportions, age
  for adults only, style) as tag pairs on eight characters: does the tag move the attribute, does its
  direction added after the adapter act as a slider, and does each slider leave the other attributes alone
  (cross-talk, and the overlap of the directions). The judge is the
  [WD EVA02-Large tagger v3](https://huggingface.co/SmilingWolf/wd-eva02-large-tagger-v3).
- **e013-e015** condition the image on a second source: a push W f + b added after the adapter, where f is
  Beatrix's feature vector for a mood phrase (read once from her checkpoint and kept in the data repo, so
  Beatrix never runs in the notebook). W and b start at zero and are the only trained numbers: Anima's own
  objective (`AnimaPipe.train_loss`: the fork's `prepare_inputs` + loss) on the LoRA arms' first-draw images,
  plain Adam with the fan-in rule's learning rates, everything else frozen; the push goes on both guidance
  branches at sampling, as a LoRA acts. e014 is the same on an untrained Beatrix (it must fail on phrases
  never trained on), e015 a free learned vector per mood (the reference). Scored on the held-out scenes.
  e013 read NOT LEARNED (the map's phrase-dependent part barely moved in 720 steps on the raw 4,096 features),
  so e016 / e017 re-run e013 / e014 on the features whitened onto the training phrases' top 16 principal
  components (fit on the training phrases only), with the map's rate set so its mood contrast moves at the
  free vector's pace. e016 moved every group the right way, unseen phrases included, but each phrase's own
  position in those 16 components pushed the image more than its mood did (and the neutral phrases moved too),
  so e018 / e019 reduce the features to a slider value: the phrase's position on the axis from the gloomy to
  the cheerful training phrases' centres (at -1 and +1) and on the axis toward the neutral training phrases'
  centre (2 numbers, both axes fit on the training phrases only). e018 moved the image for cheerful phrases,
  unseen ones too, and not for gloomy ones; e019 (the untrained trunk) came out the other way round. A linear
  map of one slider value ties both sides to one direction (gloomy phrases get the mirror of the cheerful
  push), and the image model's downbeat direction is not its upbeat direction negated (e015's free vectors are
  nearly orthogonal), so one side comes out strong and the other weak. e023 / e024 split the slider value by its sign,
  [max(a, 0), max(-a, 0), n], so each side gets its own direction (Beatrix, then the untrained trunk); e025
  maps it to [e^a, e^-a, n], both sides on for every phrase (no dead zone).
- **Progress:** every long step prints the size of the job, then done / total, time spent and time left.

> **Licence.** Anima's weights are under the CircleStone Labs Non-Commercial License (a derivative of
> NVIDIA Cosmos-Predict2-2B, NVIDIA Open Model License); LoRAs trained on it share those terms.

## Targets at a glance
| | Local (smoke-test) | Training target |
|---|---|---|
| GPU | RTX 4090, sm_89, 24 GB, Windows | RTX PRO 6000 Blackwell, sm_120, 96 GB, Linux, 1–N GPUs |
| Role | install + import + bridge + `--dry-run` | full extract + cache + multi-GPU train |
| torch | cu128 (parity) | cu128 / ≥2.7 (required for sm_120) |
| deepspeed/diffusion-pipe | source only, never runs | installed + runs |
