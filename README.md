# anima-trainer

Bridge + orchestration to finetune **CircleStone Anima** (2B anime text-to-image DiT)
with **[tdrussell/diffusion-pipe](https://github.com/tdrussell/diffusion-pipe)** — the
only trainer that natively supports Anima.

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

# 5. (target box) precache latents/text-embeds (with a live progress + ETA bar), then train
anima cache --config configs/anima_lora.toml --repo-root . --progress --log-path runs/cache.log
anima train --config configs/anima_lora.toml --num-gpus 4
#   shared box: pin specific cards instead ->  --gpu-ids 0,1
#   preview the command anywhere (incl. Windows): add --dry-run
```

`anima init-config` copies the packaged `anima_lora.toml` / `anima_dataset.toml`
templates into `./configs` to edit.

> `anima export` (step 3) is the **generic** path: it renders the JSON captions into
> Danbooru-style tag strings and routes by `source`. For `diffusion-pretrain-set-ft1`
> the intended methodology is different — use **`anima subjects`** below.

## Subject buckets (recommended for `diffusion-pretrain-set-ft1`)

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

## Targets at a glance
| | Local (smoke-test) | Training target |
|---|---|---|
| GPU | RTX 4090, sm_89, 24 GB, Windows | RTX PRO 6000 Blackwell, sm_120, 96 GB, Linux, 1–N GPUs |
| Role | install + import + bridge + `--dry-run` | full extract + cache + multi-GPU train |
| torch | cu128 (parity) | cu128 / ≥2.7 (required for sm_120) |
| deepspeed/diffusion-pipe | source only, never runs | installed + runs |
