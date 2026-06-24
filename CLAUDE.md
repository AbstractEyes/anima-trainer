# Anima Trainer — Repo Brief (CLAUDE.md)

Finetune **CircleStone Labs Anima** (2B anime text-to-image; DiT backbone is NVIDIA
Cosmos-Predict2-2B) using **tdrussell/diffusion-pipe** — the only trainer that
supports Anima. We *bridge* an HF `datasets`-format parquet repo into the directory
layout diffusion-pipe requires; we do **not** switch trainers.

## Implementation status (this repo is now a package)
The original flat `scripts/`+`configs/` brief (below) is implemented as the
installable package **`geolip_anima_trainer`** (console command **`anima`**):

- `config.py` — composable dataclasses + presets + `apply_overrides` + `sweep` +
  `validate` (the "elemental construction" engine). The Anima invariants live in
  `validate()`.
- `launch.py` — `build_plan`/`launch`: builds the `deepspeed --num_gpus=N train.py
  --deepspeed --config ...` command (native multi-GPU), guarded so it never execs
  on Windows. `launch(monitor=...)` adds the non-blocking cache-progress path;
  `rewrite_init_from_existing`/`latest_epoch_dir` drive the before_after handoff.
- `doctor.py` — environment diagnostics (`anima doctor`).
- `download_anima.py`, `hf_to_diffusion_pipe.py`, `build_multiconcept_dataset.py` —
  the original bridge scripts (still runnable standalone; now also importable).
  `build_multiconcept_dataset.py` also holds the **`dampened_repeats`** weighting and
  the `online_captions` toml emission (for MIXED mode).
- `subject_buckets.py` — **`anima subjects`**: columnar pyarrow extraction into
  semantic SUBJECT buckets with the JSON caption trained verbatim (the real
  methodology for `diffusion-pretrain-set-ft1`). Holds the 3 caption modes,
  attribute splitting, and `build_mode_tomls`. See "Subject buckets" below.
- `subject_similarity.py` — 3-tier similarity backend (sentence-transformers →
  numpy char-trigram → difflib) behind the optional `[similarity]` extra.
- `cache_monitor.py` — **`anima cache --progress`**: live %-done + ETA from the cache
  SQLite `metadata.db` COUNT(*) (the cache is sharded blobs, not per-image files).
- `api.py` / `cli.py` — the Python API and the `anima` CLI.
- `templates/anima_lora.toml`, `templates/anima_dataset.toml` — packaged templates;
  copy into `./configs` with `anima init-config`.

Two extraction paths:
- **`anima subjects`** (recommended for this dataset) → subject buckets, JSON caption
  verbatim, semantic grouping + attribute splitting + anti-overtraining weighting.
- **`anima export`** (generic) → renders JSON captions to tag strings, routes by `source`.

Workflow: `anima inspect` → `anima subjects --caption-mode before_after --build-toml configs`
→ (target) `anima cache --progress` → `anima train-before-after` (first LoRA) or
`anima train --num-gpus N`. See README.md for the full command sequence.

## Environment (fixed facts — do not re-derive)
- **Target GPU:** RTX PRO 6000 Blackwell, 96 GB, **sm_120**, one or many on a shared box.
- **bf16 throughout. No fp8** (e4m3 degrades this lineage; 96 GB makes fp8 pointless).
- **Do NOT install flash-attn** — broken on sm_120. diffusion-pipe runs on **SDPA**.
- No block swapping, no activation checkpointing (VRAM abundant).
- Python 3.12. Blackwell needs **torch ≥ 2.7 / cu128**.
- **Local box is an RTX 4090 (sm_89, Windows) — smoke-test only, never a full train.**

## Model files (circlestone-labs/Anima, split_files/)
| file | role | size |
|---|---|---|
| `diffusion_models/anima-base-v1.0.safetensors` | DiT base (recommended) | 4.18 GB |
| `text_encoders/qwen_3_06b_base.safetensors` | Qwen-3 0.6B base text encoder | ~1.2 GB |
| `vae/qwen_image_vae.safetensors` | Qwen-Image VAE | ~250 MB |

Fetch with `anima download --dest models/anima --base base-v1.0` (huggingface_hub,
never wget). It prints the exact `[model]` paths for `configs/anima_lora.toml`.

## Dataset
- Repo: `AbstractPhil/diffusion-pretrain-set-ft1`; config: **`qwen_90k`** (= the
  `data/sdxl_qwen_phase0/` folder, 83.0K rows, high-res). Uniform 17-col schema.
- Captions are **`task_1` JSON**: `caption_vlm_json` and `caption_animetimm_json`
  are `{"subjects":[{"name","attributes":[...]}],"actions":[...],"setting":...}`;
  `captions_source_json` is `{caption_kind: plain_text}`. The dataset's own
  README/CLAUDE.md say to train the JSON **verbatim** (NOT rendered to tags) — that is
  what `anima subjects` does. `anima export` is the generic render-to-tags fallback.
- A `subjects[]` entry may be a `{"name":...}` dict **OR a bare string** — parse both.
- Quality gates in-schema: `audit == "approved"`, `age_classifier_pass == True/None`.
  For qwen_90k: `audit` is always approved (keep), `age_classifier_pass` is `None`
  (unpopulated → must disable the age gate). **`anima inspect` a new config first.**
- ⚠️ **Never** put `extra_json` / `celeb_name_raw` (IMDB) into a caption — takedown-only,
  never a training signal. The narrow column read in `subject_buckets.py` excludes it.

## Non-negotiable rules
- **`llm_adapter_lr = 0`** (adapter frozen). High knowledge density, degrades easily.
  Only raise it (≤5e-6) for genuinely-new concepts, A/B'd against a frozen baseline.
  `validate()` hard-errors above 5e-6.
- **Anima is tag-order sensitive**: quality → subject → character → series → @artist →
  general. `shuffle_caption = false` (unless `keep_tokens` protects the leading block).
- **License.** This repo's *code* is Apache-2.0 (see LICENSE). The *Anima model* and any
  weights finetuned from it are **non-commercial** — CircleStone NC + NVIDIA Open Model
  License (Cosmos derivative). Any LoRA/finetune is a derivative — NC only; the permissive
  code license does not relax that.
- diffusion-pipe input = directories of image files + matching `.txt` sidecars; it
  uses the `datasets` library only as an internal latent/embedding cache.

## Open decisions — RESOLVED from `anima inspect qwen_90k` (2026-06-23, 64-row sample)
- **Caption column → `caption_vlm_json` (`--caption-format vlm`).** `caption_animetimm_json`
  is the literal string `__PARSEFAIL__` for this slice (upstream tagging failed);
  `caption_vlm_json` renders clean booru-style tags. `captions_source_json` is natural
  language (fallback only).
- **Age gate → MUST pass `--no-age-filter`.** `age_classifier_pass` is unpopulated
  (`None`) here; the default gate would drop **every** row (0/64 passed).
- **Audit gate → keep ON.** `audit` was `"approved"` for all sampled rows.
- **Schema note:** `image` is a `{bytes, path}` dict (not a decoded PIL Image); the
  bridge's `_to_pil` handles this. `conditioning_image`/`mask` were `None` in the sample.
- **Single vs multi-concept:** the stream routes by `source` and is NOT uniform (the
  first shard was `sdxl_qwen_phase0`) — so this is a **multi-concept** run
  (`anima.multi_concept_preset(...)`, rank 64, balanced). Re-inspect a larger sample
  on the target to enumerate the full `source` distribution before final balancing.

  Example generic extract (render-to-tags path):
  ```
  anima export --repo AbstractPhil/diffusion-pretrain-set-ft1 --configs qwen_90k \
      --out datasets/anima_qwen90k --caption-column caption_vlm_json \
      --caption-format vlm --route-by source --no-age-filter --limit-per-concept 8000
  ```

## Subject buckets — how it was built (and how to adapt it for new datasets)

`anima subjects` (`subject_buckets.py` + `subject_similarity.py`) is the real path for
this dataset. Read this before changing it.

### Why it exists / what it does (the methodology, with the WHY)
1. **Caption = the `caption_vlm_json` string VERBATIM** in the `.txt` sidecar (the
   dataset is meant to teach JSON-prompt conditioning), plus `caption_animetimm_json`
   as a second sample. *Do not render to tags here.* Each caption is bucketed by
   **its own** `subjects[0]` (the vlm sample by the vlm subject, the animetimm sample by
   the animetimm subject) — see "Caption modes" below for how the two are organized.
2. **Columnar read, not row-by-row.** `hf_hub_download` the shard(s) → `pyarrow
   ParquetFile.iter_batches(columns=[...])` → write the image's **raw bytes** straight
   to disk (no PIL decode/re-encode). The earlier `datasets`-streaming + PNG-encode
   path was ~1 img/s; this is ~network-bound. Download shards **lazily** (stop at
   `--limit`) — the full config is >100 GB; never pre-fetch all shards.
3. **Bucket = the dominant subject** (`subjects[0]`, normalized to a head-noun via
   `normalize_subject`: lowercase, strip articles, last word, singularize).
4. **Group the sparse tail by SEMANTIC similarity, don't drop it** (`keep_small=True`
   → leftovers pool into weighted `misc_*`, never omitted). Grouping uses
   **agglomerative average-linkage** over cosine distance (`sklearn`), NOT
   single-linkage connected-components — CC *chains* dense embeddings into one giant
   blob (we hit a 961-image `grp_bed` doing that; agglomerative gives clean groups like
   `grp_boat`=[sailboat,boat,yacht]).
5. **Keep distinct human subgroups SEPARATE** (`man`/`woman`/`player`/`person`/
   `guitarist` never merge) — Qwen-3.5-0.8B distinguishes them on purpose. The guard is
   an explicit `_HUMAN_SEED` (~12 anchors) + similarity propagation, **NOT** a cosine
   threshold: `man`/`woman` cosine is ~0.5–0.75, so a threshold alone would wrongly
   merge them. Large + human buckets are `protected` and removed from all merge
   candidates.
6. **Weighting prevents sparse overtraining** (`dampened_repeats` in
   `build_multiconcept_dataset.py`): `num_repeats = round((top/imgs)**(1-alpha))`,
   capped at `max_repeats` and `cap_mult*top` effective. `alpha=0.5` (sqrt) is the
   default; `alpha=0` is the legacy equalize-to-largest policy that repeats a 5-image
   bucket ~50×/epoch (memorization). New policy caps every bucket at ~8×.
7. **Split oversized buckets** (`split_oversized_buckets`) so no bucket exceeds a
   **data-dependent cap** (`max_bucket_size`: >10k imgs→1000, ≥1k→500, else 250;
   `--max-bucket-size` overrides). This is a 3rd tier AFTER grouping, on a **disjoint
   domain** (grouping touches `<min_bucket_size`; splitting touches `>cap`, far apart).
   An oversized bucket splits by its dominant subject's **rarest attribute**
   (animetimm-preferred, booru-style; `_ATTR_STOP` drops `1girl`/`solo`/… so it defaults
   to `blonde_hair` not `1girl`) → **secondary subject** (`man·with_dog`) → even-chunk
   (hard guarantee). One attr per image (rarest) = a true partition. Each sub-bucket is
   its own dir → weighted by its own count via `dampened_repeats`. Splitting runs
   **per caption stream**, keyed `(idslug, stream)`, because vlm/animetimm trees differ.

### Caption modes (`--caption-mode`, the two captions trained together)
Both `caption_vlm_json` (plain-english) and `caption_animetimm_json` (booru tags) train.
`export_subject_buckets` routes by `CaptionMode`:
- **`before_after`** (DEFAULT, the first LoRA): `out/vlm/<subj>` + `out/animetimm/<subj>`
  trees + **two** dataset tomls. Trained as **two sequential runs** via
  `anima train-before-after` (full VLM phase, then full animetimm phase).
- **`separate`**: same two trees, but ONE dataset.toml → diffusion-pipe globally shuffles.
- **`mixed`**: ONE image inode + a `captions.json` `{file: [vlm, animetimm, joint]}`
  (`online_captions=true`) → each image trains once with N prompts (physical dedupe).

**WHY before_after is two runs** (verified in `external/diffusion-pipe`, file:line): the
trainer builds ONE `Dataset` + ONE LR schedule, and shuffle is MANDATORY (3 hardcoded-seed
layers, `dataset.py:211,354,972`) — so "VLM strictly before animetimm" is **impossible
inside one run**. `train_before_after` chains run A → run B, handing the adapter off via
`[adapter].init_from_existing` (there is NO CLI flag — `launch.rewrite_init_from_existing`
writes a temp toml pointing at phase-1's latest `epoch{N}` dir).

### Cache facelift (`anima cache --progress`) + caching throughput
`cache_monitor.py` prints a live `[cache] N.NN GB cached · MB/s · latents/text · elapsed` line.
**Primary signal = total `shard_*.bin` BYTES** (`cache_bytes`), NOT the metadata `items`
COUNT(*): diffusion-pipe writes each item to the `.bin` immediately (`cache.py:115`) but only
**commits the items table when a 10 GB shard finalizes or a pass ends** (`cache.py:106`,
`dataset.py:91`) — so COUNT(*) is 0 for long stretches even while caching is busy (this was a
real "stuck at 0" bug; `immutable=1` made it worse and was removed). Record counts are shown
as a coarse phase marker. `launch(monitor=...)` runs the subprocess non-blocking; pair with
`--log-path` so the line isn't buried in tqdm.

**Throughput diagnosis (verified in `external/diffusion-pipe`, file:line):** `--cache_only`
is **decode/plumbing-bound, NOT GPU-bound**. The 2B DiT never loads (`train.py:511-512` quits
before `:517`), so only the VAE (~0.25 GB) + **Qwen-3 0.6B** text encoder (~1.2 GB) are
resident, forward-only — **low VRAM is correct, not a bug**. The wall-clock bottleneck: each
decode worker does `queue.put` then BLOCKS on the GPU round-trip (`dataset.py:1098`), and the
pool is capped at `min(8, cpu)` single-threaded workers (`dataset.py:33,1055`) decoding images
with PIL — so the GPU idles. Levers, ranked: (1) **`map_num_proc`** = the decode-worker count
(NOW exposed in `RunConfig`/template; set to the box's core count — the highest-impact knob),
(2) **`caching_batch_size`** 8→16+ (96 GB has headroom), (3) **`--num-gpus N`** to shard the
encode (raise `map_num_proc` to ≥N×8 first — only rank 0 spawns the producer pool). dtype stays
bf16 (Anima invariant). The Qwen-3 `max_length=512` fixed padding is the biggest GPU item but
the GPU is starved, so it's not the wall-clock bottleneck.

### Backend tiers (optional `[similarity]` extra)
`make_sim_fn` picks best-available, logs the tier: **sentence-transformers**
(`all-MiniLM-L6-v2`, ~90 MB; or `--similarity-model nomic-ai/nomic-embed-text-v1` =
0 download, already cached) → **numpy char-trigram** (zero-dep, morphological only —
can't group `truck`~`car`) → **difflib**. Real semantic grouping needs the extra
(`pip install -e ".[similarity]"`); without it nothing drops, it just groups less well.
The extra downgrades `huggingface_hub` 1.x→0.36 (pinned `<1`) — harmless, verified.

### How to ADAPT for a new / different dataset
- **Inspect first** (`anima inspect <config>` or a small pyarrow read): confirm the
  caption column name + that it's `task_1`-shaped JSON; whether `subjects[]` items are
  dicts or strings (`_subject_name` handles both); the image column form
  (`{bytes,path}` here — `_img_bytes`/`_sniff_ext` handle raw bytes + format sniff);
  and which gates are populated (`require_audit_approved` / `require_age_pass`).
- **Different caption schema** (no `subjects` field, or non-JSON): adjust
  `dominant_subject` / `normalize_subject`; the columnar read + bucketing + weighting
  scaffold is reusable as-is.
- **`misc_*` too large** (very diverse sources make most dominant subjects singletons):
  options, smallest-effort first — lower `--min-final-group-size`; raise `--sim-threshold`
  modestly (looser grouping, watch for over-merge); implement **secondary-subject
  routing** (route a singleton-dominant image by a *secondary* subject that matches a
  real bucket — the "hierarchy of fulfillment" idea, the biggest lever, not yet built);
  or stop using `misc_other` as the balancing `top` reference.
- **Tuning knobs** (all on `SubjectBucketConfig` / CLI): `caption_mode`,
  `max_bucket_size` (split cap; `None`=data-dependent), `prefer_attr_source`,
  `min_bucket_size` (large/protected floor), `human_min_size`, `sim_threshold`
  (preset-overridden per backend: ST 0.58, trigram 0.50, difflib 0.60),
  `min_final_group_size`, `balance_alpha`, `cap_mult`, `max_repeats`.
  `balance_alpha`/`cap_mult` are sweepable via `apply_overrides`.
- **Attribute splitting carries per-image data** (`ImageRecord` in pass 1: vlm/anime
  subjects + `attrs` + `secondary`). A new dataset with different attribute shapes only
  needs `normalize_attr`/`_ATTR_STOP` tuned; the tier A→B→C split is schema-agnostic.
- **Validation:** `validate()`'s ">3× effective spread" warning is gated on
  `balance_alpha == 0` — under dampening a wide spread is intended, don't re-enable it.
- **F: cache space:** dataset shards cache to `HF_HOME` (F:, ~25 GB free behind the
  model cache). One shard ≈ 1.3 GB; a full 56-shard extraction (~106 GB) won't fit —
  do full runs on the target box or point the cache at a roomier drive.
- **Tests:** `tests/test_subjects.py` covers the planner with a **stub `sim_fn`** (no
  model/download) — add cases there; keep the legacy `plan_buckets` shim green.
- **Design provenance:** the methodology came from a 3-agent design workflow
  (similarity method / human-aware clustering / weighting) + adversarial synthesis; the
  CC→agglomerative fix was a course-correction the clustering agent had flagged as a risk.
