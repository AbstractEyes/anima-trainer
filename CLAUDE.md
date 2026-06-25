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
- **Do NOT install flash-attn** — broken on sm_120 (that's the standalone Dao-AILab package).
  diffusion-pipe runs on **PyTorch SDPA**, whose flash/mem-efficient kernels ARE enabled for
  sm_120 in torch ≥2.7/cu128 (PyTorch PR 145602). The 2B DiT self-attention is **maskless**
  (`cosmos_predict2_modeling.py:303`) → efficient backend, **not** the math fallback.
- No block swapping (VRAM abundant). **`activation_checkpointing` off by default** — but it's a
  VRAM↔speed TRADE, not a free win: OFF keeps all 28 DiT blocks' activations resident (≈89 GB at
  1024² micro_batch 4 — that VRAM is **activations, not the 4 GB model**; expected, not a leak) and
  is ~25-33% FASTER. Turn it ON only to FREE VRAM for a bigger `micro_batch`/higher res.
- **Training perf (1024², single Blackwell):** ~2.8 samples/s at micro_batch 4 is **compute-bound-
  normal**, not a fallback. The one real throughput lever is **`compile = true`** (torch.compile,
  ~+10-25%; amortizes its one-time + per-AR-shape compile over a real run — net-negative for smoke
  tests). Bigger `micro_batch` grows EFFECTIVE batch, not samples/s (use grad_accum for that at no
  VRAM cost). Both `compile` and `activation_checkpointing` are exposed on `RunConfig`/template.
- Python 3.12. Blackwell needs **torch ≥ 2.7 / cu128** (cu130/2.9 only refreshes the cuDNN path — optional).
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
   - **At 90k scale it must stay fast + observable** (a real "very slow" report): shards
     download **concurrently** (`download_workers`, default 8 — over-fetch bounded to
     `download_workers` on early `limit` exit; full run = all shards, N at a time); every
     phase prints `[subjects] …` progress (shard k/N, scanned, planning over N subjects,
     wrote X/total — `print`, not `log.info`, so it shows in a notebook where the root
     logger defaults to WARNING). `progress_every` (default 20000) sets the cadence.
   - **The clustering hot path is vectorized.** `_cluster_side`'s submatrix extract was an
     O(M²) **Python** double-loop of numpy *scalar* lookups (`[[float(S[idx[a],idx[b]]) …`)
     — fine at M~hundreds (1k run) but tens of millions of slow ops at 90k. Now a single
     `S[np.ix_(sidx, sidx)]` fancy-index. The remaining O(M²) cost is sklearn's
     `AgglomerativeClustering` itself (C, on the M×M precomputed matrix) — watch the
     "planning over N unique subjects" line; if N is huge that's the next wall. **Without
     the `[similarity]` extra the backend falls to difflib, whose `sim_fn(subjects, subjects)`
     is an O(N²) `SequenceMatcher` double-loop — death at 90k; install `[similarity]`.**
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
(NOW exposed in `RunConfig`/template; set to the box's core count — the highest-impact knob;
default `min(8, cpu)` at `dataset.py:33`), (2) **`--num-gpus N`** to shard the encode (raise
`map_num_proc` to ≥N×8 first — only rank 0 spawns the producer pool). dtype stays bf16 (Anima
invariant). The Qwen-3 `max_length=512` fixed padding is the biggest GPU item but the GPU is
starved, so it's not the wall-clock bottleneck.

⚠️ **`caching_batch_size` is NOT a throughput lever — keep it small (8, ≤16).** It is read raw
(`train.py:424`, default 1, no clamp) and applied to BOTH passes (`dataset.py:1111,1124`). It only
sets how many images a worker must decode+stack into ONE VAE forward before the first shard is
written (`dataset.py:1090,1093` overrides it to the realized batch len; tqdm ticks once per
*batch*, `dataset.py:151,154`). A big value (e.g. 256) just (a) **delays the first shard / makes
warm-up look hung**, (b) **risks a VAE/text-encoder OOM** from one stacked 256-batch, (c) coarsens
resume. The qwen example uses 8. The earlier "8→16+, 96 GB has headroom" note was wrong — headroom
doesn't help a knob that only inflates first-shard latency.

**Why caching is silent for the first 1–2 min (expected, not a hang):** `--cache_only` runs a
**metadata pass** that `Image.open`s *every* file for AR bucketing (`dataset.py:1058` strictly
before `:1111`; size read at `:797-798`) BEFORE any latent encode — zero shard bytes by design.
Two facelifts make it legible: `launch.env_prefix()` sets **`PYTHONUNBUFFERED=1`** (diffusion-pipe
uses only `print()`+tqdm with no `--verbose`; on a non-TTY pipe the `print()` markers block-buffer
and tqdm auto-disables — unbuffering streams the markers), and `cache_monitor.make_monitor(log_path=)`
**tails the log's last line** during the no-shards-yet phase via `last_log_line` (splits on `\r` for
tqdm). Genuine-hang escape hatch: HF `datasets` map-over-fork can hang (`dataset.py:1052-1053`); if
nothing grows after ~10–15 min, lower `map_num_proc` (→2).

⚠️ **diffusion-pipe cache CRASH→HANG bug (`dp_compat.py` fixes it).** diffusion-pipe writes per-AR
bucket metadata via `Dataset.map(cache_file_name=<dir>/ar_frames_*/metadata/…arrow)` but only
creates `ar_frames_*/` (`dataset.py:410`), never its `metadata/` subdir (`:429`). HF `datasets`
(confirmed 2.21) opens the map temp file with `NamedTemporaryFile(dir=dirname(cache_file_name))` and
does **not** mkdir the parent → `FileNotFoundError`. The directory-level metadata path only survives
because an earlier `save_to_disk` side-effect-creates its dir. Worse, the crash is in a **forked
child** (`_cache_fn`, `dataset.py:1047`) that never puts its queue sentinel, so the parent blocks on
`queue.get()` (`:1184`) **forever** — a silent unkillable hang (this is the real "no exception
catching" the user hit; many small AR buckets trip it). Two-layer fix, both ours:
(1) **`dp_compat.patch_datasets_map_makedirs`** wraps `Dataset.map` to mkdir `dirname(cache_file_name)`
first (universally safe, idempotent). It's auto-applied in the subprocess via a bundled
`_dp_compat/sitecustomize.py` that `launch.env_prefix()` puts FIRST on `PYTHONPATH` (Python imports
`sitecustomize` at startup, before diffusion-pipe forks). The shim is self-contained (no package
import) and shipped as package-data.
(2) **`cache_monitor` hang-detection**: a Python traceback in the log + stalled bytes for
`stall_limit` (2) ticks ⇒ print the traceback and **terminate** the wedged process (→ launch raises),
instead of polling a hung `poll()==None` forever. So even a *different* child crash fails fast/loud.

### Cache preservation + resumable accumulation on HF Hub (`cache_sync.py`)
Ephemeral Colab loses the local cache on a reset (an 8 h cache vanished once). `cache_sync.py` +
`api.cache_push`/`cache_pull` + `anima cache-push`/`cache-pull` push the cache to a **private HF
*dataset* repo** as retrievable shards and resume it next session. Load-bearing mechanics (verified
in `external/diffusion-pipe/utils/{cache,dataset}.py`):
- **Mid-run copy → restore is SAFE.** Items commit to `metadata.db` only at a 10 GB shard finalize or
  pass end (`cache.py:99-106`); the in-progress `shard_<n>.bin` is re-opened `'wb'` (truncated, `:91`)
  on resume — a periodic push loses **at most the current ≤10 GB shard**, never corrupts a committed one.
- **The latent fingerprint** (one `fingerprint(value)` row per leaf `metadata.db`, `cache.py:45-56`)
  embeds **absolute image paths**; on mismatch the cache is **`clear()`ed** (`:50-52`, logs `[CACHE]
  Fingerprint changed`). ⇒ resume needs the dataset restored to an **identical absolute `DATA_ROOT`**
  with byte-identical files. So we **freeze the dataset** (ship it with the cache; re-extraction could
  re-cluster buckets → new fingerprint → wipe) and **pin `DATA_ROOT`**. `--trust_cache` (now passed
  through `launch.py` → `argv()`; was missing) loads the restored `metadata/*.arrow` without
  re-validation but does **not** bypass this check.
- **Periodic push** rides `cache_monitor`'s `on_update` hook (`cache_sync.make_periodic_pusher`):
  pushes the **cache subtree only** (frozen images uploaded once via `cache_push(include_dataset=True)`)
  on each committed-record bump (a finalize = a new restorable checkpoint) or every `backup_interval`
  (1800 s), **plus a guaranteed final push even if the run crashes** (the "8 h lost" case, now
  recoverable). A failed push is logged, never kills the run. For `before_after` pass
  `backup_root=SUBJECTS_ROOT` (the parent of vlm/+animetimm/) so periodic pushes match the freeze
  layout — `out_root_of(one toml)` only yields that tree's root.
- **Workflow:** session 1 = extract → `cache_push` (freeze) → `cache --backup-repo …` (periodic);
  session N = `cache_pull` (restore dataset+partial cache) → `cache --trust-cache --backup-repo …`
  (resume + keep pushing). `upload_folder` is incremental (only new shards re-upload). ⚠️ DATA_ROOT
  drift is the #1 footgun. The notebook `anima_full90k_train.ipynb` §6/§8 implement the first-vs-resume
  branch. **2× cost stands** under `before_after` (latents cached per tree over hardlinked pixels);
  MIXED mode would halve it but changes the validated recipe — not done.

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
