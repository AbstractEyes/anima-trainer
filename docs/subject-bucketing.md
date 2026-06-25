---
title: "Subject Bucketing: Teaching a Diffusion Model New Prompt Languages Without Forgetting"
thumbnail: ""
authors:
  - user: AbstractPhil
tags:
  - diffusion
  - dataset-curation
  - lora
  - anima
  - cosmos-predict2
  - diffusion-pipe
---

# Subject Bucketing: Teaching a Diffusion Model New Prompt Languages Without Forgetting

> How we curated an 83K-image, JSON-captioned super-dataset into **semantic subject buckets** to
> finetune a 2B anime diffusion transformer — what worked, what didn't, the technical wall we hit
> inside `diffusion-pipe`, and a cleaner dataset architecture built on `huggingface_hub` that would
> make most of those walls disappear.

This article is about **one idea taken seriously**: *subject bucketing*. Everything else — the
trainer, the cache, the captions — is downstream of getting the dataset's **organization** right.
Diffusion finetuning lives or dies on how the data is grouped, weighted, and presented; the model
architecture is almost a footnote by comparison.

---

## 1. The problem subject bucketing solves

When you finetune a text-to-image diffusion model on a large, heterogeneous dataset, the naive
approach — "point the trainer at a folder of images and let it shuffle" — fails in three predictable
ways:

1. **Frequency bias.** A dataset is never balanced. If 30% of images are `1girl` portraits and 0.1%
   are `lighthouse`, uniform sampling teaches the model that the world is mostly portraits and barely
   knows what a lighthouse is. The long tail is effectively invisible.
2. **Over-correction.** The obvious fix — repeat rare concepts until every concept is "equally seen"
   — swings the other way: a 5-image concept repeated 50×/epoch is **memorized**, not learned. You
   trade frequency bias for overfitting on exactly the concepts you cared about preserving.
3. **Broken associations.** If you globally shuffle a structured dataset, the model learns
   *co-occurrence statistics across the whole corpus* rather than *which attribute belongs to which
   subject*. A caption that says `{"subjects":[{"name":"woman","attributes":["red dress"]}]}` only
   teaches "woman → red dress" if the woman and the red dress are presented **together, consistently**
   — not smeared across a shuffle with ten thousand unrelated captions.

**Subject bucketing** is the answer to all three at once. Group images by their *dominant subject*,
weight each group so neither the head nor the tail dominates, and keep semantically-distinct groups
**separate** so the model learns clean subject↔attribute bindings. The bucket is the unit of
curation; everything else is bookkeeping.

---

## 2. The target dataset

We finetune **[CircleStone Labs Anima](https://huggingface.co/circlestone-labs/Anima)** — a 2B-parameter
anime text-to-image **diffusion transformer** whose backbone is **NVIDIA Cosmos-Predict2-2B**, paired
with a **Qwen-3 0.6B** text encoder and the **Qwen-Image VAE**. Anima is the only lineage in this
family with first-class trainer support, and it is **tag-order sensitive** and **knowledge-dense** —
which raises the stakes on data organization considerably.

The data is **[`AbstractPhil/diffusion-pretrain-set-ft1`](https://huggingface.co/datasets/AbstractPhil/diffusion-pretrain-set-ft1)**,
config **`qwen_90k`** (≈ **83K** high-resolution rows, a uniform 17-column schema). The defining
feature is its captions. Each row carries **`task_1` JSON** captions, e.g.:

```json
{"subjects": [{"name": "woman", "attributes": ["blonde hair", "red dress"]}],
 "actions": ["walking"], "setting": "city street at night"}
```

There are two structured caption columns — `caption_vlm_json` (plain-English VLM descriptions) and
`caption_animetimm_json` (Danbooru-style booru tags) — plus a natural-language `captions_source_json`.
The dataset's own documentation is explicit: **train the JSON verbatim**, and **do not** learn subject
association through a cross-subject shuffle. That single instruction is what makes generic
"folder of images + a caption" pipelines wrong for this dataset, and what makes subject bucketing
*the* methodology rather than a nice-to-have.

A few schema realities shaped the pipeline (all discovered by inspecting the config first, never
assumed): the `image` column is a `{bytes, path}` dict, not a decoded image; `audit` is `"approved"`
for every row (keep the gate); `age_classifier_pass` is **unpopulated** (`None`) for this slice, so the
default age gate would drop *every* row and must be disabled; and `caption_animetimm_json` was
`__PARSEFAIL__` on early slices, so the VLM column is the safer primary. (One hard rule, worth stating
once: the `extra_json`/`celeb_name_raw` IMDB columns are **takedown-only** and never enter a caption.)

---

## 3. The intended training outcome

The goal is **not** to teach Anima new pictures. It is to teach Anima a **new prompt interface** —
the `task_1` JSON structure and natural-English phrasing — **without degrading** the dense knowledge
it already has. Concretely, after bucketed training we want:

- **Plain-English prompt segments strengthened.** The VLM captions push natural-language conditioning
  so prompts like "a woman walking on a city street at night" do the right thing.
- **JSON structure learned as a differentiator.** The model learns to read the structured
  `{subjects, actions, setting}` shape and bind attributes to the correct subject.
- **Base knowledge preserved.** No catastrophic forgetting, no memorized sparse buckets, no collapse
  of the model's existing capabilities.

On the **preliminary 1,000-image run**, informal inspection showed exactly this signature: by roughly
epoch 12, plain-English prompting began to take without visibly degrading the base, the JSON structure
differentiated the new capacity, and training **both caption streams together appeared to out-yield
either alone** — while the weighting held the balance and the sparse buckets showed no signs of
memorization. These are *preliminary qualitative observations* (not yet a quantitative eval), but they
are the early signal that the bucketing paradigm is doing its job before we scale to the full set.

---

## 4. The subject bucketing paradigm (the core)

Here is the methodology end to end. Each step exists to defeat one of the three failure modes above.

### 4.1 Bucket = the dominant subject

Every caption already names its subjects. We bucket each image by its **dominant subject** —
`subjects[0]`, normalized to a **head noun**: lowercase, strip articles, take the last word, singularize
(`"the fire trucks"` → `truck`). A `subjects[]` entry may be a `{"name": ...}` dict **or a bare
string**, so the parser handles both. The caption is written **verbatim** into the `.txt` sidecar — we
are teaching the JSON prompt language, so we do **not** render it down to tags.

### 4.2 Group the sparse tail by *semantic* similarity — don't drop it

A head-noun bucketing of a diverse corpus produces a few large buckets and a very long tail of
singletons (`sailboat`, `yacht`, `dinghy`, each with two images). Dropping the tail throws away real
concepts; keeping every singleton as its own bucket produces thousands of un-trainable
micro-datasets. So we **merge weak buckets by meaning**:

- Embed the subject head-nouns with **sentence-transformers** (`all-MiniLM-L6-v2`, ~90 MB; or
  `nomic-ai/nomic-embed-text-v1` if already cached for a zero-download path).
- Cluster the *small* subjects with **agglomerative average-linkage over cosine distance**
  (`grp_boat = {sailboat, boat, yacht}`, `grp_car = {car, suv}`).
- Anything still ungroupable pools into a **weighted `misc_*`** catch-all. **Nothing is omitted.**

This is also the home of our **most instructive failure** (Section 5).

### 4.3 Keep distinct human subgroups separate

`man`, `woman`, `player`, `guitarist`, `person` are *meant* to be distinguished — the captioner
grouped them on purpose, and collapsing them would destroy a learning signal. But `man`↔`woman` cosine
similarity is only ~0.5–0.75, so a naive similarity threshold *would* merge them. The guard is an
explicit **`_HUMAN_SEED`** anchor set (`person, man, woman, child, boy, girl, player, …`) plus
similarity propagation: subjects close to a human anchor are flagged human, large/human buckets are
**protected**, and protected buckets are removed from all merge candidates. The boundary is drawn by
anchors, not by a cosine cutoff that can't separate the genders.

### 4.4 Weight against overtraining (dampened repeats)

This is the anti-memorization step. Instead of physically equalizing every bucket to the largest, we
use **diminishing-returns** repeats:

```
num_repeats(bucket) = round( (top / images(bucket)) ** (1 - alpha) ),  capped
```

with `alpha = 0.5` (square-root damping) as the default, an exposure ceiling `max_repeats = 8`, and an
effective-sample cap `cap_mult = 1.25 × top` so no bucket can dominate. The contrast:

| Policy | `alpha` | A 5-image bucket vs a 10K head | Result |
|---|---:|---|---|
| Equalize-to-largest (legacy) | 0.0 | repeats the 5 images ~50×/epoch | **memorization** |
| Square-root damping (default) | 0.5 | repeats capped at ~8× | learned, not memorized |
| No balancing | 1.0 | everything 1× | the tail vanishes |

`alpha = 0.5` is the sweet spot: big buckets stay ~1–2×, sparse/grouped buckets get a modest lift,
and the per-image exposure ceiling makes overfitting structurally impossible.

### 4.5 Split oversized buckets

The opposite problem: a `woman` bucket with 12,000 images is too coarse — it'll train one giant blob
of "woman-ness." So any bucket above a **data-dependent cap** (>10K images → 1000, ≥1K → 500, else
250) is **split**, in three tiers on a domain disjoint from the grouping step:

1. by the dominant subject's **rarest attribute** (animetimm-preferred, booru-style; a stop-list drops
   `1girl`/`solo`/… so it defaults to `blonde_hair`, not `1girl`) — one attribute per image (the
   rarest) is a true partition;
2. then by **secondary subject** (`man·with_dog`);
3. then **even-chunk** as a hard guarantee no sub-bucket exceeds the cap.

Each sub-bucket becomes its own directory, weighted by its own count via the same dampened-repeats
policy. Splitting runs **per caption stream** because the VLM and animetimm subject trees differ.

### 4.6 Caption modes — training both streams

Both `caption_vlm_json` (plain-English) and `caption_animetimm_json` (booru tags) are training signal.
How they're *organized* is a flag with three settings:

- **`before_after`** (the recommended first LoRA): two trees, `vlm/<subject>` and `animetimm/<subject>`,
  trained as **two sequential phases** — the full VLM set, then the full animetimm set resuming the VLM
  adapter. This was the validated, higher-yield recipe.
- **`separate`**: the same two trees but one dataset config — globally shuffled together.
- **`mixed`**: one image on disk plus a `captions.json` carrying `[vlm, animetimm, joint]` — each image
  trains once with multiple prompts (physical dedupe, no pixel duplication).

---

## 5. What worked, and what didn't

The paradigm is the product of several course-corrections. The failures are as informative as the wins.

**What worked**

- **Subject buckets, not source buckets.** The very first instinct — route images by their `source`
  column — is wrong for this dataset; subjects are the curated unit, and bucketing by source learns
  nothing about subject↔attribute binding.
- **JSON captions trained verbatim.** Rendering the structured caption down to a tag string throws
  away the exact thing we're trying to teach (the prompt *structure*). Verbatim won.
- **Agglomerative average-linkage for the tail.** Real semantic grouping (`grp_boat`, `grp_car`) with
  the human subgroups protected.
- **Dampened repeats (`alpha=0.5`, `max_repeats=8`).** On the 1K run the balance held in informal
  inspection — the sparse buckets showed no memorization.
- **`before_after` two-phase training + both caption streams.** In the prelim run both streams together
  *appeared* to out-yield either alone, with the base knowledge preserved (a qualitative read, not yet
  a measured eval).

**What didn't (and what we learned)**

- **Single-linkage connected-components clustering.** Our first grouping pass used union-find /
  connected components over the embedding graph. Dense embeddings **chain**: A is close to B, B to C,
  C to D… and the whole neighborhood collapses into one blob. We produced a **961-image `grp_bed`**
  this way. Switching to **agglomerative average-linkage** (and raising the sentence-transformers
  threshold to `0.58`) fixed it — clean groups instead of one giant chain.
- **Equalize-to-largest weighting (`alpha=0`).** The legacy policy that repeats a 5-image bucket
  ~50×/epoch. It's the textbook way to overfit the long tail you were trying to protect. Replaced by
  the dampened policy.
- **Row-by-row extraction.** The first extractor used `datasets` streaming + PNG re-encode and ran at
  **~1 image/second** — unusable at 83K. The fix is **columnar**: `pyarrow.ParquetFile.iter_batches`
  reading only the needed columns and writing the image's **raw bytes** straight to disk (no decode /
  re-encode), with **lazy** shard downloads (stop at the requested count) — the full config is >100 GB,
  so you never pre-fetch all shards.
- **difflib similarity at scale.** Without the sentence-transformers extra, the fallback similarity is
  an O(N²) `SequenceMatcher` sweep over all unique subjects — fine for a 1K test, fatal at 83K. The
  semantic backend (a single embedding matmul) is mandatory at scale; we also vectorized the cluster
  submatrix extraction after it became the hot loop.
- **2× latent duplication under `before_after`.** Because the animetimm tree hardlinks the same pixels
  as the VLM tree but lives under a separate directory, the trainer caches their latents **twice**.
  Correct, but wasteful — and it points directly at the better architecture in Section 7.

---

## 6. The functionality, briefly: producing bucketed datasets inside `diffusion-pipe`

Subject bucketing is the idea; **[`tdrussell/diffusion-pipe`](https://github.com/tdrussell/diffusion-pipe)**
is the engine, and bridging into it is where the real engineering lives. A few technical elements and
the challenges they impose:

- **The interface is a filesystem.** `diffusion-pipe` consumes **directories of image files +
  matching `.txt` sidecars**, one `[[directory]]` per bucket, with `num_repeats` and a global
  `dataset.toml`. So a columnar parquet dataset has to be *materialized* into a directory tree — which
  is exactly where the friction starts (thousands of files, lost provenance, absolute paths).
- **The latent/text-embed cache is a sharded blob store.** A `--cache_only` pass precomputes VAE
  latents + Qwen-3 text embeddings into `<dir>/cache/anima/**`: per size-bucket `metadata.db` (SQLite)
  + `shard_*.bin` (10 GB shards), plus HuggingFace-`datasets` `*.arrow` metadata. The records table
  commits only when a 10 GB shard finalizes or a pass ends, so a fresh cache reads "0 records" for a
  while even as bytes pour in. Caching is **decode-bound, not GPU-bound** (the 2B DiT never loads; only
  the VAE + 0.6B encoder do), so the real throughput lever is the **decode-worker pool**
  (`map_num_proc`), not batch size.
- **The cache is fingerprint-locked to the dataset's paths.** The fingerprint is derived from the
  metadata dataset, which includes each image's **configured path string**; because this project pins
  absolute `DATA_ROOT` paths, any path/layout change (or a non-deterministic re-extraction)
  re-fingerprints and triggers a silent **wipe**. This is the single biggest source of fragility when
  you move between machines or ephemeral runtimes.
- **The shuffle is mandatory.** `diffusion-pipe` builds one `Dataset` + one LR schedule with several
  hardcoded-seed shuffle layers, so "VLM strictly before animetimm" is **impossible inside one run** —
  which is precisely why `before_after` is implemented as **two chained runs** handing the adapter off
  via `[adapter].init_from_existing`.
- **Sharp edges we had to harden against.** A real `FileNotFoundError`→**hang** bug (a forked cache
  worker dies without signaling its parent, which then blocks forever); a metadata pass that opens
  *every* image before the first latent shard appears (so "0 shards for 2 minutes" is normal, not
  stuck); and a cache push that tried to upload ~30K tiny regenerable `*.arrow` files until we learned
  to preserve only the expensive `metadata.db` + `shard_*.bin`.

The package wraps all of this behind one command surface (`anima subjects`, `anima cache --progress`,
`anima train-before-after`) with the Anima invariants encoded in a `validate()` gate — `llm_adapter_lr
= 0` (frozen, knowledge-dense), `shuffle_caption = false` (tag-order sensitive), bf16 throughout, no
fp8, no flash-attn on Blackwell.

---

## 7. A better methodology: HF-datasets-native subject bucketing

Most of Section 6's pain is **incidental complexity** created by forcing a columnar dataset through a
filesystem interface. If we instead treat the dataset as a **first-class `huggingface_hub` dataset**
and let the *index* — not the directory tree — be the source of truth, the architecture gets
dramatically simpler. We have already **implemented the first two steps** (used in the preliminary and
full-run notebooks); the rest is the natural continuation.

### Pain point → progress

**Pain 1 — "materialize a directory tree of small files."** Bucketing into directories produces tens
of thousands of files (images + `.txt` + per-bucket cache), loses the link back to the source row, and
makes the cache absolute-path-fragile.

> **Progress (done):** an `index.jsonl` manifest. Extraction now writes one small record per image —
> `{id, slug, shard, ext, sha256, vlm_bucket, vlm_cap}` (+ `anime_bucket`/`anime_cap`, or
> `mixed_bucket`/`mixed_caps`) — that **pins the bucket assignment** (the only non-deterministic step)
> and stores the captions verbatim. The images are **not** stored; on a fresh machine they're
> **columnar-refetched from the source parquet** by id and written back **byte-identically**. The
> manifest is on the order of tens of MB against tens of GB of images — roughly three orders of
> magnitude smaller (projected for the full set) — and the rebuilt tree reproduces the same dataset
> fingerprint exactly. *This is "the dataset is a tiny index + a content store," and it works.*

**Pain 2 — "the cache is locked to absolute paths and re-extraction can wipe it."** The fingerprint
embeds the path, so the cache is portable only if you pin `DATA_ROOT` and reproduce the layout exactly.

> **Progress (done):** the cache lives on the Hub as a **dataset repo** of shards, pushed incrementally
> and pulled back to resume — already-cached items are skipped, accumulating the full cache across
> sessions. We preserve only the expensive `metadata.db` + `shard_*.bin` and **regenerate** the
> HF-`datasets` metadata on resume (it's content-deterministic), so a push is a few hundred files, not
> 30K.

**Pain 3 — "the same pixels get their latents cached twice" (the `before_after` 2× cost).** The trainer
keys latents by *dataset position + path*, so two directories over the same hardlinked pixels are two
caches.

> **Progress (proposed):** **content-addressed latents.** We already compute and store a `sha256` of
> every image's raw bytes in the manifest. Key the latent cache by **that hash** instead of by dataset
> position and absolute path. Three wins fall out immediately: (a) `before_after`'s identical pixels
> share **one** latent — the 2× duplication disappears; (b) the cache becomes **portable** across
> machines and runtimes with no path-fingerprint fragility; (c) resume is trivial — "is `sha256` already
> cached?" is a hash lookup, not a positional skip over a shuffled metadata table. Content addressing is
> the single highest-leverage change available.

**Pain 4 — "physical `num_repeats` duplication to express weights."** Repeating a bucket N× inflates
the on-disk dataset and the cache.

> **Progress (proposed):** a **bucket-aware weighted sampler** over a single HF dataset. Represent the
> bucketed corpus as **one columnar dataset** with `bucket`, `subject`, `caption_vlm`,
> `caption_animetimm`, and a `weight` column (the dampened-repeats value as a float). A
> `WeightedRandomSampler` reproduces the exact exposure distribution **without duplicating a single
> byte**, and a "no-cross-bucket-shuffle within a step" constraint preserves the subject↔attribute
> binding the directory layout was protecting by construction.

**Pain 5 — "extract everything to local disk before training."** The full config is >100 GB; ephemeral
runtimes can't hold it.

> **Progress (proposed):** **stream from the Hub.** With a content store (the source parquet) + a
> manifest + content-addressed latents, training reads a WebDataset/parquet stream directly, decoding
> only what the sampler asks for. The "dataset" is a manifest + a cache, both small and both on the Hub;
> nothing has to be fully materialized.

### What the HF-native system looks like

The **proposed** end-state (today's shipped artifact is the `index.jsonl` manifest above; the enriched
`manifest.parquet` with a `weight` column and content-addressed latents are the *proposed* additions
from Pains 3–4):

```
HF dataset repo (one place, all small/columnar):
  manifest.parquet      # id, shard, sha256, subject, bucket, weight, caption_vlm, caption_animetimm
  latents/<sha256>.*    # content-addressed VAE latents (deduped across caption modes)
  text_embeds/<hash>.*  # content-addressed text embeddings, keyed by caption hash
```

A trainer that consumes this needs only three things the filesystem layout was faking: a **`bucket`
field**, a **weighted sampler**, and a **content-addressed cache lookup**. Everything else — the
directory tree, the `.txt` sidecars, the absolute-path fingerprint, the 2× latents, the
re-extraction — is eliminated, not worked around. The subject-bucketing *paradigm* is unchanged; only
its *representation* moves from "a tree of files" to "a column in a dataset."

---

## 8. Accessing the preliminary

Everything is in **[`AbstractEyes/anima-trainer`](https://github.com/AbstractEyes/anima-trainer)**
(the package, the bridge scripts, and two end-to-end Colab/persistent-box notebooks).

```bash
pip install -e ".[similarity]"      # the sentence-transformers extra powers real semantic grouping
anima download --dest models/anima --base base-v1.0
anima inspect  --repo AbstractPhil/diffusion-pretrain-set-ft1 --config qwen_90k   # probe before extracting
anima subjects --repo AbstractPhil/diffusion-pretrain-set-ft1 --config qwen_90k \
    --out datasets/anima_subjects --limit 1000 \
    --caption-mode before_after --build-toml configs                              # the buckets
anima cache    --config configs/lora_vlm.toml --progress                          # precache latents/embeds
anima train-before-after --lora-vlm configs/lora_vlm.toml \
    --lora-animetimm configs/lora_animetimm.toml --num-gpus 1                      # two-phase LoRA
```

- The **preliminary rank-64 `before_after` LoRA** (the 1,000-image validation run) is produced by
  `notebooks/anima_colab_prelim_train.ipynb` and backed up to a HuggingFace **model** repo under your
  account (e.g. `<user>/anima-prelim-1k-r64`). It is a ComfyUI-format LoRA — drop the `epochN`
  safetensors into `ComfyUI/models/loras/`.
- The **full-run notebook** (`notebooks/anima_full90k_train.ipynb`) implements the HF-Hub cache
  accumulation + index/refetch described in Section 7 for a persistent box.

> **License — non-commercial.** Anima and any LoRA finetuned from it are **non-commercial** derivatives
> (CircleStone NC **+** the NVIDIA Open Model License, as a Cosmos-Predict2 derivative). The trainer
> *code* is Apache-2.0; that does **not** relax the model/weights terms.

---

## 9. Citations & repositories

**Model & data**

- Anima (2B anime DiT, Cosmos-Predict2-2B backbone) — <https://huggingface.co/circlestone-labs/Anima>
- Target dataset `diffusion-pretrain-set-ft1` (config `qwen_90k`, `task_1` JSON captions) —
  <https://huggingface.co/datasets/AbstractPhil/diffusion-pretrain-set-ft1>
- NVIDIA Cosmos-Predict2 (the DiT backbone lineage) — <https://github.com/nvidia-cosmos/cosmos-predict2>

**Trainer & tooling**

- `tdrussell/diffusion-pipe` (the only trainer with native Anima support) —
  <https://github.com/tdrussell/diffusion-pipe>
- `AbstractEyes/anima-trainer` (this work: the columnar bridge, subject bucketing, HF-Hub cache
  preservation, the notebooks) — <https://github.com/AbstractEyes/anima-trainer>

**Methodology dependencies**

- sentence-transformers `all-MiniLM-L6-v2` (semantic subject grouping) —
  <https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2>
- `nomic-ai/nomic-embed-text-v1` (zero-download grouping alternative) —
  <https://huggingface.co/nomic-ai/nomic-embed-text-v1>
- scikit-learn `AgglomerativeClustering` (average-linkage over cosine distance) —
  <https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html>
- `huggingface_hub` (dataset repos, `snapshot_download`, `upload_folder`) —
  <https://github.com/huggingface/huggingface_hub>
- Hugging Face `datasets` (the columnar/Arrow substrate the better methodology builds on) —
  <https://github.com/huggingface/datasets>

*Built with CircleStone Anima, NVIDIA Cosmos-Predict2, and tdrussell/diffusion-pipe. Non-commercial.*
