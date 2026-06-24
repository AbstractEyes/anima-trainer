#!/usr/bin/env python3
"""
subject_buckets.py — FAST columnar extraction of the super-dataset into SUBJECT
BUCKETS, training the structured JSON caption VERBATIM (the dataset's methodology).

Read path (per the dataset's own scripts: "pyarrow columnar read of source shard"):
  1. resolve the config's parquet shards from the README config map,
  2. hf_hub_download each shard once (sequential, xet-accelerated),
  3. pq.ParquetFile(...).iter_batches(columns=[...]) — columnar, low-memory,
  4. write the image's RAW bytes straight to disk (NO decode / NO re-encode).

This is the fix for the slow `datasets`-streaming + PNG-re-encode path.

Caption: `caption_vlm_json` (task_1 JSON) written VERBATIM as the .txt sidecar;
`caption_animetimm_json` emitted as a SECOND sample (duplicated image) when real.

Bucketing: by the DOMINANT subject (subjects[0]) already present in the JSON,
normalized to a head-noun "type"; small/similar buckets fuzzy-merged into adjacent
ones, tiniest dropped. One diffusion-pipe [[directory]] per merged subject (the
README warns subject association must NOT be trained via cross-subject shuffle).

Safety: only reads the narrow column set below — never `extra_json` / `celeb_name_raw`
(IMDB celebrity names are takedown-only, never a training signal).
"""

from __future__ import annotations

import enum
import fnmatch
import json
import logging
import math
import os
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from difflib import SequenceMatcher, get_close_matches
from pathlib import Path
from typing import Callable


class CaptionMode(enum.Enum):
    """How the VLM-JSON and animetimm-JSON caption samples are organized + trained."""
    SEPARATE = "separate"          # two bucket trees, ONE dataset.toml, globally shuffled
    MIXED = "mixed"                # one image inode, both captions via captions.json (deduped)
    BEFORE_AFTER = "before_after"  # two trees, TWO sequential runs (vlm -> animetimm). FIRST LORA.


@dataclass
class ImageRecord:
    """Per-image differentiation features carried from pass 1 to bucketing/splitting."""
    vlm_subject: str | None        # normalized dominant subject of the vlm caption
    anime_subject: str | None      # normalized dominant subject of the animetimm caption
    attrs: tuple = ()              # normalized attributes of subjects[0] (prefer-source first)
    secondary: str | None = None   # normalized subjects[1].name (prefer-source first)

log = logging.getLogger("anima.subjects")

# Narrow column set read from parquet (no extra_json / mask / conditioning / celeb name).
_READ_COLUMNS = [
    "image", "caption_vlm_json", "caption_animetimm_json",
    "id", "audit", "age_classifier_pass",
]


# =============================================================================
# CONFIG
# =============================================================================
@dataclass
class SubjectBucketConfig:
    repo: str = "AbstractPhil/diffusion-pretrain-set-ft1"
    config: str = "qwen_90k"
    split: str = "train"
    out_root: str = "datasets/anima_subjects"
    limit: int | None = 1000              # total accepted images (the first test ~1000)
    batch_size: int = 512                 # columnar record-batch size (local read)

    # caption columns, IN PRIORITY ORDER (vlm first, animetimm second if real).
    caption_columns: tuple = ("caption_vlm_json", "caption_animetimm_json")

    # subject bucketing
    head_noun: bool = True                # reduce "fire truck" -> "truck" (snip the type)
    min_bucket_size: int = 10             # buckets smaller than this get merged or dropped
    fuzzy_cutoff: float = 0.62            # difflib ratio to merge a small bucket into a big one
    drop_unmergeable: bool = True         # else send leftovers to a "misc" bucket

    # quality gates
    require_audit_approved: bool = True
    require_age_pass: bool = False        # qwen_90k: age_classifier_pass is unpopulated

    # ---- semantic grouping of the sparse tail (group + weight, don't omit) ----
    use_semantic: bool = True             # False -> legacy difflib plan_buckets
    semantic_backend: str = "auto"        # auto | sentence-transformers | trigram | difflib
    similarity_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    #   or "nomic-ai/nomic-embed-text-v1" -> reuses a model already cached on F: (0 download)
    sim_threshold: float = 0.45           # cosine edge for small<->small grouping (preset-overridden)
    human_sim_threshold: float = 0.55     # higher -> conservative human boundary (preset-overridden)
    human_min_size: int = 4               # human protection floor (<< min_bucket_size)
    min_final_group_size: int = 8         # group below this (summed images) -> misc_* (weighted, not dropped)
    max_group_members: int = 8            # connected-components fallback cap (unused by agglomerative)
    keep_small: bool = True               # weight-don't-drop: leftovers -> misc_*, never None

    # ---- dual-caption modes (bucket the vlm + animetimm samples) ----
    caption_mode: CaptionMode = CaptionMode.BEFORE_AFTER   # the FIRST LoRA uses before_after
    dedupe_mixed_concat: bool = True      # MIXED: also emit a 3rd "vlm\nanimetimm" joint caption

    # ---- oversized-bucket splitting (data-dependent cap, attribute differentiation) ----
    split_oversized: bool = True
    max_bucket_size: int | None = None    # None -> data-dependent: >10k=1000, >=1k=500, else 250
    prefer_attr_source: str = "animetimm" # which caption's attributes win ("animetimm" | "vlm")
    attr_min_split: int = 2               # don't carve a sub-bucket smaller than this
    split_separator: str = "·"       # display marker (middle dot); slugged away in dir names


# =============================================================================
# SUBJECT NORMALIZATION  ("snip out the subject type")
# =============================================================================
_ARTICLES = {"a", "an", "the"}
_IRREGULAR = {"men": "man", "women": "woman", "people": "person", "children": "child",
              "feet": "foot", "teeth": "tooth", "mice": "mouse", "geese": "goose"}


def _singularize(w: str) -> str:
    if w in _IRREGULAR:
        return _IRREGULAR[w]
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 4 and w.endswith(("ses", "xes", "zes", "ches", "shes")):
        return w[:-2]
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w


def normalize_subject(name: str | None, *, head_noun: bool = True) -> str | None:
    """Lowercase, strip punctuation/articles, optionally reduce to the head noun, and
    singularize. 'Fire Truck' -> 'truck', 'the Police Officers' -> 'officer'."""
    if not name:
        return None
    s = re.sub(r"[^a-z0-9 ]+", " ", str(name).lower()).strip()
    toks = [t for t in s.split() if t not in _ARTICLES]
    if not toks:
        return None
    key = toks[-1] if head_noun else " ".join(toks)
    return _singularize(key) or None


_SLUG = re.compile(r"[^a-zA-Z0-9_-]+")


def slug(s: str) -> str:
    return _SLUG.sub("_", str(s).strip()).strip("_") or "unknown"


# =============================================================================
# CAPTION / SUBJECT PARSING
# =============================================================================
def real_caption(v: str | None) -> str | None:
    """The caption string if it is a real JSON value, else None ("" / "__PARSEFAIL__"...)."""
    if not v or v.startswith("__"):
        return None
    return v


def _subject_name(s) -> str | None:
    """A subjects[] entry may be a dict {'name': ...} OR a bare string."""
    if isinstance(s, str):
        return s
    if isinstance(s, dict):
        return s.get("name")
    return None


def dominant_subject(vlm_json: str, *, head_noun: bool = True) -> str | None:
    """Normalized name of the first (dominant) subject in a caption_vlm_json string."""
    try:
        obj = json.loads(vlm_json)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(obj, dict):
        return None
    subs = obj.get("subjects") or []
    if not subs:
        return None
    return normalize_subject(_subject_name(subs[0]), head_noun=head_noun)


def caption_subject(caption_json: str, *, head_noun: bool = True) -> str | None:
    """Dominant subject of ONE caption (vlm OR animetimm), bucketed independently."""
    return dominant_subject(caption_json, head_noun=head_noun)


# ---- attributes & secondary subject (for splitting oversized buckets) -------
# Count/meta booru tags that don't usefully differentiate a subject bucket — so we
# split a 'woman' bucket by 'blonde_hair', NOT by '1girl' (the requirement).
_ATTR_STOP = {"1girl", "1boy", "2girls", "2boys", "solo", "general", "sensitive",
              "simple_background", "white_background", "looking_at_viewer"}


def normalize_attr(a: str | None) -> str | None:
    """Booru attribute -> stable slug-safe differentiator token. Lowercase, spaces->_,
    KEEP multiword ('blonde_hair' stays distinct from 'red_hair' — no head-noun), drop
    count/meta tags. 'Blonde Hair' -> 'blonde_hair'; '1girl' -> None."""
    if not a:
        return None
    s = re.sub(r"[^a-z0-9 _]+", " ", str(a).lower()).strip()
    s = re.sub(r"[ _]+", "_", s).strip("_")
    if not s or s in _ATTR_STOP:
        return None
    return s


def _subject_attrs(entry) -> list:
    """Attributes of a subjects[] entry: dict 'attributes' list, or [] for a bare string."""
    if isinstance(entry, dict):
        return [a for a in (entry.get("attributes") or []) if a]
    return []


def _features(caption_json: str | None, *, head_noun: bool):
    """(dominant_subject, normalized attrs of subjects[0], normalized subjects[1] name)."""
    if not caption_json:
        return None, (), None
    try:
        obj = json.loads(caption_json)
    except (json.JSONDecodeError, TypeError):
        return None, (), None
    if not isinstance(obj, dict):
        return None, (), None
    subs = obj.get("subjects") or []
    if not subs:
        return None, (), None
    subject = normalize_subject(_subject_name(subs[0]), head_noun=head_noun)
    attrs = tuple(x for x in (normalize_attr(a) for a in _subject_attrs(subs[0])) if x)
    secondary = normalize_subject(_subject_name(subs[1]), head_noun=head_noun) \
        if len(subs) > 1 else None
    return subject, attrs, secondary


def extract_features(vlm_json: str | None, anime_json: str | None, *,
                     prefer: str = "animetimm", head_noun: bool = True) -> ImageRecord:
    """Per-image record: dominant subject of EACH caption (bucketed independently) +
    attributes/secondary of subjects[0] taken from the `prefer` caption first (animetimm
    attrs are richer), the other caption as fallback."""
    vlm_subj, vlm_attrs, vlm_sec = _features(vlm_json, head_noun=head_noun)
    anime_subj, anime_attrs, anime_sec = _features(anime_json, head_noun=head_noun)
    if prefer == "vlm":
        attrs = vlm_attrs or anime_attrs
        secondary = vlm_sec or anime_sec
    else:
        attrs = anime_attrs or vlm_attrs
        secondary = anime_sec or vlm_sec
    return ImageRecord(vlm_subject=vlm_subj, anime_subject=anime_subj,
                       attrs=attrs, secondary=secondary)


# =============================================================================
# OVERSIZED-BUCKET SPLITTING  (data-dependent cap; attribute -> secondary -> chunk)
# =============================================================================
def max_bucket_size(total_images: int, override: int | None = None) -> int:
    """Per-bucket image ceiling, scaled to dataset size (manually overridable).
    >10000 imgs -> 1000 ; 1000-10000 -> 500 ; <1000 -> 250."""
    if override is not None:
        return override
    if total_images > 10_000:
        return 1_000
    if total_images >= 1_000:
        return 500
    return 250


def _even_chunks(seq: list, n: int) -> list:
    """Split seq into n near-equal contiguous chunks (no tiny remainder)."""
    n = max(1, n)
    k, m = divmod(len(seq), n)
    out, start = [], 0
    for i in range(n):
        size = k + (1 if i < m else 0)
        out.append(seq[start:start + size])
        start += size
    return [c for c in out if c]


def _split_one(records_of: dict, ids: list, subj: str, M: int,
               cfg: SubjectBucketConfig) -> dict:
    """Partition one oversized bucket's idslugs into <=M sub-buckets:
    TIER A rarest dominant-subject attribute -> TIER B secondary subject -> TIER C chunk.
    Every image lands in exactly one sub-bucket (a true partition)."""
    sep = cfg.split_separator
    attr_freq = Counter(a for i in ids for a in records_of[i].attrs)
    groups: dict = defaultdict(list)
    no_attr: list = []
    for i in ids:
        cand = [a for a in records_of[i].attrs if a]
        if cand:                                   # rarest attr = most discriminative
            key = min(cand, key=lambda a: (attr_freq[a], a))
            groups[f"{subj}{sep}{key}"].append(i)
        else:
            no_attr.append(i)
    # fold sub-threshold attribute splits back into the no-attr pool
    for name in [n for n, m in groups.items() if len(m) < cfg.attr_min_split]:
        no_attr.extend(groups.pop(name))
    for i in no_attr:                              # TIER B: secondary subject
        sec = records_of[i].secondary
        groups[f"{subj}{sep}with_{sec}" if sec else f"{subj}{sep}plain"].append(i)
    final: dict = {}                               # TIER C: even-chunk anything still > M
    for name, members in groups.items():
        if len(members) <= M:
            final[name] = members
        else:
            for c, part in enumerate(_even_chunks(sorted(members), math.ceil(len(members) / M))):
                final[f"{name}_{c:02d}"] = part
    return final


def split_oversized_buckets(records_of: dict, plan: "BucketPlan", cfg: SubjectBucketConfig,
                            *, stream: str, subject_attr: str):
    """For ONE caption stream ('vlm'/'anime'), split each FINAL bucket that exceeds the
    data-dependent cap. Returns (override, subcounts, oversized): override maps
    (idslug, stream) -> sub-bucket display name; subcounts maps name -> count."""
    M = max_bucket_size(len(records_of), cfg.max_bucket_size)
    by_bucket: dict = defaultdict(list)
    for idslug, rec in records_of.items():
        subj = getattr(rec, subject_attr)
        if subj is None:
            continue
        bucket = plan.mapping.get(subj)
        if bucket is not None:
            by_bucket[bucket].append(idslug)
    override: dict = {}
    subcounts: Counter = Counter()
    oversized: list = []
    for bucket, ids in by_bucket.items():
        if len(ids) <= M:
            continue
        oversized.append((bucket, len(ids)))
        log.warning("subject %r (%d imgs, %s stream) breaches max %d -> sub-bucketing",
                    bucket, len(ids), stream, M)
        for name, members in _split_one(records_of, ids, bucket, M, cfg).items():
            for idslug in members:
                override[(idslug, stream)] = name
            subcounts[name] += len(members)
    return override, subcounts, oversized


# ---- MIXED-mode caption list + captions.json ------------------------------
def _mixed_captions(vlm: str, anime: str | None, cfg: SubjectBucketConfig) -> list:
    caps = [vlm]
    if anime:
        caps.append(anime)
        if cfg.dedupe_mixed_concat:
            caps.append(vlm + "\n" + anime)        # joint-conditioning sample
    return caps


# =============================================================================
# IMAGE BYTES  (raw pass-through — no decode/encode)
# =============================================================================
def _sniff_ext(b: bytes) -> str:
    if b[:3] == b"\xff\xd8\xff":
        return ".jpg"
    if b[:8] == b"\x89PNG\r\n\x1a\n":
        return ".png"
    if b[:4] == b"RIFF" and b[8:12] == b"WEBP":
        return ".webp"
    if b[:2] == b"BM":
        return ".bmp"
    if b[:6] in (b"GIF87a", b"GIF89a"):
        return ".gif"
    return ".jpg"


def _img_bytes(value) -> tuple[str | None, bytes | None]:
    """Pull raw encoded bytes from a parquet image cell ({'bytes','path'} struct)."""
    b = None
    if isinstance(value, dict):
        b = value.get("bytes")
        if not b and value.get("path"):
            try:
                b = Path(value["path"]).read_bytes()
            except OSError:
                b = None
    elif isinstance(value, (bytes, bytearray)):
        b = bytes(value)
    if not b:
        return None, None
    return _sniff_ext(b), b


# =============================================================================
# BUCKET PLANNING  (fuzzy-merge small into adjacent, drop the tiniest)
# =============================================================================
@dataclass
class BucketPlan:
    mapping: dict           # normalized subject -> final bucket name (or None = drop)
    raw_counts: Counter
    actions: list           # (subject, count, action) for reporting
    groups: dict = field(default_factory=dict)            # bucket name -> [member subjects]
    group_counts: Counter = field(default_factory=Counter)  # bucket name -> summed images


def plan_buckets(subjects: list[str], cfg: SubjectBucketConfig) -> BucketPlan:
    """Dominant buckets first; merge small/similar buckets into the nearest large one
    via fuzzy match; drop (or 'misc') anything that can't merge and is below threshold."""
    counts = Counter(s for s in subjects if s)
    big = {k for k, c in counts.items() if c >= cfg.min_bucket_size}
    mapping: dict = {k: k for k in big}
    actions: list = []
    big_list = sorted(big, key=lambda k: -counts[k])  # prefer dominant targets
    for k, c in counts.most_common():
        if k in big:
            continue
        match = get_close_matches(k, big_list, n=1, cutoff=cfg.fuzzy_cutoff)
        if match:
            mapping[k] = match[0]
            actions.append((k, c, f"merge -> {match[0]}"))
        elif cfg.drop_unmergeable:
            mapping[k] = None
            actions.append((k, c, "drop"))
        else:
            mapping[k] = "misc"
            actions.append((k, c, "misc"))
    return BucketPlan(mapping=mapping, raw_counts=counts, actions=actions)


# =============================================================================
# SEMANTIC GROUPING  (group the sparse tail by similarity + protect human subgroups)
# =============================================================================
SimFn = Callable[[list, list], "object"]   # (query, cands) -> (nq, nc) cosine matrix

# ~12 anchors — the ONLY hardcoded human knowledge; small, auditable, matches qwen's
# coarse human anchors. Everything else (guitarist, dancer, officer) is pulled in by
# similarity to these, so the taxonomy is not frozen.
_HUMAN_SEED = ("person", "man", "woman", "child", "boy", "girl", "player",
               "worker", "performer", "crowd", "figure", "human")
_AGENTIVE = re.compile(r"(ist|er|man|woman|person|girl|boy)$")  # non-semantic-tier backstop


def make_sim_fn(cfg: SubjectBucketConfig) -> tuple[SimFn, dict]:
    """Pick the best available similarity backend -> (sim_fn, preset thresholds).
    Order: sentence-transformers (real semantics) -> char-trigram (numpy, morphology)
    -> difflib (lexical). Never raises; logs the tier chosen. preset carries
    backend-tuned thresholds because the cosine scales differ ~2x across backends."""
    import numpy as np
    from . import subject_similarity as _sim
    backend = cfg.semantic_backend

    def _matrix(embed, query, cands):
        uniq = list(dict.fromkeys(list(query) + list(cands)))
        emb = embed(uniq)
        pos = {p: i for i, p in enumerate(uniq)}
        qa = emb[[pos[q] for q in query]]
        ca = emb[[pos[c] for c in cands]] if cands else np.zeros((0, emb.shape[1]), "float32")
        return _sim.cosine(qa, ca)

    if backend in ("auto", "sentence-transformers") and _sim.have_semantic_backend():
        try:
            mid = cfg.similarity_model
            _sim.embed_subjects(["warmup"], model_id=mid)   # surface load errors now
            log.info("similarity backend: sentence-transformers (%s)", mid)
            return ((lambda q, c: _matrix(lambda p: _sim.embed_subjects(p, model_id=mid), q, c)),
                    {"sim_threshold": 0.58, "human_sim_threshold": 0.46, "agentive": False})
        except Exception as e:  # noqa: BLE001 — model load / CUDA / offline
            log.warning("sentence-transformers backend failed (%s); falling back", e)

    if backend in ("auto", "sentence-transformers", "trigram"):
        log.info("similarity backend: char-trigram (numpy, morphological — not semantic)")
        return ((lambda q, c: _matrix(_sim.trigram_embed, q, c)),
                {"sim_threshold": 0.50, "human_sim_threshold": 0.50, "agentive": True})

    def _difflib(query, cands):
        m = np.zeros((len(query), len(cands)), "float32")
        for i, qi in enumerate(query):
            for j, cj in enumerate(cands):
                m[i, j] = SequenceMatcher(None, qi, cj).ratio()
        return m
    log.info("similarity backend: difflib (lexical only)")
    return _difflib, {"sim_threshold": 0.60, "human_sim_threshold": 0.62, "agentive": True}


def is_human_map(subjects: list, sim_fn: SimFn, preset: dict, cfg: SubjectBucketConfig) -> dict:
    """subject -> bool. Human iff max cosine to the seed anchors >= threshold; in the
    non-semantic tiers also accept an agentive suffix (-ist/-er/-man...) as a backstop."""
    thr = preset.get("human_sim_threshold", cfg.human_sim_threshold)
    sh = sim_fn(subjects, list(_HUMAN_SEED))
    out = {}
    for i, s in enumerate(subjects):
        sem = bool(sh.shape[1]) and float(sh[i].max()) >= thr
        out[s] = sem or (preset.get("agentive", False) and bool(_AGENTIVE.search(s)))
    return out


def _cluster_side(side: list, S, idx: dict, threshold: float) -> list:
    """Cluster one side's subjects into groups. Prefers AGGLOMERATIVE average-linkage
    (sklearn, available with the [similarity] extra) over cosine distance — it does NOT
    chain dense embeddings into one blob the way single-linkage connected-components do.
    Falls back to connected-components (union-find) when sklearn is absent."""
    if len(side) <= 1:
        return [list(side)] if side else []
    try:
        import numpy as np
        from sklearn.cluster import AgglomerativeClustering
        sub = np.array([[float(S[idx[a], idx[b]]) for b in side] for a in side], "float32")
        dist = np.clip(1.0 - sub, 0.0, 2.0)
        np.fill_diagonal(dist, 0.0)
        labels = AgglomerativeClustering(
            n_clusters=None, metric="precomputed", linkage="average",
            distance_threshold=1.0 - threshold).fit(dist).labels_
        comps: dict = {}
        for s, lab in zip(side, labels):
            comps.setdefault(int(lab), []).append(s)
        return list(comps.values())
    except Exception as e:  # noqa: BLE001 — sklearn missing / degenerate matrix
        log.debug("agglomerative unavailable (%s); using connected-components", e)

    parent = {s: s for s in side}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a in range(len(side)):
        for b in range(a + 1, len(side)):
            if float(S[idx[side[a]], idx[side[b]]]) >= threshold:
                ra, rb = find(side[a]), find(side[b])
                if ra != rb:
                    parent[ra] = rb
    comps = {}
    for s in side:
        comps.setdefault(find(s), []).append(s)
    return list(comps.values())


def plan_buckets_semantic(counts: Counter, cfg: SubjectBucketConfig,
                          sim_fn: SimFn | None = None) -> BucketPlan:
    """Protect large + human buckets; group the small remainder by semantic similarity
    (small-with-small, never into protected, never across the human boundary); fold
    leftovers into weighted misc_* buckets instead of dropping. Records group metadata."""
    preset: dict = {}
    if sim_fn is None:
        sim_fn, preset = make_sim_fn(cfg)
    sim_threshold = preset.get("sim_threshold", cfg.sim_threshold)

    subjects = [s for s in counts if s]
    if not subjects:
        return BucketPlan(mapping={}, raw_counts=counts, actions=[])
    idx = {s: i for i, s in enumerate(subjects)}
    S = sim_fn(subjects, subjects)                     # one (N,N) all-pairs call
    human = is_human_map(subjects, sim_fn, preset, cfg)

    big = {s for s in subjects if counts[s] >= cfg.min_bucket_size}
    protected = big | {s for s in subjects
                       if human[s] and counts[s] >= cfg.human_min_size}

    mapping: dict = {s: s for s in protected}
    groups: dict = {s: [s] for s in protected}
    group_counts: Counter = Counter({s: counts[s] for s in protected})
    actions: list = []

    small = [s for s in subjects if s not in protected]
    sides = [("grp_h_", "misc_human", [s for s in small if human[s]]),
             ("grp_", "misc_other", [s for s in small if not human[s]])]

    for prefix, misc, side in sides:
        for members in _cluster_side(side, S, idx, sim_threshold):
            total = sum(counts[m] for m in members)
            if len(members) > 1 and total >= cfg.min_final_group_size:
                name = prefix + slug(max(members, key=lambda m: counts[m]))
                for m in members:
                    mapping[m] = name
                    actions.append((m, counts[m], f"group -> {name}"))
                groups[name] = members
                group_counts[name] = total
            elif cfg.keep_small:
                for m in members:                       # weight, don't omit
                    mapping[m] = misc
                    actions.append((m, counts[m], f"misc -> {misc}"))
                    groups.setdefault(misc, []).append(m)
                    group_counts[misc] += counts[m]
            else:
                for m in members:
                    mapping[m] = None
                    actions.append((m, counts[m], "drop"))

    return BucketPlan(mapping=mapping, raw_counts=counts, actions=actions,
                      groups=groups, group_counts=group_counts)


# =============================================================================
# SHARD RESOLUTION  (config -> parquet repo paths, metadata only)
# =============================================================================
def _resolve_shards(repo: str, config: str, split: str) -> list[str]:
    """Resolve a config's parquet shard repo-paths from the README config map."""
    import yaml
    from huggingface_hub import hf_hub_download, list_repo_files

    readme = hf_hub_download(repo, "README.md", repo_type="dataset")
    text = Path(readme).read_text(encoding="utf-8")
    parts = text.split("---")
    meta = yaml.safe_load(parts[1]) if len(parts) >= 3 else {}

    pattern = None
    for c in meta.get("configs", []) or []:
        if c.get("config_name") != config:
            continue
        dfs = c.get("data_files")
        if isinstance(dfs, str):
            pattern = dfs
        elif isinstance(dfs, list):
            for df in dfs:
                if df.get("split", "train") == split:
                    pattern = df.get("path")
                    break
        break
    if not pattern:
        pattern = f"data/{config}/*.parquet"   # fallback: source-named folder

    files = list_repo_files(repo, repo_type="dataset")
    shards = sorted(f for f in files
                    if f.endswith(".parquet") and fnmatch.fnmatch(f, pattern))
    if not shards:
        raise FileNotFoundError(
            f"no parquet shards for config '{config}' (pattern '{pattern}')")
    return shards


# =============================================================================
# EXTRACTION  (download shard -> pyarrow columnar -> raw bytes -> buckets)
# =============================================================================
def _passes(audit, age, cfg: SubjectBucketConfig) -> bool:
    if cfg.require_audit_approved and str(audit).strip().lower() != "approved":
        return False
    if cfg.require_age_pass and not bool(age):
        return False
    return True


def _link_or_copy(src: Path, dst: Path) -> None:
    """Hardlink dst -> src (no extra bytes, same volume); fall back to a copy.
    Overwrite-safe: removes an existing dst first so a re-run can't collide (a stale
    hardlink would make os.link raise FileExistsError and copyfile raise SameFileError)."""
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copyfile(src, dst)


def export_subject_buckets(cfg: SubjectBucketConfig) -> dict:
    """Fast columnar subject-bucket extraction. Two passes over the LOCAL parquet:
       (1) caption columns only -> plan buckets (no image bytes touched),
       (2) image column -> write ONLY the kept images directly into bucket dirs,
           hardlinking the animetimm caption variant. Returns a report dict."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    out_root = Path(cfg.out_root).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    vlm_col = cfg.caption_columns[0]
    anime_col = cfg.caption_columns[1] if len(cfg.caption_columns) > 1 else None

    shards = _resolve_shards(cfg.repo, cfg.config, cfg.split)
    log.info("config %s -> %d parquet shard(s)", cfg.config, len(shards))

    mixed = cfg.caption_mode is CaptionMode.MIXED

    # ---- PASS 1: caption columns -> per-image features (no image bytes) ----
    # Download shards LAZILY (stop at limit) — the full config can be >100 GB.
    cap_cols = [vlm_col] + ([anime_col] if anime_col else []) + \
               ["id", "audit", "age_classifier_pass"]
    keep: dict = {}                 # idslug -> (vlm_caption, anime_caption|None)
    records_of: dict = {}           # idslug -> ImageRecord
    cap_stats: Counter = Counter()
    used_local: list[str] = []
    scanned = n_accept = 0
    done = False
    for shard in shards:
        local = hf_hub_download(cfg.repo, shard, repo_type="dataset")
        used_local.append(local)
        pf = pq.ParquetFile(local)
        for batch in pf.iter_batches(batch_size=cfg.batch_size, columns=cap_cols):
            d = batch.to_pydict()
            ids, vlms = d["id"], d[vlm_col]
            animes = d[anime_col] if anime_col else [None] * len(ids)
            audits, ages = d["audit"], d["age_classifier_pass"]
            for i in range(len(ids)):
                scanned += 1
                if not _passes(audits[i], ages[i], cfg):
                    continue
                vlm = real_caption(vlms[i])
                if vlm is None:
                    cap_stats["no_vlm"] += 1
                    continue
                anime = real_caption(animes[i]) if anime_col else None
                rec = extract_features(vlm, anime, prefer=cfg.prefer_attr_source,
                                       head_noun=cfg.head_noun)
                if rec.vlm_subject is None:
                    cap_stats["no_subject"] += 1
                    continue
                if anime:
                    cap_stats["with_anime"] += 1
                idslug = slug(ids[i])
                keep[idslug] = (vlm, anime)
                records_of[idslug] = rec
                cap_stats["accepted"] += 1
                n_accept += 1
                if cfg.limit and n_accept >= cfg.limit:
                    done = True
                    break
            if done:
                break
        if done:
            break

    # ---- PLAN over the UNION of vlm (+ animetimm, unless MIXED) subjects ----
    subj_counter: Counter = Counter()
    for rec in records_of.values():
        if rec.vlm_subject:
            subj_counter[rec.vlm_subject] += 1
        if not mixed and rec.anime_subject:
            subj_counter[rec.anime_subject] += 1
    plan = (plan_buckets_semantic(subj_counter, cfg) if cfg.use_semantic
            else plan_buckets(list(subj_counter.elements()), cfg))

    # ---- SPLIT oversized buckets, per caption stream (disjoint from grouping) ----
    override: dict = {}
    oversized: list = []
    if cfg.split_oversized:
        ov, _, ovr = split_oversized_buckets(records_of, plan, cfg,
                                              stream="vlm", subject_attr="vlm_subject")
        override.update(ov)
        oversized += ovr
        if not mixed:
            ov, _, ovr = split_oversized_buckets(records_of, plan, cfg,
                                                 stream="anime", subject_attr="anime_subject")
            override.update(ov)
            oversized += ovr

    def _bucket(idslug, subject, stream):
        b = override.get((idslug, stream))
        if b is None and subject is not None:
            b = plan.mapping.get(subject)
        return slug(b) if b else None

    # ---- per-image write spec ----
    final: dict = {}
    dropped = 0
    for idslug, rec in records_of.items():
        vb = _bucket(idslug, rec.vlm_subject, "vlm")
        if vb is None:
            dropped += 1
            continue
        vlm, anime = keep[idslug]
        spec = {"vlm": vb, "vlm_cap": vlm}
        if mixed:
            spec["mixed_caps"] = _mixed_captions(vlm, anime, cfg)
        elif anime and rec.anime_subject:
            ab = _bucket(idslug, rec.anime_subject, "anime")
            if ab is not None:
                spec["anime"], spec["anime_cap"] = ab, anime
        final[idslug] = spec

    # ---- PASS 2: write ONLY kept images, routed per caption mode ----
    bucket_counts: Counter = Counter()
    captions_accum: dict = defaultdict(dict)     # mixed bucket dir -> {filename: [caps]}
    remaining = set(final)
    for local in used_local:
        if not remaining:
            break
        pf = pq.ParquetFile(local)
        for batch in pf.iter_batches(batch_size=cfg.batch_size, columns=["image", "id"]):
            if not remaining:
                break
            d = batch.to_pydict()
            for i in range(len(d["id"])):
                idslug = slug(d["id"][i])
                if idslug not in remaining:
                    continue
                remaining.discard(idslug)
                spec = final[idslug]
                ext, raw = _img_bytes(d["image"][i])
                if raw is None:
                    cap_stats["no_image"] += 1
                    continue
                if mixed:                                      # one inode + captions.json
                    bdir = out_root / "mixed" / spec["vlm"]
                    bdir.mkdir(parents=True, exist_ok=True)
                    img = bdir / f"{idslug}{ext}"
                    if img.exists():
                        img.unlink()
                    img.write_bytes(raw)
                    captions_accum[str(bdir)][f"{idslug}{ext}"] = spec["mixed_caps"]
                    bucket_counts[f"mixed/{spec['vlm']}"] += len(spec["mixed_caps"])
                else:                                          # vlm tree + animetimm tree
                    vb = out_root / "vlm" / spec["vlm"]
                    vb.mkdir(parents=True, exist_ok=True)
                    primary = vb / f"{idslug}{ext}"
                    if primary.exists():
                        primary.unlink()
                    primary.write_bytes(raw)
                    (vb / f"{idslug}.txt").write_text(spec["vlm_cap"], encoding="utf-8")
                    bucket_counts[f"vlm/{spec['vlm']}"] += 1
                    if "anime" in spec:
                        ab = out_root / "animetimm" / spec["anime"]
                        ab.mkdir(parents=True, exist_ok=True)
                        _link_or_copy(primary, ab / f"{idslug}{ext}")    # hardlink, no pixels
                        (ab / f"{idslug}.txt").write_text(spec["anime_cap"], encoding="utf-8")
                        bucket_counts[f"animetimm/{spec['anime']}"] += 1

    for bdir_str, mapping in captions_accum.items():           # flush MIXED captions.json
        (Path(bdir_str) / "captions.json").write_text(
            json.dumps(mapping, ensure_ascii=False), encoding="utf-8")

    return {
        "out_root": str(out_root),
        "caption_mode": cfg.caption_mode.value,
        "shards_used": len(used_local),
        "scanned": scanned,
        "accepted_images": len(keep),
        "dropped_images": dropped,
        "caption_stats": dict(cap_stats),
        "final_buckets": dict(bucket_counts.most_common()),
        "n_final_buckets": len(bucket_counts),
        "raw_subjects": len(plan.raw_counts),
        "max_bucket_size": max_bucket_size(len(records_of), cfg.max_bucket_size),
        "oversized_subjects": oversized,
        "merge_actions": plan.actions,
    }


def build_mode_tomls(out_root: str | Path, cfg: SubjectBucketConfig, *,
                     configs_dir: str | Path, resolutions=(1024,),
                     balance_alpha: float = 0.5, cap_mult: float = 1.25,
                     max_repeats: int = 8, target_effective: int | None = None) -> list:
    """Build the balanced dataset toml(s) for the extracted trees, per caption mode:
       MIXED        -> one toml over mixed/* (online_captions=true)
       SEPARATE     -> one toml over vlm/* + animetimm/* (global shuffle)
       BEFORE_AFTER -> TWO tomls: dataset_vlm.toml + dataset_animetimm.toml (sequential phases)
    Returns the written toml paths."""
    from . import build_multiconcept_dataset as _b
    out_root = Path(out_root).expanduser().resolve()
    cdir = Path(configs_dir).expanduser()
    cdir.mkdir(parents=True, exist_ok=True)

    def _build(roots, out_path: Path, *, online: bool) -> Path | None:
        bcfg = _b.BaseConfig(resolutions=list(resolutions), balance_alpha=balance_alpha,
                             cap_mult=cap_mult, max_repeats=max_repeats,
                             target_effective=target_effective, online_captions=online)
        concepts: list = []
        for r in roots:
            if Path(r).is_dir():
                concepts += _b.discover_concepts(Path(r), bcfg)
        if not concepts:
            return None
        _b.compute_repeats(concepts, bcfg)
        out_path.write_text(_b.render_toml(concepts, bcfg), encoding="utf-8")
        return out_path

    if cfg.caption_mode is CaptionMode.MIXED:
        p = _build([out_root / "mixed"], cdir / "anima_dataset.toml", online=True)
        return [p] if p else []
    if cfg.caption_mode is CaptionMode.BEFORE_AFTER:
        out = []
        for tree, name in (("vlm", "dataset_vlm.toml"), ("animetimm", "dataset_animetimm.toml")):
            p = _build([out_root / tree], cdir / name, online=False)
            if p:
                out.append(p)
        return out
    # SEPARATE
    p = _build([out_root / "vlm", out_root / "animetimm"], cdir / "anima_dataset.toml",
               online=False)
    return [p] if p else []
