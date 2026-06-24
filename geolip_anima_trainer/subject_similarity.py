#!/usr/bin/env python3
"""
subject_similarity.py — pluggable similarity backends for subject-phrase grouping.

Three tiers, best-available wins (see subject_buckets.make_sim_fn):
  A. sentence-transformers (real semantics)  — OPTIONAL extra `[similarity]`.
       default model all-MiniLM-L6-v2 (~90 MB first use); pass nomic-embed-text-v1 to
       reuse a model already cached on F: (zero download). Groups true synonyms
       (truck~car~automobile, dancer~performer) that string methods miss.
  B. char-trigram cosine (numpy only)        — zero install; MORPHOLOGICAL, not
       semantic (catches guitarist~guitar; misses truck~car). A safe default when the
       extra isn't installed — better than difflib, never claims to be semantic.
  C. difflib ratio                            — stdlib fallback (lexical only).

Nothing here raises on a missing optional dep; callers degrade gracefully.
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache

log = logging.getLogger("anima.subjects.sim")

# Already cached on F: -> selecting it downloads 0 bytes (needs trust_remote_code).
PREFERRED_LOCAL = "nomic-ai/nomic-embed-text-v1"
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"   # ~90 MB on first use


# =============================================================================
# Tier A — sentence-transformers (optional)
# =============================================================================
def have_semantic_backend() -> bool:
    try:
        import sentence_transformers  # noqa: F401
        return True
    except Exception:  # noqa: BLE001 — ImportError or a broken partial install
        return False


@lru_cache(maxsize=2)
def _load(model_id: str):
    from sentence_transformers import SentenceTransformer
    # trust_remote_code: nomic ships custom modeling; attn_implementation=eager:
    # NEVER flash-attn on this lineage (broken on sm_120, absent here). device auto.
    return SentenceTransformer(
        model_id,
        trust_remote_code=True,
        model_kwargs={"attn_implementation": "eager"},
    )


def embed_subjects(phrases: list[str], *, model_id: str = DEFAULT_MODEL):
    """Embed the (deduped, few-hundred) phrase set -> L2-normalized (N, D) float32."""
    model = _load(model_id)
    return model.encode(phrases, batch_size=256, normalize_embeddings=True,
                        convert_to_numpy=True, show_progress_bar=False).astype("float32",
                                                                              copy=False)


# =============================================================================
# Tier B — char-trigram hashed bag-of-ngrams cosine (numpy only)
# =============================================================================
def trigram_embed(phrases: list[str], *, dim: int = 1024):
    """Hashed char-trigram bag (L2-normalized rows). Morphology, not semantics."""
    import numpy as np
    mat = np.zeros((len(phrases), dim), dtype="float32")
    for r, p in enumerate(phrases):
        s = f"^^{re.sub(r'[^a-z0-9 ]+', '', (p or '').lower())}$$"
        for i in range(len(s) - 2):
            mat[r, hash(s[i:i + 3]) % dim] += 1.0
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


def cosine(a, b):
    """Cosine similarity matrix for L2-normalized row matrices a (n,D), b (m,D)."""
    return a @ b.T
