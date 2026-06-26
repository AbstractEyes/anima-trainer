"""Portable-path regression tests — the cross-box cache handoff breaks if a DATA_ROOT symlink
(factory's /workspace/anima_data -> scratch) is collapsed to the realpath in any PERSISTED path
(the toml [[directory]] path baked into diffusion-pipe's fingerprint, or the reconstruct index
out_root). These assert we keep the symlink (os.path.abspath), not resolve() it."""
from __future__ import annotations

import os
import tomllib

import pytest

from geolip_anima_trainer import build_multiconcept_dataset as bmd
from geolip_anima_trainer import subject_buckets as sb


def _symlink_or_skip(target, link):
    try:
        os.symlink(str(target), str(link))
    except OSError:
        pytest.skip("symlinks not permitted in this environment")


def test_portable_abspath_keeps_symlink(tmp_path):
    real = tmp_path / "scratch"
    real.mkdir()
    link = tmp_path / "portable"
    _symlink_or_skip(real, link)
    p = link / "anima_subjects"
    # the whole point: abspath keeps /portable/..., resolve() would collapse to /scratch/...
    assert bmd.portable_abspath(p) == str(link / "anima_subjects")
    assert os.path.realpath(p) == str(real / "anima_subjects")          # resolve() WOULD differ
    assert bmd.portable_abspath(p) != os.path.realpath(p)


def _make_bucket(root, tree, name, n=3):
    d = root / tree / name
    d.mkdir(parents=True)
    for i in range(n):
        (d / f"{i}.png").write_bytes(b"\x89PNG\r\n" + bytes(i))
        (d / f"{i}.txt").write_text("a tag, b tag", encoding="utf-8")
    return d


def test_discover_concepts_keeps_symlink_path(tmp_path):
    real = tmp_path / "scratch"
    _make_bucket(real, "vlm", "dog")
    link = tmp_path / "portable"
    _symlink_or_skip(real, link)
    cfg = bmd.BaseConfig(resolutions=[1024])
    concepts = bmd.discover_concepts(link / "vlm", cfg)
    assert concepts, "expected the dog bucket"
    paths = [c["path"] for c in concepts]
    assert all(p.startswith(str(link)) for p in paths)                 # portable symlink kept
    assert not any(str(real) == p[:len(str(real))] for p in paths)     # NOT the resolved scratch path


def test_build_mode_tomls_writes_portable_directory_path(tmp_path):
    # the [[directory]] path is baked verbatim into the latent fingerprint -> must stay portable.
    real = tmp_path / "scratch" / "anima_subjects"
    for tree in ("vlm", "animetimm"):
        _make_bucket(real, tree, "dog")
    link = tmp_path / "portable"
    _symlink_or_skip(tmp_path / "scratch", link)
    out_root = link / "anima_subjects"                                 # the SYMLINK path (as setup() would set)
    cfg = sb.SubjectBucketConfig(out_root=str(out_root), caption_mode=sb.CaptionMode.BEFORE_AFTER)
    tomls = sb.build_mode_tomls(out_root, cfg, configs_dir=str(tmp_path / "configs"), resolutions=[1024])
    assert tomls
    body = "\n".join((tomllib.loads(open(t, "rb").read().decode()) and open(t).read()) for t in tomls)
    assert str(link) in body                                           # /portable/... in the toml
    assert str((tmp_path / "scratch").resolve()) + os.sep + "anima_subjects" not in body  # not /scratch/...
