"""Reconstruct-from-index tests — the byte-identity round-trip that guarantees the cache
fingerprint survives (extract -> wipe images+txt -> rebuild from source -> identical tree).
Offline: a fake source parquet + monkeypatched hf_hub_download (no network/model/GPU)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

from geolip_anima_trainer import subject_buckets as sb


def _png(i: int) -> bytes:
    return b"\x89PNG\r\n\x1a\n" + bytes([i % 251]) * 64    # valid PNG magic + distinct payload


def _vlm(sub: str) -> str:
    return json.dumps({"subjects": [{"name": sub, "attributes": []}], "actions": [], "setting": ""})


def _write_source(path: Path, rows: list[dict]) -> None:
    img_t = pa.struct([("bytes", pa.binary()), ("path", pa.string())])
    tbl = pa.table({
        "id": pa.array([r["id"] for r in rows]),
        "caption_vlm_json": pa.array([r["vlm"] for r in rows]),
        "caption_animetimm_json": pa.array([r["anime"] for r in rows]),
        "image": pa.array([{"bytes": r["img"], "path": f"{r['id']}.png"} for r in rows], type=img_t),
        "audit": pa.array(["approved"] * len(rows)),
        "age_classifier_pass": pa.array([None] * len(rows), type=pa.bool_()),
    })
    pq.write_table(tbl, str(path))


def _snapshot(root: Path) -> dict[str, bytes]:
    out = {}
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.name != sb.INDEX_NAME:
            out[str(p.relative_to(root)).replace("\\", "/")] = p.read_bytes()
    return out


@pytest.fixture
def fake_hub(tmp_path, monkeypatch):
    rows = [{"id": f"img-{i:03d}", "vlm": _vlm(["dog", "dog", "cat", "dog", "cat"][i]),
             "anime": _vlm(["dog", "dog", "cat", "dog", "cat"][i]), "img": _png(i)} for i in range(5)]
    src = tmp_path / "src" / "0000.parquet"
    src.parent.mkdir(parents=True)
    _write_source(src, rows)
    import huggingface_hub as hf
    monkeypatch.setattr(sb, "_resolve_shards", lambda repo, config, split: ["data/0000.parquet"])
    monkeypatch.setattr(hf, "hf_hub_download", lambda repo, shard, **k: str(src))
    return src


def _cfg(out_root):
    return sb.SubjectBucketConfig(
        repo="fake/src", config="cfg", out_root=str(out_root), limit=None,
        caption_mode=sb.CaptionMode.BEFORE_AFTER, use_semantic=False,   # difflib -> no model download
        min_bucket_size=1, min_final_group_size=1, keep_small=True, split_oversized=False,
        download_workers=1, progress_every=0)


def test_extract_then_reconstruct_is_byte_identical(tmp_path, fake_hub):
    out_root = tmp_path / "subjects"
    rep = sb.export_subject_buckets(_cfg(out_root))
    assert rep["accepted_images"] == 5 and Path(rep["index_path"]).is_file()
    before = _snapshot(out_root)
    assert before, "extraction wrote no files"

    # simulate a Colab reset: drop every image + .txt, keep ONLY the index (what HF stored).
    for p in list(out_root.rglob("*")):
        if p.is_file() and p.name != sb.INDEX_NAME:
            p.unlink()

    stats = sb.reconstruct_from_index(out_root, download_workers=1, verify_sha=True)
    assert stats["reconstructed"] == 5 and stats["missing_from_source"] == 0
    assert stats["sha_mismatch"] == 0
    after = _snapshot(out_root)
    assert after == before          # same file SET + byte-identical images + captions (-> same fingerprint)


def test_index_header_and_records(tmp_path, fake_hub):
    out_root = tmp_path / "subjects"
    sb.export_subject_buckets(_cfg(out_root))
    lines = [json.loads(x) for x in (out_root / sb.INDEX_NAME).read_text(encoding="utf-8").splitlines() if x.strip()]
    header = lines[0]
    assert header.get("_index_meta") and header["repo"] == "fake/src" and header["caption_mode"] == "before_after"
    assert str(out_root.resolve()) == header["out_root"]
    recs = lines[1:]
    assert len(recs) == 5
    for r in recs:
        assert r["id"] and r["slug"] and r["shard"] and r["ext"] == ".png" and r["sha256"]
        assert r["vlm_bucket"] and r["vlm_cap"]
        assert r["shard"] in header["shards"]


def test_reconstruct_only_missing_is_idempotent(tmp_path, fake_hub, monkeypatch):
    out_root = tmp_path / "subjects"
    sb.export_subject_buckets(_cfg(out_root))
    # nothing wiped -> a second reconstruct should fetch nothing
    calls = []
    import huggingface_hub as hf
    orig = hf.hf_hub_download
    monkeypatch.setattr(hf, "hf_hub_download", lambda *a, **k: (calls.append(1), orig(*a, **k))[1])
    stats = sb.reconstruct_from_index(out_root, download_workers=1)
    assert stats["reconstructed"] == 0 and stats["skipped_present"] == 5
    assert calls == []              # no shard downloaded when everything is already on disk


def test_reconstruct_out_root_mismatch_raises(tmp_path, fake_hub):
    out_root = tmp_path / "subjects"
    sb.export_subject_buckets(_cfg(out_root))
    moved = tmp_path / "relocated"
    moved.mkdir()
    import shutil
    shutil.copy(out_root / sb.INDEX_NAME, moved / sb.INDEX_NAME)
    with pytest.raises(ValueError, match="ABSOLUTE paths"):
        sb.reconstruct_from_index(moved)     # index out_root != restore dir -> loud, not a silent wipe


def test_reconstruct_missing_source_row_is_not_fatal(tmp_path, fake_hub):
    out_root = tmp_path / "subjects"
    sb.export_subject_buckets(_cfg(out_root))
    # corrupt the index: add an id that doesn't exist in source
    idx = out_root / sb.INDEX_NAME
    lines = idx.read_text(encoding="utf-8").splitlines()
    ghost = json.loads(lines[1]); ghost["id"] = "ghost-999"; ghost["slug"] = "ghost_999"
    idx.write_text("\n".join(lines + [json.dumps(ghost)]) + "\n", encoding="utf-8")
    for p in list(out_root.rglob("*")):
        if p.is_file() and p.name != sb.INDEX_NAME:
            p.unlink()
    stats = sb.reconstruct_from_index(out_root, download_workers=1)
    assert stats["missing_from_source"] == 1 and stats["reconstructed"] == 5   # the 5 real ones still land
