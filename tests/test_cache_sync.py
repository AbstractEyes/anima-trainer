"""cache_sync tests — path targeting, fingerprint read, dry-run, round-trip, periodic push.
All offline (no network/HF/GPU): the hub calls are monkeypatched with a local fake."""
from __future__ import annotations

import shutil
import sqlite3
from pathlib import Path

from geolip_anima_trainer import cache_sync as cs


def _make_fp_db(path: Path, fp: str | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE fingerprint(value)")
    if fp is not None:
        con.execute("INSERT INTO fingerprint VALUES (?)", (fp,))
    con.execute("CREATE TABLE items(shard, shard_index)")
    con.commit()
    con.close()


def _toml(tmp_path: Path, *paths: str) -> Path:
    body = "resolutions = [1024]\n" + "".join(
        f"[[directory]]\npath = '{p}'\n" for p in paths)
    p = tmp_path / "dataset.toml"
    p.write_text(body, encoding="utf-8")
    return p


def test_path_targeting(tmp_path: Path):
    ds = _toml(tmp_path, "/data/sub/vlm/dog", "/data/sub/vlm/cat")
    roots = [str(r).replace("\\", "/") for r in cs.cache_targets_from_toml(ds)]
    assert roots == ["/data/sub/vlm/dog/cache/anima", "/data/sub/vlm/cat/cache/anima"]
    assert str(cs.out_root_of(ds)).replace("\\", "/").endswith("/data/sub/vlm")


def test_read_cache_fingerprints(tmp_path: Path):
    root = tmp_path / "vlm" / "dog" / "cache" / "anima" / "cache_1024x1024"
    _make_fp_db(root / "latents" / "metadata.db", "fp-abc")
    _make_fp_db(root / "text_embeddings_1" / "metadata.db", None)   # present, no row
    fps = cs.read_cache_fingerprints([tmp_path])
    vals = set(fps.values())
    assert "fp-abc" in vals and None in vals
    # never raises on a non-db / missing dir
    assert cs.read_cache_fingerprints([tmp_path / "nope"]) == {}


def test_dry_run_touches_no_network(tmp_path: Path, monkeypatch):
    import huggingface_hub as hf
    # any hub access would explode -> proves dry-run short-circuits before importing/using it
    monkeypatch.setattr(hf, "HfApi", lambda *a, **k: (_ for _ in ()).throw(AssertionError("net!")))
    monkeypatch.setattr(hf, "create_repo", lambda *a, **k: (_ for _ in ()).throw(AssertionError("net!")))
    monkeypatch.setattr(hf, "snapshot_download", lambda *a, **k: (_ for _ in ()).throw(AssertionError("net!")))
    assert cs.sync_up(tmp_path, "u/r", dry_run=True) == "https://huggingface.co/datasets/u/r"
    assert cs.sync_down(tmp_path / "x", "u/r", dry_run=True).endswith("x")


def test_round_trip_byte_identity(tmp_path: Path, monkeypatch):
    import huggingface_hub as hf
    remote = tmp_path / "remote"

    src = tmp_path / "subjects"
    leaf = src / "vlm" / "dog" / "cache" / "anima" / "cache_1024x1024" / "latents"
    leaf.mkdir(parents=True)
    (leaf / "shard_0.bin").write_bytes(bytes(range(256)) * 400)     # binary payload
    _make_fp_db(leaf / "metadata.db", "fp-xyz")

    class FakeApi:
        def __init__(self, token=None):
            pass
        def upload_folder(self, *, folder_path, repo_id, repo_type, path_in_repo=".",
                          allow_patterns=None, ignore_patterns=None, commit_message=None):
            remote.mkdir(exist_ok=True)
            shutil.copytree(folder_path, remote, dirs_exist_ok=True)

    def fake_snapshot(*, repo_id, repo_type, local_dir, token=None,
                      allow_patterns=None, ignore_patterns=None):
        Path(local_dir).mkdir(parents=True, exist_ok=True)
        shutil.copytree(remote, local_dir, dirs_exist_ok=True)
        return local_dir

    monkeypatch.setattr(hf, "create_repo", lambda *a, **k: None)
    monkeypatch.setattr(hf, "HfApi", FakeApi)
    monkeypatch.setattr(hf, "snapshot_download", fake_snapshot)

    cs.sync_up(src, "u/r", token="t")
    dst = tmp_path / "restored"
    cs.sync_down(dst, "u/r", token="t")

    rel = "vlm/dog/cache/anima/cache_1024x1024/latents/shard_0.bin"
    assert (dst / rel).read_bytes() == (src / rel).read_bytes()    # byte-identical shard
    assert "fp-xyz" in set(cs.read_cache_fingerprints([dst]).values())


def test_sync_up_excludes_images_and_txt(tmp_path: Path, monkeypatch):
    import huggingface_hub as hf
    captured = {}

    class FakeApi:
        def __init__(self, token=None):
            pass
        def upload_folder(self, *, folder_path, repo_id, repo_type, path_in_repo=".",
                          allow_patterns=None, ignore_patterns=None, commit_message=None):
            captured["allow"], captured["ignore"] = allow_patterns, ignore_patterns

    monkeypatch.setattr(hf, "create_repo", lambda *a, **k: None)
    monkeypatch.setattr(hf, "HfApi", FakeApi)

    cs.sync_up(tmp_path, "u/r", include_dataset=True)          # freeze: everything regenerable excluded
    ig = captured["ignore"]
    assert "**/*.txt" in ig and "**/*.arrow" in ig and "**/*-journal" in ig
    assert any("png" in p for p in ig) and any("webp" in p for p in ig)

    cs.sync_up(tmp_path, "u/r", include_dataset=False)         # periodic: ONLY metadata.db + *.bin + index
    al = captured["allow"]
    assert "index.jsonl" in al
    assert any(p.endswith("metadata.db") for p in al) and any(p.endswith("*.bin") for p in al)
    assert not any("*.arrow" in p for p in al)                 # the 30k regenerable files are NOT pushed


def test_periodic_pusher_triggers(tmp_path: Path):
    calls = []
    user_seen = []

    def fake_up(folder, repo_id, *, token=None, include_dataset=True, commit_message="", **k):
        assert include_dataset is False                     # periodic pushes are cache-only
        calls.append(commit_message)

    on_update, final = cs.make_periodic_pusher(
        tmp_path, "u/r", interval=100.0, user_on_update=user_seen.append, _sync_up=fake_up)

    on_update({"elapsed": 10.0, "latents": (0, 100), "text": (0, 100)})    # warmup, 0 committed
    assert calls == []                                      # nothing committed, < interval
    on_update({"elapsed": 20.0, "latents": (5, 100), "text": (0, 100)})    # a shard finalized (5>0)
    assert len(calls) == 1                                  # push on committed bump
    on_update({"elapsed": 30.0, "latents": (5, 100), "text": (0, 100)})    # no bump, < interval
    assert len(calls) == 1
    on_update({"elapsed": 200.0, "latents": (5, 100), "text": (0, 100)})   # interval fallback
    assert len(calls) == 2
    final()
    assert len(calls) == 3 and calls[-1].endswith("final")
    assert len(user_seen) == 4                              # user on_update fired every tick


def test_periodic_push_failure_never_propagates(tmp_path: Path):
    def boom(*a, **k):
        raise RuntimeError("hub down")
    on_update, final = cs.make_periodic_pusher(tmp_path, "u/r", interval=1.0, _sync_up=boom)
    on_update({"elapsed": 5.0, "latents": (1, 1), "text": (0, 0)})   # would push -> raises internally
    final()                                                          # must not raise
