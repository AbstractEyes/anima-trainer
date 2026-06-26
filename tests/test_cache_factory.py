"""cache_factory tests — scratch detect, config, symlink, and the runner orchestration.
All offline: no GPU / network / HF; the heavy api calls are monkeypatched, the config engine runs real."""
from __future__ import annotations

import os
import sqlite3
from types import SimpleNamespace

import pytest

from geolip_anima_trainer import cache_factory as cf


# =============================================================================
# helpers
# =============================================================================
def _fake_disk_usage(sizes):
    """shutil.disk_usage stand-in: total = sizes[path], else raise (unmounted)."""
    def _du(path):
        if path in sizes:
            return SimpleNamespace(total=sizes[path], used=0, free=sizes[path])
        raise OSError("no such mount")
    return _du


# =============================================================================
# find_scratch
# =============================================================================
def test_find_scratch_picks_biggest_writable(monkeypatch):
    mounts = ["/", "/content/drive", "/mnt/persist", "/mnt/scratch", "/proc"]
    sizes = {"/": 50 * 1024**3, "/content/drive": 999 * 1024**3,    # drive excluded despite size
             "/mnt/persist": 235 * 1024**3, "/mnt/scratch": 368 * 1024**3}
    monkeypatch.setattr(cf.shutil, "disk_usage", _fake_disk_usage(sizes))
    monkeypatch.setattr(cf.os, "access", lambda p, m: True)
    hit = cf.find_scratch(300, mounts=mounts)
    assert hit is not None and hit[0] == "/mnt/scratch"           # the 368 GB NVMe, not Drive/persist
    assert hit[1] == 368 * 1024**3


def test_find_scratch_none_when_all_too_small(monkeypatch):
    mounts = ["/", "/mnt/persist"]
    monkeypatch.setattr(cf.shutil, "disk_usage",
                        _fake_disk_usage({"/": 50 * 1024**3, "/mnt/persist": 235 * 1024**3}))
    monkeypatch.setattr(cf.os, "access", lambda p, m: True)
    assert cf.find_scratch(300, mounts=mounts) is None           # nothing >=300 GB -> fall back


# =============================================================================
# FactoryConfig
# =============================================================================
def test_factory_config_env_and_overrides(monkeypatch):
    monkeypatch.setenv("ANIMA_CACHE_REPO", "me/from-env")
    monkeypatch.setenv("ANIMA_LIMIT", "5000")
    c = cf.FactoryConfig.from_env()
    assert c.cache_repo == "me/from-env" and c.limit == 5000
    c2 = cf.FactoryConfig.from_env(limit=None, cache_repo="me/override")   # overrides win
    assert c2.limit is None and c2.cache_repo == "me/override"
    monkeypatch.setenv("ANIMA_LIMIT", "none")
    assert cf.FactoryConfig.from_env().limit is None              # "none" -> full set
    with pytest.raises(TypeError):
        cf.FactoryConfig.from_env(bogus=1)


# =============================================================================
# _refresh_symlink
# =============================================================================
def test_refresh_symlink_respects_real_dir(tmp_path):
    real = tmp_path / "real_data"
    real.mkdir()
    (real / "keep.txt").write_text("x", encoding="utf-8")
    assert cf._refresh_symlink(str(real), str(tmp_path / "target")) is False   # don't clobber real data
    assert (real / "keep.txt").exists()


def test_refresh_symlink_reclaims_empty_leftover_dir(tmp_path):
    target = tmp_path / "scratch"
    target.mkdir()
    link = tmp_path / "portable"
    link.mkdir()                                            # empty leftover from a prior session
    try:
        os.symlink(str(target), str(tmp_path / "_probe"))
    except OSError:
        pytest.skip("symlinks not permitted in this environment")
    assert cf._refresh_symlink(str(link), str(target)) is True   # empty dir reclaimed, not a silent fail
    assert os.path.realpath(link) == os.path.realpath(target)


def test_refresh_symlink_creates_and_repoints(tmp_path):
    t1, t2 = tmp_path / "scratch1", tmp_path / "scratch2"
    t1.mkdir(); t2.mkdir()
    link = tmp_path / "portable"
    try:
        os.symlink(str(t1), str(tmp_path / "_probe"))            # symlink perms? (Windows needs dev-mode/admin)
    except OSError:
        pytest.skip("symlinks not permitted in this environment")
    assert cf._refresh_symlink(str(link), str(t1)) is True
    assert os.path.realpath(link) == os.path.realpath(t1)
    assert cf._refresh_symlink(str(link), str(t2)) is True       # dangling/old -> repoint
    assert os.path.realpath(link) == os.path.realpath(t2)


# =============================================================================
# get_hf_token / _has_cache
# =============================================================================
def test_get_hf_token_env_fallback(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "tok-123")
    assert cf.get_hf_token() == "tok-123"                        # no Colab -> env


def test_has_cache(tmp_path):
    assert cf._has_cache(tmp_path) is False
    leaf = tmp_path / "vlm" / "dog" / "cache" / "anima" / "latents"
    leaf.mkdir(parents=True)
    con = sqlite3.connect(leaf / "metadata.db"); con.execute("CREATE TABLE t(x)"); con.commit(); con.close()
    assert cf._has_cache(tmp_path) is True


# =============================================================================
# the runner orchestration  (prepare -> build -> cache wiring)
# =============================================================================
def _factory(tmp_path, **kw):
    f = cf.CacheFactory(data_root=str(tmp_path / "dr"), repo_root=str(tmp_path / "repo"),
                        cache_repo="u/r", **kw)
    f.state.update(data_root=str(tmp_path / "dr"), subjects_root=str(tmp_path / "dr" / "anima_subjects"),
                   io_root=str(tmp_path / "dr"), hf_token="tok",
                   model_paths=SimpleNamespace(transformer_path="/m/t.safetensors",
                                               vae_path="/m/vae.safetensors", llm_path="/m/llm.safetensors"))
    os.makedirs(f.state["subjects_root"], exist_ok=True)
    return f


def test_prepare_dataset_extracts_when_no_cache(tmp_path, monkeypatch):
    f = _factory(tmp_path)
    calls = {}

    def _export(c):
        calls["export"] = c
        return {"accepted_images": 10, "dropped_images": 1, "n_final_buckets": 3, "index_path": "i"}

    def _push(*a, **k):
        calls["push"] = k.get("commit_message")

    def _prune(*a, **k):
        calls["prune"] = True
        return {"freed_bytes": 0}

    monkeypatch.setattr(cf._api, "cache_pull", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("nothing")))
    monkeypatch.setattr(cf._api, "export_subject_buckets", _export)
    monkeypatch.setattr(cf._api, "cache_push", _push)
    monkeypatch.setattr(cf._api, "prune_source_cache", _prune)
    resume = f.prepare_dataset()
    assert resume is False
    assert "export" in calls and calls["push"] == "store index (pre-cache)" and calls["prune"]


def test_prepare_dataset_resumes_when_cache_present(tmp_path, monkeypatch):
    f = _factory(tmp_path)
    leaf = tmp_path / "dr" / "anima_subjects" / "vlm" / "x" / "cache" / "anima" / "latents"
    leaf.mkdir(parents=True)
    con = sqlite3.connect(leaf / "metadata.db"); con.execute("CREATE TABLE t(x)"); con.commit(); con.close()
    monkeypatch.setattr(cf._api, "prune_source_cache", lambda *a, **k: {"freed_bytes": 0})
    monkeypatch.setattr(cf._api, "export_subject_buckets",
                        lambda c: (_ for _ in ()).throw(AssertionError("must NOT extract on resume")))
    assert f.prepare_dataset() is True                           # local cache -> resume, no extract


def test_setup_env_full_limit_without_scratch_raises(monkeypatch):
    monkeypatch.delenv("ANIMA_DATA_ROOT", raising=False)
    monkeypatch.setattr(cf, "find_scratch", lambda *a, **k: None)
    f = cf.CacheFactory(limit=None, data_root=None)        # full set + no scratch -> refuse, don't fill /content
    with pytest.raises(RuntimeError, match="local-scratch"):
        f._setup_env()


def test_cold_calls_raise_clear_setup_error():
    f = cf.CacheFactory(data_root="/x", cache_repo="u/r")  # never ran setup() -> empty state
    for call in (f.push, f.build_configs, f.run_cache):
        with pytest.raises(RuntimeError, match="setup"):
            call()


def test_build_and_run_cache_wiring(tmp_path, monkeypatch):
    f = _factory(tmp_path)
    f.state["subject_cfg"] = f._subject_cfg()
    dv = tmp_path / "dr" / "dataset_vlm.toml"; da = tmp_path / "dr" / "dataset_animetimm.toml"
    for p in (dv, da):
        p.write_text("resolutions=[1024]\n[[directory]]\npath='x'\n", encoding="utf-8")
    monkeypatch.setattr(cf._api, "build_mode_tomls", lambda *a, **k: [str(dv), str(da)])
    monkeypatch.setattr(cf._api, "validate", lambda c: None)
    monkeypatch.setattr(cf._api, "render_lora_toml", lambda c: "x")
    loras = f.build_configs()
    assert set(loras) == {"vlm", "animetimm"} and os.path.isfile(loras["vlm"])

    seen = []
    import contextlib
    monkeypatch.setattr(cf._api, "gpu_keepalive", lambda *a, **k: contextlib.nullcontext())
    monkeypatch.setattr(cf._api, "cache", lambda toml, **k: seen.append((toml, k.get("backup_repo"), k.get("trust_cache"))) or 0)
    pushes = []
    monkeypatch.setattr(cf._api, "cache_push", lambda *a, **k: pushes.append(k.get("commit_message")))
    held = []
    monkeypatch.setattr(cf._api, "keepalive", lambda **k: held.append(k))
    f.run_cache(hold=False)
    assert len(seen) == 2 and all(s[1] == "u/r" for s in seen) and all(s[2] is False for s in seen)
    assert pushes == ["post-cache full"] and held == []          # final push fired, no hold
    f.run_cache(hold=True)
    assert held                                                  # hold=True -> keepalive engaged
