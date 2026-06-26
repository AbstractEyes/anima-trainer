"""trainer_runner tests — config, env, pull-or-resume, build wiring, detached train/resume argv.
Offline: no GPU/network/HF; heavy api calls mocked, the config engine runs real."""
from __future__ import annotations

import os
import sqlite3
from types import SimpleNamespace

import pytest

from geolip_anima_trainer import trainer_runner as tr


def test_trainer_config_env_and_overrides(monkeypatch):
    monkeypatch.setenv("ANIMA_CACHE_REPO", "me/cache")
    monkeypatch.setenv("ANIMA_BACKUP_REPO", "me/loras")
    monkeypatch.setenv("ANIMA_LIMIT", "none")
    c = tr.TrainerConfig.from_env()
    assert c.cache_repo == "me/cache" and c.backup_repo == "me/loras" and c.limit is None
    c2 = tr.TrainerConfig.from_env(num_gpus=4, activation_checkpointing=False)
    assert c2.num_gpus == 4 and c2.activation_checkpointing is False
    with pytest.raises(TypeError):
        tr.TrainerConfig.from_env(bogus=1)


def test_setup_env_pins_data_root_and_hf_home(tmp_path, monkeypatch):
    for k in ("HF_HOME", "HF_DATASETS_CACHE", "TMPDIR", "ANIMA_DATA_ROOT"):
        monkeypatch.setenv(k, "placeholder")               # so teardown restores; setup() overrides raw
    dr = str(tmp_path / "anima_data")
    t = tr.TrainerRunner(data_root=dr, repo_root=str(tmp_path / "repo"))
    t._setup_env()
    assert os.environ["HF_HOME"].startswith(dr) and os.environ["ANIMA_DATA_ROOT"] == dr
    assert t.state["subjects_root"] == f"{dr}/anima_subjects"


def _runner(tmp_path, **kw):
    t = tr.TrainerRunner(data_root=str(tmp_path / "dr"), repo_root=str(tmp_path / "repo"),
                         cache_repo="u/r", **kw)
    t.state.update(data_root=str(tmp_path / "dr"), subjects_root=str(tmp_path / "dr" / "anima_subjects"),
                   hf_token="tok",
                   model_paths=SimpleNamespace(transformer_path="/m/t", vae_path="/m/v", llm_path="/m/l"))
    os.makedirs(t.state["subjects_root"], exist_ok=True)
    return t


def _seed_cache(subj):
    leaf = subj / "vlm" / "x" / "cache" / "anima" / "latents"
    leaf.mkdir(parents=True)
    con = sqlite3.connect(leaf / "metadata.db"); con.execute("CREATE TABLE t(x)"); con.commit(); con.close()


def test_prepare_dataset_requires_cache_repo(tmp_path):
    t = _runner(tmp_path)
    t.cfg.cache_repo = None
    with pytest.raises(RuntimeError, match="cache_repo"):
        t.prepare_dataset()


def test_prepare_dataset_pulls_then_prunes(tmp_path, monkeypatch):
    t = _runner(tmp_path)
    subj = tmp_path / "dr" / "anima_subjects"
    pulled = {}

    def _pull(s, repo, **k):
        _seed_cache(subj)                                   # cache_pull materializes the cache+images
        pulled["ok"] = True

    monkeypatch.setattr(tr._api, "cache_pull", _pull)
    monkeypatch.setattr(tr._api, "prune_source_cache", lambda *a, **k: {"freed_bytes": 5e9})
    assert t.prepare_dataset() is True and pulled["ok"] and t.state["resume"] is True


def test_prepare_dataset_errors_if_pull_yields_no_cache(tmp_path, monkeypatch):
    t = _runner(tmp_path)
    monkeypatch.setattr(tr._api, "cache_pull", lambda *a, **k: None)   # repo exists but empty -> no cache
    monkeypatch.setattr(tr._api, "prune_source_cache", lambda *a, **k: {"freed_bytes": 0})
    with pytest.raises(RuntimeError, match="no cache"):
        t.prepare_dataset()


def test_prepare_dataset_skips_pull_when_cache_present(tmp_path, monkeypatch):
    t = _runner(tmp_path)
    _seed_cache(tmp_path / "dr" / "anima_subjects")
    monkeypatch.setattr(tr._api, "cache_pull", lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not pull")))
    monkeypatch.setattr(tr._api, "prune_source_cache", lambda *a, **k: {"freed_bytes": 0})
    assert t.prepare_dataset() is True


def test_build_configs_wiring(tmp_path, monkeypatch):
    t = _runner(tmp_path)
    t.state["subject_cfg"] = t._subject_cfg()
    dv = tmp_path / "dr" / "dataset_vlm.toml"; da = tmp_path / "dr" / "dataset_animetimm.toml"
    for p in (dv, da):
        p.write_text("resolutions=[1024]\n[[directory]]\npath='x'\n", encoding="utf-8")
    monkeypatch.setattr(tr._api, "build_mode_tomls", lambda *a, **k: [str(dv), str(da)])
    monkeypatch.setattr(tr._api, "validate", lambda c: None)
    monkeypatch.setattr(tr._api, "render_lora_toml", lambda c: "x")
    loras = t.build_configs()
    assert set(loras) == {"vlm", "animetimm"} and os.path.isfile(loras["vlm"])


def _capture_launch(seen):
    def _fake(argv, log):
        seen["argv"] = argv
        return {"pid": 7, "log": log}
    return _fake


def test_train_detached_builds_cli_argv(tmp_path, monkeypatch):
    t = _runner(tmp_path, num_gpus=2)
    t.state["lora_tomls"] = {"vlm": "/c/lora_vlm.toml", "animetimm": "/c/lora_animetimm.toml"}
    seen = {}
    monkeypatch.setattr(t, "_launch_detached", _capture_launch(seen))
    t.train()
    a = seen["argv"]
    assert "train-before-after" in a and "/c/lora_vlm.toml" in a and "--num-gpus" in a and "2" in a


def test_train_blocking_calls_api(tmp_path, monkeypatch):
    t = _runner(tmp_path)
    t.state["lora_tomls"] = {"vlm": "v.toml", "animetimm": "a.toml"}
    called = {}

    def _tba(v, a, **k):
        called["args"] = (v, a, k.get("num_gpus"))
        return 0

    monkeypatch.setattr(tr._api, "train_before_after", _tba)
    assert t.train(detached=False) == 0 and called["args"][0] == "v.toml"


def test_resume_builds_resume_argv(tmp_path, monkeypatch):
    t = _runner(tmp_path)
    t.state["lora_tomls"] = {"vlm": "v.toml", "animetimm": "a.toml"}
    seen = {}
    monkeypatch.setattr(t, "_launch_detached", _capture_launch(seen))
    t.resume("vlm")
    a = seen["argv"]
    assert "train" in a and "--config" in a and "v.toml" in a and "--resume" in a


def test_cold_calls_raise_setup_error(tmp_path):
    t = tr.TrainerRunner(data_root=str(tmp_path), cache_repo="u/r")
    for call in (t.build_configs, t.train, t.resume):
        with pytest.raises(RuntimeError, match="setup"):
            call()
