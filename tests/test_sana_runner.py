"""sana_runner tests — the built-in mood set, the verdict rule, config/env, the fork lookup, the config
wiring, the train plan, the log follower, and the bootstrap's fork support (anima_colab).
Offline: no GPU/network/HF; the config engine and the launch planner run real."""
from __future__ import annotations

import json
import os
import sys
import tomllib
from pathlib import Path

import pytest

from geolip_anima_trainer import sana_runner as sr

ROOT = Path(__file__).resolve().parents[1]


def test_mood_set_holds_a_quarter_of_the_subjects_out():
    assert len(sr.SUBJECTS) == 32 and len(sr.HELD_OUT) == 8 and len(sr.TRAIN) == 24
    assert not set(sr.HELD_OUT) & set(sr.TRAIN)
    items = sr.mood_items()
    assert len(items) == 192 and len({it["name"] for it in items}) == 192
    held = {sr.SUBJECTS[i] for i in sr.HELD_OUT}
    for it in items:
        subject = it["caption"][len("a photo of "):]
        assert it["caption"] == sr.NEUTRAL_TEMPLATE.format(s=subject) and subject not in held
        assert it["prompt"] in {t.format(s=subject) for t in sr.UPBEAT_TEMPLATES}
    assert {it["prompt"].startswith("a cheerful") for it in items} == {True, False}   # both templates used


def test_mood_outcome_rule():
    assert sr.mood_outcome([1.0 + 0.1 * (i % 3) for i in range(32)])["OUTCOME"] == "FLAVOR LORA"
    assert sr.mood_outcome([(-1) ** i * 1.0 for i in range(32)])["OUTCOME"] == "NO EFFECT"
    # mean clearly positive but only 70% of the pairs positive -> neither rule -> MIXED
    mixed = [2.0] * 21 + [-0.5] * 9
    r = sr.mood_outcome(mixed)
    assert r["OUTCOME"] == "MIXED" and r["frac_pos"] == pytest.approx(0.7)


def test_config_env_and_overrides(monkeypatch):
    monkeypatch.setenv("ANIMA_SANA_VARIANT", "1600m-1024")
    monkeypatch.setenv("ANIMA_BACKUP_REPO", "me/sana")
    c = sr.SanaConfig.from_env(epochs=3)
    assert c.variant == "1600m-1024" and c.backup_repo == "me/sana" and c.epochs == 3
    with pytest.raises(TypeError):
        sr.SanaConfig.from_env(bogus=1)
    with pytest.raises(ValueError, match="source"):
        sr.SanaRunner(source="parquet")
    with pytest.raises(ValueError, match="variant"):
        sr.SanaRunner(variant="9000m")


def _fake_fork(root: Path, *, sana: bool = True) -> Path:
    (root / "models").mkdir(parents=True)
    (root / "utils").mkdir()
    (root / "train.py").write_text("", encoding="utf-8")
    (root / "utils" / "previews.py").write_text("", encoding="utf-8")
    if sana:
        (root / "models" / "sana.py").write_text("", encoding="utf-8")
    return root


def test_point_at_fork_finds_the_bootstrap_clone_and_refuses_upstream(tmp_path, monkeypatch):
    monkeypatch.setenv("ANIMA_DIFFUSION_PIPE", "")      # unset for the lookup; teardown restores the original
    repo = tmp_path / "repo"
    _fake_fork(repo / "external" / "diffusion-pipe", sana=False)          # upstream: no Sana
    s = sr.SanaRunner(repo_root=str(repo))
    with pytest.raises(RuntimeError, match="DP_FORK_URL"):
        s._point_at_fork()
    fork = _fake_fork(repo / "external" / "diffusion-pipe-fork")
    assert Path(s._point_at_fork()) == fork
    assert Path(os.environ["ANIMA_DIFFUSION_PIPE"]) == fork


def _ready_runner(tmp_path, **kw) -> sr.SanaRunner:
    model = tmp_path / "Sana_600M_512px_diffusers"
    (model / "transformer").mkdir(parents=True)
    (model / "transformer" / "config.json").write_text(json.dumps({"sample_size": 16}), encoding="utf-8")
    data = tmp_path / "data" / "datasets" / "mood_upbeat"
    data.mkdir(parents=True)
    s = sr.SanaRunner(data_root=str(tmp_path / "data"), repo_root=str(tmp_path / "repo"), **kw)
    s.state.update(data_root=str(tmp_path / "data"), resolution=512, diffusers_path=str(model),
                   dataset_dir=str(data), n_images=192)
    return s


def test_build_configs_writes_the_registered_recipe(tmp_path):
    s = _ready_runner(tmp_path)
    lora = Path(s.build_configs())
    t = tomllib.loads(lora.read_text(encoding="utf-8"))
    assert t["model"]["type"] == "sana" and t["model"]["shift"] == 3.0 and t["model"]["dtype"] == "bfloat16"
    assert t["model"]["diffusers_path"].endswith("Sana_600M_512px_diffusers") and "llm_adapter_lr" not in t["model"]
    assert t["optimizer"]["type"] == "adam" and t["optimizer"]["lr"] == 1e-4 and t["optimizer"]["weight_decay"] == 0
    assert t["adapter"]["rank"] == 32 and t["epochs"] == 10 and t["micro_batch_size_per_gpu"] == 4
    assert t["warmup_steps"] == 20 and t["save_every_n_epochs"] == 2
    held = [sr.NEUTRAL_TEMPLATE.format(s=sr.SUBJECTS[i]) for i in sr.HELD_OUT[:4]]
    assert t["samples"]["prompts"] == held and t["samples"]["width"] == 512 and t["samples"]["cfg"] == 4.5
    ds = tomllib.loads(Path(t["dataset"]).read_text(encoding="utf-8"))
    assert ds["resolutions"] == [512] and ds["directory"][0]["path"] == s.state["dataset_dir"]
    assert ds["shuffle_caption"] is False


def test_folder_source_needs_a_captioned_folder(tmp_path):
    s = sr.SanaRunner(source="folder", data_root=str(tmp_path))
    s.state["data_root"] = str(tmp_path)
    with pytest.raises(RuntimeError, match="dataset_dir"):
        s.prepare_dataset()
    d = tmp_path / "set"
    d.mkdir()
    (d / "a.png").write_bytes(b"")
    s2 = sr.SanaRunner(source="folder", dataset_dir=str(d), data_root=str(tmp_path))
    s2.state["data_root"] = str(tmp_path)
    with pytest.raises(RuntimeError, match="caption"):
        s2.prepare_dataset()
    (d / "a.txt").write_text("a red bicycle", encoding="utf-8")
    assert s2.prepare_dataset() == str(d) and s2.state["n_images"] == 1


def test_train_dry_run_plans_against_the_fork(tmp_path, monkeypatch):
    fork = _fake_fork(tmp_path / "fork")
    monkeypatch.setenv("ANIMA_DIFFUSION_PIPE", str(fork))
    s = _ready_runner(tmp_path)
    s.build_configs()
    plan = s.train(dry_run=True)
    assert Path(plan.train_py) == (fork / "train.py").resolve()


def test_train_detached_builds_cli_argv(tmp_path, monkeypatch):
    fork = _fake_fork(tmp_path / "fork")
    monkeypatch.setenv("ANIMA_DIFFUSION_PIPE", str(fork))
    s = _ready_runner(tmp_path, num_gpus=2)
    s.state["lora_toml"] = "/c/anima_lora.toml"
    seen = {}

    def _fake(argv, log):
        seen["argv"], seen["log"] = argv, log
        return {"pid": 7, "log": log}

    monkeypatch.setattr(s, "_launch_detached", _fake)
    s.train(detached=True)
    a = seen["argv"]
    assert a[a.index("--config") + 1] == "/c/anima_lora.toml" and a[a.index("--num-gpus") + 1] == "2"
    assert "--repo-root" not in a                       # the fork comes from ANIMA_DIFFUSION_PIPE
    assert seen["log"].endswith("runs/train.log")


def test_follow_prints_the_log_until_the_process_exits(tmp_path, capsys, monkeypatch):
    log = tmp_path / "train.log"
    log.write_text("step 1\nstep 2\n", encoding="utf-8")
    monkeypatch.setattr(sr.time, "sleep", lambda s: None)

    class _Proc:
        pid, polls = 1, 0

        def poll(self):
            self.polls += 1
            if self.polls == 1:                         # the trainer writes one more line, then exits
                with open(log, "a", encoding="utf-8") as f:
                    f.write("step 3\n")
                return None
            return 0

    sr.SanaRunner._follow(str(log))(_Proc())
    assert capsys.readouterr().out.splitlines()[:3] == ["step 1", "step 2", "step 3"]


def test_evaluate_needs_a_saved_lora(tmp_path):
    s = _ready_runner(tmp_path)
    s.state["output_dir"] = str(tmp_path / "data" / "runs" / "sana_lora")
    with pytest.raises(RuntimeError, match="train"):
        s.evaluate()


def test_cold_calls_raise_setup_error(tmp_path):
    s = sr.SanaRunner(data_root=str(tmp_path))
    for call in (s.prepare_dataset, s.build_configs, s.train):
        with pytest.raises(RuntimeError, match="setup"):
            call()


# ---- the bootstrap (anima_colab, stdlib-only at the repo root) -----------------------------
@pytest.fixture
def colab(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT))
    import anima_colab
    return anima_colab


def test_bootstrap_clones_the_fork_beside_upstream(tmp_path, monkeypatch, colab):
    monkeypatch.setenv("ANIMA_DIFFUSION_PIPE", "")      # ensure_repo sets these two; teardown restores them
    monkeypatch.setenv("ANIMA_REPO", "unchanged")
    repo = tmp_path / "anima-trainer"
    repo.mkdir()
    (repo / "pyproject.toml").write_text("", encoding="utf-8")
    cmds, pips = [], []

    def _sh(cmd, check=True):
        cmds.append(cmd)
        if "git clone" in cmd:                          # the clone lands train.py in the target dir
            target = Path(cmd.rsplit('"', 2)[-2])
            target.mkdir(parents=True, exist_ok=True)
            (target / "train.py").write_text("", encoding="utf-8")
        return 0

    monkeypatch.setattr(colab, "_sh", _sh)
    monkeypatch.setattr(colab, "_pip", lambda *a: pips.append(" ".join(a)))
    import importlib.metadata as md
    monkeypatch.setattr(md, "version", lambda name: "0.10.0" if name == "torchao" else "1")
    assert colab.dp_dir(str(repo)) == f"{repo}/external/diffusion-pipe"
    assert colab.dp_dir(str(repo), colab.DP_FORK_URL) == f"{repo}/external/diffusion-pipe-fork"

    assert colab.install(str(repo), dp_url=colab.DP_FORK_URL, similarity=False) is True
    assert any(colab.DP_FORK_URL in c and "diffusion-pipe-fork" in c for c in cmds)
    assert any("uninstall -y -q torchao" in c for c in cmds)      # peft refuses old torchao builds
    assert os.environ["ANIMA_DIFFUSION_PIPE"] == f"{repo}/external/diffusion-pipe-fork"
    assert any("diffusion-pipe-fork/requirements.txt" in p for p in pips)
    assert (repo / ".anima_colab_installed_fork").is_file() and not (repo / ".anima_colab_installed").exists()
    assert colab.install(str(repo), dp_url=colab.DP_FORK_URL, similarity=False) is False   # marker -> skip
    # the upstream install keeps its own marker and path
    assert colab.install(str(repo), similarity=False) is True
    assert (repo / ".anima_colab_installed").is_file()
    assert any(colab.DP_URL in c and c.rstrip().endswith('external/diffusion-pipe"') for c in cmds)
    if str(repo) in sys.path:
        sys.path.remove(str(repo))
