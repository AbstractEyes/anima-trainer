"""Launch-orchestration tests — pure command construction, no deepspeed/GPU."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from geolip_anima_trainer import launch as L


def _fake_pipe(tmp_path: Path) -> Path:
    """Create a fake external/diffusion-pipe/train.py under tmp_path (repo root)."""
    dp = tmp_path / "external" / "diffusion-pipe"
    dp.mkdir(parents=True)
    (dp / "train.py").write_text("# fake\n", encoding="utf-8")
    return tmp_path


def _lora(tmp_path: Path, *, stages=1, micro=4, grad=1) -> Path:
    p = tmp_path / "anima_lora.toml"
    p.write_text(f"pipeline_stages = {stages}\n"
                 f"micro_batch_size_per_gpu = {micro}\n"
                 f"gradient_accumulation_steps = {grad}\n", encoding="utf-8")
    return p


def test_build_plan_single_gpu(tmp_path: Path):
    root = _fake_pipe(tmp_path)
    plan = L.build_plan(config_toml=_lora(tmp_path), repo_root=root, num_gpus=1)
    argv = plan.argv()
    assert argv[0] == "deepspeed"
    assert "--num_gpus=1" in argv
    assert "--deepspeed" in argv and "--config" in argv
    assert plan.effective_batch == 4  # 4 * 1 * (1//1)


def test_master_port_is_a_launcher_option(tmp_path: Path):
    # trainings side by side on one machine each need their own rendezvous port
    root = _fake_pipe(tmp_path)
    assert not any(a.startswith("--master_port") for a in
                   L.build_plan(config_toml=_lora(tmp_path), repo_root=root).argv())
    argv = L.build_plan(config_toml=_lora(tmp_path), repo_root=root, master_port=29511).argv()
    assert argv.index("--master_port=29511") < argv.index(next(a for a in argv if a.endswith("train.py")))
    with pytest.raises(ValueError, match="master_port"):
        L.build_plan(config_toml=_lora(tmp_path), repo_root=root, master_port=80)


def test_spawn_refuses_windows(tmp_path: Path, monkeypatch):
    root = _fake_pipe(tmp_path)
    plan = L.build_plan(config_toml=_lora(tmp_path), repo_root=root)
    monkeypatch.setattr(L.platform, "system", lambda: "Windows")
    with pytest.raises(L.WindowsTrainingRefused):
        L.spawn(plan, tmp_path / "train.log")


def test_build_plan_multi_gpu_data_parallel(tmp_path: Path):
    root = _fake_pipe(tmp_path)
    plan = L.build_plan(config_toml=_lora(tmp_path, micro=4, grad=1),
                        repo_root=root, num_gpus=4)
    assert plan.world_size == 4 and plan.dp_size == 4
    assert plan.effective_batch == 16


def test_gpu_ids_use_include(tmp_path: Path):
    root = _fake_pipe(tmp_path)
    plan = L.build_plan(config_toml=_lora(tmp_path), repo_root=root, gpu_ids=[0, 1])
    argv = plan.argv()
    assert "--include" in argv and "localhost:0,1" in argv
    assert "--num_gpus" not in " ".join(argv)
    assert plan.env_prefix()["CUDA_VISIBLE_DEVICES"] == "0,1"


def test_expandable_segments_default(tmp_path: Path):
    root = _fake_pipe(tmp_path)
    plan = L.build_plan(config_toml=_lora(tmp_path), repo_root=root, num_gpus=1)
    assert plan.env_prefix()["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"


def test_pythonunbuffered_default(tmp_path: Path):
    # so diffusion-pipe's print() phase markers stream to the log instead of block-buffering
    root = _fake_pipe(tmp_path)
    plan = L.build_plan(config_toml=_lora(tmp_path), repo_root=root, num_gpus=1)
    assert plan.env_prefix()["PYTHONUNBUFFERED"] == "1"


def test_pythonpath_prepends_compat_shim(tmp_path: Path):
    # the diffusion-pipe subprocess must import our sitecustomize.py first -> shim dir leads PYTHONPATH
    from geolip_anima_trainer.dp_compat import shim_dir
    root = _fake_pipe(tmp_path)
    plan = L.build_plan(config_toml=_lora(tmp_path), repo_root=root, num_gpus=1)
    assert plan.env_prefix()["PYTHONPATH"].split(os.pathsep)[0] == shim_dir()


def test_cache_only_and_resume_flags(tmp_path: Path):
    root = _fake_pipe(tmp_path)
    plan = L.build_plan(config_toml=_lora(tmp_path), repo_root=root,
                        cache_only=True, resume_from_checkpoint="ckpt/epoch5")
    argv = plan.argv()
    assert "--cache_only" in argv
    assert "--resume_from_checkpoint" in argv and "ckpt/epoch5" in argv


def test_trust_cache_flag(tmp_path: Path):
    root = _fake_pipe(tmp_path)
    assert "--trust_cache" not in L.build_plan(config_toml=_lora(tmp_path), repo_root=root).argv()
    plan = L.build_plan(config_toml=_lora(tmp_path), repo_root=root,
                        cache_only=True, trust_cache=True)
    assert "--trust_cache" in plan.argv()


def test_topology_validation(tmp_path: Path):
    root = _fake_pipe(tmp_path)
    with pytest.raises(ValueError):  # 3 stages > 2 gpus
        L.build_plan(config_toml=_lora(tmp_path, stages=3), repo_root=root, num_gpus=2)
    with pytest.raises(ValueError):  # 4 gpus not divisible by 3 stages
        L.build_plan(config_toml=_lora(tmp_path, stages=3), repo_root=root, num_gpus=4)


def test_missing_diffusion_pipe_raises(tmp_path: Path):
    with pytest.raises(L.DiffusionPipeNotFound):
        L.build_plan(config_toml=_lora(tmp_path), repo_root=tmp_path, num_gpus=1)


def test_dry_run_returns_plan(tmp_path: Path):
    root = _fake_pipe(tmp_path)
    plan = L.build_plan(config_toml=_lora(tmp_path), repo_root=root, num_gpus=1)
    assert L.launch(plan, dry_run=True) is plan  # works on any OS, no exec


def test_api_cache_dry_run_has_trust_cache_and_pushes_nothing(tmp_path: Path, monkeypatch):
    from geolip_anima_trainer import api, cache_sync
    monkeypatch.setattr(cache_sync, "sync_up",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("pushed on dry-run!")))
    root = _fake_pipe(tmp_path)
    plan = api.cache(_lora(tmp_path), repo_root=root, dry_run=True, trust_cache=True,
                     backup_repo="u/r")            # backup gated off by dry_run -> no push
    assert "--cache_only" in plan.argv() and "--trust_cache" in plan.argv()


def test_launch_creates_missing_log_parent(tmp_path: Path, monkeypatch):
    """log_path in a not-yet-existing dir must not raise FileNotFoundError — launch()
    mkdirs the parent (caching runs before the training cell creates OUTPUT_DIR)."""
    root = _fake_pipe(tmp_path)
    plan = L.build_plan(config_toml=_lora(tmp_path), repo_root=root, num_gpus=1)
    monkeypatch.setattr(L.platform, "system", lambda: "Linux")  # bypass the Windows guard

    class _FakeProc:
        def wait(self):
            return 0
    monkeypatch.setattr(L.subprocess, "Popen", lambda *a, **k: _FakeProc())

    log_path = tmp_path / "runs" / "anima_prelim" / "cache_vlm.log"
    assert not log_path.parent.exists()
    rc = L.launch(plan, log_path=log_path)        # monitor=None + log_path set -> the fixed branch
    assert rc == 0
    assert log_path.parent.is_dir() and log_path.exists()


def _lora_with_samples(tmp_path: Path, *, stages=1) -> Path:
    p = _lora(tmp_path, stages=stages)
    p.write_text(p.read_text(encoding="utf-8") + "\n[samples]\nprompts = ['a pool']\n", encoding="utf-8")
    return p


def test_samples_warn_when_pipe_has_no_previews(tmp_path: Path, caplog):
    root = _fake_pipe(tmp_path)
    with caplog.at_level("WARNING", logger="anima.launch"):
        L.build_plan(config_toml=_lora_with_samples(tmp_path), repo_root=root, num_gpus=2)
    assert any("utils/previews.py" in r.getMessage() for r in caplog.records)


def test_samples_quiet_when_pipe_has_previews(tmp_path: Path, caplog):
    root = _fake_pipe(tmp_path)
    utils = root / "external" / "diffusion-pipe" / "utils"
    utils.mkdir()
    (utils / "previews.py").write_text("# fake\n", encoding="utf-8")
    with caplog.at_level("WARNING", logger="anima.launch"):
        L.build_plan(config_toml=_lora_with_samples(tmp_path), repo_root=root, num_gpus=2)
    assert not any("previews" in r.getMessage() for r in caplog.records)


def test_samples_refuse_pipeline_parallel(tmp_path: Path):
    root = _fake_pipe(tmp_path)
    with pytest.raises(ValueError, match="samples"):
        L.build_plan(config_toml=_lora_with_samples(tmp_path, stages=2), repo_root=root, num_gpus=2)


def _sana_lora(tmp_path: Path) -> Path:
    p = _lora(tmp_path)
    p.write_text(p.read_text(encoding="utf-8") + "\n[model]\ntype = 'sana'\ndiffusers_path = 'x'\n",
                 encoding="utf-8")
    return p


def test_sana_needs_the_fork(tmp_path: Path):
    # upstream diffusion-pipe has no models/sana.py: refuse before deepspeed starts
    root = _fake_pipe(tmp_path)
    with pytest.raises(L.DiffusionPipeNotFound, match="models/sana.py"):
        L.build_plan(config_toml=_sana_lora(tmp_path), repo_root=root, num_gpus=1)
    models = root / "external" / "diffusion-pipe" / "models"
    models.mkdir()
    (models / "sana.py").write_text("# fake\n", encoding="utf-8")
    plan = L.build_plan(config_toml=_sana_lora(tmp_path), repo_root=root, num_gpus=1)
    assert plan.argv()[0] == "deepspeed"
