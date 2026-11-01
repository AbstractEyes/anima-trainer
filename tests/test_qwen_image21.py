"""Qwen-Image-2.1 tests — the config block and its rules, the packaged templates, the fork check, the downloader, the
runner (the fork lookup, the dataset repeats, the configs, the screen's reads and pick, the uploader, the upload
layout, the release folder's licence files) and the bootstrap's pinned fork release.
Offline: no GPU/network/HF; the config engine and the launch planner run real."""
from __future__ import annotations

import json
import os
import sys
import tomllib
from pathlib import Path

import pytest

from geolip_anima_trainer import api
from geolip_anima_trainer import config as C
from geolip_anima_trainer import launch as L
from geolip_anima_trainer import qwen_image21_runner as qr

ROOT = Path(__file__).resolve().parents[1]
PATHS = {"diffusion_model": "/m/diffusion_models/qwen_image_2.1_bf16.safetensors",
         "vae": "/m/vae/qwen_image_2.1_vae_bf16.safetensors",
         "text_encoder": "/m/text_encoders/qwen3vl_8b_int8_convrot.safetensors"}
# MegaLiminal's ten class folders and their image counts (AbstractPhil/mega-liminal, data/<class>/)
MEGA_LIMINAL = {"mega_liminal": 870, "landscape": 766, "suburban": 111, "simulacrum": 36, "vanishing_point": 21,
                "empty_mall": 20, "hand_sourced": 18, "parking_garage": 13, "cityscape": 10, "movie_theatre": 8}


# ---- the config block ----------------------------------------------------------------------------
def _qwen(cfg=None) -> C.ModelConfig:
    return api.qwen_image21_model(PATHS, cfg=cfg)


def test_qwen_preset_and_round_trip(tmp_path: Path):
    cfg = C.single_concept_preset("data/c", output_dir="r", model=_qwen(4.0))
    assert (cfg.optimizer.type, cfg.optimizer.lr, cfg.optimizer.weight_decay) == ("adam", 1e-4, 0.0)
    lora_path, ds_path = C.render_train_toml(cfg, tmp_path)
    parsed = tomllib.loads(lora_path.read_text(encoding="utf-8"))
    assert parsed["model"] == {"type": "qwen_image21", "diffusion_model": PATHS["diffusion_model"], "vae": PATHS["vae"],
                               "text_encoders": [{"path": PATHS["text_encoder"], "type": "qwen_image"}],
                               "dtype": "bfloat16", "timestep_sample_method": "logit_normal", "cfg": 4.0}
    assert "Qwen-Image-2.1 reads natural-language captions" in ds_path.read_text(encoding="utf-8")
    assert "qwen_image21_lora.toml" in lora_path.read_text(encoding="utf-8")       # the 2.1 header
    loaded = C.load_train_config(lora_path, ds_path)
    assert loaded.model == cfg.model and loaded.optimizer == cfg.optimizer


def test_plain_loss_omits_cfg_and_other_models_omit_qwen_keys(tmp_path: Path):
    parsed = tomllib.loads(C.render_train_toml(C.single_concept_preset("data/c", output_dir="r", model=_qwen()),
                                               tmp_path / "q")[0].read_text(encoding="utf-8"))
    assert "cfg" not in parsed["model"] and "llm_adapter_lr" not in parsed["model"]
    anima = C.ModelConfig(transformer_path="t", vae_path="v", llm_path="l")
    parsed = tomllib.loads(C.render_train_toml(C.single_concept_preset("data/c", output_dir="r", model=anima),
                                               tmp_path / "a")[0].read_text(encoding="utf-8"))
    assert not {"diffusion_model", "vae", "text_encoders", "cfg"} & set(parsed["model"])


def test_max_steps_and_exclude_modules_render(tmp_path: Path):
    cfg = C.single_concept_preset("data/c", output_dir="r", model=_qwen())
    cfg.run.max_steps = 400
    cfg.adapter.exclude_modules = ["img_mlp.gate_up", "img_mlp.out"]
    parsed = tomllib.loads(C.render_train_toml(cfg, tmp_path)[0].read_text(encoding="utf-8"))
    assert parsed["max_steps"] == 400 and parsed["adapter"]["exclude_modules"] == ["img_mlp.gate_up", "img_mlp.out"]


def test_validate_qwen_rules(caplog):
    cfg = C.single_concept_preset("data/c", output_dir="r", model=_qwen())
    cfg.model.text_encoders = []
    with pytest.raises(C.ConfigError, match="one text_encoders entry"):
        C.validate(cfg)
    cfg.model.text_encoders = [{"path": "/m/te.safetensors", "type": "qwen3"}]
    with pytest.raises(C.ConfigError, match="one text_encoders entry"):
        C.validate(cfg)
    cfg.model.text_encoders = [{"path": "/m/te.safetensors", "type": "qwen_image"}]
    cfg.model.vae = ""
    with pytest.raises(C.ConfigError, match="diffusion_model and model.vae"):
        C.validate(cfg)
    cfg.model.vae = "/m/vae.safetensors"
    for bad in (1.0, 0.5, 11.0):
        cfg.model.cfg = bad
        with pytest.raises(C.ConfigError, match=r"\(1, 10\]"):
            C.validate(cfg)
    cfg.model.cfg = 4.0
    cfg.run.pipeline_stages = 2
    with pytest.raises(C.ConfigError, match="pipeline_stages = 1"):
        C.validate(cfg)
    cfg.run.pipeline_stages = 1
    cfg.model.llm_adapter_lr = 1e-6
    with pytest.raises(C.ConfigError, match="no LLM adapter"):
        C.validate(cfg)
    cfg.model.llm_adapter_lr = 0.0
    cfg.run.activation_checkpointing = True               # needed for the 7B transformer: no warning
    with caplog.at_level("WARNING", logger="anima.config"):
        C.validate(cfg)
    assert not any("activation_checkpointing" in r.getMessage() for r in caplog.records)
    anima = C.single_concept_preset("data/c", output_dir="r",
                                    model=C.ModelConfig(transformer_path="t", vae_path="v", llm_path="l"))
    anima.model.cfg = 4.0
    with pytest.raises(C.ConfigError, match="Qwen-Image-2.1 option"):
        C.validate(anima)


def test_packaged_qwen_templates_load_and_validate():
    from importlib import resources
    root = resources.files("geolip_anima_trainer").joinpath("templates")
    with resources.as_file(root.joinpath("qwen_image21_lora.toml")) as lp, \
            resources.as_file(root.joinpath("qwen_image21_dataset.toml")) as dp:
        cfg = C.load_train_config(lp, dp)
    assert cfg.model.type == "qwen_image21" and cfg.model.cfg is None
    assert cfg.model.text_encoders == [{"path": "/data/models/qwen_image_2.1/text_encoders/"
                                                "qwen3vl_8b_int8_convrot.safetensors", "type": "qwen_image"}]
    assert cfg.model.timestep_sample_method == "logit_normal"
    assert (cfg.optimizer.type, cfg.optimizer.lr, cfg.optimizer.weight_decay) == ("adam", 1e-4, 0.0)
    assert cfg.adapter.rank == 32 and cfg.run.activation_checkpointing is True and cfg.samples is None
    assert cfg.dataset.resolutions == [1024] and cfg.dataset.shuffle_caption is False
    C.validate(cfg)


def test_init_config_copies_the_qwen_templates(tmp_path: Path):
    from geolip_anima_trainer import cli
    assert cli.main(["init-config", "--out", str(tmp_path), "--model", "qwen_image21"]) == 0
    assert (tmp_path / "qwen_image21_lora.toml").is_file() and (tmp_path / "qwen_image21_dataset.toml").is_file()


# ---- the fork check and the downloader -----------------------------------------------------------
def test_qwen_needs_the_fork(tmp_path: Path):
    root = tmp_path / "repo"
    pipe = root / "external" / "diffusion-pipe"
    pipe.mkdir(parents=True)
    (pipe / "train.py").write_text("# fake\n", encoding="utf-8")
    lora = tmp_path / "lora.toml"
    lora.write_text("micro_batch_size_per_gpu = 1\n[model]\ntype = 'qwen_image21'\n", encoding="utf-8")
    with pytest.raises(L.DiffusionPipeNotFound, match="models/qwen_image21.py") as e:
        L.build_plan(config_toml=lora, repo_root=root, num_gpus=1)
    assert "--branch qwen-image21-v1" in str(e.value)               # the fork release that carries it
    (pipe / "models").mkdir()
    (pipe / "models" / "qwen_image21.py").write_text("# fake\n", encoding="utf-8")
    assert L.build_plan(config_toml=lora, repo_root=root, num_gpus=1).argv()[0] == "deepspeed"


def test_downloader_keeps_the_repo_layout(tmp_path: Path, monkeypatch):
    import huggingface_hub
    calls = []

    def _fake(repo, filename, local_dir=None, revision=None, **kw):
        calls.append((repo, filename, revision))
        return str(Path(local_dir) / filename)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _fake)
    paths = api.download_qwen_image21(tmp_path, revision="abc123")
    assert set(paths) == {"diffusion_model", "vae", "text_encoder"}
    assert {c[0] for c in calls} == {"Comfy-Org/Qwen-Image-2.1"} and {c[2] for c in calls} == {"abc123"}
    assert Path(paths["diffusion_model"]) == tmp_path / "diffusion_models" / "qwen_image_2.1_bf16.safetensors"
    assert api.qwen_image21_model(paths).text_encoders == [{"path": paths["text_encoder"], "type": "qwen_image"}]


# ---- the runner: the fork, the dataset, the configs ----------------------------------------------
def _fake_fork(root: Path, *, comfy: bool = True) -> Path:
    (root / "models").mkdir(parents=True)
    (root / "train.py").write_text("", encoding="utf-8")
    (root / "models" / "qwen_image21.py").write_text("", encoding="utf-8")
    if comfy:
        (root / "submodules" / "ComfyUI" / "comfy" / "ldm" / "qwen_image21").mkdir(parents=True)
    return root


def test_point_at_fork_needs_the_comfy_tree_and_prefers_the_release_clone(tmp_path, monkeypatch):
    monkeypatch.setenv("ANIMA_DIFFUSION_PIPE", "")      # unset for the lookup; teardown restores the original
    repo = tmp_path / "repo"
    _fake_fork(repo / "external" / "diffusion-pipe-fork", comfy=False)     # an older fork: no 2.1 in its ComfyUI
    q = qr.QwenImage21Runner(repo_root=str(repo))
    with pytest.raises(RuntimeError, match="QWEN21_FORK_REF"):
        q._point_at_fork()
    release = _fake_fork(repo / "external" / f"diffusion-pipe-fork-{qr.FORK_REF}")
    assert Path(q._point_at_fork()) == release and Path(os.environ["ANIMA_DIFFUSION_PIPE"]) == release


def test_class_folders_count_captioned_images(tmp_path: Path):
    for cls, names in {"b_pool": ["1.jpg", "1.txt", "2.webp", "2.txt", "3.png"],      # 3.png has no caption
                       "a_hall": ["x.jpeg", "x.txt", "notes.md"], "empty": []}.items():
        d = tmp_path / cls
        d.mkdir()
        for n in names:
            (d / n).write_text("", encoding="utf-8")
    assert [(c["name"], c["images"]) for c in qr.class_folders(tmp_path)] == [("a_hall", 1), ("b_pool", 2)]
    assert qr.class_folders(tmp_path / "missing") == []


def test_mega_liminal_repeats_reproduce_the_anima_lora_set():
    ds = C.DatasetConfig(directories=[C.DirectoryConfig(path=f"/d/{k}", name=k, image_count=v, caption_count=v)
                                      for k, v in MEGA_LIMINAL.items()])
    C.rebalance(ds)
    assert [d.num_repeats for d in ds.directories] == [1, 1, 3, 5, 6, 7, 7, 8, 8, 8]
    assert sum(d.image_count * d.num_repeats for d in ds.directories) == 2789


def _ready(tmp_path: Path, **kw) -> qr.QwenImage21Runner:
    q = qr.QwenImage21Runner(data_root=str(tmp_path / "data"), repo_root=str(tmp_path / "repo"), release_repo=None, **kw)
    rows = [{"name": k, "path": f"/d/{k}", "images": v, "repeats": r}
            for (k, v), r in zip(MEGA_LIMINAL.items(), [1, 1, 3, 5, 6, 7, 7, 8, 8, 8])]
    q.state.update(data_root=str(tmp_path / "data"), model_paths=dict(PATHS), directories=rows, samples_per_epoch=2789)
    return q


def test_build_configs_screen_arm(tmp_path: Path):
    q = _ready(tmp_path)
    prompts = qr.DRIFT_PROMPTS * 2 + qr.DOMAIN_PROMPTS
    t = tomllib.loads(Path(q.build_configs("screen_w4", cfg=4.0, epochs=1, max_steps=400, prompts=prompts,
                                           size=768)).read_text(encoding="utf-8"))
    assert t["model"]["cfg"] == 4.0 and t["max_steps"] == 400 and t["epochs"] == 1 and t["save_every_n_epochs"] == 1
    assert t["activation_checkpointing"] is True and t["adapter"]["rank"] == 32 and t["pipeline_stages"] == 1
    assert (t["optimizer"]["type"], t["optimizer"]["lr"], t["optimizer"]["weight_decay"]) == ("adam", 1e-4, 0.0)
    s = t["samples"]
    assert len(s["prompts"]) == 72 and (s["width"], s["height"], s["steps"], s["cfg"]) == (768, 768, 25, 1.0)
    assert s["shift"] == pytest.approx(1.9937) and s["seed"] == 42
    assert t["output_dir"].replace("\\", "/").endswith("data/runs/screen_w4")
    ds = tomllib.loads(Path(t["dataset"]).read_text(encoding="utf-8"))
    assert [d["num_repeats"] for d in ds["directory"]] == [1, 1, 3, 5, 6, 7, 7, 8, 8, 8]
    assert ds["resolutions"] == [1024] and ds["shuffle_caption"] is False
    assert q.state["runs"]["screen_w4"]["steps"] == 400


def test_build_configs_full_run(tmp_path: Path):
    q = _ready(tmp_path)
    t = tomllib.loads(Path(q.build_configs("r32_plain")).read_text(encoding="utf-8"))
    assert "cfg" not in t["model"] and "max_steps" not in t and t["epochs"] == 4 and t["save_every_n_epochs"] == 1
    assert t["samples"]["prompts"] == qr.SHOWCASE and t["samples"]["width"] == 1024
    assert q.state["runs"]["r32_plain"]["steps"] == 4 * 698          # ceil(2789 / 4) steps an epoch


def test_sample_shift_matches_the_models_time_shift():
    assert qr.SAMPLE_SHIFT == pytest.approx(1.9937, abs=1e-4) and qr.SAMPLE_STEPS == 25 and qr.SAMPLE_CFG == 1.0


def test_prompt_sets():
    assert len(qr.DRIFT_PROMPTS) == 32 and len(set(qr.DRIFT_PROMPTS)) == 32
    assert len(qr.SHOWCASE) == 15 and qr.DOMAIN_PROMPTS == qr.SHOWCASE[:8]
    assert all(p.startswith("liminal, ") for p in qr.SHOWCASE)
    assert not any("liminal" in p.lower() for p in qr.DRIFT_PROMPTS)


# ---- the screen's reads and pick ---------------------------------------------------------------
def _unit(*v):
    n = sum(x * x for x in v) ** 0.5
    return [x / n for x in v]


def _arm_rows(n, m, *, move, seed_gap, own_move):
    """Embeddings of one arm: n ordinary prompts (two seeds) and m own prompts, before and after training."""
    import math
    base, lora = [], []
    for i in range(2 * n + m):
        a = 0.3 * (i % n if i < 2 * n else i)                       # row n + i renders prompt i
        b = a + (seed_gap if n <= i < 2 * n else 0.0)               # at the second seed, seed_gap away
        base.append(_unit(math.cos(b), math.sin(b), 0.0))
        t = own_move if i >= 2 * n else move
        lora.append(_unit(math.cos(b) * math.cos(t), math.sin(b) * math.cos(t), math.sin(t)))
    return base, lora


def test_screen_reads_and_pick():
    n, m = 4, 2
    cb = [0.2] * (2 * n + m)
    calm_b, calm_a = _arm_rows(n, m, move=0.1, seed_gap=0.8, own_move=0.6)     # small drift, big own change
    loud_b, loud_a = _arm_rows(n, m, move=1.2, seed_gap=0.8, own_move=1.3)     # drift past a seed change
    calm = qr.screen_reads(calm_b, calm_a, cb, cb, n=n, m=m)
    loud = qr.screen_reads(loud_b, loud_a, cb, cb, n=n, m=m, first=calm_b)
    assert calm["drift"] < calm["spread"] and loud["drift"] > loud["spread"]
    assert calm["repeat"] is None and loud["repeat"] == pytest.approx(0.0, abs=1e-9)
    assert calm["contrast"] == pytest.approx(1.0) and calm["inside"] == 1.0 and calm["missing"] == 0
    pick = qr.screen_pick({"plain": {"cfg": None, **loud}, "w4": {"cfg": 4.0, **calm}})
    assert (pick["arm"], pick["cfg"], pick["verdict"]) == ("w4", 4.0, "PICK")
    assert qr.screen_pick({"plain": {"cfg": None, **loud}})["verdict"] == "NOT READY"
    faded = qr.screen_reads(calm_b, calm_a, cb, [0.1] * (2 * n + m), n=n, m=m)    # half the contrast
    assert faded["contrast"] == pytest.approx(0.5) and qr.screen_pick({"w4": {"cfg": 4.0, **faded}})["arm"] is None
    table = qr.screen_table({"plain": {"cfg": None, **loud}, "w4": {"cfg": 4.0, **calm}}, pick)
    assert "| w4 (picked) | guided, w = 4 |" in table and "| plain | plain |" in table


def test_screen_control_and_missing_previews():
    n, m = 4, 2
    cb = [0.2] * (2 * n + m)
    b, a = _arm_rows(n, m, move=0.1, seed_gap=0.8, own_move=0.6)
    other, _ = _arm_rows(n, m, move=0.0, seed_gap=0.8, own_move=0.0)
    shifted = [_unit(-x[1], x[0], x[2]) for x in other]          # 'the same renders' a quarter turn away
    broken = qr.screen_reads(b, a, cb, cb, n=n, m=m, first=shifted)
    assert broken["repeat"] > broken["spread"]
    pick = qr.screen_pick({"w4": {"cfg": 4.0, **broken}})
    assert pick["verdict"] == "MEASUREMENT FAILED" and pick["control_failed"] == ["w4"] and pick["arm"] is None
    a2 = list(a)
    a2[0] = None                                                  # a preview the trainer skipped
    r = qr.screen_reads(b, a2, cb, cb[:0] + [None] + cb[1:], n=n, m=m)
    assert r["missing"] == 1 and r["drift"] == pytest.approx(qr.screen_reads(b, a, cb, cb, n=n, m=m)["drift"], abs=0.05)


def test_names_and_parity_rows():
    assert qr.arm_name(None) == "plain" and qr.arm_name(4.0) == "w4" and qr.arm_name(3.5) == "w3.5"
    assert qr.save_label("epoch2") == "e02" and qr.save_label("step400") == "s400" and qr.save_label("x") == "x"
    rows = qr.parse_parity("loading\n[PASS] text encoder: cos 0.9999\n[FAIL] vae 512px: rel 1e-3\nnoise [INFO] x: y\n"
                           "[SKIP] cfg 4 step: no GPU\n")
    assert [(r["status"], r["check"]) for r in rows] == [("PASS", "text encoder"), ("FAIL", "vae 512px"),
                                                         ("SKIP", "cfg 4 step")]


# ---- the uploader -------------------------------------------------------------------------------
class _Err(Exception):
    def __init__(self, code):
        super().__init__(f"HTTP {code}")
        self.response = type("R", (), {"status_code": code})()


class _FakeRepo:
    def __init__(self, fails=()):
        self.fails, self.commits = list(fails), []

    def commit(self, files, msg, *, retry=True):
        assert retry is False                       # the shipper does its own waiting
        if self.fails:
            raise _Err(self.fails.pop(0))
        self.commits.append((dict(files), msg))


def test_shipper_retries_records_and_reships_changed_files(tmp_path: Path):
    f = tmp_path / "a.safetensors"
    f.write_bytes(b"12345")
    waits = []
    repo = _FakeRepo(fails=[500, 502])
    s = qr._Shipper(repo, tmp_path / "shipped.json", sleep=waits.append)
    s.submit({"x/a.safetensors": f}, "first")
    assert s.drain(10) == [] and waits == [15, 30] and len(repo.commits) == 1
    assert json.loads((tmp_path / "shipped.json").read_text(encoding="utf-8")) == {"x/a.safetensors": 5}
    s.submit({"x/a.safetensors": f}, "again")                    # same size: nothing to ship
    s.drain(10)
    assert len(repo.commits) == 1
    f.write_bytes(b"1234567")                                   # changed: ships again
    s.submit({"x/a.safetensors": f}, "changed")
    s.drain(10)
    assert len(repo.commits) == 2
    s2 = qr._Shipper(_FakeRepo(), tmp_path / "shipped.json", sleep=waits.append)    # a rerun reads the record
    assert s2.pending({"x/a.safetensors": f, "x/missing": tmp_path / "nope"}) == {}


def test_shipper_gives_up_on_a_refused_token(tmp_path: Path):
    f = tmp_path / "a.png"
    f.write_bytes(b"png")
    waits = []
    s = qr._Shipper(_FakeRepo(fails=[403]), tmp_path / "shipped.json", sleep=waits.append)
    s.submit({"x/a.png": f}, "refused")
    assert s.drain(10) == ["x/a.png"] and waits == [] and not (tmp_path / "shipped.json").exists()


# ---- the upload layout --------------------------------------------------------------------------
def _fake_run(out: Path, saves: list[str]) -> Path:
    run = out / "20261101_000000"
    t = 1_000_000
    for name in ["step0", *saves]:
        (run / "samples" / name).mkdir(parents=True)
        (run / "samples" / name / "00_a_pool.png").write_bytes(b"png")
        (run / "samples" / name / "prompts.txt").write_text("a pool\n", encoding="utf-8")
    for name in saves:
        d = run / name
        d.mkdir()
        (d / "adapter_model.safetensors").write_bytes(b"lora")
        (d / "adapter_config.json").write_text("{}", encoding="utf-8")
        t += 100
        os.utime(d, (t, t))
    (run / "events.out.tfevents.1.host").write_bytes(b"tb")
    return run


def test_ship_run_layout(tmp_path: Path, monkeypatch):
    q = _ready(tmp_path)
    out = tmp_path / "data" / "runs" / "r32_w4"
    _fake_run(out, ["epoch1", "epoch2"])
    for p in ("lora.toml", "dataset.toml", "r32_w4.log"):
        (tmp_path / p).write_text("x", encoding="utf-8")
    q.state["runs"] = {"r32_w4": {"output_dir": str(out), "cfg": 4.0, "lora_toml": str(tmp_path / "lora.toml"),
                                  "dataset_toml": str(tmp_path / "dataset.toml"), "log": str(tmp_path / "r32_w4.log")}}
    shipped = []
    monkeypatch.setattr(q, "_ship", lambda files, msg: shipped.append(dict(files)))
    q._ship_run("r32_w4", "qwen_image21/rank32", final=False)
    mid = set(shipped[-1])
    assert "qwen_image21/rank32/mega-liminal-qwen-image21-r32-w4-e01.safetensors" in mid
    assert "qwen_image21/rank32/mega-liminal-qwen-image21-r32-w4-e02.safetensors" not in mid     # still being written
    assert "qwen_image21/rank32/samples/w4/step0/00_a_pool.png" in mid
    assert "qwen_image21/rank32/samples/w4/epoch1/prompts.txt" in mid
    assert not any("/epoch2/" in p or "/training/" in p for p in mid)
    (out / "DONE.json").write_text("{}", encoding="utf-8")
    q._ship_run("r32_w4", "qwen_image21/rank32", final=True)
    end = set(shipped[-1])
    assert {"qwen_image21/rank32/mega-liminal-qwen-image21-r32-w4-e02.safetensors",
            "qwen_image21/rank32/samples/w4/epoch2/00_a_pool.png", "qwen_image21/rank32/adapter_config.json",
            "qwen_image21/rank32/training/w4/DONE.json", "qwen_image21/rank32/training/w4/lora.toml",
            "qwen_image21/rank32/training/w4/events.out.tfevents.1.host"} <= end


def test_ship_run_waits_for_the_first_save(tmp_path: Path, monkeypatch):
    q = _ready(tmp_path)
    out = tmp_path / "data" / "runs" / "screen_plain"
    _fake_run(out, [])
    q.state["runs"] = {"screen_plain": {"output_dir": str(out), "cfg": None, "lora_toml": "", "dataset_toml": "",
                                        "log": ""}}
    shipped = []
    monkeypatch.setattr(q, "_ship", lambda files, msg: shipped.append(dict(files)))
    q._ship_run("screen_plain", "qwen_image21/screen", final=False)
    assert shipped == []                                       # step0's previews may still be rendering


# ---- the release folder: licence, notice, readme -----------------------------------------------
def test_write_check_ships_the_licence_and_notice(tmp_path: Path, monkeypatch):
    import huggingface_hub
    q = _ready(tmp_path)
    q.cfg.release_repo = "me/liminal"
    q.state.update(hf_token="hf_x", hf_user="me")
    licence = tmp_path / "LICENSE"
    licence.write_text("Qwen RESEARCH LICENSE AGREEMENT\n", encoding="utf-8")
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda repo, name, **kw: str(licence))
    created = []
    monkeypatch.setattr(huggingface_hub, "HfApi",
                        lambda token=None: type("A", (), {"create_repo": lambda self, *a, **k: created.append((a, k))})())
    commits = []

    class _Repo:
        def __init__(self, repo_id, token):
            self.repo_id = repo_id

        def commit(self, files, msg, *, retry=True):
            commits.append((self.repo_id, files))

    monkeypatch.setattr(qr, "_HubRepo", _Repo)
    q._write_check()
    assert created and created[0][1] == {"repo_type": "model", "private": True, "exist_ok": True}
    repo_id, files = commits[0]
    assert repo_id == "me/liminal" and set(files) == {"qwen_image21/README.md", "qwen_image21/LICENSE",
                                                      "qwen_image21/NOTICE"}
    assert files["qwen_image21/LICENSE"].startswith("Qwen RESEARCH LICENSE AGREEMENT")
    assert qr.QWEN_NOTICE in files["qwen_image21/NOTICE"] and "trained by me" in files["qwen_image21/NOTICE"]
    readme = files["qwen_image21/README.md"]
    assert "**Built with Qwen.**" in readme and "`LICENSE`" in readme and "liminal, <class>, <description>" in readme
    assert "mega liminal, landscape" in readme and "1873 images in 10 classes" in readme


def test_write_check_needs_a_token(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(qr, "get_hf_token", lambda: None)
    q = _ready(tmp_path)
    q.cfg.release_repo = "me/liminal"
    with pytest.raises(RuntimeError, match="WRITE"):
        q._write_check()
    q.cfg.release_repo = None
    q._write_check()                                          # nothing ships: no check needed


def test_readme_without_a_caption_format(tmp_path: Path):
    q = _ready(tmp_path, caption_format=None, preview_prompts=["a hall", "a pool"])
    text = q.folder_readme()
    assert "The training captions read" not in text and "the 2 prompts the previews use" in text


# ---- the screen end to end (trainer and picture model faked) ------------------------------------
def test_screen_end_to_end_with_fakes(tmp_path: Path, monkeypatch):
    from PIL import Image
    q = _ready(tmp_path)
    q.cfg.screen_steps = 400
    monkeypatch.setattr(q, "_write_check", lambda: None)
    monkeypatch.setattr(q, "_hub_screen_reads", lambda prefix, setting: {})
    shipped = []
    monkeypatch.setattr(q, "_ship", lambda files, msg: shipped.append(dict(files)))
    n, m = len(qr.DRIFT_PROMPTS), len(qr.DOMAIN_PROMPTS)
    move = {None: 40, 4.0: 4}                                   # the plain arm moves its pictures 10x more

    def _run(tag, *, cfg, prefix, epochs, max_steps, prompts, size):
        assert epochs == 1 and max_steps == 400 and len(prompts) == 2 * n + m and size == q.cfg.screen_size
        out = Path(q.state["data_root"]) / "runs" / tag
        run = out / "20261101_000000"
        for save, shift in (("step0", 0), ("step400", move[cfg])):
            folder = run / "samples" / save
            folder.mkdir(parents=True)
            for i in range(2 * n + m):
                prompt = i % n if i < 2 * n else i                    # row n + i renders prompt i at another seed
                seed_tint = 30 if n <= i < 2 * n else 0
                shade = (prompt * 7 + seed_tint + (shift if i < 2 * n else 3 * shift)) % 256
                img = Image.new("RGB", (16, 16), (shade, 100, 255 - shade))
                img.paste((shade, 160, 255 - shade), (8, 0, 16, 16))   # the same contrast in every picture
                img.save(folder / f"{i:02d}_p.png")
        (run / "step400").mkdir()
        (run / "step400" / "adapter_model.safetensors").write_bytes(b"lora")
        q.state.setdefault("runs", {})[tag] = {"output_dir": str(out), "cfg": cfg}

    def _embed(paths):
        import math
        out = []
        for p in paths:
            r, _, _ = Image.open(p).convert("RGB").getpixel((0, 0))
            out.append([math.cos(r / 40), math.sin(r / 40)])
        return out

    monkeypatch.setattr(q, "_run", _run)
    monkeypatch.setattr(q, "_embed", _embed)
    summary = q.screen([None, 4.0])
    arms = summary["arms"]
    assert set(arms) == {"plain", "w4"} and arms["plain"]["repeat"] is None
    assert arms["w4"]["repeat"] == pytest.approx(0.0, abs=1e-9)       # same untrained pictures in both runs
    assert arms["w4"]["drift"] < arms["w4"]["spread"] < arms["plain"]["drift"]
    assert summary["pick"]["arm"] == "w4" and q.state["screen"] == {"arm": "w4", "cfg": 4.0}
    assert arms["w4"]["setting"] == q.screen_setting()
    paths = set().union(*shipped)
    assert {"qwen_image21/screen/reads/w4.json", "qwen_image21/screen/samples/w4/compare.jpg",
            "qwen_image21/screen/screen.md", "qwen_image21/screen/screen.json"} <= paths
    assert "w4 (picked)" in (Path(q.state["data_root"]) / "screen" / "screen.md").read_text(encoding="utf-8")


def test_screen_reuses_reads_an_earlier_runtime_shipped(tmp_path: Path, monkeypatch):
    q = _ready(tmp_path)
    monkeypatch.setattr(q, "_write_check", lambda: None)
    monkeypatch.setattr(q, "_ship", lambda files, msg: None)
    earlier = {"plain": {"cfg": None, "drift": 0.3, "spread": 0.2, "domain": 0.4, "contrast": 1.0, "inside": 0.1,
                         "repeat": None, "missing": 0, "selectivity": 1.3},
               "w4": {"cfg": 4.0, "drift": 0.05, "spread": 0.2, "domain": 0.2, "contrast": 0.98, "inside": 0.9,
                      "repeat": 0.001, "missing": 0, "selectivity": 4.0}}
    monkeypatch.setattr(q, "_hub_screen_reads", lambda prefix, setting: earlier)
    monkeypatch.setattr(q, "_run", lambda *a, **k: pytest.fail("a measured arm trained again"))
    assert q.screen([None, 4.0])["pick"]["arm"] == "w4" and q.state["screen"]["cfg"] == 4.0


def test_hub_screen_reads_keep_only_the_same_setting(tmp_path: Path, monkeypatch):
    import huggingface_hub
    q = _ready(tmp_path)
    q.cfg.release_repo = "me/liminal"
    q.state["hf_token"] = "hf_x"
    setting = q.screen_setting()
    files = {"qwen_image21/screen/reads/w4.json": {"cfg": 4.0, "setting": setting},
             "qwen_image21/screen/reads/w5.json": {"cfg": 5.0, "setting": {**setting, "steps": 800}}}
    for name, body in files.items():
        p = tmp_path / "hub" / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(body), encoding="utf-8")
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda token=None: type(
        "A", (), {"list_repo_files": lambda self, repo: [*files, "qwen_image21/README.md"]})())
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda repo, name, token=None: str(tmp_path / "hub" / name))
    assert set(q._hub_screen_reads("qwen_image21/screen", setting)) == {"w4"}


def test_train_needs_a_loss_choice(tmp_path: Path):
    q = _ready(tmp_path)
    with pytest.raises(RuntimeError, match="screen"):
        q.train()


def test_cold_calls_raise_setup_error(tmp_path: Path):
    q = qr.QwenImage21Runner(data_root=str(tmp_path), release_repo=None)
    for call in (q.parity, q.prepare_dataset, q.build_configs, lambda: q.train(cfg=None)):
        with pytest.raises(RuntimeError, match="setup"):
            call()


def test_config_env_and_overrides(monkeypatch):
    monkeypatch.setenv("ANIMA_QWEN21_RELEASE_REPO", "me/other")
    c = qr.QwenImage21Config.from_env(rank=64)
    assert c.release_repo == "me/other" and c.rank == 64 and c.lr == 1e-4 and c.epochs == 4
    with pytest.raises(TypeError):
        qr.QwenImage21Config.from_env(bogus=1)


# ---- the bootstrap's pinned fork release (anima_colab) and the notebook ---------------------------
@pytest.fixture
def colab(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT))
    import anima_colab
    return anima_colab


def test_bootstrap_clones_the_fork_release_into_its_own_folder(tmp_path, monkeypatch, colab):
    monkeypatch.setenv("ANIMA_DIFFUSION_PIPE", "")      # ensure_repo sets these two; teardown restores them
    monkeypatch.setenv("ANIMA_REPO", "unchanged")
    repo = tmp_path / "anima-trainer"
    repo.mkdir()
    (repo / "pyproject.toml").write_text("", encoding="utf-8")
    cmds, pips = [], []

    def _sh(cmd, check=True):
        cmds.append(cmd)
        if "git clone" in cmd:
            target = Path(cmd.rsplit('"', 2)[-2])
            target.mkdir(parents=True, exist_ok=True)
            (target / "train.py").write_text("", encoding="utf-8")
        return 0

    monkeypatch.setattr(colab, "_sh", _sh)
    monkeypatch.setattr(colab, "_pip", lambda *a: pips.append(" ".join(a)))
    import importlib.metadata as md
    monkeypatch.setattr(md, "version", lambda name: "1")
    assert colab.QWEN21_FORK_REF == qr.FORK_REF
    dp = f"{repo}/external/diffusion-pipe-fork-{qr.FORK_REF}"
    assert colab.dp_dir(str(repo), colab.DP_FORK_URL, colab.QWEN21_FORK_REF) == dp
    assert colab.dp_dir(str(repo), colab.DP_FORK_URL, "feat/x") == f"{repo}/external/diffusion-pipe-fork-feat_x"
    assert colab.install(str(repo), dp_url=colab.DP_FORK_URL, dp_ref=colab.QWEN21_FORK_REF, similarity=False) is True
    assert any(f'--branch "{qr.FORK_REF}" --recurse-submodules' in c and c.rstrip().endswith(f'{dp}"') for c in cmds)
    assert os.environ["ANIMA_DIFFUSION_PIPE"] == dp
    assert any(f"diffusion-pipe-fork-{qr.FORK_REF}/requirements.txt" in p for p in pips)
    assert (repo / f".anima_colab_installed_fork_{qr.FORK_REF}").is_file()
    assert not (repo / ".anima_colab_installed_fork").exists()          # the Sana marker stays its own
    cmds.clear()
    assert colab.install(str(repo), dp_url=colab.DP_FORK_URL, dp_ref=colab.QWEN21_FORK_REF, similarity=False) is False
    assert not any("git pull" in c for c in cmds if dp in c)           # a release tag is never pulled
    if str(repo) in sys.path:
        sys.path.remove(str(repo))


def test_notebook_is_a_thin_shell_over_the_runner():
    nb = json.loads((ROOT / "notebooks" / "qwen_image21_mega_liminal.ipynb").read_text(encoding="utf-8"))
    head = "".join(nb["cells"][0]["source"])
    assert ("https://colab.research.google.com/github/AbstractEyes/anima-trainer/blob/qwen-image21/notebooks/"
            "qwen_image21_mega_liminal.ipynb") in head
    code = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    assert any("dp_ref=anima_colab.QWEN21_FORK_REF" in c and 'BRANCH = "qwen-image21"' in c for c in code)
    calls = [step for c in code for step in ("setup", "parity", "prepare_dataset", "screen", "train", "sheets")
             if f"q.{step}()" in c]
    assert calls == ["setup", "parity", "prepare_dataset", "screen", "train", "sheets"]
    for c in code:
        compile(c.replace("!", "#"), "<cell>", "exec")
