"""The Sana experiment sequence — the verdict rules, the READMEs, and run_sequence end to end with the GPU and
the hub replaced by fakes (a fake trainer writes epoch folders, a fake pipeline's LoRA moves a fake mood score).
Offline: no GPU/network/HF."""
from __future__ import annotations

import json
import tomllib
from pathlib import Path

import numpy as np
import pytest

from geolip_anima_trainer import sana_experiments as sx
from geolip_anima_trainer import sana_runner as sr


# ---- the rules -----------------------------------------------------------------------------------
def test_arm_outcome_reads_in_the_arms_direction():
    down = [-1.0 - 0.01 * i for i in range(32)]
    assert sx.arm_outcome(down, -1)["OUTCOME"] == "FLAVOR LORA"
    assert sx.arm_outcome(down, 1)["OUTCOME"] == "NO EFFECT"          # read upward, it is 0% positive
    up = [1.0 + 0.01 * i for i in range(32)]
    assert sx.arm_outcome(up, -1)["OUTCOME"] == "NO EFFECT"
    ctl = sx.arm_outcome([0.1, -0.1] * 16, 0)
    assert ctl["OUTCOME"] == "CONTROL" and ctl["frac_pos"] == 0.5


def test_cross_arm_reads():
    assert sx.control_read(0.2, 1.0) == "CONTROL QUIET" and sx.control_read(-0.5, 1.0) == "CONTROL MOVES"
    net = sx.net_of_control([1.0 + 0.01 * i for i in range(32)], [0.0] * 32)
    assert net["OUTCOME"] == "THE MOOD COMES FROM THE IMAGES"
    assert sx.net_of_control([0.1, -0.1] * 16, [0.0] * 32)["OUTCOME"] == "NOT SHOWN"
    a = {"OUTCOME": "FLAVOR LORA", "mean": 1.0, "se": 0.1}
    assert sx.replicate_read(a, {"OUTCOME": "FLAVOR LORA", "mean": 1.1, "se": 0.1}) == "REPLICATES"
    assert sx.replicate_read(a, {"OUTCOME": "FLAVOR LORA", "mean": 2.0, "se": 0.1}) == "BOTH WORK, SIZES DIFFER"
    assert sx.replicate_read(a, {"OUTCOME": "NO EFFECT", "mean": 0.0, "se": 0.1}) == "DOES NOT REPLICATE"


def test_sequence_is_gate_ordered_and_varies_one_thing():
    assert sx.SEQUENCE_IDS[0] == "e004_lora_mood_up" and sx.SEQUENCE_IDS[1] == "e005_lora_neutral_control"
    refs = {a.seed_base: a for a in sx.SEQUENCE if a.flavor == "up" and a.lr == 1e-4}   # one reference per draw
    assert sorted(refs) == [1000, 2000] and refs[2000].id == "e007_lora_mood_up_reseed"
    for a in sx.SEQUENCE:
        ref = refs[a.seed_base]
        diffs = [k for k in ("flavor", "lr") if getattr(a, k) != getattr(ref, k)]
        assert len(diffs) == (0 if a is ref else 1), (a.id, diffs)
    roles = {}
    for a in sx.SEQUENCE:                                   # every draw carries every role exactly once
        role = next(r for r, (fl, lr) in sx.DRAW_ROLES.items() if fl == a.flavor and lr == a.lr)
        roles.setdefault(a.seed_base, []).append(role)
    assert all(sorted(v) == sorted(sx.DRAW_ROLES) for v in roles.values())
    assert len(set(sx.SEQUENCE_IDS)) == len(sx.SEQUENCE_IDS)


def _done(spec, mean, diffs, first=2):
    return sx.arm_meta(spec, "done", final={"mean": mean, "se": 0.1, "OUTCOME": "FLAVOR LORA" if mean > 0.5 else
                                            ("FLAVOR LORA" if spec.direction < 0 and mean < -0.5 else "CONTROL")},
                       final_diffs=diffs, first_epoch_beyond_3se=first)


def test_sequence_reads_per_draw_and_settled():
    by_id = {a.id: a for a in sx.SEQUENCE}
    up = [1.0 + 0.01 * i for i in range(32)]
    metas = {
        "e004_lora_mood_up": _done(by_id["e004_lora_mood_up"], 1.05, up),
        "e005_lora_neutral_control": _done(by_id["e005_lora_neutral_control"], 0.05, [0.05] * 32),
        "e006_lora_mood_down": _done(by_id["e006_lora_mood_down"], -0.9, [-0.9] * 32),
        "e007_lora_mood_up_reseed": _done(by_id["e007_lora_mood_up_reseed"], 1.1, up),
        "e010_lora_neutral_control_draw2": _done(by_id["e010_lora_neutral_control_draw2"], 0.6, [0.6] * 32),
        "e012_lora_mood_up_lr_5e-5_draw2": _done(by_id["e012_lora_mood_up_lr_5e-5_draw2"], 0.7, up, first=4),
    }
    r = sx.sequence_reads(metas)
    assert r["draw_1000"]["control"] == "CONTROL QUIET" and r["draw_2000"]["control"] == "CONTROL MOVES"
    assert r["settled"]["control"] == "UNSETTLED" and r["replicate"] == "REPLICATES"
    assert "mirror" in r["draw_1000"] and "mirror" not in r["draw_2000"]
    assert set(r["draw_2000"]["lr_dose"]) == {"5e-05", "0.0001"}
    md = sx.render_reads(r)
    assert "draw 1000-1007" in md and "draw 2000-2007" in md and "UNSETTLED" in md and "5e-05" in md


LEGACY_README = """---
license: apache-2.0
base_model: x
---
# geolip-beatrix-sana

Intro. The repo also keeps the experiments that led here. A hand-written note the runner must keep.

## Experiments
| folder | date | what | result |
|---|---|---|---|
| [`e004_lora_mood_up`](experiments/e004_lora_mood_up/) | 2026-10-03 | old | old |

## Reads across the LoRA sequence (rules fixed before the runs)
- old reads

## How the mood experiments are measured
Judge text.

## Licences
Hand-edited licence line.
"""


def test_readme_merge_keeps_hand_edits():
    metas = [sx.arm_meta(sx.SEQUENCE[0], "done", summary="NEW SUMMARY")]
    reads = {"draw_1000": {"control": "CONTROL QUIET"}}
    # a README written before the block markers: the old index is replaced, everything else kept
    once = sx.merge_repo_readme(LEGACY_README, metas, reads)
    assert "A hand-written note the runner must keep." in once and "Hand-edited licence line." in once
    assert "NEW SUMMARY" in once and "| old | old |" not in once and "old reads" not in once
    assert once.count("## How the mood experiments are measured") == 1 and once.count(sx.INDEX_START) == 1
    # with the markers: a hand edit outside the block survives a rewrite, the block follows the metas
    edited = once.replace("Judge text.", "Judge text, edited by hand.")
    twice = sx.merge_repo_readme(edited, [sx.arm_meta(sx.SEQUENCE[0], "done", summary="NEWER")], reads)
    assert "Judge text, edited by hand." in twice and "NEWER" in twice and "NEW SUMMARY" not in twice
    assert sx.merge_repo_readme(twice, [sx.arm_meta(sx.SEQUENCE[0], "done", summary="NEWER")], reads) == twice
    # no README yet, or only a metadata stub: the whole README
    full = sx.render_repo_readme(metas, reads)
    assert sx.merge_repo_readme(None, metas, reads) == full
    assert sx.merge_repo_readme("---\nlicense: mit\n---", metas, reads) == full
    # a fully hand-written README without an index: the block is appended, nothing removed
    own = "# My notes\n\n## Anything\ntext\n"
    out = sx.merge_repo_readme(own, metas, reads)
    assert out.startswith(own.rstrip("\n")) and sx.INDEX_END in out


def test_publish_index_keeps_hand_edits_and_always_writes(runner):
    s, repo, pipe, calls = runner
    repo.files_["README.md"] = LEGACY_README.encode()
    s._publish_index(repo, {})
    first = repo.files_["README.md"].decode()
    assert "A hand-written note the runner must keep." in first and sx.INDEX_START in first
    index = json.loads(repo.files_["index.json"])
    assert index["experiments"] == {} and "updated_utc" in index
    n = len(repo.commits)
    s._publish_index(repo, {})                                # nothing new: still one commit (the write check)
    assert len(repo.commits) == n + 1


def test_readmes_are_plain_and_complete():
    spec = sx.SEQUENCE[0]
    txt = sx.render_arm_readme(spec, {"optimizer": "Adam"})
    assert spec.id in txt and "fixed before the run" in txt and "Running." in txt
    top = sx.render_repo_readme([sx.arm_meta(spec, "running")], {"draw_1000": {"control": "CONTROL QUIET"}})
    assert "license: apache-2.0" in top and f"experiments/{spec.id}/" in top and "CONTROL QUIET" in top
    assert "2410.10629" in top and "2311.12092" in top
    for word in ("S-1", "Phil", "docket", "canon/"):
        assert word not in txt and word not in top


def test_epoch_dirs_reads_the_newest_run(tmp_path):
    old, new = tmp_path / "20260101_000000", tmp_path / "20260102_000000"
    for run, eps in ((old, (2, 4, 6)), (new, (2, 10, 4))):
        for n in eps:
            (run / f"epoch{n}").mkdir(parents=True)
            (run / f"epoch{n}" / "adapter_model.safetensors").write_bytes(b"x")
    import os
    os.utime(old, (1, 1))
    assert [n for n, _ in sr._epoch_dirs(tmp_path)] == [2, 4, 10]
    assert sr._epoch_dirs(tmp_path / "missing") == []


# ---- run_sequence end to end, with fakes ---------------------------------------------------------------
class FakeRepo:
    """The _HubRepo surface: commit() stores every file's bytes; put() is one file."""

    def __init__(self):
        self.files_: dict[str, bytes] = {}
        self.commits: list[str] = []
        self.fail_writes = False

    def files(self):
        return list(self.files_)

    def read(self, path):
        b = self.files_.get(path)
        return b.decode() if b is not None else None

    def metas(self):
        return {p.split("/")[1]: json.loads(b) for p, b in self.files_.items()
                if p.startswith("experiments/") and p.endswith("/meta.json")}

    def commit(self, files, msg, *, retry=True):
        if self.fail_writes:
            raise PermissionError("403 Forbidden")
        for p, v in files.items():
            assert isinstance(v, (Path, bytes, str)), type(v)
            self.files_[p] = v.read_bytes() if isinstance(v, Path) else v.encode() if isinstance(v, str) else v
        self.commits.append(msg)

    def put(self, path, data, msg, *, retry=True):
        self.commit({path: data}, msg)


def test_folder_files_keeps_allowed_names_and_skips_caches(tmp_path):
    (tmp_path / "images" / "cache").mkdir(parents=True)
    (tmp_path / "images" / "cache" / "x.bin").write_bytes(b"c")
    (tmp_path / "images" / "a.png").write_bytes(b"p")
    (tmp_path / "items.jsonl").write_text("{}", encoding="utf-8")
    (tmp_path / "global_step10").mkdir()
    (tmp_path / "global_step10" / "s.pt").write_bytes(b"s")
    assert sorted(sr._folder_files(tmp_path, "d")) == ["d/images/a.png", "d/items.jsonl"]
    assert list(sr._folder_files(tmp_path, "d", allow=["items.jsonl"])) == ["d/items.jsonl"]
    assert all(isinstance(v, Path) for v in sr._folder_files(tmp_path, "d").values())


class FakePipe:
    """A LoRA whose folder path names a mood arm moves every image's fake mood by +-scale."""

    def __init__(self):
        self.loaded, self.current = {}, None

    def load_lora_weights(self, path, weight_name, adapter_name):
        assert weight_name == "adapter_model.safetensors" and Path(path, weight_name).is_file()
        self.loaded[adapter_name] = -1 if "mood_down" in path else 0 if "neutral_control" in path else 1

    def set_adapters(self, names, adapter_weights):
        self.current = (names[0], adapter_weights[0])

    def unload_lora_weights(self):
        self.loaded, self.current = {}, None

    def get_list_adapters(self):
        return {"transformer": list(self.loaded)} if self.loaded else {}


@pytest.fixture
def runner(tmp_path, monkeypatch):
    from PIL import Image
    model = tmp_path / "Sana_600M_512px_diffusers"
    (model / "transformer").mkdir(parents=True)
    (model / "transformer" / "config.json").write_text(json.dumps({"sample_size": 16}), encoding="utf-8")
    s = sr.SanaRunner(data_root=str(tmp_path / "data"), repo_root=str(tmp_path / "repo"), seeds_per_subject=2)
    s.state.update(data_root=str(tmp_path / "data"), resolution=512, diffusers_path=str(model), hf_token="tok")
    repo, pipe, calls = FakeRepo(), FakePipe(), {"train": []}
    monkeypatch.setattr(sr, "_HubRepo", lambda repo_id, token: repo)
    monkeypatch.setattr(sr, "_drop_torchao", lambda: False)      # never touch this environment's packages
    monkeypatch.setattr(s, "_point_at_fork", lambda: "fork")
    s._pipe = pipe

    def render(prompts, seeds):
        return [Image.new("RGB", (8, 8), (100, 100, 100)) for _ in prompts]

    def score(imgs):
        n = len(imgs)
        sign = pipe.loaded.get(pipe.current[0], 0) if pipe.current else 0
        scale = pipe.current[1] if pipe.current else 0.0
        base = np.array([0.05 * (i % 5) for i in range(n)])
        moved = base + (sign * scale + 0.02 * np.arange(n) * (sign != 0))
        return np.tile(np.eye(4)[0], (n, 1)), moved

    monkeypatch.setattr(s, "_render", render)
    monkeypatch.setattr(s, "_score", score)
    calls.update(ports=[], running={}, max_running=0, overlaps=[], exit_codes={})

    def build_plan(config_toml, num_gpus, master_port=None):
        calls["ports"].append(master_port)
        return config_toml

    monkeypatch.setattr(sr._launch, "build_plan", build_plan)

    def fake_train(plan, log_path):
        cfg = tomllib.loads(Path(plan).read_text(encoding="utf-8"))
        calls["train"].append(cfg["optimizer"]["lr"])
        run = Path(cfg["output_dir"]) / "20261003_160000"
        for n in (2, 4, 6, 8, 10):
            (run / f"epoch{n}").mkdir(parents=True, exist_ok=True)
            (run / f"epoch{n}" / "adapter_model.safetensors").write_bytes(b"lora")
            (run / f"epoch{n}" / "adapter_config.json").write_text("{}", encoding="utf-8")
        (run / "samples" / "step0").mkdir(parents=True, exist_ok=True)
        (run / "samples" / "step0" / "0.png").write_bytes(b"png")
        Path(log_path).write_text("steps: 10 loss: 1.0 iter time (s): 0.1 samples/sec: 40.0\n"
                                  "steps: 480 loss: 0.9 iter time (s): 0.1 samples/sec: 41.3\n", encoding="utf-8")
        return Path(cfg["output_dir"]).parent.name            # the arm id

    def fake_launch(plan, log_path=None, monitor=None, **kw):
        fake_train(plan, log_path)

        class Done:
            pid = 1

            def poll(self):
                return 0

        monitor(Done())
        return 0

    def fake_spawn(plan, log_path):
        """A background trainer that finishes on its third poll; tracks which arms train at once."""
        arm = fake_train(plan, log_path)
        key = {a.id: a for a in sx.SEQUENCE}[arm].data_key
        if key in calls["running"].values():
            calls["overlaps"].append(arm)
        calls["running"][arm] = key
        calls["max_running"] = max(calls["max_running"], len(calls["running"]))

        class Proc:
            pid, polls = 7, 0

            def poll(self):
                self.polls += 1
                if self.polls < 3:
                    return None
                calls["running"].pop(arm, None)
                return calls["exit_codes"].get(arm, 0)

        return Proc()

    monkeypatch.setattr(sr._launch, "launch", fake_launch)
    monkeypatch.setattr(sr._launch, "spawn", fake_spawn)
    monkeypatch.setattr(sr.time, "sleep", lambda s: None)
    return s, repo, pipe, calls


def test_run_sequence_end_to_end(runner):
    s, repo, pipe, calls = runner
    arms = ["e004_lora_mood_up", "e005_lora_neutral_control", "e006_lora_mood_down"]
    metas = s.run_sequence(arms)
    assert calls["train"] == [1e-4, 1e-4, 1e-4]
    assert metas["e004_lora_mood_up"]["final"]["OUTCOME"] == "FLAVOR LORA"
    assert metas["e005_lora_neutral_control"]["final"]["OUTCOME"] == "CONTROL"
    assert metas["e006_lora_mood_down"]["final"]["OUTCOME"] == "FLAVOR LORA"
    assert metas["e006_lora_mood_down"]["final"]["mean"] < 0
    m4 = metas["e004_lora_mood_up"]
    assert [(r["epoch"], r["scale"]) for r in m4["epochs"]] == [(2, 1.0), (4, 1.0), (6, 1.0), (8, 1.0), (10, 0.5), (10, 1.0)]
    assert len(m4["final_diffs"]) == 32 and m4["first_epoch_beyond_3se"] == 2
    # the repo: every arm's folder, every epoch, the data list, the eval files, the index with the reads
    for a in arms:
        base = f"experiments/{a}"
        for n in (2, 4, 6, 8, 10):
            assert repo.files_[f"{base}/lora/epoch{n}/adapter_model.safetensors"] == b"lora"
        data = sorted(p for p in repo.files_ if p.startswith(f"{base}/data/"))
        assert data == [f"{base}/data/items.jsonl", f"{base}/data/sheet.jpg"]
        for f in ("final.json", "epochs.json", "sheet_final.jpg", "sheet_epochs.jpg"):
            assert f"{base}/eval/{f}" in repo.files_
        assert f"{base}/config/anima_lora.toml" in repo.files_ and f"{base}/logs/train.log" in repo.files_
        assert f"{base}/samples/step0/0.png" in repo.files_
        assert json.loads(repo.files_[f"{base}/meta.json"])["status"] == "done"
        assert "## Result" in repo.files_[f"{base}/README.md"].decode()
    arm_commits = [c for c in repo.commits if c.startswith("e004")]
    assert len(arm_commits) == 3                              # started, weights (before the eval), result
    top = repo.files_["README.md"].decode()
    assert "CONTROL QUIET" in top and "THE MOOD COMES FROM THE IMAGES" in top and "e006_lora_mood_down" in top
    # the training sets were rendered stock, beside (not inside) the image folders the trainer scans
    up = Path(s.state["data_root"]) / "datasets" / "up_1000"
    assert (up / "items.jsonl").is_file() and (up / "sheet.jpg").is_file()
    assert sorted(p.suffix for p in (up / "images").iterdir()) == [".png"] * 48 + [".txt"] * 48
    assert not pipe.loaded                                   # every LoRA unloaded after its evaluation


def test_index_keeps_folders_added_while_the_sequence_runs(runner):
    s, repo, pipe, calls = runner
    real_commit = repo.commit

    def commit(files, msg, *, retry=True):         # another writer adds a folder mid-sequence
        real_commit(files, msg)
        meta = files.get("experiments/e004_lora_mood_up/meta.json")
        if meta is not None and b'"done"' in meta:
            real_commit({"experiments/e001_flavor_test/meta.json":
                         json.dumps({"id": "e001_flavor_test", "title": "t", "status": "done", "summary": "DIAL"}).encode()}, "x")

    repo.commit = commit
    s.run_sequence(["e004_lora_mood_up"])
    assert "e001_flavor_test" in repo.files_["README.md"].decode()


def test_rerun_skips_done_arms_and_runs_the_rest(runner):
    s, repo, pipe, calls = runner
    s.run_sequence(["e004_lora_mood_up"])
    s._base = None
    metas = s.run_sequence(["e004_lora_mood_up", "e008_lora_mood_up_lr_5e-5"])
    assert calls["train"] == [1e-4, 5e-5]                    # e004 was not trained twice
    assert metas["e008_lora_mood_up_lr_5e-5"]["status"] == "done"
    assert "learning rate" in repo.files_["README.md"].decode()   # the lr dose table appears with two lrs


def test_a_failing_arm_is_recorded_and_stops_the_sequence(runner, monkeypatch):
    s, repo, pipe, calls = runner

    def boom(*a, **k):
        raise RuntimeError("deepspeed exploded")

    monkeypatch.setattr(sr._launch, "launch", boom)
    with pytest.raises(RuntimeError, match="exploded"):
        s.run_sequence(["e004_lora_mood_up", "e005_lora_neutral_control"])
    meta = json.loads(repo.files_["experiments/e004_lora_mood_up/meta.json"])
    assert meta["status"] == "failed" and "exploded" in meta["error"]
    assert "experiments/e005_lora_neutral_control/meta.json" not in repo.files_


def test_no_write_access_stops_before_any_gpu_work(runner):
    s, repo, pipe, calls = runner
    repo.fail_writes = True
    with pytest.raises(RuntimeError, match="WRITE"):
        s.run_sequence()
    assert calls["train"] == [] and not (Path(s.state["data_root"]) / "datasets").exists()


def test_drop_torchao_uninstalls_only_when_present(monkeypatch):
    import importlib.metadata as md
    ran = []
    monkeypatch.setattr(sr.subprocess, "run", lambda argv, check=False: ran.append(argv))

    def version_missing(name):
        raise md.PackageNotFoundError(name)

    monkeypatch.setattr(md, "version", version_missing)
    assert sr._drop_torchao() is False and ran == []
    monkeypatch.setattr(md, "version", lambda name: "0.10.0")
    assert sr._drop_torchao() is True
    assert ran and ran[0][-3:] == ["-y", "-q", "torchao"] and "uninstall" in ran[0]


def test_unknown_arm_is_refused(runner):
    s, *_ = runner
    with pytest.raises(ValueError, match="unknown arm"):
        s.run_sequence(["e999_nope"])


DRAW2 = ["e010_lora_neutral_control_draw2", "e011_lora_mood_down_draw2",
         "e012_lora_mood_up_lr_5e-5_draw2", "e013_lora_mood_up_lr_2e-4_draw2"]


def test_parallel_trains_side_by_side_never_on_a_shared_training_set(runner):
    s, repo, pipe, calls = runner
    metas = s.run_sequence(DRAW2, parallel=3)
    assert calls["max_running"] == 3                          # e010, e011, e012 together
    assert calls["overlaps"] == []                            # e013 waited for e012 (same images)
    ports = [p for p in calls["ports"] if p is not None]
    assert len(ports) == 4 and len(set(ports)) == 4           # one rendezvous port per trainer
    assert all(metas[a]["status"] == "done" for a in DRAW2)
    assert metas["e011_lora_mood_down_draw2"]["final"]["mean"] < 0
    for a in DRAW2:
        assert len([c for c in repo.commits if c.startswith(a)]) == 3
        assert repo.files_[f"experiments/{a}/lora/epoch10/adapter_model.safetensors"] == b"lora"
    assert metas["e010_lora_neutral_control_draw2"]["trained_beside"] == DRAW2[1:3]
    assert "same card at the same time" in repo.files_["experiments/e010_lora_neutral_control_draw2/README.md"].decode()
    assert not pipe.loaded


def test_parallel_matches_the_one_at_a_time_results(runner):
    s, repo, pipe, calls = runner
    side = s.run_sequence(DRAW2, parallel=4)
    repo.files_ = {k: v for k, v in repo.files_.items() if not k.startswith("experiments/")}
    s._base = None
    one = s.run_sequence(DRAW2)
    for a in DRAW2:
        assert side[a]["final"] == one[a]["final"] and side[a]["final_diffs"] == one[a]["final_diffs"]


def test_parallel_failure_lets_the_running_arms_finish_then_raises(runner):
    s, repo, pipe, calls = runner
    calls["exit_codes"]["e010_lora_neutral_control_draw2"] = 1
    with pytest.raises(RuntimeError, match="exited with code 1"):
        s.run_sequence(DRAW2, parallel=3)
    metas = repo.metas()
    assert metas["e010_lora_neutral_control_draw2"]["status"] == "failed"
    assert metas["e011_lora_mood_down_draw2"]["status"] == "done"     # was training beside it: finished + evaluated
    assert metas["e012_lora_mood_up_lr_5e-5_draw2"]["status"] == "done"
    assert "e013_lora_mood_up_lr_2e-4_draw2" not in metas            # nothing new starts after a failure
    assert "experiments/e010_lora_neutral_control_draw2/logs/train.log" in repo.files_


def test_progress_reads_the_trainer_log(tmp_path):
    log = tmp_path / "train.log"
    assert sr._progress(log) == "starting"
    log.write_text("caching latents\n", encoding="utf-8")
    assert sr._progress(log) == "starting"
    log.write_text("x\nsteps: 120 loss: 0.9 iter time (s): 0.097 samples/sec: 41.321\n", encoding="utf-8")
    assert sr._progress(log) == "step 120 (41 samples/s)"
    log.write_text("steps: 480 loss: 0.9 iter time (s): 0.1 samples/sec: 41.3\nTRAINING COMPLETE!\n", encoding="utf-8")
    assert sr._progress(log) == "saving"
