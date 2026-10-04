"""The Anima bed: the registry and READMEs, the LoRA hooks on a tiny CPU model, the trainer config, and the sequence +
the e001 flavor test end to end with the GPU and the hub replaced by fakes. Offline: no GPU/network/HF."""
from __future__ import annotations

import json
import tomllib
from pathlib import Path

import numpy as np
import pytest

from geolip_anima_trainer import anima_experiments as ax
from geolip_anima_trainer import anima_runner as ar
from geolip_anima_trainer import sana_experiments as sx
from geolip_anima_trainer import sana_runner as sr


# ---- the registry --------------------------------------------------------------------------------
def test_sequence_is_gate_ordered_and_varies_one_thing():
    assert ax.SEQUENCE_IDS[:2] == ["e002_lora_mood_up", "e003_lora_neutral_control"]
    refs = {a.seed_base: a for a in ax.SEQUENCE if a.flavor == "up" and a.lr == ax.LR}
    assert sorted(refs) == [1000, 2000] and refs[2000].id == "e005_lora_mood_up_reseed"
    for a in ax.SEQUENCE:
        ref = refs[a.seed_base]
        diffs = [k for k in ("flavor", "lr") if getattr(a, k) != getattr(ref, k)]
        assert len(diffs) == (0 if a is ref else 1), (a.id, diffs)
    roles = {}
    for a in ax.SEQUENCE:
        role = next(r for r, (fl, lr) in ax.DRAW_ROLES.items() if fl == a.flavor and lr == a.lr)
        roles.setdefault(a.seed_base, []).append(role)
    assert all(sorted(v) == sorted(ax.DRAW_ROLES) for v in roles.values())
    assert {a.lr for a in ax.SEQUENCE} == {1e-5, 2e-5, 4e-5} and ax.ANIMA.reference_lr == 2e-5
    assert len(set(ax.SEQUENCE_IDS)) == len(ax.SEQUENCE_IDS) and ax.FLAVOR_TEST_ID not in ax.SEQUENCE_IDS


def test_prompts_follow_the_model_card():
    for tps in ax.TEMPLATES.values():
        assert all(t.startswith(ax.PREFIX) and "photo" not in t for t in tps)
    assert ax.NEUTRAL_CAPTION == ax.TEMPLATES["neutral"][0]
    assert ax.PREFIX == "masterpiece, best quality, score_7, safe, "
    assert ax.NEGATIVE.startswith("worst quality, low quality, score_1")


def test_readmes_are_plain_and_name_the_right_references():
    by_id = {a.id: a for a in ax.SEQUENCE}
    top = sx.render_repo_readme([sx.arm_meta(ax.SEQUENCE[0], "running")], None, ax.ANIMA)
    assert "license: other" in top and "license_name: circlestone-labs-non-commercial-license" in top
    assert "base_model: circlestone-labs/Anima" in top and "# geolip-beatrix-anima" in top
    assert "2501.03575" in top and "2106.09685" in top and "ComfyUI" in top
    ctl1 = sx.render_arm_readme(by_id["e003_lora_neutral_control"], {"x": "y"}, None, ax.ANIMA)
    ctl2 = sx.render_arm_readme(by_id["e008_lora_neutral_control_draw2"], {"x": "y"}, None, ax.ANIMA)
    assert "read against e002" in ctl1 and "read against e005" in ctl2
    arm = sx.render_arm_readme(by_id["e002_lora_mood_up"], {"x": "y"}, None, ax.ANIMA)
    assert "Anima-Base v1.0" in arm and "model type `anima`" in arm and "an illustration of <scene>" in arm
    e001 = ax.render_flavor_test_readme({"status": "running"}, {"model": "Anima"})
    assert "fixed before the run" in e001 and "Running." in e001
    for txt in (top, ctl1, arm, e001):
        for word in ("S-1", "Phil", "docket", "canon/"):
            assert word not in txt


def test_cell_slopes_and_flavor_outcome():
    by_alpha = {-2.0: [-2.0, 0.0], -1.0: [-1.0, 0.0], 0.0: [0.0, 0.0], 1.0: [1.0, 0.0], 2.0: [2.0, 0.0]}
    assert ax.cell_slopes(by_alpha) == pytest.approx([1.0, 0.0])
    r = ax.flavor_outcome([1.0 + 0.01 * i for i in range(64)], 1, label="A DIAL")
    assert r["OUTCOME"] == "A DIAL"
    assert ax.flavor_outcome([0.1, -0.1] * 32, 1, label="A DIAL")["OUTCOME"] == "NO EFFECT"


# ---- LoRA by forward hooks -----------------------------------------------------------------------
torch = pytest.importorskip("torch")
nn = torch.nn


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([nn.ModuleDict({"q": nn.Linear(8, 6), "o": nn.Linear(6, 8)}) for _ in range(2)])
        self.llm_adapter = nn.ModuleDict({"k": nn.Linear(8, 8)})

    def forward(self, x):
        for b in self.blocks:
            x = b["o"](b["q"](x))
        return self.llm_adapter["k"](x)


def _lora_sd(seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"diffusion_model.blocks.0.q.lora_A.weight": torch.randn(2, 8, generator=g),
            "diffusion_model.blocks.0.q.lora_B.weight": torch.randn(6, 2, generator=g),
            "diffusion_model.blocks.1.o.lora_A.weight": torch.randn(2, 6, generator=g),
            "diffusion_model.blocks.1.o.lora_B.weight": torch.randn(8, 2, generator=g),
            "diffusion_model.llm_adapter.k.lora_A.weight": torch.randn(2, 8, generator=g),
            "diffusion_model.llm_adapter.k.lora_B.weight": torch.zeros(8, 2)}     # frozen at zero: skipped


def _merged(model, sd, scale):
    """The reference: the LoRA merged into a copy's weights (the copy's hooks dropped: deepcopy carries them)."""
    import copy
    m = copy.deepcopy(model)
    for mod in m.modules():
        mod._forward_hooks.clear()
    for path, ab in ar.LoraHooks.parse(sd).items():
        mod = m.get_submodule(path)
        mod.weight.data += scale * ab["B"] @ ab["A"]
    return m


def test_lora_hooks_equal_merged_weights_and_remove_exactly():
    torch.manual_seed(0)
    model, x = Tiny(), torch.randn(5, 8)
    stock = model(x)
    hooks, sd = ar.LoraHooks(model), _lora_sd()
    info = hooks.load("a", sd, scaling=1.0)
    assert info == {"modules": 2, "skipped_zero": 1}
    for scale in (1.0, 0.5):
        hooks.set("a", scale)
        assert torch.allclose(model(x), _merged(model, sd, scale)(x), atol=1e-5)
    hooks.remove()
    assert torch.equal(model(x), stock)                     # the stock forward, bit for bit
    hooks.load("b", sd, scaling=2.0)                         # alpha / r folds into the scale
    hooks.set("b", 0.5)
    assert torch.allclose(model(x), _merged(model, sd, 1.0)(x), atol=1e-5)
    hooks.unload()
    assert hooks.adapters == {} and torch.equal(model(x), stock)


def test_lora_hooks_refuse_what_does_not_fit():
    hooks = ar.LoraHooks(Tiny())
    with pytest.raises(AttributeError):
        hooks.load("x", {"diffusion_model.blocks.7.q.lora_A.weight": torch.ones(2, 8),
                         "diffusion_model.blocks.7.q.lora_B.weight": torch.ones(6, 2)})
    with pytest.raises(ValueError, match="do not fit"):
        hooks.load("x", {"blocks.0.q.lora_A.weight": torch.ones(2, 7), "blocks.0.q.lora_B.weight": torch.ones(6, 2)})
    with pytest.raises(TypeError, match="not a Linear"):
        hooks.load("x", {"blocks.0.lora_A.weight": torch.ones(2, 8), "blocks.0.lora_B.weight": torch.ones(6, 2)})
    with pytest.raises(ValueError, match="unexpected LoRA key"):
        ar.LoraHooks.parse({"blocks.0.q.alpha": torch.ones(1)})
    with pytest.raises(ValueError, match="both A and B"):
        ar.LoraHooks.parse({"blocks.0.q.lora_A.weight": torch.ones(2, 8)})


def test_read_lora_takes_the_scaling_from_its_config(tmp_path):
    from safetensors.torch import save_file
    save_file({k: v.contiguous() for k, v in _lora_sd().items()}, str(tmp_path / "adapter_model.safetensors"))
    sd, scaling = ar._read_lora(tmp_path)
    assert scaling == 1.0 and len(sd) == 6
    (tmp_path / "adapter_config.json").write_text(json.dumps({"r": 4, "lora_alpha": 8}), encoding="utf-8")
    assert ar._read_lora(tmp_path)[1] == 2.0


# ---- the runner with fakes ----------------------------------------------------------------------------
class FakeRepo:
    def __init__(self):
        self.files_: dict = {}
        self.commits: list = []

    def files(self):
        return list(self.files_)

    def read(self, path):
        b = self.files_.get(path)
        return b.decode() if b is not None else None

    def metas(self):
        return {p.split("/")[1]: json.loads(b) for p, b in self.files_.items()
                if p.startswith("experiments/") and p.endswith("/meta.json")}

    def commit(self, files, msg, *, retry=True):
        for p, v in files.items():
            self.files_[p] = v.read_bytes() if isinstance(v, Path) else v.encode() if isinstance(v, str) else v
        self.commits.append(msg)

    def put(self, path, data, msg, *, retry=True):
        self.commit({path: data}, msg)


class FakePipe:
    """A LoRA whose folder names a mood arm moves every image's fake mood by +-scale (as in the Sana tests)."""

    def __init__(self):
        self.loaded, self.current = {}, None

    def load_lora_weights(self, path, weight_name, adapter_name):
        assert Path(path, weight_name).is_file()
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
    s = ar.AnimaRunner(data_root=str(tmp_path / "data"), repo_root=str(tmp_path / "repo"), seeds_per_subject=2,
                       data_repo_id=None)          # offline; the training-set test points it at a fake data repo
    models = tmp_path / "models"
    models.mkdir()
    paths = {k: str(models / n) for k, n in (("transformer_path", "anima-base-v1.0.safetensors"),
                                             ("vae_path", "qwen_image_vae.safetensors"),
                                             ("llm_path", "qwen_3_06b_base.safetensors"))}
    s.state.update(data_root=str(tmp_path / "data"), resolution=768, hf_token="tok", **paths)
    repo, pipe, calls = FakeRepo(), FakePipe(), {"train": [], "tomls": []}
    for mod in (sr, ar):
        monkeypatch.setattr(mod, "_HubRepo", lambda repo_id, token: repo)
        monkeypatch.setattr(mod, "_drop_torchao", lambda: False)
    monkeypatch.setattr(s, "_point_at_fork", lambda: "fork")
    s._pipe = pipe

    def mood_of(prompt):
        return 1.0 if "cheerful" in prompt or "joyful" in prompt else -1.0 if "gloomy" in prompt or "somber" in prompt else 0.0

    def render(prompts, seeds, source_add=None, context_add=None):
        shift = sum(float(v.mean()) for v in (source_add, context_add) if v is not None)
        return [Image.new("RGB", (8, 8), (min(255, max(0, int(round(128 + 20 * (mood_of(p) + shift))))),) * 3)
                for p in prompts]

    def score(imgs):
        sign = pipe.loaded.get(pipe.current[0], 0) if pipe.current else 0
        scale = pipe.current[1] if pipe.current else 0.0
        lvl = np.array([(int(np.asarray(im)[0, 0, 0]) - 128) / 20 for im in imgs], dtype=float)
        noise = 0.02 * np.arange(len(imgs)) * (sign != 0)
        return np.tile(np.eye(4)[0], (len(imgs), 1)), lvl + 0.05 * (np.arange(len(imgs)) % 5) + sign * scale + noise

    monkeypatch.setattr(s, "_render", render)
    monkeypatch.setattr(s, "_score", score)

    def site_states(prompts):                 # tiny fake states: upbeat prompts +1, downbeat -1 on every own token
        m = torch.tensor([[1, 1, 1, 0]] * len(prompts))
        x = torch.stack([torch.full((4, 3), 2.0 + mood_of(p)) for p in prompts])
        return {"source": (x, m), "context": (x * 3, m)}

    monkeypatch.setattr(s, "_site_states", site_states)

    def build_plan(config_toml, num_gpus, master_port=None):
        return config_toml

    def fake_launch(plan, log_path=None, monitor=None, **kw):
        cfg = tomllib.loads(Path(plan).read_text(encoding="utf-8"))
        calls["train"].append(cfg["optimizer"]["lr"])
        calls["tomls"].append(cfg)
        run = Path(cfg["output_dir"]) / "20261003_180000"
        for n in (2, 4, 6, 8, 10):
            (run / f"epoch{n}").mkdir(parents=True, exist_ok=True)
            (run / f"epoch{n}" / "adapter_model.safetensors").write_bytes(b"lora")
        (run / "samples" / "step0").mkdir(parents=True, exist_ok=True)
        (run / "samples" / "step0" / "0.png").write_bytes(b"png")
        Path(log_path).write_text("steps: 480 loss: 0.9 iter time (s): 0.3 samples/sec: 13.0\n", encoding="utf-8")

        class Done:
            pid = 1

            def poll(self):
                return 0

        monitor(Done())
        return 0

    monkeypatch.setattr(sr._launch, "build_plan", build_plan)
    monkeypatch.setattr(sr._launch, "launch", fake_launch)
    monkeypatch.setattr(s, "_after_train", lambda spec, ctx, epochs: {"step0": {"status": "MATCHES"}})
    return s, repo, pipe, calls


def test_the_trainer_config_is_the_registered_recipe(runner, tmp_path):
    s, *_ = runner
    lora, ds = s._render_config(str(tmp_path / "imgs"), str(tmp_path / "out"), str(tmp_path / "cfg"), lr=4e-5,
                                held_out_previews=True)
    t = tomllib.loads(Path(lora).read_text(encoding="utf-8"))
    assert t["model"]["type"] == "anima" and t["model"]["llm_adapter_lr"] == 0.0 and t["model"]["dtype"] == "bfloat16"
    assert t["optimizer"] == {"type": "adam", "lr": 4e-5, "betas": [0.9, 0.99], "weight_decay": 0.0, "eps": 1e-8}
    assert t["bf16_master_weights"] is True and t["adapter"]["rank"] == 32
    assert t["samples"]["negative_prompt"] == ax.NEGATIVE and t["samples"]["steps"] == 30
    assert t["samples"]["cfg"] == 4.5 and t["samples"]["width"] == 768
    assert all(p.startswith(ax.PREFIX) for p in t["samples"]["prompts"]) and len(t["samples"]["prompts"]) == 4
    assert tomllib.loads(Path(ds).read_text(encoding="utf-8"))["resolutions"] == [768]
    rec = s._recipe(2e-5)
    assert "fp32 master weights" in rec["optimizer"] and "LLM adapter frozen" in rec["adapter"]


def test_sequence_runs_on_the_anima_bed(runner):
    s, repo, pipe, calls = runner
    arms = ["e002_lora_mood_up", "e003_lora_neutral_control", "e004_lora_mood_down"]
    metas = s.run_sequence(arms)
    assert calls["train"] == [2e-5, 2e-5, 2e-5]
    assert all(c["bf16_master_weights"] for c in calls["tomls"])
    assert metas["e002_lora_mood_up"]["final"]["OUTCOME"] == "FLAVOR LORA"
    assert metas["e004_lora_mood_down"]["final"]["mean"] < 0
    assert "against e002" in metas["e003_lora_neutral_control"]["summary"]
    items = (Path(s.state["data_root"]) / "datasets" / "up_1000" / "items.jsonl").read_text(encoding="utf-8")
    first = json.loads(items.splitlines()[0])
    assert first["caption"] == ax.NEUTRAL_CAPTION.format(s=sr.SUBJECTS[0]) and "cheerful" in first["prompt"]
    top = repo.files_["README.md"].decode()
    assert "# geolip-beatrix-anima" in top and "CONTROL QUIET" in top and "license: other" in top
    assert "Anima-Base v1.0" in repo.files_["experiments/e002_lora_mood_up/README.md"].decode()
    assert metas["e002_lora_mood_up"]["renderer_parity"] == {"step0": {"status": "MATCHES"}}
    assert not pipe.loaded


def test_parity_check_reads_the_trainers_previews(tmp_path, monkeypatch):
    import sys
    import types
    from PIL import Image
    prev = types.ModuleType("utils.previews")
    prev.image_filename = lambda i, p: f"{i:02d}_x.png"
    monkeypatch.setitem(sys.modules, "utils", types.ModuleType("utils"))
    monkeypatch.setitem(sys.modules, "utils.previews", prev)
    run = tmp_path / "runs" / "20261003_180000"
    for name, level in (("step0", 100), ("epoch10", 140)):
        (run / "samples" / name).mkdir(parents=True)
        for i in range(4):
            Image.new("RGB", (8, 8), (level,) * 3).save(run / "samples" / name / f"{i:02d}_x.png")
    (run / "epoch10").mkdir()

    class Pipe:
        lora = 0

        def load_lora_weights(self, path, weight_name, adapter_name):
            self.lora = 1

        def set_adapters(self, names, adapter_weights):
            pass

        def unload_lora_weights(self):
            self.lora = 0

        def generate(self, prompts, seeds, **kw):
            assert seeds == [42, 43, 44, 45] and kw["batch"] == 1
            return [Image.new("RGB", (8, 8), ((101, 150)[self.lora],) * 3) for _ in prompts]

    s = ar.AnimaRunner(data_root=str(tmp_path))
    s.state.update(resolution=768)
    s._pipe = Pipe()
    out = s._after_train(None, {}, [(10, run / "epoch10")])
    assert out["step0"] == {"mean_levels": 1.0, "max_levels": 1.0, "status": "MATCHES"}
    assert out["epoch10"]["status"] == "DIFFERS" and out["epoch10"]["mean_levels"] == 10.0
    assert s._after_train(None, {}, [(10, run / "epoch10")]) is None          # once per session


def test_flavor_test_end_to_end(runner):
    s, repo, pipe, calls = runner
    meta = s.run_flavor_test()
    r = meta["result"]
    assert meta["status"] == "done" and meta["kind"] == "flavor_test"
    assert r["up_words"]["OUTCOME"] == "UPBEAT WORDS MOVE IT" and r["down_words"]["OUTCOME"] == "DOWNBEAT WORDS MOVE IT"
    for site in ax.DIAL_SITES:
        assert r["dial"][site]["OUTCOME"] == "A DIAL" and r["dial"][site]["mean"] > 0
        assert set(r["dial"][site]["by_alpha"]) == {"-2.0", "-1.0", "0.0", "1.0", "2.0"}
        assert r["census"][site]["d_norm"] == pytest.approx(3 ** 0.5 * (1 if site == "source" else 3))
    assert r["census"]["source"]["mean"] == pytest.approx(2 * 3 ** 0.5)
    base = f"experiments/{ax.FLAVOR_TEST_ID}"
    for f in ("meta.json", "README.md", "result.json", "sheet_words.jpg", "sheet_dial_source.jpg",
              "sheet_dial_context.jpg"):
        assert f"{base}/{f}" in repo.files_
    assert "A DIAL" in repo.files_[f"{base}/README.md"].decode()
    assert ax.FLAVOR_TEST_ID in repo.files_["README.md"].decode()
    assert len(json.loads(repo.files_[f"{base}/result.json"])["cells"]) == 64
    again = s.run_flavor_test()                               # done already: skipped
    assert again["status"] == "done" and sum(c.startswith(ax.FLAVOR_TEST_ID) for c in repo.commits) == 2


def test_attribute_sets_are_the_registered_design():
    sets = ax.attr_sets()
    assert len(sets) == 22 and len(ax.attr_cells()) == 16
    assert sets["age@+2"] == (None, ("age", 2.0)) and "age@-2" not in sets      # adults only: pushed older only
    assert all(f"{a}-" in sets for a, (_, minus, _) in ax.ATTRIBUTES.items() if minus)
    assert ax.attr_prompt("very long hair", 0) == (ax.PREFIX + "1girl, solo, very long hair, upper body, office suit, "
                                                    "modern office.")
    assert ax.attr_prompt(None, 1) == ax.PREFIX + "1girl, solo, upper body, apron over a sweater, cozy cafe."
    assert not any("school" in o or "school" in st for o, st in ax.CHARACTERS)
    readme = ax.render_attribute_screen_readme({"status": "running"}, {"model": "Anima"})
    assert "fixed before the run" in readme and "2403.17064" in readme and "Running." in readme
    for word in ("S-1", "Phil", "docket", "canon/"):
        assert word not in readme


def test_attribute_screen_end_to_end(runner, monkeypatch):
    """Each tag owns one axis of a tiny conditioning space; a fake tagger reads the prompt and the push back. Built so
    that hair length / hair colour / age are clean sliders, the eye-colour slider drags hair colour along, the
    proportions push does nothing and the judge never sees style."""
    from PIL import Image
    s, repo, pipe, calls = runner
    names = list(ax.ATTRIBUTES)

    def vec(prompt):
        v = torch.zeros(8)
        for k, (plus, minus, _) in enumerate(ax.ATTRIBUTES.values()):
            v[k] += float(plus in prompt) - float(bool(minus) and minus in prompt)
        return v

    def site_states(prompts):
        m = torch.tensor([[1, 1, 1, 0]] * len(prompts))
        x = torch.stack([vec(p).expand(4, 8) + 2.0 for p in prompts])
        return {"source": (x, m), "context": (x, m)}

    def render(prompts, seeds, source_add=None, context_add=None):
        out = []
        for p, sd in zip(prompts, seeds):
            im = Image.new("RGB", (8, 8), (128, 128, 128))
            im.info.update(prompt=p, seed=sd, push=context_add)
            out.append(im)
        return out

    def tag_scores(imgs):
        res = {a: [] for a in names}
        for im in imgs:
            p, push = im.info["prompt"], im.info["push"]
            ci = next(i for i, (o, _) in enumerate(ax.CHARACTERS) if o in p)
            noise, v = 0.05 * ((ci + im.info["seed"]) % 3), vec(p)
            for k, a in enumerate(names):
                sc = 3 * float(v[k]) + noise
                if push is not None and a != "proportions":
                    sc += 2 * float(push[k])
                if push is not None and a == "hair_colour":
                    sc += 1.5 * float(push[names.index("eye_colour")])
                res[a].append(noise if a == "style" else sc)
        return res

    monkeypatch.setattr(s, "_site_states", site_states)
    monkeypatch.setattr(s, "_render", render)
    monkeypatch.setattr(s, "_tag_scores", tag_scores)
    loaded = []
    monkeypatch.setattr(s, "_tagger", lambda: loaded.append("tagger"))
    monkeypatch.setattr(s, "_judge", lambda: loaded.append("clip"))
    meta = s.run_attribute_screen()
    assert loaded == ["tagger", "clip"]                          # both judges load before the first image
    r = meta["result"]["attributes"]
    assert meta["status"] == "done" and meta["kind"] == "attribute_screen"
    assert {a: v["OUTCOME"] for a, v in r.items()} == {
        "hair_length": "SLIDER", "hair_colour": "SLIDER", "eye_colour": "SLIDER WITH CROSS-TALK",
        "proportions": "WORDS ONLY", "age": "SLIDER", "style": "NO HANDLE"}
    assert r["hair_length"]["reach"] == pytest.approx(2 / 6) and r["age"]["reach"] == pytest.approx(1 / 3)
    assert r["eye_colour"]["cross_talk"]["hair_colour"] == pytest.approx(1.5 / 6)
    assert r["hair_colour"]["words_down"]["OUTCOME"] == "THE WORD MOVES IT" and r["age"]["words_down"] is None
    ov = meta["result"]["overlap"]["context"]
    assert ov["hair_length"]["hair_colour"] == pytest.approx(0.0) and ov["age"]["age"] == pytest.approx(1.0)
    assert meta["result"]["direction_norms"]["context"]["age"] == pytest.approx(0.5)
    base = f"experiments/{ax.ATTR_TEST_ID}"
    for f in ["meta.json", "README.md", "result.json"] + [f"sheet_{a}.jpg" for a in names]:
        assert f"{base}/{f}" in repo.files_
    assert "SLIDER WITH CROSS-TALK" in repo.files_[f"{base}/README.md"].decode()
    assert ax.ATTR_TEST_ID in repo.files_["README.md"].decode()
    assert len(json.loads(repo.files_[f"{base}/result.json"])["cells"]) == 16
    again = s.run_attribute_screen()                           # done already: skipped
    assert again["status"] == "done" and sum(c.startswith(ax.ATTR_TEST_ID) for c in repo.commits) == 2


def test_training_sets_are_kept_and_reused_by_a_fresh_runtime(runner, fake_hub, tmp_path, capsys):
    s, repo, pipe, calls = runner
    s.cfg.data_repo_id = "AbstractPhil/geolip-beatrix-anima-data"
    s.run_sequence(["e002_lora_mood_up"])
    out = capsys.readouterr().out
    assert "rendering 48 up training images" in out and "saved to AbstractPhil/geolip-beatrix-anima-data" in out
    sets = [p.name for p in (fake_hub.d / "sets").iterdir()]
    assert len(sets) == 1 and sets[0].startswith("up_1000-") and len(fake_hub.commits) == 1
    assert repo.metas()["e002_lora_mood_up"]["recipe"]["training images"].endswith(f"/tree/main/sets/{sets[0]}")
    # a fresh runtime: an empty data folder, the arm not done yet -> the set comes back from the data repo, undrawn
    s.state["data_root"] = str(tmp_path / "data2")
    for k in [k for k in repo.files_ if k.startswith("experiments/e002_lora_mood_up/")]:
        del repo.files_[k]
    s.run_sequence(["e002_lora_mood_up"])
    out = capsys.readouterr().out
    assert "training set up_1000: downloaded from AbstractPhil/geolip-beatrix-anima-data" in out
    assert "rendering 48 up training images" not in out and len(fake_hub.commits) == 1
    assert len(list((tmp_path / "data2" / "datasets" / "up_1000" / "images").glob("*.png"))) == 48
    assert repo.metas()["e002_lora_mood_up"]["status"] == "done"


# ---- e013-e015: the connector experiments ------------------------------------------------------------------
PHRASES = ([{"class": "up", "split": "train", "text": t} for t in ("cheerful and upbeat", "joyful and uplifting", "happy",
                                                                   "sunny")]
           + [{"class": "up", "split": "heldout", "text": t} for t in ("elated", "blissful")]
           + [{"class": "down", "split": "train", "text": t} for t in ("gloomy and downbeat", "somber and melancholy",
                                                                     "sad", "bleak")]
           + [{"class": "down", "split": "heldout", "text": t} for t in ("mournful", "dismal")]
           + [{"class": "neutral", "split": "train", "text": t} for t in ("plain", "everyday")]
           + [{"class": "neutral", "split": "heldout", "text": "typical"}])


def test_connector_registry_and_rules():
    assert ax.CONNECTOR_IDS == ["e013_beatrix_mood_connector", "e014_beatrix_random_trunk_connector",
                                "e015_free_vector_connector", "e016_beatrix_mood_connector_whitened",
                                "e017_beatrix_random_trunk_connector_whitened", "e018_beatrix_mood_slider",
                                "e019_beatrix_random_trunk_slider"]
    assert [a.source for a in ax.CONNECTOR_ARMS] == ["trained", "random", "onehot", "trained", "random", "trained", "random"]
    assert [a.whiten_k for a in ax.CONNECTOR_ARMS] == [None, None, None, 16, 16, None, None]
    assert [a.axis for a in ax.CONNECTOR_ARMS] == [False] * 5 + [True] * 2
    ids = ax.CONNECTOR_IDS
    assert ax.CONNECTOR_PAIRS == ((ids[0], ids[1]), (ids[3], ids[4]), (ids[5], ids[6]))
    assert len({a.seed for a in ax.CONNECTOR_ARMS}) == 7 and not set(ax.CONNECTOR_IDS) & set(ax.SEQUENCE_IDS)
    assert ax.CONNECTOR_STEPS * ax.CONNECTOR_BATCH == 5 * 576 and ax.CONNECTOR_SAVE_EVERY * ax.CONNECTOR_BATCH == 576
    assert ax.connector_lrs("trained", 4096) == {"W": 1e-3 / 4096, "b": 1e-3}          # the fan-in rule (e013, e014)
    assert ax.connector_lrs("onehot", 3) == {"W": 1e-3, "b": 1e-3}
    assert ax.connector_lrs("trained", 16, contrast_l1=5.0) == {"W": 1e-3 * 2 / 5.0, "b": 1e-3}   # the free vector's pace
    sets = ax.connector_eval_sets(PHRASES, "trained")
    assert list(sets)[:4] == ["up/train/cheerful and upbeat", "up/train/joyful and uplifting", "up/heldout/elated",
                              "up/heldout/blissful"]
    assert len(sets) == 9 and sets["neutral/heldout/typical"] == ("neutral", "heldout", "typical")
    assert ax.connector_eval_sets(PHRASES, "onehot") == {c: (c, "train", None) for c in ax.CONNECTOR_CLASSES}
    with pytest.raises(ValueError, match="no training phrase"):
        ax.connector_eval_sets([p for p in PHRASES if p["text"] != "joyful and uplifting"], "trained")

    def diffs(up_tr, up_ho, dn_tr, dn_ho, neu):
        vals = {"up/train": up_tr, "up/heldout": up_ho, "down/train": dn_tr, "down/heldout": dn_ho,
                "neutral/heldout": neu}
        return {k: [vals[k.rsplit("/", 1)[0]] + 0.01 * (i % 4) for i in range(16)] for k in sets}

    good = ax.connector_reads(diffs(1.0, 0.6, -0.9, -0.5, 0.1), sets)
    assert good["TRAINED"] == "TRAINED WORDS MOVE IT" and good["HELD_OUT"] == "HELD-OUT WORDS CARRY IT"
    assert good["NEUTRAL"] == "NEUTRAL QUIET" and good["groups"]["heldout_up"]["n"] == 32
    assert good["heldout_effect"] == pytest.approx(0.55) and good["trained_effect"] == pytest.approx(0.95)
    unlearned = ax.connector_reads(diffs(1.0, 0.6, 0.0, -0.5, 0.5), sets)
    assert unlearned["TRAINED"] == "NOT LEARNED" and unlearned["HELD_OUT"] == "NOT LEARNED"
    assert unlearned["NEUTRAL"] == "NEUTRAL MOVES"
    assert ax.connector_reads(diffs(1.0, 0.6, -0.9, 0.2, 0.0), sets)["HELD_OUT"] == "HELD-OUT WORDS CARRY IT ONE WAY"
    flat = ax.connector_reads(diffs(1.0, 0.05, -0.9, 0.05, 0.0), sets)
    ids = ax.CONNECTOR_IDS

    def control(reads, pair=0):
        b_id, r_id = ax.CONNECTOR_PAIRS[pair]
        return ax.connector_cross_reads(reads)["controls"][f"{b_id} vs {r_id}"]["OUTCOME"]

    assert control({ids[0]: good, ids[1]: flat}) == "THE CONTROL FAILS AS IT SHOULD"
    assert control({ids[3]: good, ids[4]: flat}, pair=1) == "THE CONTROL FAILS AS IT SHOULD"
    leaky = ax.connector_reads(diffs(1.0, 0.4, -0.9, -0.3, 0.0), sets)
    assert control({ids[0]: good, ids[1]: leaky}) == "THE RANDOM TRUNK CARRIES IT TOO"
    assert control({ids[0]: unlearned, ids[1]: flat}).startswith("NOT READABLE")    # no control read without learning
    free = ax.connector_reads({"up": [1.9 + 0.01 * (i % 4) for i in range(16)],
                               "down": [-1.9 + 0.01 * (i % 4) for i in range(16)], "neutral": [0.1] * 16},
                              ax.connector_eval_sets(PHRASES, "onehot"), "onehot")
    assert free["TRAINED"] == "THE CLASS VECTORS MOVE IT" and "HELD_OUT" not in free and free["NEUTRAL"] == "NEUTRAL QUIET"
    assert ax.connector_cross_reads({ids[0]: good, ids[2]: free})["of_free_vector"][ids[0]] == pytest.approx(0.5, abs=0.01)
    readme = ax.render_connector_readme(ax.CONNECTOR_ARMS[0], {"status": "running"}, {"input": "x"}, PHRASES)
    assert "fixed before the run" in readme and "Running." in readme and "2208.01618" in readme
    assert "elated, blissful" in readme and "both branches" in readme and "principal components" not in readme
    whitened = ax.render_connector_readme(ax.CONNECTOR_ARMS[3], {"status": "running"}, {"input": "x"}, PHRASES)
    assert "top 16 principal components" in whitened and "(f - mu) @ V.T / scale" in whitened
    slider = ax.render_connector_readme(ax.CONNECTOR_ARMS[5], {"status": "running"}, {"input": "x"}, PHRASES)
    assert "a slider value" in slider and "(f - mu) @ V.T / scale" in slider and "principal components" not in slider
    import re
    for word in (r"S-1", r"\bPhil\b", r"docket", r"canon/"):           # the repo owner's handle in a repo id is fine
        assert not re.search(word, readme), word


def test_train_forward_builds_a_graph_for_the_push_only():
    """The fork's layer protocol on tiny modules: the embeddings and the adapter run without a graph, the push reaches
    every caption token after the adapter (padding untouched) with the exact gradient through the checkpointed blocks,
    and train_loss = the fork's prepare_inputs + its loss function with an empty mask."""
    from torch import nn

    class Initial(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.ones(()))

        def forward(self, inputs):
            x, t, pe, am, ids, tm = inputs
            return (x * self.w, t, pe, ids, am, tm, torch.zeros(1), torch.zeros(1), t)

    class Adapter(nn.Module):
        def forward(self, inputs):
            x, temb, pe, ids, am, tm, rope, adaln, t = inputs
            return (x, temb, pe * 2, rope, adaln, t)

    class Block(nn.Module):
        def forward(self, inputs):
            x, temb, ctx, rope, adaln, t = inputs
            return (x + ctx.sum(1, keepdim=True), temb, ctx, rope, adaln, t)

    class Final(nn.Module):
        def forward(self, inputs):
            return inputs[0]

    p = object.__new__(ar.AnimaPipe)
    p.layers, p.adapter_at = [Initial(), Adapter(), Block(), Block(), Final()], 1
    x, pe = torch.zeros(2, 1, 3), torch.ones(2, 4, 3)
    tm = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]])
    push = torch.zeros(2, 3, requires_grad=True)
    out = p.train_forward(x, torch.zeros(2, 1), (pe, tm, tm, tm), push)
    assert torch.allclose(out, torch.full((2, 1, 3), 16.0))          # 2 blocks x 4 tokens x 2 (the adapter)
    out.sum().backward()
    assert p.layers[0].w.grad is None                                  # nothing before the push is in the graph
    assert torch.allclose(push.grad, torch.tensor([[4.0] * 3, [6.0] * 3]))   # 2 blocks x the image's caption tokens

    class Model:
        def prepare_inputs(self, inputs):
            assert inputs["mask"] is None
            lat = inputs["latents"]
            return ((lat, torch.zeros(lat.shape[0], 1), inputs["prompt_embeds"], inputs["attn_mask"],
                     inputs["t5_input_ids"], inputs["t5_attn_mask"]), (torch.ones_like(lat), None))

    seen = {}

    def loss_fn(output, label):
        seen["mask"] = label[1]
        return ((output - label[0]) ** 2).mean()

    p.model, p._loss_fn = Model(), loss_fn
    loss = p.train_loss(x, (pe, tm, tm, tm), torch.zeros(2, 3, requires_grad=True))
    assert seen["mask"].numel() == 0 and float(loss.detach()) == pytest.approx(15.0 ** 2)


class FakeConnectorPipe:
    """Anima reduced to one number: an image's latent is its mood (+1 upbeat, -1 downbeat, 0 neutral) and the loss asks
    the push's first coordinate to equal it."""
    device, context_width = "cpu", 8

    def get_list_adapters(self):
        return {}

    def encode(self, prompts):
        n = len(prompts)
        return (torch.zeros(n, 4, 8), torch.ones(n, 4, dtype=torch.long), torch.zeros(n, 4, dtype=torch.long),
                torch.ones(n, 4, dtype=torch.long))

    def encode_images(self, imgs, res):
        assert res == 768
        return torch.stack([torch.full((2, 1, 2, 2), (int(np.asarray(im)[0, 0, 0]) - 128) / 20) for im in imgs])

    def train_loss(self, latents, conds, push):
        assert len(conds) == 4 and conds[0].shape[0] == latents.shape[0] == push.shape[0]
        return ((push[:, 0] - latents.flatten(1).mean(1)) ** 2).mean() + 0.01 * (push[:, 1:] ** 2).mean()


def _fake_features(dim: int = 16) -> dict:
    """Beatrix's features carry the class on a fixed direction (held-out phrases too); the untrained trunk's are noise
    for the training phrases and zero for the held-out ones (so its held-out push is its bias alone)."""
    g = torch.Generator().manual_seed(0)
    u = torch.sign(torch.randn(dim, generator=g))
    sign = {"up": 1.0, "down": -1.0, "neutral": 0.0}
    return {"trained": torch.stack([sign[p["class"]] * u + 0.1 * torch.randn(dim, generator=g) for p in PHRASES]),
            "random": torch.stack([torch.zeros(dim) if p["split"] == "heldout" else torch.randn(dim, generator=g)
                                   for p in PHRASES])}


def test_beatrix_connectors_end_to_end(runner, monkeypatch):
    from PIL import Image
    from safetensors.torch import load
    s, repo, _, _ = runner
    s._pipe = FakeConnectorPipe()
    for k, v in (("CONNECTOR_STEPS", 72), ("CONNECTOR_SAVE_EVERY", 36), ("CONNECTOR_TRACE_EVERY", 12),
                 ("CONNECTOR_LR", 0.05)):
        monkeypatch.setattr(ax, k, v)
    monkeypatch.setattr(s, "_connector_features", lambda: (_fake_features(), PHRASES))
    pushes, loaded = [], []

    def mood(p):
        return 1.0 if "cheerful" in p or "joyful" in p else -1.0 if "gloomy" in p or "somber" in p else 0.0

    def render(prompts, seeds, source_add=None, context_add=None, uncond_add=None):
        assert source_add is None and (context_add is None) == (uncond_add is None)
        if context_add is not None:
            assert torch.equal(context_add, uncond_add)              # a trained push goes on both guidance branches
            pushes.append(context_add)
        shift = float(context_add[0]) if context_add is not None else 0.0
        return [Image.new("RGB", (8, 8), (min(255, max(0, int(round(128 + 20 * (mood(p) + shift))))),) * 3)
                for p in prompts]

    monkeypatch.setattr(s, "_render", render)
    monkeypatch.setattr(s, "_judge", lambda: loaded.append("clip"))
    out = s.run_beatrix_connectors()
    assert loaded == ["clip"] and set(out) == set(ax.CONNECTOR_IDS)
    for c in ax.CONNECTOR_CLASSES:                                    # the three first-draw training sets were drawn
        assert len(list((Path(s.state["data_root"]) / "datasets" / f"{c}_1000" / "images").glob("*.png"))) == 48
    m = repo.metas()
    r13, r14, r15, r16, r17, r18, r19 = (m[k]["result"]["reads"] for k in ax.CONNECTOR_IDS)
    for r in (r13, r16, r18):
        assert r["TRAINED"] == "TRAINED WORDS MOVE IT" and r["HELD_OUT"] == "HELD-OUT WORDS CARRY IT"
        assert r["NEUTRAL"] == "NEUTRAL QUIET"
    for r in (r14, r17, r19):
        assert r["heldout_effect"] == pytest.approx(0.0, abs=1e-6) and r["HELD_OUT"] != "HELD-OUT WORDS CARRY IT"
    assert r15["TRAINED"] == "THE CLASS VECTORS MOVE IT" and r15["NEUTRAL"] == "NEUTRAL QUIET" and "HELD_OUT" not in r15
    cross = m[ax.CONNECTOR_IDS[1]]["result"]["cross"]
    assert [c["OUTCOME"] for c in cross["controls"].values()] == ["THE CONTROL FAILS AS IT SHOULD"] * 3
    assert m[ax.CONNECTOR_IDS[1]]["summary"].endswith("against e013: THE CONTROL FAILS AS IT SHOULD")
    assert m[ax.CONNECTOR_IDS[4]]["summary"].endswith("against e016: THE CONTROL FAILS AS IT SHOULD")
    assert m[ax.CONNECTOR_IDS[6]]["summary"].endswith("against e018: THE CONTROL FAILS AS IT SHOULD")
    assert set(cross["of_free_vector"]) == {ax.CONNECTOR_IDS[i] for i in (0, 1, 3, 4, 5, 6)}
    assert m[ax.CONNECTOR_IDS[0]]["result"]["learning_rates"] == {"W": 0.05 / 16, "b": 0.05}
    assert m[ax.CONNECTOR_IDS[2]]["result"]["learning_rates"] == {"W": 0.05, "b": 0.05}
    lr16 = m[ax.CONNECTOR_IDS[3]]["result"]["learning_rates"]
    assert lr16["b"] == 0.05 and 0.05 * 2 / 20 < lr16["W"] < 0.05 * 2 / 0.5      # 2 / the class contrast's L1 size
    assert "principal components" in m[ax.CONNECTOR_IDS[3]]["recipe"]["input"]
    w16 = load(repo.files_[f"experiments/{ax.CONNECTOR_IDS[3]}/connector/step0072.safetensors"])
    assert w16["V"].shape == (9, 16) and w16["W"].shape == (8, 9)       # k capped at the 10 training phrases' rank
    f = _fake_features()["trained"][1]                                   # 'joyful and uplifting' through the shipped projection
    push16 = ((f - w16["mu"]) @ w16["V"].T / w16["scale"]) @ w16["W"].T + w16["b"]
    assert any(torch.allclose(push16, p, atol=1e-5) for p in pushes)
    lr18 = m[ax.CONNECTOR_IDS[5]]["result"]["learning_rates"]
    assert lr18["b"] == 0.05 and lr18["W"] <= 0.05 * 2 / 2           # the slider's class contrast is at least 2 wide
    assert "a slider value" in m[ax.CONNECTOR_IDS[5]]["recipe"]["input"]
    w18 = load(repo.files_[f"experiments/{ax.CONNECTOR_IDS[5]}/connector/step0072.safetensors"])
    assert w18["V"].shape == (2, 16) and w18["W"].shape == (8, 2)
    push18 = ((f - w18["mu"]) @ w18["V"].T / w18["scale"]) @ w18["W"].T + w18["b"]
    assert any(torch.allclose(push18, p, atol=1e-5) for p in pushes)
    for k in ax.CONNECTOR_IDS:
        base = f"experiments/{k}"
        assert m[k]["kind"] == "beatrix_connector" and m[k]["status"] == "done"
        for f in ("meta.json", "README.md", "result.json", "sheet.jpg", "trace.json", "connector/step0036.safetensors",
                  "connector/step0072.safetensors"):
            assert f"{base}/{f}" in repo.files_, f
        assert k in repo.files_["README.md"].decode()
    assert len(json.loads(repo.files_[f"experiments/{ax.CONNECTOR_IDS[0]}/result.json"])["per_cell"]) == 9
    w = load(repo.files_[f"experiments/{ax.CONNECTOR_IDS[0]}/connector/step0072.safetensors"])
    push = _fake_features()["trained"][1] @ w["W"].T + w["b"]          # 'joyful and uplifting' from the shipped weights
    assert any(torch.allclose(push, p) for p in pushes)
    readme = repo.files_[f"experiments/{ax.CONNECTOR_IDS[0]}/README.md"].decode()
    assert "HELD-OUT WORDS CARRY IT" in readme and "| elated (held out) | up |" in readme
    n = len(repo.commits)
    again = s.run_beatrix_connectors()                                # done already: skipped
    assert all(v["status"] == "done" for v in again.values())
    assert not any(c.startswith(tuple(k[:4] for k in ax.CONNECTOR_IDS)) for c in repo.commits[n:])


def test_connector_whitening_is_fit_on_the_training_rows_only():
    g = torch.Generator().manual_seed(1)
    F = torch.randn(30, 12, generator=g) * torch.linspace(3, 0.1, 12)
    train = list(range(20))
    p = ar.connector_whitening(F, train, 4)
    Z = (F - p["mu"]) @ p["V"].T / p["scale"]
    assert p["V"].shape == (4, 12)
    assert torch.allclose(Z[train].mean(0), torch.zeros(4), atol=1e-5)
    assert torch.allclose(Z[train].std(0), torch.ones(4), atol=1e-4)         # unit variance on the training rows
    held = F.clone()
    held[20:] += 100.0                                                         # held-out rows never move the fit
    q = ar.connector_whitening(held, train, 4)
    assert torch.allclose(q["mu"], p["mu"]) and torch.allclose(q["scale"], p["scale"])
    assert ar.connector_whitening(F, [0, 1, 2], 16)["V"].shape[0] == 2         # k capped at the training rows' rank


def test_connector_axis_puts_the_training_centres_on_the_slider_marks():
    g = torch.Generator().manual_seed(2)
    F = torch.randn(30, 12, generator=g)
    pool = {"up": [0, 1, 2], "down": [3, 4, 5], "neutral": [6, 7, 8]}
    p = ar.connector_axis(F, pool)
    Z = (F - p["mu"]) @ p["V"].T / p["scale"]
    assert p["V"].shape == (2, 12) and torch.equal(p["scale"], torch.ones(2))
    assert float(Z[pool["up"]].mean(0)[0]) == pytest.approx(1.0, abs=1e-5)      # the cheerful centre at +1
    assert float(Z[pool["down"]].mean(0)[0]) == pytest.approx(-1.0, abs=1e-5)    # the gloomy centre at -1
    assert float(Z[pool["neutral"]].mean(0)[1]) == pytest.approx(1.0, abs=1e-5)  # the neutral centre at 1 on its axis
    mid = (F[pool["up"]].mean(0) + F[pool["down"]].mean(0)) / 2
    assert torch.allclose((mid - p["mu"]) @ p["V"].T, torch.zeros(2), atol=1e-5)
    held = F.clone()
    held[9:] += 100.0                                                          # rows outside the pool never move the fit
    q = ar.connector_axis(held, pool)
    assert torch.allclose(q["mu"], p["mu"]) and torch.allclose(q["V"], p["V"])


def test_route_reads_split_the_words_between_the_two_readings():
    n = 64
    sc = {"neutral": [0.1 * (i % 4) for i in range(n)]}

    def shifted(d):
        return [v + d + 0.01 * (i % 3) for i, v in enumerate(sc["neutral"])]

    sc.update(words_up=shifted(2.0), words_down=shifted(-1.0), qwen_up=shifted(0.0), qwen_down=shifted(0.0),
              t5_up=shifted(2.0), t5_down=shifted(-1.0))
    r = ax.route_reads(sc)
    assert r["routes"]["t5"]["OUTCOME"] == "CARRIES THE WORDS" and r["routes"]["qwen"]["OUTCOME"] == "CARRIES NOTHING"
    assert r["sets"]["words_down"]["OUTCOME"] == "THE WORDS MOVE IT" and r["sets"]["t5_up"]["OUTCOME"] == "THE T5 IDS MOVE IT"
    assert r["routes"]["t5"]["share_of_words"]["up"] == pytest.approx(1.0, abs=0.01)
    assert r["sum_of_routes"]["down"] == pytest.approx(1.0, abs=0.02)
    assert ax.route_summary(r).startswith("the words +2.01 / -0.99")


def test_route_split_end_to_end(runner, monkeypatch):
    from PIL import Image
    s, repo, _, _ = runner
    passed = []

    def mood(p):
        return 1.0 if "cheerful" in p else -1.0 if "gloomy" in p else 0.0

    def render(prompts, seeds, t5_prompts=None):
        """A model that reads the mood from the T5 ids, and the upbeat words a little from Qwen3's states too."""
        passed.append(t5_prompts is not None)
        t5 = t5_prompts or prompts
        assert len(t5) == len(prompts) == len(seeds)
        return [Image.new("RGB", (8, 8), (int(round(128 + 20 * (mood(t) + 0.25 * max(0.0, mood(q))))),) * 3)
                for q, t in zip(prompts, t5)]

    monkeypatch.setattr(s, "_render", render)
    meta = s.run_route_split()
    r = meta["result"]
    assert meta["status"] == "done" and meta["kind"] == "route_split"
    assert r["routes"]["t5"]["OUTCOME"] == "CARRIES THE WORDS" and r["routes"]["qwen"]["OUTCOME"] == "ONE WAY"
    assert r["sets"]["qwen_down"]["OUTCOME"] == "NO EFFECT" and r["sets"]["words_up"]["OUTCOME"] == "THE WORDS MOVE IT"
    assert r["routes"]["qwen"]["share_of_words"]["up"] == pytest.approx(0.25 / 1.25, abs=0.01)
    assert r["sum_of_routes"]["up"] == pytest.approx(1.0, abs=0.01)
    assert any(passed) and not all(passed)             # only the split sets hand the adapter another prompt's T5 ids
    base = f"experiments/{ax.ROUTE_TEST_ID}"
    for f in ("meta.json", "README.md", "result.json", "sheet_routes.jpg"):
        assert f"{base}/{f}" in repo.files_, f
    readme = repo.files_[f"{base}/README.md"].decode()
    assert "**CARRIES THE WORDS**" in readme and "position-exact" in readme and ax.ROUTE_TEST_ID in repo.files_["README.md"].decode()
    import re
    for word in (r"S-1", r"\bPhil\b", r"docket", r"canon/"):
        assert not re.search(word, readme), word
    assert len(json.loads(repo.files_[f"{base}/result.json"])["cells"]) == 64
    n = len(repo.commits)
    assert s.run_route_split()["status"] == "done"                    # done already: skipped
    assert not any(c.startswith(ax.ROUTE_TEST_ID) for c in repo.commits[n:])


def test_config_validation_refuses_master_weights_without_plain_adam():
    from geolip_anima_trainer import api
    cfg = api.TrainConfig(run=api.RunConfig(output_dir="o", bf16_master_weights=True), model=api.ModelConfig(),
                          adapter=api.AdapterConfig(), optimizer=api.OptimizerConfig(),
                          dataset=api.DatasetConfig(directories=[api.DirectoryConfig(path="d")]))
    with pytest.raises(api.ConfigError, match="bf16_master_weights"):
        api.validate(cfg)
