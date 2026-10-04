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
    s = ar.AnimaRunner(data_root=str(tmp_path / "data"), repo_root=str(tmp_path / "repo"), seeds_per_subject=2)
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


def test_config_validation_refuses_master_weights_without_plain_adam():
    from geolip_anima_trainer import api
    cfg = api.TrainConfig(run=api.RunConfig(output_dir="o", bf16_master_weights=True), model=api.ModelConfig(),
                          adapter=api.AdapterConfig(), optimizer=api.OptimizerConfig(),
                          dataset=api.DatasetConfig(directories=[api.DirectoryConfig(path="d")]))
    with pytest.raises(api.ConfigError, match="bf16_master_weights"):
        api.validate(cfg)
