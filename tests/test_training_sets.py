"""training_sets: the key, the upload of complete local sets (skipping incomplete ones, the trainer's cache and sets the
Hub has), and the round trip back into a fresh runtime, against a fake Hub in a temp folder. Offline."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from geolip_anima_trainer import training_sets as ts

RENDER = {"bed": "anima", "model": "anima-base-v1.0.safetensors", "resolution": 768, "steps": 30, "guidance": 4.5,
          "shift": 3.0, "negative": "worst quality", "batch": 8}


def _items(n=3, flavor="up"):
    return [{"name": f"00_{k:02d}", "seed": 1000 + k, "prompt": f"{flavor} prompt {k}", "caption": "neutral caption"}
            for k in range(n)]


def _draw(root: Path, items, *, upto=None):
    img = root / "images"
    img.mkdir(parents=True, exist_ok=True)
    (root / "items.jsonl").write_text("".join(json.dumps(it) + "\n" for it in items), encoding="utf-8")
    for it in items[:upto]:
        (img / f"{it['name']}.png").write_bytes(b"png-" + it["name"].encode())
        (img / f"{it['name']}.txt").write_text(it["caption"], encoding="utf-8")
    if upto is None:
        (root / "sheet.jpg").write_bytes(b"jpg")
    return root


@pytest.fixture
def hub(fake_hub):                    # tests/conftest.py: a dataset repo in a temp folder
    return fake_hub


def test_the_key_is_the_inputs():
    k = ts.set_key(RENDER, _items())
    assert k == ts.set_key(dict(reversed(list(RENDER.items()))), _items()) and len(k) == 10
    assert k != ts.set_key({**RENDER, "steps": 20}, _items())
    assert k != ts.set_key(RENDER, _items(flavor="down"))
    assert ts.hub_folder("up_1000", RENDER, _items()) == f"sets/up_1000-{k}"


def test_render_spec_reads_any_runner_version():
    r = SimpleNamespace(BED=SimpleNamespace(key="anima"), GEN_STEPS=30, GEN_CFG=4.5, GEN_SHIFT=3, NEGATIVE="worst quality",
                        cfg=SimpleNamespace(gen_batch=8),
                        state={"transformer_path": "/content/m/anima-base-v1.0.safetensors", "resolution": "768"})
    spec = ts.render_spec_of(r)
    assert spec == RENDER and isinstance(spec["shift"], float) and isinstance(spec["resolution"], int)


def test_upload_local_sets_then_reuse_in_a_fresh_runtime(tmp_path, hub):
    data = tmp_path / "data"
    up, neutral = _items(), _items(flavor="neutral")
    _draw(data / "datasets" / "up_1000", up)
    _draw(data / "datasets" / "neutral_1000", neutral, upto=1)                # still being drawn
    (data / "datasets" / "up_1000" / "images" / "cache" / "anima").mkdir(parents=True)
    (data / "datasets" / "up_1000" / "images" / "cache" / "anima" / "shard.png").write_bytes(b"latents")
    runner = SimpleNamespace(BED=SimpleNamespace(key="anima"), GEN_STEPS=30, GEN_CFG=4.5, GEN_SHIFT=3.0,
                             NEGATIVE="worst quality", cfg=SimpleNamespace(gen_batch=8),
                             state={"data_root": str(data), "resolution": 768, "hf_token": "tok",
                                    "transformer_path": "anima-base-v1.0.safetensors"})
    out = ts.upload_local_sets(runner)
    folder = ts.hub_folder("up_1000", RENDER, up)
    assert out == {"neutral_1000": "incomplete", "up_1000": folder} and len(hub.commits) == 1
    on_hub = sorted(p.relative_to(hub.d / folder).as_posix() for p in (hub.d / folder).rglob("*") if p.is_file())
    assert on_hub == sorted(["items.jsonl", "sheet.jpg", "render.json"]
                            + [f"images/{it['name']}.{x}" for it in up for x in ("png", "txt")])     # no cache
    meta = json.loads((hub.d / folder / "render.json").read_text(encoding="utf-8"))
    assert meta["render"] == RENDER and meta["images"] == 3
    assert ts.upload_local_sets(runner)["up_1000"] == "already there" and len(hub.commits) == 1

    fresh = tmp_path / "fresh" / "datasets" / "up_1000"                    # a new runtime: nothing on disk
    assert ts.download_set("tok", ts.DATA_REPO, fresh, RENDER, up) is True
    for it in up:
        assert (fresh / "images" / f"{it['name']}.png").read_bytes() == b"png-" + it["name"].encode()
    assert ts.read_items(fresh) == up and (fresh / "sheet.jpg").is_file() and not ts.missing_images(fresh, up)
    assert not (fresh.parent / ".pull_up_1000").exists()
    assert ts.download_set("tok", ts.DATA_REPO, tmp_path / "x" / "up_1000", {**RENDER, "steps": 20}, up) is False
    assert ts.download_set("tok", ts.DATA_REPO, tmp_path / "x" / "down_1000", RENDER, _items(flavor="down")) is False


def test_a_hub_set_that_does_not_check_out_is_not_used(tmp_path, hub):
    up = _items()
    root = _draw(tmp_path / "a" / "up_1000", up)
    folder = ts.upload_set("tok", ts.DATA_REPO, root, RENDER, up)
    (hub.d / folder / "images" / "00_01.txt").write_text("another caption", encoding="utf-8")
    assert ts.download_set("tok", ts.DATA_REPO, tmp_path / "b" / "up_1000", RENDER, up) is False
    assert not (tmp_path / "b" / "up_1000" / "images").exists()
    with pytest.raises(ValueError, match="incomplete"):
        ts.upload_set("tok", ts.DATA_REPO, _draw(tmp_path / "c" / "down_1000", up, upto=2), RENDER, up)
