"""connectors_run on the CPU: the batch probe's choice (the least time per image, a size within 5% going to the smaller one,
a size the card cannot hold skipped, the runner's batch restored) and the arm check before any setup."""
import pytest
import torch

from geolip_anima_trainer import connectors_run as cr


class FakeRunner:
    """_render costs `per_image[b]` seconds an image at batch b on a fake clock; it runs out of memory at the sizes in `oom`,
    or at every call from call number `oom_from` on."""

    def __init__(self, per_image, oom=(), oom_from=None):
        self.cfg = type("Cfg", (), {"gen_batch": 8})()
        self.BED = type("Bed", (), {"neutral_caption": "a photo of {s}"})()
        self.per_image, self.oom, self.oom_from = per_image, set(oom), oom_from
        self.calls, self.clock = [], 0.0

    def _render(self, prompts, seeds):
        b = self.cfg.gen_batch
        assert len(prompts) == len(seeds) == b and min(seeds) >= 9000       # throwaway seeds, never a scored cell
        self.calls.append(b)
        if b in self.oom or (self.oom_from is not None and len(self.calls) > self.oom_from):
            raise torch.cuda.OutOfMemoryError("fake")
        self.clock += self.per_image.get(b, 1.0) * b
        return [None] * b


@pytest.fixture
def fake_clock(monkeypatch):
    holder = {}
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda *a, **k: None)
    monkeypatch.setattr(cr.time, "time", lambda: holder["r"].clock)
    return holder


def test_the_probe_keeps_the_fastest_size_and_restores_the_batch(fake_clock):
    r = fake_clock["r"] = FakeRunner({8: 1.0, 16: 0.80, 32: 0.79})
    chosen, per = cr.probe_batches(r)
    assert chosen == 16                                   # 32 is within 5% of 16's time: the smaller size wins
    assert per == pytest.approx({8: 1.0, 16: 0.80, 32: 0.79})
    assert r.calls == [8, 8, 16, 32] and r.cfg.gen_batch == 8          # a warm-up batch first; the runner's batch kept


def test_the_probe_skips_a_size_the_card_cannot_hold(fake_clock):
    r = fake_clock["r"] = FakeRunner({8: 1.0, 16: 0.5, 32: 0.1}, oom={32})
    chosen, per = cr.probe_batches(r)
    assert chosen == 16 and set(per) == {8, 16} and r.cfg.gen_batch == 8


def test_the_probe_fails_when_no_size_fits(fake_clock):
    r = fake_clock["r"] = FakeRunner({8: 1.0}, oom_from=1)              # the warm-up fits, then nothing does
    with pytest.raises(RuntimeError, match="no render batch"):
        cr.probe_batches(r)
    assert r.calls == [8, 8, 16, 32] and r.cfg.gen_batch == 8
    r = fake_clock["r"] = FakeRunner({8: 1.0}, oom={8, 16, 32})         # not even the warm-up: the card's own error
    with pytest.raises(torch.cuda.OutOfMemoryError):
        cr.probe_batches(r)
    assert r.cfg.gen_batch == 8


def test_unknown_arms_stop_before_any_setup(monkeypatch):
    import geolip_anima_trainer.anima_runner as ar
    monkeypatch.setattr(ar, "AnimaRunner", lambda *a, **k: pytest.fail("setup reached"))
    with pytest.raises(SystemExit, match="unknown connector arm"):
        cr.main(["--arms", "e031_beatrix_stream_closing_slider,e999_nothing", "--data-root", "x", "--models-dir", "y"])


def test_the_arguments():
    a = cr.parse(["--arms", "e031_a, e032_b", "--data-root", "d", "--models-dir", "m"])
    assert a.eval_batch == "auto" and not a.offline and not a.force and a.repo_root is None
