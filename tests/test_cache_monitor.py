"""Cache-monitor tests — SQLite counting, totals, rate/ETA, monitor loop (no GPU/torch)."""
from __future__ import annotations

import sqlite3

from geolip_anima_trainer import cache_monitor as cm


def _make_db(path, n):
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE items (i INTEGER)")
    con.executemany("INSERT INTO items VALUES (?)", [(x,) for x in range(n)])
    con.commit()
    con.close()


def test_db_count_and_count_done(tmp_path):
    root = tmp_path / "img" / "cache" / "anima"
    _make_db(root / "ar_1024x1024" / "latents_" / "metadata.db", 12)
    _make_db(root / "ar_1024x1024" / "text_embeddings_1_" / "metadata.db", 9)
    done = cm.count_done([root])
    assert done["latents_"] == 12 and done["text_embeddings_"] == 9
    # missing db -> 0, never raises
    assert cm._db_count(str(tmp_path / "nope.db")) == 0


def test_cache_bytes_is_the_continuous_signal(tmp_path):
    # bytes grow continuously even while the items table stays uncommitted (the fix)
    root = tmp_path / "man" / "cache" / "anima" / "latents_"
    root.mkdir(parents=True)
    (root / "shard_0.bin").write_bytes(b"x" * 1000)
    (root / "shard_1.bin").write_bytes(b"y" * 2000)
    (root / "metadata.db").write_bytes(b"z" * 99)        # not a .bin -> excluded
    assert cm.cache_bytes([tmp_path / "man" / "cache" / "anima"]) == 3000


def test_count_total_images(tmp_path):
    d = tmp_path / "man"
    d.mkdir()
    for name in ("a.png", "b.jpg", "c.webp", "a.txt", "captions.json"):
        (d / name).write_bytes(b"x")
    assert cm.count_total_images([d]) == 3   # only the 3 image files


def test_dataset_dirs_from_toml(tmp_path):
    ds = tmp_path / "dataset.toml"
    ds.write_text("resolutions = [1024]\n[[directory]]\npath = '/data/man'\n"
                  "[[directory]]\npath = '/data/woman'\n", encoding="utf-8")
    dirs = cm.dataset_dirs_from_toml(ds)
    assert [str(p).replace("\\", "/") for p in dirs] == ["/data/man", "/data/woman"]


def test_last_log_line_handles_tqdm_carriage_returns(tmp_path):
    log = tmp_path / "cache_vlm.log"
    # tqdm overwrites in place with \r (never a newline when piped to a file)
    log.write_bytes(b"loading models\ncaching latents: (1024, 1024)\n"
                    b"Grouping examples:  10%\rGrouping examples:  55%\r")
    assert cm.last_log_line(log) == "Grouping examples:  55%"
    # missing file / None -> '' (never raises)
    assert cm.last_log_line(tmp_path / "nope.log") == ""
    assert cm.last_log_line(None) == ""


def test_warmup_line_shows_log_tail(tmp_path, capsys):
    # no shards yet + a log_path -> the warm-up line surfaces diffusion-pipe's real phase
    root = tmp_path / "man" / "cache" / "anima"
    (root / "latents_").mkdir(parents=True)
    imgdir = tmp_path / "man"
    imgdir.mkdir(exist_ok=True)
    (imgdir / "a.png").write_bytes(b"x")
    log = tmp_path / "cache.log"
    log.write_text("caching latents: (1024, 1024)\n", encoding="utf-8")

    class _Proc:
        def poll(self):
            return 0      # exits immediately -> one final _tick()

    mon = cm.make_monitor(cache_roots=[root], dataset_dirs=[imgdir], captions_per_image=1,
                          interval=0, log_path=log, sleep=lambda s: None)
    mon(_Proc())
    out = capsys.readouterr().out
    assert "warming up" in out and "caching latents: (1024, 1024)" in out


def test_rate_and_fmt():
    r = cm._Rate(window_s=100)
    r.update(0.0, 0)
    r.update(10.0, 100)
    assert abs(r.rate() - 10.0) < 1e-6
    assert abs(r.eta(50) - 5.0) < 1e-6
    assert cm._fmt(125) == "02:05"
    assert cm._fmt(float("inf")) == "--:--"


def test_monitor_loop_runs_to_completion(tmp_path, capsys):
    root = tmp_path / "man" / "cache" / "anima"
    _make_db(root / "latents_" / "metadata.db", 5)
    imgdir = tmp_path / "man"
    for i in range(5):
        (imgdir / f"{i}.png").write_bytes(b"x")

    class _Proc:  # fake Popen: alive for 2 polls then exits
        def __init__(self):
            self.n = 0
        def poll(self):
            self.n += 1
            return None if self.n <= 2 else 0

    seen = []
    mon = cm.make_monitor(cache_roots=[root], dataset_dirs=[imgdir],
                          captions_per_image=1, interval=0,
                          on_update=seen.append, sleep=lambda s: None)
    mon(_Proc())
    out = capsys.readouterr().out
    assert "[cache]" in out and seen          # printed progress + fired callback
