"""diffusion-pipe compat shim — the datasets Dataset.map mkdir fix."""
from __future__ import annotations

from pathlib import Path

import pytest

from geolip_anima_trainer import dp_compat


def test_shim_dir_has_sitecustomize():
    # launch puts this dir on PYTHONPATH; Python auto-imports its sitecustomize.py
    d = Path(dp_compat.shim_dir())
    assert (d / "sitecustomize.py").is_file()


def test_patch_makes_map_create_missing_cache_dir(tmp_path):
    # reproduces the diffusion-pipe crash: Dataset.map(cache_file_name=...) into a dir that
    # doesn't exist. Before the patch -> FileNotFoundError; after -> the dir is created.
    datasets = pytest.importorskip("datasets")
    ds = datasets.Dataset.from_dict({"x": [1, 2, 3]})
    target = tmp_path / "ar_frames_1.000_1" / "metadata" / "metadata_x.arrow"
    assert not target.parent.exists()

    assert dp_compat.patch_datasets_map_makedirs() is True
    out = ds.map(lambda e: {"y": e["x"] * 2}, cache_file_name=str(target))
    assert target.parent.is_dir()           # the missing parent was created
    assert out["y"] == [2, 4, 6]            # map still works + passes kwargs through

    # idempotent: a second call is a no-op that still reports success
    assert dp_compat.patch_datasets_map_makedirs() is True
    assert getattr(datasets.Dataset.map, "_anima_mkdir_patched", False) is True
