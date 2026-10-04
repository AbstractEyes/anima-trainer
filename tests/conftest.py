"""Shared test doubles: a Hub dataset repo in a temp folder for training_sets (and the runners that use it)."""
from __future__ import annotations

import fnmatch
import shutil
from pathlib import Path

import pytest


class FakeHub:
    """A dataset repo in a folder: file_exists / upload_folder (fnmatch patterns, as the hub client) / snapshot_download."""

    def __init__(self, d: Path):
        self.d, self.commits = d, []

    def file_exists(self, repo_id, path, repo_type=None):
        assert repo_type == "dataset"
        return (self.d / path).is_file()

    def upload_folder(self, repo_id, repo_type, folder_path, path_in_repo, allow_patterns, ignore_patterns,
                      commit_message):
        assert repo_type == "dataset"
        for f in Path(folder_path).rglob("*"):
            rel = f.relative_to(folder_path).as_posix()
            if f.is_file() and any(fnmatch.fnmatch(rel, p) for p in allow_patterns) \
                    and not any(fnmatch.fnmatch(rel, p) for p in ignore_patterns):
                (self.d / path_in_repo / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(f, self.d / path_in_repo / rel)
        self.commits.append(commit_message)

    def snapshot_download(self, repo_id, repo_type=None, allow_patterns=None, local_dir=None, token=None):
        for f in self.d.rglob("*"):
            rel = f.relative_to(self.d).as_posix()
            if f.is_file() and any(fnmatch.fnmatch(rel, p) for p in allow_patterns):
                (Path(local_dir) / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(f, Path(local_dir) / rel)
        return local_dir


@pytest.fixture
def fake_hub(tmp_path, monkeypatch):
    import huggingface_hub

    from geolip_anima_trainer import training_sets as ts
    h = FakeHub(tmp_path / "hub")
    h.d.mkdir()
    monkeypatch.setattr(ts, "_api", lambda token: h)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", h.snapshot_download)
    return h
