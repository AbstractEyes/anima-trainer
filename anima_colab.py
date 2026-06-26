#!/usr/bin/env python3
"""
anima_colab.py — the ONLY repo code the Colab cache-factory notebook imports *before* the
package's heavy deps exist. It lives at the repo ROOT (not inside geolip_anima_trainer) and
imports stdlib ONLY, so `import anima_colab` works on a bare Colab runtime where torch /
pyarrow / huggingface_hub are not installed yet (importing the package would fail there).

The whole point of the third notebook: keep the NOTEBOOK static (a handful of calls) and put
every behavior that might change in the REPO, so iterating = `git pull`, never re-pasting
cells. This module is step 0 (clone + install); geolip_anima_trainer.cache_factory.CacheFactory
is everything after (env / auth / extract / cache / push), driven from stable one-line cells.

    import anima_colab
    REPO = anima_colab.ensure_repo()                  # clone-or-pull this repo
    if anima_colab.install(REPO):                     # idempotent; True => torch (re)installed
        import os; os.kill(os.getpid(), 9)            # restart ONCE so the new torch loads
    # (reconnect, re-run the cell: install now skips, returns False -> no restart)
    from geolip_anima_trainer.cache_factory import CacheFactory
    CacheFactory().run_all()
"""

from __future__ import annotations

import os
import subprocess
import sys

REPO_URL = "https://github.com/AbstractEyes/anima-trainer.git"
DP_URL = "https://github.com/tdrussell/diffusion-pipe.git"
TORCH_INDEX = "https://download.pytorch.org/whl/cu128"   # cu128 wheels cover sm_80 (A100) .. sm_120
_MARKER = ".anima_colab_installed"                       # written after a full install on THIS runtime


def _sh(cmd: str, *, check: bool = True) -> int:
    print("$", cmd, flush=True)
    rc = subprocess.run(cmd, shell=True).returncode
    if check and rc:
        raise SystemExit(f"command failed ({rc}): {cmd}")
    return rc


def _pip(*args: str) -> None:
    _sh(f'"{sys.executable}" -m pip -q install ' + " ".join(args))


def repo_root() -> str:
    """Where the repo lives — $ANIMA_REPO or /content/anima-trainer (Colab default)."""
    return os.environ.get("ANIMA_REPO", "/content/anima-trainer")


def ensure_repo(repo: str | None = None, *, url: str = REPO_URL, pull: bool = True) -> str:
    """Clone the trainer repo (and diffusion-pipe with submodules) if missing; else `git pull`
    so the latest repo logic is what runs. Returns the repo path; also puts it on sys.path so
    `import geolip_anima_trainer` resolves from source even without an editable install."""
    repo = repo or repo_root()
    if not os.path.isfile(f"{repo}/pyproject.toml"):
        _sh(f'git clone "{url}" "{repo}"')
    elif pull:
        _sh(f'cd "{repo}" && git pull --ff-only', check=False)  # stay on the latest logic; never fatal
    dp = f"{repo}/external/diffusion-pipe"
    if not os.path.isfile(f"{dp}/train.py"):
        _sh(f'git clone --recurse-submodules "{DP_URL}" "{dp}"')
    assert os.path.isfile(f"{dp}/train.py"), "diffusion-pipe/train.py missing — clone failed"
    if repo not in sys.path:
        sys.path.insert(0, repo)
    os.environ["ANIMA_REPO"] = repo
    return repo


def install(repo: str | None = None, *, similarity: bool = True, force: bool = False) -> bool:
    """Install torch (cu128) + diffusion-pipe requirements + this package ([similarity]) + the
    datasets<3 pin. IDEMPOTENT: a marker file under the repo means 'already done on this runtime'
    so a re-run after the restart skips. Returns True iff a restart is recommended (torch was
    (re)installed) — the caller restarts ONCE, reconnects, and re-runs (which then returns False).

    This is the one spot most likely to need tweaking for a new Colab image; it's in the repo on
    purpose, so a `git pull` changes it with no notebook edit. NOTE: the marker tracks 'an install
    ran on THIS runtime', not 'deps match the current repo' — a `git pull` that ADDS a dependency
    needs `install(force=True)` (or a fresh runtime) to be picked up; pure-Python logic changes need
    neither."""
    repo = ensure_repo(repo)
    marker = os.path.join(repo, _MARKER)
    if os.path.exists(marker) and not force:
        print("[anima_colab] deps already installed on this runtime -> skipping (no restart).", flush=True)
        return False
    dp = f"{repo}/external/diffusion-pipe"
    _pip(f"--index-url {TORCH_INDEX} torch torchvision")          # Blackwell/Ampere-safe torch
    _pip(f"-r {dp}/requirements.txt")
    extra = "[similarity]" if similarity else ""
    _pip(f'-e "{repo}{extra}"')                                  # the bridge package + its deps
    _pip('"datasets>=2.19,<3"')                                  # diffusion-pipe needs datasets<3
    _pip(f"--index-url {TORCH_INDEX} torch torchvision")          # re-pin torch LAST so nothing overrode it
    with open(marker, "w", encoding="utf-8") as f:
        f.write("ok\n")
    print("\n[anima_colab] install complete. RESTART once so the new torch loads, then re-run.", flush=True)
    return True
