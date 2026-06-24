"""Auto-loaded compat shim for diffusion-pipe subprocesses.

Python imports a module named `sitecustomize` at interpreter startup if one is on
sys.path. `geolip_anima_trainer.launch.env_prefix()` puts THIS directory on PYTHONPATH
so the patch below is applied in every diffusion-pipe subprocess BEFORE diffusion-pipe
forks its caching child process.

Self-contained on purpose (no `geolip_anima_trainer` import): importing the package would
pull its whole API surface at every subprocess startup, and the shim must work even if only
this directory is on the path. This is a deliberate mirror of
`geolip_anima_trainer.dp_compat.patch_datasets_map_makedirs` — see that module for the full
rationale (diffusion-pipe writes Dataset.map cache files into an `ar_frames_X/metadata/`
dir it never creates; datasets won't mkdir it; the resulting crash hangs the run).
"""

import os


def _patch_datasets_map_makedirs():
    try:
        from datasets import Dataset
    except Exception:  # datasets not importable in this process -> nothing to do
        return
    if getattr(Dataset.map, "_anima_mkdir_patched", False):
        return
    _orig = Dataset.map

    def map(self, *args, **kwargs):
        cfn = kwargs.get("cache_file_name")
        if cfn:
            d = os.path.dirname(str(cfn))
            if d:
                os.makedirs(d, exist_ok=True)
        return _orig(self, *args, **kwargs)

    map._anima_mkdir_patched = True
    try:
        Dataset.map = map
    except Exception:
        pass


_patch_datasets_map_makedirs()
