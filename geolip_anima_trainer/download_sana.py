#!/usr/bin/env python3
"""
download_sana.py — fetch one diffusers-format Sana checkpoint for diffusion-pipe.

Sana (NVlabs, arXiv 2410.10629) trains through model type 'sana' in the AbstractEyes
diffusion-pipe fork, which loads a whole diffusers folder (transformer + DC-AE autoencoder +
Gemma-2 text encoder + tokenizer). The diffusers repos carry each weight twice (a plain file
plus an fp16 / bf16 / int4 variant); the loader reads the plain files, so the variants are
skipped (several GB per repo).

Licences: the Sana diffusers checkpoints are Apache-2.0; the bundled Gemma-2-2B-IT text encoder
is under Google's Gemma Terms of Use and Prohibited Use Policy (see the model card).

Run:
    python download_sana.py --dest /data/models/sana --variant 600m-512
Then paste the printed path into sana_lora.toml's [model] block (diffusers_path).
"""

import argparse
from pathlib import Path

# Short --variant names -> diffusers repos (all checked on the Hub; Apache-2.0). The pixel
# size is the one each checkpoint was trained at: use it as the dataset resolution.
SANA_REPOS = {
    "600m-512":        ("Efficient-Large-Model/Sana_600M_512px_diffusers", 512),
    "600m-1024":       ("Efficient-Large-Model/Sana_600M_1024px_diffusers", 1024),
    "1600m-512":       ("Efficient-Large-Model/Sana_1600M_512px_diffusers", 512),
    "1600m-1024":      ("Efficient-Large-Model/Sana_1600M_1024px_diffusers", 1024),
    "1600m-1024-bf16": ("Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers", 1024),
}
DEFAULT_VARIANT = "600m-512"

# The precision variants the loader never reads (it loads the plain weight files).
VARIANT_FILES = ["*.fp16.*", "*.fp16-*", "*.bf16.*", "*.bf16-*", "*.int4.*"]


def fetch(repo_id: str, dest: Path) -> str:
    """Download one Sana diffusers repo into dest/<repo name>, without the precision variants.
    Returns the local folder (the [model] diffusers_path).

    huggingface_hub is imported HERE, not at module top, so `import geolip_anima_trainer`
    stays hf-free (the hub cache dir is fixed from HF_HOME at hf's FIRST import)."""
    from huggingface_hub import snapshot_download
    local = Path(dest) / repo_id.split("/")[-1]
    return snapshot_download(repo_id=repo_id, local_dir=str(local), ignore_patterns=VARIANT_FILES)


def main() -> None:
    ap = argparse.ArgumentParser(description="Download a diffusers-format Sana checkpoint for diffusion-pipe.")
    ap.add_argument("--dest", default="./sana_models", help="Local directory to download into.")
    ap.add_argument("--variant", default=DEFAULT_VARIANT, choices=list(SANA_REPOS),
                    help="Which Sana checkpoint to fetch.")
    args = ap.parse_args()

    dest = Path(args.dest).expanduser().resolve()
    dest.mkdir(parents=True, exist_ok=True)
    repo_id, native = SANA_REPOS[args.variant]
    print(f"Repo:        {repo_id}")
    print(f"Destination: {dest}\n")
    path = fetch(repo_id, dest)
    print("\n# ---- paste into sana_lora.toml [model] ----")
    print(f"diffusers_path = '{path}'")
    print(f"# dataset resolutions = [{native}]")


if __name__ == "__main__":
    main()
