#!/usr/bin/env python3
"""
download_anima.py — fetch the three Anima ComfyUI files diffusion-pipe needs.

Uses huggingface_hub exclusively (NO wget). Downloads only the files required,
not the whole 16GB repo. Verified file layout (circlestone-labs/Anima):

    split_files/diffusion_models/
        anima-base-v1.0.safetensors      (4.18 GB)  <- finalized base, recommended
        anima-preview.safetensors        (4.18 GB)
        anima-preview2.safetensors       (4.18 GB)
        anima-preview3-base.safetensors  (4.18 GB)  <- latest preview (more 1024 training)
    split_files/text_encoders/
        qwen_3_06b_base.safetensors
    split_files/vae/
        qwen_image_vae.safetensors

Run:
    python download_anima.py --dest /data/models/anima --base base-v1.0
Then paste the printed paths into anima_lora.toml's [model] block.
"""

import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_ID = "circlestone-labs/Anima"

# Map a short --base choice to the actual repo filename.
BASE_CHOICES = {
    "base-v1.0": "anima-base-v1.0.safetensors",
    "preview":   "anima-preview.safetensors",
    "preview2":  "anima-preview2.safetensors",
    "preview3":  "anima-preview3-base.safetensors",
}

# These two are shared by every workflow regardless of which base you pick.
TEXT_ENCODER = "split_files/text_encoders/qwen_3_06b_base.safetensors"
VAE          = "split_files/vae/qwen_image_vae.safetensors"


def fetch(repo_filename: str, dest: Path) -> str:
    """Download one repo file into `dest`, preserving no nesting. Returns local path."""
    # local_dir keeps the repo's internal subfolders; we resolve the real path after.
    local_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=repo_filename,
        local_dir=str(dest),
    )
    return local_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Download Anima ComfyUI files for diffusion-pipe.")
    ap.add_argument("--dest", default="./anima_models",
                    help="Local directory to download into.")
    ap.add_argument("--base", default="base-v1.0", choices=list(BASE_CHOICES),
                    help="Which diffusion checkpoint to use as the training base.")
    args = ap.parse_args()

    dest = Path(args.dest).expanduser().resolve()
    dest.mkdir(parents=True, exist_ok=True)

    base_file = f"split_files/diffusion_models/{BASE_CHOICES[args.base]}"

    print(f"Repo:        {REPO_ID}")
    print(f"Destination: {dest}")
    print(f"Base ckpt:   {args.base} -> {BASE_CHOICES[args.base]}\n")

    transformer_path = fetch(base_file, dest)
    llm_path         = fetch(TEXT_ENCODER, dest)
    vae_path         = fetch(VAE, dest)

    # Print exactly what to drop into the [model] block of anima_lora.toml.
    print("\n# ---- paste into anima_lora.toml [model] ----")
    print(f"transformer_path = '{transformer_path}'")
    print(f"vae_path         = '{vae_path}'")
    print(f"llm_path         = '{llm_path}'")


if __name__ == "__main__":
    main()