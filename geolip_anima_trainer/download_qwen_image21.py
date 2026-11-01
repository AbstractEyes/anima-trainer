#!/usr/bin/env python3
"""
download_qwen_image21.py — fetch the three Qwen-Image-2.1 files diffusion-pipe trains from.

Qwen-Image-2.1 trains through model type 'qwen_image21' in the AbstractEyes diffusion-pipe fork, which
loads the ComfyUI split files of Comfy-Org/Qwen-Image-2.1: the bf16 transformer (train on this one; the
int8 file of ComfyUI's template is for inference), the RGBA VAE and the Qwen3-VL-8B text encoder (int8,
as ComfyUI's template loads it; it only runs while caching). The repo's folder layout is kept under dest.

Licence: the Qwen Research License (research and evaluation; commercial use needs a licence from Qwen;
derivatives display "Built with Qwen"). See the model card.

Run:
    python download_qwen_image21.py --dest /data/models/qwen_image_2.1
Then paste the printed paths into qwen_image21_lora.toml's [model] block.
"""

import argparse
from pathlib import Path

REPO = "Comfy-Org/Qwen-Image-2.1"
FILES = {
    "diffusion_model": "diffusion_models/qwen_image_2.1_bf16.safetensors",   # 14.2 GB
    "vae": "vae/qwen_image_2.1_vae_bf16.safetensors",                        # 0.7 GB
    "text_encoder": "text_encoders/qwen3vl_8b_int8_convrot.safetensors",     # 9.4 GB
}


def fetch(dest: Path, revision: "str | None" = None) -> dict:
    """Download the three files into dest (the repo's folder layout). Returns {role: local path}.

    huggingface_hub is imported HERE, not at module top, so `import geolip_anima_trainer`
    stays hf-free (the hub cache dir is fixed from HF_HOME at hf's FIRST import)."""
    from huggingface_hub import hf_hub_download
    return {role: hf_hub_download(REPO, rel, local_dir=str(dest), revision=revision) for role, rel in FILES.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description="Download the Qwen-Image-2.1 files for diffusion-pipe.")
    ap.add_argument("--dest", default="./qwen_image_2.1", help="Local directory to download into.")
    ap.add_argument("--revision", default=None, help="A commit of Comfy-Org/Qwen-Image-2.1 (default: the newest).")
    args = ap.parse_args()

    dest = Path(args.dest).expanduser().resolve()
    dest.mkdir(parents=True, exist_ok=True)
    print(f"Repo:        {REPO}")
    print(f"Destination: {dest}\n")
    paths = fetch(dest, args.revision)
    print("\n# ---- paste into qwen_image21_lora.toml [model] ----")
    print(f"diffusion_model = '{paths['diffusion_model']}'")
    print(f"vae = '{paths['vae']}'")
    print(f"text_encoders = [{{path = '{paths['text_encoder']}', type = 'qwen_image'}}]")


if __name__ == "__main__":
    main()
