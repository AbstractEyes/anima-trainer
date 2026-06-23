# Anima Trainer — Repo Brief (CLAUDE.md)

Finetune **CircleStone Labs Anima** (2B anime text-to-image; DiT backbone is NVIDIA
Cosmos-Predict2-2B) using **tdrussell/diffusion-pipe** — the only trainer that
supports Anima. We *bridge* an HF `datasets`-format parquet repo into the directory
layout diffusion-pipe requires; we do **not** switch trainers.

## Implementation status (this repo is now a package)
The original flat `scripts/`+`configs/` brief (below) is implemented as the
installable package **`geolip_anima_trainer`** (console command **`anima`**):

- `config.py` — composable dataclasses + presets + `apply_overrides` + `sweep` +
  `validate` (the "elemental construction" engine). The Anima invariants live in
  `validate()`.
- `launch.py` — `build_plan`/`launch`: builds the `deepspeed --num_gpus=N train.py
  --deepspeed --config ...` command (native multi-GPU), guarded so it never execs
  on Windows.
- `doctor.py` — environment diagnostics (`anima doctor`).
- `download_anima.py`, `hf_to_diffusion_pipe.py`, `build_multiconcept_dataset.py` —
  the original bridge scripts (still runnable standalone; now also importable).
- `api.py` / `cli.py` — the Python API and the `anima` CLI.
- `templates/anima_lora.toml`, `templates/anima_dataset.toml` — packaged templates;
  copy into `./configs` with `anima init-config`.

Workflow: `anima inspect` → `anima export` → `anima build` → (target) `anima cache`
→ `anima train --num-gpus N`. See README.md for the full command sequence.

## Environment (fixed facts — do not re-derive)
- **Target GPU:** RTX PRO 6000 Blackwell, 96 GB, **sm_120**, one or many on a shared box.
- **bf16 throughout. No fp8** (e4m3 degrades this lineage; 96 GB makes fp8 pointless).
- **Do NOT install flash-attn** — broken on sm_120. diffusion-pipe runs on **SDPA**.
- No block swapping, no activation checkpointing (VRAM abundant).
- Python 3.12. Blackwell needs **torch ≥ 2.7 / cu128**.
- **Local box is an RTX 4090 (sm_89, Windows) — smoke-test only, never a full train.**

## Model files (circlestone-labs/Anima, split_files/)
| file | role | size |
|---|---|---|
| `diffusion_models/anima-base-v1.0.safetensors` | DiT base (recommended) | 4.18 GB |
| `text_encoders/qwen_3_06b_base.safetensors` | Qwen-3 0.6B base text encoder | ~1.2 GB |
| `vae/qwen_image_vae.safetensors` | Qwen-Image VAE | ~250 MB |

Fetch with `anima download --dest models/anima --base base-v1.0` (huggingface_hub,
never wget). It prints the exact `[model]` paths for `configs/anima_lora.toml`.

## Dataset
- Repo: `AbstractPhil/diffusion-pretrain-set-ft1`; config: **`qwen_90k`** (83.0K rows,
  high-res, good for 1024). Uniform 17-col schema.
- Captions are **JSON**: `caption_animetimm_json` (booru-style), `caption_vlm_json`,
  `captions_source_json`. The bridge renders them to tag strings.
- Quality gates in-schema: `audit == "approved"`, `age_classifier_pass == True`
  (on by default). **May be unpopulated for qwen_90k — `anima inspect` first.**

## Non-negotiable rules
- **`llm_adapter_lr = 0`** (adapter frozen). High knowledge density, degrades easily.
  Only raise it (≤5e-6) for genuinely-new concepts, A/B'd against a frozen baseline.
  `validate()` hard-errors above 5e-6.
- **Anima is tag-order sensitive**: quality → subject → character → series → @artist →
  general. `shuffle_caption = false` (unless `keep_tokens` protects the leading block).
- **License.** This repo's *code* is Apache-2.0 (see LICENSE). The *Anima model* and any
  weights finetuned from it are **non-commercial** — CircleStone NC + NVIDIA Open Model
  License (Cosmos derivative). Any LoRA/finetune is a derivative — NC only; the permissive
  code license does not relax that.
- diffusion-pipe input = directories of image files + matching `.txt` sidecars; it
  uses the `datasets` library only as an internal latent/embedding cache.

## Open decisions — RESOLVED from `anima inspect qwen_90k` (2026-06-23, 64-row sample)
- **Caption column → `caption_vlm_json` (`--caption-format vlm`).** `caption_animetimm_json`
  is the literal string `__PARSEFAIL__` for this slice (upstream tagging failed);
  `caption_vlm_json` renders clean booru-style tags. `captions_source_json` is natural
  language (fallback only).
- **Age gate → MUST pass `--no-age-filter`.** `age_classifier_pass` is unpopulated
  (`None`) here; the default gate would drop **every** row (0/64 passed).
- **Audit gate → keep ON.** `audit` was `"approved"` for all sampled rows.
- **Schema note:** `image` is a `{bytes, path}` dict (not a decoded PIL Image); the
  bridge's `_to_pil` handles this. `conditioning_image`/`mask` were `None` in the sample.
- **Single vs multi-concept:** the stream routes by `source` and is NOT uniform (the
  first shard was `sdxl_qwen_phase0`) — so this is a **multi-concept** run
  (`anima.multi_concept_preset(...)`, rank 64, balanced). Re-inspect a larger sample
  on the target to enumerate the full `source` distribution before final balancing.

  Example extract command (target box):
  ```
  anima export --repo AbstractPhil/diffusion-pretrain-set-ft1 --configs qwen_90k \
      --out datasets/anima_qwen90k --caption-column caption_vlm_json \
      --caption-format vlm --route-by source --no-age-filter --limit-per-concept 8000
  ```
