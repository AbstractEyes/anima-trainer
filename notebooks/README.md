# Colab notebooks

## `anima_colab_prelim_train.ipynb`
Preliminary **~1000-image, rank-64 Anima LoRA** trained on **semantic subject buckets**, on a
Google Colab **RTX PRO 6000 Blackwell** (sm_120, 96 GB) runtime — the repo's real training
target. Drives the `geolip_anima_trainer` package + diffusion-pipe; backs checkpoints up to
HuggingFace every few minutes so the LoRA survives a Colab disconnect.

### Before you run
- Select a **RTX PRO 6000 Blackwell** GPU runtime (the notebook installs the cu128 torch
  build Blackwell/sm_120 requires and verifies it).
- Add a **write-scoped** HuggingFace token to Colab **Secrets** (🔑) named **`HF_TOKEN`**
  (used to read the dataset/model and to push checkpoint backups).

### Run order
Top to bottom. There is **one runtime restart** right after the install cell (§2) — expected.
§1 GPU → §2 clone+install(→restart) → §3 verify+`anima doctor` → §4 HF auth → §5 download
model → **§6 extract ~1000 into subject buckets** → **§7 build rank-64 config + dampened
weighting** → §8 cache → §9 train(bg)+periodic backup → §10 notes.

### Methodology (baked in, from `CLAUDE.md` for `qwen_90k`)
- Caption = **`caption_vlm_json` `task_1` JSON trained VERBATIM** (+ `caption_animetimm_json`
  as a hardlinked 2nd sample) — not rendered to tags.
- **Subject buckets** (`anima subjects`, columnar pyarrow): dominant-subject keys; sparse
  subjects grouped by **semantic similarity** (`[similarity]` extra: sentence-transformers,
  falls back to char-trigram/difflib), never dropped; **human subgroups kept separate**.
- **Anti-overtraining weighting:** `num_repeats` via the diminishing-returns policy
  (`balance_alpha=0.5`, capped at `max_repeats=8`), not the legacy 50× equalization.
- `require_age_pass=False` (age col unpopulated; audit gate **on**), `limit=1000`,
  `llm_adapter_lr=0` (frozen), `shuffle_caption=false` (tag-order sensitive).

### License
Anima and any LoRA from it are **NON-COMMERCIAL** (CircleStone NC + NVIDIA Open Model
License / Cosmos derivative). The backup model card is labelled accordingly.
