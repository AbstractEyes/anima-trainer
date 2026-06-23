"""Bridge tests — image coercion + caption rendering (pure, no network)."""
from __future__ import annotations

import io

from PIL import Image

from geolip_anima_trainer import hf_to_diffusion_pipe as B


def _png_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (4, 4), (10, 20, 30)).save(buf, "PNG")
    return buf.getvalue()


def test_to_pil_handles_all_forms():
    assert B._to_pil(None) is None
    assert B._to_pil({"bytes": None, "path": None}) is None
    # the real AbstractPhil schema: image is a {bytes, path} dict
    assert B._to_pil({"bytes": _png_bytes(), "path": None}).size == (4, 4)
    # already a PIL image
    assert B._to_pil(Image.new("L", (2, 2))).size == (2, 2)


def test_render_caption_vlm_structured():
    # caption_vlm_json renders the subjects/actions/setting structure to tags.
    raw = '{"subjects":[{"name":"officer","attributes":["police"]}],' \
          '"actions":["standing"],"setting":"street"}'
    out = B.render_caption(raw, "vlm")
    assert "police officer" in out and "standing" in out and "street" in out


def test_render_caption_raw_and_parsefail():
    assert B.render_caption("__PARSEFAIL__", "animetimm") == "__PARSEFAIL__"
    assert B.render_caption("just text", "raw") == "just text"
    assert B.render_caption(None, "vlm") == ""


def test_render_caption_source_takes_first_string():
    raw = '{"sdxl_caption": "a red car on a road"}'
    assert B.render_caption(raw, "source") == "a red car on a road"


def test_row_passes_gates():
    cfg = B.BridgeConfig()  # audit+age gates ON by default
    assert B.row_passes({"audit": "approved", "age_classifier_pass": True}, cfg)
    assert not B.row_passes({"audit": "approved", "age_classifier_pass": None}, cfg)
    # qwen_90k: age col unpopulated -> must disable the age gate
    cfg2 = B.BridgeConfig(require_age_pass=False)
    assert B.row_passes({"audit": "approved", "age_classifier_pass": None}, cfg2)
