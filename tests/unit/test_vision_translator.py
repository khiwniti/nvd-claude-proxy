from __future__ import annotations

import base64
import io

from PIL import Image

from nvd_claude_proxy.translators.vision_translator import anthropic_image_to_openai


def _b64_png(size=(4, 4), mode="RGB") -> str:
    img = Image.new(mode, size, (1, 2, 3))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _b64_gif() -> str:
    img = Image.new("P", (4, 4))
    buf = io.BytesIO()
    img.save(buf, format="GIF")
    return base64.b64encode(buf.getvalue()).decode()


def test_url_passthrough():
    b = {"type": "image", "source": {"type": "url", "url": "https://x/y.png"}}
    out = anthropic_image_to_openai(b)
    assert out == {"type": "image_url", "image_url": {"url": "https://x/y.png"}}


def test_png_base64_passthrough():
    data = _b64_png()
    b = {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": data},
    }
    out = anthropic_image_to_openai(b)
    assert out["image_url"]["url"].startswith("data:image/png;base64,")


def test_gif_transcoded_to_png():
    data = _b64_gif()
    b = {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/gif", "data": data},
    }
    out = anthropic_image_to_openai(b)
    assert out["image_url"]["url"].startswith("data:image/png;base64,")
