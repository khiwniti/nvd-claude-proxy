"""Anthropic image blocks → OpenAI `image_url` objects.

NVIDIA VLMs accept JPG/PNG only; GIF/WEBP must be transcoded to PNG.
"""
from __future__ import annotations

import base64
import io

from PIL import Image

_SUPPORTED_NIM = {"image/jpeg", "image/png"}


def _transcode_to_png(raw: bytes) -> bytes:
    img = Image.open(io.BytesIO(raw))
    # Flatten transparency to a white background so the PNG stays JPEG-safe
    # even if a downstream component re-encodes it.
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, img).convert("RGB")
    else:
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def anthropic_image_to_openai(block: dict) -> dict:
    src = block["source"]
    stype = src["type"]
    if stype == "url":
        return {"type": "image_url", "image_url": {"url": src["url"]}}
    if stype == "base64":
        media_type = src["media_type"]
        data_b64 = src["data"]
        if media_type not in _SUPPORTED_NIM:
            raw = base64.b64decode(data_b64)
            data_b64 = base64.b64encode(_transcode_to_png(raw)).decode()
            media_type = "image/png"
        return {
            "type": "image_url",
            "image_url": {"url": f"data:{media_type};base64,{data_b64}"},
        }
    if stype == "file":
        raise ValueError(
            "Anthropic Files-API image source is not supported; send base64 or URL."
        )
    raise ValueError(f"Unknown image source type: {stype!r}")
