"""PDF → plain-text extraction for Anthropic `document` content blocks.

Uses pypdf when available (optional dependency).  Falls back to a warning
message so the model at least knows a PDF was supplied even if we cannot
extract its text.

Anthropic `document` block shape::

    {
        "type": "document",
        "source": {
            "type": "base64",          # only base64 supported today
            "media_type": "application/pdf",
            "data": "<base64-string>",
        },
        "title": "optional title",
        "context": "optional context hint",
        "citations": {"enabled": false},
    }

We convert this into a text block whose text is:

    [Document: <title>]
    <extracted text>
"""
from __future__ import annotations

import base64
import io
import logging

_log = logging.getLogger("nvd_claude_proxy.pdf")

try:
    from pypdf import PdfReader as _PdfReader

    _PYPDF_AVAILABLE = True
except ImportError:
    _PYPDF_AVAILABLE = False
    _log.warning(
        "pypdf not installed — PDF document blocks will be replaced with a "
        "placeholder. `pip install pypdf` to enable PDF extraction."
    )


def extract_pdf_text(b64_data: str) -> str:
    """Decode a base64-encoded PDF and return its full text content."""
    raw = base64.b64decode(b64_data)
    if not _PYPDF_AVAILABLE:
        return f"[PDF document — {len(raw)} bytes — text extraction unavailable; install pypdf]"
    try:
        reader = _PdfReader(io.BytesIO(raw))
        parts: list[str] = []
        for page in reader.pages:
            text = page.extract_text() or ""
            if text.strip():
                parts.append(text)
        return "\n\n".join(parts) if parts else "[PDF document — no extractable text]"
    except Exception as exc:
        _log.warning("pdf.extraction_failed", error=str(exc))
        return f"[PDF document — extraction failed: {exc}]"


def document_block_to_text(block: dict) -> str:
    """Convert an Anthropic `document` block to a plain-text string.

    Handles:
      - ``source.type == "base64"`` + ``media_type == "application/pdf"``
      - ``source.type == "text"`` (plain-text documents)
      - ``source.type == "url"`` — we cannot fetch at proxy time; emit placeholder

    Prepends the ``title`` and ``context`` hints if provided so the model
    understands what it is reading.
    """
    source = block.get("source") or {}
    title = block.get("title") or ""
    context = block.get("context") or ""
    header_parts: list[str] = []
    if title:
        header_parts.append(f"[Document: {title}]")
    if context:
        header_parts.append(f"[Context: {context}]")
    header = "\n".join(header_parts)

    src_type = source.get("type")
    media_type = source.get("media_type", "")

    if src_type == "text":
        body = source.get("data") or ""
    elif src_type == "base64" and "pdf" in media_type:
        body = extract_pdf_text(source.get("data") or "")
    elif src_type == "base64":
        body = f"[Binary document ({media_type}) — {len(source.get('data') or '')} chars base64]"
    elif src_type == "url":
        body = f"[Document URL: {source.get('url', '?')} — cannot fetch at proxy]"
    else:
        body = f"[Document (unknown source type: {src_type})]"

    return f"{header}\n{body}".strip() if header else body
