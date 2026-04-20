from __future__ import annotations

import base64
import secrets


def new_message_id() -> str:
    return "msg_" + secrets.token_urlsafe(18)


def new_tool_use_id() -> str:
    return "toolu_" + secrets.token_urlsafe(18)


def new_thinking_signature() -> str:
    """Opaque proxy-local signature.

    NOT cryptographically valid vs Anthropic's API; used only so Anthropic-format
    clients that round-trip thinking blocks internally don't choke. Thinking
    blocks produced by this proxy cannot be replayed into the real Anthropic API.
    """
    return "proxy-" + base64.urlsafe_b64encode(secrets.token_bytes(36)).decode().rstrip("=")
