"""Scenario-based routing logic for nvd-claude-proxy.

Ported from claude-code-router's ``router.ts``.

This module decides which model to use based on the request content:
- ``long_context``: Input token count exceeds a threshold.
- ``background``: Fast models (Haiku).
- ``web_search``: Request contains web search tools.
- ``think``: Request enables thinking mode.
- ``vision``: Request contains image content.
- ``default``: Fallback.
"""

from __future__ import annotations

from typing import Any

import structlog

from nvd_claude_proxy.config.models import ModelRegistry

_log = structlog.get_logger("nvd_claude_proxy.router")


def get_use_model(
    anthropic_body: dict,
    token_count: int,
    registry: ModelRegistry,
) -> str:
    """Decide which Claude model alias to use for this request."""
    router = registry.router
    requested_model = anthropic_body.get("model") or ""

    # 1. Long Context Scenario
    if token_count > router.long_context_threshold and router.long_context:
        _log.info(
            "router.scenario",
            scenario="long_context",
            tokens=token_count,
            threshold=router.long_context_threshold,
            model=router.long_context,
        )
        return router.long_context

    # 2. Vision Scenario
    has_vision = False
    for msg in anthropic_body.get("messages") or []:
        content = msg.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "image":
                    has_vision = True
                    break
        if has_vision:
            break
    
    if has_vision and router.vision:
        _log.info("router.scenario", scenario="vision", model=router.vision)
        return router.vision

    # 3. Web Search Scenario
    # Only match Anthropic server-tool types (dated pattern) or explicit web search
    # tool names. Substring "search" is intentionally NOT used — it matches
    # legitimate code-search tools like search_files, grep_search, codebase_search.
    has_web_search = False
    _WEB_SEARCH_TYPE_PREFIX = "web_search_"
    _WEB_SEARCH_NAMES = frozenset({"web_search", "brave_search", "tavily_search", "bing_search"})
    tools = anthropic_body.get("tools") or []
    for tool in tools:
        ttype = tool.get("type") or ""
        tname = (tool.get("name") or "").lower()
        if ttype.startswith(_WEB_SEARCH_TYPE_PREFIX) or tname in _WEB_SEARCH_NAMES:
            has_web_search = True
            break
    
    if has_web_search and router.web_search:
        _log.info("router.scenario", scenario="web_search", model=router.web_search)
        return router.web_search

    # 4. Thinking Scenario
    if anthropic_body.get("thinking") and router.think:
        _log.info("router.scenario", scenario="think", model=router.think)
        return router.think

    # 5. Background Scenario (Fast models)
    if ("haiku" in requested_model.lower()) and router.background:
        _log.info("router.scenario", scenario="background", model=router.background)
        return router.background

    # 6. Default
    return requested_model or router.default
