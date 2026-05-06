from __future__ import annotations

from typing import Set
import structlog

_log = structlog.get_logger("nvd_claude_proxy.beta")

# Official Anthropic betas that nvd-claude-proxy has implemented or explicitly supports.
SUPPORTED_BETAS = {
    "prompt-caching-2024-07-31",
    "computer-use-2024-10-22",
    "pdfs-2024-09-25",
    "token-counting-2024-11-01",
    "extended-cache-ttl-2025-04-11",
    "web-search-2025-03-05",
    "code-execution-2025-05-22",
    "mcp-client-2025-04-04",
}

# Features that REQUIRE a specific beta header to be present.
# Maps a top-level request body key to the beta name.
BODY_KEY_TO_BETA = {
    "mcp_servers": "mcp-client-2025-04-04",
}

# Features inside content blocks that require a beta.
# Maps a block 'type' to the beta name.
BLOCK_TYPE_TO_BETA = {
    "web_search_tool_result": "web-search-2025-03-05",
    "code_execution_tool_result": "code-execution-2025-05-22",
}

class BetaNegotiator:
    """Handles negotiation and validation of 'anthropic-beta' feature flags."""

    def __init__(self, presented_betas: Set[str]) -> None:
        self.presented = presented_betas

    def validate_request(self, body: dict) -> None:
        """Raise ValueError if the request requires a beta that was not presented."""
        # 1. Check top-level keys
        for key, beta in BODY_KEY_TO_BETA.items():
            if key in body and beta not in self.presented:
                raise ValueError(f"Feature '{key}' requires the '{beta}' beta header.")

        # 2. Check content blocks
        for msg in body.get("messages", []):
            content = msg.get("content")
            if isinstance(content, list):
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    btype = block.get("type")
                    if btype in BLOCK_TYPE_TO_BETA:
                        beta = BLOCK_TYPE_TO_BETA[btype]
                        if beta not in self.presented:
                            raise ValueError(f"Content block type '{btype}' requires the '{beta}' beta header.")

        # 3. Check cache_control ttl
        def walk_cache_control(obj):
            if isinstance(obj, dict):
                cc = obj.get("cache_control")
                if isinstance(cc, dict) and cc.get("ttl") == "1h":
                    if "extended-cache-ttl-2025-04-11" not in self.presented:
                        raise ValueError("cache_control.ttl='1h' requires 'extended-cache-ttl-2025-04-11' beta.")
                for v in obj.values():
                    walk_cache_control(v)
            elif isinstance(obj, list):
                for item in obj:
                    walk_cache_control(item)

        walk_cache_control(body)

    def is_supported(self, beta: str) -> bool:
        return beta in SUPPORTED_BETAS

    def get_unsupported(self) -> Set[str]:
        return self.presented - SUPPORTED_BETAS
