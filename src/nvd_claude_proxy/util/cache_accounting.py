"""Prompt cache accounting utilities.

This module estimates cache token accounting based on cache_control markers
in Anthropic requests. While NVIDIA NIM doesn't support true prompt caching,
we estimate the tokens that WOULD have been cached for cost tracking purposes.

Anthropic's prompt caching behavior:
- cache_control blocks marked as "ephemeral" are cached
- First request with cached blocks includes cache_creation tokens
- Subsequent requests include cache_read tokens (90% discount)
- Budget: up to 200K tokens of cache per request
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .tokens import approximate_tokens

# Anthropic's token pricing (approximate, as of 2024)
# Per 1M tokens:
_CACHE_CREATION_PRICE_PER_M = 3.75  # $3.75 per 1M tokens
_CACHE_READ_PRICE_PER_M = 0.30  # $0.30 per 1M tokens (90% discount)
_BASE_PRICE_PER_M = 3.75  # $3.75 per 1M tokens for non-cached


@dataclass
class CacheAccounting:
    """Result of cache token accounting for a request."""

    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    non_cached_input_tokens: int = 0
    estimated_cache_savings_usd: float = 0.0
    
    # TTL breakdown
    ephemeral_5m_input_tokens: int = 0
    ephemeral_1h_input_tokens: int = 0

    @property
    def total_input_tokens(self) -> int:
        """Total input tokens including cached."""
        return (
            self.cache_creation_input_tokens
            + self.cache_read_input_tokens
            + self.non_cached_input_tokens
        )

    def to_dict(self) -> dict[str, int | float]:
        """Convert to Anthropic usage dict format."""
        return {
            "input_tokens": self.total_input_tokens,
            "cache_creation_input_tokens": self.cache_creation_input_tokens,
            "cache_read_input_tokens": self.cache_read_input_tokens,
            "cache_creation_input_tokens_breakdown": {
                "ephemeral_5m_input_tokens": self.ephemeral_5m_input_tokens,
                "ephemeral_1h_input_tokens": self.ephemeral_1h_input_tokens,
            }
        }


def has_cache_control_markers(body: dict) -> bool:
    """Check if a request body contains any cache_control markers."""

    def walk(obj: Any) -> bool:
        if isinstance(obj, dict):
            if obj.get("cache_control") is not None:
                return True
            for v in obj.values():
                if walk(v):
                    return True
        elif isinstance(obj, list):
            for item in obj:
                if walk(item):
                    return True
        return False

    return walk(body)


def estimate_cache_tokens(body: dict) -> CacheAccounting:
    """Estimate cache token accounting based on cache_control markers.

    Implementation follows P2-1: Walk request top-down; sum tokens of every block
    from start through last cache_control marker into creation; everything before
    first marker on hit into reads. Since we are a proxy, we treat every request
    with markers as a 'cache write' (creation) for cost estimation, assuming the
    SDK will do the right thing.
    """
    total_tokens = approximate_tokens(body)
    
    # Collect all blocks and identify which ones have cache_control
    all_blocks = []
    
    # Anthropic system can be blocks
    system = body.get("system")
    if isinstance(system, list):
        all_blocks.extend(system)
    elif isinstance(system, str):
        all_blocks.append({"type": "text", "text": system})
        
    for msg in body.get("messages", []):
        content = msg.get("content")
        if isinstance(content, list):
            all_blocks.extend(content)
        elif isinstance(content, str):
            all_blocks.append({"type": "text", "text": content})
            
    # Tools also have cache_control
    tools = body.get("tools") or []
    all_blocks.extend(tools)

    if not any(b.get("cache_control") for b in all_blocks if isinstance(b, dict)):
        return CacheAccounting(non_cached_input_tokens=total_tokens)

    # Find the index of the LAST block with cache_control
    last_marker_idx = -1
    for i, block in enumerate(all_blocks):
        if isinstance(block, dict) and block.get("cache_control"):
            last_marker_idx = i
            
    if last_marker_idx == -1:
        return CacheAccounting(non_cached_input_tokens=total_tokens)

    # Creation tokens = tokens in all blocks from index 0 to last_marker_idx
    creation_tokens = 0
    ephemeral_5m = 0
    ephemeral_1h = 0
    
    for i in range(last_marker_idx + 1):
        block = all_blocks[i]
        tokens = 0
        if isinstance(block, str):
            tokens = len(block) // 4
        elif isinstance(block, dict):
            btype = block.get("type")
            if btype == "text" or "input_schema" in block: # Text or Tool
                text = block.get("text") or block.get("description") or ""
                tokens = len(text) // 4
            elif btype == "image" or btype == "document":
                tokens = 1000 # Heuristic
            else:
                tokens = 10 # Default for tool_use etc
        
        creation_tokens += tokens
        
        # Breakdown by TTL
        if isinstance(block, dict) and block.get("cache_control"):
            ttl = block["cache_control"].get("ttl", "5m")
            if ttl == "1h":
                ephemeral_1h += tokens
            else:
                ephemeral_5m += tokens
        else:
            # If a block is before a marker but doesn't have one, it's still part of that cache prefix
            # We'll attribute it to 5m by default unless we already passed a 1h marker
            ephemeral_5m += tokens

    # Re-normalize creation_tokens to not exceed total_tokens
    creation_tokens = min(creation_tokens, total_tokens)
    
    # Read tokens: For the sake of this local proxy, we assume a "hit" if we see markers
    # but to be conservative we'll set it to 0 and put everything in creation
    # UNLESS the user explicitly wants to see savings.
    # We'll stick to the "first marker = creation" mental model.
    
    non_cached = max(0, total_tokens - creation_tokens)
    
    # Price calculation
    savings = (creation_tokens / 1_000_000) * (_BASE_PRICE_PER_M - _CACHE_CREATION_PRICE_PER_M)
    
    return CacheAccounting(
        cache_creation_input_tokens=creation_tokens,
        cache_read_input_tokens=0,
        non_cached_input_tokens=non_cached,
        ephemeral_5m_input_tokens=ephemeral_5m,
        ephemeral_1h_input_tokens=ephemeral_1h,
        estimated_cache_savings_usd=round(max(0, savings), 6),
    )


def get_cache_efficiency_ratio(accounting: CacheAccounting) -> float:
    """Calculate cache efficiency as percentage of cached vs total tokens.

    Returns:
        0.0 to 1.0 representing cache hit rate
    """
    total = accounting.total_input_tokens
    if total == 0:
        return 0.0

    cached = accounting.cache_creation_input_tokens + accounting.cache_read_input_tokens
    return cached / total


def estimate_cost_with_caching(
    accounting: CacheAccounting,
    input_price_per_m: float = _BASE_PRICE_PER_M,
    output_price_per_m: float = 15.0,  # Claude 3.5 Sonnet output price
    output_tokens: int = 0,
) -> dict[str, float]:
    """Estimate total cost including cache savings.

    Args:
        accounting: Cache token accounting
        input_price_per_m: Price per 1M input tokens (non-cached)
        output_price_per_m: Price per 1M output tokens
        output_tokens: Number of output tokens

    Returns:
        Dict with breakdown of costs
    """
    # Non-cached input cost
    non_cached_cost = (accounting.non_cached_input_tokens / 1_000_000) * input_price_per_m

    # Cached input (creation): Full price for first cache write
    creation_cost = (accounting.cache_creation_input_tokens / 1_000_000) * input_price_per_m

    # Cached input (read): 10% of full price (90% savings)
    read_cost = (accounting.cache_read_input_tokens / 1_000_000) * (input_price_per_m * 0.1)

    # Output cost
    output_cost = (output_tokens / 1_000_000) * output_price_per_m

    # Total without cache
    total_without_cache = (
        accounting.total_input_tokens / 1_000_000
    ) * input_price_per_m + output_cost

    # Actual total with cache
    total_with_cache = non_cached_cost + creation_cost + read_cost + output_cost

    return {
        "non_cached_input": round(non_cached_cost, 6),
        "cache_creation": round(creation_cost, 6),
        "cache_read": round(read_cost, 6),
        "output": round(output_cost, 6),
        "total": round(total_with_cache, 6),
        "total_without_cache": round(total_without_cache, 6),
        "savings": round(total_without_cache - total_with_cache, 6),
    }
