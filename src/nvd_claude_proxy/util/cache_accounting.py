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
    
    @property
    def total_input_tokens(self) -> int:
        """Total input tokens including cached."""
        return self.cache_creation_input_tokens + self.cache_read_input_tokens + self.non_cached_input_tokens
    
    def to_dict(self) -> dict[str, int | float]:
        """Convert to Anthropic usage dict format."""
        return {
            "input_tokens": self.total_input_tokens,
            "cache_creation_input_tokens": self.cache_creation_input_tokens,
            "cache_read_input_tokens": self.cache_read_input_tokens,
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
    
    This is an approximation since:
    1. We don't have access to the actual cache hit/miss status
    2. NVIDIA NIM doesn't expose cache token breakdowns
    3. Token estimation is inherently imprecise
    
    We use a heuristic: if cache_control markers exist, assume:
    - First occurrence of a block = cache creation
    - Subsequent references to same content = cache read
    
    Args:
        body: The request body dict (Anthropic format)
        
    Returns:
        CacheAccounting with estimated token breakdowns
    """
    # Collect all blocks that have cache_control
    cached_blocks: list[tuple[str, dict]] = []  # (path, block)
    all_text_tokens = 0
    
    def walk(obj: Any, path: str = "") -> None:
        if isinstance(obj, dict):
            # Check for cache_control
            if obj.get("cache_control"):
                cached_blocks.append((path, obj))
            
            # Collect text content for token estimation
            if obj.get("type") == "text":
                text = obj.get("text", "")
                all_text_tokens += len(text.split())  # Rough word count
            
            # Recurse into children
            for k, v in obj.items():
                if k not in ("cache_control", "type"):
                    walk(v, f"{path}.{k}" if path else k)
                    
        elif isinstance(obj, list):
            for idx, item in enumerate(obj):
                walk(item, f"{path}[{idx}]")
    
    walk(body)
    
    if not cached_blocks:
        # No cache markers - all tokens are non-cached
        total = approximate_tokens(body)
        return CacheAccounting(
            non_cached_input_tokens=total,
            estimated_cache_savings_usd=0.0,
        )
    
    # Estimate tokens for cached vs non-cached content
    # This is simplified - real implementation would track actual token counts
    total_tokens = approximate_tokens(body)
    cached_token_estimate = 0
    non_cached_token_estimate = total_tokens
    
    for path, block in cached_blocks:
        # Get text from text blocks or content from document blocks
        text = ""
        if block.get("type") == "text":
            text = block.get("text", "")
        elif block.get("type") == "document":
            source = block.get("source", {})
            text = source.get("data", "") or ""
        elif block.get("type") == "image":
            # Images are expensive to cache
            source = block.get("source", {})
            data = source.get("data", "")
            # Estimate ~1000 tokens per image
            cached_token_estimate += 1000
            continue
        
        if text:
            # Estimate tokens in this block
            block_tokens = max(1, len(text) // 4)  # Rough: 4 chars per token
            cached_token_estimate += block_tokens
    
    # Recalculate non-cached as remainder
    non_cached_token_estimate = max(0, total_tokens - cached_token_estimate)
    
    # Calculate cost savings
    # Non-cached price for cached tokens vs cached price
    non_cached_cost = (cached_token_estimate / 1_000_000) * _BASE_PRICE_PER_M
    cached_cost = (
        (cached_token_estimate * 0.1 / 1_000_000) * _CACHE_READ_PRICE_PER_M  # 90% discount
    )
    savings = max(0, non_cached_cost - cached_cost)
    
    return CacheAccounting(
        cache_creation_input_tokens=cached_token_estimate // 10,  # ~10% as creation
        cache_read_input_tokens=cached_token_estimate - cached_token_estimate // 10,
        non_cached_input_tokens=non_cached_token_estimate,
        estimated_cache_savings_usd=round(savings, 6),
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
        (accounting.total_input_tokens / 1_000_000) * input_price_per_m +
        output_cost
    )
    
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