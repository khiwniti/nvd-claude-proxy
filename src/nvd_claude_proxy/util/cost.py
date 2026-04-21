"""Very rough cost estimation for structured logs.

NVIDIA NIM usage isn't billed per token the same way Anthropic is, but
knowing the equivalent Anthropic cost helps with budget comparisons and
anomaly detection in logs.

Prices are *approximate* Anthropic list prices in USD as of early 2026.
Adjust via the ``COST_INPUT_PER_MTK`` / ``COST_OUTPUT_PER_MTK`` env vars
or simply ignore the ``cost_usd_est`` log field.

MTK = 1 million tokens.
"""

from __future__ import annotations

# Per-model overrides (alias → (input_per_mtk, output_per_mtk)).
# If a model isn't listed we fall back to `_DEFAULT_COST`.
_MODEL_COSTS: dict[str, tuple[float, float]] = {
    "claude-opus-4-7": (15.00, 75.00),  # Anthropic Opus 4 list price proxy
    "claude-sonnet-4-6": (3.00, 15.00),  # Sonnet 4 list price proxy
    "claude-haiku-4-5": (0.80, 4.00),  # Haiku 4 list price proxy
}
_DEFAULT_COST: tuple[float, float] = (3.00, 15.00)  # Sonnet-tier default


def estimate_cost_usd(
    model_alias: str,
    input_tokens: int,
    output_tokens: int,
) -> float:
    """Return estimated cost in USD (sum of input + output)."""
    inp_mtk, out_mtk = _MODEL_COSTS.get(model_alias, _DEFAULT_COST)
    return (input_tokens * inp_mtk + output_tokens * out_mtk) / 1_000_000
