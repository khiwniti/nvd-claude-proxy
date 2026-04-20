from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml

ReasoningStyle = Literal[
    "detailed-thinking-v1", "slash-think", "qwen-kwargs", "always-on", "none"
]


@dataclass(slots=True)
class ModelSpec:
    alias: str
    nvidia_id: str
    supports_tools: bool = True
    supports_vision: bool = False
    supports_reasoning: bool = False
    reasoning_style: ReasoningStyle = "none"
    max_context: int = 131072
    max_output: int = 16384
    temperature_override: float | None = None
    # Ordered list of alias names to try if this model returns 5xx.
    # Example: failover_to: ["claude-sonnet-4-6", "claude-haiku-4-5"]
    failover_to: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ModelRegistry:
    specs: dict[str, ModelSpec] = field(default_factory=dict)
    prefix_fallbacks: dict[str, str] = field(default_factory=dict)
    default_big: str = "claude-opus-4-7"
    default_small: str = "claude-haiku-4-5"

    def resolve_chain(self, claude_model_name: str | None) -> list[ModelSpec]:
        """Return the primary spec followed by any failover specs, deduped."""
        primary = self.resolve(claude_model_name)
        chain: list[ModelSpec] = [primary]
        seen = {primary.alias}
        for alias in primary.failover_to:
            if alias in self.specs and alias not in seen:
                chain.append(self.specs[alias])
                seen.add(alias)
        return chain

    def resolve(self, claude_model_name: str | None) -> ModelSpec:
        """Resolve a Claude-style model name to a configured ModelSpec.

        Order:
          1. exact alias match
          2. longest-prefix fallback
          3. default big model
        """
        name = (claude_model_name or "").strip()
        if name and name in self.specs:
            return self.specs[name]
        # Longest prefix wins.
        best: tuple[int, str] | None = None
        for prefix, alias in self.prefix_fallbacks.items():
            if name.startswith(prefix) and (best is None or len(prefix) > best[0]):
                best = (len(prefix), alias)
        if best and best[1] in self.specs:
            return self.specs[best[1]]
        if self.default_big in self.specs:
            return self.specs[self.default_big]
        # Last resort — return any spec so the proxy stays usable.
        return next(iter(self.specs.values()))


def _bundled_models_path() -> Path:
    """Return the path to the models.yaml bundled inside the package."""
    try:
        from importlib.resources import files  # Python 3.9+
        return Path(str(files("nvd_claude_proxy.data").joinpath("models.yaml")))
    except Exception:
        return Path(__file__).parent.parent / "data" / "models.yaml"


def load_model_registry(path: str | Path | None = None) -> ModelRegistry:
    """Load the model registry from *path*, falling back to the bundled default."""
    resolved = Path(path) if path else None
    if resolved is None or not resolved.exists():
        bundled = _bundled_models_path()
        if resolved is not None and not resolved.exists():
            import warnings
            warnings.warn(
                f"models.yaml not found at '{resolved}'; using bundled default. "
                "Set MODEL_CONFIG_PATH to override.",
                stacklevel=2,
            )
        resolved = bundled
    data = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    specs = {
        alias: ModelSpec(alias=alias, **spec)
        for alias, spec in (data.get("aliases") or {}).items()
    }
    return ModelRegistry(
        specs=specs,
        prefix_fallbacks=data.get("prefix_fallbacks") or {},
        default_big=(data.get("defaults") or {}).get("big", "claude-opus-4-7"),
        default_small=(data.get("defaults") or {}).get("small", "claude-haiku-4-5"),
    )
