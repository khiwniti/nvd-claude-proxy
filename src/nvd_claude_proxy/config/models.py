from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml  # type: ignore[import-untyped]

ReasoningStyle = Literal["detailed-thinking-v1", "slash-think", "qwen-kwargs", "always-on", "none"]


@dataclass(slots=True)
class ReasoningConfig:
    style: ReasoningStyle = "none"
    adaptive: bool = False
    effort_mapping: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class ToolConfig:
    supports: bool = True
    parallel: bool = True
    arg_validation: bool = True


@dataclass(slots=True)
class CapabilityManifest:
    alias: str
    nvidia_id: str
    supports_tools: bool = True
    supports_vision: bool = False
    reasoning: ReasoningConfig = field(default_factory=ReasoningConfig)
    tools: ToolConfig = field(default_factory=ToolConfig)
    max_context: int = 1000000
    max_output: int = 16384
    temperature_override: float | None = None
    failover_to: list[str] = field(default_factory=list)
    # Legacy support
    supports_reasoning: bool = False
    reasoning_style: ReasoningStyle = "none"

    def __post_init__(self):
        if self.reasoning_style != "none":
            self.reasoning.style = self.reasoning_style


@dataclass(slots=True)
class ModelRegistry:
    specs: dict[str, CapabilityManifest] = field(default_factory=dict)
    prefix_fallbacks: dict[str, str] = field(default_factory=dict)
    default_big: str = "claude-opus-4-7"
    default_small: str = "claude-haiku-4-5"

    def resolve_chain(self, claude_model_name: str | None) -> list[CapabilityManifest]:
        """Return the primary spec followed by any failover specs, deduped."""
        primary = self.resolve(claude_model_name)
        chain: list[CapabilityManifest] = [primary]
        seen = {primary.alias}
        for alias in primary.failover_to:
            if alias in self.specs and alias not in seen:
                chain.append(self.specs[alias])
                seen.add(alias)
        return chain

    def resolve(self, claude_model_name: str | None) -> CapabilityManifest:
        """Resolve a Claude-style model name to a configured CapabilityManifest."""
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
        from importlib.resources import files

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
    specs = {}
    for alias, spec_data in (data.get("aliases") or {}).items():
        specs[alias] = CapabilityManifest(
            alias=alias,
            nvidia_id=spec_data["nvidia_id"],
            supports_vision=spec_data.get("supports_vision", False),
            reasoning=ReasoningConfig(
                style=spec_data.get("reasoning_style", "none"),
            ),
            tools=ToolConfig(
                supports=spec_data.get("supports_tools", True),
            ),
            max_context=spec_data.get("max_context", 1000000),
            max_output=spec_data.get("max_output", 16384),
            temperature_override=spec_data.get("temperature_override"),
            failover_to=spec_data.get("failover_to") or [],
            reasoning_style=spec_data.get("reasoning_style", "none"),
            supports_reasoning=spec_data.get("supports_reasoning", False),
        )

    return ModelRegistry(
        specs=specs,
        prefix_fallbacks=data.get("prefix_fallbacks") or {},
        default_big=(data.get("defaults") or {}).get("big", "claude-opus-4-7"),
        default_small=(data.get("defaults") or {}).get("small", "claude-haiku-4-5"),
    )
