from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any


@dataclass
class DegradationContext:
    dropped_fields: list[str] = field(default_factory=list)
    unsupported_blocks: list[str] = field(default_factory=list)
    unsupported_betas: list[str] = field(default_factory=list)
    disabled_parallel_tools: bool = False
    downgraded_thinking: bool = False
    routing_fallback: str | None = None
    schema_repair_attempts: int = 0
    blocked_tool_calls: list[str] = field(default_factory=list)
    provider_capability_mismatch: list[str] = field(default_factory=list)
    approximated_token_usage: bool = False

    def add_dropped_field(self, field: str) -> None:
        self.dropped_fields.append(field)

    def add_unsupported_block(self, block_type: str) -> None:
        self.unsupported_blocks.append(block_type)

    def add_unsupported_beta(self, beta: str) -> None:
        self.unsupported_betas.append(beta)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dropped_fields": self.dropped_fields,
            "unsupported_blocks": self.unsupported_blocks,
            "unsupported_betas": self.unsupported_betas,
            "disabled_parallel_tools": self.disabled_parallel_tools,
            "downgraded_thinking": self.downgraded_thinking,
            "routing_fallback": self.routing_fallback,
            "schema_repair_attempts": self.schema_repair_attempts,
            "blocked_tool_calls": self.blocked_tool_calls,
            "provider_capability_mismatch": self.provider_capability_mismatch,
            "approximated_token_usage": self.approximated_token_usage,
        }

    def has_degradation(self) -> bool:
        return any(
            [
                self.dropped_fields,
                self.unsupported_blocks,
                self.unsupported_betas,
                self.disabled_parallel_tools,
                self.downgraded_thinking,
                self.routing_fallback is not None,
                self.schema_repair_attempts > 0,
                self.blocked_tool_calls,
                self.provider_capability_mismatch,
                self.approximated_token_usage,
            ]
        )
