from __future__ import annotations

import json
import re
import secrets
from typing import Any, Protocol, runtime_checkable, Callable

import structlog
from json_repair import repair_json

_log = structlog.get_logger("nvd_claude_proxy.transformers")


@runtime_checkable
class Transformer(Protocol):
    on_fix: Callable[[str, Any], None] | None = None

    def transform_request(self, payload: dict[str, Any]) -> dict[str, Any]:
        return payload

    def transform_response(self, response: dict[str, Any]) -> dict[str, Any]:
        return response

    def transform_stream_chunk(self, chunk: dict[str, Any]) -> dict[str, Any] | None:
        return chunk

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.__class__.__name__}


class TransformerChain:
    def __init__(
        self, transformers: list[Transformer], on_fix: Callable[[str, Any], None] | None = None
    ) -> None:
        self.transformers = transformers
        self.on_fix = on_fix
        for t in self.transformers:
            t.on_fix = on_fix

    def transform_request(self, payload: dict[str, Any]) -> dict[str, Any]:
        for transformer in self.transformers:
            payload = transformer.transform_request(payload)
        return payload

    def transform_response(self, response: dict[str, Any]) -> dict[str, Any]:
        for transformer in self.transformers:
            response = transformer.transform_response(response)
        return response

    def transform_stream_chunk(self, chunk: dict[str, Any]) -> dict[str, Any] | None:
        current_chunk: dict[str, Any] | None = chunk
        for transformer in self.transformers:
            if current_chunk is None:
                return None
            current_chunk = transformer.transform_stream_chunk(current_chunk)
        return current_chunk

    def to_dict(self) -> list[dict[str, Any]]:
        return [t.to_dict() for t in self.transformers]

    @classmethod
    def from_dict(
        cls, data: list[dict[str, Any]], on_fix: Callable[[str, Any], None] | None = None
    ) -> TransformerChain:
        transformers: list[Transformer] = []
        for item in data:
            t_type = item.get("type")
            if t_type == "CharFixerTransformer":
                transformers.append(CharFixerTransformer(on_fix=on_fix))
            elif t_type == "JSONRepairTransformer":
                transformers.append(JSONRepairTransformer(on_fix=on_fix))
            elif t_type == "WebSearchTransformer":
                transformers.append(WebSearchTransformer())
            elif t_type == "ReasoningTransformer":
                transformers.append(ReasoningTransformer())
            elif t_type == "ExitToolTransformer":
                # Note: ExitToolTransformer might have state, but for now we re-init
                transformers.append(ExitToolTransformer())
        return cls(transformers, on_fix=on_fix)


class CharFixerTransformer:
    """Strips control characters from JSON strings that can cause parsing issues."""

    def __init__(self, on_fix: Callable[[str, Any], None] | None = None) -> None:
        self.on_fix = on_fix

    def _fix_value(self, val: Any) -> Any:
        if isinstance(val, str):
            # Strip ASCII control characters except \n, \r, \t
            fixed = "".join(ch for ch in val if ord(ch) >= 32 or ch in "\n\r\t")
            if fixed != val and self.on_fix:
                self.on_fix("char_fix", {"before": val, "after": fixed})
            return fixed
        if isinstance(val, list):
            return [self._fix_value(v) for v in val]
        if isinstance(val, dict):
            return {k: self._fix_value(v) for k, v in val.items()}
        return val

    def transform_request(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._fix_value(payload)

    def transform_response(self, response: dict[str, Any]) -> dict[str, Any]:
        return self._fix_value(response)

    def transform_stream_chunk(self, chunk: dict[str, Any]) -> dict[str, Any] | None:
        return self._fix_value(chunk)

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.__class__.__name__}


class JSONRepairTransformer:
    """Uses json_repair.repair_json to fix truncated JSON blocks."""

    def __init__(self, on_fix: Callable[[str, Any], None] | None = None) -> None:
        self.on_fix = on_fix

    def transform_request(self, payload: dict[str, Any]) -> dict[str, Any]:
        return payload

    def transform_response(self, response: dict[str, Any]) -> dict[str, Any]:
        choices = response.get("choices") or []
        for choice in choices:
            msg = choice.get("message") or {}
            tool_calls = msg.get("tool_calls") or []
            for tc in tool_calls:
                fn = tc.get("function") or {}
                args = fn.get("arguments")
                if args and isinstance(args, str):
                    repaired = repair_json(args)
                    if repaired != args and self.on_fix:
                        self.on_fix("json_repair", {"before": args, "after": repaired})
                    fn["arguments"] = repaired
        return response

    def transform_stream_chunk(self, chunk: dict[str, Any]) -> dict[str, Any] | None:
        # Do NOT repair streaming fragments — they are intentionally partial.
        # The StreamTranslator accumulates complete args before emitting.
        return chunk

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.__class__.__name__}


class WebSearchTransformer:
    """Detects annotations in OpenAI responses and maps to Anthropic web_search blocks."""

    def __init__(self, on_fix: Callable[[str, Any], None] | None = None) -> None:
        self.on_fix = on_fix

    def transform_request(self, payload: dict[str, Any]) -> dict[str, Any]:
        return payload

    def transform_response(self, response: dict[str, Any]) -> dict[str, Any]:
        choices = response.get("choices") or []
        for choice in choices:
            msg = choice.get("message") or {}
            if annotations := msg.get("annotations"):
                content = msg.get("content") or []
                if isinstance(content, str):
                    content = [{"type": "text", "text": content}]

                for ann in annotations:
                    if ann.get("type") == "web_search":
                        # Map to Anthropic web_search_tool_result
                        content.append(
                            {
                                "type": "web_search_tool_result",
                                "tool_use_id": f"srvtoolu_{secrets.token_urlsafe(12)}",
                                "content": [
                                    {
                                        "type": "web_search_result",
                                        "title": ann.get("title", ""),
                                        "url": ann.get("url", ""),
                                    }
                                ],
                            }
                        )
                msg["content"] = content
        return response

    def transform_stream_chunk(self, chunk: dict[str, Any]) -> dict[str, Any] | None:
        # Note: Streaming search results is difficult because one OpenAI chunk
        # with multiple annotations would need to explode into multiple
        # Anthropic events. For now, we pass through.
        return chunk

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.__class__.__name__}


class ReasoningTransformer:
    """Captures <think> tags and maps reasoning_content to Anthropic thinking blocks."""

    _THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)

    def __init__(self, on_fix: Callable[[str, Any], None] | None = None) -> None:
        self.on_fix = on_fix

    def transform_request(self, payload: dict[str, Any]) -> dict[str, Any]:
        return payload

    def transform_response(self, response: dict[str, Any]) -> dict[str, Any]:
        choices = response.get("choices") or []
        for choice in choices:
            msg = choice.get("message") or {}
            content = msg.get("content")
            if content and isinstance(content, str):
                m = self._THINK_RE.search(content)
                if m:
                    reasoning = m.group(1).strip()
                    if not msg.get("reasoning_content"):
                        msg["reasoning_content"] = reasoning
                    msg["content"] = self._THINK_RE.sub("", content, count=1).lstrip()
        return response

    def transform_stream_chunk(self, chunk: dict[str, Any]) -> dict[str, Any] | None:
        # Handling <think> tags in streaming is complex because they can span chunks.
        # This is usually handled in the StreamTranslator state machine.
        return chunk

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.__class__.__name__}


class ExitToolTransformer:
    """Ported from exit_tool.py. Injects ExitTool and filters it from responses."""

    EXIT_TOOL_NAME = "ExitTool"
    EXIT_TOOL_SCHEMA: dict[str, Any] = {
        "type": "function",
        "function": {
            "name": EXIT_TOOL_NAME,
            "description": (
                "Use this tool when you are in tool mode and have completed the task. "
                "This is the only valid way to exit tool mode.\n"
                "IMPORTANT: Before using this tool, ensure that none of the available "
                "tools are applicable to the current task."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "response": {
                        "type": "string",
                        "description": "Your response will be forwarded to the user exactly as returned.",
                    },
                },
                "required": ["response"],
            },
        },
    }

    _SYSTEM_REMINDER = (
        "<system-reminder>Tool mode is active. The user expects you to proactively "
        "execute the most suitable tool to help complete the task.\n"
        "Before invoking a tool, you must carefully evaluate whether it matches the "
        "current task. If no available tool is appropriate for the task, you MUST call "
        "the `ExitTool` to exit tool mode.\n"
        "Always prioritize completing the user's task effectively and efficiently by "
        "using tools whenever appropriate.</system-reminder>"
    )

    def __init__(self, on_fix: Callable[[str, Any], None] | None = None) -> None:
        self.on_fix = on_fix
        self._exit_tool_index: int = -1
        self._exit_tool_response: str = ""
        self._exit_tool_detected: bool = False

    def transform_request(self, payload: dict[str, Any]) -> dict[str, Any]:
        tools = payload.get("tools")
        if not tools:
            return payload

        messages = payload.get("messages", [])
        modified_tools = list(tools)
        modified_tools.append(self.EXIT_TOOL_SCHEMA)

        modified_messages = list(messages)
        # Check if there is a system message to append to, or add a new one
        system_found = False
        for msg in modified_messages:
            if msg.get("role") == "system":
                msg["content"] = (msg.get("content") or "") + "\n\n" + self._SYSTEM_REMINDER
                system_found = True
                break

        if not system_found:
            modified_messages.insert(0, {"role": "system", "content": self._SYSTEM_REMINDER})

        payload["tools"] = modified_tools
        payload["messages"] = modified_messages
        return payload

    def transform_response(self, response: dict[str, Any]) -> dict[str, Any]:
        choices = response.get("choices") or [{}]
        choice = choices[0]
        msg = choice.get("message") or {}
        tool_calls = msg.get("tool_calls") or []

        if not tool_calls:
            return response

        first_tc = tool_calls[0]
        fn = first_tc.get("function") or {}
        if fn.get("name") != self.EXIT_TOOL_NAME:
            return response

        try:
            args = json.loads(fn.get("arguments", "{}"))
            text = args.get("response", "")
        except (json.JSONDecodeError, ValueError):
            text = fn.get("arguments", "")

        msg["content"] = text
        if "tool_calls" in msg:
            del msg["tool_calls"]

        _log.debug("exit_tool.stripped_from_response")
        return response

    def transform_stream_chunk(self, chunk: dict[str, Any]) -> dict[str, Any] | None:
        choices = chunk.get("choices") or []
        if not choices:
            return chunk

        choice = choices[0]
        delta = choice.get("delta") or {}
        tool_calls = delta.get("tool_calls") or []

        for tc in tool_calls:
            fn = tc.get("function") or {}

            if fn.get("name") == self.EXIT_TOOL_NAME:
                self._exit_tool_index = tc.get("index", 0)
                self._exit_tool_detected = True
                return None

            if self._exit_tool_index >= 0 and tc.get("index") == self._exit_tool_index:
                args_fragment = fn.get("arguments", "")
                if args_fragment:
                    self._exit_tool_response += args_fragment
                    try:
                        parsed = json.loads(self._exit_tool_response)
                        text = parsed.get("response", "")
                        converted = dict(chunk)
                        converted["choices"] = [
                            {
                                **choice,
                                "delta": {"role": "assistant", "content": text},
                            }
                        ]
                        return converted
                    except (json.JSONDecodeError, ValueError):
                        pass
                return None

        finish = choice.get("finish_reason")
        if finish == "tool_calls" and self._exit_tool_detected:
            modified = dict(chunk)
            modified["choices"] = [{**choice, "finish_reason": "stop", "delta": delta}]
            return modified

        return chunk

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.__class__.__name__}
