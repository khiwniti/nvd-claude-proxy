"""Robust JSON repair for tool-call arguments.

Ported from claude-code-router's ``parseToolArguments()`` (JSON5 + jsonrepair).

Strategy order (cheapest first):
1. ``json.loads`` -- fast-path for well-formed JSON.
2. Strip markdown fences + leading prose -> ``json.loads``.
3. Balanced-brace extraction -> ``json.loads``.
4. Truncated-JSON repair: add missing closing braces/brackets.
5. Fallback: return ``{}`` so downstream doesn't crash.
"""

from __future__ import annotations

import json
import re

import structlog

_log = structlog.get_logger("nvd_claude_proxy.tool_args_parser")

_FENCE_RE = re.compile(r"^```[a-z]*\s*", re.MULTILINE)
_JSON_START_RE = re.compile(r"[{\[]")


def _try_json(s: str) -> dict | list | None:
    try:
        result = json.loads(s)
        if isinstance(result, (dict, list)):
            return result
        return {"value": result}
    except (json.JSONDecodeError, ValueError):
        return None


def _strip_fences(s: str) -> str:
    if s.startswith("```"):
        s = _FENCE_RE.sub("", s, count=1)
        if s.rstrip().endswith("```"):
            s = s.rstrip()[:-3].rstrip()
    return s.strip()


def _strip_leading_prose(s: str) -> str:
    m = _JSON_START_RE.search(s)
    if m:
        return s[m.start():]
    return s


def _balanced_extract(s: str) -> str | None:
    for start_idx in range(len(s)):
        ch = s[start_idx]
        if ch not in ("{", "["):
            continue
        close_ch = "}" if ch == "{" else "]"
        depth, in_str, esc = 0, False, False
        for end_idx in range(start_idx, len(s)):
            c = s[end_idx]
            if esc:
                esc = False
                continue
            if c == "\\" and in_str:
                esc = True
                continue
            if c == '"':
                in_str = not in_str
                continue
            if not in_str:
                if c == ch:
                    depth += 1
                elif c == close_ch:
                    depth -= 1
                    if depth == 0:
                        return s[start_idx : end_idx + 1]
        break
    return None


def _repair_truncated(s: str) -> str | None:
    stripped = _strip_fences(s)
    stripped = _strip_leading_prose(stripped)
    if not stripped:
        return None

    stack: list[str] = []
    in_str, esc = False, False
    for c in stripped:
        if esc:
            esc = False
            continue
        if c == "\\" and in_str:
            esc = True
            continue
        if c == '"':
            in_str = not in_str
            continue
        if not in_str:
            if c in ("{", "["):
                stack.append(c)
            elif c == "}" and stack and stack[-1] == "{":
                stack.pop()
            elif c == "]" and stack and stack[-1] == "[":
                stack.pop()

    if not stack:
        return None

    closers = {"[": "]", "{": "}"}
    repaired = stripped
    last_meaningful = repaired.rstrip()
    if last_meaningful and last_meaningful[-1] not in ('"', "}", "]", ",", ":", "{", "["):
        repaired = repaired.rstrip() + '"'

    for opener in reversed(stack):
        repaired += closers[opener]

    return repaired


def parse_tool_arguments(raw: str) -> str:
    """Parse and repair tool-call arguments string.

    Returns a valid JSON string. Never raises -- falls back to ``{}`` on
    catastrophic failure.
    """
    if not raw or raw.strip() == "" or raw.strip() == "{}":
        return "{}"

    s = raw.strip()

    # 1) Fast path: standard JSON.
    result = _try_json(s)
    if result is not None:
        return s

    # 2) Strip fences + prose -> JSON.
    cleaned = _strip_fences(s)
    cleaned = _strip_leading_prose(cleaned)
    result = _try_json(cleaned)
    if result is not None:
        _log.debug("tool_args.fence_stripped_parse_ok")
        return cleaned

    # 3) Balanced-brace extraction.
    balanced = _balanced_extract(s)
    if balanced:
        result = _try_json(balanced)
        if result is not None:
            _log.debug("tool_args.balanced_extract_ok")
            return balanced

    # 4) Truncated-JSON repair.
    repaired = _repair_truncated(s)
    if repaired:
        result = _try_json(repaired)
        if result is not None:
            _log.debug("tool_args.truncated_repair_ok")
            return repaired

    # 5) Fallback.
    _log.warning("tool_args.all_strategies_failed", raw_length=len(raw))
    return "{}"
