# Changelog

All notable changes are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [0.3.4] — 2026-04-20

### Fixed
- **Hallucinated tool blocking** — the proxy now detects and blocks known hallucinated tools (`Skill`, `Read`, `migrate`, `status`) that the model may fabricate from training data mismatch with the actual Claude Code tool registry. Previously these caused infinite loops of fake tool calls. Blocked tools emit a text warning instead of being forwarded.
- **Infinite tool-loop detection** — added repetition tracking: if the same tool pattern repeats 3+ times in the last 5 calls, the proxy forces stream termination with `stop_reason: refusal` and an error message, preventing token waste on stuck loops.

---

## [0.3.0] — 2026-04-20

### Fixed
- **Tool name restoration** — proxy now correctly returns the original tool name (before schema sanitisation) to Claude Code in both streaming and non-streaming paths. Fixes `Invalid tool parameters` errors for any tool whose name contained characters not in `[a-zA-Z0-9_-]`.
- `ToolIdMap.register_tool_rename` and `original_tool_name` were dead code; wired up in request and response translators.

---

## [0.2.9] — 2026-04-20

### Fixed
- **Context overflow pre-flight guard** — when estimated input tokens exceed the model's context window, the proxy now returns a clean Anthropic `invalid_request_error` 400 before the request reaches NVIDIA (previously NVIDIA returned a confusing `400 … None`).
- Raised `_CONTEXT_HEADROOM` from 8 192 → 16 384 tokens to absorb the ~9% cl100k undercount observed on large Context7 MCP payloads.
- Trailing ` None` stripped from NVIDIA 400 error message bodies.

---

## [0.2.8] — 2026-04-20

### Added
- **Sliding-window rate limiter** — replaces fixed-window; prevents double-quota burst at window boundaries.
- `/v1/models` now returns Anthropic-spec `type`, `display_name`, and stable `created_at` fields.
- `tool_result` image blocks are now converted to a text placeholder instead of being silently dropped.

### Fixed
- `stop_sequence` echo now scans full message text (previously only checked the tail of the last block).

---

## [0.2.7] — 2026-04-20

### Added
- Shared `NvidiaClient` (httpx connection pool) created once at startup via FastAPI lifespan — eliminates per-request TLS setup overhead (~50–150 ms per call).
- `message_start` now carries an estimated `input_tokens` count so SDK cost tracking works.
- Upstream NVIDIA `retry-after` header forwarded on 429 responses.
- `User-Agent` now reflects the actual package version from `_version.py`.

---

## [0.2.6] — 2026-04-19

### Added
- `ncp kill` command — terminates stuck proxy processes on the configured port.

### Fixed
- `anthropic-version: 2023-06-01` header added to all responses (required by TypeScript SDK).
- SSE `Content-Type` corrected to `text/event-stream; charset=utf-8`.
- API key pasting now works in all terminals (`hide_input=False`).
- Fast-path settings propagate `NVIDIA_API_KEY` to subprocess environment.

---

## [0.2.5] — 2026-04-18

### Added
- Model failover chains — automatic retry on upstream 5xx with configured fallback models.
- SIGHUP hot-reload of `models.yaml` without restart.
- Per-client rate limiter middleware (`RATE_LIMIT_RPM`).
- Request body size guard (`MAX_REQUEST_BODY_MB`).
- Stub endpoints for Batch and Files APIs returning 501.

---

## [0.2.0] — 2026-04-17

### Added
- Initial public release.
- `ncp` CLI with `code`, `proxy`, `init`, `models` commands.
- Full Anthropic Messages API → NVIDIA NIM translation.
- Streaming SSE with keepalive ping events.
- Tool use, reasoning/thinking, vision, PDF extraction.
- Prometheus metrics endpoint.
