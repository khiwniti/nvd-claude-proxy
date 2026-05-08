# Changelog

All notable changes are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [1.2.0] — 2026-05-08

### Honest scoping

NVIDIA NIM containers now expose a **native Anthropic-compatible `/v1/messages` endpoint** ([official guide](https://docs.nvidia.com/nim/large-language-models/latest/ai-assistant-integrations/claude-code.html)). For self-hosted NIM users, this proxy is unnecessary — point Claude Code at NIM directly. The README now leads with that path; the proxy is positioned as the residual translation shim for NVIDIA's hosted OpenAI-compatible endpoint (`integrate.api.nvidia.com`).

### Removed

- **Dashboard / live-monitor pubsub** — Deleted `routes/dashboard.py`, the `/dashboard` static mount, the `PubSub` WebSocket manager in `app.py`, and all in-stream `_fanout_pubsub` broadcasts (5 hot-path call sites). Removes ~250 LOC and one source of streaming overhead. Operators wanting observability should use the `/metrics` Prometheus endpoint and structured JSON logs.

### Fixed (carried over from the streaming-fidelity remediation)

- **TTFT** — `message_start` is now the first wire byte. Synthesised pre-flight from the request envelope per Anthropic spec, before the upstream NIM POST is initiated. Eliminates the dead-air period that left Claude Code's agent-state UI blank.
- **Partial `<think>` tag splitting** — New holdback scanner in `StreamTranslator` holds back at most 7 chars of trailing text that could form the prefix of `<think>` / `</think>`. Fragments like `"<th"` or `"</thi"` no longer leak as visible `text_delta`.
- **Singleton `message_delta`** — Removed the per-block-close cumulative `message_delta` emission in `core/processors.py`. Matches Anthropic spec (`message_delta` exactly once at end).
- **Inert transformer-chain fast path** — Cached the "no transformers" verdict in `StreamTranslator.__post_init__`, skipping per-event chain dispatch in the common case.
- **HTTP/2 + streaming-friendly timeouts** — `httpx[http2]` dependency, `read=None` for streaming reads, `UPSTREAM_HTTP2` env knob with graceful fallback.
- **`BetaNegotiator` type bug** — Fixed list/set mismatch that 500'd any request carrying an `anthropic-beta` header.

### Tests

- `tests/unit/test_think_tag_holdback.py` — 6 cases including 1-byte chunking, EOF flush, reasoning→content transition.
- `tests/unit/test_message_start_ttft.py` — first-frame structural assertion under hung upstream.

---

## [1.1.11] — 2026-05-06

### Fixed
- **Auth Robustness** — The proxy now automatically strips leading/trailing whitespace from the presented API key and the configured secret. This prevents common 401 errors caused by trailing newlines in Secret Manager values.

---

## [1.1.10] — 2026-05-06

### Added
- **Dynamic OpenAPI Generation (Phase 5)** — The `/v1/openapi.json` and `/v1/messages/schema` endpoints now dynamically generate their schemas from the underlying Pydantic v2 `CanonicalRequest` and `CanonicalResponse` models.
- **Proactive Stream Pings** — The streaming pipeline now features an independent background task that guarantees a `ping` event every 10 seconds, preventing SDK idle timeouts during extended reasoning phases.
- **Detailed Health Diagnostics** — The `/readyz` endpoint now distinguishes between upstream errors (`auth_failed`, `rate_limited`, `upstream_5xx`, `circuit_open`), greatly improving load-balancer visibility.

### Fixed
- **Mid-Stream Circuit Breaker** — Streaming connections that encounter transient mid-stream errors (`ReadTimeout`, `ProtocolError`, etc.) now correctly record failures to the global `CircuitBreaker`.
- **SSRF Protection** — Generalized SSRF protection middleware to walk all arbitrary payload nodes shaped as `{"type": "url", "url": ...}` to safely block dangerous internal URLs across any block type.
- **Billing Error Mapping** — Added explicit mapping for upstream `402` to Anthropic's `billing_error`.

---

## [1.1.9] — 2026-05-06

### Added
- **Canonical IR (Phase 3)** — Replaced soft dictionaries with frozen Pydantic v2 `CanonicalRequest` objects to enforce strict schema parity and discriminated unions at the translation boundary.
- **Server Tool Registry (Phase 4)** — Completely re-architected the server tool injector to be YAML data-driven (`config/server_tools.yaml`), supporting dynamic injection of schemas for Anthropic-specific tools (web_search, bash, computer, etc.).
- **Strict Version & Beta Negotiation** — Enforces `anthropic-version: 2023-06-01` and implements `BetaNegotiator` to validate requested beta features.
- **Token Reconciliation (P1-15)** — Compares upstream NVIDIA output tokens with local `tiktoken` re-tokenization of accumulated output to detect usage drift and automatically reconcile missed tokens.
- **TTL Cache Breakdown** — `cache_creation_input_tokens` is now accurately broken down into `ephemeral_5m` and `ephemeral_1h` buckets based on the `cache_control` TTL marker.

### Changed
- Flattened `anyOf`/`oneOf`/`allOf` JSON-Schemas to simple object schemas to guarantee compatibility with strict upstream vLLM parsers.
- Collision-safe SHA-256 truncation is now applied to generated tool names exceeding the 64-character limit.

---

## [1.1.8] — 2026-05-06

### Added
- **Idempotency Support (GAP-008)** — Full implementation of the `anthropic-idempotency-key` header to ensure safe request deduplication across Redis, SQLite, and In-Memory storage engines.
- **Vision Streaming Parity (GAP-009)** — The `StreamTranslator` and core `Pipeline` now perfectly translate upstream multimodal (`image_url`) chunks into standard Anthropic `image` content blocks during streaming.

### Fixed
- **Protocol Adjacency (GAP-004)** — Resolved a strict OpenAI compatibility bug where `tool_result` messages were incorrectly sequenced after user text in translated requests. Tool results now maintain the exact sequential ordering generated by the model in the preceding assistant turn.
- **Progressive Tool Streaming** — Restored fine-grained progressive tool streaming. The proxy now emits `input_json_delta` characters instantly as they arrive rather than buffering the entire tool payload, drastically improving responsiveness during agentic tool-use loops.
- **Tool Validation Unification** — Centralized all tool-schema validation and checks via `ToolInvocationController.is_declared()`.

---

## [1.1.7] — 2026-05-04

### Added
- **Final Official-Grade API Compliance** — Implemented the remaining missing features from the official Anthropic API specification. 
  - **Cache Accounting Simulation**: The proxy now correctly identifies `cache_control` markers and injects simulated `cache_creation_input_tokens` and `cache_read_input_tokens` into the `usage` reporting block, satisfying SDK clients that strictly validate cache accounting payloads.
  - **API Stubs for Unsupported Endpoints**: Added formal `501 Not Implemented` stubs for the Message Batches API and Files API. Instead of returning a cryptic `404 Not Found`, the proxy now returns a properly formatted Anthropic Error object explaining that the NVIDIA NIM backend does not support these features.
  - **OpenAPI Schema**: Added the formal `/v1/openapi.json` endpoint that perfectly matches the Anthropic API shape, allowing automatic SDK generation and client validation.

---

## [1.1.6] — 2026-05-04

### Fixed
- **Tool Parsing Error** — Fixed an issue where tool calls could not be parsed by the client. Some upstream models wrap JSON arguments in Markdown code blocks (e.g. ` ```json `). The new modular `ToolProcessor` now automatically detects and strips these Markdown fences before emitting the `input_json_delta` event, guaranteeing clean, valid JSON for Claude Code.

---

## [1.1.5] — 2026-05-04

### Changed
- **Hyper-Fast Agentic Flagship** — Re-assigned the primary `claude-opus-4-7` alias to `Qwen2.5-Coder-32B`. After extensive live-traffic testing, massive parameter models (70B+) were observed taking too long to respond. The new 32B parameter flagship is universally recognized as punching far above its weight class (rivaling GPT-4 in coding capabilities) while delivering near-instant latency and massively improved token throughput, creating a vastly superior user experience for autonomous agent loops.

---

## [1.1.4] — 2026-05-04

### Changed
- **Extreme Low-Latency Flagship** — Re-assigned the primary `claude-opus-4-7` alias to `MiniMax-2.7`. This model provides extreme speed and high token throughput under heavy API traffic while maintaining exceptional coding capabilities.
- **MoE Fallback** — Re-assigned the `claude-sonnet-4-6` alias to `DeepSeek-V4-Flash` as a blazing-fast, massive-scale MoE alternative for heavy workloads.

---

## [1.1.3] — 2026-05-04

### Changed
- **Flagship Model Reassignment** — Shifted the primary `claude-opus-4-7` alias from `GLM-5.1` to `DeepSeek-V4-Pro`. DeepSeek-V4-Pro's highly efficient Sparse MoE architecture maintains top-tier agentic reasoning capabilities and a 1M-token context window while providing significantly higher token throughput and lower latency under heavy API traffic.
- **Model Fleet Update** — Shifted the `claude-sonnet-4-6` alias to use `MiniMax-2.7` to maintain a balanced, high-speed alternative to the flagship model.

---

## [1.1.2] — 2026-05-04

### Fixed
- **Plugin and MCP Compatibility** — Addressed an issue where official Claude Code plugins or MCP servers using Anthropic's "Server Tools" beta formats (e.g., `bash`, `computer`, `text_editor`) were being incorrectly dropped during translation. The proxy now intercepts these specific beta types and injects their complete, implicit OpenAI-compatible JSON schemas on the fly, allowing NVIDIA NIM models to leverage them natively.
- **Claude Code Launcher** — Changed the default CLI mode from `--bare` to `--full-claude`. This ensures that Claude Code officially loads its user-configured plugins, hooks, and MCP discovery servers out of the box when run via `ncp code`.

---

## [1.1.0] — 2026-05-03

### Added
- **Official-Grade Architectural Overhaul** — Major refactor to decouple core translation logic from the transport layer. Core logic now lives in a framework-agnostic `core` package.
- **Modular Streaming Pipeline** — Replaced monolithic state machine with an event-driven "Chain of Responsibility" pipeline. This improves reliability for complex tool-calling and reasoning sequences.
- **Enterprise Resilience** — Added a sophisticated `CircuitBreaker` registry to protect upstream calls and prevent cascading failures.
- **Distributed State Management** — Introduced a new `StorageEngine` abstraction supporting **Redis** for horizontal scaling, with **SQLite** and **In-Memory** fallbacks for local dev.
- **Global Auth Enforcement** — Centralized security in a unified `AuthMiddleware`, ensuring every endpoint (including metrics and metadata) is protected by the `PROXY_API_KEY`.
- **Automatic Fallback Factory** — The storage layer now automatically selects the best available backend based on environment configuration, ensuring zero-config robustness.

### Fixed
- **Configuration Drift** — Synchronized default ports, versions, and documentation across the entire project. Default `PROXY_PORT` is now consistently `8788`.
- **Strict SDK Compliance** — Refactored the middleware order to ensure response headers and error shapes perfectly mirror the official Anthropic API.
- **Static Analysis** — Achieved 100% compliance with strict `mypy` typing and `ruff` linting across the entire codebase.

---

## [0.4.1] — 2026-04-21

### Added
- **Tokeniser fallback** — added `approximate_tokens_fast` fallback that uses a simple character-based heuristic when `tiktoken` initialization fails (e.g. in restricted environments or during network issues). Prevents proxy crashes on startup.
- **CI publish gating** — updated GitHub Actions to safely handle PyPI publishing when Trusted Publishers are not yet configured, falling back to API tokens if present.

### Fixed
- **Undeclared tool blocking** — improved safety by strictly blocking only tools that were NOT provided in the original Anthropic `tools` list. Legitimate tools with names like `Skill` or `Read` are no longer blocked if they are correctly declared in the session.
- **Stream robustness** — hardened the SSE translator to better handle partial JSON fragments and malformed reasoning blocks from upstream models.
- **Lint & Type safety** — comprehensive fix of `mypy` and `ruff` issues across core modules, CLI, and test suite.

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
