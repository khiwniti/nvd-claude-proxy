# Remediation Plan: nvidia-claude-proxy Claude Code / Anthropic Parity Hardening

## Correction

The root repository for this plan is `nvidia-claude-proxy`, present in this workspace as:

- `/Users/khiwn/custom-claude-code/nvd-claude-proxy`

The earlier broader audit referenced sibling projects (`free-claude-code` and `claude-code-router`) for comparative context, but the implementation plan below is scoped to the root repo: `nvd-claude-proxy`.

## Goal

Harden `nvd-claude-proxy` into a production-grade Anthropic Messages API / Claude Code compatibility layer with:

- Strict Anthropic request/response validation.
- Correct SSE event ordering and streaming lifecycle semantics.
- Safe, deterministic, parallel-capable tool-use handling.
- Fail-closed declared-tool allowlisting and JSON Schema validation.
- Capability-aware routing across configured NVIDIA/OpenAI-compatible models.
- Anthropic version/beta negotiation.
- Zero-trust input handling and sanitized error/log behavior.
- Explicit graceful degradation for unsupported provider features.

## Current Context / Assumptions

Primary codebase:

- `/Users/khiwn/custom-claude-code/nvd-claude-proxy`

Key existing architecture:

- FastAPI app and routes under `src/nvd_claude_proxy/routes/`.
- Anthropic/OpenAI schema models under `src/nvd_claude_proxy/schemas/`.
- Translation pipeline under `src/nvd_claude_proxy/translators/`.
- NVIDIA client under `src/nvd_claude_proxy/clients/nvidia_client.py`.
- Model registry/routing under `src/nvd_claude_proxy/config/models.py` and `src/nvd_claude_proxy/util/router.py`.
- Security/rate/body middleware under `src/nvd_claude_proxy/middleware/`.

Highest-risk gaps from audit:

1. Interleaved parallel streamed tool calls can produce invalid Anthropic SSE ordering.
2. `tool_result` translation can violate OpenAI-compatible adjacency ordering.
3. Non-streaming undeclared/invalid tool calls are logged but not reliably blocked.
4. Claude Code internal-looking tools can bypass declared request tool allowlists in streaming paths.
5. Anthropic content block support is incomplete and some blocks are dropped silently.
6. Request validation is not strict enough and validated models are not consistently used downstream.
7. `anthropic-version` / `anthropic-beta` behavior is not negotiated into translator capabilities.
8. Routing is scenario-based but not capability-safe.
9. JSON Schema validation for tool args is optional/non-enforcing.
10. Mid-stream error/cancellation and token usage fallback semantics need explicit lifecycle policy.

## Proposed Approach

Implement in six phases:

1. Add conformance tests first.
2. Build strict Anthropic protocol boundaries.
3. Fix tool-use lifecycle and parallel invocation.
4. Harden SSE streaming correctness.
5. Add capability-aware routing and Anthropic version/beta negotiation.
6. Add zero-trust security, degradation reporting, and production observability.

Do not start by patching isolated symptoms. First create tests that reproduce the protocol gaps, then fix the translator and routing architecture against those tests.

## Phase 0 — Conformance Harness

### Objective

Capture expected Claude Code / Anthropic Messages API behavior before implementation.

### Tasks

1. Add fixtures under:

- `tests/fixtures/anthropic/`

2. Add golden request/response fixtures for:

- Non-streaming text response.
- Streaming text response.
- Streaming single `tool_use`.
- Streaming two interleaved parallel `tool_use` calls.
- User message with multiple `tool_result` blocks.
- `tool_result` with `is_error: true`.
- Thinking blocks.
- Redacted thinking blocks.
- Prompt cache/control blocks.
- Vision/image content.
- Web search/server-tool content.
- Invalid request bodies.
- Invalid tool schemas.
- Unknown model IDs.
- Provider 401/429/500 translation.
- Mid-stream provider disconnect.
- Split UTF-8 and split SSE chunk boundaries.

3. Add protocol validator helpers:

- Validate Anthropic request schema.
- Validate Anthropic non-streaming response schema.
- Validate Anthropic SSE event ordering.
- Validate tool lifecycle invariants.

### Likely Files to Add / Change

- `tests/test_anthropic_protocol_conformance.py`
- `tests/test_streaming_conformance.py`
- `tests/test_tool_lifecycle_conformance.py`
- `tests/fixtures/anthropic/*.json`

### Validation

```sh
cd /Users/khiwn/custom-claude-code/nvd-claude-proxy
make test
```

Expected first result: tests should expose current failures before fixes are implemented.

## Phase 1 — Strict Anthropic Protocol Boundary

### Objective

Ensure every request and response crosses a typed Anthropic protocol boundary before translation or return.

### Tasks

1. Expand `schemas/anthropic.py` with strict Pydantic v2 models for:

- `MessagesRequest`
- `MessagesResponse`
- `Message`
- `ContentBlock` discriminated union
- `TextBlock`
- `ImageBlock`
- `ToolUseBlock`
- `ToolResultBlock` including `is_error`
- `ThinkingBlock`
- `RedactedThinkingBlock`
- `ServerToolUseBlock`
- `WebSearchToolResultBlock`
- search/citation-like blocks where supported
- `Tool`
- strict `ToolChoice` union
- `Usage`
- `AnthropicError`

2. Add strict request-level fields:

- `model`
- `messages`
- `system`
- `max_tokens`
- `metadata`
- `stop_sequences`
- `stream`
- `temperature`
- `top_k`
- `top_p`
- `tools`
- `tool_choice`
- `thinking`
- `service_tier`
- `container`
- `context_management`
- `mcp_servers`

3. Set strict model config for public wire models:

- Use `ConfigDict(extra="forbid")` for strict mode.
- If compatibility mode is needed, preserve unknown beta/extensions explicitly rather than silently ignoring them.

4. Update validation flow:

- `routes/messages.py` must use the validated/canonical request object downstream.
- Do not validate and then continue with raw request body.
- Add Anthropic-shaped `invalid_request_error` for validation failures.

5. Add response validation:

- Validate non-streaming `MessagesResponse` before returning.
- Validate event object shapes in streaming tests.

### Likely Files to Change

- `src/nvd_claude_proxy/schemas/anthropic.py`
- `src/nvd_claude_proxy/schemas/validators.py`
- `src/nvd_claude_proxy/routes/messages.py`
- `src/nvd_claude_proxy/errors/mapper.py`

### Validation

```sh
cd /Users/khiwn/custom-claude-code/nvd-claude-proxy
make test
```

Add specific tests proving:

- invalid fields fail in strict mode;
- unsupported fields are not silently dropped;
- invalid `tool_choice` fails;
- invalid `tool.input_schema` fails;
- system role inside `messages` is rejected unless explicit compatibility mode exists.

## Phase 2 — Tool-Use Lifecycle Hardening

### Objective

Make tool-use deterministic, safe, and parallel-capable.

### Tasks

1. Implement a `ToolCallAssembler` / tool state machine.

Track each tool call by:

- provider stream index;
- provider tool call id;
- Anthropic `tool_use.id`;
- content block index.

States:

- `pending`
- `started`
- `receiving_name`
- `receiving_input_json`
- `complete`
- `invalid`
- `closed`

2. Enforce invariants:

- No duplicate `tool_use.id`.
- No `content_block_delta` after `content_block_stop`.
- No executable undeclared tool names.
- Every `tool_result.tool_use_id` references a previous assistant `tool_use`.
- Parallel tool results must be accounted for before the next assistant continuation.
- Tool args must validate against the declared `input_schema` before executable `tool_use` emission.

3. Fix streamed parallel tools.

Current risk:

- Interleaved OpenAI/NIM tool chunks can cause a tool block to be closed and later receive more deltas, violating Anthropic SSE ordering.

Required behavior:

- Buffer interleaved tool-call fragments per index.
- Emit each Anthropic tool block contiguously.
- Never emit deltas after block stop.
- If complete JSON cannot be assembled, emit a safe error/degradation, not executable tool use.

4. Fix `tool_result` translation.

Current risk:

- User text can be emitted before role=`tool` messages, violating OpenAI-compatible adjacency requirements.

Required behavior:

- Translate `tool_result` blocks to role=`tool` messages immediately after the assistant tool calls they answer.
- Emit residual user text only after all tool results.
- Preserve `tool_result.is_error`.

5. Enforce declared-tool allowlist.

Required behavior:

- Remove unconditional Claude Code internal tool bypass.
- `Read`, `Write`, `Edit`, `Bash`, `Task`, `Skill`, `WebSearch`, etc. are allowed only if present in the request’s `tools` list.
- Fuzzy matching is disabled by default and opt-in only.

6. Make JSON Schema validation fail-closed.

- `jsonschema` should be required in Claude Code strict mode.
- Invalid schemas fail request validation.
- Invalid model-produced args are blocked or converted to safe text/error, not executable `tool_use`.

### Likely Files to Change

- `src/nvd_claude_proxy/translators/stream_translator.py`
- `src/nvd_claude_proxy/translators/request_translator.py`
- `src/nvd_claude_proxy/translators/response_translator.py`
- `src/nvd_claude_proxy/translators/tool_translator.py`
- `src/nvd_claude_proxy/translators/tool_controller.py`
- `src/nvd_claude_proxy/translators/tool_fuzzy_mapper.py`
- `src/nvd_claude_proxy/util/tool_args_parser.py`
- `src/nvd_claude_proxy/translators/schema_sanitizer.py`

### Validation

Add tests for:

- two interleaved streamed tool calls;
- args arriving before name/id;
- invalid JSON args;
- undeclared tool name;
- internal-looking tool name not declared;
- fuzzy match disabled;
- multiple tool_result blocks;
- unknown/stale tool_result id;
- `tool_result.is_error` preservation.

Run:

```sh
cd /Users/khiwn/custom-claude-code/nvd-claude-proxy
make test
```

## Phase 3 — SSE Streaming Correctness

### Objective

Make streaming behavior conform to Anthropic SSE semantics and robust against malformed provider streams.

### Tasks

1. Centralize SSE parsing/serialization.

Parser must support:

- split UTF-8 boundaries;
- multi-line `data:` fields;
- `event:` fields;
- comments and ping events;
- CRLF/LF;
- `id:`;
- `retry:`;
- arbitrary provider chunk fragmentation;
- provider `[DONE]` where applicable.

2. Add an `AnthropicStreamEventBuilder`.

Enforce event order:

- `message_start`
- `content_block_start`
- `content_block_delta` zero or more times
- `content_block_stop`
- `message_delta`
- `message_stop`

3. Add terminal lifecycle policy:

- normal completion;
- provider error before `message_start`;
- provider error after `message_start`;
- timeout;
- client disconnect;
- cancellation;
- malformed upstream stream;
- partial/invalid tool-call stream.

4. Fix usage accounting.

- Preserve upstream usage when provided.
- If upstream omits usage, estimate only as an explicit fallback.
- Avoid returning output_tokens=0 for completed streams when text was emitted and provider usage is missing.
- Track cache creation/read token estimates where applicable.

5. Add ping support for long-running streams.

### Likely Files to Change

- `src/nvd_claude_proxy/translators/stream_translator.py`
- `src/nvd_claude_proxy/util/sse.py`
- `src/nvd_claude_proxy/routes/messages.py`
- `src/nvd_claude_proxy/clients/nvidia_client.py`
- `src/nvd_claude_proxy/util/tokens.py`

### Validation

- Golden stream fixtures pass.
- Fuzz split SSE chunks and UTF-8 boundaries.
- No invalid event ordering under parallel tool calls.
- Mid-stream provider error terminates according to documented policy.
- Client disconnect/cancellation is recorded distinctly in metrics/logs.

## Phase 4 — Routing, Capabilities, and Version/Beta Negotiation

### Objective

Ensure requests route only to capable models and that Anthropic header semantics drive behavior.

### Tasks

1. Add provider/model capability descriptors.

Model capability fields:

- target provider model;
- context window;
- max output tokens;
- supports tools;
- supports parallel tools;
- supports vision;
- supports documents;
- supports extended thinking;
- supports signed thinking;
- supports prompt caching;
- supports server tools/web search;
- supports MCP;
- supports strict JSON schema;
- supports streaming usage;
- tokenizer/counting strategy.

2. Add request requirement extraction.

Detect whether request requires:

- tools;
- parallel tools;
- vision;
- documents;
- thinking;
- signed thinking;
- prompt caching;
- server tools/web search;
- MCP;
- long context;
- large output;
- strict schema.

3. Add post-routing capability validation.

- If selected model cannot satisfy requirements, choose a capable fallback only if policy allows.
- Otherwise return Anthropic-shaped error.
- Never silently route to an incapable model.

4. Add Anthropic version/beta negotiation.

- Parse `anthropic-version`.
- Parse `anthropic-beta` into feature flags.
- Route beta-dependent behavior into translators.
- Reject or explicitly degrade unsupported beta features.
- Expose supported versions/betas in health/debug metadata.

5. Improve count_tokens.

- Share canonical request processing with `/v1/messages` in dry-run mode.
- Account for tools, system, image/document estimates, prompt cache hints, and provider-specific tokenizer where possible.

### Likely Files to Change

- `src/nvd_claude_proxy/config/models.py`
- `src/nvd_claude_proxy/util/router.py`
- `src/nvd_claude_proxy/util/anthropic_headers.py`
- `src/nvd_claude_proxy/routes/messages.py`
- `src/nvd_claude_proxy/routes/models.py`
- `src/nvd_claude_proxy/routes/health.py`
- `src/nvd_claude_proxy/routes/count_tokens.py`
- `src/nvd_claude_proxy/util/tokens.py`

### Validation

- Unknown model fails deterministically unless compatibility fallback is enabled.
- Vision requests route only to vision-capable models.
- Tool requests route only to tool-capable models.
- Parallel tool requests route only to parallel-capable models or explicitly degrade.
- Server-tool/web-search requests do not silently drop semantics.
- `/v1/models` reflects configured registry.
- `/v1/count_tokens` matches canonicalization behavior of `/v1/messages`.

## Phase 5 — Zero-Trust Security Hardening

### Objective

Make the proxy safe for production exposure.

### Tasks

1. Redact sensitive logs.

Redact:

- `Authorization`;
- `x-api-key`;
- provider credentials;
- cookies;
- raw request bodies in production defaults;
- likely secret-bearing tool input fields.

2. Sanitize client-visible errors.

- No stack traces.
- Include request id.
- Map to Anthropic error object.

3. Harden SSRF/URL policy.

- Validate only fields that are dereferenced or forwarded as provider-fetchable URLs.
- Use robust `ipaddress` checks.
- Handle IPv6, private/link-local/reserved ranges, encoded IP forms, and DNS rebinding concerns where applicable.

4. Add endpoint-specific body limits.

5. Ensure rate limiting is scoped by provider/key/model policy.

6. Ensure failover does not leak partial stream state after first byte has been sent.

### Likely Files to Change

- `src/nvd_claude_proxy/middleware/security.py`
- `src/nvd_claude_proxy/middleware/logging.py`
- `src/nvd_claude_proxy/middleware/body_limit.py`
- `src/nvd_claude_proxy/middleware/rate_limiter.py`
- `src/nvd_claude_proxy/middleware/load_shedding.py`
- `src/nvd_claude_proxy/errors/mapper.py`
- `src/nvd_claude_proxy/util/circuit_breaker.py`

### Validation

- Secrets do not appear in logs during tests.
- Stack traces do not appear in production error responses.
- SSRF tests cover private IPv4/IPv6, localhost variants, encoded IP forms, and harmless text containing URLs.
- Rate-limit/load-shedding tests show scoped, predictable behavior.

## Phase 6 — Graceful Degradation and Observability

### Objective

Make unsupported features explicit and measurable.

### Tasks

1. Add `DegradationContext`.

Track:

- dropped fields;
- unsupported content blocks;
- unsupported beta flags;
- disabled parallel tools;
- downgraded thinking;
- approximated token usage;
- routing fallback;
- schema repair attempts;
- blocked tool calls;
- provider capability mismatch.

2. Expose degradation via:

- structured logs;
- optional debug response headers;
- diagnostics endpoint;
- tests.

3. Add/extend metrics for:

- request count by provider/model;
- streaming terminal state;
- tool-call count;
- blocked tool-call count;
- invalid tool args;
- degraded features;
- provider errors by class;
- retry/failover count;
- token usage and estimate markers.

### Likely Files to Change

- `src/nvd_claude_proxy/util/degradation.py` or new equivalent
- `src/nvd_claude_proxy/util/metrics.py`
- `src/nvd_claude_proxy/util/metrics_enhanced.py`
- `src/nvd_claude_proxy/routes/metrics_route.py`
- `src/nvd_claude_proxy/routes/dashboard.py`
- all translators/routes that collect degradation data

### Validation

- Tests assert degradation entries for unsupported beta/content blocks.
- Unsupported blocks are never silently ignored.
- Debug logs show routing and degradation decisions.

## Consolidated Files Likely to Change

- `src/nvd_claude_proxy/schemas/anthropic.py`
- `src/nvd_claude_proxy/schemas/validators.py`
- `src/nvd_claude_proxy/routes/messages.py`
- `src/nvd_claude_proxy/routes/models.py`
- `src/nvd_claude_proxy/routes/health.py`
- `src/nvd_claude_proxy/routes/count_tokens.py`
- `src/nvd_claude_proxy/routes/metrics_route.py`
- `src/nvd_claude_proxy/translators/request_translator.py`
- `src/nvd_claude_proxy/translators/response_translator.py`
- `src/nvd_claude_proxy/translators/stream_translator.py`
- `src/nvd_claude_proxy/translators/tool_translator.py`
- `src/nvd_claude_proxy/translators/tool_controller.py`
- `src/nvd_claude_proxy/translators/tool_fuzzy_mapper.py`
- `src/nvd_claude_proxy/translators/schema_sanitizer.py`
- `src/nvd_claude_proxy/util/sse.py`
- `src/nvd_claude_proxy/util/router.py`
- `src/nvd_claude_proxy/util/anthropic_headers.py`
- `src/nvd_claude_proxy/util/tokens.py`
- `src/nvd_claude_proxy/util/tool_args_parser.py`
- `src/nvd_claude_proxy/util/metrics.py`
- `src/nvd_claude_proxy/util/metrics_enhanced.py`
- `src/nvd_claude_proxy/config/models.py`
- `src/nvd_claude_proxy/config/settings.py`
- `src/nvd_claude_proxy/clients/nvidia_client.py`
- `src/nvd_claude_proxy/middleware/security.py`
- `src/nvd_claude_proxy/middleware/logging.py`
- `src/nvd_claude_proxy/middleware/body_limit.py`
- `src/nvd_claude_proxy/middleware/rate_limiter.py`
- `src/nvd_claude_proxy/middleware/load_shedding.py`
- `src/nvd_claude_proxy/errors/mapper.py`
- `tests/**`

## Verification Commands

```sh
cd /Users/khiwn/custom-claude-code/nvd-claude-proxy
make fmt
make lint
make test
```

If direct commands are needed:

```sh
cd /Users/khiwn/custom-claude-code/nvd-claude-proxy
python -m pytest -q
ruff check .
mypy src tests
```

Use the repository’s configured command set as authoritative if `Makefile` and `pyproject.toml` differ.

## Risks and Tradeoffs

1. Strict validation may break existing permissive proxy behavior.

Mitigation:

- Add explicit compatibility mode.
- Default Claude Code mode to strict.
- Emit structured degradation/warnings.

2. Some Anthropic features cannot be faithfully represented by NVIDIA/OpenAI-compatible models.

Mitigation:

- Capability-aware routing.
- Explicit degradation or fail-fast errors.
- Do not fake signed thinking, server tools, MCP, or prompt cache support.

3. Parallel tool streaming is complex.

Mitigation:

- Disable upstream parallel tool calls for unsupported providers.
- Implement buffered contiguous Anthropic emission for providers that stream interleaved calls.
- Require golden tests before enabling.

4. Token accounting will not always be exact.

Mitigation:

- Preserve upstream truth when present.
- Mark estimates internally.
- Use shared canonicalization for count_tokens and messages.

5. Over-broad SSRF blocking can reject harmless prompts.

Mitigation:

- Validate URL-bearing fields rather than arbitrary text when possible.
- Keep policy configurable and observable.

## Open Questions

1. Should strict Claude Code compatibility mode become the default for all deployments?

2. Which Anthropic beta features are required for the first parity milestone?

3. Should unsupported `server_tool_use`/web-search blocks fail fast or degrade to text placeholders?

4. Should fuzzy tool-name mapping be removed entirely or retained behind an explicit config flag?

5. What provider/model registry format should be treated as stable public configuration?

6. Should signed thinking be represented only when provider-native support exists, or should non-signed reasoning be exposed as ordinary text?

## Recommended First Implementation Slice

Start with P0/P1 focused on tool safety and protocol validation:

1. Add failing tests for:

- parallel streamed tool calls;
- tool_result ordering;
- undeclared internal tools;
- invalid tool args;
- strict request validation.

2. Fix:

- `stream_translator.py` parallel tool event ordering;
- `request_translator.py` tool_result adjacency;
- declared-tool allowlist enforcement;
- strict/fail-closed tool schema validation;
- use validated request body in `routes/messages.py`.

3. Verify:

```sh
cd /Users/khiwn/custom-claude-code/nvd-claude-proxy
make fmt
make lint
make test
```

This first slice removes the most dangerous Claude Code execution correctness risks before broader routing/security/observability work.
