# Architecture

**Analysis Date:** 2026-04-21

## Pattern Overview

**Overall:** Layered FastAPI proxy translating Anthropic Messages API semantics to NVIDIA NIM OpenAI-compatible endpoints.

**Assumptions:**
- Primary codebase target is `nvd-claude-proxy/` under workspace root.
- Runtime deployment is single-process `uvicorn` with async request handling.

**Key Characteristics:**
- API compatibility boundary exposed via FastAPI routes in `src/nvd_claude_proxy/routes/`.
- Protocol translation core isolated in `src/nvd_claude_proxy/translators/`.
- Upstream communication encapsulated in `src/nvd_claude_proxy/clients/nvidia_client.py`.
- Cross-cutting middleware for logging/rate/body controls in `src/nvd_claude_proxy/middleware/`.
- Shared app state (`settings`, `model_registry`, `nvidia_client`) initialized in `src/nvd_claude_proxy/app.py`.

## Subsystems and Responsibilities

### Bootstrap and Lifecycle
- `src/nvd_claude_proxy/main.py`: process entrypoint; resolves runtime options and launches app factory via Uvicorn.
- `src/nvd_claude_proxy/app.py`: FastAPI composition root; loads settings/model registry, installs SIGHUP model reload handler, wires middleware and routers, and manages `NvidiaClient` lifecycle in app lifespan.

### Request/Response API Boundary
- `src/nvd_claude_proxy/routes/messages.py`: main Anthropic-compatible `/v1/messages` endpoint handling auth check, request translation, failover routing, streaming/non-streaming response mapping, and error conversion.
- `src/nvd_claude_proxy/routes/count_tokens.py`: `/v1/messages/count_tokens` estimator boundary.
- `src/nvd_claude_proxy/routes/models.py`: model alias discovery and lookup endpoints.
- `src/nvd_claude_proxy/routes/health.py`, `src/nvd_claude_proxy/routes/metrics_route.py`, `src/nvd_claude_proxy/routes/stubs.py`: operational and compatibility support endpoints.

### Translation Engine
- `src/nvd_claude_proxy/translators/request_translator.py`: Anthropic request blocks -> OpenAI/NIM payload, context window guard, tool/vision/thinking conversion, and tool filtering.
- `src/nvd_claude_proxy/translators/response_translator.py`: non-stream OpenAI response -> Anthropic schema blocks.
- `src/nvd_claude_proxy/translators/stream_translator.py`: streaming state machine enforcing Anthropic SSE event ordering and block semantics.
- `src/nvd_claude_proxy/translators/tool_translator.py`, `tool_controller.py`, `schema_sanitizer.py`: tool schema conversion, id mapping, validation and sanitization.
- `src/nvd_claude_proxy/translators/thinking_translator.py`, `vision_translator.py`: modality-specific transformations.

### Upstream Client Layer
- `src/nvd_claude_proxy/clients/nvidia_client.py`: resilient async HTTP wrapper around `/chat/completions` and `/models` with retries/backoff and SSE parsing.

### Configuration and Capability Registry
- `src/nvd_claude_proxy/config/settings.py`: env-driven runtime config model and cached settings accessor.
- `src/nvd_claude_proxy/config/models.py`: `models.yaml` loader with alias/prefix resolution and failover chain construction.
- `src/nvd_claude_proxy/data/models.yaml`: bundled model capability manifest source.

### Middleware and Operational Utilities
- `src/nvd_claude_proxy/middleware/logging.py`: request logging, request-id propagation.
- `src/nvd_claude_proxy/middleware/rate_limiter.py`: optional per-client request throttling.
- `src/nvd_claude_proxy/middleware/body_limit.py`: optional body-size guardrail.
- `src/nvd_claude_proxy/util/`: headers, metrics, token estimation, SSE encoding, cost estimation, pdf extraction, ids.

### CLI Orchestration
- `src/nvd_claude_proxy/cli/main.py`: operational CLI (`ncp`) for init/run/status/model inspection and Claude Code launcher orchestration.

## Data and Control Flow

### Core Messages Flow (non-stream)
1. Client calls `POST /v1/messages` in `src/nvd_claude_proxy/routes/messages.py`.
2. Route validates proxy key (if configured), resolves model/failover via `app.state.model_registry`.
3. `translate_request()` in `src/nvd_claude_proxy/translators/request_translator.py` produces NVIDIA payload and enforces context guard.
4. `NvidiaClient.chat_completions()` in `src/nvd_claude_proxy/clients/nvidia_client.py` sends upstream request with retry/backoff policy.
5. Errors map through `src/nvd_claude_proxy/errors/mapper.py`; successful payload maps through `translate_response()`.
6. Route records metrics/cost and returns Anthropic-shaped JSON response.

### Core Messages Flow (stream)
1. Same pre-processing as non-stream path in `src/nvd_claude_proxy/routes/messages.py`.
2. Route drives async generator around `NvidiaClient.astream_chat_completions()`.
3. `StreamTranslator.feed()` and `finalize()` in `src/nvd_claude_proxy/translators/stream_translator.py` convert OpenAI chunk stream into ordered Anthropic SSE events.
4. Route emits heartbeat `ping` events and fallback behavior for early transient upstream failures.
5. Final `message_delta` and `message_stop` close stream; metrics/logs updated.

### Configuration and Startup Flow
1. `run()` in `src/nvd_claude_proxy/main.py` calls `get_settings()` from `src/nvd_claude_proxy/config/settings.py`.
2. `create_app()` in `src/nvd_claude_proxy/app.py` loads settings and model registry from `src/nvd_claude_proxy/config/models.py`.
3. App lifespan creates shared `NvidiaClient`; middleware and routers attach; optional SIGHUP reload handler installed.

### CLI Flow
1. User invokes `ncp` in `src/nvd_claude_proxy/cli/main.py`.
2. CLI reads settings and model registry, can launch proxy subprocess via `python -m nvd_claude_proxy.main`.
3. CLI optionally bootstraps/launches Claude Code process with environment wiring (`ANTHROPIC_BASE_URL`, model vars).

## Architectural Boundaries

### External Interfaces
- Anthropic-compatible HTTP API boundary: `src/nvd_claude_proxy/routes/`.
- NVIDIA NIM upstream boundary: `src/nvd_claude_proxy/clients/nvidia_client.py`.
- Operator boundary via CLI: `src/nvd_claude_proxy/cli/main.py`.

### Internal Boundaries
- Routes orchestrate request lifecycle but delegate all schema/protocol conversion to `translators/`.
- Translators are pure transformation/state-machine units and do not own transport.
- Client layer owns HTTP retry/stream mechanics and no Anthropic semantics.
- Config layer owns env/model resolution and no routing logic.

### State Ownership
- Application state in `FastAPI.app.state`:
  - `settings`: from `src/nvd_claude_proxy/config/settings.py`
  - `model_registry`: from `src/nvd_claude_proxy/config/models.py`
  - `nvidia_client`: from `src/nvd_claude_proxy/clients/nvidia_client.py`

## Error Handling Strategy

- Upstream/OpenAI-style errors are normalized to Anthropic-like error payloads through `src/nvd_claude_proxy/errors/mapper.py`.
- Context overflow is preemptively intercepted by `ContextOverflowError` in `src/nvd_claude_proxy/translators/request_translator.py`, returning client-facing 400 before upstream call.
- Streaming path emits `error` SSE events for mid-stream failures in `src/nvd_claude_proxy/routes/messages.py`.
- Middleware-level unhandled exceptions are logged in `src/nvd_claude_proxy/middleware/logging.py`.

## Cross-Cutting Concerns

- **Observability:** structured logs (`structlog`) and optional Prometheus metrics (`src/nvd_claude_proxy/util/metrics.py`, endpoint in `src/nvd_claude_proxy/routes/metrics_route.py`).
- **Security:** optional proxy API key enforcement at route boundary (`src/nvd_claude_proxy/routes/messages.py`); optional rate limiting/body-size controls in middleware.
- **Compatibility guarantees:** strict Anthropic event ordering and model alias behavior centralized in translators and model registry.

---

*Architecture analysis: 2026-04-21*
