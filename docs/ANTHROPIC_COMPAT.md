# Anthropic Messages API — Compatibility Gap Analysis

Audit of `nvd-claude-proxy` against the public Anthropic Messages API as
consumed by the official Python/TypeScript SDKs and by Claude Code.

Legend:
- ✅ Done · ⚠️ Partial · ❌ Missing · ➖ N/A (no NVIDIA equivalent)

---

## 1. Endpoints

| Method | Path                                    | Status | Notes |
|--------|-----------------------------------------|--------|-------|
| POST   | `/v1/messages`                          | ✅     | Streaming + non-stream. |
| POST   | `/v1/messages/count_tokens`             | ⚠️     | Was cl100k estimate; ignored `tools`. Fixed in this pass. |
| GET    | `/v1/models`                            | ✅     | Alias list. |
| GET    | `/v1/models/{id}`                       | ✅     | Single-model lookup with alias fallback. |
| POST   | `/v1/messages/batches`                  | ❌     | Batch API. |
| GET    | `/v1/messages/batches/{id}`             | ❌     | |
| POST   | `/v1/files`                             | ❌     | Files API (PDFs). |
| —      | Admin API (orgs/workspaces/keys)        | ➖     | Out of scope. |

## 2. Request fields

| Field                     | Status | Notes |
|---------------------------|--------|-------|
| `model`                   | ✅     | Alias registry + longest-prefix fallback. |
| `messages`                | ✅     | |
| `system` (string or blocks) | ✅   | `cache_control` dropped (no NVIDIA equivalent). |
| `max_tokens`              | ✅     | Clamped against model cap + context budget. |
| `stream`                  | ✅     | |
| `temperature`, `top_p`, `top_k` | ✅ | |
| `stop_sequences`          | ✅     | But `stop_sequence` not echoed in response. |
| `tools`                   | ✅     | Sanitized, capped, MCP-passthrough. |
| `tool_choice`             | ✅     | `auto` / `any` / `none` / `tool`. |
| `tool_choice.disable_parallel_tool_use` | ✅ | Maps to `parallel_tool_calls: false`. |
| `thinking.type`           | ✅     | Drives reasoning toggle. |
| `thinking.budget_tokens`  | ✅     | Enforced in stream translator (chars/4 heuristic). |
| `metadata.user_id`        | ✅     | Forwarded to structured log for tenant analytics. |
| `service_tier`            | ❌     | `auto`/`standard_only` ignored. |
| `container`               | ➖     | Server-tool only. |
| `mcp_servers`             | ❌     | Anthropic-side MCP bindings silently ignored. |

## 3. Content block types in request

| Type                      | Status | Notes |
|---------------------------|--------|-------|
| `text`                    | ✅     | |
| `image` (base64/url)      | ✅     | GIF/WEBP transcoded to PNG. |
| `tool_use`                | ✅     | |
| `tool_result`             | ✅     | List content flattened to text. |
| `thinking` / `redacted_thinking` | ✅ | Dropped from history (signatures proxy-local). |
| `document` (PDF base64)   | ✅     | Text extracted via pypdf (optional dep); plain-text and URL sources handled. |
| `server_tool_use`         | ➖     | No NVIDIA analog. |
| `web_search_tool_result`  | ➖     | |
| `search_result`           | ➖     | |
| `container_upload`        | ➖     | |

## 4. Streaming event types

| Event                     | Status | Notes |
|---------------------------|--------|-------|
| `message_start`           | ✅     | |
| `content_block_start`     | ✅     | text / thinking / tool_use. |
| `content_block_delta`     | ✅     | `text_delta`, `thinking_delta`, `signature_delta`, `input_json_delta`. |
| `content_block_stop`      | ✅     | |
| `message_delta`           | ✅     | `stop_reason`, `usage.output_tokens`. |
| `message_stop`            | ✅     | |
| `ping` (keepalive)        | ✅     | Emitted every 15 s during silent reasoning streams. |
| `error` (mid-stream)      | ✅     | |
| `citations_delta`         | ➖     | Web search only. |

## 5. Response/error headers

| Header                             | Status | Notes |
|------------------------------------|--------|-------|
| `anthropic-request-id`             | ✅     | `req_<hex20>` generated per request. |
| `request-id` (alias)               | ✅     | Same value; both headers emitted. |
| `anthropic-organization-id`        | ✅     | Static `nvd-proxy-local`. |
| `anthropic-ratelimit-requests-*`   | ✅     | Fabricated from settings (conservative Build-tier limits). |
| `anthropic-ratelimit-tokens-*`     | ✅     | Same. |
| `retry-after` on 429/529           | ⚠️     | Passed through if upstream sends; otherwise absent. |

## 6. Error type mapping

| HTTP | Anthropic type            | Status | Notes |
|------|---------------------------|--------|-------|
| 400  | `invalid_request_error`   | ✅     | |
| 401  | `authentication_error`    | ✅     | |
| 403  | `permission_error`        | ✅     | |
| 404  | `not_found_error`         | ✅     | |
| 413  | `request_too_large`       | ✅     | |
| 429  | `rate_limit_error`        | ✅     | |
| 500  | `api_error`               | ✅     | |
| 529  | `overloaded_error`        | ✅     | Also maps 503 → `overloaded_error`. |

## 7. Anthropic beta features

| Beta flag                                   | Status | Notes |
|---------------------------------------------|--------|-------|
| `prompt-caching-2024-07-31`                 | ⚠️     | `cache_control` silently dropped; usage reports 0. |
| `extended-cache-ttl-2025-04-11`             | ⚠️     | Same. |
| `interleaved-thinking-2025-05-14`           | ✅     | State machine switches blocks. |
| `fine-grained-tool-streaming-2025-05-14`    | ✅     | Char-level `input_json_delta` supported. |
| `mcp-client-2025-04-04`                     | ⚠️     | Custom-type tools pass through; no server-side MCP. |
| `pdfs-2024-09-25`                           | ❌     | `document` blocks ignored. |
| `computer-use-2024-10-22`                   | ➖     | Server tool; dropped with warning. |
| `message-batches-2024-09-24`                | ❌     | No batch endpoints. |
| `output-128k-2025-02-19`                    | ❌     | Hard-capped by model `max_output`. |
| `token-efficient-tools-2025-02-19`          | ❌     | |
| `files-api-2025-04-14`                      | ❌     | No file endpoints. |
| `search-results-2025-06-09`                 | ❌     | |

## 8. Response correctness

| Behavior                                       | Status | Notes |
|------------------------------------------------|--------|-------|
| `id` prefixed `msg_`                           | ✅     | |
| `type: "message"`, `role: "assistant"`         | ✅     | |
| `model` echoes the alias the client sent       | ✅     | |
| `content: []` guarantee (≥1 block)             | ✅     | |
| `stop_reason` mapping                          | ✅     | |
| `stop_sequence` echo when matched              | ✅     | Tail-scan of last text block; sets `stop_reason: stop_sequence`. |
| `usage.input_tokens` / `output_tokens`         | ✅     | From upstream. |
| `usage.cache_*` tokens                         | ⚠️     | Always 0 (no caching). |
| Message body `usage.server_tool_use`           | ➖     | |

## 9. Claude Code-specific conventions

| Concern                                          | Status |
|--------------------------------------------------|--------|
| `?beta=true` query param accepted                | ✅ (FastAPI ignores unknown query params) |
| Long-running streams without SDK timeout         | ✅ (ping every 15 s) |
| Parallel tool-call execution                     | ✅ |
| 273+ tool schemas in a single request            | ✅ (desc caps + budget clamp) |
| Multi-turn tool_use ↔ tool_result loops          | ✅ |
| `anthropic-beta` header multi-value parsing      | ✅ (logged per-request) |
| Request cancel/reconnect during stream           | ✅ (StreamingResponse handles client disconnect) |

## 10. Security / ops hardening

| Concern                               | Status |
|---------------------------------------|--------|
| Per-client rate limit                 | ❌ |
| Request body size limit               | ❌ (FastAPI default) |
| Tool schema jsonschema validation     | ⚠️ (shape only) |
| Structured audit log                  | ✅ |
| Prometheus metrics                    | ✅ (`/metrics`; optional `prometheus-client` dep) |
| Cost estimation per request           | ❌ |
| Fallback model chain on 5xx           | ❌ |
| Hot reload of `models.yaml`           | ❌ |

---

## Prioritized roadmap

### P1 — ✅ complete

1. `anthropic-request-id` + `request-id` + `anthropic-ratelimit-*` headers.
2. `ping` SSE events every 15 s during reasoning-heavy streams.
3. `stop_sequence` echo when upstream stop matches a requested sequence.
4. 413 → `request_too_large`, 529 → `overloaded_error` mapping.
5. `anthropic-beta` header parsing + per-request log.
6. `metadata.user_id` pass-through to structured log.
7. `count_tokens` now includes tool schemas + system.
8. `GET /v1/models/{id}` single-model endpoint.
9. Retry/backoff on 429/5xx in NVIDIA client.

### P2 — ✅ complete

10. `thinking.budget_tokens` enforcement in stream translator (chars/4 heuristic).
11. `disable_parallel_tool_use` → `parallel_tool_calls: false` in NVIDIA payload.
12. `document` (PDF/text/URL) block support via optional `pypdf` extraction.
13. Prometheus `/metrics` endpoint (optional `prometheus-client` dep).

### P3 — next

14. Model-fallback chain (`failover_to: [...]` in `models.yaml`).
15. Per-client rate limit (token bucket keyed on `metadata.user_id` or IP).
16. Hot reload of `models.yaml` on SIGHUP.

### P3 — compliance extras (likely unused but correct)

17. Message Batches API stub endpoints (202 → 501 with clear error).
18. Fake `cache_read_input_tokens` accounting based on `cache_control` markers.
19. OpenAPI spec published at `/v1/openapi.json` in Anthropic shape.
20. Files API stub endpoints.

### Out of scope (documented)

- True prompt caching (no NVIDIA equivalent).
- Thinking-block signature replay against real Anthropic.
- Server tools (web_search, computer, bash, code_execution).
- Admin API.
