# nvd-claude-proxy — Production Remediation Blueprint
**Audit lens:** Anthropic Messages API spec + Claude Code (TS SDK) runtime expectations

---

## 1. Executive Verdict

The codebase is approximately **88–92% feature-complete** as a translation shim, with sound architectural bones (chain-of-responsibility processors, decoupled translators, pluggable storage). However, it is **not yet "officially-sanctioned-launch-grade"** because of a small set of **protocol-fidelity defects** that will silently corrupt SDK state, drop revenue-bearing telemetry, or diverge from Anthropic semantics under real Claude Code load.

The defects cluster in five fault domains, each with a deterministic fix path:

| Fault Domain | Status | Blast Radius |
|---|---|---|
| **A. SSE event grammar** (signature_delta, input_json_delta, event-id, citations_delta) | **P0 — protocol-violating** | All streaming clients |
| **B. Tool-use determinism** (parallel ordering, fuzzy mapping, `is_error` loss, MCP) | **P0 — semantic drift** | Every Claude Code turn |
| **C. Type/schema parity** (file sources, cache TTL, `pause_turn`/`refusal`, server-tool result blocks) | **P1 — surface gaps** | Beta features, multi-turn |
| **D. Resilience & security** (constant-time auth, body limit default, idempotency TTL, stream cancellation) | **P0 — production hazard** | All requests |
| **E. Telemetry truthfulness** (cumulative output_tokens, cache_creation breakdown, fabricated rate-limit headers) | **P1 — billing/quota drift** | Claude Code budgeting |

The remediation below is **structural, not patch-level**: it re-architects the translation pipeline so that protocol invariants are enforced by types and pipeline stages, not by ad-hoc string handling.

---

## 2. Target Architecture (the "north star")

```
                        ┌──────────────────────────────────────────────────┐
HTTP request ──►        │  AnthropicEdge   (FastAPI router + middleware)   │
                        │  • AuthGuard (constant-time)                     │
                        │  • VersionGate (anthropic-version=2023-06-01)    │
                        │  • BetaNegotiator (CSV → BetaSet, gate features) │
                        │  • IdempotencyVault (key+req-hash, 24h TTL)      │
                        │  • BodyLimiter (default 32 MiB, hard cap)        │
                        └────────────────────┬─────────────────────────────┘
                                             ▼
                        ┌──────────────────────────────────────────────────┐
                        │  CanonicalIR  (single immutable Pydantic v2 model)│
                        │  – Discriminated unions for every block type    │
                        │  – Validators reject invalid alternations,      │
                        │    orphan tool_results, schema violations       │
                        └────────────────────┬─────────────────────────────┘
                                             ▼
                        ┌──────────────────────────────────────────────────┐
                        │  Router  → ModelSpec (max_output, beta caps)    │
                        │  RequestProjector (Anthropic IR → OpenAI IR)    │
                        │   – Strict JSON-Schema sanitiser (oneOf/anyOf   │
                        │     flattened, $ref resolved/dropped)           │
                        │   – Tool registry: server-tool injector +       │
                        │     dated-suffix collapser (data-driven)        │
                        └────────────────────┬─────────────────────────────┘
                                             ▼
                        ┌──────────────────────────────────────────────────┐
                        │  UpstreamClient (httpx + circuit breaker +      │
                        │   retry on {408,409,429,5xx} w/ jitter)         │
                        │  Streaming uses async-iterator with aclose()    │
                        └────────────────────┬─────────────────────────────┘
                                             ▼
                        ┌──────────────────────────────────────────────────┐
                        │  ResponseAssembler                              │
                        │  Non-stream: deterministic block ordering,       │
                        │              cumulative usage, stop_sequence    │
                        │              tail-scan across all block kinds   │
                        │  Stream:     EventGrammar (FSM) emits exact     │
                        │              Anthropic SSE sequence with         │
                        │              monotonic indices, signature_delta │
                        │              as final thinking delta, char-     │
                        │              level input_json_delta, ping       │
                        │              proactive scheduler, event ids     │
                        └──────────────────────────────────────────────────┘
```

The two new constructs that don't exist today and **must be added**:

1. **EventGrammar FSM** — replaces the imperative emitters in `stream_translator.py:543-582` and `processors.py:248,326`. State machine guarantees content-block-start/delta/stop ordering, monotonic indices, and proper signature delta placement.
2. **CanonicalIR** — replaces the soft duck-typed dicts that flow between `request_translator`, `response_translator`, and `stream_translator`. Pydantic v2 discriminated unions catch malformed blocks at the boundary, not in production.

---

## 3. Remediation Plan — Ordered by Priority

### P0 (block launch) — protocol & safety

| # | File:Line | Defect | Required Change |
|---|---|---|---|
| P0-1 | `translators/stream_translator.py:411,764`; `core/processors.py:248,326` | `input_json_delta.partial_json` carries accumulated args, not incremental fragments | Buffer raw JSON characters from upstream; emit each chunk as a fragment; never re-flush the accumulator. Add property test: `concat(deltas) == final_input_json` and each delta is non-empty. |
| P0-2 | `translators/stream_translator.py:242-254` | `signature_delta` emitted bundled with `content_block_stop`, not as last delta | EventGrammar must emit `thinking_delta`*, `signature_delta`, `content_block_stop` in that order. Required for Anthropic SDK signature validation on multi-turn extended thinking. |
| P0-3 | `util/sse.py:9-16`; `routes/messages.py:568,624,629` | SSE `id:` field never written; `SSEEvent.id` declared but unused | Encode `id: <monotonic>` per event; persist last-seen-id per session for `Last-Event-ID` resumption (ties to idempotency vault). |
| P0-4 | `core/processors.py`, `translators/stream_translator.py` (entire) | Server-side tool deltas (`web_search_tool_result`, `code_execution_tool_result`) not handled | Add server-tool handlers in EventGrammar: open `server_tool_use` block, then a sibling `*_tool_result` block, with content/error/citations sub-deltas. Gate on `web-search-2025-03-05` / `code-execution-2025-05-22` betas. |
| P0-5 | `translators/tool_fuzzy_mapper.py:75-108` | `difflib.get_close_matches` non-deterministic; can dispatch wrong tool silently | **Delete fuzzy fallback entirely.** Replace with strict registry lookup → on miss, emit `tool_use` block unchanged AND set `is_error=true` only if model later requires execution. Log + counter, never silent rewrite. |
| P0-6 | `translators/request_translator.py:239-245` | `tool_result.is_error` flag lost when projecting to OpenAI `role:"tool"` | Prepend a sentinel marker into the tool message content (`<error>...</error>`) **and** add a structured field via OpenAI's `tool_call_id` metadata channel; preserve in CanonicalIR for round-trip. |
| P0-7 | `middleware/security.py:474` | API-key compare is non-constant-time | `hmac.compare_digest(presented or "", s.proxy_api_key)`. |
| P0-8 | `config/settings.py:48` | `MAX_REQUEST_BODY_MB` defaults to 0 (unlimited) | Default `32`, validator clamps to ≤32. Reject 413 with Anthropic `request_too_large` envelope. |
| P0-9 | `routes/messages.py` streaming generator | Client disconnect doesn't propagate to upstream `httpx.Response` | Wrap generator in `try/finally`; call `await upstream.aclose()` and `task.cancel()` on `GeneratorExit`. Add metric `stream_cancelled_total`. |
| P0-10 | `routes/messages.py:131-141` | Idempotency cache keyed only by `idempotency-key` — replay attack & cross-model collision | Key = `sha256(idempotency-key ‖ model ‖ canonical-request-hash)`; TTL=24h; validate request hash on hit and 400 if mismatched. |
| P0-11 | `clients/nvidia_client.py:20` | Retry set excludes 408/409 | `_RETRY_STATUSES = {408, 409, 429, 500, 502, 503, 504}`. |

### P1 (close before public release) — parity gaps

| # | File:Line | Defect | Required Change |
|---|---|---|---|
| P1-1 | `schemas/anthropic.py:22-24` | `cache_control` lacks `ttl` discriminator | Add `ttl: Literal["5m","1h"] \| None`; gate `1h` behind `extended-cache-ttl-2025-04-11` beta. Wire into `util/cache_accounting.py`. |
| P1-2 | `schemas/anthropic.py:35-49,67-72` | `image.source` & `document.source` lack `file` (file_id) variant | Add `FileSource(type="file", file_id: str)` discriminated branch; resolve via `/v1/files` stub (return 501 with explicit "files-api beta required"). |
| P1-3 | `schemas/anthropic.py:222`; `core/processors.py` (Finalizer) | Stop reasons missing `pause_turn`, `refusal` | Extend `StopReason` literal; map upstream finish_reason `length→max_tokens`, `content_filter→refusal`, async-yield→`pause_turn`. |
| P1-4 | `schemas/anthropic.py:111-122` | Missing block types: `search_result`, `web_search_tool_result`, `code_execution_tool_result`, `mcp_tool_use`, `mcp_tool_result` | Add all five; wire into validator unions; required for #P0-4. |
| P1-5 | `schemas/anthropic.py:190-207` | Request lacks `mcp_servers`, `container` | Add fields; mcp_servers passes through to upstream when MCP beta active; container returns 400 if upstream lacks support. |
| P1-6 | `schemas/anthropic.py:209-214` | `Usage` lacks `cache_creation` breakdown (`ephemeral_5m_input_tokens`, `ephemeral_1h_input_tokens`) and `server_tool_use` counters | Extend `Usage`; populate from `cache_accounting` per-block walk. |
| P1-7 | `translators/schema_sanitizer.py:71-72` | `oneOf/anyOf/allOf` preserved — vLLM upstream rejects | Implement schema flattener: `oneOf` of object schemas → merge required as union, properties as union with `additionalProperties:true`. Unit-test against the 12 most common Claude Code tool shapes. |
| P1-8 | `translators/schema_sanitizer.py:29,51` | Tool-name regex caps at 64 chars (NIM limit), Anthropic spec is 128 | Hash-suffix collision-safe: if name >64, truncate to 56 + 8-char b64(sha256) suffix; maintain bidirectional alias map for round-trip. |
| P1-9 | `translators/stream_translator.py:451-456,793-794` | `output_tokens` only set from final upstream chunk; `message_delta` fires once at end | Emit `message_delta` with cumulative `output_tokens` after **every** content block close (not every delta — too chatty); use tokenizer estimate as floor when upstream usage missing. Critical for Claude Code budget UI. |
| P1-10 | `routes/messages.py:147-148` | `anthropic-version` accepts anything not on a denylist | Strict equality check vs `ANTHROPIC_VERSION_ALLOWLIST = {"2023-06-01"}` — return 400 `invalid_request_error` otherwise. |
| P1-11 | `routes/messages.py:162-172` | `anthropic-beta` parsed, unsupported betas ignored | `BetaNegotiator` class: enumerate supported betas; raise `400` if request *requires* an unsupported beta (e.g., body has `mcp_servers` but `mcp-client-2025-04-04` not in header). |
| P1-12 | `util/router.py`, `config/models.py` | Versioned model IDs (`claude-haiku-4-5-20251001`) miss alias map | Longest-prefix matcher: strip trailing `-YYYYMMDD` or `-vN` then retry; cache resolution. |
| P1-13 | `routes/messages.py` (request handler) | `service_tier` parsed but dropped silently | Forward to upstream where supported; otherwise echo back in response `service_tier` field as `"standard"` and emit `service_tier_downgraded_total` counter. |
| P1-14 | `util/anthropic_headers.py:44-49` | `anthropic-ratelimit-*` headers fabricated with static values | Source from `RateLimiter.snapshot()` for the bucket actually applied. If upstream NVIDIA quota unknown, **omit** the headers rather than lie. SDKs handle absence; lying breaks backoff. |
| P1-15 | `core/processors.py` (Finalizer) | `output_tokens` may diverge from sum of block contents | Re-tokenize emitted content with `tiktoken` proxy → cross-check; if drift >5%, log `usage_drift_warning` + emit reconciled value. |

### P2 (polish for "official-grade")

| # | File:Line | Defect | Required Change |
|---|---|---|---|
| P2-1 | `util/cache_accounting.py:157-162` | `cache_creation_input_tokens = cached_estimate // 10` is a stub | Walk request top-down; sum tokens of every block from start through last `cache_control` marker into creation; everything before first marker on cache hit into reads. Per-TTL bucket. |
| P2-2 | `routes/messages.py` ping logic | Reactive (only on queue timeout) — slow-but-steady streams never ping | Add proactive `asyncio.create_task(_ping_scheduler)` independent of queue; cancel on stream close. 10s cadence (Anthropic uses 15s; safer to be ahead). |
| P2-3 | `routes/health.py:35-37` | `/readyz` collapses all upstream errors to "degraded" | Distinguish `auth_failed`, `rate_limited`, `upstream_5xx`, `circuit_open`; expose in body for ops dashboards. |
| P2-4 | `errors/mapper.py:5-18` | No 402 → `billing_error` mapping | Add. Rare but spec-mandated. |
| P2-5 | `routes/messages.py:137` | Idempotency key logged in full | Truncate to 12 chars + length suffix (consistent with session_middleware). |
| P2-6 | `util/circuit_breaker.py:89-92` | Mid-stream failures don't trip breaker | Pass a `failure_callback` to streaming generator; invoke on `IncompleteRead`/`ReadTimeout`/`ProtocolError`. |
| P2-7 | `middleware/security.py` (SSRF) | Walks image URLs; misses document blocks with `source.type="url"` | Generalise traversal to any `{type:"url", url:...}` node anywhere in payload. |
| P2-8 | `routes/openapi.py` | Custom OpenAPI not Anthropic-shape-faithful | Generate from CanonicalIR Pydantic models; cross-validate against Anthropic-published OAS. |
