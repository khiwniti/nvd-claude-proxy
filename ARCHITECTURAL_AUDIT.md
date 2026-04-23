# nvd-claude-proxy: Production-Grade Claude Code Parity Remediation Blueprint

**Document Type:** Architectural Audit & Implementation Roadmap  
**Author:** Senior Principal Engineer / Core Architect (Anthropic SDK)  
**Date:** 2026-04-22  
**Version:** 1.0  
**Status:** AUTHORITATIVE  

---

## Executive Summary

This document provides an exhaustive, production-critical audit of `nvd-claude-proxy` against Anthropic's official Messages API specification and Claude Code's operational requirements. The audit identifies structural deficiencies, missing paradigms, and implementation gaps that prevent full feature parity. A phased remediation blueprint is provided with prioritized implementations targeting a stable, officially-sanctioned launch.

**Assumption:** This proxy operates in **translation-only mode** (no native tool execution). Claude Code executes tools client-side; the proxy's role is faithful translation of Anthropic ↔ NVIDIA NIM protocols.

---

## Part I: Current State Assessment

### 1.1 Architecture Overview

The proxy consists of these layers:

```
┌─────────────────────────────────────────────────────────────────┐
│                        FastAPI Application                       │
├─────────────────────────────────────────────────────────────────┤
│  Middleware: Logging │ Rate Limiting │ Body Limits │ SIGHUP     │
├─────────────────────────────────────────────────────────────────┤
│  Routes: /v1/messages │ /v1/models │ /v1/messages/count_tokens  │
├─────────────────────────────────────────────────────────────────┤
│  Translation Layer                                               │
│  ┌──────────────┐ ┌──────────────┐ ┌───────────────┐           │
│  │   Request    │ │   Response   │ │    Stream     │           │
│  │  Translator  │ │  Translator  │ │  Translator   │           │
│  └──────────────┘ └──────────────┘ └───────────────┘           │
│  ┌──────────────┐ ┌──────────────┐ ┌───────────────┐           │
│  │    Tool      │ │   Thinking   │ │    Vision     │           │
│  │  Translator  │ │  Translator  │ │  Translator   │           │
│  └──────────────┘ └──────────────┘ └───────────────┘           │
├─────────────────────────────────────────────────────────────────┤
│  Client Layer: NvidiaClient (httpx async)                       │
├─────────────────────────────────────────────────────────────────┤
│  Config: Model Registry │ Settings │ Error Mapper               │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 What's Implemented Well

| Component | Status | Notes |
|-----------|--------|-------|
| SSE streaming | ✅ Solid | Correct event ordering, ping keepalive, proper chunking |
| Request translation | ✅ Good | Most Anthropic → OpenAI conversions handled |
| Response translation | ✅ Good | Non-streaming path complete |
| Model registry | ✅ Good | YAML config, prefix fallback, failover chains |
| Error mapping | ✅ Complete | Anthropic error types mapped correctly |
| Token estimation | ✅ Good | tiktoken with fallback, tool-inclusive |
| Tool schema sanitization | ✅ Good | JSON schema normalization, collision detection |
| Schema validation | ⚠️ Optional | jsonschema dependency is optional |
| Rate limiting | ⚠️ Simple | IP-based, no user_id prioritization |
| Body limits | ✅ Implemented | Configurable max_request_body_mb |

### 1.3 Critical Gaps Inventory

#### Category A: Protocol-Level Parity (P0)

| Gap | Impact | Location |
|-----|--------|----------|
| No `/v1/messages/batches` endpoints | ❌ Missing API | routes/messages.py |
| No `/v1/files` endpoints | ❌ Missing API | routes/ |
| No prompt cache accounting | ⚠️ Incomplete | Always reports 0 cache tokens |
| No `output-128k` support | ⚠️ Limited | Hard-capped by max_output per model |
| No `token-efficient-tools` support | ❌ Missing beta | No implementation |
| `service_tier` silently ignored | ⚠️ Incomplete | Dropped, not even logged |
| `container` block not handled | ❌ Missing | Documented as N/A but could be stubbed |

#### Category B: Tool System Parity (P0)

| Gap | Impact | Location |
|-----|--------|----------|
| No MCP tool execution | ❌ Major | Only custom-type passthrough, no execution |
| Parallel tool_result ordering | ⚠️ Risk | No guarantee of ordering in multi-result responses |
| `disable_parallel_tool_use` | ✅ Done | Mapped to `parallel_tool_calls: false` |
| Tool name collision handling | ✅ Done | Drop later duplicate to preserve determinism |
| Undeclared tool blocking | ⚠️ Partial | Only blocks in stream; non-stream passes through |
| Tool repetition detection | ✅ Done | Implemented in StreamTranslator |

#### Category C: Streaming & Real-time (P1)

| Gap | Impact | Location |
|-----|--------|----------|
| No mid-stream error recovery | ⚠️ Limited | Errors after first chunk end stream |
| No request cancellation support | ⚠️ Limited | Client disconnect ends stream |
| `citations_delta` support | ➖ N/A | Web search only |
| Vision in streaming | ⚠️ Unclear | Image blocks not tested in stream path |

#### Category D: Type Safety & Validation (P1)

| Gap | Impact | Location |
|-----|--------|----------|
| No request validation layer | ❌ Risk | Raw dicts accepted throughout |
| No response validation | ❌ Risk | No verification of upstream response shape |
| Optional jsonschema | ⚠️ Risk | Schema validation fails silently if dep missing |
| No input sanitization | ⚠️ Risk | Tool names/args sanitized; other fields not |
| Type safety gaps | ⚠️ Risk | `Any` types used liberally |

#### Category E: Resilience & Operations (P1)

| Gap | Impact | Location |
|-----|--------|----------|
| No circuit breaker | ❌ Risk | Cascading failures possible |
| No request deduplication | ⚠️ Risk | Idempotency not enforced |
| No proper backoff for streaming | ⚠️ Risk | Non-stream only |
| Cost estimation incomplete | ⚠️ Incomplete | No per-request USD tracking in logs |
| Hot reload partial | ⚠️ Partial | SIGHUP handler exists; not fully tested |

#### Category F: Testing & Documentation (P2)

| Gap | Impact | Location |
|-----|--------|----------|
| No integration tests | ❌ Risk | Only unit tests |
| No e2e with real Claude Code | ❌ Risk | test_with_claude_code.md exists but untested |
| OpenAPI spec not Anthropic-shaped | ❌ Missing | No custom schema at /v1/openapi.json |
| No chaos engineering tests | ⚠️ Missing | No failure injection |

---

## Part II: Detailed Gap Analysis & Remediation

### GAP-001: No Batch API Endpoints

**Current State:** No endpoints exist.  
**Required:** `POST /v1/messages/batches` and `GET /v1/messages/batches/{id}`.  
**Anthropic Spec:** Batches API handles large job queues with `expires_at`, `id`, `status`, `results`, `errors`.  
**Implementation:**

```python
# routes/batches.py (NEW)
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
from typing import Literal

router = APIRouter()

class BatchRequest(BaseModel):
    """Minimal batch request - full spec in schemas/anthropic.py"""
    endpoint: Literal["/v1/messages"]
    arguments: dict
    completion_window: Literal["1h", "24h"] = "1h"
    metadata: dict | None = None

class BatchResponse(BaseModel):
    id: str
    endpoint: str
    arguments: dict
    completion_window: str
    status: Literal["pending", "in_progress", "ended", "expired", "cancelled"]
    created_at: int
    expires_at: int
    processing_at: int | None = None
    completed_at: int | None = None
    error: dict | None = None
    results: dict | None = None

@router.post("/v1/messages/batches", response_model=BatchResponse)
async def create_batch(request: Request, body: BatchRequest):
    """Create a batch message request."""
    # NVIDIA NIM doesn't support batches. Options:
    # 1. Reject with clear error explaining limitation
    # 2. Simulate batch with immediate execution (not recommended)
    # 3. Queue locally and process (complex)
    raise HTTPException(
        status_code=501,
        detail={
            "type": "error",
            "error": {
                "type": "api_error",
                "message": "Batch API is not supported by the NVIDIA NIM backend. "
                          "Submit requests individually via POST /v1/messages."
            }
        }
    )
```

**Risk:** Low (graceful 501 with clear message)  
**Priority:** P2 (batches rarely used by Claude Code)

---

### GAP-002: No Files API Endpoints

**Current State:** No endpoints.  
**Required:** `POST /v1/files`, `GET /v1/files`, `GET /v1/files/{id}`, `DELETE /v1/files/{id}`.  
**Anthropic Spec:** File upload/download for PDFs, images, documents.  
**Implementation:**

```python
# routes/files.py (NEW)
from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel

router = APIRouter()

class FileObject(BaseModel):
    id: str
    object: Literal["file"] = "file"
    bytes: int
    created_at: int
    filename: str
    purpose: Literal["batch", "processor"]
    status: Literal["uploaded", "processed", "error"]
    error: dict | None = None

@router.post("/v1/files")
async def upload_file(
    file: UploadFile = File(...),
    purpose: str = "batch"
) -> FileObject:
    """Upload a file for batch processing."""
    # NVIDIA NIM file handling is not standardized.
    # Option 1: Convert to base64 and embed in request
    # Option 2: Reject with clear error
    content = await file.read()
    return FileObject(
        id=f"file_{hash(content)[:16]}",
        bytes=len(content),
        created_at=int(time.time()),
        filename=file.filename or "unknown",
        purpose=purpose,
        status="uploaded"
    )
```

**Risk:** Medium (file handling is complex)  
**Priority:** P2

---

### GAP-003: No Prompt Cache Accounting

**Current State:** `cache_creation_input_tokens` and `cache_read_input_tokens` always return 0.  
**Required:** Realistic accounting when `cache_control` markers are present.  
**Anthropic Spec:** Prompt caching shows 90% cost reduction on cache reads.  

**Implementation:**

```python
# Enhanced in routes/messages.py or new util/cache_accounting.py

def estimate_cache_tokens(body: dict, cache_markers: list[str]) -> tuple[int, int]:
    """Estimate cache creation vs read tokens based on markers.
    
    Anthropic caching:
    - First appearance of cache_control marker = cache creation
    - Subsequent references = cache reads (90% discount)
    """
    # Walk the body and identify cache_control markers
    creation_tokens = 0
    read_tokens = 0
    
    # Simplified: treat all cached content as creation on first encounter
    # Real implementation would track per-block markers
    return creation_tokens, read_tokens

# In response translation, when cache_control blocks are present:
if has_cache_control_markers(body):
    creation, read = estimate_cache_tokens(body, markers)
    usage["cache_creation_input_tokens"] = creation
    usage["cache_read_input_tokens"] = read
    # Cost = (creation_tokens * 1.0 + read_tokens * 0.1) * price_per_token
```

**Risk:** Medium (estimation may differ from actual)  
**Priority:** P2 (Claude Code doesn't heavily use caching)

---

### GAP-004: No Token-Efficient Tools Support

**Current State:** Beta flag `token-efficient-tools-2025-02-19` exists but no implementation.  
**Required:** Tool schemas optimized for token efficiency.  
**Anthropic Spec:** Minimizes token overhead for tool definitions.

**Implementation:**

```python
# Enhanced in translators/tool_translator.py

def anthropic_tools_to_openai_efficient(
    tools: list[dict] | None,
    *,
    tool_id_map: "ToolIdMap | None" = None,
    token_budget: int | None = None,  # NEW: max tokens for tool schemas
) -> list[dict]:
    """Convert tools with token-efficient encoding.
    
    When token_budget is set, aggressively truncate descriptions
    and reduce schema complexity to stay within budget.
    """
    if not token_budget:
        return anthropic_tools_to_openai(tools, tool_id_map=tool_id_map)
    
    # Start with minimal descriptions
    MINIMAL_DESC_CHARS = 50
    output = anthropic_tools_to_openai(
        tools, 
        tool_id_map=tool_id_map,
        description_cap=MINIMAL_DESC_CHARS
    )
    
    # If still over budget, drop non-required parameters
    for tool in output:
        params = tool["function"]["parameters"]
        if "required" in params:
            # Keep only required params
            required = set(params["required"])
            params["properties"] = {
                k: v for k, v in params["properties"].items() 
                if k in required
            }
    
    return output

# In request translator, check beta flag:
if "token-efficient-tools-2025-02-19" in betas:
    payload["tools"] = anthropic_tools_to_openai_efficient(
        tools, tool_id_map=tool_id_map, token_budget=4096
    )
```

**Risk:** Medium (affects tool behavior)  
**Priority:** P2

---

### GAP-005: No Request Validation Layer

**Current State:** Raw dicts accepted. Malformed requests pass through.  
**Required:** Pydantic validation of inbound Anthropic requests.  
**Implementation:**

```python
# schemas/validators.py (NEW)
from pydantic import BaseModel, Field, field_validator
from typing import Annotated, Any, Literal, Union

class ValidatedMessagesRequest(BaseModel):
    """Strictly validated Anthropic Messages request."""
    model: str = Field(..., min_length=1)
    messages: list[dict] = Field(..., min_length=1)
    system: str | list[dict] | None = None
    max_tokens: int = Field(default=1024, ge=1, le=200000)
    stream: bool = False
    temperature: float | None = Field(default=None, ge=0, le=2)
    top_p: float | None = Field(default=None, ge=0, le=1)
    top_k: int | None = Field(default=None, ge=1)
    stop_sequences: list[str] | None = None
    tools: list[dict] | None = None
    tool_choice: Any = None
    thinking: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None
    service_tier: str | None = None
    
    @field_validator('messages')
    @classmethod
    def validate_messages(cls, v):
        for idx, msg in enumerate(v):
            if msg.get('role') not in ('user', 'assistant', 'system'):
                raise ValueError(f"Invalid role at message {idx}: {msg.get('role')}")
        return v
    
    @field_validator('tools')
    @classmethod
    def validate_tools(cls, v):
        if v:
            for idx, tool in enumerate(v):
                if not tool.get('name'):
                    raise ValueError(f"Tool at index {idx} missing 'name'")
                if not tool.get('input_schema'):
                    raise ValueError(f"Tool {tool.get('name')} missing 'input_schema'")
        return v

# In routes/messages.py:
try:
    validated = ValidatedMessagesRequest(**body)
except ValidationError as e:
    return ORJSONResponse(
        {"type": "error", "error": {"type": "invalid_request_error", "message": str(e)}},
        status_code=400
    )
```

**Risk:** Low (stricter validation prevents downstream issues)  
**Priority:** P1

---

### GAP-006: Parallel Tool Result Ordering Not Guaranteed

**Current State:** When multiple tool_results are submitted in parallel, ordering depends on arrival.  
**Required:** Claude Code expects tool results in same order as tool_use calls.  
**Implementation:**

```python
# Enhanced in routes/messages.py

async def _process_tool_results_parallel(
    tool_uses: list[dict],
    tool_results: list[dict]
) -> list[dict]:
    """Reorder tool_results to match tool_use order.
    
    Anthropic requires: tool_result[i] corresponds to tool_use[i]
    NVIDIA NIM may return in different order.
    """
    # Build position map from tool_use ids
    tool_order = {tu['id']: idx for idx, tu in enumerate(tool_uses)}
    
    # Sort tool_results by corresponding tool_use position
    def get_order(tr):
        tu_id = tr.get('tool_use_id', '')
        return tool_order.get(tu_id, len(tool_order))
    
    return sorted(tool_results, key=get_order)

# In request handling:
if tool_results:
    tool_results = await _process_tool_results_parallel(
        tool_uses, tool_results
    )
```

**Risk:** Low (reordering is safe)  
**Priority:** P1

---

### GAP-007: No Circuit Breaker Pattern

**Current State:** Failures cascade without protection.  
**Required:** Circuit breaker for upstream NVIDIA API.  
**Implementation:**

```python
# New file: util/circuit_breaker.py
import time
from enum import Enum
from dataclasses import dataclass
from typing import Callable, TypeVar

class CircuitState(Enum):
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject immediately
    HALF_OPEN = "half_open"  # Testing recovery

@dataclass
class CircuitBreakerConfig:
    failure_threshold: int = 5      # Failures before opening
    success_threshold: int = 3      # Successes to close
    timeout: float = 30.0           # Seconds before half-open
    half_open_max_calls: int = 3    # Max test calls in half-open

class CircuitBreaker:
    def __init__(self, config: CircuitBreakerConfig = None):
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self.failures = 0
        self.successes = 0
        self.last_failure_time: float | None = None
        
    async def call(self, func: Callable, *args, **kwargs):
        if self.state == CircuitState.OPEN:
            if time.time() - self.last_failure_time > self.config.timeout:
                self.state = CircuitState.HALF_OPEN
                self.successes = 0
            else:
                raise CircuitBreakerOpenError("Circuit breaker is OPEN")
        
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise
    
    def _on_success(self):
        self.failures = 0
        if self.state == CircuitState.HALF_OPEN:
            self.successes += 1
            if self.successes >= self.config.success_threshold:
                self.state = CircuitState.CLOSED
                
    def _on_failure(self):
        self.failures += 1
        self.last_failure_time = time.time()
        if self.failures >= self.config.failure_threshold:
            self.state = CircuitState.OPEN
```

**Risk:** Low (protects upstream)  
**Priority:** P2

---

### GAP-008: No Request Deduplication / Idempotency

**Current State:** Identical requests may be processed multiple times.  
**Required:** Idempotency key support for safe retries.  
**Implementation:**

```python
# Enhanced in routes/messages.py
from fastapi import Header

@router.post("/v1/messages")
async def messages(
    request: Request,
    idempotency_key: str | None = Header(None, alias="anthropic-idempotency-key")
):
    if idempotency_key:
        # Check cache (use Redis in production, dict for single-instance)
        cached = await get_idempotency_cache(idempotency_key)
        if cached:
            _log.info("messages.idempotent_replay", key=idempotency_key)
            return cached
        
        # Process request
        response = await _do_messages(request)
        
        # Cache successful response
        await set_idempotency_cache(idempotency_key, response, ttl=24*3600)
        return response
    
    return await _do_messages(request)

# For multi-instance deployment, use Redis:
# await redis.setex(f"idempotency:{key}", 86400, json.dumps(response))
```

**Risk:** Low (idempotency is safe)  
**Priority:** P2

---

### GAP-009: Vision Content in Streaming Mode Not Fully Tested

**Current State:** Vision translator handles images but stream_translator doesn't explicitly handle image blocks.  
**Required:** Test and verify image content in streaming responses.  
**Implementation:**

```python
# Enhanced stream_translator.py
def _handle_content_block(self, delta: dict) -> Iterator[dict]:
    """Handle all content block types including images in streams."""
    # Existing: text, thinking, tool_use
    # NEW: image blocks in streaming response
    
    if image_url := delta.get("image_url"):
        # NVIDIA may return images in streaming (rare but possible)
        yield self._emit(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": self._next_index,
                "content_block": {
                    "type": "image",
                    "source": image_url_to_anthropic(image_url)
                }
            }
        )
        yield self._emit(
            "content_block_stop",
            {"type": "content_block_stop", "index": self._next_index}
        )
        self._next_index += 1

# Add test case:
def test_vision_stream_with_image_delta():
    """Images can appear in streaming responses."""
    chunks = [
        {"choices": [{"index": 0, "delta": {"content": "Here's the image:"}}]},
        # Hypothetical image delta from NVIDIA
        # {"choices": [{"index": 0, "delta": {"image_url": {...}}}]},
        {"choices": [{"index": 0, "delta": {}}]},
    ]
    # Verify image block is properly formatted
```

**Risk:** Medium (depends on NVIDIA behavior)  
**Priority:** P1

---

### GAP-010: No OpenAPI Spec in Anthropic Shape

**Current State:** FastAPI generates default OpenAPI spec.  
**Required:** Anthropic-compatible API documentation at `/v1/openapi.json`.  
**Implementation:**

```python
# Enhanced in app.py or new routes/openapi.py
from fastapi.openapi.utils import get_openapi

def anthropic_openapi_schema() -> dict:
    """Generate Anthropic-shaped OpenAPI spec."""
    anthropic_paths = {
        "/v1/messages": {
            "post": {
                "operationId": "createMessage",
                "summary": "Create a Message",
                "description": "Create a model response for a given prompt.",
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/MessageRequest"}
                        }
                    }
                },
                "responses": {
                    "201": {
                        "description": "Message created",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/MessageResponse"}
                            }
                        }
                    }
                }
            }
        }
        # ... other endpoints
    }
    
    return {
        "openapi": "3.0.0",
        "info": {
            "title": "Anthropic Messages API",
            "version": "2023-06-01"
        },
        "paths": anthropic_paths,
        "components": {
            "schemas": {
                "MessageRequest": {...},  # From schemas/anthropic.py
                "MessageResponse": {...}
            }
        }
    }
```

**Risk:** Low (documentation only)  
**Priority:** P3

---

## Part III: Production Hardening Checklist

### III.A Security Hardening

| Item | Current | Required | Implementation |
|------|---------|----------|----------------|
| Zero-trust input sanitization | ⚠️ Partial | ✅ Complete | Sanitize ALL fields, not just tools |
| Rate limiting | ⚠️ Simple | ✅ Advanced | Token bucket per user_id, burst allowance |
| Request body limits | ✅ Done | ✅ | FastAPI middleware |
| API key validation | ✅ Done | ✅ | Header + Bearer token check |
| SQL injection | ➖ N/A | ✅ | No DB |
| SSRF prevention | ⚠️ Not implemented | ✅ | Validate all URLs in image/document blocks |
| Log sanitization | ⚠️ Partial | ✅ | Mask API keys, PII in logs |
| CORS configuration | ⚠️ Not set | ✅ | Restrictive CORS for browser clients |

**Implementation:**

```python
# Enhanced security middleware in middleware/security.py
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Content-Security-Policy"] = "default-src 'none'"
        return response

class SSRFProtectionMiddleware(BaseHTTPMiddleware):
    """Validate all URLs in request body."""
    BLOCKED_SCHEMES = {"javascript", "file", "ftp"}
    BLOCKED_HOSTS = {"localhost", "127.0.0.1", "169.254.169.254"}
    
    async def dispatch(self, request: Request, call_next):
        if request.url.path in ("/v1/messages", "/v1/messages/batches"):
            body = await request.json()
            urls = extract_all_urls(body)  # Recursive extraction
            for url in urls:
                parsed = urlparse(url)
                if parsed.scheme.lower() in self.BLOCKED_SCHEMES:
                    raise HTTPException(400, "Blocked URL scheme")
                if parsed.hostname in self.BLOCKED_HOSTS:
                    raise HTTPException(400, "Blocked hostname")
        return await call_next(request)
```

---

### III.B Resilience Patterns

| Pattern | Current | Required | Implementation |
|---------|---------|----------|----------------|
| Circuit breaker | ❌ Missing | ✅ | See GAP-007 |
| Rate limiting | ⚠️ Simple | ✅ Advanced | Token bucket with Redis |
| Retry with backoff | ⚠️ Non-stream only | ✅ Complete | Unified retry for all paths |
| Graceful degradation | ⚠️ Partial | ✅ Complete | Fallback to text-only on vision failure |
| Timeout management | ⚠️ Simple | ✅ | Per-request, per-operation timeouts |
| Load shedding | ❌ Missing | ✅ | Reject requests under heavy load |

**Implementation:**

```python
# Enhanced load shedding in middleware/load_shedding.py
import psutil

class LoadSheddingMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, max_queue_depth: int = 100, max_cpu_percent: float = 90.0):
        super().__init__(app)
        self.max_queue_depth = max_queue_depth
        self.max_cpu_percent = max_cpu_percent
        
    async def dispatch(self, request: Request, call_next):
        # Check CPU load
        cpu_percent = psutil.cpu_percent(interval=0.1)
        if cpu_percent > self.max_cpu_percent:
            return ORJSONResponse(
                {"type": "error", "error": {"type": "overloaded_error", 
                 "message": "Server under heavy load, please retry"}},
                status_code=529
            )
        
        # Check queue depth (approximate via active requests)
        active = len([r for r in asyncio.all_tasks() if not r.done()])
        if active > self.max_queue_depth:
            return ORJSONResponse(
                {"type": "error", "error": {"type": "overloaded_error",
                 "message": "Request queue full, please retry"}},
                status_code=529
            )
        
        return await call_next(request)
```

---

### III.C Observability

| Metric | Current | Required | Implementation |
|--------|---------|----------|----------------|
| Request latency histogram | ✅ Prometheus | ✅ | Per-model, per-endpoint |
| Token usage counter | ✅ Prometheus | ✅ | input/output/cache breakdown |
| Error rate by type | ⚠️ Partial | ✅ | Per error type, per model |
| Upstream latency | ⚠️ Basic | ✅ | Separate from proxy overhead |
| Cache hit rate | ❌ Missing | ✅ | Track cache_control usage |
| Circuit breaker state | ❌ Missing | ✅ | Prometheus gauge |

**Implementation:**

```python
# Enhanced metrics in util/metrics.py
from prometheus_client import Counter, Histogram, Gauge

# New metrics
STREAM_CHUNK_LATENCY = Histogram(
    'nvd_stream_chunk_latency_seconds',
    'Time to translate each stream chunk',
    ['model', 'chunk_type']
)

CIRCUIT_BREAKER_STATE = Gauge(
    'nvd_circuit_breaker_state',
    'Circuit breaker state (0=closed, 1=half-open, 2=open)',
    ['upstream']
)

CACHE_TOKENS = Counter(
    'nvd_cache_tokens_total',
    'Cache token accounting',
    ['type'],  # 'creation' or 'read'
)

# Usage:
def observe_chunk_translation_latency(model: str, chunk_type: str, latency: float):
    STREAM_CHUNK_LATENCY.labels(model=model, chunk_type=chunk_type).observe(latency)
```

---

## Part IV: Implementation Phases

### Phase 1: Core Parity (Weeks 1-2)

**Goal:** Achieve P0 feature parity and security hardening.

| Task | Owner | Effort | Priority |
|------|-------|--------|----------|
| Add request validation layer | Senior Eng | 1 day | P0 |
| Implement SSRF protection | Senior Eng | 0.5 day | P0 |
| Add security headers middleware | Senior Eng | 0.5 day | P0 |
| Fix parallel tool result ordering | Senior Eng | 1 day | P1 |
| Test vision in streaming | QA | 1 day | P1 |
| Add circuit breaker | Senior Eng | 1 day | P1 |
| Implement load shedding | Senior Eng | 1 day | P1 |
| Add idempotency support | Senior Eng | 1 day | P2 |

**Deliverables:**
- `schemas/validators.py` - Strict Pydantic validation
- `middleware/security.py` - Security middleware stack
- `util/circuit_breaker.py` - Circuit breaker pattern
- All P0 gaps closed

---

### Phase 2: Feature Completion (Weeks 3-4)

**Goal:** Complete P1 features and advanced resilience.

| Task | Owner | Effort | Priority |
|------|-------|--------|----------|
| Batch API stubs (501) | Junior Eng | 0.5 day | P2 |
| Files API stubs (501) | Junior Eng | 0.5 day | P2 |
| Token-efficient tools | Senior Eng | 1 day | P2 |
| Cache accounting estimate | Senior Eng | 1 day | P2 |
| Enhanced rate limiting | Senior Eng | 1 day | P1 |
| Circuit breaker integration | Senior Eng | 1 day | P1 |
| Enhanced metrics | Senior Eng | 1 day | P1 |
| Integration tests | QA | 3 days | P1 |

**Deliverables:**
- `routes/batches.py` - Batch API with clear 501
- `routes/files.py` - Files API with clear 501
- `translators/tool_translator.py` - Token-efficient mode
- `middleware/rate_limiter.py` - Advanced token bucket
- `tests/integration/` - New integration test suite

---

### Phase 3: Production Hardening (Weeks 5-6)

**Goal:** Complete resilience patterns and observability.

| Task | Owner | Effort | Priority |
|------|-------|--------|----------|
| OpenAPI spec in Anthropic shape | Junior Eng | 1 day | P3 |
| Prometheus metrics expansion | Junior Eng | 1 day | P1 |
| Load shedding integration | Senior Eng | 1 day | P1 |
| Cost estimation in logs | Junior Eng | 0.5 day | P2 |
| SIGHUP handler testing | QA | 1 day | P2 |
| Chaos engineering tests | Senior Eng | 2 days | P3 |
| Performance benchmarks | QA | 2 days | P2 |
| Documentation update | Tech Writer | 1 day | P1 |

**Deliverables:**
- `routes/openapi.py` - Anthropic-shaped spec
- `middleware/load_shedding.py` - Load shedding
- `tests/chaos/` - Failure injection tests
- Performance benchmark report

---

## Part V: Testing Strategy

### V.A Unit Tests (Existing)

Continue maintaining current unit test coverage for:
- `test_stream_translator.py` - 6 test cases
- `test_request_translator.py` - Tool/schema handling
- `test_response_translator.py` - Non-stream path
- `test_tool_translator.py` - Name sanitization, collision detection
- `test_routes_anthropic_compat.py` - API compatibility

### V.B New Integration Tests (Required)

```python
# tests/integration/test_messages_endpoint.py
import pytest
from httpx import AsyncClient, ASGITransport
from nvd_claude_proxy.app import create_app

@pytest.fixture
async def client():
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c

async def test_stream_with_parallel_tools(client, respx_mock):
    """Test streaming with parallel tool calls maintains ordering."""
    # Mock NVIDIA API
    # Send request with 3 parallel tool calls
    # Verify each tool_result appears in correct order
    pass

async def test_validation_rejects_malformed(client):
    """Test that malformed requests are rejected with clear errors."""
    response = await client.post("/v1/messages", json={"invalid": "body"})
    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"

async def test_circuit_breaker_opens_on_failures(client, respx_mock):
    """Test circuit breaker trips after threshold failures."""
    # Mock 5 consecutive 500 errors
    # Verify subsequent requests get 503
    pass

async def test_ssrf_blocked_urls(client):
    """Test SSRF protection blocks dangerous URLs."""
    response = await client.post("/v1/messages", json={
        "model": "claude-opus-4-7",
        "messages": [{"role": "user", "content": [
            {"type": "image", "source": {"type": "url", "url": "file:///etc/passwd"}}
        ]}],
        "max_tokens": 100
    })
    assert response.status_code == 400
```

### V.C E2E Tests with Claude Code

Document test scenarios in `tests/e2e/test_with_claude_code.md`:

```markdown
## E2E Test Scenarios

### Scenario 1: Basic Text Request
- Claude Code → POST /v1/messages
- Verify streaming response
- Verify token counts

### Scenario 2: Tool Execution Loop
- Claude Code → tool_use
- Claude Code → tool_result
- Claude Code → tool_use (chain continues)
- Verify round-trip identity preservation

### Scenario 3: 273-Tool Session (Claude Code /init)
- Initialize with max tools
- Verify schema truncation
- Verify context overflow handling
- Verify all tools callable

### Scenario 4: Long Reasoning Session
- Request thinking enabled
- Verify >15s reasoning with ping events
- Verify reasoning block in response

### Scenario 5: Vision Request
- Send image in user message
- Verify response contains interpretation
```

---

## Part VI: Migration & Rollout

### VI.A Backward Compatibility

**Critical:** All changes must maintain backward compatibility with existing Claude Code sessions.

| Change | Compatibility Risk | Mitigation |
|--------|-------------------|------------|
| Request validation | Breaking for malformed requests | Opt-in initially, mandatory after deprecation |
| New required headers | None | Add optional headers only |
| Error message format | Breaking if clients parse errors | Maintain error structure |
| Circuit breaker | May return 529 instead of upstream error | Clear error message |

### VI.B Feature Flags

Implement feature flags for gradual rollout:

```python
# config/feature_flags.py
from pydantic import BaseModel

class FeatureFlags(BaseModel):
    request_validation: bool = False  # Enable strict validation
    circuit_breaker: bool = True      # Always on for resilience
    load_shedding: bool = True        # Always on for protection
    token_efficient_tools: bool = False  # New feature, off by default
    enhanced_rate_limiting: bool = False  # New feature, off by default

# Environment variable override
# FEATURE_FLAGS=request_validation,circuit_breaker
```

### VI.C Rollout Checklist

1. **Internal dogfood**: Run proxy against Claude Code internally for 1 week
2. **Staged rollout**: 10% → 25% → 50% → 100% of traffic
3. **Monitor dashboards**: Latency, error rates, Claude Code success rates
4. **Rollback procedure**: Feature flag toggle, no redeploy required

---

## Part VII: API Versioning & Lifecycle

### VII.A Anthropic API Versioning

**Current:** Hardcoded to `2023-06-01` in `anthropic_headers.py`.

**Required:** Dynamic versioning with backward compatibility:

```python
# routes/messages.py
SUPPORTED_API_VERSIONS = {
    "2023-06-01": {"min_version": "2023-06-01", "features": {...}},
    "2024-01-01": {"min_version": "2024-01-01", "features": {...}},  # Future
}

def get_api_version(request: Request) -> str:
    version = request.headers.get("anthropic-version", "2023-06-01")
    if version not in SUPPORTED_API_VERSIONS:
        return "2023-06-01"  # Fallback to oldest supported
    return version

def version_aware_response(version: str, data: dict) -> dict:
    """Return response formatted for specific API version."""
    if version >= "2024-01-01":
        # Include new fields
        return {**data, "server_tool_use": 0}  # New field
    return data  # Original format
```

### VII.B Deprecation Policy

| Deprecation | Timeline | Communication |
|-------------|----------|---------------|
| Old request format | 6 months notice | Release notes, warning headers |
| Error message format | 6 months notice | Release notes, warning headers |
| Endpoint removal | 12 months notice | Release notes, warning headers |

---

## Part VIII: Performance Targets

| Metric | Current | Target | Measurement |
|--------|---------|--------|-------------|
| P50 latency (text only) | Unknown | <100ms | Histogram |
| P95 latency (text only) | Unknown | <500ms | Histogram |
| P50 latency (streaming start) | Unknown | <200ms | First chunk |
| P95 latency (streaming start) | Unknown | <1s | First chunk |
| Throughput (req/s) | Unknown | >50 RPS | Load test |
| Memory per request | Unknown | <10MB | Memory profiler |
| CPU per request | Unknown | <50ms | CPU profiler |

---

## Part IX: Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| NVIDIA API changes break translation | Medium | High | Monitoring, rapid response |
| Claude Code behavior diverges | Low | High | Keep test suite current |
| Performance regression | Medium | Medium | Benchmark before/after |
| Security vulnerability discovered | Low | Critical | Regular security audits |
| Request validation breaks valid clients | Medium | Medium | Opt-in, gradual rollout |
| Circuit breaker too aggressive | Low | Medium | Tune thresholds |

---

## Appendix A: File Structure After Remediation

```
src/nvd_claude_proxy/
├── __init__.py
├── _version.py
├── main.py
├── app.py
├── cli/
│   └── main.py
├── clients/
│   ├── __init__.py
│   └── nvidia_client.py
├── config/
│   ├── __init__.py
│   ├── feature_flags.py       # NEW
│   ├── models.py
│   └── settings.py
├── data/
│   └── models.yaml
├── errors/
│   ├── __init__.py
│   └── mapper.py
├── middleware/
│   ├── __init__.py
│   ├── body_limit.py
│   ├── logging.py
│   ├── rate_limiter.py
│   ├── security.py            # NEW: SecurityHeadersMiddleware, SSRFProtection
│   └── load_shedding.py       # NEW: LoadSheddingMiddleware
├── routes/
│   ├── __init__.py
│   ├── batches.py             # NEW: Batch API (501 stubs)
│   ├── count_tokens.py
│   ├── files.py               # NEW: Files API (501 stubs)
│   ├── health.py
│   ├── messages.py
│   ├── metrics_route.py
│   ├── models.py
│   ├── openapi.py             # NEW: Anthropic-shaped OpenAPI
│   └── stubs.py
├── schemas/
│   ├── __init__.py
│   ├── anthropic.py           # Extended for all types
│   ├── canonical.py
│   ├── openai.py
│   └── validators.py          # NEW: Strict request validation
├── translators/
│   ├── __init__.py
│   ├── request_translator.py
│   ├── response_translator.py
│   ├── schema_sanitizer.py
│   ├── stream_translator.py   # Enhanced with image handling
│   ├── thinking_translator.py
│   ├── tool_controller.py
│   ├── tool_translator.py     # Enhanced with token-efficient mode
│   └── vision_translator.py
└── util/
    ├── __init__.py
    ├── anthropic_headers.py
    ├── circuit_breaker.py      # NEW
    ├── cost.py
    ├── ids.py
    ├── metrics.py              # Enhanced
    ├── pdf_extractor.py
    ├── sse.py
    └── tokens.py
```

---

## Appendix B: Quick Win Implementation Orders

### B.1 High-Impact, Low-Effort (Do First)

1. **Add request validation** (1 day effort)
   - Prevents downstream crashes
   - Clear error messages
   
2. **Add security headers** (0.5 day effort)
   - X-Content-Type-Options, X-Frame-Options, CSP
   
3. **Fix parallel tool result ordering** (1 day effort)
   - Guarantees Claude Code compatibility
   
4. **Enhance metrics with cache tokens** (0.5 day effort)
   - Better observability

### B.2 Medium-Effort, High-Impact

5. **Implement circuit breaker** (1 day)
   - Prevents cascading failures
   
6. **Add SSRF protection** (1 day)
   - Security critical
   
7. **Enhanced rate limiting** (2 days)
   - Token bucket per user
   
### B.3 Long-Term Investments

8. **Full integration test suite** (1 week)
9. **Performance optimization** (1 week)
10. **Chaos engineering** (1 week)

---

## Appendix C: Reference Specifications

- [Anthropic Messages API](https://docs.anthropic.com/en/api/messages)
- [Anthropic SDK Messages](https://github.com/anthropics/anthropic-sdk-python)
- [OpenAI Chat Completions](https://platform.openai.com/docs/api-reference/chat)
- [NVIDIA NIM API](https://docs.nvidia.com/nim/)
- [RFC 2119](https://www.rfc-editor.org/rfc/rfc2119) - Key words for requirements

---

*Document ends.*