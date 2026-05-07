from __future__ import annotations

import asyncio
import time
from typing import AsyncIterator, Callable, Any

import httpx
import structlog
from fastapi import APIRouter, Request
from fastapi.responses import ORJSONResponse, StreamingResponse

from ..clients.nvidia_client import NvidiaClient
from ..errors.mapper import openai_error_to_anthropic
from ..translators.request_translator import ContextOverflowError, translate_request
from ..translators.response_translator import translate_response
from ..core.events import StreamState
from ..core.pipeline import Pipeline
from ..core.processors import (
    MetadataProcessor,
    TextProcessor,
    ToolProcessor,
    SafetyProcessor,
    FinalizerProcessor,
)
from ..translators.tool_controller import ToolInvocationController
from ..translators.tool_translator import ToolIdMap
from ..translators.transformers import (
    CharFixerTransformer,
    ExitToolTransformer,
    JSONRepairTransformer,
    ReasoningTransformer,
    TransformerChain,
    WebSearchTransformer,
)
from ..util.anthropic_headers import new_request_id, standard_response_headers
from ..util.cost import estimate_cost_usd
from ..util.metrics import inc_requests, inc_tokens, observe_duration
from ..util.tokens import approximate_tokens
from ..util.router import get_use_model
from ..util.sse import encode_sse
from ..util.cache_accounting import estimate_cache_tokens, has_cache_control_markers
from ..util.circuit_breaker import get_circuit_breaker_registry, CircuitBreakerOpenError
from ..config.models import CapabilityManifest
from ..services.session_service import SessionService
from ..util.degradation import DegradationContext

# Import request validators for production hardening
try:
    from ..schemas.validators import validate_messages_request

    _HAS_VALIDATORS = True
except ImportError:
    _HAS_VALIDATORS = False

router = APIRouter()
_log = structlog.get_logger("nvd_claude_proxy.messages")

# Cadence at which we emit SSE `ping` events during otherwise-silent periods
# (e.g. long `reasoning_content` generation on Nemotron Ultra). Matches
# Anthropic's ~15 s server-side heartbeat and keeps SDK idle-timeouts happy.
_PING_INTERVAL_SECONDS = 15.0


def _build_transformer_chain(
    spec: CapabilityManifest, on_fix: Callable[[str, Any], None] | None = None
) -> TransformerChain:
    """Instantiate a TransformerChain based on model capabilities."""
    from ..translators.transformers import Transformer

    transformers: list[Transformer] = []
    # Always clean control chars
    transformers.append(CharFixerTransformer(on_fix=on_fix))

    if spec.supports_tools:
        transformers.append(JSONRepairTransformer(on_fix=on_fix))
        if spec.tools.exit_tool_enabled:
            transformers.append(ExitToolTransformer())

    if spec.supports_reasoning:
        transformers.append(ReasoningTransformer())

    # Always include web search for future-proofing
    transformers.append(WebSearchTransformer())

    return TransformerChain(transformers, on_fix=on_fix)


def _parse_beta_header(request: Request) -> list[str]:
    raw = request.headers.get("anthropic-beta", "")
    return [b.strip() for b in raw.split(",") if b.strip()]


# Bounded fan-out for the live-monitor pubsub: a slow subscriber must never
# delay a streamed token reaching the client. The semaphore caps in-flight
# broadcast tasks so a stuck subscriber cannot leak unbounded tasks either.
_PUBSUB_FANOUT_SEM = asyncio.Semaphore(64)


def _fanout_pubsub(request: Request, payload: dict) -> None:
    """Schedule a pubsub broadcast as a detached task; never await."""
    pubsub = getattr(request.app.state, "pubsub", None)
    if pubsub is None:
        return

    async def _go() -> None:
        async with _PUBSUB_FANOUT_SEM:
            try:
                await pubsub.broadcast(payload)
            except Exception as exc:  # noqa: BLE001
                _log.debug("pubsub.drop", err=str(exc))

    task = asyncio.create_task(_go())
    # Prevent "Task was destroyed but it is pending!" warnings on shutdown.
    task.add_done_callback(lambda t: t.exception() if not t.cancelled() else None)


def _echo_stop_sequence(anthropic_body: dict, resp: dict) -> None:
    """If the upstream stopped on text that matches a requested stop_sequence,
    set `stop_sequence` on the Anthropic response so SDKs can detect it.

    NVIDIA/OpenAI finish_reason="stop" doesn't reveal *which* sequence matched.
    We reconstruct by scanning the suffix of all text content — stop sequences
    are always at or near the end of the generated text (never mid-sentence).
    """
    seqs = anthropic_body.get("stop_sequences") or []
    if not seqs or resp.get("stop_reason") != "end_turn":
        return
    full_text = "".join(
        (b.get("text") or "") for b in (resp.get("content") or []) if b.get("type") == "text"
    )
    # Prefer the stop sequence that appears latest (closest to end of output).
    best_pos = -1
    best_seq = None
    for s in seqs:
        if not s:
            continue
        pos = full_text.rfind(s)
        if pos > best_pos:
            best_pos = pos
            best_seq = s
    if best_seq is not None:
        resp["stop_sequence"] = best_seq
        resp["stop_reason"] = "stop_sequence"


def _build_tool_schemas(body: dict) -> dict[str, dict]:
    """Build a name→schema map from the raw Anthropic request body for validation."""
    return {
        t["name"]: t.get("input_schema", {}) for t in (body.get("tools") or []) if t.get("name")
    }


@router.post("/v1/messages")
async def messages(request: Request):
    body = await request.json()
    request_id = new_request_id()

    idempotency_key = request.headers.get("anthropic-idempotency-key")
    req_hash = None
    if idempotency_key:
        import hashlib
        import json
        req_hash = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
        
        storage = getattr(request.app.state, "storage", None)
        if storage and hasattr(storage, "get_idempotency"):
            cached_data = await storage.get_idempotency(idempotency_key)
            if cached_data:
                if cached_data.get("req_hash") != req_hash:
                    _log.warning("messages.idempotency_mismatch", key=idempotency_key[:12] + "...")
                    return ORJSONResponse(
                        {
                            "type": "error",
                            "error": {
                                "type": "invalid_request_error",
                                "message": "Idempotency key mismatch: request body differs from original request.",
                            },
                        },
                        status_code=400,
                        headers=standard_response_headers(request_id),
                    )
                _log.info("messages.idempotent_replay", key=idempotency_key[:12] + "...")
                return ORJSONResponse(
                    cached_data.get("response", {}),
                    headers=standard_response_headers(new_request_id()),
                )

    # ── Phase 4: Anthropic Version/Beta Negotiation ──────────────────────────
    anthropic_version = request.headers.get("anthropic-version")
    if anthropic_version != "2023-06-01":
        _log.warning("messages.invalid_version", version=anthropic_version)
        return ORJSONResponse(
            {
                "type": "error",
                "error": {
                    "type": "invalid_request_error",
                    "message": f"Unsupported or missing anthropic-version: {anthropic_version}. Only '2023-06-01' is supported.",
                },
            },
            status_code=400,
            headers=standard_response_headers(request_id),
        )

    betas = _parse_beta_header(request)
    from ..util.beta_negotiator import BetaNegotiator

    # BetaNegotiator expects a set for set-difference operations; the parsed
    # CSV header arrives as a list.
    negotiator = BetaNegotiator(set(betas))
    try:
        negotiator.validate_request(body)
    except ValueError as exc:
        _log.warning("messages.beta_validation_failed", error=str(exc))
        return ORJSONResponse(
            {
                "type": "error",
                "error": {
                    "type": "invalid_request_error",
                    "message": str(exc),
                },
            },
            status_code=400,
            headers=standard_response_headers(request_id),
        )

    unsupported_betas = negotiator.get_unsupported()
    degradation = DegradationContext()
    for b in unsupported_betas:
        degradation.add_unsupported_beta(b)

    # ── Request Validation (Production Hardening) ─────────────────────────
    if _HAS_VALIDATORS:
        is_valid, result = validate_messages_request(body)
        if not is_valid:
            error_dict = result if isinstance(result, dict) else {}
            _log.warning(
                "messages.validation_failed",
                request_id=request_id,
                error=error_dict.get("error", {}).get("message", ""),
            )
            return ORJSONResponse(
                error_dict,
                status_code=400,
                headers=standard_response_headers(request_id),
            )
        # Validation passed - use validated model for downstream processing
        from ..schemas.validators import MessagesRequest

        if isinstance(result, MessagesRequest):
            body = result.model_dump(exclude_none=True)
        _log.debug("messages.validated")

    registry = request.app.state.model_registry

    def on_fix(fix_type: str, payload: Any) -> None:
        """Broadcast transformer fixes to the live monitor."""
        if hasattr(request.app.state, "pubsub"):
            asyncio.create_task(
                request.app.state.pubsub.broadcast(
                    {
                        "type": "transformer_fix",
                        "fix_type": fix_type,
                        "payload": payload,
                        "request_id": request_id,
                    }
                )
            )

    # Phase 3: Scenario-based routing (port from claude-code-router)
    # Estimate tokens of the Anthropic request body to inform routing.
    rough_input_tokens = approximate_tokens(body)
    requested_model = get_use_model(body, rough_input_tokens, registry)

    spec_chain = registry.resolve_chain(requested_model)
    spec = spec_chain[0]

    # ── Phase 4: Post-Routing Capability Validation ─────────────────────────
    requires_tools = bool(body.get("tools"))
    requires_vision = any(
        b.get("type") == "image"
        for m in body.get("messages", [])
        for b in (m.get("content") or [])
        if isinstance(b, dict)
    )
    if requires_tools and not spec.supports_tools:
        degradation.provider_capability_mismatch.append("tools")
        return ORJSONResponse(
            {
                "type": "error",
                "error": {
                    "type": "invalid_request_error",
                    "message": f"Routed model {spec.alias} does not support tools.",
                },
            },
            status_code=400,
            headers=standard_response_headers(request_id),
        )
    if requires_vision and not getattr(spec, "supports_vision", True):
        degradation.provider_capability_mismatch.append("vision")
        return ORJSONResponse(
            {
                "type": "error",
                "error": {
                    "type": "invalid_request_error",
                    "message": f"Routed model {spec.alias} does not support vision.",
                },
            },
            status_code=400,
            headers=standard_response_headers(request_id),
        )

    # Load isolated state from session if available
    session_obj = getattr(request.state, "session", None)
    tool_id_map = await SessionService.get_isolated_tool_id_map(request, session_obj)
    transformer_chain = await SessionService.get_isolated_transformer_chain(
        request, session_obj, spec, _build_transformer_chain, on_fix=on_fix
    )

    # Pass tool schemas so the controller can perform deterministic arg validation.
    tool_schemas = _build_tool_schemas(body)
    tool_controller = ToolInvocationController(spec, tool_id_map, tool_schemas=tool_schemas)

    try:
        payload = translate_request(
            body, 
            spec, 
            tool_id_map, 
            transformer_chain=transformer_chain,
            server_tool_registry=getattr(request.app.state, "server_tool_registry", None)
        )
    except ContextOverflowError as exc:
        _log.warning(
            "messages.context_overflow",
            est_input=exc.est_input,
            max_context=exc.max_context,
            model=exc.model,
        )
        return ORJSONResponse(
            {
                "type": "error",
                "error": {
                    "type": "invalid_request_error",
                    "message": str(exc),
                },
            },
            status_code=400,
            headers=standard_response_headers(request_id),
        )
    stream = payload["stream"]
    # `rough_input_tokens` (computed once at routing time) is the value
    # exposed in `message_start.usage.input_tokens` per Anthropic spec.
    # The earlier double-tokenisation here added 20-50ms of TTFT for zero
    # observable benefit — output_tokens is always 0 at message_start.
    est_input_tokens = rough_input_tokens
    betas = _parse_beta_header(request)
    user_id = (body.get("metadata") or {}).get("user_id")

    std_headers = standard_response_headers(request_id)

    if degradation.has_degradation():
        _log.warning("messages.degraded", request_id=request_id, degradation=degradation.to_dict())

    _log.info(
        "messages.request",
        request_id=request_id,
        claude_model=requested_model,
        nvidia_id=spec.nvidia_id,
        failover_chain=[s.alias for s in spec_chain[1:]],
        stream=stream,
        tool_count=len(payload.get("tools") or []),
        effective_max_tokens=payload.get("max_tokens"),
        input_tokens_requested=body.get("max_tokens"),
        context_window=spec.max_context,
        betas=betas,
        user_id=user_id,
    )

    client: NvidiaClient = request.app.state.nvidia_client
    circuit_breaker = await get_circuit_breaker_registry().get_or_create("nvidia_api")

    if not stream:
        # Non-streaming: walk the failover chain on 5xx / 429.
        t0 = time.monotonic()
        resp = None

        for attempt_idx, try_spec in enumerate(spec_chain):
            if attempt_idx > 0:
                tool_id_map = ToolIdMap()
                payload = translate_request(body, try_spec, tool_id_map)
                _log.warning(
                    "messages.failover",
                    attempt=attempt_idx + 1,
                    fallback_model=try_spec.alias,
                    nvidia_id=try_spec.nvidia_id,
                )
            try:
                # Use circuit breaker to protect against cascading failures
                resp = await circuit_breaker.call(lambda p=payload: client.chat_completions(p))
            except CircuitBreakerOpenError:
                rid = new_request_id()
                _log.error(
                    "messages.circuit_breaker_open",
                    request_id=rid,
                    upstream="nvidia_api",
                    retry_after=circuit_breaker.config.recovery_timeout
                    if hasattr(circuit_breaker, "config")
                    else 30,
                )
                err_headers = standard_response_headers(rid)
                return ORJSONResponse(
                    {
                        "type": "error",
                        "error": {
                            "type": "api_error",
                            "message": "Upstream API temporarily unavailable. Please retry.",
                        },
                    },
                    status_code=503,
                    headers=err_headers,
                )
            except httpx.HTTPStatusError as exc:
                if attempt_idx < len(spec_chain) - 1:
                    continue
                raise exc from None
            except Exception as exc:
                if attempt_idx < len(spec_chain) - 1:
                    continue
                raise exc from None
            if (resp.status_code >= 500 or resp.status_code == 429) and attempt_idx < len(
                spec_chain
            ) - 1:
                continue
            break
        if resp is None:
            raise RuntimeError("No upstream response received from NVIDIA API")
        elapsed = time.monotonic() - t0
        observe_duration(requested_model, elapsed)
        if resp.status_code >= 400:
            try:
                err_body = resp.json() if resp.content else {}
            except Exception:
                err_body = {"message": resp.text}
            status, anth = openai_error_to_anthropic(resp.status_code, err_body)
            inc_requests(requested_model, False, status)
            err_headers = dict(std_headers)
            if status == 429 and (ra := resp.headers.get("retry-after")):
                err_headers["retry-after"] = ra
            return ORJSONResponse(anth, status_code=status, headers=err_headers)

        out = translate_response(
            resp.json(),
            requested_model,
            tool_id_map,
            tool_controller=tool_controller,
            transformer_chain=transformer_chain,
        )
        # Wire cache accounting for cost tracking
        if has_cache_control_markers(body):
            acct = estimate_cache_tokens(body)
            usage = out.setdefault("usage", {})
            usage["cache_creation_input_tokens"] = acct.cache_creation_input_tokens
            usage["cache_read_input_tokens"] = acct.cache_read_input_tokens

        _echo_stop_sequence(body, out)
        # Validate tool args from non-streaming response against declared schemas.
        failing_tools = tool_controller.validate_all(
            [b for b in (out.get("content") or []) if b.get("type") == "tool_use"]
        )
        if failing_tools:
            _log.warning("messages.tool_arg_validation_failed", tools=failing_tools)
        usage = out.get("usage") or {}
        in_tok = usage.get("input_tokens", 0)
        out_tok = usage.get("output_tokens", 0)
        inc_tokens(requested_model, in_tok, out_tok)
        inc_requests(requested_model, False, 200)

        # Persist session state (tool maps, transformer settings)
        if session_obj:
            await SessionService.save_session_state(
                request=request,
                session_id=session_obj.id,
                tool_id_map=tool_id_map,
                transformer_chain=transformer_chain,
                tokens_inc=in_tok + out_tok,
            )

        if idempotency_key and req_hash:
            storage = getattr(request.app.state, "storage", None)
            if storage and hasattr(storage, "save_idempotency"):
                await storage.save_idempotency(
                    idempotency_key, {"req_hash": req_hash, "response": out}
                )

        _log.info(
            "messages.complete",
            request_id=request_id,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cost_usd_est=round(estimate_cost_usd(requested_model, in_tok, out_tok), 6),
            elapsed_s=round(elapsed, 2),
        )
        return ORJSONResponse(out, headers=std_headers)

    async def gen() -> AsyncIterator[bytes]:
        """Multiplex upstream chunks with periodic ping events so SDKs never
        idle-timeout on a slow reasoning stream.

        TTFT optimisation: emit a synthesised `message_start` SSE frame as
        the very first wire byte — before the upstream POST is even
        initiated. Anthropic's spec defines `message_start` as a synchronous
        echo of the request envelope, so emitting it pre-flight is
        spec-conformant and slashes perceived TTFT for the agent-state UI.

        Failover: if the first item from the upstream is a 5xx HTTPStatusError
        AND there is a next spec in the chain, we cancel that attempt and
        restart with the fallback model.  Once any non-error chunk has been
        yielded we cannot failover (HTTP 200 headers are already sent), but
        `message_start` itself is identical across all chain members so
        emitting it early is safe.
        """
        thinking_cfg = body.get("thinking") or {}
        budget_tokens = (
            thinking_cfg.get("budget_tokens") if isinstance(thinking_cfg, dict) else None
        )
        active_tool_id_map = tool_id_map
        active_payload = payload
        active_tool_controller = tool_controller

        # Mint a stable message_id ONCE so the synthesised message_start and
        # all subsequent events agree, even across failover attempts.
        stable_message_id = new_request_id().replace("req_", "msg_")
        synth_event_id = 1

        # ── Phase A: synthesised message_start — FIRST WIRE BYTE ──────────
        msg_start_payload = {
            "type": "message_start",
            "message": {
                "id": stable_message_id,
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": requested_model,
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {
                    "input_tokens": est_input_tokens,
                    "output_tokens": 0,
                },
            },
        }
        yield encode_sse("message_start", msg_start_payload, event_id=str(synth_event_id))
        synth_event_id += 1

        try:
            for attempt_idx, try_spec in enumerate(spec_chain):
                if attempt_idx > 0:
                    active_tool_id_map = ToolIdMap()
                    active_payload = translate_request(
                        body, 
                        try_spec, 
                        active_tool_id_map,
                        server_tool_registry=getattr(request.app.state, "server_tool_registry", None)
                    )
                    active_tool_controller = ToolInvocationController(
                        try_spec, active_tool_id_map, tool_schemas=tool_schemas
                    )
                    _log.warning(
                        "stream.failover",
                        attempt=attempt_idx + 1,
                        fallback_model=try_spec.alias,
                    )

                state = StreamState(
                    message_id=stable_message_id,
                    model_name=requested_model,
                    budget_tokens=budget_tokens,
                    estimated_input_tokens=est_input_tokens,
                )
                # Tell the pipeline that message_start has already been
                # emitted on the wire so MetadataProcessor._ensure_started
                # is a no-op for this stream.
                state.started = True
                if has_cache_control_markers(body):
                    acct = estimate_cache_tokens(body)
                    state.cache_creation_input_tokens = acct.cache_creation_input_tokens
                    state.cache_read_input_tokens = acct.cache_read_input_tokens
                    state.ephemeral_5m_input_tokens = acct.ephemeral_5m_input_tokens
                    state.ephemeral_1h_input_tokens = acct.ephemeral_1h_input_tokens

                pipeline = Pipeline(
                    processors=[
                        MetadataProcessor(),
                        TextProcessor(),
                        ToolProcessor(active_tool_id_map, active_tool_controller),
                        SafetyProcessor(),
                        FinalizerProcessor(),
                    ],
                    state=state,
                )
                upstream_queue: asyncio.Queue = asyncio.Queue(maxsize=256)
                SENTINEL = object()

                async def pump(p=active_payload) -> None:
                    try:
                        async for chunk in client.astream_chat_completions(p):
                            await upstream_queue.put(chunk)
                    except Exception as exc:  # noqa: BLE001
                        await upstream_queue.put(("__error__", exc))
                    finally:
                        await upstream_queue.put(SENTINEL)

                async def _ping_scheduler() -> None:
                    try:
                        while True:
                            await asyncio.sleep(10.0)
                            await upstream_queue.put(("__ping__", None))
                    except asyncio.CancelledError:
                        pass

                task = asyncio.create_task(pump())
                ping_task = asyncio.create_task(_ping_scheduler())
                try:
                    def _encode_event(event_type: str, data: dict) -> bytes:
                        nonlocal synth_event_id
                        if event_type == "ping":
                            return encode_sse(event_type, data)
                        ev_id = str(synth_event_id)
                        synth_event_id += 1
                        return encode_sse(event_type, data, event_id=ev_id)

                    # Peek at the first item to detect 5xx before any output.
                    try:
                        first_item: object = await asyncio.wait_for(
                            upstream_queue.get(), timeout=_PING_INTERVAL_SECONDS
                        )
                    except asyncio.TimeoutError:
                        yield _encode_event("ping", {"type": "ping"})
                        first_item = await upstream_queue.get()

                    # Failover status (5xx or 429) before first chunk.
                    if isinstance(first_item, tuple) and first_item[0] == "__error__":
                        exc = first_item[1]
                        is_failover_status = isinstance(exc, httpx.HTTPStatusError) and (
                            exc.response.status_code >= 500 or exc.response.status_code == 429
                        )
                        if is_failover_status and attempt_idx < len(spec_chain) - 1:
                            try:
                                await asyncio.wait_for(upstream_queue.get(), timeout=5.0)
                            except asyncio.TimeoutError:
                                pass
                            continue  # outer for-loop

                        # Final attempt or non-retryable error — emit error SSE.
                        if isinstance(exc, httpx.HTTPStatusError):
                            try:
                                upstream_body = exc.response.json()
                            except Exception:
                                upstream_body = {"message": exc.response.text}
                            _, anth = openai_error_to_anthropic(
                                exc.response.status_code, upstream_body
                            )
                            # Propagate retry-after from NVIDIA 429 to client.
                            if exc.response.status_code == 429:
                                if ra := exc.response.headers.get("retry-after"):
                                    anth.setdefault("error", {})["retry_after"] = ra
                        else:
                            _log.exception("stream.unhandled", err=str(exc))
                            anth = {
                                "type": "error",
                                "error": {"type": "api_error", "message": str(exc)},
                            }
                        yield _encode_event("error", anth)
                        return

                    # Normal path: process first_item then drain the queue.
                    sentinel_seen = first_item is SENTINEL
                    if not sentinel_seen:
                        if isinstance(first_item, dict):
                            _fanout_pubsub(
                                request,
                                {
                                    "type": "openai_chunk",
                                    "payload": first_item,
                                    "request_id": request_id,
                                },
                            )
                            for ev in pipeline.feed(first_item):
                                _fanout_pubsub(
                                    request,
                                    {
                                        "type": "anthropic_event",
                                        "payload": {"event": ev.event, "data": ev.data},
                                        "request_id": request_id,
                                    },
                                )
                                yield _encode_event(ev.event, ev.data)

                    while not sentinel_seen:
                        try:
                            item = await asyncio.wait_for(
                                upstream_queue.get(), timeout=_PING_INTERVAL_SECONDS
                            )
                        except asyncio.TimeoutError:
                            yield _encode_event("ping", {"type": "ping"})
                            continue
                        if item is SENTINEL:
                            sentinel_seen = True
                        elif isinstance(item, tuple) and item and item[0] == "__ping__":
                            yield _encode_event("ping", {"type": "ping"})
                        elif isinstance(item, tuple) and item and item[0] == "__error__":
                            exc = item[1]
                            is_transient = isinstance(
                                exc,
                                (
                                    httpx.ReadError,
                                    httpx.ReadTimeout,
                                    httpx.LocalProtocolError,
                                    httpx.RemoteProtocolError,
                                ),
                            )
                            if is_transient or (
                                isinstance(exc, httpx.HTTPStatusError)
                                and exc.response.status_code >= 500
                            ):
                                await circuit_breaker.record_failure()

                            if isinstance(exc, httpx.HTTPStatusError):
                                try:
                                    upstream_body = exc.response.json()
                                except Exception:
                                    upstream_body = {"message": exc.response.text}
                                _, anth = openai_error_to_anthropic(
                                    exc.response.status_code, upstream_body
                                )
                                if exc.response.status_code == 429:
                                    if ra := exc.response.headers.get("retry-after"):
                                        anth.setdefault("error", {})["retry_after"] = ra
                            else:
                                _log.exception("stream.unhandled", err=str(exc))
                                anth = {
                                    "type": "error",
                                    "error": {"type": "api_error", "message": str(exc)},
                                }
                            _fanout_pubsub(
                                request,
                                {"type": "error", "payload": anth, "request_id": request_id},
                            )
                            yield _encode_event("error", anth)
                            return
                        else:
                            if isinstance(item, dict):
                                _fanout_pubsub(
                                    request,
                                    {
                                        "type": "openai_chunk",
                                        "payload": item,
                                        "request_id": request_id,
                                    },
                                )
                                for ev in pipeline.feed(item):
                                    _fanout_pubsub(
                                        request,
                                        {
                                            "type": "anthropic_event",
                                            "payload": {"event": ev.event, "data": ev.data},
                                            "request_id": request_id,
                                        },
                                    )
                                    yield _encode_event(ev.event, ev.data)

                    # Finalize with double-close protection (port from claude-code-router)
                    try:
                        for ev in pipeline.finalize():
                            yield _encode_event(ev.event, ev.data)
                    except Exception:
                        _log.debug("stream.finalize_error", exc_info=True)

                    # Persist session state (tool maps, transformer settings)
                    if session_obj:
                        await SessionService.save_session_state(
                            request=request,
                            session_id=session_obj.id,
                            tool_id_map=active_tool_id_map,
                            transformer_chain=transformer_chain,
                            tokens_inc=state.usage_input + state.usage_output,
                        )

                    inc_requests(requested_model, True, 200)
                    inc_tokens(requested_model, state.usage_input, state.usage_output)
                    _log.info(
                        "stream.complete",
                        request_id=request_id,
                        input_tokens=state.usage_input,
                        output_tokens=state.usage_output,
                        cost_usd_est=round(
                            estimate_cost_usd(
                                requested_model, state.usage_input, state.usage_output
                            ),
                            6,
                        ),
                    )
                    return  # success — stop iterating the failover chain

                finally:
                    if not sentinel_seen:
                        _log.warning("stream.cancelled", request_id=request_id)
                        from ..util.metrics import COUNTER_REQUESTS
                        if COUNTER_REQUESTS:
                            COUNTER_REQUESTS.labels(model=requested_model, stream="true", status="cancelled").inc()
                    task.cancel()
                    ping_task.cancel()
                    try:
                        await task
                        await ping_task
                    except (asyncio.CancelledError, Exception):
                        pass

        finally:
            pass  # client is shared — do not close here

    try:
        await circuit_breaker._before_call()
    except CircuitBreakerOpenError:
        rid = new_request_id()
        _log.error(
            "messages.circuit_breaker_open",
            request_id=rid,
            upstream="nvidia_api",
            retry_after=circuit_breaker.config.recovery_timeout
            if hasattr(circuit_breaker, "config")
            else 30,
        )
        err_headers = standard_response_headers(rid)
        return ORJSONResponse(
            {
                "type": "error",
                "error": {
                    "type": "api_error",
                    "message": "Upstream API temporarily unavailable. Please retry.",
                },
            },
            status_code=503,
            headers=err_headers,
        )

    return StreamingResponse(
        gen(),
        media_type="text/event-stream; charset=utf-8",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "Transfer-Encoding": "chunked",
            **std_headers,
        },
    )
