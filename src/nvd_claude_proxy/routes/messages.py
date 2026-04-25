from __future__ import annotations

import asyncio
import time
from typing import AsyncIterator, Callable, Any

import httpx
import structlog
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import ORJSONResponse, StreamingResponse

from ..clients.nvidia_client import NvidiaClient
from ..errors.mapper import openai_error_to_anthropic
from ..translators.request_translator import ContextOverflowError, translate_request
from ..translators.response_translator import translate_response
from ..translators.stream_translator import StreamTranslator
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


def _build_transformer_chain(spec: CapabilityManifest, on_fix: Callable[[str, Any], None] | None = None) -> TransformerChain:
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


def _check_proxy_key(request: Request) -> None:
    s = request.app.state.settings
    if not s.proxy_api_key:
        return

    # Trusted sessions must be authenticated. Freshly created sessions are
    # not authenticated until the key is checked once.
    session_obj = getattr(request.state, "session", None)
    if session_obj and getattr(session_obj, "authenticated", False):
        return

    presented = request.headers.get("x-api-key")
    if not presented:
        auth = request.headers.get("authorization", "")
        if auth.lower().startswith("bearer "):
            presented = auth[7:].strip()

    if presented != s.proxy_api_key:
        raise HTTPException(
            401,
            detail={
                "type": "error",
                "error": {
                    "type": "authentication_error",
                    "message": "invalid proxy api key",
                },
            },
        )

    # Mark session as authenticated for this process lifetime
    if session_obj:
        session_obj.authenticated = True


def _parse_beta_header(request: Request) -> list[str]:
    raw = request.headers.get("anthropic-beta", "")
    return [b.strip() for b in raw.split(",") if b.strip()]


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
    _check_proxy_key(request)
    body = await request.json()

    # ── Request Validation (Production Hardening) ─────────────────────────
    if _HAS_VALIDATORS:
        is_valid, result = validate_messages_request(body)
        if not is_valid:
            error_dict = result if isinstance(result, dict) else {}
            rid = new_request_id()
            _log.warning(
                "messages.validation_failed",
                request_id=rid,
                error=error_dict.get("error", {}).get("message", ""),
            )
            return ORJSONResponse(
                error_dict,
                status_code=400,
                headers=standard_response_headers(rid),
            )
        # Validation passed - use validated model for downstream processing
        from ..schemas.validators import MessagesRequest
        if isinstance(result, MessagesRequest):
            body = result.model_dump(exclude_none=True)
        _log.debug("messages.validated")

    registry = request.app.state.model_registry
    request_id = new_request_id()

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

    # Load isolated state from session if available
    session_obj = getattr(request.state, "session", None)
    tool_id_map = SessionService.get_isolated_tool_id_map(session_obj)
    transformer_chain = SessionService.get_isolated_transformer_chain(
        session_obj, spec, _build_transformer_chain, on_fix=on_fix
    )

    # Pass tool schemas so the controller can perform deterministic arg validation.
    tool_schemas = _build_tool_schemas(body)
    tool_controller = ToolInvocationController(spec, tool_id_map, tool_schemas=tool_schemas)

    try:
        payload = translate_request(body, spec, tool_id_map, transformer_chain=transformer_chain)
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
    est_input_tokens = approximate_tokens(
        {"messages": payload.get("messages", []), "tools": payload.get("tools", [])}
    )
    betas = _parse_beta_header(request)
    user_id = (body.get("metadata") or {}).get("user_id")

    std_headers = standard_response_headers(request_id)

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

    if not stream:
        # Non-streaming: walk the failover chain on 5xx / 429.
        t0 = time.monotonic()
        resp = None
        circuit_breaker = await get_circuit_breaker_registry().get_or_create("nvidia_api")

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
                    retry_after=circuit_breaker.config.timeout
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
                session_id=session_obj.id,
                tool_id_map=tool_id_map,
                transformer_chain=transformer_chain,
                tokens_inc=in_tok + out_tok,
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

        Failover: if the first item from the upstream is a 5xx HTTPStatusError
        AND there is a next spec in the chain, we cancel that attempt and
        restart with the fallback model.  Once any non-error chunk has been
        yielded we cannot failover (HTTP 200 headers are already sent).
        """
        thinking_cfg = body.get("thinking") or {}
        budget_tokens = (
            thinking_cfg.get("budget_tokens") if isinstance(thinking_cfg, dict) else None
        )
        active_tool_id_map = tool_id_map
        active_payload = payload
        active_tool_controller = tool_controller

        try:
            for attempt_idx, try_spec in enumerate(spec_chain):
                if attempt_idx > 0:
                    active_tool_id_map = ToolIdMap()
                    active_payload = translate_request(body, try_spec, active_tool_id_map)
                    active_tool_controller = ToolInvocationController(
                        try_spec, active_tool_id_map, tool_schemas=tool_schemas
                    )
                    _log.warning(
                        "stream.failover",
                        attempt=attempt_idx + 1,
                        fallback_model=try_spec.alias,
                    )

                st = StreamTranslator(
                    model_name=requested_model,
                    tool_id_map=active_tool_id_map,
                    tool_controller=active_tool_controller,
                    budget_tokens=budget_tokens,
                    estimated_input_tokens=est_input_tokens,
                    transformer_chain=transformer_chain,
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

                task = asyncio.create_task(pump())
                try:
                    # Peek at the first item to detect 5xx before any output.
                    try:
                        first_item: object = await asyncio.wait_for(
                            upstream_queue.get(), timeout=_PING_INTERVAL_SECONDS
                        )
                    except asyncio.TimeoutError:
                        for ev in st._emit("ping", {"type": "ping"}):
                            yield encode_sse(ev["event"], ev["data"])
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
                        yield encode_sse("error", anth)
                        return

                    # Normal path: process first_item then drain the queue.
                    sentinel_seen = first_item is SENTINEL
                    if not sentinel_seen:
                        if isinstance(first_item, dict):
                            if hasattr(request.app.state, "pubsub"):
                                await request.app.state.pubsub.broadcast(
                                    {
                                        "type": "openai_chunk",
                                        "payload": first_item,
                                        "request_id": request_id,
                                    }
                                )
                            for ev in st.feed(first_item):
                                if hasattr(request.app.state, "pubsub"):
                                    await request.app.state.pubsub.broadcast(
                                        {
                                            "type": "anthropic_event",
                                            "payload": ev,
                                            "request_id": request_id,
                                        }
                                    )
                                yield encode_sse(ev["event"], ev["data"])

                    while not sentinel_seen:
                        try:
                            item = await asyncio.wait_for(
                                upstream_queue.get(), timeout=_PING_INTERVAL_SECONDS
                            )
                        except asyncio.TimeoutError:
                            for ev in st._emit("ping", {"type": "ping"}):
                                yield encode_sse(ev["event"], ev["data"])
                            continue
                        if item is SENTINEL:
                            sentinel_seen = True
                        elif isinstance(item, tuple) and item and item[0] == "__error__":
                            exc = item[1]
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
                            if hasattr(request.app.state, "pubsub"):
                                await request.app.state.pubsub.broadcast(
                                    {"type": "error", "payload": anth, "request_id": request_id}
                                )
                            yield encode_sse("error", anth)
                            return
                        else:
                            if isinstance(item, dict):
                                if hasattr(request.app.state, "pubsub"):
                                    await request.app.state.pubsub.broadcast(
                                        {
                                            "type": "openai_chunk",
                                            "payload": item,
                                            "request_id": request_id,
                                        }
                                    )
                                for ev in st.feed(item):
                                    if hasattr(request.app.state, "pubsub"):
                                        await request.app.state.pubsub.broadcast(
                                            {
                                                "type": "anthropic_event",
                                                "payload": ev,
                                                "request_id": request_id,
                                            }
                                        )
                                    yield encode_sse(ev["event"], ev["data"])

                    # Finalize with double-close protection (port from claude-code-router)
                    try:
                        for ev in st.finalize():
                            yield encode_sse(ev["event"], ev["data"])
                    except Exception:
                        _log.debug("stream.finalize_error", exc_info=True)

                    # Persist session state (tool maps, transformer settings)
                    if session_obj:
                        await SessionService.save_session_state(
                            session_id=session_obj.id,
                            tool_id_map=active_tool_id_map,
                            transformer_chain=transformer_chain,
                            tokens_inc=st._usage_input + st._usage_output,
                        )

                    inc_requests(requested_model, True, 200)
                    inc_tokens(requested_model, st._usage_input, st._usage_output)
                    _log.info(
                        "stream.complete",
                        request_id=request_id,
                        input_tokens=st._usage_input,
                        output_tokens=st._usage_output,
                        cost_usd_est=round(
                            estimate_cost_usd(requested_model, st._usage_input, st._usage_output), 6
                        ),
                    )
                    return  # success — stop iterating the failover chain

                finally:
                    task.cancel()
                    try:
                        await task
                    except (asyncio.CancelledError, Exception):
                        pass

        finally:
            pass  # client is shared — do not close here

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
