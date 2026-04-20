from __future__ import annotations

import asyncio
import time
from typing import AsyncIterator

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
from ..util.anthropic_headers import new_request_id, standard_response_headers
from ..util.cost import estimate_cost_usd
from ..util.metrics import inc_requests, inc_tokens, observe_duration
from ..util.tokens import approximate_tokens
from ..util.sse import encode_sse

router = APIRouter()
_log = structlog.get_logger("nvd_claude_proxy.messages")

# Cadence at which we emit SSE `ping` events during otherwise-silent periods
# (e.g. long `reasoning_content` generation on Nemotron Ultra). Matches
# Anthropic's ~15 s server-side heartbeat and keeps SDK idle-timeouts happy.
_PING_INTERVAL_SECONDS = 15.0


def _check_proxy_key(request: Request) -> None:
    s = request.app.state.settings
    if not s.proxy_api_key:
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
        (b.get("text") or "")
        for b in (resp.get("content") or [])
        if b.get("type") == "text"
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
        t["name"]: t.get("input_schema", {})
        for t in (body.get("tools") or [])
        if t.get("name")
    }


@router.post("/v1/messages")
async def messages(request: Request):
    _check_proxy_key(request)
    body = await request.json()

    registry = request.app.state.model_registry
    settings = request.app.state.settings
    requested_model = body.get("model") or registry.default_big
    spec_chain = registry.resolve_chain(requested_model)
    spec = spec_chain[0]
    tool_id_map = ToolIdMap()
    # Pass tool schemas so the controller can perform deterministic arg validation.
    tool_schemas = _build_tool_schemas(body)
    tool_controller = ToolInvocationController(spec, tool_id_map, tool_schemas=tool_schemas)
    try:
        payload = translate_request(body, spec, tool_id_map)
    except ContextOverflowError as exc:
        _log.warning(
            "messages.context_overflow",
            est_input=exc.est_input,
            max_context=exc.max_context,
            model=exc.model,
        )
        rid = new_request_id()
        return ORJSONResponse(
            {
                "type": "error",
                "error": {
                    "type": "invalid_request_error",
                    "message": str(exc),
                },
            },
            status_code=400,
            headers=standard_response_headers(rid),
        )
    stream = payload["stream"]
    est_input_tokens = approximate_tokens(
        {"messages": payload.get("messages", []), "tools": payload.get("tools", [])}
    )
    request_id = new_request_id()
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
                resp = await client.chat_completions(payload)
            except Exception as exc:
                if attempt_idx < len(spec_chain) - 1:
                    continue
                raise exc from None
            if (resp.status_code >= 500 or resp.status_code == 429) and attempt_idx < len(spec_chain) - 1:
                continue
            break
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
        out = translate_response(resp.json(), requested_model, tool_id_map)
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
            thinking_cfg.get("budget_tokens")
            if isinstance(thinking_cfg, dict)
            else None
        )
        active_tool_id_map = tool_id_map
        active_payload = payload

        try:
            for attempt_idx, try_spec in enumerate(spec_chain):
                if attempt_idx > 0:
                    active_tool_id_map = ToolIdMap()
                    active_payload = translate_request(body, try_spec, active_tool_id_map)
                    _log.warning(
                        "stream.failover",
                        attempt=attempt_idx + 1,
                        fallback_model=try_spec.alias,
                    )

                st = StreamTranslator(
                    model_name=requested_model,
                    tool_id_map=active_tool_id_map,
                    tool_controller=tool_controller,
                    budget_tokens=budget_tokens,
                    estimated_input_tokens=est_input_tokens,
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
                        yield encode_sse("ping", {"type": "ping"})
                        first_item = await upstream_queue.get()

                    # Failover status (5xx or 429) before first chunk.
                    if isinstance(first_item, tuple) and first_item[0] == "__error__":
                        exc = first_item[1]
                        is_failover_status = (
                            isinstance(exc, httpx.HTTPStatusError)
                            and (exc.response.status_code >= 500 or exc.response.status_code == 429)
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
                        for ev in st.feed(first_item):
                            yield encode_sse(ev["event"], ev["data"])

                    while not sentinel_seen:
                        try:
                            item = await asyncio.wait_for(
                                upstream_queue.get(), timeout=_PING_INTERVAL_SECONDS
                            )
                        except asyncio.TimeoutError:
                            yield encode_sse("ping", {"type": "ping"})
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
                            yield encode_sse("error", anth)
                            return
                        else:
                            for ev in st.feed(item):
                                yield encode_sse(ev["event"], ev["data"])

                    for ev in st.finalize():
                        yield encode_sse(ev["event"], ev["data"])
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
