"""Tests for P2/P3 features: failover chain, PDF extraction, budget_tokens,
disable_parallel_tool_use, stubs, SIGHUP reload."""
from __future__ import annotations

import base64
import signal
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from nvd_claude_proxy.app import create_app
from nvd_claude_proxy.config.models import ModelRegistry, CapabilityManifest, load_model_registry
from nvd_claude_proxy.translators.request_translator import translate_request
from nvd_claude_proxy.translators.stream_translator import StreamTranslator
from nvd_claude_proxy.translators.tool_translator import ToolIdMap
from nvd_claude_proxy.translators.tool_controller import ToolInvocationController
from nvd_claude_proxy.util.pdf_extractor import document_block_to_text


# ── CapabilityManifest failover_to ──────────────────────────────────────────────────────

def _make_registry() -> ModelRegistry:
    big = CapabilityManifest(
        alias="claude-opus-4-7",
        nvidia_id="nvidia/big",
        failover_to=["claude-sonnet-4-6", "claude-haiku-4-5"],
    )
    mid = CapabilityManifest(
        alias="claude-sonnet-4-6",
        nvidia_id="nvidia/mid",
        failover_to=["claude-haiku-4-5"],
    )
    small = CapabilityManifest(alias="claude-haiku-4-5", nvidia_id="nvidia/small")
    return ModelRegistry(
        specs={s.alias: s for s in [big, mid, small]},
        default_big="claude-opus-4-7",
        default_small="claude-haiku-4-5",
    )


def test_resolve_chain_full():
    reg = _make_registry()
    chain = reg.resolve_chain("claude-opus-4-7")
    assert [s.alias for s in chain] == [
        "claude-opus-4-7",
        "claude-sonnet-4-6",
        "claude-haiku-4-5",
    ]


def test_resolve_chain_no_failover():
    reg = _make_registry()
    chain = reg.resolve_chain("claude-haiku-4-5")
    assert len(chain) == 1
    assert chain[0].alias == "claude-haiku-4-5"


def test_resolve_chain_deduplicates():
    """If failover list contains the primary alias, it must be skipped."""
    spec = CapabilityManifest(
        alias="claude-opus-4-7",
        nvidia_id="nvidia/big",
        failover_to=["claude-opus-4-7", "claude-haiku-4-5"],
    )
    reg = ModelRegistry(
        specs={"claude-opus-4-7": spec, "claude-haiku-4-5": CapabilityManifest(alias="claude-haiku-4-5", nvidia_id="x")},
        default_big="claude-opus-4-7",
    )
    chain = reg.resolve_chain("claude-opus-4-7")
    assert [s.alias for s in chain] == ["claude-opus-4-7", "claude-haiku-4-5"]


def test_failover_to_unknown_alias_ignored():
    spec = CapabilityManifest(
        alias="claude-opus-4-7",
        nvidia_id="nvidia/big",
        failover_to=["does-not-exist"],
    )
    reg = ModelRegistry(specs={"claude-opus-4-7": spec}, default_big="claude-opus-4-7")
    chain = reg.resolve_chain("claude-opus-4-7")
    assert len(chain) == 1  # unknown alias silently dropped


# ── disable_parallel_tool_use ─────────────────────────────────────────────────

def _spec() -> CapabilityManifest:
    return CapabilityManifest(alias="claude-opus-4-7", nvidia_id="nvidia/big", supports_tools=True)


def test_disable_parallel_tool_use_sets_flag():
    body = {
        "model": "claude-opus-4-7",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 100,
        "tools": [{"name": "get_weather", "input_schema": {"type": "object", "properties": {}}}],
        "tool_choice": {"type": "auto", "disable_parallel_tool_use": True},
    }
    payload = translate_request(body, _spec(), ToolIdMap())
    assert payload.get("parallel_tool_calls") is False


def test_disable_parallel_tool_use_absent_by_default():
    body = {
        "model": "claude-opus-4-7",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 100,
        "tools": [{"name": "get_weather", "input_schema": {"type": "object", "properties": {}}}],
        "tool_choice": {"type": "auto"},
    }
    payload = translate_request(body, _spec(), ToolIdMap())
    assert "parallel_tool_calls" not in payload


# ── PDF document_block_to_text ────────────────────────────────────────────────

def test_document_plain_text_source():
    block = {"type": "document", "source": {"type": "text", "data": "Hello PDF"}}
    assert document_block_to_text(block) == "Hello PDF"


def test_document_with_title_and_context():
    block = {
        "type": "document",
        "title": "My Doc",
        "context": "Some hint",
        "source": {"type": "text", "data": "Content"},
    }
    result = document_block_to_text(block)
    assert "[Document: My Doc]" in result
    assert "[Context: Some hint]" in result
    assert "Content" in result


def test_document_url_source_returns_placeholder():
    block = {"source": {"type": "url", "url": "https://example.com/a.pdf"}}
    result = document_block_to_text(block)
    assert "https://example.com/a.pdf" in result


def test_document_base64_pdf_with_pypdf():
    """Test with a minimal valid PDF (pypdf may or may not extract text)."""
    pytest.importorskip("pypdf")
    # Minimal valid 1-page PDF with no text — should not raise.
    minimal_pdf = (
        b"%PDF-1.0\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj "
        b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj "
        b"3 0 obj<</Type/Page/MediaBox[0 0 3 3]>>endobj\n"
        b"xref\n0 4\n0000000000 65535 f \n0000000009 00000 n \n"
        b"0000000058 00000 n \n0000000115 00000 n \n"
        b"trailer<</Size 4/Root 1 0 R>>\nstartxref\n190\n%%EOF"
    )
    b64 = base64.b64encode(minimal_pdf).decode()
    block = {"source": {"type": "base64", "media_type": "application/pdf", "data": b64}}
    result = document_block_to_text(block)
    assert isinstance(result, str)  # no exception


def test_document_base64_pdf_without_pypdf():
    """When pypdf is absent the result is a placeholder, not an exception."""
    block = {
        "source": {
            "type": "base64",
            "media_type": "application/pdf",
            "data": base64.b64encode(b"fake").decode(),
        }
    }
    with patch("nvd_claude_proxy.util.pdf_extractor._PYPDF_AVAILABLE", False):
        result = document_block_to_text(block)
    assert "unavailable" in result or "pypdf" in result.lower()


def test_document_block_wired_into_request_translator():
    """document blocks must appear as text in the translated messages."""
    body = {
        "model": "claude-opus-4-7",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Summarise this:"},
                    {
                        "type": "document",
                        "source": {"type": "text", "data": "Annual revenue $1M"},
                    },
                ],
            }
        ],
        "max_tokens": 100,
    }
    payload = translate_request(body, _spec(), ToolIdMap())
    combined = payload["messages"][-1]["content"]
    assert "Annual revenue" in combined


# ── thinking.budget_tokens ────────────────────────────────────────────────────

def _run_stream(chunks: list[dict], budget_tokens: int | None = None) -> list[dict]:
    spec = CapabilityManifest(alias="claude-opus-4-7", nvidia_id="nvidia/big")
    tool_controller = ToolInvocationController(spec, ToolIdMap())
    st = StreamTranslator(
        model_name="claude-opus-4-7",
        tool_id_map=ToolIdMap(),
        tool_controller=tool_controller,
        budget_tokens=budget_tokens,
    )
    events = []
    for c in chunks:
        events.extend(st.feed(c))
    events.extend(st.finalize())
    return events


def _reasoning_chunk(text: str, finish: str | None = None) -> dict:
    choice: dict = {"delta": {"reasoning_content": text}, "finish_reason": finish}
    return {"choices": [choice]}


def test_budget_tokens_none_no_truncation():
    """Without a budget, all reasoning passes through."""
    chunk = _reasoning_chunk("A" * 400)
    events = _run_stream([chunk], budget_tokens=None)
    thinking_deltas = [
        e["data"]["delta"]["thinking"]
        for e in events
        if e["event"] == "content_block_delta"
        and e["data"]["delta"].get("type") == "thinking_delta"
    ]
    total = "".join(thinking_deltas)
    assert len(total) == 400


def test_budget_tokens_truncates_at_limit():
    """budget_tokens=10 means ≤40 chars of reasoning (10*4)."""
    chunk = _reasoning_chunk("B" * 200)
    events = _run_stream([chunk], budget_tokens=10)
    thinking_deltas = [
        e["data"]["delta"]["thinking"]
        for e in events
        if e["event"] == "content_block_delta"
        and e["data"]["delta"].get("type") == "thinking_delta"
    ]
    total = "".join(thinking_deltas)
    assert len(total) <= 40


def test_budget_tokens_stops_further_reasoning():
    """After budget hit, subsequent reasoning chunks are dropped."""
    chunks = [
        _reasoning_chunk("C" * 40),   # fills budget_tokens=10
        _reasoning_chunk("D" * 100),  # should be dropped
    ]
    events = _run_stream(chunks, budget_tokens=10)
    thinking_deltas = [
        e["data"]["delta"]["thinking"]
        for e in events
        if e["event"] == "content_block_delta"
        and e["data"]["delta"].get("type") == "thinking_delta"
    ]
    total = "".join(thinking_deltas)
    assert "D" not in total


# ── Batch / Files stubs ───────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def client():
    app = create_app()
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


def test_batch_create_returns_501(client):
    r = client.post("/v1/messages/batches", json={})
    assert r.status_code == 501
    assert r.json()["type"] == "error"


def test_batch_get_returns_501(client):
    r = client.get("/v1/messages/batches/batch_abc")
    assert r.status_code == 501


def test_batch_list_returns_501(client):
    r = client.get("/v1/messages/batches")
    assert r.status_code == 501


def test_files_upload_returns_501(client):
    r = client.post("/v1/files", content=b"data")
    assert r.status_code == 501


def test_files_get_returns_501(client):
    r = client.get("/v1/files/file_xyz")
    assert r.status_code == 501


def test_files_content_returns_501(client):
    r = client.get("/v1/files/file_xyz/content")
    assert r.status_code == 501


# ── SIGHUP hot reload ─────────────────────────────────────────────────────────

def test_sighup_handler_registered():
    """The SIGHUP signal handler must be the proxy's reload function."""
    if not hasattr(signal, "SIGHUP"):
        pytest.skip("SIGHUP not available on this platform")
    app = create_app()
    handler = signal.getsignal(signal.SIGHUP)
    assert callable(handler) and handler is not signal.SIG_DFL


def test_sighup_reload_updates_registry(tmp_path):
    """Simulating SIGHUP should replace app.state.model_registry."""
    if not hasattr(signal, "SIGHUP"):
        pytest.skip("SIGHUP not available on this platform")
    import yaml
    from nvd_claude_proxy.config.models import load_model_registry
    from nvd_claude_proxy.app import _install_sighup_handler
    from fastapi import FastAPI

    cfg = {
        "defaults": {"big": "model-a", "small": "model-a"},
        "aliases": {
            "model-a": {
                "nvidia_id": "vendor/model-a",
                "supports_tools": True,
                "supports_vision": False,
                "supports_reasoning": False,
            }
        },
        "prefix_fallbacks": {},
    }
    config_path = tmp_path / "models.yaml"
    config_path.write_text(yaml.dump(cfg))

    # Build a minimal fake app with a writable settings-like object.
    app = FastAPI()
    app.state.model_registry = load_model_registry(config_path)
    app.state.settings = MagicMock()
    app.state.settings.model_config_path = str(config_path)
    _install_sighup_handler(app)

    # Add a new alias and rewrite the config.
    cfg["aliases"]["model-b"] = {
        "nvidia_id": "vendor/model-b",
        "supports_tools": False,
        "supports_vision": False,
        "supports_reasoning": False,
    }
    config_path.write_text(yaml.dump(cfg))

    signal.raise_signal(signal.SIGHUP)

    assert "model-b" in app.state.model_registry.specs
