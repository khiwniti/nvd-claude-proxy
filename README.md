# nvd-claude-proxy

[![PyPI](https://img.shields.io/pypi/v/nvd-claude-proxy)](https://pypi.org/project/nvd-claude-proxy/)
[![Python](https://img.shields.io/pypi/pyversions/nvd-claude-proxy)](https://pypi.org/project/nvd-claude-proxy/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Run **Claude Code** on **NVIDIA NIM** models.

---

## Which path do you need?

There are two ways to run Claude Code against NVIDIA models. **Pick the one that matches your deployment.**

### ✅ Path A — Self-hosted NIM container (no proxy needed)

If you run a NIM container yourself (Docker, on-prem, NGC, k8s), it already exposes a **native Anthropic-compatible `/v1/messages` endpoint**. Claude Code talks to it directly — this proxy adds nothing.

Follow [NVIDIA's official guide: *Use Claude Code with NIM*](https://docs.nvidia.com/nim/large-language-models/latest/ai-assistant-integrations/claude-code.html):

```bash
export ANTHROPIC_BASE_URL="http://${NIM_HOST}:${NIM_PORT}"
export ANTHROPIC_API_KEY="not-used"           # NIM ignores it
export ANTHROPIC_DEFAULT_OPUS_MODEL="<your-nim-model-id>"
export ANTHROPIC_DEFAULT_SONNET_MODEL="<your-nim-model-id>"
export ANTHROPIC_DEFAULT_HAIKU_MODEL="<your-nim-model-id>"
export ANTHROPIC_CUSTOM_MODEL_OPTION="<your-nim-model-id>"
claude
```

That's the entire integration. Streaming, tool-use, thinking, count-tokens — all native.

### ⚙️ Path B — NVIDIA hosted API (this proxy)

If you can only hit NVIDIA's hosted endpoint at `https://integrate.api.nvidia.com/v1` with an `nvapi-…` key, that endpoint is **OpenAI-compatible only** — Claude Code cannot talk to it directly. This proxy translates Anthropic Messages API ↔ OpenAI Chat Completions on the wire.

```bash
pip install nvd-claude-proxy[full]
export NVIDIA_API_KEY=nvapi-...
ncp run                              # listens on :8788

# in another shell
export ANTHROPIC_BASE_URL=http://localhost:8788
export ANTHROPIC_API_KEY=not-used
claude
```

---

## Configuration (Path B)

| Environment variable | Default | Purpose |
|---|---|---|
| `NVIDIA_API_KEY` | *(required)* | Your `nvapi-…` key |
| `NVIDIA_BASE_URL` | `https://integrate.api.nvidia.com/v1` | Upstream OpenAI-compatible endpoint |
| `PROXY_PORT` | `8788` | Local listen port |
| `PROXY_API_KEY` | `None` | Optional key to protect the proxy itself |
| `STORAGE_ENGINE` | `sqlite` | `sqlite`, `redis`, or `memory` |
| `REDIS_URL` | `None` | Required when `STORAGE_ENGINE=redis` |
| `RATE_LIMIT_RPM` | `0` | Global RPM cap (`0` = disabled) |
| `MAX_REQUEST_BODY_MB` | `32` | Per-request body cap |
| `UPSTREAM_HTTP2` | `true` | HTTP/2 to upstream (graceful fallback to 1.1) |

Model aliases (`claude-opus-4-7`, `claude-sonnet-4-6`, `claude-haiku-4-5`) and routing chains are defined in [`config/models.yaml`](config/models.yaml). Override with `MODEL_CONFIG_PATH` or your own `custom_models.yaml`.

---

## What this proxy does (and doesn't)

**Does:**
- Translates Anthropic Messages API ↔ OpenAI Chat Completions in both directions, including streaming SSE.
- Pre-emits `message_start` synthetically so Claude Code's agent-state UI never sits on a blank screen.
- Streams `text_delta` / `thinking_delta` / `input_json_delta` faithfully, with a partial-tag holdback scanner for inline `<think>` reasoning.
- Maps OpenAI `finish_reason` → Anthropic `stop_reason` (`end_turn`, `max_tokens`, `tool_use`, `pause_turn`, `refusal`).
- Validates request shape, sanitises tool input schemas (`oneOf`/`anyOf`/`$ref` flattening for vLLM), enforces body-size and rate limits, maps errors to Anthropic's envelope.
- Provides `/v1/messages`, `/v1/messages/count_tokens`, `/v1/models`, `/healthz`, `/readyz`.

**Doesn't:**
- Add features NIM doesn't have. Prompt caching is reported as zero. Server-side tools (`web_search`, `code_execution`, computer use) work only if the upstream model supports them.
- Replace NIM's native Anthropic endpoint when you self-host. **Use Path A above** if you can — it's faster (one less hop) and more conformant (Anthropic spec implemented by vLLM directly).

---

## Development

```bash
git clone https://github.com/khiwniti/nvd-claude-proxy
cd nvd-claude-proxy
uv sync --extra dev
uv run --extra dev pytest tests/ -q
```

---

## License

MIT — see [LICENSE](LICENSE).
