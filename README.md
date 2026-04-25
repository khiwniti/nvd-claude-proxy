# nvd-claude-proxy

[![PyPI](https://img.shields.io/pypi/v/nvd-claude-proxy)](https://pypi.org/project/nvd-claude-proxy/)
[![Python](https://img.shields.io/pypi/pyversions/nvd-claude-proxy)](https://pypi.org/project/nvd-claude-proxy/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Run Claude Code — and any Anthropic SDK client — on free NVIDIA NIM models.**

A local HTTP proxy that speaks the [Anthropic Messages API](https://docs.anthropic.com/en/api/messages)
and forwards requests to `https://integrate.api.nvidia.com/v1` (NVIDIA NIM /
build.nvidia.com). Point your `ANTHROPIC_BASE_URL` at the proxy and your tools
work unchanged while inference runs on Nemotron Ultra, Qwen3, DeepSeek-R1,
or any other NIM model.

---

## Install

```sh
# Recommended: isolated global install
pipx install nvd-claude-proxy

# Or plain pip
pip install nvd-claude-proxy

# Optional extras
pip install "nvd-claude-proxy[metrics]"   # Prometheus /metrics endpoint
pip install "nvd-claude-proxy[pdf]"       # PDF document block extraction
pip install "nvd-claude-proxy[full]"      # everything above
```

---

## Quick start — `ncp` CLI  *(recommended)*

```sh
# First run: save your API key permanently
ncp init
# → prompts for NVIDIA_API_KEY and saves to ~/.config/nvd-claude-proxy/.env

# Launch proxy + Claude Code in one command
ncp code

# Launch the visual dashboard (New in v1.0!)
ncp dashboard
```

That's it. `ncp code` starts the proxy in the background, waits until it is
ready, then launches `claude`. When Claude exits the proxy stops cleanly.

### All `ncp` commands

| Command | Description |
|---|---|
| `ncp code` | Start proxy → launch Claude Code |
| `ncp proxy` | Start proxy only (foreground) |
| `ncp dashboard`| Launch the web-based management UI |
| `ncp init` | Save `NVIDIA_API_KEY` to global config |
| `ncp models list` | Show all configured model aliases |
| `ncp kill` | Terminate any stuck proxy process on port 8788 |
| `ncp version` | Print version |

Pass `--api-key nvapi-…` to any command for a one-shot override without saving.

---

## Force custom proxy URL

If you have trouble getting your Claude client to honor the `ANTHROPIC_BASE_URL` environment variable, use the provided wrapper script which forces the configuration:

```sh
# Use this wrapper instead of calling claude directly
./scripts/launch_claude.sh
```

You can add this to your shell profile to make it permanent:
```sh
# Add to ~/.bashrc or ~/.zshrc
alias claude='./path/to/nvd-claude-proxy/scripts/launch_claude.sh'
```

---

## Default model mapping

| Claude alias             | NVIDIA NIM model                              | Notes                   |
| ------------------------ | --------------------------------------------- | ----------------------- |
| `claude-opus-4-7`        | `nvidia/llama-3.1-nemotron-ultra-253b-v1`     | Reasoning, best quality |
| `claude-sonnet-4-6`      | `nvidia/llama-3.3-nemotron-super-49b-v1`      | Balanced                |
| `claude-haiku-4-5`       | `nvidia/nvidia-nemotron-nano-9b-v2`           | Fast, small             |
| `claude-opus-4-7-vision` | `meta/llama-4-maverick-17b-128e-instruct`     | Vision-capable          |
| `claude-qwen3`           | `qwen/qwen3-235b-a22b`                        | Qwen3 thinking          |
| `claude-r1`              | `deepseek-ai/deepseek-r1`                     | DeepSeek-R1             |

Legacy Claude 3.x names (`claude-3-5-sonnet-*`, `claude-3-opus-*`, etc.) are
automatically routed to the matching tier via prefix fallbacks.

Override by setting `MODEL_CONFIG_PATH=/path/to/your/models.yaml`.

---

## Environment variables

| Variable                  | Default                                 | Description                                          |
| ------------------------- | --------------------------------------- | ---------------------------------------------------- |
| `NVIDIA_API_KEY`          | **required**                            | `nvapi-…` key from [build.nvidia.com](https://build.nvidia.com) |
| `NVIDIA_BASE_URL`         | `https://integrate.api.nvidia.com/v1`   | Override for self-hosted NIM                         |
| `PROXY_HOST`              | `127.0.0.1`                             | Bind address (`0.0.0.0` for Docker/remote)           |
| `PROXY_PORT`              | `8788`                                  | Bind port                                            |
| `PROXY_API_KEY`           | *(unset)*                               | Require clients to present this key as Bearer token  |
| `LOG_LEVEL`               | `INFO`                                  | `DEBUG` / `INFO` / `WARNING` / `ERROR`               |
| `MODEL_CONFIG_PATH`       | *(bundled)*                             | Path to a custom `models.yaml`                       |
| `REQUEST_TIMEOUT_SECONDS` | `600`                                   | Total request timeout (long for reasoning streams)   |
| `MAX_RETRIES`             | `2`                                     | Upstream retry budget for transient 5xx              |
| `RATE_LIMIT_RPM`          | `0` *(off)*                             | Per-client sliding-window requests/minute; 0 = off   |
| `MAX_REQUEST_BODY_MB`     | `0` *(off)*                             | Reject bodies larger than this; 0 = unlimited        |

Variables can be placed in:
- `.env` in the current directory
- `~/.config/nvd-claude-proxy/.env` (written by `ncp init`)

---

## API endpoints

| Method | Path                        | Purpose                                        |
| ------ | --------------------------- | ---------------------------------------------- |
| `POST` | `/v1/messages`              | Anthropic Messages — streaming & non-streaming |
| `POST` | `/v1/messages/count_tokens` | Approximate token count (cl100k_base)          |
| `GET`  | `/v1/models`                | List model aliases                             |
| `GET`  | `/v1/models/{id}`           | Single model lookup                            |
| `GET`  | `/healthz`                  | Liveness probe                                 |
| `GET`  | `/metrics`                  | Prometheus metrics (`[metrics]` extra)         |
| `POST` | `/v1/messages/batches`      | 501 stub (not supported by NIM)                |
| `POST` | `/v1/files`                 | 501 stub                                       |

---

## Features

- **Full Anthropic SDK compatibility** — `anthropic-version` header, correct SSE `Content-Type`, proper `message_start` token counts.
- **Persistent Session Manager** — SQLite-backed isolation of tool states and conversation histories via `sk-ncp-*` keys.
- **Web Dashboard** — Modern dark-mode UI with **Live Monitor** dual-window stream visualization and model mapping.
- **Modular Transformer Pipeline** — Chain-of-responsibility pattern for model-specific fixes like **JSON Repair** and control-character stripping.
- **Agent Skills** — Native mapping of NVIDIA/OpenAI citations to Anthropic `web_search_tool_result` blocks.
- **Streaming** — strict Anthropic SSE event ordering with keepalive `ping` events every 15 s.
- **Tool use** — Parallel tool call buffering; schema validation and auto-sanitization for NIM models.
- **Reasoning / thinking** — `thinking.budget_tokens` enforced; `<think>` tags captured and extracted in real-time.
- **Vision** — JPEG/PNG pass-through; GIF/WEBP transcoded to PNG.
- **PDF documents** — base64 PDF blocks extracted to plain text (requires `[pdf]` extra).
- **Model failover** — automatic retry on 5xx with the next model in the configured chain.
- **Context overflow guard** — pre-flight check returns a clean 400 before the request reaches NVIDIA; automatic message truncation to salvage requests.
- **Shared connection pool** — one `httpx.AsyncClient` for all requests (no per-request TLS setup).
- **SIGHUP reload** — `kill -HUP <pid>` reloads `models.yaml` without restart.
- **Cost estimation** — `cost_usd_est` field in every structured log line and dashboard view.

---

## Custom model config

Create a `models.yaml` (start from the [bundled default](config/models.yaml)):

```yaml
defaults:
  big: "my-model"
  small: "my-model"

aliases:
  my-model:
    nvidia_id: "org/my-nim-model"
    supports_tools: true
    supports_vision: false
    supports_reasoning: false
    max_context: 131072
    max_output: 16384
    failover_to: []

prefix_fallbacks:
  "claude-": "my-model"
```

```sh
MODEL_CONFIG_PATH=./my_models.yaml nvd-claude-proxy
# or
ncp code --model-config ./my_models.yaml
```

---

## Docker

```sh
docker run --rm -p 8788:8788 \
  -e NVIDIA_API_KEY=nvapi-... \
  ghcr.io/khiwn/nvd-claude-proxy:latest
```

Or clone the repo and run:

```sh
cp .env.example .env      # fill in NVIDIA_API_KEY
docker compose up
```

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `NVIDIA_API_KEY Field required` | Run `ncp init` to save your key globally |
| `proxy did not start in time` | Run `ncp kill` then `ncp code` again |
| 132183 input tokens > 131072 | Context overflow — proxy now attempts automatic message truncation |
| 429 rate_limit_error | Hit NIM free-tier 40 RPM cap; wait 60 s or upgrade your NIM plan |
| Claude Code shows tool errors | Open `ncp dashboard` and check **Live Monitor** to see repair results |

---

## Known limitations

- **Prompt caching** is silently ignored (NVIDIA NIM has no equivalent).
- **`thinking.signature`** is proxy-local — do not forward proxy-generated thinking blocks to the real Anthropic API.
- **DeepSeek-R1 + tool use** is unreliable; use Nemotron models for agentic workloads.
- **Batch and Files APIs** return 501 — NIM has no equivalent.


---

## Development

```sh
git clone https://github.com/khiwn/nvd-claude-proxy
cd nvd-claude-proxy
cp .env.example .env      # fill in NVIDIA_API_KEY
make dev                  # pip install -e ".[dev,full]"
make test                 # pytest
make lint                 # ruff + mypy
make run                  # uvicorn on :8788
```

---

## License

MIT — see [LICENSE](LICENSE).
