# nvd-claude-proxy - Project Instructions

## Project Overview
`nvd-claude-proxy` is a local HTTP proxy that translates between the [Anthropic Messages API](https://docs.anthropic.com/en/api/messages) and the NVIDIA NIM / OpenAI-compatible API. It enables users to run Claude Code and other Anthropic SDK clients using NVIDIA-hosted models (like Nemotron, Qwen3, DeepSeek-R1) by simply pointing their `ANTHROPIC_BASE_URL` at this proxy.

### Core Technologies
- **Language:** Python 3.11+
- **Web Framework:** FastAPI
- **Asynchronous I/O:** httpx, uvloop
- **Data Validation:** Pydantic v2
- **Logging:** structlog (JSON formatted)
- **CLI:** Typer & Rich

### Architecture
- **Translation Layer (`src/nvd_claude_proxy/translators/`):** Handles complex mapping of streaming events, tool definitions, vision content, and reasoning blocks.
- **Model Registry (`src/nvd_claude_proxy/config/models.py`):** Resolves Claude model aliases (e.g., `claude-opus-4-7`) to NVIDIA NIM identifiers and capabilities. Supports automatic failover chains.
- **Streaming Logic (`src/nvd_claude_proxy/translators/stream_translator.py`):** Reconstructs Anthropic-style SSE streams from OpenAI-style chunks, including heartbeat support to prevent client timeouts during slow reasoning.

---

## Building and Running

### Development Setup
```bash
# Install with dev and full optional dependencies
make full
```

### Key Commands
- **Run Server:** `make run` (starts uvicorn on port 8788 by default)
- **Run CLI:** `ncp` (managed by `src/nvd_claude_proxy/cli/main.py`)
- **Test:** `make test` (runs pytest)
- **Lint:** `make lint` (ruff check + mypy)
- **Format:** `make fmt` (ruff format + ruff fix)

### Environment Configuration
The proxy loads configuration from `.env` files (local or `~/.config/nvd-claude-proxy/.env`).
- `NVIDIA_API_KEY`: Required for upstream authentication.
- `PROXY_PORT`: Default is 8788 (CLI) or 8787 (default settings).
- `MODEL_CONFIG_PATH`: Path to `models.yaml` defining aliases and capabilities.

---

## Development Conventions

### Coding Style
- **Formatting:** Handled by [Ruff](https://docs.astral.sh/ruff/). Line length is 100.
- **Imports:** Use `from __future__ import annotations` in all files.
- **Type Safety:** Use type hints throughout; checked via `mypy`.

### Testing
- **Framework:** Pytest with `asyncio` support.
- **Structure:**
  - `tests/unit/`: Component-level tests (translators, registry, etc.).
  - `tests/e2e/`: Higher-level integration/compatibility checks.
- **Mocks:** Uses `respx` and `pytest-httpx` for mocking upstream NVIDIA API calls.

### Logging & Metrics
- All logs are structured JSON via `structlog`.
- Key events (request start, completion, failover, errors) include `request_id` and cost estimation.
- If the `metrics` extra is installed, a `/metrics` endpoint provides Prometheus metrics for token usage and latency.

### Model Failover
If a model is configured with `failover_to`, the proxy will automatically retry 5xx errors using the next model in the list. This logic is implemented in `src/nvd_claude_proxy/routes/messages.py`.
