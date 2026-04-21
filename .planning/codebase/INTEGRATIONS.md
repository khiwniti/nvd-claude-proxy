# External Integrations

**Analysis Date:** 2026-04-21

## APIs & External Services

**LLM Upstream:**
- NVIDIA NIM API (`https://integrate.api.nvidia.com/v1`) - upstream chat/model provider.
  - SDK/Client: `httpx` in `nvd-claude-proxy/src/nvd_claude_proxy/clients/nvidia_client.py`.
  - Auth: `NVIDIA_API_KEY` via settings in `nvd-claude-proxy/src/nvd_claude_proxy/config/settings.py`.
  - Coupling points:
    - Request adaptation: `nvd-claude-proxy/src/nvd_claude_proxy/translators/request_translator.py`
    - Response adaptation: `nvd-claude-proxy/src/nvd_claude_proxy/translators/response_translator.py`
    - Streaming adaptation: `nvd-claude-proxy/src/nvd_claude_proxy/translators/stream_translator.py`

**Anthropic-compatible Client Surface (consumer-facing protocol):**
- Anthropic Messages-compatible API exposed by proxy routes in `nvd-claude-proxy/src/nvd_claude_proxy/routes/messages.py` and `nvd-claude-proxy/src/nvd_claude_proxy/routes/count_tokens.py`.
  - SDK/Client: internal FastAPI routes (no separate external SDK dependency detected).
  - Auth: optional `PROXY_API_KEY` enforced in `_check_proxy_key` (`routes/messages.py`).
  - Coupling points: strict header and SSE compatibility utilities in `nvd-claude-proxy/src/nvd_claude_proxy/util/anthropic_headers.py` and `nvd-claude-proxy/src/nvd_claude_proxy/util/sse.py`.

## Data Storage

**Databases:**
- Not detected (no SQL/NoSQL client dependency or migration tooling found in `pyproject.toml`).

**File/Config-backed Data:**
- Model registry loaded from YAML (`nvd-claude-proxy/config/models.yaml`) through `nvd-claude-proxy/src/nvd_claude_proxy/config/models.py`.
  - Connection: `MODEL_CONFIG_PATH` env var in `nvd-claude-proxy/src/nvd_claude_proxy/config/settings.py`.
  - Client: `pyyaml` parser.

**Caching:**
- No external cache service detected (Redis/Memcached not detected).
- Internal in-process caching only: settings memoization via `@lru_cache` in `nvd-claude-proxy/src/nvd_claude_proxy/config/settings.py`.

## Authentication & Identity

**Auth Provider:**
- External identity provider: unknown/not used.
- Local API-key based auth:
  - Upstream NVIDIA bearer auth in `nvidia_client.py`.
  - Optional inbound proxy key check in `routes/messages.py`.

## Monitoring & Observability

**Error/Exception Tracking:**
- Dedicated external service (Sentry/etc.): unknown/not detected.

**Metrics:**
- Prometheus exposition endpoint at `/metrics` in `nvd-claude-proxy/src/nvd_claude_proxy/routes/metrics_route.py`.
  - Library: optional `prometheus-client` in `nvd-claude-proxy/src/nvd_claude_proxy/util/metrics.py`.

**Logs:**
- Structured JSON logs via `structlog` configured in `nvd-claude-proxy/src/nvd_claude_proxy/app.py`.

## CI/CD & Deployment

**Hosting/Runtime:**
- Containerized runtime supported via `nvd-claude-proxy/Dockerfile` and `nvd-claude-proxy/docker-compose.yml`.
- Non-container runtime supported via Python module execution (`nvd-claude-proxy/src/nvd_claude_proxy/main.py`).

**CI Pipeline:**
- GitHub Actions workflow in `nvd-claude-proxy/.github/workflows/ci-cd.yml` for lint, typecheck, test, and publish.
- Package publishing to PyPI via OIDC trusted publishing in same workflow.

## Environment Configuration

**Required env vars (critical):**
- `NVIDIA_API_KEY` (required) in `nvd-claude-proxy/src/nvd_claude_proxy/config/settings.py`.

**Optional env vars (integration-affecting):**
- `NVIDIA_BASE_URL`, `MODEL_CONFIG_PATH`, `PROXY_API_KEY`, `RATE_LIMIT_RPM`, `MAX_REQUEST_BODY_MB` (`settings.py` and `README.md`).

**Secrets location:**
- `.env` at project level and user config env path supported (`settings.py`); values intentionally not inspected.

## Webhooks & Callbacks

**Incoming webhooks:**
- Not detected (no webhook route pattern found).

**Outgoing callbacks/webhooks:**
- Not detected.

## Internal Module Integrations

**Route ↔ Translator ↔ Client pipeline (core coupling):**
- Entry route: `nvd-claude-proxy/src/nvd_claude_proxy/routes/messages.py`.
- Translation layer: `nvd-claude-proxy/src/nvd_claude_proxy/translators/*.py`.
- Upstream transport: `nvd-claude-proxy/src/nvd_claude_proxy/clients/nvidia_client.py`.
- Error mapping boundary: `nvd-claude-proxy/src/nvd_claude_proxy/errors/mapper.py`.

**Middleware ↔ App integration points:**
- Middleware registration in `nvd-claude-proxy/src/nvd_claude_proxy/app.py`:
  - `middleware/logging.py`
  - `middleware/rate_limiter.py`
  - `middleware/body_limit.py`

**Model capability coupling:**
- Runtime capability checks and model fallback chain use registry from `config/models.yaml` through `config/models.py` and `routes/messages.py`.

---

*Integration audit: 2026-04-21*
