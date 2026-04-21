# Technology Stack

**Analysis Date:** 2026-04-21

## Languages

**Primary:**
- Python 3.11+ (`requires-python >=3.11`) - application/runtime code in `nvd-claude-proxy/src/nvd_claude_proxy/`.

**Secondary:**
- YAML - model/capability configuration in `nvd-claude-proxy/config/models.yaml`.
- Markdown - operational docs in `nvd-claude-proxy/README.md`, `nvd-claude-proxy/docs/ANTHROPIC_COMPAT.md`.

## Runtime

**Environment:**
- CPython (tested in CI on 3.11 and 3.12 via `.github/workflows/ci-cd.yml`; Docker image uses `python:3.12-slim` in `nvd-claude-proxy/Dockerfile`).

**Package Manager:**
- `pip` (install flow in `nvd-claude-proxy/Makefile` and CI).
- Lockfile: unknown/not detected (`requirements*.txt`, `poetry.lock`, `uv.lock` not detected at repo root or package root).

## Frameworks

**Core:**
- FastAPI (`fastapi>=0.115`) - HTTP API and routing in `nvd-claude-proxy/src/nvd_claude_proxy/app.py`, `nvd-claude-proxy/src/nvd_claude_proxy/routes/`.
- Uvicorn (`uvicorn[standard]>=0.32`) - ASGI server entrypoint via `nvd-claude-proxy/src/nvd_claude_proxy/main.py`.
- Pydantic + pydantic-settings - schema/env configuration in `nvd-claude-proxy/src/nvd_claude_proxy/config/settings.py`.

**Testing:**
- Pytest (+ `pytest-asyncio`, `pytest-httpx`, `respx`) from `nvd-claude-proxy/pyproject.toml`.

**Build/Dev:**
- Setuptools build backend in `nvd-claude-proxy/pyproject.toml`.
- Ruff + mypy for static quality (`nvd-claude-proxy/pyproject.toml`, `nvd-claude-proxy/Makefile`).
- Typer + Rich for CLI UX in `nvd-claude-proxy/src/nvd_claude_proxy/cli/main.py`.

## Key Dependencies

**Critical runtime:**
- `httpx` - outbound NVIDIA API transport in `nvd-claude-proxy/src/nvd_claude_proxy/clients/nvidia_client.py`.
- `structlog` - JSON structured logs in `nvd-claude-proxy/src/nvd_claude_proxy/app.py`.
- `orjson` - response serialization (`ORJSONResponse`) in `nvd-claude-proxy/src/nvd_claude_proxy/app.py`.
- `tiktoken` - token approximation in `nvd-claude-proxy/src/nvd_claude_proxy/util/tokens.py`.
- `Pillow` - image transcoding path referenced by vision translators and Docker system libs (`nvd-claude-proxy/Dockerfile`).

**Optional/runtime feature deps:**
- `prometheus-client` (`[metrics]` extra) used by `nvd-claude-proxy/src/nvd_claude_proxy/util/metrics.py`.
- `pypdf` (`[pdf]` extra) used by `nvd-claude-proxy/src/nvd_claude_proxy/util/pdf_extractor.py`.

## Configuration

**Environment:**
- Pydantic settings load from local `.env` and user config env files (`nvd-claude-proxy/src/nvd_claude_proxy/config/settings.py`).
- Runtime env vars documented in `nvd-claude-proxy/README.md` (e.g., `NVIDIA_API_KEY`, `NVIDIA_BASE_URL`, `MODEL_CONFIG_PATH`).

**Build:**
- Python packaging via `pyproject.toml` (`setuptools.build_meta`).
- Container build in `nvd-claude-proxy/Dockerfile`.
- Compose runtime in `nvd-claude-proxy/docker-compose.yml`.
- CI pipeline in `nvd-claude-proxy/.github/workflows/ci-cd.yml`.

## Platform Requirements

**Development:**
- Python 3.11+.
- Local env file present support (`nvd-claude-proxy/.env.example`); actual `.env` exists (contents intentionally not inspected).
- Optional Docker runtime via `docker compose`.

**Production:**
- HTTP service target exposing proxy API (FastAPI + Uvicorn).
- Deployment tooling detected: Docker image, Docker Compose, GitHub Actions CI/CD, PyPI publish (`nvd-claude-proxy/.github/workflows/ci-cd.yml`).

---

*Stack analysis: 2026-04-21*
