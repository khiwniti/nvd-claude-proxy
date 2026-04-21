# Coding Conventions

**Analysis Date:** 2026-04-21

## Naming Patterns

**Files:**
- Python modules use `snake_case` (for example `src/nvd_claude_proxy/translators/request_translator.py`, `src/nvd_claude_proxy/routes/messages.py`).
- Test files use `test_*.py` under `tests/unit/` (for example `tests/unit/test_request_translator.py`).

**Functions:**
- Functions use `snake_case`; private helpers are `_prefixed` (for example `_flatten_system()` in `src/nvd_claude_proxy/translators/request_translator.py`, `_wait_for_proxy()` in `src/nvd_claude_proxy/cli/main.py`).

**Variables:**
- Locals and params are `snake_case`; module-level constants are `UPPER_CASE` (for example `_CONTEXT_HEADROOM`, `_MIN_OUTPUT` in `src/nvd_claude_proxy/translators/request_translator.py`).

**Types:**
- Type hints are used broadly (`dict[str, Any]`, return annotations, `Optional`) in source and tests (for example `src/nvd_claude_proxy/routes/messages.py`, `src/nvd_claude_proxy/cli/main.py`).

## Code Style

**Formatting:**
- Tool: Ruff formatter (`[tool.ruff]` in `pyproject.toml`).
- Key setting: line length `100`, target version `py311`.
- Canonical command: `make fmt` in `Makefile` (`ruff format src tests` + `ruff check --fix src tests`).

**Linting:**
- Tooling: Ruff + mypy via `make lint` in `Makefile`.
- CI enforces `ruff check src/ tests/` and `ruff format --check src/ tests/` in `.github/workflows/ci-cd.yml`.
- mypy runs in CI as `mypy src/nvd_claude_proxy` (`.github/workflows/ci-cd.yml`); config is non-strict (`strict = false`) in `pyproject.toml`.

## Import Organization

**Order:**
1. `from __future__ import annotations`
2. Standard library imports
3. Third-party imports
4. Local package imports

Pattern is consistent in `src/nvd_claude_proxy/routes/messages.py`, `src/nvd_claude_proxy/translators/request_translator.py`, and `tests/unit/test_stream_translator.py`.

**Path Aliases:**
- Not detected; standard Python package imports are used (for example `from ..util.tokens import approximate_tokens`).

## Error Handling

**Patterns:**
- Domain-specific exceptions are introduced and translated to API errors (for example `ContextOverflowError` in `src/nvd_claude_proxy/translators/request_translator.py`, caught in `src/nvd_claude_proxy/routes/messages.py`).
- FastAPI uses structured `HTTPException` payloads for Anthropic-compatible errors in `src/nvd_claude_proxy/routes/messages.py`.
- Non-fatal resilience pattern uses guarded imports and fallbacks (`try/except ImportError`) in `src/nvd_claude_proxy/main.py`.

## Logging

**Framework:** `structlog` in runtime code.

**Patterns:**
- Route-level structured events with stable keys (`messages.request`, `messages.complete`, `stream.complete`) in `src/nvd_claude_proxy/routes/messages.py`.
- CLI output uses `rich` consoles/panels instead of logging in `src/nvd_claude_proxy/cli/main.py`.

## Comments

**When to Comment:**
- Comments explain protocol compatibility, sizing heuristics, and failover behavior; they are used for intent-heavy logic rather than obvious lines (see `src/nvd_claude_proxy/translators/request_translator.py`, `src/nvd_claude_proxy/routes/messages.py`).

**JSDoc/TSDoc:**
- Not applicable (Python project). Python docstrings are used for modules/functions.

## Function Design

**Size:** 
- Core translators/routes include long functions with internal helper extraction (for example `translate_request()` in `src/nvd_claude_proxy/translators/request_translator.py`, `messages()` in `src/nvd_claude_proxy/routes/messages.py`).

**Parameters:**
- Prefer explicit typed parameters and dependency injection of collaborators (for example `translate_request(anthropic_body, spec, tool_id_map)`).

**Return Values:**
- Return concrete dict/list payloads matching upstream API contracts; tests assert exact response structure in `tests/unit/test_request_translator.py` and `tests/unit/test_routes_anthropic_compat.py`.

## Module Design

**Exports:**
- Explicit top-level entrypoints via scripts in `pyproject.toml`:
  - `nvd-claude-proxy = nvd_claude_proxy.main:run`
  - `ncp = nvd_claude_proxy.cli.main:main`

**Barrel Files:**
- Not detected.

## Repo Workflow (Detectable)

- Local workflow documented in `README.md`: `make dev`, `make test`, `make lint`, `make run`.
- CI workflow in `.github/workflows/ci-cd.yml` gates publish on `lint`, `typecheck`, and `test`.
- Branch patterns accepted by CI triggers: `main`, `claude/**`, `feat/**`, `fix/**` (`.github/workflows/ci-cd.yml`).

---

*Convention analysis: 2026-04-21*
