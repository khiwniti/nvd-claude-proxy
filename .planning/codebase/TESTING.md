# Testing Patterns

**Analysis Date:** 2026-04-21

## Test Framework

**Runner:**
- `pytest` (declared in `pyproject.toml` optional dev dependencies).
- Config: `pyproject.toml` under `[tool.pytest.ini_options]` (`asyncio_mode = "auto"`, `testpaths = ["tests"]`).

**Assertion Library:**
- Built-in `assert` statements in pytest tests (for example `tests/unit/test_request_translator.py`, `tests/unit/test_routes_anthropic_compat.py`).

**Run Commands:**
```bash
make test                 # pytest -q
pytest tests/unit -v --tb=short
make lint                 # quality gate used with test workflow
```

## Test File Organization

**Location:**
- Unit tests are grouped in `tests/unit/`.
- Shared test bootstrap fixture is in `tests/conftest.py`.

**Naming:**
- `test_*.py` naming convention throughout `tests/unit/`.

**Structure:**
```
tests/
├── conftest.py
└── unit/
    ├── test_request_translator.py
    ├── test_stream_translator.py
    ├── test_routes_anthropic_compat.py
    └── ...
```

## Test Structure

**Suite Organization:**
```python
def _spec(**kw) -> CapabilityManifest:
    ...

def test_reasoning_toggle_on_when_thinking_present():
    body = {...}
    out = translate_request(body, _spec(), ToolIdMap())
    assert out["messages"][0]["content"] == "detailed thinking on"
```

Pattern shown in `tests/unit/test_request_translator.py`.

**Patterns:**
- Setup helper functions for fixtures/builders (`_spec`, `_collect`, `_client`) in test modules.
- Test lifecycle defaults are centralized in `tests/conftest.py` (adds `src/` to path, sets test env vars).
- Assertions focus on protocol-level outputs and compatibility contracts.

## Mocking

**Framework:** 
- `unittest.mock` usage detected (for example `MagicMock` import in `tests/unit/test_stream_translator.py`).
- `respx` and `pytest-httpx` are declared and documented, but concrete usage was not verified in sampled files.

**Patterns:**
```python
from unittest.mock import MagicMock
```
from `tests/unit/test_stream_translator.py`.

**What to Mock:**
- Upstream HTTP/API boundaries (documented in `README.md`/`GEMINI.md`), especially NVIDIA API calls.

**What NOT to Mock:**
- Core translation logic is tested directly with real function calls in `tests/unit/test_request_translator.py`.

## Fixtures and Factories

**Test Data:**
```python
@pytest.fixture
def model_registry():
    from nvd_claude_proxy.config.models import load_model_registry
    return load_model_registry(os.environ["MODEL_CONFIG_PATH"])
```
from `tests/conftest.py`.

**Location:**
- Shared fixtures in `tests/conftest.py`.
- Per-file builders/helpers inside each test module.

## Coverage

**Requirements:** 
- Unknown: no explicit minimum coverage threshold found in `pyproject.toml`, `Makefile`, or `.github/workflows/ci-cd.yml`.

**View Coverage:**
```bash
Unknown (no dedicated coverage command/config detected)
```

## Test Types

**Unit Tests:**
- Primary test type; focus on translators, routing compatibility, token counting, and model registry behavior in `tests/unit/`.

**Integration Tests:**
- Limited API-level checks via FastAPI `TestClient` in `tests/unit/test_routes_anthropic_compat.py` (still located under unit tree).

**E2E Tests:**
- Not detected (no playwright/cypress/selenium config found).

## Common Patterns

**Async Testing:**
```python
[tool.pytest.ini_options]
asyncio_mode = "auto"
```
from `pyproject.toml`.

**Error Testing:**
```python
if r.status_code == 404:
    assert body.get("detail", {}).get("error", {}).get("type") == "not_found_error"
```
from `tests/unit/test_routes_anthropic_compat.py`.

## Coverage Posture and Gaps (Evidence-Based)

- Strong coverage around request/stream translation paths (`tests/unit/test_request_translator.py`, `tests/unit/test_stream_translator.py`).
- API compatibility expectations validated in route tests (`tests/unit/test_routes_anthropic_compat.py`).
- Gap: no explicit coverage reporting or threshold gate detected.
- Gap: E2E/black-box runtime testing is not detectable in repository configs.

---

*Testing analysis: 2026-04-21*
