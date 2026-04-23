# Codebase Structure

**Last Updated:** 2026-04-23 (v0.8.7)

## Area Ownership and Purpose

### Application Runtime (`src/nvd_claude_proxy/`)
- **App Core:** `app.py`, `main.py`
- **Session DB:** `db/database.py`, `db/models.py`
- **Dashboard API:** `routes/dashboard.py`
- **Translation Engine:** `translators/transformers.py`, `translators/stream_translator.py`
- **Isolated State:** `services/session_service.py`
- **Session Routing:** `middleware/session_middleware.py`
- **Static Assets:** `static/index.html`, `static/js/dashboard.js`
- **Configuration:** `config/settings.py`, `data/models.yaml`

### Deployment & Packaging
- **`MANIFEST.in`**: Ensures static assets and data files are included in the PyPI wheel.
- **`pyproject.toml`**: Unified project configuration and dependency tree.
- **`Makefile`**: Developer automation and release management.

## Key Directory Map

```text
nvd-claude-proxy/
├── src/nvd_claude_proxy/
│   ├── db/               # SQLite persistence layer
│   ├── routes/           # FastAPI endpoints (Messages, Dashboard)
│   ├── translators/      # Modular Transformer Pipeline
│   ├── services/         # Persistent session isolation
│   ├── middleware/       # Session interception and security
│   ├── static/           # Dashboard Web UI (Showcase)
│   └── util/             # Token metrics, cost estimation, headers
├── .planning/            # Architecture and roadmap docs
├── docs/                 # Detailed Design documents
├── tests/                # Async test suite
└── MANIFEST.in           # Package distribution rules
```
