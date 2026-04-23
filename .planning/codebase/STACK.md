# Tech Stack

**Last Updated:** 2026-04-23 (v0.8.7)

## Core Language & Runtime
- **Python 3.11+**: Primary runtime for the proxy server.
- **FastAPI / Starlette**: High-performance async web framework for API routing and middleware.
- **Uvicorn / uvloop**: Lightning-fast ASGI server and event loop for production deployment.

## Persistence & State
- **SQLite**: Reliable local database for persistent session storage and model mappings.
- **SQLAlchemy (Async)**: Modern ORM for database interactions.
- **aiosqlite**: Non-blocking SQLite driver.

## Translation & Repair
- **json-repair**: Proactive library for fixing truncated or malformed JSON in tool calls.
- **json5**: Relaxed JSON parsing for handling model hallucinations.
- **tiktoken**: Precise token counting and estimation for context window management.
- **Pydantic v2**: Strict schema validation for all API inputs and internal data models.

## Client & Networking
- **httpx**: Resilient async HTTP client with advanced retry and stream-parsing capabilities.
- **WebSocket (FastAPI)**: Real-time telemetry for the Live Monitor dashboard.

## Frontend (Dashboard)
- **Tailwind CSS**: Modern utility-first styling for the dark-mode console.
- **React / JavaScript**: Dynamic SPA for managing sessions and visual model mapping.
- **Lucide Icons**: Professional iconography for the UI.

## Development & Quality
- **Ruff**: Extremely fast Python linter and formatter.
- **mypy**: Static type checking for production safety.
- **pytest-asyncio**: Comprehensive async test suite.
- **structlog**: Structured JSON logging for observability.
