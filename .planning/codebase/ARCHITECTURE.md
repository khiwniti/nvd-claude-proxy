# Architecture

**Last Updated:** 2026-04-23 (v0.8.7)

## Pattern Overview

**Overall:** A production-hardened FastAPI proxy translating Anthropic Messages API semantics to NVIDIA NIM (OpenAI-compatible) endpoints.

**Key Architectural Upgrades:**
- **Modular Transformer Pipeline:** A "Chain of Responsibility" pattern for model-specific fixes (JSON repair, character fixing, reasoning extraction).
- **Persistent Session Management:** SQLite-backed isolation of tool states and conversation histories via `sk-ncp-*` keys.
- **Showcase Dashboard:** Built-in web UI with real-time stream monitoring and visual model mapping.

## Subsystems and Responsibilities

### Modular Transformer Pipeline (`src/nvd_claude_proxy/translators/transformers.py`)
- **JSONRepairTransformer:** Real-time repair of truncated tool arguments using `json-repair`.
- **ReasoningTransformer:** Standardizes thought extraction from `<think>` tags and `reasoning_content`.
- **CharFixerTransformer:** Strips illegal control characters from model outputs.
- **WebSearchTransformer:** Maps NVIDIA/OpenAI citations to Anthropic `web_search_tool_result`.

### Session & Persistence Layer (`src/nvd_claude_proxy/db/`, `services/`)
- **SQLite Database:** Stores session metadata, API keys, and model mappings in `sessions.db`.
- **SessionService:** Manages isolated `ToolIdMap` and `TransformerChain` instances per session to prevent cross-window state contamination.
- **SessionMiddleware:** Intercepts `x-api-key` headers to route requests to the correct persistent context.

### API & Routing Layer
- **Messages Route (`routes/messages.py`):** Orchestrates the request lifecycle, failure-handling, and transformer-driven streaming.
- **Dashboard API (`routes/dashboard.py`):** CRUD endpoints for managing sessions and model aliases.
- **Real-time Monitor:** WebSocket-driven telemetry for visual verification of proxy transformations.

## Data and Control Flow

1. **Request Interception:** `SessionMiddleware` extracts the session key (`sk-ncp-*`) and loads the context from SQLite.
2. **Translation:** `TransformerChain` modifies the request (e.g., injecting thinking prompts).
3. **Upstream Call:** `NvidiaClient` handles the resilient async connection to NVIDIA NIM.
4. **Stream Repair:** `StreamTranslator` scanners detect `<think>` tags and truncated JSON, applying fixes via the Transformer Pipeline before yielding events to Claude Code.
5. **Persistence:** Token counts and tool ID mappings are synced back to SQLite on request completion.

---
*Documentation updated for v0.8.7 production release.*
