# nvd-claude-proxy Design Document

## 1. Overview
`nvd-claude-proxy` is a local HTTP proxy that translates Anthropic Messages API requests to NVIDIA NIM (OpenAI-compatible) API requests. This document outlines the Phase 02 architecture, focusing on session persistence, a web dashboard, and enhanced model management.

## 2. Persistent Session Manager
To support complex workflows like Claude Code, the proxy must maintain state across multiple requests.

### 2.1. Key Identification
Clients identify themselves using a custom API key format: `sk-ncp-[unique-id]`.
- This key acts as a session identifier.
- The proxy looks up the session in its local SQLite database.
- If no session exists, it creates one (or falls back to default behavior if a standard NVIDIA key is provided).

### 2.2. Isolated State
Each session maintains its own isolated:
- **ToolIdMap**: Persists the mapping between Anthropic `toolu_` IDs and NVIDIA `call_` IDs across turns, ensuring the model can correctly reference previous tool results.
- **Transformer Chain**: Individual transformers (Reasoning, JSON Repair, etc.) can be toggled or configured per session.
- **Metadata**: Tracks token usage, model aliases, and friendly names.

### 2.3. SQLite Storage
A local SQLite database (`nvd_claude_proxy.db`) is used for persistence.
- **Engine**: SQLAlchemy with `aiosqlite` for asynchronous I/O.
- **Migration**: Simple `Base.metadata.create_all` for initial setup.

## 3. Web Dashboard
A built-in web dashboard allows users to monitor and configure the proxy in real-time.

### 3.1. Features
- **Live Stream Visualization**: A dual-window view showing the raw Anthropic protocol events alongside the translated NVIDIA NIM stream.
- **Session Management**: List active sessions, view token usage, and modify transformer settings.
- **Usage Graphs**: Historical token consumption and latency metrics.
- **Configuration**: Edit model aliases and failover chains via a GUI.

### 3.2. Technology Stack
- **Backend**: FastAPI endpoints (SSE for live logs).
- **Frontend**: Modern Vanilla JS or React-based UI with Dark Mode support.
- **Real-time**: WebSockets or SSE for streaming log data to the browser.

## 4. Database Schema

### 4.1. `sessions` table
| Column | Type | Description |
|--------|------|-------------|
| id | Integer | Primary Key |
| api_key | String | The `sk-ncp-*` identifier |
| friendly_name | String | User-defined label for the session |
| model_alias | String | The model currently bound to this session |
| transformer_settings | JSON | Toggles for JSON repair, thinking, etc. |
| tool_id_map | JSON | Persistent ToolIdMap state |
| tokens_used | Integer | Cumulative token count |
| last_active | DateTime | Last request timestamp |
| created_at | DateTime | Session creation timestamp |

### 4.2. `model_mappings` table
Allows overriding default NIM models via the dashboard.
| Column | Type | Description |
|--------|------|-------------|
| id | Integer | Primary Key |
| anthropic_model | String | e.g. `claude-3-5-sonnet` |
| nvd_model | String | e.g. `nvidia/llama-3.1-70b-instruct` |
| overrides | JSON | Specific capability flags |

## 5. Middleware Workflow
1. **Request Interception**: Extract `x-api-key` (or `Authorization: Bearer`) header.
2. **Session Lookup**: If key starts with `sk-ncp-`, resolve it to a database record.
3. **Context Injection**: Initialize `ToolIdMap` and `TransformerChain` from the persisted SQLite state.
4. **Upstream Call**: Translate and execute request to NVIDIA NIM.
5. **State Persistence**: Sync updated `tool_id_map` and token metrics back to SQLite on request completion.
