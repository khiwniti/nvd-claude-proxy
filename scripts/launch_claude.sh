#!/bin/bash
export ANTHROPIC_BASE_URL="http://localhost:8787"
# Optional: force a dummy key if your proxy doesn't check for validity yet
export ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-sk-local-proxy}"
claude "$@"
