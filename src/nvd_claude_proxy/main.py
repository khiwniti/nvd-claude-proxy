"""Entrypoint: `python -m nvd_claude_proxy.main` or `nvd-claude-proxy`."""

from __future__ import annotations

import uvicorn

from .config.settings import get_settings


def run() -> None:
    s = get_settings()
    # uvloop/httptools aren't available on Windows; fall back gracefully.
    loop = "auto"
    http = "auto"
    try:
        import uvloop  # noqa: F401

        loop = "uvloop"
    except ImportError:
        pass
    try:
        import httptools  # noqa: F401

        http = "httptools"
    except ImportError:
        pass

    uvicorn.run(
        "nvd_claude_proxy.app:create_app",
        host=s.proxy_host,
        port=s.proxy_port,
        factory=True,
        log_level=s.log_level.lower(),
        http=http,
        loop=loop,
    )


if __name__ == "__main__":
    run()
