from __future__ import annotations

import logging
import signal
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from fastapi.responses import ORJSONResponse

from .clients.nvidia_client import NvidiaClient
from .config.models import load_model_registry
from .config.settings import get_settings
from .middleware.body_limit import BodyLimitMiddleware
from .middleware.logging import LoggingMiddleware
from .middleware.rate_limiter import RateLimiterMiddleware
from .routes import count_tokens, health, messages, metrics_route, models, stubs

_log = structlog.get_logger("nvd_claude_proxy.app")


def _configure_logging(level: str) -> None:
    logging.basicConfig(level=level.upper(), format="%(message)s")
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.getLevelName(level.upper())),
        cache_logger_on_first_use=True,
    )


def _install_sighup_handler(app: FastAPI) -> None:
    """Register a SIGHUP handler that reloads models.yaml without restart.

    Send ``kill -HUP <pid>`` (or ``kill -1 <pid>``) to trigger a reload.
    Not available on Windows; silently skipped there.
    """
    try:

        def _reload(signum, frame) -> None:  # noqa: ARG001
            try:
                new_registry = load_model_registry(app.state.settings.model_config_path)
                app.state.model_registry = new_registry
                _log.info(
                    "models.reloaded",
                    aliases=list(new_registry.specs.keys()),
                )
            except Exception as exc:  # noqa: BLE001
                _log.error("models.reload_failed", error=str(exc))

        signal.signal(signal.SIGHUP, _reload)
    except (AttributeError, OSError):
        # Windows: SIGHUP doesn't exist; just skip.
        pass


def create_app() -> FastAPI:
    settings = get_settings()
    _configure_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Create a shared NvidiaClient (and its httpx connection pool) once.
        app.state.nvidia_client = NvidiaClient(app.state.settings)
        _log.info("nvidia_client.created")
        yield
        await app.state.nvidia_client.aclose()
        _log.info("nvidia_client.closed")

    app = FastAPI(
        title="nvd-claude-proxy",
        version="0.2.6",
        default_response_class=ORJSONResponse,
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.model_registry = load_model_registry(settings.model_config_path)
    _install_sighup_handler(app)
    # Middleware is applied in reverse registration order (last added = outermost).
    app.add_middleware(LoggingMiddleware)
    if settings.rate_limit_rpm > 0:
        app.add_middleware(RateLimiterMiddleware, rpm_limit=settings.rate_limit_rpm)
    if settings.max_request_body_mb > 0:
        max_bytes = int(settings.max_request_body_mb * 1024 * 1024)
        app.add_middleware(BodyLimitMiddleware, max_bytes=max_bytes)
    app.include_router(messages.router)
    app.include_router(count_tokens.router)
    app.include_router(models.router)
    app.include_router(health.router)
    app.include_router(metrics_route.router)
    app.include_router(stubs.router)
    return app
