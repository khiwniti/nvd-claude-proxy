from __future__ import annotations

import logging
import signal
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from fastapi.responses import ORJSONResponse

from ._version import __version__
from .clients.nvidia_client import NvidiaClient
from .config.models import load_model_registry
from .config.settings import get_settings
from .middleware.body_limit import BodyLimitMiddleware
from .middleware.logging import LoggingMiddleware
from .middleware.rate_limiter import DistributedRateLimiterMiddleware
from .routes import count_tokens, health, messages, metrics_route, models, stubs, openapi
from .services.storage.factory import create_storage_engine

_log = structlog.get_logger("nvd_claude_proxy.app")

# Import security middleware for production hardening
try:
    from .middleware.security import (
        SecurityHeadersMiddleware,
        SSRFProtectionMiddleware,
        SuspiciousRequestDetectionMiddleware,
        RequestTimingMiddleware,
        AuditLoggerMiddleware,
        AuthMiddleware,
    )

    _HAS_SECURITY_MIDDLEWARE = True
except ImportError:
    _HAS_SECURITY_MIDDLEWARE = False
    _log.warning("Security middleware not available - install all dependencies")


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
    """Register a SIGHUP handler that reloads models.yaml without restart."""
    try:

        def _reload(signum, frame) -> None:
            try:
                new_registry = load_model_registry(app.state.settings.model_config_path)
                app.state.model_registry = new_registry
                _log.info("models.reloaded", aliases=list(new_registry.specs.keys()))
            except Exception as exc:
                _log.error("models.reload_failed", error=str(exc))

        signal.signal(signal.SIGHUP, _reload)
    except (AttributeError, OSError):
        pass


def create_app() -> FastAPI:
    settings = get_settings()
    _configure_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # 1. Initialize DB (Legacy support/fallback)
        from .db.database import init_db

        try:
            await init_db()
        except Exception as exc:
            _log.error("database.init_failed", error=str(exc))

        # 2. Initialize Storage Engine (Redis or SQLite)
        app.state.storage = create_storage_engine(app.state.settings)
        _log.info("storage_engine.initialized", engine=app.state.settings.storage_engine)

        # 3. Create Upstream Client
        app.state.nvidia_client = NvidiaClient(app.state.settings)
        _log.info("nvidia_client.created")

        yield

        await app.state.nvidia_client.aclose()
        _log.info("nvidia_client.closed")

    app = FastAPI(
        title="nvd-claude-proxy",
        version=__version__,
        default_response_class=ORJSONResponse,
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.model_registry = load_model_registry(settings.model_config_path)

    from .config.server_tools import load_server_tool_registry
    app.state.server_tool_registry = load_server_tool_registry()

    _install_sighup_handler(app)

    # Middleware Pipeline (Outermost -> Innermost)

    # 1. Logging & Metrics
    app.add_middleware(LoggingMiddleware)

    # 2. Global Rate Limiting (Distributed)
    app.add_middleware(DistributedRateLimiterMiddleware)

    # 3. Request Hardening
    if settings.max_request_body_mb > 0:
        max_bytes = int(settings.max_request_body_mb * 1024 * 1024)
        app.add_middleware(BodyLimitMiddleware, max_bytes=max_bytes)

    # 4. Session Persistence
    from .middleware.session_middleware import SessionMiddleware

    app.add_middleware(SessionMiddleware)

    # 5. Security Suite
    if _HAS_SECURITY_MIDDLEWARE:
        app.add_middleware(AuditLoggerMiddleware)
        app.add_middleware(RequestTimingMiddleware, slow_request_threshold=30.0)
        app.add_middleware(SuspiciousRequestDetectionMiddleware)
        app.add_middleware(SSRFProtectionMiddleware)
        app.add_middleware(AuthMiddleware)
        app.add_middleware(SecurityHeadersMiddleware)
        _log.info("security.middleware.enabled")

    # Routes
    app.include_router(messages.router)
    app.include_router(count_tokens.router)
    app.include_router(models.router)
    app.include_router(health.router)
    app.include_router(metrics_route.router)
    app.include_router(stubs.router)
    app.include_router(openapi.router)

    return app
