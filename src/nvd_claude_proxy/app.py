from __future__ import annotations

import logging
import signal
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import ORJSONResponse
from fastapi.staticfiles import StaticFiles

from ._version import __version__
from .clients.nvidia_client import NvidiaClient
from .config.models import load_model_registry
from .config.settings import get_settings
from .middleware.body_limit import BodyLimitMiddleware
from .middleware.logging import LoggingMiddleware
from .middleware.rate_limiter import RateLimiterMiddleware
from .routes import count_tokens, dashboard, health, messages, metrics_route, models, stubs, openapi

_log = structlog.get_logger("nvd_claude_proxy.app")

# Import security middleware for production hardening
try:
    from .middleware.security import (
        SecurityHeadersMiddleware,
        SSRFProtectionMiddleware,
        SuspiciousRequestDetectionMiddleware,
        RequestTimingMiddleware,
        AuditLoggerMiddleware,
    )
    _HAS_SECURITY_MIDDLEWARE = True
except ImportError:
    _HAS_SECURITY_MIDDLEWARE = False
    _log.warning("Security middleware not available - install all dependencies")


class PubSub:
    """Simple PubSub manager for real-time monitoring events via WebSockets."""

    def __init__(self) -> None:
        self.subscribers: list[WebSocket] = []

    async def subscribe(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self.subscribers.append(websocket)
        _log.debug("pubsub.subscribed", count=len(self.subscribers))

    def unsubscribe(self, websocket: WebSocket) -> None:
        if websocket in self.subscribers:
            self.subscribers.remove(websocket)
            _log.debug("pubsub.unsubscribed", count=len(self.subscribers))

    async def broadcast(self, message: dict) -> None:
        if not self.subscribers:
            return

        # Create a copy of the list to avoid modification during iteration
        for subscriber in list(self.subscribers):
            try:
                await subscriber.send_json(message)
            except Exception:  # noqa: BLE001
                self.unsubscribe(subscriber)


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
        version=__version__,
        default_response_class=ORJSONResponse,
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.model_registry = load_model_registry(settings.model_config_path)
    app.state.pubsub = PubSub()
    _install_sighup_handler(app)
    # Middleware is applied in reverse registration order (last added = outermost).
    
    @app.websocket("/ws/monitor")
    async def websocket_monitor(websocket: WebSocket) -> None:
        """Endpoint for the real-time monitoring dashboard."""
        await app.state.pubsub.subscribe(websocket)
        try:
            while True:
                # Keep the connection alive; the dashboard doesn't currently send back data
                await websocket.receive_text()
        except WebSocketDisconnect:
            app.state.pubsub.unsubscribe(websocket)
        except Exception:  # noqa: BLE001
            app.state.pubsub.unsubscribe(websocket)

    app.add_middleware(LoggingMiddleware)
    if settings.rate_limit_rpm > 0:
        app.add_middleware(RateLimiterMiddleware, rpm_limit=settings.rate_limit_rpm)
    if settings.max_request_body_mb > 0:
        max_bytes = int(settings.max_request_body_mb * 1024 * 1024)
        app.add_middleware(BodyLimitMiddleware, max_bytes=max_bytes)
    
    # Session isolation middleware
    from .middleware.session_middleware import SessionMiddleware
    app.add_middleware(SessionMiddleware)
    
    # Security middleware (production hardening)
    if _HAS_SECURITY_MIDDLEWARE:
        # Outermost middleware runs first
        app.add_middleware(AuditLoggerMiddleware)
        app.add_middleware(RequestTimingMiddleware, slow_request_threshold=30.0)
        app.add_middleware(SuspiciousRequestDetectionMiddleware)
        app.add_middleware(SSRFProtectionMiddleware)
        app.add_middleware(SecurityHeadersMiddleware)
        _log.info("security.middleware.enabled")
    else:
        _log.warning("security.middleware.disabled")
    
    app.include_router(messages.router)
    app.include_router(dashboard.router)
    app.include_router(count_tokens.router)
    app.include_router(models.router)
    app.include_router(health.router)
    app.include_router(metrics_route.router)
    app.include_router(stubs.router)
    app.include_router(openapi.router)

    # Mount static files for dashboard frontend
    from pathlib import Path
    static_dir = Path(__file__).parent / "static"
    app.mount("/dashboard", StaticFiles(directory=str(static_dir), html=True), name="static")

    return app
