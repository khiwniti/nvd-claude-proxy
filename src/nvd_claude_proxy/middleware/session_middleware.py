from __future__ import annotations

from typing import Callable

import structlog
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from sqlalchemy.future import select

from ..db.database import async_session_factory
from ..db.models import Session

_log = structlog.get_logger("nvd_claude_proxy.session_middleware")

class SessionMiddleware(BaseHTTPMiddleware):
    """
    Middleware that identifies the user session via the x-api-key header.
    If the key starts with 'sk-ncp-', it lookups or creates a persistent
    session in the database and attaches it to request.state.session.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        api_key = request.headers.get("x-api-key")
        if not api_key:
            auth = request.headers.get("authorization", "")
            if auth.lower().startswith("bearer "):
                api_key = auth[7:].strip()

        if api_key and api_key.startswith("sk-ncp-"):
            async with async_session_factory() as db_session:
                stmt = select(Session).where(Session.api_key == api_key)
                result = await db_session.execute(stmt)
                session_obj = result.scalar_one_or_none()

                if not session_obj:
                    _log.info("session.auto_create", api_key=api_key[:12] + "...")
                    session_obj = Session(api_key=api_key)
                    db_session.add(session_obj)
                    await db_session.commit()
                    await db_session.refresh(session_obj)
                
                # Attach to request.state for downstream use
                request.state.session = session_obj
        else:
            request.state.session = None

        return await call_next(request)
