from __future__ import annotations

import os
from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

# Default to a local SQLite database in the current directory.
# In a production environment, this should be set via environment variable.
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./nvd_claude_proxy.db")

# Create the async engine.
engine = create_async_engine(DATABASE_URL)

# Create a session factory.
async_session_factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency for providing a database session to FastAPI routes."""
    async with async_session_factory() as session:
        yield session


async def init_db() -> None:
    """Initialize the database, creating all defined tables."""
    from .models import Base

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
