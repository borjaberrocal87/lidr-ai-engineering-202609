"""Postgres checkpointer wiring for the estimation graph (Session 13, Level 2).

The graph persists state per ``thread_id`` in the SAME project Postgres that holds
the pgvector embeddings — the checkpointer creates its own tables (``checkpoints``,
``checkpoint_writes``, ``checkpoint_blobs``) and coexists with them. No new
infrastructure and no Alembic migration.

LangGraph's ``AsyncPostgresSaver`` is built on **psycopg3 (async)**, so it wants a
plain libpq DSN (``postgresql://user:pass@host/db``) — NOT the SQLAlchemy
``postgresql+psycopg://`` / ``+asyncpg`` forms. ``saver_conninfo`` derives it from
the single ``database_url`` by stripping the driver token, mirroring
``_async_database_url`` in ``app/foundation/persistence/database.py``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import structlog
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from app.config import Settings, get_settings

log = structlog.get_logger(__name__)


def saver_conninfo(settings: Settings | None = None) -> str:
    """Return a plain libpq DSN for ``AsyncPostgresSaver`` from ``database_url``.

    ``postgresql+psycopg://…`` / ``postgresql+asyncpg://…`` → ``postgresql://…``.
    """
    url = (settings or get_settings()).database_url
    if "+psycopg" in url:
        return url.replace("+psycopg", "")
    if "+asyncpg" in url:
        return url.replace("+asyncpg", "")
    return url


@asynccontextmanager
async def open_checkpointer(settings: Settings | None = None) -> AsyncIterator[AsyncPostgresSaver]:
    """Open an ``AsyncPostgresSaver`` over the project Postgres and set it up.

    ``setup()`` is idempotent — it creates the checkpointer tables on first run and
    is a no-op afterwards — so calling it on every startup is safe. Use as an async
    context manager (entered in the app lifespan) so the connection is closed on
    shutdown.
    """
    conninfo = saver_conninfo(settings)
    async with AsyncPostgresSaver.from_conn_string(conninfo) as checkpointer:
        await checkpointer.setup()
        log.info("graph.checkpointer_ready")
        yield checkpointer
