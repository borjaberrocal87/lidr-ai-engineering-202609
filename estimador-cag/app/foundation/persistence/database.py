"""Motor de SQLAlchemy, factorías de sesión y ayudantes de sesión por petición.

Dos stacks conviven a propósito:

* **Sync (psycopg)** — la ingesta de la Sesión 06. No está en la ruta caliente
  de peticiones de usuario, así que cambiamos ergonomía asíncrona por
  simplicidad.
* **Async (asyncpg)** — el store RAG de la Sesión 08. ``POST
  /embeddings/ingest`` y ``POST /search`` sí están en la ruta caliente y no
  deben bloquear el event loop en I/O de Postgres.

Ambos motores leen ``Settings.database_url``; el async cambia el token del
driver, de modo que una sola variable configura todo el servicio.
"""

from __future__ import annotations

from collections.abc import Iterator
from functools import lru_cache

from sqlalchemy import Engine, create_engine
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import Session, sessionmaker

from app.config import get_settings


@lru_cache
def create_engine_from_settings() -> Engine:
    """Construye el motor global a partir de ``Settings.DATABASE_URL`` (singleton)."""
    return create_engine(
        get_settings().database_url,
        pool_pre_ping=True,
        future=True,
    )


SessionLocal = sessionmaker(
    bind=create_engine_from_settings(),
    autoflush=False,
    autocommit=False,
    expire_on_commit=False,
    future=True,
)


def get_session() -> Iterator[Session]:
    """Dependencia de FastAPI que produce una Session y la cierra al salir."""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


def _async_database_url() -> str:
    """Deriva la URL de asyncpg desde ``Settings.database_url``.

    La URL canónica usa el driver síncrono (``postgresql+psycopg://``) porque
    Alembic y los repositorios de la Sesión 06 son síncronos. El store RAG
    cambia el token del driver en lugar de introducir una segunda variable.
    """
    url = get_settings().database_url
    if "+asyncpg" in url:
        return url
    if "+psycopg" in url:
        return url.replace("+psycopg", "+asyncpg")
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+asyncpg://", 1)
    return url


@lru_cache
def create_async_engine_from_settings() -> AsyncEngine:
    """Motor async global para el store RAG (singleton)."""
    return create_async_engine(
        _async_database_url(),
        pool_pre_ping=True,
    )


@lru_cache
def get_async_session_factory() -> async_sessionmaker[AsyncSession]:
    """Factoría de sesiones async.

    ``expire_on_commit=False`` para que el ``document_id`` recién persistido
    siga siendo legible después del commit, sin un SELECT extra.
    """
    return async_sessionmaker(
        bind=create_async_engine_from_settings(),
        autoflush=False,
        expire_on_commit=False,
    )
