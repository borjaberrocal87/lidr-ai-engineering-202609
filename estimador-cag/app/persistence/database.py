"""Motor de SQLAlchemy, factoría de sesiones y ayudantes de sesión por petición.

Usamos la API *síncrona* de SQLAlchemy 2.0. Las rutas de ingesta de este módulo
no están en el camino caliente de peticiones de usuario — se ejecutan como
BackgroundTasks u operaciones administrativas puntuales —, así que cambiamos
ergonomía asíncrona por simplicidad y menos infraestructura en movimiento
durante la formación.
"""

from __future__ import annotations

from collections.abc import Iterator
from functools import lru_cache

from sqlalchemy import Engine, create_engine
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
