"""Unit of Work ligero para el subsistema de ingesta.

Agrupa una ``Session`` de SQLAlchemy con sus repositorios y expone ``commit()``
para que la **frontera** (el router o la BackgroundTask) decida cuándo cerrar la
transacción. Los repositorios sólo hacen ``flush()``; nunca ``commit()``. Así el
caso de uso controla la atomicidad y los tests pueden sustituir el UoW por un
doble en memoria sin arrastrar Postgres.
"""

from __future__ import annotations

from collections.abc import Callable
from types import TracebackType

from sqlalchemy.orm import Session

from app.persistence.database import SessionLocal
from app.persistence.repositories.jobs import JobsRepository
from app.persistence.repositories.mappings import MappingsRepository


class UnitOfWork:
    """Sesión + repositorios, con control explícito de la transacción."""

    def __init__(self, session_factory: Callable[[], Session] = SessionLocal) -> None:
        self._session = session_factory()
        self.jobs = JobsRepository(self._session)
        self.mappings = MappingsRepository(self._session)

    @property
    def session(self) -> Session:
        """Sesión subyacente (solo lectura). La usa la frontera para construir
        adaptadores ligados a esta transacción, p. ej. el pseudonimizador."""
        return self._session

    def commit(self) -> None:
        self._session.commit()

    def rollback(self) -> None:
        self._session.rollback()

    def close(self) -> None:
        self._session.close()

    def __enter__(self) -> UnitOfWork:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if exc_type is None:
            self.commit()
        else:
            self.rollback()
        self.close()
