"""Repositorio de correspondencias de pseudonimización.

La operación ``lookup_or_create`` es la que el pseudonimizador llama por cada
entidad detectada. Debe ser idempotente bajo contención: dos llamadas paralelas
con el mismo ``(entity_type, original_hash)`` DEBEN converger al mismo
seudónimo — de lo contrario el corpus adquiriría dos seudónimos distintos para
la misma persona y se rompería la garantía de consistencia.

Nos apoyamos en la restricción UNIQUE de Postgres para imponerlo: insertar
primero y recurrir a un SELECT cuando el insert entra en conflicto. Esto es más
barato que una carrera SELECT-y-luego-INSERT y sobrevive a escritores
concurrentes sin bloqueo explícito.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from app.persistence.models import PseudonymMappingRow


@dataclass(frozen=True)
class Mapping:
    """Vista en memoria de una fila. Los repositorios nunca filtran tipos de SQLAlchemy."""

    entity_type: str
    original_hash: str
    pseudonym: str
    created_at: datetime


class MappingsRepository:
    """Envoltura fina alrededor de la tabla ``pseudonym_mappings``."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def lookup(self, entity_type: str, original_hash: str) -> Mapping | None:
        row = self._session.execute(
            select(PseudonymMappingRow).where(
                PseudonymMappingRow.entity_type == entity_type,
                PseudonymMappingRow.original_hash == original_hash,
            )
        ).scalar_one_or_none()
        return _to_mapping(row) if row is not None else None

    def lookup_or_create(
        self,
        entity_type: str,
        original_hash: str,
        new_pseudonym_factory: Callable[[], str],
    ) -> Mapping:
        """Devuelve la correspondencia existente o inserta una nueva, de forma determinista.

        ``new_pseudonym_factory`` es un callable sin argumentos que se invoca
        solo cuando no existe una correspondencia previa. Se pasa de forma
        perezosa porque generar un valor de Faker es barato pero inútil si hay
        acierto de caché.
        """
        existing = self.lookup(entity_type, original_hash)
        if existing is not None:
            return existing

        pseudonym = new_pseudonym_factory()
        stmt = (
            pg_insert(PseudonymMappingRow)
            .values(
                entity_type=entity_type,
                original_hash=original_hash,
                pseudonym=pseudonym,
            )
            .on_conflict_do_nothing(index_elements=["entity_type", "original_hash"])
            .returning(PseudonymMappingRow)
        )
        row = self._session.execute(stmt).scalar_one_or_none()
        self._session.flush()

        if row is not None:
            return _to_mapping(row)

        # Otro escritor ganó la carrera; lee la fila ganadora.
        winner = self.lookup(entity_type, original_hash)
        if winner is None:  # pragma: no cover - la unique constraint garantiza ganador
            raise RuntimeError("unique constraint guaranteed a winner exists")
        return winner

    def forget(self, entity_type: str, original_hash: str) -> bool:
        """Elimina una correspondencia. Devuelve True si existía una fila."""
        if self.lookup(entity_type, original_hash) is None:
            return False
        self._session.execute(
            delete(PseudonymMappingRow).where(
                PseudonymMappingRow.entity_type == entity_type,
                PseudonymMappingRow.original_hash == original_hash,
            )
        )
        self._session.flush()
        return True


def _to_mapping(row: PseudonymMappingRow) -> Mapping:
    return Mapping(
        entity_type=row.entity_type,
        original_hash=row.original_hash,
        pseudonym=row.pseudonym,
        created_at=row.created_at,
    )
