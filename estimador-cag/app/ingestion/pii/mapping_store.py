"""El almacén de correspondencias: respaldo persistente para una pseudonimización consistente.

Dos implementaciones:

* :class:`PostgresMappingStore` — la ruta de producción. Envuelve el
  :class:`MappingsRepository`. A la base de datos llega el HMAC del valor
  original (no el valor en sí); el valor nunca llega. Esa asimetría es lo que
  hace auditable el "derecho al olvido": borrar la fila por su hash elimina la
  reversibilidad sin haber almacenado jamás el texto en claro.

* :class:`InMemoryMappingStore` — para tests y scripts puntuales. Mismo
  contrato, sin base de datos.

Ambos almacenes son *idempotentes*: consultar el mismo ``(entity_type, hash)``
dos veces devuelve el mismo seudónimo, sea cual sea el orden de las llamadas.
Las llamadas concurrentes convergen gracias a la restricción de unicidad de
más arriba.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, runtime_checkable

from sqlalchemy.orm import Session

from app.persistence.repositories.mappings import MappingsRepository


@runtime_checkable
class MappingStore(Protocol):
    def lookup_or_create(
        self,
        entity_type: str,
        original_hash: str,
        new_pseudonym_factory: Callable[[], str],
    ) -> str:  # pragma: no cover - stub del protocolo
        ...

    def forget(self, entity_type: str, original_hash: str) -> bool: ...


class PostgresMappingStore:
    """Almacén de correspondencias persistente respaldado por Postgres."""

    def __init__(self, session: Session) -> None:
        self._repo = MappingsRepository(session)

    def lookup_or_create(
        self,
        entity_type: str,
        original_hash: str,
        new_pseudonym_factory: Callable[[], str],
    ) -> str:
        mapping = self._repo.lookup_or_create(
            entity_type=entity_type,
            original_hash=original_hash,
            new_pseudonym_factory=new_pseudonym_factory,
        )
        return mapping.pseudonym

    def forget(self, entity_type: str, original_hash: str) -> bool:
        return self._repo.forget(entity_type, original_hash)


class InMemoryMappingStore:
    """Almacén de correspondencias simple basado en dict para tests y scripts."""

    def __init__(self) -> None:
        self._mappings: dict[tuple[str, str], str] = {}

    def lookup_or_create(
        self,
        entity_type: str,
        original_hash: str,
        new_pseudonym_factory: Callable[[], str],
    ) -> str:
        key = (entity_type, original_hash)
        if key not in self._mappings:
            self._mappings[key] = new_pseudonym_factory()
        return self._mappings[key]

    def forget(self, entity_type: str, original_hash: str) -> bool:
        return self._mappings.pop((entity_type, original_hash), None) is not None
