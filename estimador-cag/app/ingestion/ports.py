"""Puertos que el caso de uso de ingesta necesita del exterior.

El núcleo de ``app.ingestion`` no importa SQLAlchemy ni Presidio: depende de
estos contratos (`Protocol`), que los adaptadores de ``app.persistence`` y
``app.ingestion.pii`` satisfacen estructuralmente. Esto es lo que permite
testear el orquestador con dobles en memoria y mantener la regla de dependencia
(ver ``tests/test_ingestion_layering.py``).
"""

from __future__ import annotations

import uuid
from typing import Protocol


class JobTracker(Protocol):
    """Persistencia del ciclo de vida de un job de ingesta."""

    def mark_running(self, job_id: uuid.UUID) -> None:  # pragma: no cover - stub
        ...

    def mark_completed(
        self, job_id: uuid.UUID, *, documents_count: int
    ) -> None:  # pragma: no cover - stub
        ...

    def mark_failed(
        self, job_id: uuid.UUID, *, error_message: str
    ) -> None:  # pragma: no cover - stub
        ...


class IngestionUnitOfWork(Protocol):
    """Frontera transaccional del caso de uso.

    Los repositorios hacen ``flush()``; el ``commit()`` lo decide la frontera
    (el router o la BackgroundTask) a través de este objeto, que a su vez expone
    el :class:`JobTracker`. ``jobs`` se declara como propiedad de solo lectura
    porque los atributos mutables son invariantes en un Protocol y rechazarían
    un subtipo concreto como ``JobsRepository``.
    """

    @property
    def jobs(self) -> JobTracker:  # pragma: no cover - stub
        ...

    def commit(self) -> None:  # pragma: no cover - stub
        ...


class TextPseudonymizer(Protocol):
    """Adaptador de pseudonimización: texto crudo → texto pseudonimizado.

    Se modela como callable para que el núcleo no conozca el tipo de resultado
    de Presidio: el adaptador concreto (construido en ``app.dependencies``)
    encapsula la llamada a ``ConsistentPseudonymizer``.
    """

    def __call__(self, text: str) -> str:  # pragma: no cover - stub
        ...
