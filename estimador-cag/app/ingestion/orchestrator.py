"""Caso de uso: ejecuta la ingesta de una fuente incluida del catálogo.

Une catálogo + loader + parser en una única corrida **síncrona** que invoca una
BackgroundTask de FastAPI. Garantías:

1. La fuente debe existir y tener ``decision == include``; ``ensure_ingestible``
   lo valida y lanza ``UnknownSource`` / ``SourceNotIncluded`` (el router las
   traduce a 404/400). El router llama a ``ensure_ingestible`` en la petición y
   pasa la fuente ya resuelta a ``ingest_source``: la validación ocurre una sola
   vez.
2. El estado del job avanza ``pending → running → completed | failed`` a través
   del puerto ``IngestionUnitOfWork``; el commit lo gobierna el caso de uso en la
   frontera de cada transición (UoW ligero).
3. El núcleo no conoce SQLAlchemy: sólo depende de los puertos de
   ``app.ingestion.ports``.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import structlog

from app.ingestion.catalog.models import CatalogDecision, CatalogSource, DataCatalog
from app.ingestion.documents.models import Document
from app.ingestion.errors import SourceNotIncluded, UnknownSource
from app.ingestion.loaders.filesystem import FileSystemLoader
from app.ingestion.parsers.protocol import ParseContext
from app.ingestion.parsers.registry import ParserRegistry
from app.ingestion.ports import IngestionUnitOfWork, TextPseudonymizer

log = structlog.get_logger(__name__)


def ensure_ingestible(catalog: DataCatalog, source_name: str) -> CatalogSource:
    """Valida que ``source_name`` existe y está incluida; devuelve la fuente.

    Es el único punto de validación de la decisión del catálogo: lo usan el
    router (para fallar rápido antes de crear el job) y, defensivamente, los
    scripts.
    """
    source = catalog.find(source_name)
    if source is None:
        raise UnknownSource(source_name)
    if source.decision is not CatalogDecision.INCLUDE:
        raise SourceNotIncluded(source_name, source.decision.value, source.decision_reason)
    return source


def ingest_source(
    *,
    source: CatalogSource,
    source_version: str,
    loader: FileSystemLoader,
    registry: ParserRegistry,
    uow: IngestionUnitOfWork,
    job_id: uuid.UUID,
    pseudonymizer: TextPseudonymizer | None = None,
) -> list[Document]:
    """Ejecuta la ingesta de una fuente ya validada.

    Si se inyecta ``pseudonymizer``, el texto de cada ``Document`` se sustituye
    por su versión pseudonimizada antes de completar el job (los mappings se
    persisten con el mismo commit). La decisión de *cuándo* pseudonimizar (flag
    de configuración + ``sensitivity.has_pii``) la toma la frontera, no el caso
    de uso.

    Efectos secundarios:
      * Actualiza la fila ``ingestion_jobs`` referenciada por ``job_id``.
      * Emite eventos structlog en las fronteras de la corrida.
    """
    bound = log.bind(job_id=str(job_id), source_name=source.name)

    uow.jobs.mark_running(job_id)
    uow.commit()
    bound.info("ingestion.started", format=source.format)

    parser = registry.get(source.format)
    context = ParseContext(
        source=source,
        source_version=source_version,
        ingested_at=datetime.now(UTC),
    )

    documents: list[Document] = []
    try:
        for blob in loader.iter_blobs(source.location, {source.format}):
            documents.extend(parser.parse(blob, context))

        if pseudonymizer is not None:
            documents = [
                document.model_copy(update={"text": pseudonymizer(document.text)})
                for document in documents
            ]
            bound.info("ingestion.pseudonymized", documents_count=len(documents))
    except Exception as exc:
        bound.error("ingestion.failed", error=str(exc))
        uow.jobs.mark_failed(job_id, error_message=str(exc))
        uow.commit()
        raise

    uow.jobs.mark_completed(job_id, documents_count=len(documents))
    uow.commit()
    bound.info("ingestion.completed", documents_count=len(documents))
    return documents
