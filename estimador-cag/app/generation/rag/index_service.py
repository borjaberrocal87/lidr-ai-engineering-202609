"""Expansión incremental del corpus (Sesión 11, directo).

Crecer el corpus no debe ser un "re-ingest de todo": esta pieza añade documentos
NUEVOS reutilizando :class:`RagIngestService` (chunk → embed → persist, con la
guarda de duplicado), sin reimplementar nada del pipeline de ingesta.

Diferencias con una ingesta a ciegas:

* un documento ya presente (mismo ``source_path``) se **omite**, no rompe el lote;
* un documento inválido no tumba a los demás: se cuenta como fallido y se sigue;
* ``on_progress`` reporta el avance tras cada documento para que el job asíncrono
  muestre un contador vivo en la UI.

No se re-embebe el corpus existente (eso sería un cambio de modelo / migración
blue-green, fuera de alcance): los documentos nuevos se embeben con el MISMO
modelo y caen en su tabla, donde el índice HNSW de la migración 0005 los recoge
de forma incremental.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import structlog

from app.generation.rag.ingest_service import DuplicateDocumentError, RagIngestService
from app.generation.rag.schemas import Budget

log = structlog.get_logger()


@dataclass(frozen=True, slots=True)
class CorpusExpansionResult:
    """Balance de un lote de expansión."""

    documents_indexed: int
    documents_skipped: int
    documents_failed: int
    chunks_created: int

    @property
    def documents_processed(self) -> int:
        """Documentos procesados (indexados + omitidos + fallidos)."""
        return self.documents_indexed + self.documents_skipped + self.documents_failed


class CorpusIndexService:
    """Añade documentos nuevos al corpus en lote, reutilizando la ingesta."""

    def __init__(self, ingest: RagIngestService) -> None:
        self._ingest = ingest

    async def expand(
        self,
        documents: list[Budget],
        *,
        document_type: str = "historical_budget",
        chunk_type: str = "budget_component",
        source_prefix: str = "corpus-expansion",
        on_progress: Callable[[int], None] | None = None,
    ) -> CorpusExpansionResult:
        """Añade cada documento del lote, omitiendo duplicados y fallos aislados.

        ``on_progress`` recibe el número de documentos ya procesados (indexados +
        omitidos + fallidos) tras cada uno, para que el job reporte avance real.
        """
        indexed = skipped = failed = chunks = 0
        for document in documents:
            source_path = f"{source_prefix}::{document.budget_id}"
            try:
                response = await self._ingest.ingest(
                    source_path=source_path,
                    document_type=document_type,
                    budget=document,
                    chunk_type=chunk_type,
                )
                indexed += 1
                chunks += response.chunks_created
            except DuplicateDocumentError:
                skipped += 1
                log.info("corpus_expansion_skip_duplicate", source_path=source_path)
            except Exception as exc:
                failed += 1
                log.warning(
                    "corpus_expansion_document_failed",
                    source_path=source_path,
                    error=str(exc)[:300],
                )
            if on_progress is not None:
                on_progress(indexed + skipped + failed)

        result = CorpusExpansionResult(
            documents_indexed=indexed,
            documents_skipped=skipped,
            documents_failed=failed,
            chunks_created=chunks,
        )
        log.info(
            "corpus_expansion_done",
            documents_indexed=indexed,
            documents_skipped=skipped,
            documents_failed=failed,
            chunks_created=chunks,
        )
        return result
