"""Orquestación de la ingesta: chunk → embed → persist, en UNA transacción.

Compone el chunker estructural, el embedder de OpenAI y el store de pgvector.
La transacción única es el punto: la fila del documento se inserta primero
(para que los chunks puedan referenciarla tras el ``flush``), pero si la API de
embeddings falla después, el rollback elimina todo y no sobrevive ningún
``documents`` huérfano.
"""

from __future__ import annotations

import asyncio
import time

import structlog
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.embedding_pipeline.embedder import OpenAIEmbedder, estimate_cost_usd
from app.embedding_pipeline.schemas import Budget, IngestResponse
from app.embedding_pipeline.store.repository import ChunkStore
from app.embedding_pipeline.structural import JSONStructuralChunker

log = structlog.get_logger()


class DuplicateDocumentError(Exception):
    """Ya existe un documento con el mismo ``source_path``."""

    def __init__(self, document_id: int) -> None:
        super().__init__(f"Document already ingested (id={document_id})")
        self.document_id = document_id


class RagIngestService:
    """Persiste un presupuesto como documento + sus chunks embebidos."""

    def __init__(
        self,
        chunker: JSONStructuralChunker,
        embedder: OpenAIEmbedder,
        session_factory: async_sessionmaker[AsyncSession],
        store: ChunkStore,
    ) -> None:
        self._chunker = chunker
        self._embedder = embedder
        self._session_factory = session_factory
        self._store = store

    async def ingest(
        self,
        *,
        source_path: str,
        document_type: str,
        budget: Budget,
    ) -> IngestResponse:
        started = time.perf_counter()

        async with self._session_factory() as session, session.begin():
            # 1. Guardia de duplicado. Check-then-insert a nivel de aplicación:
            #    no es race-proof bajo ingestas idénticas concurrentes; a escala
            #    de formación es suficiente y es material del directo.
            existing_id = await self._store.find_document_id(session, source_path)
            if existing_id is not None:
                log.info(
                    "embedding.ingest.duplicate",
                    source_path=source_path,
                    document_id=existing_id,
                )
                raise DuplicateDocumentError(existing_id)

            # 2. Chunking estructural (un chunk por componente del presupuesto).
            chunks = self._chunker.chunk([budget])

            # 3. Embedding por lotes. El cliente de OpenAI es síncrono; se
            #    ejecuta en un hilo para no bloquear el event loop. Si falla, la
            #    transacción hace rollback: no queda documento huérfano.
            embedded = await asyncio.to_thread(self._embedder.embed_many, chunks)

            # 4. Documento + todos los chunks, aún sin commit.
            document_id = await self._store.persist_document_with_chunks(
                session,
                source_path=source_path,
                document_type=document_type,
                doc_metadata={
                    "budget_id": budget.budget_id,
                    "client_sector": budget.client_metadata.sector.value,
                    "year": budget.year,
                },
                embedded_chunks=embedded,
            )
            # 5. El commit ocurre al salir del `session.begin()`.

        elapsed_ms = int((time.perf_counter() - started) * 1000)
        total_tokens = sum(chunk.token_count for chunk in embedded)
        response = IngestResponse(
            document_id=document_id,
            chunks_created=len(embedded),
            embedding_dimension=len(embedded[0].embedding) if embedded else 0,
            ingestion_time_ms=elapsed_ms,
        )
        log.info(
            "embedding.ingest.end",
            source_path=source_path,
            total_tokens=total_tokens,
            estimated_cost_usd=estimate_cost_usd(total_tokens),
            **response.model_dump(),
        )
        return response
