"""Retriever semántico sobre el store de pgvector (Sesión 08).

Embebe la query con el MISMO modelo usado en la ingesta (mezclar modelos de
embedding hace que las distancias no signifiquen nada) y rankea los chunks por
distancia coseno vía SQL. Sin índice vectorial y sin filtros de metadata: ambas
cosas se construyen en directo sobre este baseline.
"""

from __future__ import annotations

import asyncio
import time

import structlog
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.generation.rag.embedding.embedder import OpenAIEmbedder
from app.generation.rag.schemas import (
    RetrievalOutcome,
    RetrievedChunk,
    SearchHit,
    SearchResponse,
)
from app.generation.rag.store.repository import ChunkMatch, ChunkStore

log = structlog.get_logger()


class SemanticRetriever:
    """k-NN: embebe la query y rankea chunks por distancia coseno."""

    def __init__(
        self,
        embedder: OpenAIEmbedder,
        session_factory: async_sessionmaker[AsyncSession],
        store: ChunkStore,
    ) -> None:
        self._embedder = embedder
        self._session_factory = session_factory
        self._store = store

    async def search(self, *, query: str, k: int) -> SearchResponse:
        started = time.perf_counter()

        # Cliente de OpenAI síncrono → hilo, mismo razonamiento que en ingest.
        query_vector = await asyncio.to_thread(self._embedder.embed_one, query)

        async with self._session_factory() as session:
            matches = await self._store.search(
                session,
                query_vector=query_vector,
                k=k,
            )

        elapsed_ms = int((time.perf_counter() - started) * 1000)
        response = SearchResponse(
            query=query,
            k=k,
            search_time_ms=elapsed_ms,
            results=[
                SearchHit(
                    chunk_id=match.chunk_id,
                    document_id=match.document_id,
                    chunk_type=match.chunk_type,
                    content=match.content,
                    distance=match.distance,
                    metadata=match.metadata,
                )
                for match in matches
            ],
        )
        log.info(
            "embedding.search.end",
            query=query[:80],
            k=k,
            results=len(response.results),
            search_mode=self._store.search_mode,
            search_time_ms=elapsed_ms,
        )
        return response

    @staticmethod
    def _to_retrieved_chunk(match: ChunkMatch) -> RetrievedChunk:
        """Aplana sector/año desde el JSONB del chunk a la forma tipada S09."""
        raw_year = match.metadata.get("year")
        try:
            project_year = int(raw_year) if raw_year is not None else None
        except (TypeError, ValueError):
            project_year = None
        sector = match.metadata.get("client_sector")
        return RetrievedChunk(
            id=match.chunk_id,
            document_id=match.document_id,
            content=match.content,
            sector=str(sector) if sector is not None else None,
            project_year=project_year,
            chunk_type=match.chunk_type,
            distance=match.distance,
        )

    async def retrieve_chunks(
        self,
        *,
        query_text: str,
        top_k: int = 10,
        distance_threshold: float = 0.6,
        sectors: list[str] | None = None,
        project_year_min: int | None = None,
        project_year_max: int | None = None,
        chunk_types: list[str] | None = None,
    ) -> RetrievalOutcome:
        """Retrieval S09: filtros estructurales + umbral y soft-fail.

        Si ningún chunk cruza el umbral, ``low_confidence`` queda a ``True`` y
        la lista vacía: el pipeline decide entonces generar una estimación
        insuficiente en vez de inventar cifras.
        """
        started = time.perf_counter()
        query_vector = await asyncio.to_thread(self._embedder.embed_one, query_text)

        async with self._session_factory() as session:
            matches, candidates = await self._store.search_with_filters(
                session,
                query_vector=query_vector,
                top_k=top_k,
                distance_threshold=distance_threshold,
                sectors=sectors,
                project_year_min=project_year_min,
                project_year_max=project_year_max,
                chunk_types=chunk_types,
            )

        chunks = [self._to_retrieved_chunk(match) for match in matches]
        outcome = RetrievalOutcome(
            chunks=chunks,
            low_confidence=not chunks,
            candidates_evaluated=candidates,
        )
        log.info(
            "rag.retrieval.end",
            query=query_text[:80],
            top_k=top_k,
            distance_threshold=distance_threshold,
            chunks=len(chunks),
            candidates_evaluated=candidates,
            low_confidence=outcome.low_confidence,
            search_mode=self._store.search_mode,
            search_time_ms=int((time.perf_counter() - started) * 1000),
        )
        return outcome
