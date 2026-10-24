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

from app.embedding_pipeline.embedder import OpenAIEmbedder
from app.embedding_pipeline.schemas import SearchHit, SearchResponse
from app.embedding_pipeline.store.repository import ChunkStore

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
            search_time_ms=elapsed_ms,
        )
        return response
