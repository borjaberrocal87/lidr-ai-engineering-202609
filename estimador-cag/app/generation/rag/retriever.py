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
from app.generation.rag.retrieval.pipeline import retrieve as retrieve_pipeline
from app.generation.rag.retrieval.reranker import CrossEncoderReranker
from app.generation.rag.schemas import (
    RetrievalOutcome,
    SearchHit,
    SearchResponse,
)
from app.generation.rag.store.repository import ChunkStore

log = structlog.get_logger()


class SemanticRetriever:
    """k-NN: embebe la query y rankea chunks por distancia coseno."""

    def __init__(
        self,
        embedder: OpenAIEmbedder,
        session_factory: async_sessionmaker[AsyncSession],
        store: ChunkStore,
        reranker: CrossEncoderReranker | None = None,
    ) -> None:
        self._embedder = embedder
        self._session_factory = session_factory
        self._store = store
        self._reranker = reranker

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

    async def retrieve(
        self,
        *,
        query_text: str,
        search_mode: str = "vector",
        rerank: bool = False,
        top_k: int = 10,
        recall_k: int = 50,
        rerank_top_n: int = 5,
        distance_threshold: float = 0.6,
        rrf_k: int = 60,
        sectors: list[str] | None = None,
        project_year_min: int | None = None,
        project_year_max: int | None = None,
        chunk_types: list[str] | None = None,
    ) -> RetrievalOutcome:
        """Embed the query and run vector/hybrid retrieval with optional rerank.

        ``search_mode`` and ``rerank`` are resolved by the caller (request →
        runtime → settings). The reranker is injected in the constructor; when
        ``rerank`` is True and no reranker exists, the pipeline raises.
        """
        query_embedding = await asyncio.to_thread(self._embedder.embed_one, query_text)
        return await retrieve_pipeline(
            store=self._store,
            session_factory=self._session_factory,
            query_embedding=query_embedding,
            query_text=query_text,
            search_mode=search_mode,
            rerank=rerank,
            top_k=top_k,
            recall_k=recall_k,
            rerank_top_n=rerank_top_n,
            distance_threshold=distance_threshold,
            rrf_k=rrf_k,
            sectors=sectors,
            project_year_min=project_year_min,
            project_year_max=project_year_max,
            chunk_types=chunk_types,
            reranker=self._reranker,
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
        search_mode: str = "vector",
        rerank: bool = False,
        recall_k: int = 50,
        rerank_top_n: int = 5,
        rrf_k: int = 60,
    ) -> RetrievalOutcome:
        """Retrieval with optional Session 10 hybrid search and reranking.

        The defaults (``vector``, no rerank) preserve the Session 09 behaviour
        for the grounded orchestrator, the stage endpoints and existing callers.
        If no chunk crosses the threshold, ``low_confidence`` is True and the
        list is empty so the caller can short-circuit to an insufficient
        estimate instead of inventing numbers.
        """
        return await self.retrieve(
            query_text=query_text,
            search_mode=search_mode,
            rerank=rerank,
            top_k=top_k,
            recall_k=recall_k,
            rerank_top_n=rerank_top_n,
            distance_threshold=distance_threshold,
            rrf_k=rrf_k,
            sectors=sectors,
            project_year_min=project_year_min,
            project_year_max=project_year_max,
            chunk_types=chunk_types,
        )
