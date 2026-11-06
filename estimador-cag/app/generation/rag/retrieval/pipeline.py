"""Recall-then-rerank retrieval pipeline (Session 10).

The single ``retrieve()`` entrypoint composes the four configurations the
exercise measures, behind two switches the caller resolves (request → runtime →
settings):

* ``search_mode="vector"`` — dense k-NN only (the Session 9 behaviour).
* ``search_mode="hybrid"`` — dense + lexical branches fused with RRF.
* ``rerank=False`` — keep the top ``top_k`` of the recall ordering.
* ``rerank=True``  — recall WIDE (``recall_k``, e.g. 50) then let the
  cross-encoder rescore and keep the top ``rerank_top_n`` (e.g. 5). The recall
  stage's only job is not to lose the relevant document; the reranker's job is
  to float it up.

It returns the same :class:`RetrievalOutcome` as ``retrieve_chunks`` so every
existing consumer (orchestrator, stage endpoints) is unaffected.
"""

from __future__ import annotations

import asyncio
import time

import structlog
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.generation.rag.retrieval.fusion import reciprocal_rank_fusion
from app.generation.rag.retrieval.reranker import CrossEncoderReranker
from app.generation.rag.schemas import RetrievalOutcome, RetrievedChunk
from app.generation.rag.store.repository import (
    LEXICAL_ONLY_DISTANCE,
    ChunkMatch,
    ChunkStore,
)

log = structlog.get_logger()


def _match_to_chunk(match: ChunkMatch, *, distance: float) -> RetrievedChunk:
    """Map a store DTO onto the typed retrieval contract."""
    raw_year = match.metadata.get("year")
    try:
        project_year = int(raw_year) if raw_year is not None else None
    except (TypeError, ValueError):
        project_year = None
    sector = match.metadata.get("client_sector")
    budget_id = match.metadata.get("budget_id")
    return RetrievedChunk(
        id=match.chunk_id,
        document_id=match.document_id,
        content=match.content,
        sector=str(sector) if sector is not None else None,
        project_year=project_year,
        chunk_type=match.chunk_type,
        distance=distance,
        budget_id=str(budget_id) if budget_id is not None else None,
    )


async def retrieve(
    *,
    store: ChunkStore,
    session_factory: async_sessionmaker[AsyncSession],
    query_embedding: list[float],
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
    reranker: CrossEncoderReranker | None = None,
) -> RetrievalOutcome:
    """Run vector/hybrid retrieval with optional cross-encoder reranking.

    ``query_text`` is required for the lexical branch and the reranker even when
    ``search_mode`` is ``"vector"`` (the reranker scores against the raw query).
    ``reranker`` is injected by the caller; when ``rerank`` is True it must be
    present. Returns the top results best-first (cross-encoder order when
    reranking, else distance/RRF order).
    """
    if search_mode not in {"vector", "hybrid"}:
        raise ValueError(f"Unsupported search_mode: {search_mode!r}")
    if rerank and reranker is None:
        raise ValueError("rerank=True requires an injected reranker")

    started = time.perf_counter()

    # Recall wide whenever a later stage (fusion or rerank) will re-sort; recall
    # exactly top_k only for the plain vector path, where the order is final.
    wide = rerank or search_mode == "hybrid"
    vector_limit = recall_k if wide else top_k

    async with session_factory() as session:
        vector_matches, candidates_evaluated = await store.search_with_filters(
            session,
            query_vector=query_embedding,
            top_k=vector_limit,
            distance_threshold=distance_threshold,
            sectors=sectors,
            project_year_min=project_year_min,
            project_year_max=project_year_max,
            chunk_types=chunk_types,
        )
        lexical_matches: list[ChunkMatch] = []
        if search_mode == "hybrid":
            lexical_matches = await store.search_lexical(
                session,
                query_text=query_text,
                top_k=recall_k,
                sectors=sectors,
                project_year_min=project_year_min,
                project_year_max=project_year_max,
                chunk_types=chunk_types,
            )

    # Build the candidate pool once (id → chunk); vector distance wins over the
    # lexical sentinel when an id is in both branches.
    candidates: dict[int, RetrievedChunk] = {}
    for match in vector_matches:
        candidates[match.chunk_id] = _match_to_chunk(match, distance=match.distance)
    for match in lexical_matches:
        candidates.setdefault(
            match.chunk_id,
            _match_to_chunk(match, distance=LEXICAL_ONLY_DISTANCE),
        )

    # Recall ordering: RRF for hybrid, raw distance order for vector.
    if search_mode == "hybrid":
        fused = reciprocal_rank_fusion(
            [
                [match.chunk_id for match in vector_matches],
                [match.chunk_id for match in lexical_matches],
            ],
            k=rrf_k,
        )
        ordered = [candidates[chunk_id] for chunk_id, _score in fused]
    else:
        ordered = [candidates[match.chunk_id] for match in vector_matches]

    if rerank and reranker is not None:
        final = await asyncio.to_thread(
            reranker.rerank,
            query_text,
            ordered[:recall_k],
            top_n=rerank_top_n,
        )
    else:
        final = ordered[:top_k]

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    log.info(
        "rag.retrieval.end",
        query=query_text[:80],
        search_mode=search_mode,
        rerank=rerank,
        vector_hits=len(vector_matches),
        lexical_hits=len(lexical_matches),
        results=len(final),
        candidates_evaluated=candidates_evaluated,
        store_search_mode=store.search_mode,
        search_time_ms=elapsed_ms,
    )
    return RetrievalOutcome(
        chunks=final,
        low_confidence=not final,
        candidates_evaluated=candidates_evaluated,
    )
