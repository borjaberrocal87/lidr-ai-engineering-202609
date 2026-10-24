"""Capa HTTP del pipeline de embeddings.

Router fino: orquesta ``chunker → embedder → respuesta`` y traduce los fallos a
códigos de estado. Aquí no vive lógica de negocio.
"""

from __future__ import annotations

from typing import Annotated

import structlog
from fastapi import APIRouter, Depends, HTTPException, status

from app.dependencies import ALL_STRATEGIES, build_chunkers, get_chunker, get_embedder
from app.embedding_pipeline.analysis.comparison import (
    ChunkingComparator,
    CompareRequest,
    CompareResponse,
)
from app.embedding_pipeline.embedder import OpenAIEmbedder, estimate_cost_usd
from app.embedding_pipeline.schemas import IngestRequest, IngestResponse, IngestStats
from app.embedding_pipeline.structural import JSONStructuralChunker

log = structlog.get_logger()

router = APIRouter(prefix="/embeddings", tags=["embeddings"])


@router.post("/ingest", response_model=IngestResponse)
def ingest(
    request: IngestRequest,
    chunker: Annotated[JSONStructuralChunker, Depends(get_chunker)],
    embedder: Annotated[OpenAIEmbedder | None, Depends(get_embedder)],
) -> IngestResponse:
    """Chunquea los presupuestos, embebe cada chunk y devuelve vectores + stats."""
    if embedder is None:
        log.error("embedding.ingest.failed", reason="embedder_unavailable")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="No se pudieron generar los embeddings.",
        )

    chunks = chunker.chunk(request.budgets)
    log.info(
        "embedding.ingest.received",
        total_budgets=len(request.budgets),
        total_chunks=len(chunks),
    )

    try:
        embedded = embedder.embed_many(chunks)
    except Exception as exc:
        log.error(
            "embedding.ingest.failed",
            reason="provider_error",
            error_type=type(exc).__name__,
            error=str(exc)[:300],
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="No se pudieron generar los embeddings.",
        ) from exc

    total_tokens = sum(chunk.token_count for chunk in embedded)
    stats = IngestStats(
        total_budgets=len(request.budgets),
        total_chunks=len(embedded),
        total_tokens=total_tokens,
        estimated_cost_usd=estimate_cost_usd(total_tokens),
    )
    log.info("embedding.ingest.end", **stats.model_dump())
    return IngestResponse(chunks=embedded, stats=stats)


@router.post("/compare", response_model=CompareResponse)
def compare(
    request: CompareRequest,
    embedder: Annotated[OpenAIEmbedder | None, Depends(get_embedder)],
) -> CompareResponse:
    """Compara varias estrategias de chunking sobre el mismo corpus.

    Sin ``queries`` solo se calculan estadísticos (no hace falta API key de
    embeddings). Con ``queries`` se calcula el top-k por similitud coseno y el
    embedder pasa a ser obligatorio.
    """
    if request.queries and embedder is None:
        log.error("embedding.compare.failed", reason="embedder_unavailable")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="No se pudieron generar los embeddings.",
        )

    names = request.strategies or ALL_STRATEGIES
    try:
        chunkers = build_chunkers(names)
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Estrategia desconocida: {exc.args[0]}",
        ) from exc
    except RuntimeError as exc:
        log.error("embedding.compare.failed", reason="missing_api_key", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc

    comparator = ChunkingComparator(chunkers, embedder)
    log.info(
        "embedding.compare.received",
        total_budgets=len(request.budgets),
        strategies=names,
        n_queries=len(request.queries),
    )
    try:
        stats = comparator.compute_stats(request.budgets)
        queries = comparator.run_queries(request.budgets, request.queries, request.top_k)
    except Exception as exc:
        log.error(
            "embedding.compare.failed",
            reason="comparison_error",
            error_type=type(exc).__name__,
            error=str(exc)[:300],
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="No se pudo ejecutar la comparativa de chunking.",
        ) from exc

    return CompareResponse(stats_per_strategy=stats, queries_per_strategy=queries)
