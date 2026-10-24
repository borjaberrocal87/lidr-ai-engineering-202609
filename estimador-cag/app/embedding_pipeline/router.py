"""Capa HTTP del pipeline de embeddings.

Router fino: orquesta ``chunker → embedder → persist`` en la ingesta y delega
la búsqueda en su router propio. Aquí no vive lógica de negocio.
"""

from __future__ import annotations

from typing import Annotated

import structlog
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse

from app.dependencies import (
    ALL_STRATEGIES,
    build_chunkers,
    get_embedder,
    get_rag_ingest_service,
)
from app.embedding_pipeline.analysis.comparison import (
    ChunkingComparator,
    CompareRequest,
    CompareResponse,
)
from app.embedding_pipeline.embedder import OpenAIEmbedder
from app.embedding_pipeline.ingest_service import (
    DuplicateDocumentError,
    RagIngestService,
)
from app.embedding_pipeline.schemas import IngestRequest, IngestResponse

log = structlog.get_logger()

router = APIRouter(prefix="/embeddings", tags=["embeddings"])


@router.post("/ingest", response_model=IngestResponse)
async def ingest(
    request: IngestRequest,
    service: Annotated[RagIngestService | None, Depends(get_rag_ingest_service)],
) -> IngestResponse | JSONResponse:
    """Chunquea, embebe y persiste un presupuesto en una sola transacción."""
    if service is None:
        log.error("embedding.ingest.failed", reason="ingest_service_unavailable")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="No se pudieron persistir los embeddings.",
        )

    try:
        return await service.ingest(
            source_path=request.source_path,
            document_type=request.document_type,
            budget=request.content,
        )
    except DuplicateDocumentError as exc:
        # El contrato pide ``detail`` y ``document_id`` en el nivel superior,
        # no anidados dentro de ``detail``. Por eso se devuelve un JSONResponse
        # en vez de un HTTPException con dict.
        return JSONResponse(
            status_code=status.HTTP_409_CONFLICT,
            content={
                "detail": "Document already ingested",
                "document_id": exc.document_id,
            },
        )
    except Exception as exc:
        log.error(
            "embedding.ingest.failed",
            reason="ingest_error",
            error_type=type(exc).__name__,
            error=str(exc)[:300],
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="No se pudieron persistir los embeddings.",
        ) from exc


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
