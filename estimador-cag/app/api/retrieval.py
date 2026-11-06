"""Capa HTTP del retrieval filtrado (Sesión 9).

Router fino: valida el payload, aplica auth y rate limit, y delega el ranking y
el soft-fail en ``SemanticRetriever.retrieve_chunks``. Un corpus vacío o sin
chunks bajo el umbral es un 200 con ``low_confidence=True``, no un error.
"""

from __future__ import annotations

from typing import Annotated

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request, status

from app.api.rate_limiting import limiter
from app.api.security import require_retrieval_key
from app.config import get_settings
from app.dependencies import get_runtime_retrieval_config, get_semantic_retriever
from app.generation.rag.retriever import SemanticRetriever
from app.generation.rag.schemas import RetrievalOutcome, RetrievalQuery

log = structlog.get_logger()

router = APIRouter(
    prefix="/retrieval",
    tags=["retrieval"],
    dependencies=[Depends(require_retrieval_key)],
)


@router.post("/search", response_model=RetrievalOutcome)
@limiter.limit("120/minute")
async def search(
    request: Request,
    payload: RetrievalQuery,
    retriever: Annotated[SemanticRetriever | None, Depends(get_semantic_retriever)],
) -> RetrievalOutcome:
    """Busca chunks históricos con filtros estructurales y umbral de distancia."""
    if retriever is None:
        log.error("rag.retrieval.failed", reason="retriever_unavailable")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="No se pudo ejecutar el retrieval.",
        )

    settings = get_settings()
    runtime = get_runtime_retrieval_config()
    search_mode = payload.search_mode or runtime.effective_search_mode()
    rerank = payload.rerank if payload.rerank is not None else runtime.effective_rerank()

    try:
        return await retriever.retrieve_chunks(
            query_text=payload.query_text,
            search_mode=search_mode,
            rerank=rerank,
            top_k=payload.top_k,
            recall_k=settings.retrieval_recall_top_k,
            rerank_top_n=settings.rerank_top_n,
            distance_threshold=payload.distance_threshold,
            rrf_k=settings.rrf_k,
            sectors=payload.sectors,
            project_year_min=payload.project_year_min,
            project_year_max=payload.project_year_max,
            chunk_types=payload.chunk_types,
        )
    except Exception as exc:
        log.error(
            "rag.retrieval.failed",
            reason="retrieval_error",
            error_type=type(exc).__name__,
            error=str(exc)[:300],
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="No se pudo ejecutar el retrieval.",
        ) from exc
