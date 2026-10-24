"""Capa HTTP de la búsqueda semántica (Sesión 08).

Router fino: la validación vive en ``SearchRequest`` (k entre 1 y 50 → 422) y
el ranking vive en ``SemanticRetriever``. Un corpus vacío es un 200 con
``results: []``, no un error.
"""

from __future__ import annotations

from typing import Annotated

import structlog
from fastapi import APIRouter, Depends, HTTPException, status

from app.dependencies import get_semantic_retriever
from app.embedding_pipeline.retriever import SemanticRetriever
from app.embedding_pipeline.schemas import SearchRequest, SearchResponse

log = structlog.get_logger()

router = APIRouter(prefix="/search", tags=["search"])


@router.post("", response_model=SearchResponse)
async def search(
    request: SearchRequest,
    retriever: Annotated[SemanticRetriever | None, Depends(get_semantic_retriever)],
) -> SearchResponse:
    """Devuelve los ``k`` chunks más cercanos a la query por distancia coseno."""
    if retriever is None:
        log.error("embedding.search.failed", reason="retriever_unavailable")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="No se pudo ejecutar la búsqueda semántica.",
        )

    try:
        return await retriever.search(query=request.query, k=request.k)
    except Exception as exc:
        log.error(
            "embedding.search.failed",
            reason="search_error",
            error_type=type(exc).__name__,
            error=str(exc)[:300],
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="No se pudo ejecutar la búsqueda semántica.",
        ) from exc
