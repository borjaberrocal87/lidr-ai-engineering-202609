"""Capa HTTP del pipeline de embeddings.

Router fino: orquesta ``chunker → embedder → respuesta`` y traduce los fallos a
códigos de estado. Aquí no vive lógica de negocio.
"""

from __future__ import annotations

from typing import Annotated

import structlog
from fastapi import APIRouter, Depends, HTTPException, status

from app.dependencies import get_chunker, get_embedder
from app.embedding_pipeline.chunker import JSONStructuralChunker
from app.embedding_pipeline.embedder import OpenAIEmbedder, estimate_cost_usd
from app.embedding_pipeline.schemas import IngestRequest, IngestResponse, IngestStats

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
        # Sin clave/endpoint de embeddings configurados. Mensaje genérico al
        # cliente; el motivo real queda en los logs.
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
