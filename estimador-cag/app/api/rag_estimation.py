"""Capa HTTP de la estimación RAG end-to-end (Sesión 9).

Dos superficies:

* ``POST /estimate/from-transcript`` — pipeline completo, idempotente y con el
  rate limit más estricto (cada llamada dispara varias llamadas LLM).
* ``POST /estimate/stages/*`` — cada etapa por separado para el wizard, de modo
  que el usuario pueda revisar y re-ejecutar antes de gastar en generación.

La validación vive en los modelos Pydantic (422), la auth en
``require_estimate_key`` (401) y el rate limit en el decorador (429). Cualquier
fallo del pipeline se traduce a 502; el detalle queda en los logs.
"""

from __future__ import annotations

from typing import Annotated

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request, status

from app.api.rate_limiting import limiter
from app.api.security import require_estimate_key
from app.config import settings
from app.dependencies import get_semantic_retriever, get_token_encoder
from app.generation.rag.brief_extractor import compose_retrieval_text, extract_project_brief
from app.generation.rag.context_builder import assemble_context, fit_chunks_to_budget
from app.generation.rag.errors import RagPipelineError
from app.generation.rag.grounded_estimator import (
    estimate_from_transcript,
    generate_grounded_estimate,
)
from app.generation.rag.grounding import find_fabricated_citations, is_coherent_estimate
from app.generation.rag.retriever import SemanticRetriever
from app.generation.rag.schemas import (
    BriefOutcome,
    BriefRequest,
    ContextAssemblyOutcome,
    ContextAssemblyRequest,
    GroundedEstimate,
    GroundedGenerationOutcome,
    GroundedGenerationRequest,
    RetrievalOutcome,
    RetrievalQuery,
    TranscriptEstimateRequest,
)

log = structlog.get_logger()

router = APIRouter(
    prefix="/estimate",
    tags=["estimate"],
    dependencies=[Depends(require_estimate_key)],
)


def _pipeline_error(exc: Exception, *, surface: str) -> HTTPException:
    log.error(
        f"rag.{surface}.failed",
        error_type=type(exc).__name__,
        error=str(exc)[:300],
    )
    return HTTPException(
        status_code=status.HTTP_502_BAD_GATEWAY,
        detail="No se pudo completar la estimación RAG.",
    )


@router.post(
    "/from-transcript",
    response_model=GroundedEstimate,
    summary="Transcripción → estimación grounded con citas",
)
@limiter.limit("10/minute")
async def from_transcript(
    request: Request,
    payload: TranscriptEstimateRequest,
) -> GroundedEstimate:
    """Ejecuta el pipeline completo (idempotente por ``idempotency_key``)."""
    try:
        return await estimate_from_transcript(
            payload.transcript,
            idempotency_key=payload.idempotency_key,
        )
    except RagPipelineError as exc:
        raise _pipeline_error(exc, surface="estimate") from exc


@router.post("/stages/brief", response_model=BriefOutcome)
@limiter.limit("30/minute")
async def brief(
    request: Request,
    payload: BriefRequest,
) -> BriefOutcome:
    """Etapa 1: destila el brief y compone el texto de búsqueda."""
    try:
        project_brief = await extract_project_brief(payload.transcript)
    except RagPipelineError as exc:
        raise _pipeline_error(exc, surface="brief") from exc
    return BriefOutcome(
        brief=project_brief,
        retrieval_text=compose_retrieval_text(project_brief),
    )


@router.post("/stages/retrieve", response_model=RetrievalOutcome)
@limiter.limit("60/minute")
async def retrieve(
    request: Request,
    payload: RetrievalQuery,
    retriever: Annotated[SemanticRetriever | None, Depends(get_semantic_retriever)],
) -> RetrievalOutcome:
    """Etapa 2: retrieval filtrado con umbral y soft-fail."""
    if retriever is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="No se pudo ejecutar el retrieval.",
        )
    try:
        return await retriever.retrieve_chunks(
            query_text=payload.query_text,
            top_k=payload.top_k,
            distance_threshold=payload.distance_threshold,
            sectors=payload.sectors,
            project_year_min=payload.project_year_min,
            project_year_max=payload.project_year_max,
            chunk_types=payload.chunk_types,
        )
    except Exception as exc:
        raise _pipeline_error(exc, surface="retrieval") from exc


@router.post("/stages/assemble", response_model=ContextAssemblyOutcome)
@limiter.limit("60/minute")
async def assemble(
    request: Request,
    payload: ContextAssemblyRequest,
) -> ContextAssemblyOutcome:
    """Etapa 3: recorta por presupuesto y ensambla el bloque ``<source>``."""
    max_tokens = payload.max_context_tokens or settings.rag_max_context_tokens
    kept = fit_chunks_to_budget(payload.chunks, max_tokens, get_token_encoder())
    context_block = assemble_context(kept)
    token_count = len(get_token_encoder().encode(context_block))
    return ContextAssemblyOutcome(
        context_block=context_block,
        kept_chunks=kept,
        dropped_count=len(payload.chunks) - len(kept),
        token_count=token_count,
    )


@router.post("/stages/generate", response_model=GroundedGenerationOutcome)
@limiter.limit("15/minute")
async def generate(
    request: Request,
    payload: GroundedGenerationRequest,
) -> GroundedGenerationOutcome:
    """Etapa 4: genera la estimación y devuelve las señales de grounding.

    A diferencia del pipeline completo, aquí NO se reintenta automáticamente:
    el wizard muestra las citas inventadas y la incoherencia para que el usuario
    decida si re-ejecuta o corrige.
    """
    try:
        estimate = await generate_grounded_estimate(payload.context_block, payload.brief)
    except RagPipelineError as exc:
        raise _pipeline_error(exc, surface="generation") from exc
    return GroundedGenerationOutcome(
        estimate=estimate,
        fabricated_source_ids=find_fabricated_citations(estimate, payload.kept_chunks),
        coherent=is_coherent_estimate(estimate),
    )
