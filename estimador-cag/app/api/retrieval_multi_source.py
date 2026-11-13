"""Capa HTTP de la recuperación multifuente (Sesión 10).

El contrapunto del ``POST /v1/retrieval/search`` de la Sesión 9: añade plan de
consulta (expansión/descomposición), enrutado en cascada entre fuentes, filtros
duros de metadata, fusión diferenciada, reranking y vigencia temporal. Cada
etapa es conmutable. El endpoint de la Sesión 9 se queda intacto por
compatibilidad.

Transporte fino: auth en ``require_retrieval_key`` (401), rate limit en el
decorador (429) y validación en el modelo de petición (422). La respuesta no
solo trae los chunks: también CÓMO se obtuvieron (nivel y razón del enrutado,
técnica, sub-consultas y cardinalidad por fuente), para que la sesión en vivo
pueda mostrar cada decisión.
"""

from __future__ import annotations

from datetime import date, datetime

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

from app.api.rate_limiting import limiter
from app.api.security import require_retrieval_key
from app.config import get_settings
from app.dependencies import (
    get_chunk_store,
    get_embedder,
    get_llm_wrapper,
    get_reranker,
    get_runtime_retrieval_config,
)
from app.foundation.persistence.database import get_async_session_factory
from app.generation.rag.errors import RagRetrievalError
from app.generation.rag.retrieval.multi_source import StageConfig, retrieve_multi_source
from app.generation.rag.retrieval.sources import Collection, HardFilters
from app.generation.rag.schemas import RetrievedChunk

log = structlog.get_logger()

router = APIRouter(
    prefix="/retrieval",
    tags=["retrieval"],
    dependencies=[Depends(require_retrieval_key)],
)


class MultiSourceRequest(BaseModel):
    """Payload de ``POST /api/v1/retrieval/multi-source``."""

    query_text: str = Field(min_length=10, max_length=2000)
    # Nivel 0 del enrutado: el llamante nombra las fuentes y no se clasifica nada.
    collections: list[Collection] | None = Field(
        default=None, description="Fuentes explícitas (omite el enrutado)."
    )
    top_k: int = Field(default=5, ge=1, le=30)
    # Filtros duros de metadata (se embeben en la consulta; nunca post-filtran).
    technologies: list[str] | None = None
    sectors: list[str] | None = None
    min_date: date | None = Field(
        default=None,
        description="Excluye chunks anteriores a esta fecha (presupuestos: por año).",
    )
    doc_version: str | None = Field(
        default=None, description="Filtro de versión para documentación técnica."
    )
    # Overrides por petición (None ⇒ cae a runtime y luego a settings).
    search_mode: str | None = Field(default=None, description="'vector' o 'hybrid'.")
    rerank: bool | None = None
    routing_enabled: bool | None = None
    query_plan_enabled: bool | None = None
    freshness_enabled: bool | None = None


class RoutingInfo(BaseModel):
    level: str
    targets: list[str]
    reason: str


class SubQueryInfo(BaseModel):
    topic: str
    query: str


class MultiSourceResult(BaseModel):
    """Resultados + diagnóstico de cómo se recuperaron."""

    chunks: list[RetrievedChunk]
    low_confidence: bool
    routing: RoutingInfo
    technique: str
    subqueries: list[SubQueryInfo]
    cardinality: dict[str, int] = Field(
        description="Candidatos por fuente tras los filtros duros "
        "(un 0 es un vacío silencioso: hay que mirarlo)."
    )


@router.post("/multi-source", response_model=MultiSourceResult)
@limiter.limit("120/minute")
async def multi_source(
    request: Request,
    payload: MultiSourceRequest,
) -> MultiSourceResult:
    """Ejecuta el pipeline multifuente; precedencia explícito → runtime → .env."""
    embedder = get_embedder()
    if embedder is None:
        log.error("multi_source_failed", reason="embedder_unavailable")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Embedding service is not available.",
        )

    settings = get_settings()
    runtime = get_runtime_retrieval_config()
    search_mode = payload.search_mode or runtime.effective_search_mode()
    rerank = payload.rerank if payload.rerank is not None else runtime.effective_rerank()
    routing_enabled = (
        payload.routing_enabled
        if payload.routing_enabled is not None
        else runtime.effective_routing()
    )
    query_plan_enabled = (
        payload.query_plan_enabled
        if payload.query_plan_enabled is not None
        else runtime.effective_query_plan()
    )
    freshness_enabled = (
        payload.freshness_enabled
        if payload.freshness_enabled is not None
        else runtime.effective_freshness()
    )

    stages = StageConfig.from_settings(
        settings,
        search_mode=search_mode,
        rerank=rerank,
        routing_enabled=routing_enabled,
        query_plan_enabled=query_plan_enabled,
        freshness_enabled=freshness_enabled,
        top_k=payload.top_k,
    )
    hard_filters = HardFilters(
        technologies=tuple(payload.technologies or ()),
        sectors=tuple(payload.sectors or ()),
        min_date=payload.min_date,
        version=payload.doc_version,
    )

    try:
        outcome = await retrieve_multi_source(
            query_text=payload.query_text,
            embedder=embedder,
            store=get_chunk_store(),
            session_factory=get_async_session_factory(),
            stages=stages,
            reference_date=datetime.now().date(),
            llm_wrapper=get_llm_wrapper(),
            reranker=get_reranker() if rerank else None,
            explicit_collections=payload.collections,
            hard_filters=hard_filters,
        )
    except RagRetrievalError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Retrieval failed.",
        ) from exc
    except Exception as exc:
        log.error(
            "multi_source_failed",
            reason="search_error",
            error_type=type(exc).__name__,
            error=str(exc)[:300],
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Failed to run multi-source retrieval.",
        ) from exc

    return MultiSourceResult(
        chunks=outcome.chunks,
        low_confidence=outcome.low_confidence,
        routing=RoutingInfo(
            level=outcome.routing.level,
            targets=[t.value for t in outcome.routing.targets],
            reason=outcome.routing.reason,
        ),
        technique=outcome.technique,
        subqueries=[SubQueryInfo(topic=sq.topic, query=sq.query) for sq in outcome.subqueries],
        cardinality=outcome.cardinality,
    )
