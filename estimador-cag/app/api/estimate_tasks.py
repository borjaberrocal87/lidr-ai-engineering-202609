"""``POST /v1/estimate/tasks/hours`` — horas por tarea vía búsqueda vectorial (S10).

La generación estructura-primero del wizard devuelve módulos → tareas SIN horas.
Este endpoint las rellena: cada tarea se busca contra el corpus histórico de
tareas y recibe un consenso ponderado por distancia de las horas de sus vecinos
más cercanos, con un score de fiabilidad. Las tareas sin vecino bajo el umbral
vuelven con ``has_match=False`` (sin horas) para que la UI marque la fila en rojo.

Transporte fino, misma postura que los routers hermanos: validación en
``TaskHoursRequest`` (422), auth en ``require_estimate_key`` (401), rate limit en
el decorador (429) y fallos del pipeline → 502. La búsqueda por tarea reutiliza
el retriever de la Sesión 9; el consenso vive en ``task_hours``.
"""

from __future__ import annotations

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request, status

from app.api.rate_limiting import limiter
from app.api.security import require_estimate_key
from app.config import get_settings
from app.dependencies import (
    get_chunk_store,
    get_embedder,
    get_reranker,
    get_runtime_retrieval_config,
)
from app.foundation.persistence.database import get_async_session_factory
from app.generation.rag.errors import RagRetrievalError
from app.generation.rag.schemas import TaskHoursRequest, TaskHoursResult
from app.generation.rag.task_hours import estimate_all

log = structlog.get_logger()

router = APIRouter(
    prefix="/estimate/tasks",
    tags=["estimate"],
    dependencies=[Depends(require_estimate_key)],
)


@router.post("/hours", response_model=TaskHoursResult)
@limiter.limit("30/minute")
async def hours(request: Request, payload: TaskHoursRequest) -> TaskHoursResult:
    """Estima las horas de todas las tareas enviadas desde el corpus histórico."""
    embedder = get_embedder()
    if embedder is None:
        log.error("stage_failed", stage="task_hours", reason="embedder_unavailable")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Embedding service is not available.",
        )

    settings = get_settings()
    runtime = get_runtime_retrieval_config()
    search_mode = runtime.effective_search_mode()
    rerank = runtime.effective_rerank()
    task_count = sum(len(module.tasks) for module in payload.modules)

    try:
        return await estimate_all(
            payload.modules,
            store=get_chunk_store(),
            session_factory=get_async_session_factory(),
            embedder=embedder,
            reranker=get_reranker() if rerank else None,
            top_k=runtime.effective_task_hours_top_k(),
            distance_threshold=runtime.effective_task_hours_distance_threshold(),
            search_mode=search_mode,
            rerank=rerank,
            recall_k=settings.retrieval_recall_top_k,
            rerank_top_n=settings.rerank_top_n,
            rrf_k=settings.rrf_k,
        )
    except RagRetrievalError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Retrieval failed.",
        ) from exc
    except Exception as exc:
        log.error(
            "stage_failed",
            stage="task_hours",
            tasks=task_count,
            error_type=type(exc).__name__,
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Failed to estimate task hours.",
        ) from exc
