"""Capa HTTP del grafo de estimación (Sesión 13).

``POST /api/v1/graph/estimate/from-transcript`` — mismo contrato externo que el
resto de estimadores (transcripción dentro, estimación estructurada con su
``status`` fuera), pero por debajo conduce el ``StateGraph`` compilado al
arrancar (``app.state.graph``) usando el identificador de la estimación como
``thread_id`` del checkpointer, de modo que la ejecución queda persistida.

Transporte fino, misma postura que los routers hermanos: validación en el request
(422), rate limit en el decorador (429), grafo no disponible → 503 y fallo de un
nodo/LLM → 502. La respuesta incluye ``run_text`` renderizado en servidor, como el
``trace_text`` del agente de la Sesión 12.
"""

from __future__ import annotations

from uuid import uuid4

import structlog
from fastapi import APIRouter, HTTPException, Request, status

from app.api.rate_limiting import limiter
from app.generation.agentic.estimation_graph.render import render_run
from app.generation.agentic.estimation_graph.schemas import (
    BudgetMatchView,
    ComponentOut,
    GraphEstimateRequest,
    GraphEstimateResponse,
    NodeRunRecord,
)

log = structlog.get_logger(__name__)

router = APIRouter(prefix="/graph", tags=["graph"])


@router.post(
    "/estimate/from-transcript",
    response_model=GraphEstimateResponse,
    summary="Transcripción → estimación del grafo con su status",
)
@limiter.limit("10/minute")
async def estimate_with_graph(
    request: Request,
    payload: GraphEstimateRequest,
) -> GraphEstimateResponse:
    """Ejecuta el grafo de estimación y devuelve estimación + status + run_text."""
    graph = getattr(request.app.state, "graph", None)
    if graph is None:
        log.error("graph.unavailable")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="El grafo de estimación no está disponible en este despliegue.",
        )

    estimation_id = payload.estimation_id or str(uuid4())
    config = {"configurable": {"thread_id": estimation_id}}
    try:
        state = await graph.ainvoke(
            {"transcript": payload.transcript, "estimation_id": estimation_id},
            config,
        )
    except Exception as exc:
        log.error(
            "graph.estimate.failed",
            estimation_id=estimation_id,
            error_type=type(exc).__name__,
            error=str(exc)[:300],
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="No se pudo completar la estimación con el grafo.",
        ) from exc

    return GraphEstimateResponse(
        estimation_id=estimation_id,
        status=state.get("status"),
        estimate=state.get("estimate"),
        requirements=state.get("requirements") or [],
        components=[ComponentOut(**component) for component in state.get("components") or []],
        budget_matches=[BudgetMatchView(**match) for match in state.get("budget_matches") or []],
        node_runs=[NodeRunRecord(**run) for run in state.get("node_runs") or []],
        errors=state.get("errors") or [],
        run_text=render_run(state),
    )
