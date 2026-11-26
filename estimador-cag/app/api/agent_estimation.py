"""Capa HTTP del agente de estimación (Sesión 12).

Tres superficies:

* ``POST /agent/estimate/from-transcript`` — el shot autónomo (pre-work): una
  transcripción entra, un informe con traza sale. Se conserva como comparativa en
  vivo frente al flujo por fases.
* ``POST /agent/estimate/structure`` — fase 1. El agente propone la estructura
  módulo→tarea; devuelve el MISMO ``GroundedGenerationOutcome`` que
  ``/estimate/stages/structure`` (más la traza), para que el wizard la parsee sin
  un contrato nuevo.
* ``POST /agent/estimate/hours`` — fase 2 HÍBRIDA. El consenso determinista corre
  primero y el agente solo re-busca las tareas no groundadas; devuelve el mismo
  ``TaskHoursResult`` que ``/estimate/tasks/hours`` (más la traza).

La composición de los hermanos ``agentic`` + ``rag`` vive en el conductor
(``app.domain.agent_estimation``), no aquí. Transporte fino, misma postura que los
routers hermanos: validación en el request (422), auth en ``require_estimate_key``
(401), rate limit en el decorador (429), fallos del bucle → 502 y cliente sin
configurar → 500.
"""

from __future__ import annotations

from typing import Annotated

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request, status
from openai import AsyncOpenAI
from pydantic import BaseModel, ConfigDict, Field

from app.api.rate_limiting import limiter
from app.api.security import require_estimate_key
from app.config import settings
from app.dependencies import (
    get_async_openai_client,
    get_budget_search_backend,
    get_chunk_store,
    get_embedder,
    get_reranker,
    get_runtime_retrieval_config,
    make_budget_search_backend,
)
from app.domain.agent_estimation import (
    agent_estimate_task_hours,
    agent_propose_structure,
)
from app.foundation.persistence.database import get_async_session_factory
from app.generation.agentic.estimation_agent.runner import run_estimation_agent
from app.generation.agentic.estimation_agent.schemas import AgentRunResult
from app.generation.agentic.estimation_agent.tools import BudgetSearchBackend
from app.generation.rag.grounding import (
    is_coherent_estimate,
    verify_citations_for_chunks,
)
from app.generation.rag.schemas import (
    AgentOverrides,
    GroundedGenerationOutcome,
    StructureRequest,
    TaskHoursRequest,
    TaskHoursResult,
)
from app.generation.rag.task_hours import distance_weighted_consensus, estimate_all

log = structlog.get_logger(__name__)

router = APIRouter(prefix="/agent", tags=["agent"])


class AgentEstimateRequest(BaseModel):
    """Payload de ``POST /api/v1/agent/estimate/from-transcript``."""

    model_config = ConfigDict(extra="forbid")

    transcript: str = Field(min_length=100, max_length=50_000)


class AgentEstimateResponse(AgentRunResult):
    """Informe + traza estructurada + traza renderizada para la UI."""

    trace_text: str = Field(description="AgentTrace.render() output, ready to download.")


def _not_configured() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="El agente de estimación no está configurado en este despliegue.",
    )


@router.post(
    "/estimate/from-transcript",
    response_model=AgentEstimateResponse,
    summary="Transcripción → informe del agente con traza",
)
@limiter.limit("10/minute")
async def estimate_with_agent(
    request: Request,
    payload: AgentEstimateRequest,
    client: Annotated[AsyncOpenAI | None, Depends(get_async_openai_client)],
    backend: Annotated[BudgetSearchBackend | None, Depends(get_budget_search_backend)],
) -> AgentEstimateResponse:
    """Conduce el bucle del agente y devuelve el informe con su traza."""
    if client is None or backend is None:
        raise _not_configured()

    try:
        result = await run_estimation_agent(
            payload.transcript,
            client=client,
            model=settings.agent_model,
            reasoning_effort=settings.agent_reasoning_effort,
            max_iterations=settings.agent_max_iterations,
            backend=backend,
        )
    except Exception as exc:
        log.error(
            "agent.estimate.failed",
            error_type=type(exc).__name__,
            error=str(exc)[:300],
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="No se pudo completar la estimación con el agente.",
        ) from exc

    return AgentEstimateResponse(
        **result.model_dump(),
        trace_text=result.trace.render(),
    )


# --- Fase 1 y fase 2: el agente conduce el wizard -----------------------------


def _agent_knobs(overrides: AgentOverrides | None) -> tuple[str, str, int]:
    """Resuelve (model, effort, max_iterations) desde el perfil o los defaults."""
    model = overrides.model if overrides and overrides.model else settings.agent_model
    effort = (
        overrides.reasoning_effort
        if overrides and overrides.reasoning_effort
        else settings.agent_reasoning_effort
    )
    max_iterations = (
        overrides.max_iterations
        if overrides and overrides.max_iterations
        else settings.agent_max_iterations
    )
    return model, effort, max_iterations


@router.post(
    "/estimate/structure",
    response_model=GroundedGenerationOutcome,
    dependencies=[Depends(require_estimate_key)],
    summary="Fase 1: el agente propone la estructura módulo→tarea",
)
@limiter.limit("15/minute")
async def structure_with_agent(
    request: Request,
    payload: StructureRequest,
    client: Annotated[AsyncOpenAI | None, Depends(get_async_openai_client)],
) -> GroundedGenerationOutcome:
    """El agente descompone el brief y lo mapea al contrato del wizard."""
    if client is None:
        raise _not_configured()

    model, effort, _max_iterations = _agent_knobs(payload.agent)
    try:
        proposal = await agent_propose_structure(
            payload.query,
            client=client,
            model=model,
            reasoning_effort=effort,
        )
    except Exception as exc:
        log.error(
            "agent.structure.failed",
            error_type=type(exc).__name__,
            error=str(exc)[:300],
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="No se pudo generar la estructura con el agente.",
        ) from exc

    citation_report = verify_citations_for_chunks(proposal.estimate, [])
    return GroundedGenerationOutcome(
        estimate=proposal.estimate,
        fabricated_source_ids=[],
        citation_report=citation_report,
        coherent=is_coherent_estimate(proposal.estimate),
        agent_trace=proposal.trace,
    )


@router.post(
    "/estimate/hours",
    response_model=TaskHoursResult,
    dependencies=[Depends(require_estimate_key)],
    summary="Fase 2: horas híbridas (determinista + recuperación del agente)",
)
@limiter.limit("15/minute")
async def hours_with_agent(
    request: Request,
    payload: TaskHoursRequest,
    client: Annotated[AsyncOpenAI | None, Depends(get_async_openai_client)],
) -> TaskHoursResult:
    """Consenso determinista por tarea y, para las no groundadas, el agente."""
    if client is None:
        raise _not_configured()
    embedder = get_embedder()
    if embedder is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Embedding service is not available.",
        )

    overrides = payload.agent
    backend = make_budget_search_backend(
        top_k=overrides.search_top_k if overrides else None,
        distance_threshold=overrides.search_distance_threshold if overrides else None,
    )
    if backend is None:
        raise _not_configured()

    runtime = get_runtime_retrieval_config()
    search_mode = runtime.effective_search_mode()
    rerank = runtime.effective_rerank()

    async def base_estimator() -> TaskHoursResult:
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
            synthesis_enabled=runtime.effective_synthesis(),
            synthesis_threshold=settings.synthesis_contradiction_threshold,
        )

    model, effort, max_iterations = _agent_knobs(overrides)
    try:
        return await agent_estimate_task_hours(
            payload.modules,
            client=client,
            backend=backend,
            consensus_fn=distance_weighted_consensus,
            base_estimator=base_estimator,
            model=model,
            reasoning_effort=effort,
            max_iterations=max_iterations,
            low_reliability=settings.agent_recovery_reliability_threshold,
        )
    except Exception as exc:
        log.error(
            "agent.hours.failed",
            error_type=type(exc).__name__,
            error=str(exc)[:300],
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="No se pudieron estimar las horas con el agente.",
        ) from exc
