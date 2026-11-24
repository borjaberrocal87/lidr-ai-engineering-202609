"""Capa HTTP del agente de estimación (Sesión 12).

Superficie única:

``POST /agent/estimate/from-transcript``

El backend de negocio no ve el bucle: envía una transcripción y recibe el informe
estructurado, igual que en la Sesión 9. La diferencia es que aquí viaja también la
traza (razonamiento + acción + observación por paso), ya renderizada en
``trace_text`` para que la UI la pinte y la descargue sin duplicar el formato.

Sin persistencia ni idempotencia: cada petición es un bucle fresco. La ejecución
puede tardar varias llamadas al modelo; el endpoint es síncrono y comparte el
límite de peticiones del resto del pipeline RAG. Los fallos de configuración
(cliente o backend a ``None``) se traducen a 500 genérico; los fallos del bucle se
traducen a 502, con el detalle solo en los logs.
"""

from __future__ import annotations

from typing import Annotated

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request, status
from openai import AsyncOpenAI
from pydantic import BaseModel, ConfigDict, Field

from app.api.rate_limiting import limiter
from app.config import settings
from app.dependencies import get_async_openai_client, get_budget_search_backend
from app.generation.agentic.estimation_agent.runner import run_estimation_agent
from app.generation.agentic.estimation_agent.schemas import AgentRunResult
from app.generation.agentic.estimation_agent.tools import BudgetSearchBackend

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
