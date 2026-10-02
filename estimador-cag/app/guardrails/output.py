"""Guardrails de salida: refuerzo sobre los validadores de Pydantic.

Los validadores del esquema (`app.schemas.estimations`) son la primera línea:
cuando fallan, el wrapper re-promptea al modelo. Si el modelo se empeña,
`complete_structured` acaba lanzando y la petición falla.

`enforce_scope_response` es un **filtro** (no una excepción): reescribe el
summary cuando el modelo produjo una respuesta de baja confianza sin el prefijo
`Out of scope:`. En la práctica el validador habrá saltado antes, así que esto
cubre bordes (p. ej. `confidence_pct == 30` exacto o futuros ajustes del umbral).
"""

from __future__ import annotations

import structlog

from app.schemas.estimations import (
    LOW_CONFIDENCE_THRESHOLD,
    OUT_OF_SCOPE_PREFIX,
    EstimationResult,
    Phase,
)

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)

_NOT_ESTIMATED_PHASE = Phase(
    name="Not estimated",
    duration_weeks=1,
    cost_eur=0,
    summary="No se puede dimensionar sin más información de alcance e integraciones.",
)


def enforce_scope_response(result: EstimationResult) -> EstimationResult:
    """Reescribe el resultado si hay baja confianza y el summary no lo declara.

    Política *filter*: nunca lanza, siempre devuelve un `EstimationResult` bien
    formado. El usuario recibe un mensaje claro en lugar de un error.
    """
    is_low_confidence = result.confidence_pct < LOW_CONFIDENCE_THRESHOLD
    already_marked = result.summary.startswith(OUT_OF_SCOPE_PREFIX)

    if not is_low_confidence or already_marked:
        return result

    logger.info(
        "enforce_scope_response_filtering",
        confidence_pct=result.confidence_pct,
        original_summary_chars=len(result.summary),
    )
    new_summary = (
        f"{OUT_OF_SCOPE_PREFIX} no hay información suficiente para estimar con confianza. "
        f"Razonamiento original del modelo: {result.summary[:400]}"
    )
    return EstimationResult(
        summary=new_summary[:1200],
        confidence_pct=result.confidence_pct,
        phases=[_NOT_ESTIMATED_PHASE],
        total_duration_weeks=1,
        total_cost_eur=0,
    )
