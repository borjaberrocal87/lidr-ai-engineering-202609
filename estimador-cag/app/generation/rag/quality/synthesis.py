"""Síntesis de horas en dos etapas (Sesión 11).

El consenso de :mod:`app.generation.rag.task_hours` promedia los análogos más
cercanos en un único número. Eso es correcto cuando los vecinos están de acuerdo
y engañoso cuando no: si un análogo registra 40h y otro 90h, el punto medio (65h)
parece una estimación segura mientras esconde un desacuerdo de más del doble.

La síntesis hace el desacuerdo de primera clase:

1. **Ancla determinista** — reutiliza la dispersión (coeficiente de variación)
   que ya calcula el consenso; no reimplementa el promedio.
2. **Juicio** — solo cuando la dispersión cruza el umbral de contradicción se
   emite un :class:`HourRange` (low..high) con una razón. La razón es
   determinista por defecto (min/max de los vecinos + plantilla); un *rephraser*
   inyectable puede redactarla mejor, y si falla se degrada a la determinista.

``estimated_hours`` no se toca: la persona sigue editando un número; el rango
viaja al lado como señal de incertidumbre honesta.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

import structlog

from app.generation.rag.schemas import HourRange, TaskNeighbor

log = structlog.get_logger()

# El *rephraser* recibe los vecinos y el rango determinista y devuelve una frase.
# Se inyecta (normalmente un wrapper LLM) para no acoplar este módulo al
# composition root y poder probarlo sin red.
Rephraser = Callable[[list[TaskNeighbor], HourRange], Awaitable[str]]


def is_contradiction(dispersion: float | None, threshold: float) -> bool:
    """True cuando la dispersión de los vecinos cruza el umbral de contradicción."""
    return dispersion is not None and dispersion > threshold


def deterministic_reason(neighbors: list[TaskNeighbor]) -> str:
    """Razón sin LLM: nombra las horas extremas y cuántos análogos discrepan."""
    hours = sorted(neighbor.estimated_hours for neighbor in neighbors)
    return (
        f"los análogos históricos discrepan ({hours[0]}h vs {hours[-1]}h en "
        f"{len(hours)} referencias); el alcance aún no está cerrado y el esfuerzo "
        "depende de esa decisión"
    )


async def synthesize_hour_range(
    neighbors: list[TaskNeighbor],
    dispersion: float | None,
    *,
    threshold: float,
    rephrase: Rephraser | None = None,
) -> HourRange | None:
    """Devuelve un rango si los vecinos se contradicen; si no, ``None``.

    ``rephrase`` es un callable ``async (neighbors, base) -> str`` opcional. Si
    no se pasa, o si falla, la razón es la determinista: el rango se produce
    siempre offline.
    """
    if not neighbors or not is_contradiction(dispersion, threshold):
        return None

    hours = sorted(neighbor.estimated_hours for neighbor in neighbors)
    base = HourRange(
        low_hours=hours[0],
        high_hours=hours[-1],
        reason=deterministic_reason(neighbors),
    )
    if rephrase is None:
        return base

    try:
        reason = await rephrase(neighbors, base)
    except Exception as exc:
        log.warning("synthesis_reason_failed", error=str(exc)[:200])
        return base

    cleaned = str(reason).strip()
    return base.model_copy(update={"reason": cleaned}) if cleaned else base
