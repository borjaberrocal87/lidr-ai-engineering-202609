"""Auditoría semántica de la estimación grounded (Sesión 11, directo).

``grounding.verify_citations`` prueba integridad **referencial**: cada
``chunk_id`` citado existía en el contexto. Es necesario pero no suficiente — una
línea puede citar una fuente real y aun así inventarse la cifra. Este módulo
añade la capa **semántica**, en dos piezas que se combinan por línea:

1. :func:`evidence_ceiling_days` — **ancla determinista** (sin LLM, sin coste,
   sin varianza): suma las horas históricas que registran los chunks citados y
   las convierte a días-ingeniero. Es el techo que la evidencia soporta.
2. :class:`HallucinationGate` — **juez estricto** (una sola llamada batched):
   ¿la evidencia citada implica de verdad la cifra y el alcance declarados? El
   juez se inyecta (``StructuredLLM``) para no acoplar el módulo al composition
   root y poder probarlo sin red; si falla, el gate sigue con el ancla sola.

La regla es asimétrica a propósito: quedarse **por debajo** del techo es legítimo
(la línea es una rebanada de un componente histórico mayor); pasarse **por
encima** de la tolerancia, o no poder atar la cifra a la evidencia, es una
alucinación vestida de cita y se gradúa ``degraded``.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Mapping
from typing import Any, Protocol, TypeVar

import structlog
from pydantic import BaseModel, Field

from app.foundation.prompts.loader import render_rag_quality_judge_prompt
from app.generation.rag.schemas import (
    EstimationTask,
    GroundedEstimate,
    HallucinationReport,
    LineQualityGate,
    LineVerdict,
    RetrievedChunk,
)

log = structlog.get_logger()

# El corpus registra horas; la estimación habla en días-ingeniero.
_HOURS_PER_DAY = 8
# Dos formatos conviven en el corpus: "48 h / 48 hours" (cifra antes) y
# "Estimated hours: 48" (etiqueta antes). Se prueban en ese orden.
_HOURS_AFTER_NUMBER = re.compile(r"(\d+(?:[.,]\d+)?)\s*(?:h|hrs|horas|hours)\b", re.IGNORECASE)
_HOURS_AFTER_LABEL = re.compile(r"(?:horas|hours)\s*[:=]?\s*(\d+(?:[.,]\d+)?)", re.IGNORECASE)

StructuredModelT = TypeVar("StructuredModelT", bound=BaseModel)


class StructuredLLM(Protocol):
    """Contrato mínimo del wrapper para una llamada estructurada (inyectable)."""

    def complete_structured(
        self,
        *,
        system_prompt: str,
        user_message: str,
        response_model: type[StructuredModelT],
        temperature: float | None,
        max_tokens: int,
        max_retries: int = ...,
        model_override: str | None = ...,
    ) -> tuple[StructuredModelT, dict[str, Any]]: ...


class _VerdictPanel(BaseModel):
    """Salida batched del juez: un dictamen por línea auditada."""

    verdicts: list[LineVerdict] = Field(default_factory=list)


def recorded_hours(chunk: RetrievedChunk) -> float | None:
    """Horas históricas del chunk: metadata primero, luego parseo del texto."""
    if chunk.estimated_hours is not None:
        return float(chunk.estimated_hours)
    text = chunk.content or ""
    matches = _HOURS_AFTER_NUMBER.findall(text) or _HOURS_AFTER_LABEL.findall(text)
    if not matches:
        return None
    return float(matches[-1].replace(",", "."))


def evidence_ceiling_days(
    task: EstimationTask,
    chunks_by_id: Mapping[str, RetrievedChunk],
) -> float | None:
    """Techo determinista de una línea, en días-ingeniero.

    Suma las horas históricas de los chunks que la línea cita. ``None`` cuando
    ninguno de ellos registra horas: sin techo, el gate decide solo por el juez.
    """
    hours = [
        value
        for citation in task.citations
        if (chunk := chunks_by_id.get(citation.chunk_id)) is not None
        and (value := recorded_hours(chunk)) is not None
    ]
    if not hours:
        return None
    return sum(hours) / _HOURS_PER_DAY


def combine_line_gate(
    task: EstimationTask,
    module: str,
    ceiling_days: float | None,
    verdict: LineVerdict | None,
    *,
    tolerance: float,
) -> LineQualityGate:
    """Combina ancla y dictamen en el estado de UNA línea (función pura).

    ``verdict`` es ``None`` en modo solo-ancla. Una línea no grounded es
    ``insufficient``; un exceso sobre el techo por encima de la tolerancia o un
    veredicto ``entailed=False`` la degradan.
    """
    if not task.grounded:
        return LineQualityGate(module=module, component=task.name, status="insufficient")

    deviation: float | None = None
    over_ceiling = False
    if ceiling_days and ceiling_days > 0 and task.engineer_days is not None:
        delta = task.engineer_days - ceiling_days
        deviation = abs(delta) / ceiling_days
        # Solo pasarse del techo es señal de alucinación: una línea puede ser
        # legítimamente una fracción de su componente padre.
        over_ceiling = (delta / ceiling_days) > tolerance

    judge_failed = verdict is not None and not verdict.entailed

    if over_ceiling or judge_failed:
        reasons: list[str] = []
        if over_ceiling and ceiling_days is not None:
            reasons.append(
                f"declara {task.engineer_days}d pero la evidencia citada soporta "
                f"~{ceiling_days:.0f}d"
            )
        if judge_failed and verdict is not None and verdict.reason:
            reasons.append(verdict.reason)
        return LineQualityGate(
            module=module,
            component=task.name,
            status="degraded",
            evidence_days=ceiling_days,
            deviation=deviation,
            reason="; ".join(reasons) or "la evidencia citada no sostiene la cifra",
        )

    return LineQualityGate(
        module=module,
        component=task.name,
        status="grounded",
        evidence_days=ceiling_days,
        deviation=deviation,
    )


class HallucinationGate:
    """Audita una estimación completa: ancla numérica por línea + juez batched."""

    def __init__(
        self,
        *,
        judge: StructuredLLM | None,
        model: str,
        tolerance: float,
        max_tokens: int = 4096,
        temperature: float | None = 0.0,
    ) -> None:
        self._judge = judge
        self._model = model
        self._tolerance = tolerance
        self._max_tokens = max_tokens
        self._temperature = temperature

    async def _ask_judge(
        self,
        estimate: GroundedEstimate,
    ) -> dict[tuple[str, str], LineVerdict]:
        """Una única llamada batched sobre todas las líneas con evidencia."""
        auditable = [
            (module.name, task)
            for module in estimate.modules
            for task in module.tasks
            if task.grounded and task.citations
        ]
        if not auditable or self._judge is None:
            return {}

        lines: list[dict[str, object]] = [
            {
                "module": module_name,
                "component": task.name,
                "claimed_days": task.engineer_days,
                "evidence": [citation.evidence for citation in task.citations],
            }
            for module_name, task in auditable
        ]
        system_prompt, user_message = render_rag_quality_judge_prompt(lines=lines)

        try:
            panel, _meta = await asyncio.to_thread(
                self._judge.complete_structured,
                system_prompt=system_prompt,
                user_message=user_message,
                response_model=_VerdictPanel,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
                max_retries=0,
                model_override=self._model,
            )
        except Exception as exc:
            log.warning("hallucination_judge_failed", error=str(exc)[:200])
            return {}

        return {(verdict.module, verdict.component): verdict for verdict in panel.verdicts}

    async def audit(
        self,
        estimate: GroundedEstimate,
        chunks: list[RetrievedChunk],
        *,
        use_judge: bool = True,
    ) -> HallucinationReport:
        """Gradúa cada línea: ``grounded`` / ``degraded`` / ``insufficient``.

        ``use_judge=False`` deja el camino solo-ancla (sin LLM, sin coste), útil
        para medir el juez en aislamiento.
        """
        chunks_by_id = {str(chunk.id): chunk for chunk in chunks}
        verdicts = await self._ask_judge(estimate) if use_judge else {}

        lines: list[LineQualityGate] = []
        grounded = degraded = insufficient = 0
        for module in estimate.modules:
            for task in module.tasks:
                gate = combine_line_gate(
                    task,
                    module.name,
                    evidence_ceiling_days(task, chunks_by_id),
                    verdicts.get((module.name, task.name)),
                    tolerance=self._tolerance,
                )
                lines.append(gate)
                if gate.status == "grounded":
                    grounded += 1
                elif gate.status == "degraded":
                    degraded += 1
                else:
                    insufficient += 1

        return HallucinationReport(
            total_lines=len(lines),
            grounded_lines=grounded,
            degraded_lines=degraded,
            insufficient_lines=insufficient,
            lines=lines,
        )
