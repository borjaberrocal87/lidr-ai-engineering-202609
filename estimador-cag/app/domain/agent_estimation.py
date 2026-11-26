"""Conductor del agente que conduce el wizard de estimación (Sesión 12).

Es el ÚNICO sitio donde los hermanos ``generation.agentic`` y ``generation.rag``
se encuentran, así que vive en ``domain``: los conductores de dominio pueden
componer infraestructura resuelta por el composition root (ver
``tests/test_layering.py``). El cliente async de OpenAI y el backend de búsqueda
los INYECTA el router; este módulo no resuelve infraestructura.

Dos fases, espejo del wizard y alrededor de la revisión humana:

* :func:`agent_propose_structure` — fase 1. El agente descompone el brief en
  módulos→tareas y lo mapeamos al ``GroundedEstimate`` que el wizard ya renderiza
  (sin horas, sin citas: eso llega después).
* :func:`agent_estimate_task_hours` — fase 2, HÍBRIDA. Primero corre el consenso
  determinista por tarea; solo las tareas que quedaron sin análogo — o con
  fiabilidad demasiado baja — pasan al bucle de recuperación del agente, y lo
  recuperado se fusiona en el mismo ``TaskHoursResult``.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

import structlog

from app.domain.schemas.agent_trace import AgentTrace
from app.generation.agentic.estimation_agent.recovery_agent import (
    run_task_hours_recovery_agent,
)
from app.generation.agentic.estimation_agent.schemas import AgentTaskRef
from app.generation.agentic.estimation_agent.structure_agent import run_structure_agent
from app.generation.agentic.estimation_agent.tools import (
    BudgetSearchBackend,
    ConsensusFn,
)
from app.generation.rag.schemas import (
    Assumption,
    EstimationModule,
    EstimationTask,
    GroundedEstimate,
    ProjectBrief,
    SourceReference,
    TaskHoursEstimate,
    TaskHoursModuleInput,
    TaskHoursResult,
)

log = structlog.get_logger(__name__)

# Una tarea groundada por debajo de esta fiabilidad merece una re-búsqueda: el
# paso determinista casó con ALGO, pero tan flojo que el agente puede mejorarlo.
DEFAULT_LOW_RELIABILITY = 0.35

BaseEstimator = Callable[[], Awaitable[TaskHoursResult]]


@dataclass(frozen=True)
class StructureProposal:
    """Resultado de la fase 1: estimación para el wizard + traza del agente."""

    estimate: GroundedEstimate
    trace: AgentTrace


def compose_brief_text(brief: ProjectBrief) -> str:
    """Convierte el brief destilado en el mensaje de usuario del agente.

    Se compone aquí (y no en ``agentic``) porque ``ProjectBrief`` es un contrato
    de ``generation.rag`` y el hermano agentic no puede importarlo.
    """
    lines = [f"Functionality: {brief.functionality}"]
    if brief.technologies:
        lines.append(f"Technologies: {', '.join(brief.technologies)}")
    if brief.sector:
        lines.append(f"Sector: {brief.sector}")
    if brief.scale and brief.scale != "unknown":
        lines.append(f"Scale: {brief.scale}")
    if brief.country:
        lines.append(f"Country: {brief.country}")
    if brief.regulations:
        lines.append(f"Regulations: {', '.join(brief.regulations)}")
    if brief.constraints:
        lines.append(f"Constraints: {', '.join(brief.constraints)}")
    lines.append("")
    lines.append("Decompose this brief into the module→task structure now.")
    return "\n".join(lines)


def _structure_to_estimate(structure: Any) -> GroundedEstimate:
    """Mapea la estructura rag-free del agente al ``GroundedEstimate`` del wizard.

    Solo estructura: cada tarea queda sin horas y sin fuentes (``grounded=False``
    y ``engineer_days=None``), que es justo el contrato de la generación
    estructura-primero de la Sesión 10. El paso de horas la rellena después.
    """
    modules = [
        EstimationModule(
            name=module.name,
            description=module.description,
            tasks=[
                EstimationTask(
                    name=task.name,
                    description=task.description,
                    engineer_days=None,
                    grounded=False,
                    citations=[],
                    sources=[],
                )
                for task in module.tasks
            ],
        )
        for module in structure.modules
    ]
    assumptions = [
        Assumption(
            description=text,
            impact="medium",
            rationale="Inferido por el agente al descomponer el brief.",
        )
        for text in structure.assumptions
    ]
    return GroundedEstimate(
        total_engineer_days=None,
        modules=modules,
        duration_weeks=None,
        sources=[],
        assumptions=assumptions,
        confidence=structure.confidence,
        reasoning=structure.reasoning or "Estructura propuesta por el agente.",
        insufficient_context_explanation=structure.insufficient_context_explanation,
    )


async def agent_propose_structure(
    brief: ProjectBrief,
    *,
    client: Any,
    model: str,
    reasoning_effort: str,
) -> StructureProposal:
    """Fase 1: el agente propone la estructura del brief."""
    run = await run_structure_agent(
        compose_brief_text(brief),
        client=client,
        model=model,
        reasoning_effort=reasoning_effort,
    )
    estimate = _structure_to_estimate(run.structure)
    log.info(
        "agent.estimate.structure",
        modules=len(estimate.modules),
        tasks=sum(len(module.tasks) for module in estimate.modules),
    )
    return StructureProposal(estimate=estimate, trace=run.trace)


def _task_descriptions(modules: list[TaskHoursModuleInput]) -> dict[tuple[str, str], str | None]:
    """Índice (módulo, tarea) → descripción para etiquetar las tareas marcadas."""
    lookup: dict[tuple[str, str], str | None] = {}
    for module in modules:
        for task in module.tasks:
            lookup[(module.name, task.name)] = task.description
    return lookup


def _is_flagged(estimate: TaskHoursEstimate, low_reliability: float) -> bool:
    """Una tarea entra al agente si no tiene análogo o su fiabilidad es baja."""
    if not estimate.has_match or estimate.estimated_hours is None:
        return True
    if estimate.reliability is None:
        return False
    return estimate.reliability < low_reliability


def _recovered_estimate(module: str, task: str, derivation: Any) -> TaskHoursEstimate:
    """Convierte una derivación del agente en la línea del wizard.

    ``grounded`` se omite a propósito: el validador lo deriva de ``has_match``,
    que es la fuente de verdad y tolera la recuperación sin ids de chunk.
    """
    sources = [
        SourceReference(
            chunk_id=str(chunk_id),
            document_id="agent-recovery",
            evidence=f"Recuperado por el agente: {derivation.rationale or 'análogo histórico'}",
        )
        for chunk_id in derivation.source_chunk_ids
    ]
    return TaskHoursEstimate(
        module=module,
        task=task,
        estimated_hours=derivation.estimated_hours,
        reliability=derivation.reliability,
        has_match=True,
        dispersion=derivation.dispersion,
        neighbors=[],
        sources=sources,
    )


async def agent_estimate_task_hours(
    modules: list[TaskHoursModuleInput],
    *,
    client: Any,
    backend: BudgetSearchBackend,
    consensus_fn: ConsensusFn,
    base_estimator: BaseEstimator,
    model: str,
    reasoning_effort: str,
    max_iterations: int,
    low_reliability: float = DEFAULT_LOW_RELIABILITY,
) -> TaskHoursResult:
    """Fase 2 híbrida: consenso determinista + recuperación de lo no groundado.

    ``base_estimator`` corre el pipeline determinista con la infraestructura que
    resuelve el router (store, embedder, reranker...). Si ninguna tarea queda
    marcada, se devuelve tal cual y el agente ni se invoca.
    """
    base = await base_estimator()
    descriptions = _task_descriptions(modules)
    flagged = [
        AgentTaskRef(
            module=estimate.module,
            task=estimate.task,
            description=descriptions.get((estimate.module, estimate.task)),
        )
        for estimate in base.tasks
        if _is_flagged(estimate, low_reliability)
    ]

    if not flagged:
        log.info("agent.estimate.hours", flagged=0, recovered=0)
        return base

    recovery = await run_task_hours_recovery_agent(
        flagged,
        client=client,
        model=model,
        backend=backend,
        consensus_fn=consensus_fn,
        reasoning_effort=reasoning_effort,
        max_iterations=max_iterations,
    )

    recovered_by_key = {
        (derivation.module, derivation.task): derivation for derivation in recovery.recovered
    }
    merged: list[TaskHoursEstimate] = []
    for estimate in base.tasks:
        derivation = recovered_by_key.get((estimate.module, estimate.task))
        if derivation is not None:
            merged.append(_recovered_estimate(estimate.module, estimate.task, derivation))
        else:
            merged.append(estimate)

    log.info(
        "agent.estimate.hours",
        flagged=len(flagged),
        recovered=len(recovered_by_key),
        iterations=recovery.iterations,
    )
    return TaskHoursResult(tasks=merged, agent_trace=recovery.trace)
