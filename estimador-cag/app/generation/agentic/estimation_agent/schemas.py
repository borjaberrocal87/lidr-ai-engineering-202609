"""Contratos tipados del agente de estimación (Sesión 12).

Dos familias de modelos viven aquí:

* **Argumentos de tools** — el runner valida cada ``function_call.arguments``
  contra uno de estos modelos ANTES de ejecutar la tool. Un argumento inventado
  se convierte en una observación de error que el modelo puede corregir en la
  siguiente vuelta, nunca en una excepción que mate el bucle.
* **Resultado** — un informe ligero y propio del agente, distinto del
  ``GroundedEstimate`` de S9-S11 (que exige citas verificadas por línea).

La **traza** (razonamiento → acción → observación) ya no se define aquí: es un
contrato compartido y vive en ``app/domain/schemas/agent_trace.py`` para que los
schemas RAG puedan transportarla sin que ``rag`` importe ``agentic``. Se
re-exporta desde este módulo (con ``AgentStepRecord`` como alias histórico) para
no romper los imports existentes.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from app.domain.schemas.agent_trace import AgentStep, AgentTrace, StoppedReason

# Nombre histórico del paso; hoy es un alias del contrato compartido.
AgentStepRecord = AgentStep

Confidence = Literal["low", "medium", "high"]

__all__ = [
    "AgentStep",
    "AgentStepRecord",
    "AgentTrace",
    "StoppedReason",
    "Confidence",
    "SearchFilters",
    "SearchBudgetsArgs",
    "ComponentCostInput",
    "CalculateEstimateArgs",
    "ValidationLine",
    "ValidateEstimateArgs",
    "AgentComponent",
    "AgentEstimateReport",
    "AgentRunResult",
    "AgentTask",
    "AgentModule",
    "AgentStructure",
    "AgentStructureRun",
    "AgentTaskRef",
    "AgentNeighbor",
    "DeriveTaskHoursArgs",
    "AgentTaskDerivation",
    "AgentTaskHoursRun",
]


# --------------------------------------------------------------------------- #
# Argumentos de las tools                                                      #
# --------------------------------------------------------------------------- #
class SearchFilters(BaseModel):
    """Filtros estructurales opcionales de una búsqueda de presupuestos."""

    sectors: list[str] | None = Field(
        default=None,
        description="Restrict the search to these client sectors (e.g. ['logistics']).",
    )
    component_type: str | None = Field(
        default=None,
        description="Free-text hint about the kind of component (e.g. 'mobile app').",
    )


class SearchBudgetsArgs(BaseModel):
    """Argumentos validados de ``search_budgets``."""

    query: str = Field(min_length=1, description="One component or requirement to cost.")
    filters: SearchFilters | None = Field(
        default=None,
        description="Optional structural filters; None searches across every sector.",
    )


class ComponentCostInput(BaseModel):
    """Un componente que el agente quiere costear, con sus referencias históricas."""

    name: str = Field(min_length=1)
    reference_amounts: list[float] = Field(
        default_factory=list,
        description="Historical engineer-hours returned by search_budgets.",
    )


class CalculateEstimateArgs(BaseModel):
    """Argumentos validados de ``calculate_estimate``."""

    components: list[ComponentCostInput]


class ValidationLine(BaseModel):
    """Una línea del informe que se pasa a los guardrails."""

    name: str = Field(min_length=1)
    estimated_hours: float
    reference_amounts: list[float] = Field(default_factory=list)


class ValidateEstimateArgs(BaseModel):
    """Argumentos validados de ``validate_estimate``."""

    components: list[ValidationLine]
    total_hours: float


# --------------------------------------------------------------------------- #
# Resultado                                                                    #
# --------------------------------------------------------------------------- #
class AgentComponent(BaseModel):
    """Un componente costeado en el informe final."""

    name: str
    estimated_hours: float = Field(ge=0)
    source_chunk_ids: list[int] = Field(
        default_factory=list,
        description="Database ids of the historical chunks that grounded this component.",
    )
    rationale: str = Field(description="Why this number, in one or two sentences.")


class AgentEstimateReport(BaseModel):
    """Informe final estructurado del agente (ligero, sin citas obligatorias)."""

    components: list[AgentComponent]
    total_hours: float = Field(ge=0)
    assumptions: list[str] = Field(default_factory=list)
    confidence: Confidence


class AgentRunResult(BaseModel):
    """Todo lo que produce una ejecución: informe (o None) más su traza."""

    report: AgentEstimateReport | None = Field(
        default=None,
        description="None when the loop stopped before producing a parseable report.",
    )
    trace: AgentTrace
    iterations: int = Field(ge=0, description="Responses API round-trips.")
    stopped_reason: StoppedReason = "completed"


# --------------------------------------------------------------------------- #
# Fase 1 — el agente propone la estructura módulo→tarea                       #
# --------------------------------------------------------------------------- #
class AgentTask(BaseModel):
    """Una tarea de la estructura propuesta (sin horas: las pone la fase 2)."""

    name: str = Field(min_length=1)
    description: str | None = None


class AgentModule(BaseModel):
    """Un módulo funcional de la estructura propuesta."""

    name: str = Field(min_length=1)
    description: str | None = None
    tasks: list[AgentTask] = Field(default_factory=list)


class AgentStructure(BaseModel):
    """Estructura módulo→tarea que el agente propone en la fase 1.

    Es un contrato ligero y SIN dependencias de RAG (sin citas ni horas): el
    conductor lo mapea al ``GroundedEstimate`` que el wizard ya renderiza.
    """

    modules: list[AgentModule] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    confidence: Confidence
    reasoning: str = ""
    insufficient_context_explanation: str | None = None


class AgentStructureRun(BaseModel):
    """Lo que produce la fase 1: la estructura propuesta más su traza."""

    structure: AgentStructure
    trace: AgentTrace
    iterations: int = Field(ge=0, description="Responses API round-trips.")
    stopped_reason: StoppedReason = "completed"


# --------------------------------------------------------------------------- #
# Fase 2 — recuperación de horas solo para las tareas no groundadas           #
# --------------------------------------------------------------------------- #
class AgentTaskRef(BaseModel):
    """Una tarea que el paso determinista no pudo groundar."""

    module: str = Field(min_length=1)
    task: str = Field(min_length=1)
    description: str | None = None


class AgentNeighbor(BaseModel):
    """Un análogo histórico que el agente encontró y pasa al consenso."""

    source_id: int | None = Field(
        default=None,
        description="Database id of the historical chunk, when known.",
    )
    estimated_hours: int = Field(ge=0)
    distance: float = Field(ge=0.0)


class DeriveTaskHoursArgs(BaseModel):
    """Argumentos validados de ``derive_task_hours``."""

    module: str = Field(min_length=1)
    task: str = Field(min_length=1)
    neighbors: list[AgentNeighbor] = Field(min_length=1)


class AgentTaskDerivation(BaseModel):
    """Horas que la fase 2 derivó para una tarea, con su fiabilidad."""

    module: str
    task: str
    estimated_hours: int = Field(ge=0)
    reliability: float = Field(ge=0.0, le=1.0)
    dispersion: float = Field(ge=0.0)
    source_chunk_ids: list[int] = Field(default_factory=list)
    rationale: str | None = None


class AgentTaskHoursRun(BaseModel):
    """Resultado de la fase 2: horas recuperadas más la traza del bucle.

    No hay parse final: las horas salen de las observaciones de
    ``derive_task_hours``, que aplica la MISMA matemática de consenso que el
    camino determinista. Así el agente no puede inventar una cifra.
    """

    recovered: list[AgentTaskDerivation] = Field(default_factory=list)
    trace: AgentTrace
    iterations: int = Field(ge=0, description="Responses API round-trips.")
    stopped_reason: StoppedReason = "completed"
