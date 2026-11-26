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
