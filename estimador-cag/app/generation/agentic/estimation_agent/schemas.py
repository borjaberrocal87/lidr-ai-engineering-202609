"""Contratos tipados del agente de estimación (Sesión 12).

Tres familias de modelos:

* **Argumentos de tools** — el runner valida cada ``function_call.arguments``
  contra uno de estos modelos ANTES de ejecutar la tool. Un argumento inventado
  se convierte en una observación de error que el modelo puede corregir en la
  siguiente vuelta, nunca en una excepción que mate el bucle.
* **Traza** — el registro razonamiento → acción → observación que exige el
  ejercicio. ``AgentTrace.render`` produce el formato ``STEP N`` de consola.
* **Resultado** — un informe ligero y propio del agente, distinto del
  ``GroundedEstimate`` de S9-S11 (que exige citas verificadas por línea).
"""

from __future__ import annotations

import json
import re
from typing import Any, Literal

from pydantic import BaseModel, Field

Confidence = Literal["low", "medium", "high"]
StoppedReason = Literal["completed", "max_iterations", "no_final_report"]

# Un modelo puede colar un carácter de control dentro de un escape unicode
# malformado. Se limpia solo al renderizar la traza para que el .txt quede
# legible sin tocar los argumentos que usa el dispatch.
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


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
# Traza                                                                        #
# --------------------------------------------------------------------------- #
class AgentStepRecord(BaseModel):
    """Un paso razonamiento → acción → observación del bucle."""

    step: int = Field(ge=1)
    reasoning_summary: str | None = Field(
        default=None,
        description="Reasoning summary emitted by the Responses API for this turn.",
    )
    tool: str = Field(description="Invoked tool name.")
    tool_args: dict[str, Any] = Field(description="Arguments the model passed.")
    observation: str = Field(description="Human-readable summary of the tool result.")


class AgentTrace(BaseModel):
    """Registro ordenado de la ejecución, para auditar y para el entregable."""

    steps: list[AgentStepRecord] = Field(default_factory=list)
    model: str | None = None
    iterations: int | None = None
    stopped_reason: StoppedReason | None = None

    def render(self) -> str:
        """Render the trace in the ``STEP N`` console format required by the exercise."""
        header = ["=" * 78, "AGENT TRACE", "=" * 78]
        meta = []
        if self.model:
            meta.append(f"model={self.model}")
        if self.iterations is not None:
            meta.append(f"iterations={self.iterations}")
        if self.stopped_reason:
            meta.append(f"stopped={self.stopped_reason}")
        if meta:
            header.append("  " + "   ".join(meta))
            header.append("")

        if not self.steps:
            return "\n".join([*header, "(no tool steps — the agent answered without tools)"])

        blocks: list[str] = []
        for step in self.steps:
            reasoning = step.reasoning_summary or "(no reasoning summary emitted)"
            args = _CONTROL_CHARS.sub(
                "",
                json.dumps(step.tool_args, ensure_ascii=False, default=str),
            )
            blocks.append(
                f"STEP {step.step}\n"
                f"  reasoning:   {reasoning}\n"
                f"  action:      {step.tool}({args})\n"
                f"  observation: {step.observation}"
            )
        return "\n".join([*header, "\n\n".join(blocks)])


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
