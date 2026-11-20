"""Paquete de la capa agéntica de estimación (Sesión 12).

El agente vive junto a ``boss`` y ``critic`` y se reparte en tres módulos:

* :mod:`schemas` — contratos tipados (argumentos de tools, traza, informe).
* :mod:`tools` — schemas planos de la Responses API e implementaciones puras.
* :mod:`runner` — el bucle manual razona → actúa → observa → repite.

Nada de este paquete conoce el transporte HTTP ni el retrieval: el cliente de la
Responses API y el backend de búsqueda se inyectan desde el composition root.
"""

from app.generation.agentic.estimation_agent.runner import run_estimation_agent
from app.generation.agentic.estimation_agent.schemas import (
    AgentComponent,
    AgentEstimateReport,
    AgentRunResult,
    AgentStepRecord,
    AgentTrace,
    CalculateEstimateArgs,
    ComponentCostInput,
    SearchBudgetsArgs,
    SearchFilters,
    ValidateEstimateArgs,
    ValidationLine,
)
from app.generation.agentic.estimation_agent.tools import (
    TOOL_SCHEMAS,
    BudgetSearchBackend,
    calculate_estimate,
    dispatch_tool,
    search_budgets,
    validate_estimate,
)

__all__ = [
    "TOOL_SCHEMAS",
    "AgentComponent",
    "AgentEstimateReport",
    "AgentRunResult",
    "AgentStepRecord",
    "AgentTrace",
    "BudgetSearchBackend",
    "CalculateEstimateArgs",
    "ComponentCostInput",
    "SearchBudgetsArgs",
    "SearchFilters",
    "ValidateEstimateArgs",
    "ValidationLine",
    "calculate_estimate",
    "dispatch_tool",
    "run_estimation_agent",
    "search_budgets",
    "validate_estimate",
]
