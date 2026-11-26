"""Tools del agente de estimación: schemas planos y funciones deterministas.

Tres tools:

* ``search_budgets`` — envuelve el pipeline de recuperación de S9-S10 a través de
  un backend **inyectado**. Esta capa no sabe qué hay detrás: el backend real lo
  construye ``app/dependencies.py`` y el stub offline vive en
  ``examples/session-12/offline_budget_stub.py``.
* ``calculate_estimate`` — aritmética pura (mediana + contingencia). No llama al
  LLM.
* ``validate_estimate`` — guardrails de S4 sobre el informe final (componentes sin
  referencia, horas fuera de rango, totales incoherentes).

Los schemas usan la forma **plana** de la Responses API
(``{"type": "function", "name": ..., "parameters": {...}}``), no la de Chat
Completions, y ``strict: true``: todas las propiedades aparecen en ``required``
(la opcionalidad se expresa con uniones ``["...", "null"]``) y
``additionalProperties: false`` en todos los niveles.

Las descripciones son lo ÚNICO que el modelo lee para decidir cuándo usar cada
tool. Están escritas para un modelo que no ve este código.
"""

from __future__ import annotations

import statistics
from collections.abc import Callable
from typing import Any, Protocol

import structlog

from app.generation.agentic.estimation_agent.schemas import (
    CalculateEstimateArgs,
    DeriveTaskHoursArgs,
    SearchBudgetsArgs,
    ValidateEstimateArgs,
)

log = structlog.get_logger(__name__)

# Buffer de contingencia plano por componente. Un número fijo y visible mantiene
# el modelo de costes auditable: no hay multiplicadores escondidos.
CONTINGENCY_FACTOR = 0.15

# Un proyecto por encima de ~10 personas-año en un único encargo casi siempre es
# un error de unidades o de duplicación de componentes.
MAX_PLAUSIBLE_HOURS = 20_000.0

CONTENT_PREVIEW_CHARS = 200


# --------------------------------------------------------------------------- #
# Schemas planos para la Responses API                                         #
# --------------------------------------------------------------------------- #
SEARCH_BUDGETS_TOOL: dict[str, Any] = {
    "type": "function",
    "name": "search_budgets",
    "description": (
        "Look up historical project budgets for work analogous to ONE component or "
        "requirement and return the matching items with their recorded effort in "
        "engineer-hours. Call it once per component you need to cost (for example "
        "once for the orders backend and again for the mobile app); do NOT try to "
        "cover the whole project with a single broad query. Use a focused query "
        "that describes that one component. Each returned item carries an id, a "
        "text preview, its sector, its chunk type and its recorded engineer-hours; "
        "pass those hours as reference_amounts to calculate_estimate."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "Natural-language description of the single component to find "
                    "analogous budgets for, e.g. 'SAP ERP integration with IDoc "
                    "mapping and retry handling'."
                ),
            },
            "filters": {
                "type": ["object", "null"],
                "description": ("Optional structural filters. Pass null to search every sector."),
                "properties": {
                    "sectors": {
                        "type": ["array", "null"],
                        "items": {"type": "string"},
                        "description": (
                            "Restrict the search to these client sectors, e.g. "
                            "['logistics', 'industrial']."
                        ),
                    },
                    "component_type": {
                        "type": ["string", "null"],
                        "description": (
                            "Free-text hint about the kind of component, e.g. "
                            "'mobile app' or 'erp integration'."
                        ),
                    },
                },
                "required": ["sectors", "component_type"],
                "additionalProperties": False,
            },
        },
        "required": ["query", "filters"],
        "additionalProperties": False,
    },
    "strict": True,
}

CALCULATE_ESTIMATE_TOOL: dict[str, Any] = {
    "type": "function",
    "name": "calculate_estimate",
    "description": (
        "Deterministically compute an effort estimate in engineer-hours from a set "
        "of components and their historical reference amounts. For each component "
        "it takes the median of the reference amounts and adds a fixed contingency "
        "buffer, then sums the components into a total. Call it once you have "
        "gathered reference hours (from search_budgets) for every component. A "
        "component with an empty reference_amounts list is costed at zero hours and "
        "flagged as unbudgeted: do not invent a number, search again for it instead. "
        "This tool does NOT call a model; its output is reproducible arithmetic."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "components": {
                "type": "array",
                "description": "The components to cost.",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Component name, taken from the transcript.",
                        },
                        "reference_amounts": {
                            "type": "array",
                            "items": {"type": "number"},
                            "description": (
                                "Historical engineer-hours for analogous work, taken "
                                "from search_budgets results. May be empty when no "
                                "analog was found."
                            ),
                        },
                    },
                    "required": ["name", "reference_amounts"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["components"],
        "additionalProperties": False,
    },
    "strict": True,
}

VALIDATE_ESTIMATE_TOOL: dict[str, Any] = {
    "type": "function",
    "name": "validate_estimate",
    "description": (
        "Run sanity-check guardrails over a finished estimate before returning it. "
        "It flags components with no historical reference, components whose hours "
        "fall far outside the range implied by their references, a total that does "
        "not match the sum of its components, and non-positive or implausibly large "
        "totals. Call it as the LAST tool step, once you have a full estimate, and "
        "fix whatever it reports (for example by searching again for an unbudgeted "
        "component) before you give your final answer. Returns {ok, issues}."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "components": {
                "type": "array",
                "description": (
                    "The estimate's components, with their final hours and their reference amounts."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "estimated_hours": {"type": "number"},
                        "reference_amounts": {
                            "type": "array",
                            "items": {"type": "number"},
                        },
                    },
                    "required": ["name", "estimated_hours", "reference_amounts"],
                    "additionalProperties": False,
                },
            },
            "total_hours": {
                "type": "number",
                "description": "The estimate's grand total in engineer-hours.",
            },
        },
        "required": ["components", "total_hours"],
        "additionalProperties": False,
    },
    "strict": True,
}

TOOL_SCHEMAS: list[dict[str, Any]] = [
    SEARCH_BUDGETS_TOOL,
    CALCULATE_ESTIMATE_TOOL,
    VALIDATE_ESTIMATE_TOOL,
]


# --------------------------------------------------------------------------- #
# Fase 2 — consenso de horas con la matemática del camino determinista        #
# --------------------------------------------------------------------------- #
DERIVE_TASK_HOURS_TOOL: dict[str, Any] = {
    "type": "function",
    "name": "derive_task_hours",
    "description": (
        "Compute the hours for ONE task from the historical analogs you found with "
        "search_budgets, using the SAME distance-weighted consensus the "
        "deterministic pipeline uses. Pass every analog you consider relevant as a "
        "neighbor with its recorded engineer-hours and its cosine distance "
        "(smaller = closer). The tool returns the consensus hours, a 0..1 "
        "reliability and a dispersion; do NOT invent hours yourself. If no "
        "neighbor is close enough, search again with a better reformulation "
        "instead of calling this tool with weak matches."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "module": {
                "type": "string",
                "description": "Module the task belongs to, taken from the flagged list.",
            },
            "task": {
                "type": "string",
                "description": "Task name, exactly as given in the flagged list.",
            },
            "neighbors": {
                "type": "array",
                "description": (
                    "Historical analogs for this task, each with its recorded hours "
                    "and the cosine distance returned by search_budgets."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "source_id": {
                            "type": ["integer", "null"],
                            "description": "Chunk id of the analog, when known.",
                        },
                        "estimated_hours": {
                            "type": "integer",
                            "description": "Recorded engineer-hours of the analog.",
                        },
                        "distance": {
                            "type": "number",
                            "description": "Cosine distance to the task (smaller = closer).",
                        },
                    },
                    "required": ["source_id", "estimated_hours", "distance"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["module", "task", "neighbors"],
        "additionalProperties": False,
    },
    "strict": True,
}

RECOVERY_TOOL_SCHEMAS: list[dict[str, Any]] = [
    SEARCH_BUDGETS_TOOL,
    DERIVE_TASK_HOURS_TOOL,
    VALIDATE_ESTIMATE_TOOL,
]


# --------------------------------------------------------------------------- #
# Backend de búsqueda (inyectado)                                              #
# --------------------------------------------------------------------------- #
class BudgetSearchBackend(Protocol):
    """Recibe una búsqueda validada y devuelve ítems históricos como diccionarios.

    Una función async cumple el protocolo. El backend real (que envuelve
    ``retrieve()``) se construye en el composition root; el stub offline vive en
    ``examples/session-12/``.
    """

    async def __call__(self, args: SearchBudgetsArgs) -> list[dict[str, Any]]: ...


# Una función de consenso ``[(horas, distancia), ...] -> (horas, fiabilidad,
# dispersión)``. El conductor inyecta la MISMA que usa el camino determinista
# (``task_hours.distance_weighted_consensus``); así esta capa no importa rag.
ConsensusFn = Callable[[list[tuple[int, float]]], tuple[int, float, float]]


# --------------------------------------------------------------------------- #
# Implementaciones                                                             #
# --------------------------------------------------------------------------- #
async def search_budgets(
    raw_args: dict[str, Any],
    *,
    backend: BudgetSearchBackend,
) -> dict[str, Any]:
    """Recupera ítems históricos para un componente y resume el resultado."""
    args = SearchBudgetsArgs.model_validate(raw_args)
    items = await backend(args)
    hours = [item["estimated_hours"] for item in items if item.get("estimated_hours") is not None]
    summary = (
        f"{len(items)} historical items for {args.query!r}; hours={hours}"
        if items
        else f"no historical items for {args.query!r}"
    )
    log.info(
        "agent.tool.search_budgets",
        query=args.query,
        results=len(items),
        hours=len(hours),
    )
    return {"items": items, "count": len(items), "summary": summary}


def calculate_estimate(raw_args: dict[str, Any]) -> dict[str, Any]:
    """Costea los componentes de forma determinista. Sin LLM."""
    args = CalculateEstimateArgs.model_validate(raw_args)
    breakdown: list[dict[str, Any]] = []
    total = 0.0

    for component in args.components:
        references = component.reference_amounts
        if references:
            central = statistics.median(references)
            hours = round(central * (1 + CONTINGENCY_FACTOR), 1)
            unbudgeted = False
        else:
            # Sin referencia no se inventa una cifra: 0 h y flag visible para que
            # el modelo lo detecte y vuelva a buscar.
            hours = 0.0
            unbudgeted = True
        total += hours
        breakdown.append(
            {
                "name": component.name,
                "reference_count": len(references),
                "estimated_hours": hours,
                "unbudgeted": unbudgeted,
            }
        )

    total = round(total, 1)
    log.info(
        "agent.tool.calculate_estimate",
        components=len(breakdown),
        total_hours=total,
    )
    return {
        "components": breakdown,
        "total_hours": total,
        "contingency_factor": CONTINGENCY_FACTOR,
        "summary": f"total={total}h across {len(breakdown)} components",
    }


def validate_estimate(raw_args: dict[str, Any]) -> dict[str, Any]:
    """Guardrails de S4 sobre el informe final. Sin LLM."""
    args = ValidateEstimateArgs.model_validate(raw_args)
    issues: list[str] = []
    component_sum = 0.0

    for component in args.components:
        component_sum += component.estimated_hours
        if not component.reference_amounts:
            issues.append(f"{component.name!r} has no historical reference (unbudgeted).")
            continue
        low = min(component.reference_amounts) * 0.5
        high = max(component.reference_amounts) * 2.0
        if not low <= component.estimated_hours <= high:
            issues.append(
                f"{component.name!r} estimate {component.estimated_hours}h is outside the "
                f"plausible range [{round(low, 1)}, {round(high, 1)}]h implied by its "
                "references."
            )

    if args.total_hours <= 0:
        issues.append("Total hours is non-positive.")
    if abs(component_sum - args.total_hours) > 0.5:
        issues.append(
            f"Total {args.total_hours}h does not match the sum of components "
            f"({round(component_sum, 1)}h)."
        )
    if args.total_hours > MAX_PLAUSIBLE_HOURS:
        issues.append(f"Total {args.total_hours}h is implausibly large for a single project.")

    ok = not issues
    log.info("agent.tool.validate_estimate", ok=ok, issues=len(issues))
    return {
        "ok": ok,
        "issues": issues,
        "summary": "estimate passed all guardrails" if ok else f"{len(issues)} issue(s) found",
    }


# --------------------------------------------------------------------------- #
# Dispatch                                                                     #
# --------------------------------------------------------------------------- #
def derive_task_hours(
    raw_args: dict[str, Any],
    *,
    consensus_fn: ConsensusFn,
) -> dict[str, Any]:
    """Deriva las horas de una tarea con el consenso inyectado. Sin LLM."""
    args = DeriveTaskHoursArgs.model_validate(raw_args)
    neighbors = [(neighbor.estimated_hours, neighbor.distance) for neighbor in args.neighbors]
    hours, reliability, dispersion = consensus_fn(neighbors)
    source_chunk_ids = sorted(
        {neighbor.source_id for neighbor in args.neighbors if neighbor.source_id is not None}
    )
    log.info(
        "agent.tool.derive_task_hours",
        module=args.module,
        task=args.task,
        hours=hours,
        reliability=reliability,
    )
    return {
        "module": args.module,
        "task": args.task,
        "estimated_hours": hours,
        "reliability": reliability,
        "dispersion": dispersion,
        "source_chunk_ids": source_chunk_ids,
        "summary": (
            f"{args.task!r}: {hours}h (reliability {reliability}, dispersion {dispersion}) "
            f"from {len(neighbors)} analogs"
        ),
    }


async def dispatch_tool(
    name: str,
    raw_args: dict[str, Any],
    *,
    backend: BudgetSearchBackend,
    consensus_fn: ConsensusFn | None = None,
) -> dict[str, Any]:
    """Enruta una tool por nombre.

    Lanza en tool desconocida; el runner convierte cualquier excepción (incluida
    la ``ValidationError`` de argumentos inválidos) en una observación de error
    que el modelo puede corregir, sin romper el bucle.
    """
    if name == "search_budgets":
        return await search_budgets(raw_args, backend=backend)
    if name == "calculate_estimate":
        return calculate_estimate(raw_args)
    if name == "validate_estimate":
        return validate_estimate(raw_args)
    if name == "derive_task_hours":
        if consensus_fn is None:
            raise ValueError("derive_task_hours needs an injected consensus function")
        return derive_task_hours(raw_args, consensus_fn=consensus_fn)
    raise ValueError(f"Unknown tool: {name!r}")
