"""The five graph nodes (Session 13) — ``state -> partial update`` functions.

Each node is a factory that receives its dependencies injected and returns the
node callable. That keeps the layering contract intact: this package never
imports ``app.dependencies`` or ``app.generation.rag``; the composition root
builds the LLM wrapper and the budget-search backend and passes them here.

* ``extract_requirements`` / ``classify_components`` are the only LLM nodes and go
  through ``LLMWrapper.complete_structured`` (LiteLLM + Instructor).
* ``search_budgets`` reuses the Session 12 ``BudgetSearchBackend`` (the real
  S9-S10 retrieval wrapped by the composition root), sequentially for now.
* ``generate_estimate`` reuses the Session 12 deterministic cost model (median +
  contingency, unbudgeted at zero hours) from ``estimation_agent/tools.py``.
* ``validate_and_consolidate`` ports the Session 12 guardrails and fixes the
  output ``status``.

Every node runs inside one ``logfire`` span (via ``traced_node``) and records its
duration in the ``node_runs`` accumulator, so a run has a visible timeline even
without an external observability backend.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from functools import wraps
from typing import Any

import structlog

from app.foundation.llm.wrapper import LLMWrapper
from app.generation.agentic.estimation_agent.schemas import SearchBudgetsArgs
from app.generation.agentic.estimation_agent.tools import (
    CONTINGENCY_FACTOR,
    MAX_PLAUSIBLE_HOURS,
    BudgetSearchBackend,
    calculate_estimate,
)
from app.generation.agentic.estimation_graph.observability import span
from app.generation.agentic.estimation_graph.schemas import (
    ComponentClassification,
    EstimateComponent,
    EstimateReport,
    RequirementsExtraction,
)
from app.generation.agentic.estimation_graph.state import (
    BudgetMatch,
    EstimationState,
)

log = structlog.get_logger(__name__)

NodeFn = Callable[[EstimationState], Awaitable[dict[str, Any]]]

_EXTRACT_SYSTEM_PROMPT = (
    "You are a software-delivery analyst. Read a raw, messy client meeting "
    "transcript and extract a flat list of the concrete requirements the client "
    "wants built. One atomic requirement per item, concise technical English, "
    "regardless of the transcript language. Ignore small talk, anecdotes and "
    "digressions. Never invent requirements the transcript gives no evidence for."
)

_CLASSIFY_SYSTEM_PROMPT = (
    "You are a solution architect. Group a list of project requirements into the "
    "distinct functional COMPONENTS needed to deliver them. Each component has a "
    "short name and a coarse category (e.g. backend, integration, mobile, "
    "analytics, frontend, infrastructure). Merge requirements that belong to the "
    "same component; keep genuinely unrelated pieces separate."
)


def _traced_node(name: str, fn: NodeFn) -> NodeFn:
    """Wrap a node: one Logfire span per node + a ``NodeRun`` appended to state."""

    @wraps(fn)
    async def wrapper(state: EstimationState) -> dict[str, Any]:
        with span(f"node: {name}"):
            started = time.perf_counter()
            update = await fn(state)
            duration_ms = round((time.perf_counter() - started) * 1000, 3)
        update.setdefault("node_runs", [{"name": name, "duration_ms": duration_ms}])
        return update

    return wrapper


# --------------------------------------------------------------------------- #
# LLM nodes                                                                    #
# --------------------------------------------------------------------------- #
def make_extract_requirements_node(
    llm_wrapper: LLMWrapper,
    *,
    model: str,
    max_tokens: int,
    temperature: float,
) -> NodeFn:
    """Transcript → a flat list of requirements (structured LLM call)."""

    async def extract_requirements(state: EstimationState) -> dict[str, Any]:
        result, _meta = await asyncio.to_thread(
            llm_wrapper.complete_structured,
            system_prompt=_EXTRACT_SYSTEM_PROMPT,
            user_message=state["transcript"],
            response_model=RequirementsExtraction,
            temperature=temperature,
            max_tokens=max_tokens,
            model_override=model,
        )
        requirements = [item.strip() for item in result.requirements if item.strip()]
        log.info("graph.node.extract_requirements", requirements=len(requirements))
        return {"requirements": requirements}

    return _traced_node("extract_requirements", extract_requirements)


def make_classify_components_node(
    llm_wrapper: LLMWrapper,
    *,
    model: str,
    max_tokens: int,
    temperature: float,
) -> NodeFn:
    """Requirements → components with a category (structured LLM call)."""

    async def classify_components(state: EstimationState) -> dict[str, Any]:
        requirements = state.get("requirements") or []
        user_message = "Requirements:\n" + "\n".join(f"- {item}" for item in requirements)
        result, _meta = await asyncio.to_thread(
            llm_wrapper.complete_structured,
            system_prompt=_CLASSIFY_SYSTEM_PROMPT,
            user_message=user_message,
            response_model=ComponentClassification,
            temperature=temperature,
            max_tokens=max_tokens,
            model_override=model,
        )
        components = [
            {"name": item.name.strip(), "category": item.category.strip()}
            for item in result.components
            if item.name.strip()
        ]
        log.info("graph.node.classify_components", components=len(components))
        return {"components": components}

    return _traced_node("classify_components", classify_components)


# --------------------------------------------------------------------------- #
# Retrieval node                                                               #
# --------------------------------------------------------------------------- #
def make_search_budgets_node(backend: BudgetSearchBackend) -> NodeFn:
    """For each component, retrieve reference budgets (SEQUENTIAL for now).

    Returns only the NEW matches; the ``operator.add`` reducer appends them. A
    per-component retrieval failure is soft: it goes to ``errors`` and the loop
    continues (the live session upgrades this to real error handling).
    """

    async def search_budgets(state: EstimationState) -> dict[str, Any]:
        components = state.get("components") or []
        matches: list[BudgetMatch] = []
        errors: list[str] = []
        for component in components:
            query = f"{component['name']} ({component['category']})"
            try:
                items = await backend(SearchBudgetsArgs(query=query, filters=None))
            except Exception as exc:
                log.warning(
                    "graph.node.search_budgets_error",
                    component=component["name"],
                    error=str(exc)[:200],
                )
                errors.append(f"search_budgets failed for {component['name']!r}: {exc}")
                continue
            for item in items:
                hours = item.get("estimated_hours")
                if hours is None:
                    continue
                matches.append(
                    {
                        "component": component["name"],
                        "reference_budget_id": item.get("budget_id"),
                        "amount": float(hours),
                        "distance": float(item.get("distance") or 0.0),
                    }
                )
        log.info(
            "graph.node.search_budgets",
            components=len(components),
            matches=len(matches),
            errors=len(errors),
        )
        update: dict[str, Any] = {"budget_matches": matches}
        if errors:
            update["errors"] = errors
        return update

    return _traced_node("search_budgets", search_budgets)


# --------------------------------------------------------------------------- #
# Deterministic nodes (Session 12 cost model + guardrails)                     #
# --------------------------------------------------------------------------- #
def _norm(name: str) -> str:
    """Normalise a component name for matching.

    Component names come from the LLM and may echo a ``[category]`` suffix or
    drift in casing; the ``budget_matches`` names come from classify verbatim.
    """
    base = name.split("[", 1)[0]
    return base.strip().lower()


def _references_for(component: str, matches: list[BudgetMatch]) -> list[BudgetMatch]:
    target = _norm(component)
    return [match for match in matches if _norm(match["component"]) == target]


def make_generate_estimate_node() -> NodeFn:
    """Consolidate components + their matches into a deterministic estimate."""

    async def generate_estimate(state: EstimationState) -> dict[str, Any]:
        components = state.get("components") or []
        matches = state.get("budget_matches") or []

        references_by_component: dict[str, list[BudgetMatch]] = {}
        payload: list[dict[str, Any]] = []
        for component in components:
            refs = _references_for(component["name"], matches)
            references_by_component[component["name"]] = refs
            payload.append(
                {
                    "name": component["name"],
                    "reference_amounts": [match["amount"] for match in refs],
                }
            )

        # Reuse the Session 12 cost model verbatim: median + contingency, and
        # unbudgeted components at zero hours (never an invented number).
        calculation = calculate_estimate({"components": payload})
        breakdown: list[dict[str, Any]] = calculation["components"]

        report_components: list[EstimateComponent] = []
        for item in breakdown:
            name = str(item["name"])
            refs = references_by_component.get(name, [])
            reference_ids = sorted(
                {match["reference_budget_id"] for match in refs if match["reference_budget_id"]}
            )
            if item["unbudgeted"]:
                rationale = "No historical reference found; not estimated (unbudgeted)."
            else:
                rationale = (
                    f"Median of {item['reference_count']} historical reference(s) plus a "
                    f"{CONTINGENCY_FACTOR:.0%} contingency."
                )
            report_components.append(
                EstimateComponent(
                    name=name,
                    estimated_hours=float(item["estimated_hours"]),
                    unbudgeted=bool(item["unbudgeted"]),
                    reference_budget_ids=reference_ids,
                    rationale=rationale,
                )
            )

        report = EstimateReport(
            components=report_components,
            total_hours=float(calculation["total_hours"]),
        )
        log.info(
            "graph.node.generate_estimate",
            components=len(report_components),
            total_hours=report.total_hours,
        )
        return {"estimate": report.model_dump()}

    return _traced_node("generate_estimate", generate_estimate)


def _validation_issues(
    estimate: dict[str, Any],
    matches: list[BudgetMatch],
) -> list[str]:
    """Deterministic guardrails over the estimate (ported from Session 12)."""
    issues: list[str] = []
    components: list[dict[str, Any]] = estimate.get("components") or []
    component_sum = 0.0
    for component in components:
        name = str(component.get("name", "?"))
        hours = component.get("estimated_hours")
        refs = _references_for(name, matches)
        if hours is None:
            issues.append(f"{name!r} has no estimated hours.")
            continue
        component_sum += float(hours)
        if component.get("unbudgeted") or not refs:
            issues.append(f"{name!r} has no historical reference (unbudgeted).")
            continue
        reference_hours = [match["amount"] for match in refs]
        low = min(reference_hours) * 0.5
        high = max(reference_hours) * 2.0
        if not low <= float(hours) <= high:
            issues.append(
                f"{name!r} estimate {hours}h is outside the plausible range "
                f"[{round(low, 1)}, {round(high, 1)}]h implied by its references."
            )

    total = estimate.get("total_hours")
    if total is None:
        issues.append("Total hours is missing.")
    else:
        if float(total) <= 0:
            issues.append("Total hours is non-positive.")
        if abs(component_sum - float(total)) > 0.5:
            issues.append(
                f"Total {total}h does not match the sum of components ({round(component_sum, 1)}h)."
            )
        if float(total) > MAX_PLAUSIBLE_HOURS:
            issues.append(f"Total {total}h is implausibly large for a single project.")
    return issues


def make_validate_and_consolidate_node() -> NodeFn:
    """Run guardrails over the estimate and fix the output ``status``."""

    async def validate_and_consolidate(state: EstimationState) -> dict[str, Any]:
        estimate = state.get("estimate") or {}
        matches = state.get("budget_matches") or []
        issues = _validation_issues(estimate, matches)
        status = "validated" if not issues else "needs_review"
        log.info("graph.node.validate", status=status, issues=len(issues))
        update: dict[str, Any] = {"status": status}
        if issues:
            update["errors"] = issues
        return update

    return _traced_node("validate_and_consolidate", validate_and_consolidate)
