"""Graph nodes: structured LLM output, sequential retrieval and cost model."""

from __future__ import annotations

from typing import Any

from app.generation.agentic.estimation_agent.schemas import SearchBudgetsArgs
from app.generation.agentic.estimation_agent.tools import CONTINGENCY_FACTOR
from app.generation.agentic.estimation_graph.nodes import (
    make_classify_components_node,
    make_extract_requirements_node,
    make_generate_estimate_node,
    make_search_budgets_node,
    make_validate_and_consolidate_node,
)
from app.generation.agentic.estimation_graph.schemas import (
    ComponentClassification,
    ComponentOut,
    RequirementsExtraction,
)
from app.generation.agentic.estimation_graph.state import EstimationState

LLM_KWARGS = {"model": "gpt-5-mini", "max_tokens": 4096, "temperature": 0.0}


class FakeWrapper:
    """Structured-output double: returns a scripted model per response type."""

    def __init__(
        self,
        requirements: list[str] | None = None,
        components: list[ComponentOut] | None = None,
    ) -> None:
        self.requirements = requirements or []
        self.components = components or []

    def complete_structured(self, *, response_model: Any, **_kwargs: Any) -> tuple[Any, dict]:
        if response_model is RequirementsExtraction:
            return RequirementsExtraction(requirements=self.requirements), {}
        return ComponentClassification(components=self.components), {}


def _component(name: str, category: str = "backend") -> dict[str, str]:
    return {"name": name, "category": category}


async def test_extract_requirements_trims_and_drops_blanks() -> None:
    node = make_extract_requirements_node(
        FakeWrapper(requirements=["  orders backend  ", "", "mobile app"]),
        **LLM_KWARGS,
    )
    update = await node({"transcript": "x" * 120})
    assert update["requirements"] == ["orders backend", "mobile app"]
    assert update["node_runs"][0]["name"] == "extract_requirements"


async def test_classify_components_returns_name_and_category() -> None:
    node = make_classify_components_node(
        FakeWrapper(
            components=[
                ComponentOut(name="Orders backend", category="backend"),
                ComponentOut(name="Mobile app", category="mobile"),
            ]
        ),
        **LLM_KWARGS,
    )
    update = await node({"requirements": ["orders", "mobile"]})
    assert update["components"] == [
        _component("Orders backend"),
        _component("Mobile app", "mobile"),
    ]


async def test_search_budgets_queries_once_per_component_and_skips_missing_hours() -> None:
    queries: list[str] = []

    async def backend(args: SearchBudgetsArgs) -> list[dict[str, Any]]:
        queries.append(args.query)
        if "orders" in args.query.lower():
            return [
                {"budget_id": "B1", "estimated_hours": 100.0, "distance": 0.1},
                {"budget_id": "B2", "estimated_hours": None, "distance": 0.2},
            ]
        return [{"budget_id": "B3", "estimated_hours": 190.0, "distance": 0.3}]

    node = make_search_budgets_node(backend)
    state: EstimationState = {
        "components": [_component("Orders backend"), _component("Mobile app", "mobile")]
    }
    update = await node(state)

    assert queries == ["Orders backend (backend)", "Mobile app (mobile)"]
    assert [(m["component"], m["amount"]) for m in update["budget_matches"]] == [
        ("Orders backend", 100.0),
        ("Mobile app", 190.0),
    ]
    assert "errors" not in update


async def test_search_budgets_soft_fails_one_component() -> None:
    async def backend(args: SearchBudgetsArgs) -> list[dict[str, Any]]:
        if "orders" in args.query.lower():
            raise RuntimeError("retrieval down")
        return [{"budget_id": "B3", "estimated_hours": 190.0, "distance": 0.3}]

    node = make_search_budgets_node(backend)
    state: EstimationState = {
        "components": [_component("Orders backend"), _component("Mobile app", "mobile")]
    }
    update = await node(state)

    assert len(update["budget_matches"]) == 1
    assert "Orders backend" in update["errors"][0]


def _matches() -> list[dict[str, Any]]:
    return [
        {
            "component": "Orders backend",
            "reference_budget_id": "B1",
            "amount": 100.0,
            "distance": 0.1,
        },
        {
            "component": "Orders backend",
            "reference_budget_id": "B2",
            "amount": 120.0,
            "distance": 0.2,
        },
        {
            "component": "Mobile app [mobile]",
            "reference_budget_id": "B3",
            "amount": 190.0,
            "distance": 0.3,
        },
    ]


async def test_generate_estimate_is_deterministic_and_uses_medians() -> None:
    node = make_generate_estimate_node()
    state: EstimationState = {
        "components": [_component("Orders backend"), _component("Mobile app", "mobile")],
        "budget_matches": _matches(),  # type: ignore[list-item]
    }
    first = await node(state)
    second = await node(state)

    # The estimate is reproducible; node_runs carries a wall-clock duration.
    assert first["estimate"] == second["estimate"]
    components = first["estimate"]["components"]
    by_name = {component["name"]: component for component in components}
    orders_hours = round(110.0 * (1 + CONTINGENCY_FACTOR), 1)
    mobile_hours = round(190.0 * (1 + CONTINGENCY_FACTOR), 1)
    assert by_name["Orders backend"]["estimated_hours"] == orders_hours
    assert by_name["Mobile app"]["estimated_hours"] == mobile_hours
    assert by_name["Orders backend"]["reference_budget_ids"] == ["B1", "B2"]
    assert first["estimate"]["total_hours"] == round(
        by_name["Orders backend"]["estimated_hours"] + by_name["Mobile app"]["estimated_hours"], 1
    )


async def test_generate_estimate_marks_unbudgeted_at_zero() -> None:
    node = make_generate_estimate_node()
    state: EstimationState = {
        "components": [_component("Data migration", "backend")],
        "budget_matches": [],
    }
    update = await node(state)

    component = update["estimate"]["components"][0]
    assert component["unbudgeted"] is True
    assert component["estimated_hours"] == 0.0
    assert update["estimate"]["total_hours"] == 0.0


async def test_validate_and_consolidate_clean_estimate_is_validated() -> None:
    generate = make_generate_estimate_node()
    validate = make_validate_and_consolidate_node()
    state: EstimationState = {
        "components": [_component("Orders backend"), _component("Mobile app", "mobile")],
        "budget_matches": _matches(),  # type: ignore[list-item]
    }
    estimate_update = await generate(state)
    state.update(estimate_update)  # type: ignore[typeddict-item]

    update = await validate(state)
    assert update["status"] == "validated"
    assert "errors" not in update


async def test_validate_and_consolidate_flags_unbudgeted_and_bad_totals() -> None:
    validate = make_validate_and_consolidate_node()
    state: EstimationState = {
        "budget_matches": [],
        "estimate": {
            "components": [
                {
                    "name": "Data migration",
                    "estimated_hours": 0.0,
                    "unbudgeted": True,
                    "reference_budget_ids": [],
                    "rationale": "unbudgeted",
                }
            ],
            "total_hours": 42.0,
        },
    }
    update = await validate(state)

    assert update["status"] == "needs_review"
    issues = update["errors"]
    assert any("unbudgeted" in issue for issue in issues)
    assert any("does not match the sum" in issue for issue in issues)
