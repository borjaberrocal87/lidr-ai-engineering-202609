"""Graph wiring: sequential order, accumulator growth and thread isolation."""

from __future__ import annotations

from typing import Any

from langgraph.checkpoint.memory import MemorySaver

from app.generation.agentic.estimation_agent.schemas import SearchBudgetsArgs
from app.generation.agentic.estimation_graph.build import build_graph, route_on_status
from app.generation.agentic.estimation_graph.schemas import (
    ComponentClassification,
    ComponentOut,
    RequirementsExtraction,
)

REQUIREMENTS = ["orders backend", "mobile app"]


class FakeWrapper:
    def complete_structured(self, *, response_model: Any, **_kwargs: Any) -> tuple[Any, dict]:
        if response_model is RequirementsExtraction:
            return RequirementsExtraction(requirements=list(REQUIREMENTS)), {}
        return (
            ComponentClassification(
                components=[
                    ComponentOut(name="Orders backend", category="backend"),
                    ComponentOut(name="Mobile app", category="mobile"),
                ]
            ),
            {},
        )


async def _backend(args: SearchBudgetsArgs) -> list[dict[str, Any]]:
    if "orders" in args.query.lower():
        return [
            {"budget_id": "B1", "estimated_hours": 100.0, "distance": 0.1},
            {"budget_id": "B2", "estimated_hours": 120.0, "distance": 0.2},
        ]
    return [{"budget_id": "B3", "estimated_hours": 190.0, "distance": 0.3}]


def _graph() -> Any:
    return build_graph(
        llm_wrapper=FakeWrapper(),
        backend=_backend,
        checkpointer=MemorySaver(),
    )


async def test_graph_runs_the_five_nodes_in_order_to_a_validated_estimate() -> None:
    state = await _graph().ainvoke(
        {"transcript": "x" * 120, "estimation_id": "run-1"},
        {"configurable": {"thread_id": "run-1"}},
    )

    assert [run["name"] for run in state["node_runs"]] == [
        "extract_requirements",
        "classify_components",
        "search_budgets",
        "generate_estimate",
        "validate_and_consolidate",
    ]
    assert state["requirements"] == REQUIREMENTS
    assert [component["name"] for component in state["components"]] == [
        "Orders backend",
        "Mobile app",
    ]
    # Two components searched sequentially => the accumulator holds both sets.
    assert len(state["budget_matches"]) == 3
    assert state["status"] == "validated"
    assert state["estimate"]["total_hours"] > 0
    assert state["node_runs"][2]["duration_ms"] >= 0


async def test_thread_ids_do_not_share_state() -> None:
    graph = _graph()
    first = await graph.ainvoke(
        {"transcript": "x" * 120, "estimation_id": "thread-a"},
        {"configurable": {"thread_id": "thread-a"}},
    )
    second = await graph.ainvoke(
        {"transcript": "y" * 120, "estimation_id": "thread-b"},
        {"configurable": {"thread_id": "thread-b"}},
    )

    assert first["estimation_id"] == "thread-a"
    assert second["estimation_id"] == "thread-b"
    # Each run starts from its own input: no accumulated matches leak across threads.
    assert len(first["budget_matches"]) == 3
    assert len(second["budget_matches"]) == 3


def test_route_on_status_branches_on_the_validation_result() -> None:
    assert route_on_status({"status": "needs_review"}) == "needs_review"
    assert route_on_status({"status": "validated"}) == "validated"
    assert route_on_status({}) == "validated"
