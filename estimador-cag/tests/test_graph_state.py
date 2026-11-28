"""Graph state contracts: reducers accumulate, partial updates are allowed."""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from app.generation.agentic.estimation_graph.state import EstimationState


async def _first(_state: EstimationState) -> dict[str, Any]:
    return {
        "requirements": ["one requirement"],
        "budget_matches": [
            {"component": "A", "reference_budget_id": "b1", "amount": 10.0, "distance": 0.1}
        ],
        "node_runs": [{"name": "first", "duration_ms": 1.0}],
        "errors": ["first error"],
    }


async def _second(_state: EstimationState) -> dict[str, Any]:
    return {
        "budget_matches": [
            {"component": "B", "reference_budget_id": "b2", "amount": 20.0, "distance": 0.2}
        ],
        "node_runs": [{"name": "second", "duration_ms": 2.0}],
        "errors": ["second error"],
    }


def _two_node_graph() -> Any:
    builder = StateGraph(EstimationState)
    builder.add_node("first", _first)
    builder.add_node("second", _second)
    builder.add_edge(START, "first")
    builder.add_edge("first", "second")
    builder.add_edge("second", END)
    return builder.compile()


async def test_reducers_append_instead_of_replacing() -> None:
    state = await _two_node_graph().ainvoke({"transcript": "x"})
    assert [match["component"] for match in state["budget_matches"]] == ["A", "B"]
    assert state["errors"] == ["first error", "second error"]
    assert [run["name"] for run in state["node_runs"]] == ["first", "second"]


async def test_partial_updates_do_not_require_every_key() -> None:
    state = await _two_node_graph().ainvoke({"transcript": "x"})
    # Only ``_first`` wrote requirements; the second node returns no such key.
    assert state["requirements"] == ["one requirement"]
    assert "estimate" not in state
    assert "status" not in state
