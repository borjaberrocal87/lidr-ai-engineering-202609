"""Graph endpoint: availability, validation, success and failure mapping."""

from __future__ import annotations

from typing import Any

from starlette.testclient import TestClient

from app.main import app

TRANSCRIPT = "x" * 120

STATE: dict[str, Any] = {
    "estimation_id": "est-1",
    "status": "validated",
    "requirements": ["orders backend"],
    "components": [{"name": "Orders backend", "category": "backend"}],
    "budget_matches": [
        {
            "component": "Orders backend",
            "reference_budget_id": "B1",
            "amount": 100.0,
            "distance": 0.1,
        }
    ],
    "node_runs": [{"name": "extract_requirements", "duration_ms": 12.5}],
    "estimate": {"components": [], "total_hours": 0.0},
    "errors": [],
}


class FakeGraph:
    def __init__(self, state: dict[str, Any] | None = None) -> None:
        self.state = state or STATE
        self.config: dict[str, Any] | None = None

    async def ainvoke(self, _input: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
        self.config = config
        return self.state


class FailingGraph:
    async def ainvoke(self, _input: dict[str, Any], _config: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError("node blew up")


def _post(graph: Any, payload: dict[str, Any] | None = None) -> Any:
    app.state.graph = graph
    try:
        return TestClient(app).post(
            "/api/v1/graph/estimate/from-transcript",
            json=payload if payload is not None else {"transcript": TRANSCRIPT},
        )
    finally:
        app.state.graph = None


def test_graph_endpoint_returns_estimate_status_and_run_text() -> None:
    graph = FakeGraph()
    response = _post(graph, {"transcript": TRANSCRIPT, "estimation_id": "est-1"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["estimation_id"] == "est-1"
    assert payload["status"] == "validated"
    assert payload["components"][0]["name"] == "Orders backend"
    assert payload["budget_matches"][0]["reference_budget_id"] == "B1"
    assert payload["node_runs"][0]["name"] == "extract_requirements"
    assert "SESSION 13 — ESTIMATION GRAPH RUN" in payload["run_text"]
    assert graph.config == {"configurable": {"thread_id": "est-1"}}


def test_graph_endpoint_generates_a_thread_id_when_omitted() -> None:
    graph = FakeGraph()
    response = _post(graph)

    assert response.status_code == 200
    assert response.json()["estimation_id"]
    assert graph.config == {
        "configurable": {"thread_id": response.json()["estimation_id"]}
    }


def test_graph_endpoint_returns_503_without_a_graph() -> None:
    response = _post(None, {"transcript": TRANSCRIPT})
    assert response.status_code == 503
    assert "no está disponible" in response.json()["detail"]


def test_graph_endpoint_returns_502_when_a_node_fails() -> None:
    response = _post(FailingGraph(), {"transcript": TRANSCRIPT})
    assert response.status_code == 502


def test_graph_endpoint_rejects_a_short_transcript() -> None:
    response = _post(FakeGraph(), {"transcript": "too short"})
    assert response.status_code == 422
