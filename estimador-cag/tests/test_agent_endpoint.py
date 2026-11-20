"""Endpoint del agente: configuración, validación y respuesta con traza."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

from starlette.testclient import TestClient

from app.dependencies import get_async_openai_client, get_budget_search_backend
from app.generation.agentic.estimation_agent.schemas import AgentEstimateReport
from app.main import app

TRANSCRIPT = "x" * 120

REPORT = AgentEstimateReport(
    components=[],
    total_hours=0.0,
    assumptions=["Nothing to estimate."],
    confidence="low",
)


def _call(call_id: str, name: str, arguments: dict[str, Any]) -> Any:
    return SimpleNamespace(
        type="function_call",
        call_id=call_id,
        name=name,
        arguments=json.dumps(arguments),
    )


class FakeClient:
    def __init__(self) -> None:
        self.responses = SimpleNamespace(
            create=self._create,
            parse=self._parse,
        )
        self._turn = 0

    async def _create(self, **_kwargs: Any) -> Any:
        self._turn += 1
        if self._turn == 1:
            return SimpleNamespace(
                id="resp-1",
                output=[
                    SimpleNamespace(type="reasoning", summary=[]),
                    _call("call-1", "validate_estimate", {"components": [], "total_hours": 0.0}),
                ],
            )
        return SimpleNamespace(id="resp-2", output=[])

    async def _parse(self, **_kwargs: Any) -> Any:
        return SimpleNamespace(output_parsed=REPORT)


async def _backend(_args: Any) -> list[dict[str, Any]]:
    return []


def _client() -> TestClient:
    app.dependency_overrides[get_async_openai_client] = lambda: FakeClient()
    app.dependency_overrides[get_budget_search_backend] = lambda: _backend
    return TestClient(app)


def test_agent_endpoint_returns_the_report_and_the_rendered_trace() -> None:
    try:
        response = _client().post(
            "/api/v1/agent/estimate/from-transcript",
            json={"transcript": TRANSCRIPT},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    payload = response.json()
    assert payload["stopped_reason"] == "completed"
    assert payload["iterations"] >= 2
    assert payload["trace"]["steps"][0]["tool"] == "validate_estimate"
    assert "STEP 1" in payload["trace_text"]
    assert "action:      validate_estimate" in payload["trace_text"]


def test_agent_endpoint_rejects_a_short_transcript() -> None:
    try:
        response = _client().post(
            "/api/v1/agent/estimate/from-transcript",
            json={"transcript": "too short"},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 422


def test_agent_endpoint_returns_500_without_a_client() -> None:
    try:
        app.dependency_overrides[get_async_openai_client] = lambda: None
        app.dependency_overrides[get_budget_search_backend] = lambda: _backend
        response = TestClient(app).post(
            "/api/v1/agent/estimate/from-transcript",
            json={"transcript": TRANSCRIPT},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 500
    assert "no está configurado" in response.json()["detail"]
