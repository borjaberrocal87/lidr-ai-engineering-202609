"""Endpoints del wizard agéntico: auth, contrato y traza en la respuesta."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

from pydantic import SecretStr
from starlette.testclient import TestClient

from app.config import settings
from app.dependencies import get_async_openai_client
from app.generation.agentic.estimation_agent.schemas import (
    AgentModule,
    AgentStructure,
    AgentTask,
)
from app.generation.rag.schemas import TaskHoursEstimate, TaskHoursResult
from app.main import app

HEADERS = {"X-API-Key": "estimate-secret"}

STRUCTURE = AgentStructure(
    modules=[
        AgentModule(name="Backend", tasks=[AgentTask(name="Modelo de pedidos")]),
    ],
    assumptions=[],
    confidence="medium",
    reasoning="Descomposición directa.",
)


class StructureClient:
    def __init__(self) -> None:
        self.responses = SimpleNamespace(parse=self._parse)

    async def _parse(self, **_kwargs: Any) -> Any:
        return SimpleNamespace(output_parsed=STRUCTURE)


def _derive_call() -> Any:
    return SimpleNamespace(
        type="function_call",
        call_id="call-1",
        name="derive_task_hours",
        arguments=json.dumps(
            {
                "module": "Backend",
                "task": "Modelo de pedidos",
                "neighbors": [{"source_id": 3, "estimated_hours": 40, "distance": 0.1}],
            }
        ),
    )


class RecoveryClient:
    def __init__(self) -> None:
        self.responses = SimpleNamespace(create=self._create)

    async def _create(self, **kwargs: Any) -> Any:
        if kwargs.get("previous_response_id"):
            return SimpleNamespace(id="resp-2", output=[])
        return SimpleNamespace(id="resp-1", output=[_derive_call()])


class FakeRuntime:
    def effective_search_mode(self) -> str:
        return "vector"

    def effective_rerank(self) -> bool:
        return False

    def effective_task_hours_top_k(self) -> int:
        return 5

    def effective_task_hours_distance_threshold(self) -> float:
        return 0.45

    def effective_synthesis(self) -> bool:
        return False


def _brief() -> dict[str, Any]:
    return {
        "functionality": "App de pedidos",
        "technologies": ["FastAPI"],
        "scale": "unknown",
    }


def _hours_payload() -> dict[str, Any]:
    return {
        "modules": [
            {
                "name": "Backend",
                "tasks": [{"name": "Modelo de pedidos", "description": "CRUD"}],
            }
        ]
    }


async def _empty_backend(_args: Any) -> list[dict[str, Any]]:
    return []


def _base_hours() -> TaskHoursResult:
    return TaskHoursResult(
        tasks=[
            TaskHoursEstimate(module="Backend", task="Modelo de pedidos", has_match=False)
        ]
    )


def _patch_hours_dependencies(monkeypatch) -> None:
    monkeypatch.setattr("app.api.agent_estimation.get_embedder", lambda: object())
    monkeypatch.setattr("app.api.agent_estimation.get_chunk_store", lambda: object())
    monkeypatch.setattr("app.api.agent_estimation.get_async_session_factory", lambda: object)
    monkeypatch.setattr("app.api.agent_estimation.get_reranker", lambda: object())
    monkeypatch.setattr(
        "app.api.agent_estimation.get_runtime_retrieval_config", lambda: FakeRuntime()
    )
    monkeypatch.setattr(
        "app.api.agent_estimation.make_budget_search_backend",
        lambda **_kwargs: _empty_backend,
    )

    async def fake_estimate_all(*_args, **_kwargs) -> TaskHoursResult:
        return _base_hours()

    monkeypatch.setattr("app.api.agent_estimation.estimate_all", fake_estimate_all)


def test_structure_endpoint_requires_the_estimate_key(monkeypatch) -> None:
    monkeypatch.setattr(settings, "estimate_api_key", SecretStr("estimate-secret"))

    response = TestClient(app).post("/api/v1/agent/estimate/structure", json={"query": _brief()})

    assert response.status_code == 401


def test_structure_endpoint_returns_the_wizard_contract_and_the_trace(monkeypatch) -> None:
    monkeypatch.setattr(settings, "estimate_api_key", SecretStr("estimate-secret"))
    app.dependency_overrides[get_async_openai_client] = lambda: StructureClient()
    try:
        response = TestClient(app).post(
            "/api/v1/agent/estimate/structure",
            json={"query": _brief()},
            headers=HEADERS,
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    payload = response.json()
    assert payload["estimate"]["modules"][0]["name"] == "Backend"
    assert payload["estimate"]["modules"][0]["tasks"][0]["engineer_days"] is None
    assert payload["coherent"] is True
    assert payload["agent_trace"]["stopped_reason"] == "completed"


def test_structure_endpoint_returns_500_without_a_client(monkeypatch) -> None:
    monkeypatch.setattr(settings, "estimate_api_key", SecretStr("estimate-secret"))
    app.dependency_overrides[get_async_openai_client] = lambda: None
    try:
        response = TestClient(app).post(
            "/api/v1/agent/estimate/structure",
            json={"query": _brief()},
            headers=HEADERS,
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 500


def test_hours_endpoint_merges_recovered_hours_and_returns_the_trace(
    monkeypatch,
) -> None:
    monkeypatch.setattr(settings, "estimate_api_key", SecretStr("estimate-secret"))
    _patch_hours_dependencies(monkeypatch)
    app.dependency_overrides[get_async_openai_client] = lambda: RecoveryClient()
    try:
        response = TestClient(app).post(
            "/api/v1/agent/estimate/hours",
            json=_hours_payload(),
            headers=HEADERS,
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    payload = response.json()
    assert payload["tasks"][0]["has_match"] is True
    assert payload["tasks"][0]["estimated_hours"] is not None
    assert payload["agent_trace"]["steps"][0]["tool"] == "derive_task_hours"


def test_hours_endpoint_rejects_unknown_override_knobs(monkeypatch) -> None:
    monkeypatch.setattr(settings, "estimate_api_key", SecretStr("estimate-secret"))

    payload = _hours_payload()
    payload["agent"] = {"not_a_knob": 1}

    response = TestClient(app).post(
        "/api/v1/agent/estimate/hours", json=payload, headers=HEADERS
    )

    assert response.status_code == 422
