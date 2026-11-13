"""Endpoint de horas por tarea (``POST /api/v1/estimate/tasks/hours``)."""

from __future__ import annotations

from pydantic import SecretStr
from starlette.testclient import TestClient

from app.config import settings
from app.generation.rag.schemas import TaskHoursEstimate, TaskHoursResult

HEADERS = {"X-API-Key": "estimate-secret"}


class FakeRuntime:
    def effective_search_mode(self) -> str:
        return "vector"

    def effective_rerank(self) -> bool:
        return False

    def effective_task_hours_top_k(self) -> int:
        return 5

    def effective_task_hours_distance_threshold(self) -> float:
        return 0.45


def _payload() -> dict:
    return {
        "modules": [
            {
                "name": "Auth",
                "tasks": [{"name": "OAuth2 login", "description": "PKCE flow"}],
            }
        ]
    }


def test_hours_requires_estimate_key(client: TestClient, monkeypatch) -> None:
    monkeypatch.setattr(settings, "estimate_api_key", SecretStr("estimate-secret"))

    response = client.post("/api/v1/estimate/tasks/hours", json=_payload())

    assert response.status_code == 401


def test_hours_returns_the_consensus(client: TestClient, monkeypatch) -> None:
    monkeypatch.setattr(settings, "estimate_api_key", SecretStr("estimate-secret"))
    monkeypatch.setattr("app.api.estimate_tasks.get_embedder", lambda: object())
    monkeypatch.setattr("app.api.estimate_tasks.get_chunk_store", lambda: object())
    monkeypatch.setattr("app.api.estimate_tasks.get_async_session_factory", lambda: object)
    monkeypatch.setattr("app.api.estimate_tasks.get_reranker", lambda: object())
    monkeypatch.setattr(
        "app.api.estimate_tasks.get_runtime_retrieval_config", lambda: FakeRuntime()
    )

    async def fake_estimate_all(*_args, **_kwargs) -> TaskHoursResult:
        return TaskHoursResult(
            tasks=[
                TaskHoursEstimate(
                    module="Auth",
                    task="OAuth2 login",
                    estimated_hours=40,
                    reliability=0.82,
                    has_match=True,
                )
            ]
        )

    monkeypatch.setattr("app.api.estimate_tasks.estimate_all", fake_estimate_all)

    response = client.post(
        "/api/v1/estimate/tasks/hours", json=_payload(), headers=HEADERS
    )

    assert response.status_code == 200
    body = response.json()
    assert body["tasks"][0]["estimated_hours"] == 40
    assert body["tasks"][0]["has_match"] is True


def test_hours_rejects_empty_modules(client: TestClient, monkeypatch) -> None:
    monkeypatch.setattr(settings, "estimate_api_key", SecretStr("estimate-secret"))

    response = client.post(
        "/api/v1/estimate/tasks/hours", json={"modules": []}, headers=HEADERS
    )

    assert response.status_code == 422
