"""Endpoint de recuperación multifuente (``POST /api/v1/retrieval/multi-source``)."""

from __future__ import annotations

from pydantic import SecretStr
from starlette.testclient import TestClient

from app.config import settings
from app.generation.rag.retrieval.multi_source import MultiSourceOutcome
from app.generation.rag.retrieval.query_plan import SubQuery
from app.generation.rag.retrieval.routing import RoutingDecision
from app.generation.rag.retrieval.sources import Collection

HEADERS = {"X-API-Key": "retrieval-secret"}


class FakeRuntime:
    def effective_search_mode(self) -> str:
        return "vector"

    def effective_rerank(self) -> bool:
        return False

    def effective_routing(self) -> bool:
        return False

    def effective_query_plan(self) -> bool:
        return False

    def effective_freshness(self) -> bool:
        return False


def _payload() -> dict:
    return {"query_text": "OAuth authentication runbook", "top_k": 5}


def test_multi_source_requires_key(client: TestClient, monkeypatch) -> None:
    monkeypatch.setattr(settings, "retrieval_api_key", SecretStr("retrieval-secret"))

    response = client.post("/api/v1/retrieval/multi-source", json=_payload())

    assert response.status_code == 401


def test_multi_source_can_be_disabled(client: TestClient, monkeypatch) -> None:
    monkeypatch.setattr(settings, "retrieval_api_key", SecretStr("retrieval-secret"))
    monkeypatch.setattr(settings, "retrieval_multi_source_enabled", False)

    response = client.post(
        "/api/v1/retrieval/multi-source", json=_payload(), headers=HEADERS
    )

    assert response.status_code == 404


def test_multi_source_returns_diagnostics(client: TestClient, monkeypatch) -> None:
    monkeypatch.setattr(settings, "retrieval_api_key", SecretStr("retrieval-secret"))
    monkeypatch.setattr(settings, "retrieval_multi_source_enabled", True)
    monkeypatch.setattr("app.api.retrieval_multi_source.get_embedder", lambda: object())
    monkeypatch.setattr("app.api.retrieval_multi_source.get_chunk_store", lambda: object())
    monkeypatch.setattr(
        "app.api.retrieval_multi_source.get_async_session_factory", lambda: object
    )
    monkeypatch.setattr("app.api.retrieval_multi_source.get_llm_wrapper", lambda: object())
    monkeypatch.setattr("app.api.retrieval_multi_source.get_reranker", lambda: object())
    monkeypatch.setattr(
        "app.api.retrieval_multi_source.get_runtime_retrieval_config", lambda: FakeRuntime()
    )

    async def fake_retrieve(**_kwargs) -> MultiSourceOutcome:
        return MultiSourceOutcome(
            chunks=[],
            routing=RoutingDecision(
                targets=[Collection.TECHNICAL_DOC], level="rules", reason="runbook"
            ),
            technique="direct",
            subqueries=[SubQuery(topic="original", query="OAuth runbook")],
            cardinality={"technical_doc": 3},
            low_confidence=True,
        )

    monkeypatch.setattr(
        "app.api.retrieval_multi_source.retrieve_multi_source", fake_retrieve
    )

    response = client.post(
        "/api/v1/retrieval/multi-source", json=_payload(), headers=HEADERS
    )

    assert response.status_code == 200
    body = response.json()
    assert body["routing"]["targets"] == ["technical_doc"]
    assert body["technique"] == "direct"
    assert body["cardinality"] == {"technical_doc": 3}
    assert body["low_confidence"] is True
