"""Auth por API key y rate limiting de los endpoints S09."""

from __future__ import annotations

from pydantic import SecretStr
from starlette.requests import Request
from starlette.testclient import TestClient

from app.api.rate_limiting import rate_limit_handler
from app.config import settings
from app.dependencies import get_semantic_retriever
from app.generation.rag.schemas import RetrievalOutcome
from app.main import app


class FakeRetriever:
    async def retrieve_chunks(self, **kwargs) -> RetrievalOutcome:
        return RetrievalOutcome(chunks=[], low_confidence=True, candidates_evaluated=0)


def _payload() -> dict:
    return {
        "query_text": "historico de pagos para fintech",
        "top_k": 5,
        "distance_threshold": 0.6,
        "sectors": ["finance"],
    }


def _request(path: str = "/api/v1/retrieval/search") -> Request:
    return Request({"type": "http", "method": "POST", "path": path, "headers": []})


def test_retrieval_without_key_returns_401(client: TestClient, monkeypatch) -> None:
    monkeypatch.setattr(settings, "retrieval_api_key", SecretStr("retrieval-secret"))

    response = client.post("/api/v1/retrieval/search", json=_payload())

    assert response.status_code == 401


def test_retrieval_with_wrong_key_returns_401(client: TestClient, monkeypatch) -> None:
    monkeypatch.setattr(settings, "retrieval_api_key", SecretStr("retrieval-secret"))

    response = client.post(
        "/api/v1/retrieval/search",
        json=_payload(),
        headers={"X-API-Key": "otra"},
    )

    assert response.status_code == 401


def test_retrieval_with_valid_key_calls_the_retriever(client: TestClient, monkeypatch) -> None:
    monkeypatch.setattr(settings, "retrieval_api_key", SecretStr("retrieval-secret"))
    app.dependency_overrides[get_semantic_retriever] = lambda: FakeRetriever()
    try:
        response = client.post(
            "/api/v1/retrieval/search",
            json=_payload(),
            headers={"X-API-Key": "retrieval-secret"},
        )
    finally:
        app.dependency_overrides.pop(get_semantic_retriever, None)

    assert response.status_code == 200
    assert response.json()["low_confidence"] is True


def test_estimate_endpoint_requires_its_own_key(client: TestClient, monkeypatch) -> None:
    monkeypatch.setattr(settings, "estimate_api_key", SecretStr("estimate-secret"))
    monkeypatch.setattr(settings, "retrieval_api_key", SecretStr("retrieval-secret"))

    response = client.post(
        "/api/v1/estimate/from-transcript",
        json={"transcript": "x" * 120},
        headers={"X-API-Key": "retrieval-secret"},
    )

    assert response.status_code == 401


def test_rate_limit_handler_returns_429_with_retry_after() -> None:
    response = rate_limit_handler(_request(), RuntimeError("limit"))

    assert response.status_code == 429
    assert response.headers["Retry-After"] == "60"


def test_retrieval_rate_limit_returns_429(client: TestClient, monkeypatch) -> None:
    monkeypatch.setattr(settings, "retrieval_api_key", SecretStr("rate-limit-key"))
    app.dependency_overrides[get_semantic_retriever] = lambda: FakeRetriever()
    try:
        statuses = [
            client.post(
                "/api/v1/retrieval/search",
                json=_payload(),
                headers={"X-API-Key": "rate-limit-key"},
            ).status_code
            for _ in range(121)
        ]
    finally:
        app.dependency_overrides.pop(get_semantic_retriever, None)

    assert statuses[:120] == [200] * 120
    assert statuses[120] == 429
