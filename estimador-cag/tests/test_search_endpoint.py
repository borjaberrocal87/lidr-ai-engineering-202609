"""Tests del endpoint ``POST /api/v1/search`` con un retriever falso."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from starlette.testclient import TestClient

from app.dependencies import get_semantic_retriever
from app.embedding_pipeline.schemas import SearchHit, SearchResponse
from app.main import app


class FakeRetriever:
    """Doble del retriever: no embebe ni consulta Postgres."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    async def search(self, *, query: str, k: int) -> SearchResponse:
        self.calls.append((query, k))
        return SearchResponse(
            query=query,
            k=k,
            search_time_ms=7,
            results=[
                SearchHit(
                    chunk_id=1,
                    document_id=2,
                    chunk_type="budget_component",
                    content="Backend service implementation with JWT...",
                    distance=0.231,
                    metadata={"budget_id": "BUD-2024-001"},
                )
            ],
        )


@pytest.fixture
def fake_retriever() -> Iterator[FakeRetriever]:
    retriever = FakeRetriever()
    app.dependency_overrides[get_semantic_retriever] = lambda: retriever
    yield retriever
    app.dependency_overrides.pop(get_semantic_retriever, None)


def test_search_returns_ranked_chunks(client: TestClient, fake_retriever: FakeRetriever) -> None:
    response = client.post(
        "/api/v1/search",
        json={"query": "REST API with OAuth authentication", "k": 5},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["query"] == "REST API with OAuth authentication"
    assert body["k"] == 5
    assert body["search_time_ms"] == 7
    assert body["results"] == [
        {
            "chunk_id": 1,
            "document_id": 2,
            "chunk_type": "budget_component",
            "content": "Backend service implementation with JWT...",
            "distance": 0.231,
            "metadata": {"budget_id": "BUD-2024-001"},
        }
    ]
    assert fake_retriever.calls == [("REST API with OAuth authentication", 5)]


@pytest.mark.parametrize("k", [0, 51])
def test_search_rejects_out_of_range_k(
    client: TestClient, fake_retriever: FakeRetriever, k: int
) -> None:
    response = client.post("/api/v1/search", json={"query": "valid query", "k": k})

    assert response.status_code == 422
    assert fake_retriever.calls == []


def test_search_rejects_empty_query(client: TestClient, fake_retriever: FakeRetriever) -> None:
    response = client.post("/api/v1/search", json={"query": "", "k": 5})

    assert response.status_code == 422
    assert fake_retriever.calls == []


def test_search_without_configuration_returns_500(client: TestClient) -> None:
    app.dependency_overrides[get_semantic_retriever] = lambda: None
    try:
        response = client.post("/api/v1/search", json={"query": "valid query", "k": 5})
    finally:
        app.dependency_overrides.pop(get_semantic_retriever, None)

    assert response.status_code == 500
    assert response.json() == {"detail": "No se pudo ejecutar la búsqueda semántica."}
