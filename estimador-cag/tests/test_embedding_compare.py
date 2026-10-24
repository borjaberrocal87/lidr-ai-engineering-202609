"""Tests del endpoint POST /api/v1/embeddings/compare con dobles."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from app.dependencies import get_embedder
from app.embedding_pipeline.schemas import Chunk, EmbeddedChunk
from app.main import app

DATASET = Path(__file__).resolve().parents[1] / "data" / "budgets_sample.json"


class FakeEmbedder:
    def embed_one(self, text: str) -> list[float]:
        return [1.0, float(len(text) % 5)]

    def embed_many(self, chunks: list[Chunk]) -> list[EmbeddedChunk]:
        return [
            EmbeddedChunk(**chunk.model_dump(), embedding=self.embed_one(chunk.text))
            for chunk in chunks
        ]


def _payload(queries: list[str] | None = None) -> dict:
    budgets = json.loads(DATASET.read_text(encoding="utf-8"))[:1]
    return {
        "budgets": budgets,
        "queries": queries or [],
        "strategies": ["structural", "hierarchical"],
        "top_k": 2,
    }


@pytest.fixture
def fake_embedder() -> Iterator[FakeEmbedder]:
    embedder = FakeEmbedder()
    app.dependency_overrides[get_embedder] = lambda: embedder
    yield embedder
    app.dependency_overrides.pop(get_embedder, None)


def test_compare_stats_only_does_not_need_embedder(client: TestClient) -> None:
    app.dependency_overrides[get_embedder] = lambda: None
    try:
        response = client.post("/api/v1/embeddings/compare", json=_payload())
    finally:
        app.dependency_overrides.pop(get_embedder, None)

    assert response.status_code == 200
    body = response.json()
    assert set(body["stats_per_strategy"]) == {"structural", "hierarchical"}
    assert body["stats_per_strategy"]["structural"]["n_chunks"] == 4
    assert body["queries_per_strategy"] == {}


def test_compare_with_queries_returns_top_k(
    client: TestClient, fake_embedder: FakeEmbedder
) -> None:
    response = client.post(
        "/api/v1/embeddings/compare",
        json=_payload(["OAuth authentication"]),
    )

    assert response.status_code == 200
    body = response.json()
    assert len(body["queries_per_strategy"]["structural"]) == 1
    assert len(body["queries_per_strategy"]["structural"][0]["top_k"]) == 2


def test_compare_with_queries_without_embedder_returns_500(client: TestClient) -> None:
    app.dependency_overrides[get_embedder] = lambda: None
    try:
        response = client.post(
            "/api/v1/embeddings/compare",
            json=_payload(["OAuth authentication"]),
        )
    finally:
        app.dependency_overrides.pop(get_embedder, None)

    assert response.status_code == 500


def test_compare_unknown_strategy_returns_400(
    client: TestClient, fake_embedder: FakeEmbedder
) -> None:
    payload = _payload()
    payload["strategies"] = ["not_a_strategy"]

    response = client.post("/api/v1/embeddings/compare", json=payload)

    assert response.status_code == 400
    assert "desconocida" in response.json()["detail"]
