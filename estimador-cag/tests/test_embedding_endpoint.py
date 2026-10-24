"""Tests del endpoint ``POST /api/v1/embeddings/ingest`` con un embedder falso."""

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
FAKE_VECTOR = [0.01] * 1536


class FakeEmbedder:
    """Doble del embedder: no llama a la API y registra los batches recibidos."""

    def __init__(self) -> None:
        self.batches: list[int] = []

    def embed_many(self, chunks: list[Chunk]) -> list[EmbeddedChunk]:
        self.batches.append(len(chunks))
        return [EmbeddedChunk(**chunk.model_dump(), embedding=FAKE_VECTOR) for chunk in chunks]


def _payload() -> dict:
    return {"budgets": json.loads(DATASET.read_text(encoding="utf-8"))}


@pytest.fixture
def fake_embedder() -> Iterator[FakeEmbedder]:
    embedder = FakeEmbedder()
    app.dependency_overrides[get_embedder] = lambda: embedder
    yield embedder
    app.dependency_overrides.pop(get_embedder, None)


def test_ingest_returns_vectorized_chunks_and_stats(
    client: TestClient, fake_embedder: FakeEmbedder
) -> None:
    response = client.post("/api/v1/embeddings/ingest", json=_payload())

    assert response.status_code == 200
    body = response.json()
    assert len(body["chunks"]) == 60
    assert len(body["chunks"][0]["embedding"]) == 1536
    assert body["stats"] == {
        "total_budgets": 17,
        "total_chunks": 60,
        "total_tokens": 5970,
        "estimated_cost_usd": pytest.approx(0.0001194),
    }
    # 60 chunks caben en un único batch de 100.
    assert fake_embedder.batches == [60]


def test_invalid_sector_returns_422(client: TestClient, fake_embedder: FakeEmbedder) -> None:
    payload = _payload()
    payload["budgets"][0]["client_metadata"]["sector"] = "retail"

    response = client.post("/api/v1/embeddings/ingest", json=payload)

    assert response.status_code == 422


def test_unknown_field_returns_422(client: TestClient, fake_embedder: FakeEmbedder) -> None:
    payload = _payload()
    payload["budgets"][0]["surprise_field"] = "nope"

    response = client.post("/api/v1/embeddings/ingest", json=payload)

    assert response.status_code == 422


def test_missing_embeddings_configuration_returns_500(client: TestClient) -> None:
    app.dependency_overrides[get_embedder] = lambda: None
    try:
        response = client.post("/api/v1/embeddings/ingest", json=_payload())
    finally:
        app.dependency_overrides.pop(get_embedder, None)

    assert response.status_code == 500
    assert response.json() == {"detail": "No se pudieron generar los embeddings."}
