"""Tests del endpoint ``POST /api/v1/embeddings/ingest`` con un servicio falso."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from app.dependencies import get_rag_ingest_service
from app.embedding_pipeline.ingest_service import DuplicateDocumentError
from app.embedding_pipeline.schemas import Budget, IngestResponse
from app.main import app

DATASET = Path(__file__).resolve().parents[1] / "data" / "budgets_sample.json"


class FakeIngestService:
    """Doble del servicio de ingesta: no toca Postgres ni la API de embeddings."""

    def __init__(
        self,
        *,
        response: IngestResponse | None = None,
        error: Exception | None = None,
    ) -> None:
        self.response = response or IngestResponse(
            document_id=42,
            chunks_created=3,
            embedding_dimension=1536,
            ingestion_time_ms=12,
        )
        self.error = error
        self.calls: list[dict[str, object]] = []

    async def ingest(
        self,
        *,
        source_path: str,
        document_type: str,
        budget: Budget,
    ) -> IngestResponse:
        self.calls.append(
            {
                "source_path": source_path,
                "document_type": document_type,
                "budget": budget,
            }
        )
        if self.error is not None:
            raise self.error
        return self.response


def _budget() -> dict:
    return json.loads(DATASET.read_text(encoding="utf-8"))[0]


def _payload() -> dict:
    return {
        "source_path": "data/budgets_sample.json::BUD-2024-001",
        "document_type": "historical_budget",
        "content": _budget(),
    }


@pytest.fixture
def fake_service() -> Iterator[FakeIngestService]:
    service = FakeIngestService()
    app.dependency_overrides[get_rag_ingest_service] = lambda: service
    yield service
    app.dependency_overrides.pop(get_rag_ingest_service, None)


def test_ingest_persists_and_returns_ids_and_metrics(
    client: TestClient, fake_service: FakeIngestService
) -> None:
    response = client.post("/api/v1/embeddings/ingest", json=_payload())

    assert response.status_code == 200
    assert response.json() == {
        "document_id": 42,
        "chunks_created": 3,
        "embedding_dimension": 1536,
        "ingestion_time_ms": 12,
    }
    assert fake_service.calls[0]["source_path"] == "data/budgets_sample.json::BUD-2024-001"
    assert fake_service.calls[0]["document_type"] == "historical_budget"


def test_duplicate_source_path_returns_409(client: TestClient) -> None:
    service = FakeIngestService(error=DuplicateDocumentError(42))
    app.dependency_overrides[get_rag_ingest_service] = lambda: service
    try:
        response = client.post("/api/v1/embeddings/ingest", json=_payload())
    finally:
        app.dependency_overrides.pop(get_rag_ingest_service, None)

    assert response.status_code == 409
    assert response.json() == {
        "detail": "Document already ingested",
        "document_id": 42,
    }


def test_invalid_sector_returns_422(client: TestClient, fake_service: FakeIngestService) -> None:
    payload = _payload()
    payload["content"]["client_metadata"]["sector"] = "retail"

    response = client.post("/api/v1/embeddings/ingest", json=payload)

    assert response.status_code == 422
    assert fake_service.calls == []


def test_unknown_budget_field_returns_422(
    client: TestClient, fake_service: FakeIngestService
) -> None:
    payload = _payload()
    payload["content"]["surprise_field"] = "nope"

    response = client.post("/api/v1/embeddings/ingest", json=payload)

    assert response.status_code == 422
    assert fake_service.calls == []


def test_empty_source_path_returns_422(client: TestClient, fake_service: FakeIngestService) -> None:
    payload = _payload()
    payload["source_path"] = ""

    response = client.post("/api/v1/embeddings/ingest", json=payload)

    assert response.status_code == 422
    assert fake_service.calls == []


def test_document_type_too_long_returns_422(
    client: TestClient, fake_service: FakeIngestService
) -> None:
    payload = _payload()
    payload["document_type"] = "x" * 51

    response = client.post("/api/v1/embeddings/ingest", json=payload)

    assert response.status_code == 422
    assert fake_service.calls == []


def test_missing_ingest_configuration_returns_500(client: TestClient) -> None:
    app.dependency_overrides[get_rag_ingest_service] = lambda: None
    try:
        response = client.post("/api/v1/embeddings/ingest", json=_payload())
    finally:
        app.dependency_overrides.pop(get_rag_ingest_service, None)

    assert response.status_code == 500
    assert response.json() == {"detail": "No se pudieron persistir los embeddings."}
