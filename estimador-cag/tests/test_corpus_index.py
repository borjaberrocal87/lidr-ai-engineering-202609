"""Endpoints de expansión del corpus: job 202 + polling y panel de stats."""

from __future__ import annotations

import uuid
from collections.abc import Iterator

import pytest
from pydantic import SecretStr
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool
from starlette.testclient import TestClient

from app.api import corpus_index
from app.config import settings
from app.dependencies import get_chunk_store, get_corpus_index_service
from app.foundation.persistence.database import get_session
from app.foundation.persistence.models import IngestionJobRow
from app.foundation.persistence.repositories.jobs import JobsRepository
from app.generation.rag.index_service import CorpusExpansionResult
from app.generation.rag.schemas import Budget
from app.generation.rag.store.repository import CollectionCorpusStats
from app.main import app

HEADERS = {"X-API-Key": "estimate-secret"}


def _budget(budget_id: str) -> dict:
    return {
        "budget_id": budget_id,
        "client_metadata": {"name": "ACME", "sector": "finance", "country": "ES"},
        "project_summary": "proyecto de prueba",
        "main_technology": "python",
        "year": 2024,
        "total_estimated_hours": 40,
        "components": [
            {
                "component_id": "C-1",
                "name": "API",
                "description": "API REST",
                "estimated_hours": 40,
                "complexity": "medium",
            }
        ],
    }


class FakeIndexService:
    async def expand(
        self,
        documents: list[Budget],
        *,
        document_type: str,
        chunk_type: str,
        on_progress=None,
    ) -> CorpusExpansionResult:
        for index in range(1, len(documents) + 1):
            if on_progress is not None:
                on_progress(index)
        return CorpusExpansionResult(
            documents_indexed=len(documents),
            documents_skipped=0,
            documents_failed=0,
            chunks_created=len(documents),
        )


class FakeStore:
    async def corpus_stats(self, _session: object) -> list[CollectionCorpusStats]:
        return [
            CollectionCorpusStats("budget", documents=2, chunks=10, hnsw_indexed=True),
            CollectionCorpusStats("transcript", documents=1, chunks=4, hnsw_indexed=False),
        ]


class _FakeAsyncSession:
    async def __aenter__(self) -> _FakeAsyncSession:
        return self

    async def __aexit__(self, *_exc: object) -> bool:
        return False


@pytest.fixture
def sqlite_sessions() -> Iterator[sessionmaker[Session]]:
    """Job store en SQLite compartido (una sola conexión, sin Postgres real)."""
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    IngestionJobRow.__table__.create(engine)
    factory = sessionmaker(engine, expire_on_commit=False)

    def _override_session() -> Iterator[Session]:
        session = factory()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[get_session] = _override_session
    yield factory
    app.dependency_overrides.pop(get_session, None)
    engine.dispose()


@pytest.fixture
def index_service() -> Iterator[FakeIndexService]:
    app.dependency_overrides[get_corpus_index_service] = lambda: FakeIndexService()
    yield FakeIndexService()
    app.dependency_overrides.pop(get_corpus_index_service, None)


def test_create_index_run_returns_202_and_the_job_completes(
    client: TestClient,
    monkeypatch,
    sqlite_sessions: sessionmaker[Session],
    index_service: FakeIndexService,
) -> None:
    monkeypatch.setattr(settings, "estimate_api_key", SecretStr("estimate-secret"))

    def fake_record_job(
        job_id: uuid.UUID,
        event: str,
        *,
        documents_count: int = 0,
        error_message: str | None = None,
    ) -> None:
        session = sqlite_sessions()
        try:
            repo = JobsRepository(session)
            if event == "running":
                repo.mark_running(job_id)
            elif event == "progress":
                repo.set_documents_count(job_id, documents_count)
            elif event == "completed":
                repo.mark_completed(job_id, documents_count=documents_count)
            else:
                repo.mark_failed(job_id, error_message=error_message or "boom")
            session.commit()
        finally:
            session.close()

    monkeypatch.setattr(corpus_index, "_record_job", fake_record_job)

    response = client.post(
        "/api/v1/embeddings/index/runs",
        json={"documents": [_budget("B-1"), _budget("B-2")]},
        headers=HEADERS,
    )

    assert response.status_code == 202
    body = response.json()
    assert body["documents_total"] == 2

    polled = client.get(f"/api/v1/embeddings/index/jobs/{body['job_id']}", headers=HEADERS)

    assert polled.status_code == 200
    assert polled.json()["status"] == "completed"
    assert polled.json()["documents_processed"] == 2


def test_index_run_without_embeddings_returns_500(
    client: TestClient, monkeypatch, sqlite_sessions: sessionmaker[Session]
) -> None:
    monkeypatch.setattr(settings, "estimate_api_key", SecretStr("estimate-secret"))
    app.dependency_overrides[get_corpus_index_service] = lambda: None
    try:
        response = client.post(
            "/api/v1/embeddings/index/runs",
            json={"documents": [_budget("B-1")]},
            headers=HEADERS,
        )
    finally:
        app.dependency_overrides.pop(get_corpus_index_service, None)

    assert response.status_code == 500


def test_unknown_job_returns_404(
    client: TestClient, monkeypatch, sqlite_sessions: sessionmaker[Session]
) -> None:
    monkeypatch.setattr(settings, "estimate_api_key", SecretStr("estimate-secret"))

    response = client.get(f"/api/v1/embeddings/index/jobs/{uuid.uuid4()}", headers=HEADERS)

    assert response.status_code == 404


def test_stats_reports_hnsw_state_per_collection(
    client: TestClient, monkeypatch, sqlite_sessions: sessionmaker[Session]
) -> None:
    monkeypatch.setattr(settings, "estimate_api_key", SecretStr("estimate-secret"))
    app.dependency_overrides[get_chunk_store] = lambda: FakeStore()
    monkeypatch.setattr(
        corpus_index,
        "get_async_session_factory",
        lambda: lambda: _FakeAsyncSession(),
    )
    try:
        response = client.get("/api/v1/embeddings/index/stats", headers=HEADERS)
    finally:
        app.dependency_overrides.pop(get_chunk_store, None)

    assert response.status_code == 200
    body = response.json()
    assert body["total_chunks"] == 14
    assert body["collections"][0]["collection"] == "budget"
    assert body["collections"][0]["hnsw_indexed"] is True
    assert body["collections"][1]["hnsw_indexed"] is False
