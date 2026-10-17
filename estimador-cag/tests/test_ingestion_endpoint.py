"""Pruebas a nivel HTTP de los endpoints de ingesta.

Prescindimos por completo de Postgres. ``get_unit_of_work`` se sustituye por un
doble en memoria que imita el contrato del repositorio real
(pending → running → completed) y el ``UnitOfWork`` del router se monkey-patchea
al mismo doble, de modo que la BackgroundTask y el endpoint GET comparten estado.
Así aislamos el test de la infraestructura sin dejar de ejercitar el cableado del
router + orquestador.
"""

from __future__ import annotations

import textwrap
import uuid
from datetime import UTC, datetime
from typing import ClassVar

import pytest
from fastapi.testclient import TestClient

from app.dependencies import (
    get_catalog,
    get_filesystem_loader,
    get_parser_registry,
    get_unit_of_work,
)
from app.ingestion.catalog.loader import load_catalog
from app.ingestion.loaders.filesystem import FileSystemLoader
from app.ingestion.parsers.registry import default_registry
from app.main import app


class _InMemoryJob:
    def __init__(self, source_name: str):
        self.job_id = uuid.uuid4()
        self.source_name = source_name
        self.status = "pending"
        self.documents_count = 0
        self.error_message: str | None = None
        self.started_at = datetime.now(UTC)
        self.finished_at: datetime | None = None


class _InMemoryJobsRepo:
    """Almacén global de módulo para que la BackgroundTask y el endpoint GET compartan estado."""

    _store: ClassVar[dict[uuid.UUID, _InMemoryJob]] = {}

    def create(self, *, source_name: str):
        job = _InMemoryJob(source_name)
        self._store[job.job_id] = job
        return job

    def get(self, job_id):
        return self._store.get(job_id)

    def list(self, *, limit=20, offset=0):
        jobs = sorted(self._store.values(), key=lambda job: job.started_at, reverse=True)
        return jobs[offset : offset + limit]

    def mark_running(self, job_id):
        if job := self._store.get(job_id):
            job.status = "running"

    def mark_completed(self, job_id, *, documents_count):
        if job := self._store.get(job_id):
            job.status = "completed"
            job.documents_count = documents_count
            job.finished_at = datetime.now(UTC)

    def mark_failed(self, job_id, *, error_message):
        if job := self._store.get(job_id):
            job.status = "failed"
            job.error_message = error_message
            job.finished_at = datetime.now(UTC)


class _InMemoryUnitOfWork:
    """Doble del ``UnitOfWork``: sin Postgres, store compartido."""

    _store = _InMemoryJobsRepo._store

    def __init__(self, *args, **kwargs) -> None:
        self.jobs = _InMemoryJobsRepo()

    def commit(self) -> None:  # no-op: el store es compartido
        pass

    def rollback(self) -> None:
        pass

    def close(self) -> None:
        pass


def _write_minimal_catalog(tmp_path):
    yaml = """
    version: "1.0.0"
    sources:
      - name: tiny_json
        location: budgets
        format: json
        quality: {completeness: 4, consistency: 4, actuality: 5, reliability: 4}
        sensitivity: {has_pii: false, pii_flags: []}
        decision: include
      - name: excluded_xlsx
        location: rate_card.xlsx
        format: xlsx
        quality: {completeness: 4, consistency: 5, actuality: 1, reliability: 4}
        sensitivity: {has_pii: false, pii_flags: []}
        decision: exclude
        decision_reason: "actuality=1 — outdated"
    """
    catalog_path = tmp_path / "catalog.yaml"
    catalog_path.write_text(textwrap.dedent(yaml), encoding="utf-8")

    budgets_dir = tmp_path / "budgets"
    budgets_dir.mkdir()
    (budgets_dir / "BUDGET-2024-0001.json").write_text(
        '{"budget_id":"BUDGET-2024-0001","currency":"EUR",'
        '"total_amount":100,"signed_at":"2024-01-01","phases":[]}',
        encoding="utf-8",
    )
    return catalog_path


@pytest.fixture
def ingestion_client(tmp_path, monkeypatch):
    catalog_path = _write_minimal_catalog(tmp_path)

    catalog = load_catalog(catalog_path)
    loader = FileSystemLoader(data_root=tmp_path)
    registry = default_registry()

    # Sobrescribe las dependencias de FastAPI para que el router use nuestros fixtures de tmp_path.
    app.dependency_overrides[get_catalog] = lambda: catalog
    app.dependency_overrides[get_filesystem_loader] = lambda: loader
    app.dependency_overrides[get_parser_registry] = lambda: registry
    app.dependency_overrides[get_unit_of_work] = lambda: _InMemoryUnitOfWork()

    # La BackgroundTask construye su propio UnitOfWork; redirígelo también al doble.
    monkeypatch.setattr("app.routers.ingestion.UnitOfWork", _InMemoryUnitOfWork)

    # Reinicia el estado global de módulo entre tests.
    _InMemoryJobsRepo._store.clear()

    yield TestClient(app)

    app.dependency_overrides.clear()


def test_post_run_returns_202_and_creates_job(ingestion_client):
    response = ingestion_client.post("/api/v1/ingestion/runs", json={"source_name": "tiny_json"})
    assert response.status_code == 202, response.text
    body = response.json()
    assert body["status"] == "pending"
    assert body["source_name"] == "tiny_json"
    assert "job_id" in body


def test_get_job_reflects_completed_state(ingestion_client):
    # TestClient ejecuta las BackgroundTasks de forma síncrona tras la respuesta.
    response = ingestion_client.post("/api/v1/ingestion/runs", json={"source_name": "tiny_json"})
    job_id = response.json()["job_id"]
    get = ingestion_client.get(f"/api/v1/ingestion/jobs/{job_id}")
    assert get.status_code == 200
    body = get.json()
    assert body["status"] == "completed"
    assert body["documents_count"] >= 1


def test_unknown_source_returns_404(ingestion_client):
    response = ingestion_client.post(
        "/api/v1/ingestion/runs", json={"source_name": "does_not_exist"}
    )
    assert response.status_code == 404
    assert response.json()["detail"]["reason"] == "unknown_source"


def test_excluded_source_returns_400(ingestion_client):
    response = ingestion_client.post(
        "/api/v1/ingestion/runs", json={"source_name": "excluded_xlsx"}
    )
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert detail["reason"] == "source_not_included"
    assert detail["decision"] == "exclude"


def test_get_unknown_job_returns_404(ingestion_client):
    fake = "00000000-0000-0000-0000-000000000000"
    assert ingestion_client.get(f"/api/v1/ingestion/jobs/{fake}").status_code == 404


def test_list_sources_returns_catalog(ingestion_client):
    response = ingestion_client.get("/api/v1/ingestion/sources")
    assert response.status_code == 200
    body = response.json()
    assert body["version"] == "1.0.0"
    names = {source["name"] for source in body["sources"]}
    assert {"tiny_json", "excluded_xlsx"} <= names


def test_list_sources_filters_by_decision(ingestion_client):
    response = ingestion_client.get("/api/v1/ingestion/sources?decision=exclude")
    assert response.status_code == 200
    body = response.json()
    assert [source["name"] for source in body["sources"]] == ["excluded_xlsx"]
    assert body["sources"][0]["decision"] == "exclude"
    assert body["sources"][0]["decision_reason"]


def test_list_sources_rejects_invalid_decision(ingestion_client):
    response = ingestion_client.get("/api/v1/ingestion/sources?decision=bogus")
    assert response.status_code == 422


def test_list_jobs_returns_created_jobs(ingestion_client):
    ingestion_client.post("/api/v1/ingestion/runs", json={"source_name": "tiny_json"})
    response = ingestion_client.get("/api/v1/ingestion/jobs")
    assert response.status_code == 200
    body = response.json()
    assert body["limit"] == 20
    assert body["offset"] == 0
    assert len(body["items"]) == 1
    assert body["items"][0]["status"] == "completed"
