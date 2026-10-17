"""Tests del cableado de pseudonimización en el caso de uso de ingesta.

No tocan Presidio ni la base de datos: inyectan un pseudo-analizador falso y un
UoW en memoria. Verifican que `ingest_source` aplica el pseudonimizador al texto
de cada `Document` cuando se le pasa uno, y que no lo hace cuando no se le pasa.
"""

from __future__ import annotations

import json
import uuid

import pytest

from app.ingestion.catalog.models import CatalogSource, QualityScore, Sensitivity
from app.ingestion.loaders.filesystem import FileSystemLoader
from app.ingestion.orchestrator import ingest_source
from app.ingestion.parsers.registry import default_registry


class _FakeJobs:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def mark_running(self, job_id: uuid.UUID) -> None:
        self.calls.append(("running", {"job_id": job_id}))

    def mark_completed(self, job_id: uuid.UUID, *, documents_count: int) -> None:
        self.calls.append(("completed", {"job_id": job_id, "documents_count": documents_count}))

    def mark_failed(self, job_id: uuid.UUID, *, error_message: str) -> None:
        self.calls.append(("failed", {"job_id": job_id, "error_message": error_message}))


class _FakeUnitOfWork:
    def __init__(self) -> None:
        self.jobs = _FakeJobs()
        self.commits = 0

    def commit(self) -> None:
        self.commits += 1


class _RecordingPseudonymizer:
    def __init__(self) -> None:
        self.texts: list[str] = []

    def __call__(self, text: str) -> str:
        self.texts.append(text)
        return "<<pseudonimizado>>"


def _source(data_root, *, has_pii: bool) -> CatalogSource:
    (data_root / "budgets").mkdir(exist_ok=True)
    (data_root / "budgets" / "BUDGET-2024-0001.json").write_text(
        json.dumps(
            {
                "budget_id": "BUDGET-2024-0001",
                "client_name": "Laura Fernández",
                "currency": "EUR",
                "total_amount": 100,
                "signed_at": "2024-01-01",
                "phases": [],
            }
        ),
        encoding="utf-8",
    )
    return CatalogSource(
        name="presupuestos_json",
        location="budgets",
        format="json",
        quality=QualityScore(completeness=4, consistency=4, actuality=5, reliability=4),
        sensitivity=Sensitivity(has_pii=has_pii, pii_flags=["PERSON"] if has_pii else []),
        decision="include",
    )


@pytest.fixture
def data_root(tmp_path):
    return tmp_path


def test_pseudonymizer_applied_when_provided(data_root):
    source = _source(data_root, has_pii=True)
    uow = _FakeUnitOfWork()
    recorder = _RecordingPseudonymizer()

    documents = ingest_source(
        source=source,
        source_version="1.0.0",
        loader=FileSystemLoader(data_root=data_root),
        registry=default_registry(),
        uow=uow,
        job_id=uuid.uuid4(),
        pseudonymizer=recorder,
    )

    assert recorder.texts, "el pseudonimizador debería recibir el texto del documento"
    assert all(doc.text == "<<pseudonimizado>>" for doc in documents)
    # El job se completa con el número de documentos (ya pseudonimizados).
    assert uow.jobs.calls[-1][0] == "completed"
    assert uow.jobs.calls[-1][1]["documents_count"] == len(documents)


def test_no_pseudonymizer_keeps_original_text(data_root):
    source = _source(data_root, has_pii=True)
    uow = _FakeUnitOfWork()

    documents = ingest_source(
        source=source,
        source_version="1.0.0",
        loader=FileSystemLoader(data_root=data_root),
        registry=default_registry(),
        uow=uow,
        job_id=uuid.uuid4(),
    )

    assert any("Laura Fernández" in doc.text for doc in documents)
